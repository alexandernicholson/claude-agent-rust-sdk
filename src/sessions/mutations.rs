//! Portable session mutations: rename, tag, delete, and fork.
//!
//! Ports the official Python `_internal/session_mutations.py`. Filesystem and
//! `SessionStore`-backed variants share the fork transform (UUID/parent/session
//! remapping, `forkedFrom` stamping, stale-field stripping) so both paths
//! produce identical output.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

use crate::error::ClaudeError;
use crate::sessions::filesystem::{
    canonicalize_path, extract_first_prompt_from_head, extract_last_json_string_field,
    find_project_dir, get_projects_dir, get_worktree_paths, validate_uuid, LITE_READ_BUF_SIZE,
};
use crate::sessions::key::project_key_for_directory;
use crate::sessions::store::{SessionStore, SessionStoreEntry};
use crate::sessions::SessionKey;

const TRANSCRIPT_TYPES: [&str; 5] = ["user", "assistant", "attachment", "system", "progress"];

/// Result of a fork operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForkSessionResult {
    /// ID of the new forked session (a freshly generated canonical UUID).
    pub session_id: String,
}

// ---------------------------------------------------------------------------
// Filesystem: rename / tag / delete
// ---------------------------------------------------------------------------

/// Rename a session by appending a `custom-title` entry.
///
/// `list_sessions` reads the LAST custom-title from the tail, so repeated calls
/// are safe — the most recent wins.
///
/// # Errors
/// Returns [`ClaudeError::InvalidConfig`] for an invalid UUID or empty title,
/// and [`ClaudeError::TransportError`] if the session file cannot be found.
pub fn rename_session(
    session_id: &str,
    title: &str,
    directory: Option<&str>,
) -> Result<(), ClaudeError> {
    require_uuid(session_id)?;
    let stripped = title.trim();
    if stripped.is_empty() {
        return Err(ClaudeError::InvalidConfig("title must be non-empty".into()));
    }
    let data = compact_line(&json!({
        "type": "custom-title",
        "customTitle": stripped,
        "sessionId": session_id,
    }));
    append_to_session(session_id, &data, directory)
}

/// Tag a session. Pass `None` to clear the tag (appends an empty-string tag).
///
/// Tags are Unicode-sanitized (zero-width, directional marks, private-use, etc.)
/// before storing.
///
/// # Errors
/// Returns [`ClaudeError::InvalidConfig`] for an invalid UUID or an empty tag
/// after sanitization, and [`ClaudeError::TransportError`] if not found.
pub fn tag_session(
    session_id: &str,
    tag: Option<&str>,
    directory: Option<&str>,
) -> Result<(), ClaudeError> {
    require_uuid(session_id)?;
    let stored_tag = match tag {
        Some(t) => {
            let sanitized = sanitize_unicode(t);
            let sanitized = sanitized.trim();
            if sanitized.is_empty() {
                return Err(ClaudeError::InvalidConfig(
                    "tag must be non-empty (use None to clear)".into(),
                ));
            }
            sanitized.to_string()
        }
        None => String::new(),
    };
    let data = compact_line(&json!({
        "type": "tag",
        "tag": stored_tag,
        "sessionId": session_id,
    }));
    append_to_session(session_id, &data, directory)
}

/// Delete a session: removes `{id}.jsonl` and the sibling `{id}/` subagents dir.
///
/// # Errors
/// Returns [`ClaudeError::InvalidConfig`] for an invalid UUID and
/// [`ClaudeError::TransportError`] if the session file cannot be found.
pub fn delete_session(session_id: &str, directory: Option<&str>) -> Result<(), ClaudeError> {
    require_uuid(session_id)?;
    let path = find_session_file(session_id, directory)
        .ok_or_else(|| ClaudeError::TransportError(not_found_message(session_id, directory)))?;
    match std::fs::remove_file(&path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(ClaudeError::TransportError(format!(
                "Session {session_id} not found"
            )));
        }
        Err(e) => return Err(ClaudeError::TransportError(e.to_string())),
    }
    // Subagent transcripts live in a sibling {session_id}/ dir; often absent.
    if let Some(parent) = path.parent() {
        let _ = std::fs::remove_dir_all(parent.join(session_id));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Filesystem: fork
// ---------------------------------------------------------------------------

/// Fork a session into a new branch with fresh UUIDs.
///
/// Copies transcript messages into a new session file, remapping every message
/// UUID, preserving the `parentUuid` chain (skipping progress ancestors),
/// stamping `forkedFrom`, and stripping session-specific fields. Supports
/// `up_to_message_id` for branching from a specific point.
///
/// # Errors
/// Returns [`ClaudeError::InvalidConfig`] for invalid UUIDs or a session with
/// no forkable messages, and [`ClaudeError::TransportError`] if the source is
/// not found.
pub fn fork_session(
    session_id: &str,
    directory: Option<&str>,
    up_to_message_id: Option<&str>,
    title: Option<&str>,
) -> Result<ForkSessionResult, ClaudeError> {
    require_uuid(session_id)?;
    if let Some(up) = up_to_message_id {
        if validate_uuid(up).is_none() {
            return Err(ClaudeError::InvalidConfig(format!(
                "Invalid up_to_message_id: {up}"
            )));
        }
    }

    let (file_path, project_dir) = find_session_file_with_dir(session_id, directory)
        .ok_or_else(|| ClaudeError::TransportError(not_found_message(session_id, directory)))?;

    let content =
        std::fs::read(&file_path).map_err(|e| ClaudeError::TransportError(e.to_string()))?;
    if content.is_empty() {
        return Err(ClaudeError::InvalidConfig(format!(
            "Session {session_id} has no messages to fork"
        )));
    }

    let (transcript, content_replacements) = parse_fork_transcript(&content, session_id);

    let content_for_title = content.clone();
    let derive_title = move || -> Option<String> {
        let buf_len = content_for_title.len();
        let head_end = buf_len.min(LITE_READ_BUF_SIZE);
        let head = String::from_utf8_lossy(&content_for_title[..head_end]).into_owned();
        let tail_start = buf_len.saturating_sub(LITE_READ_BUF_SIZE);
        let tail = String::from_utf8_lossy(&content_for_title[tail_start..]).into_owned();
        extract_last_json_string_field(&tail, "customTitle")
            .or_else(|| extract_last_json_string_field(&head, "customTitle"))
            .or_else(|| extract_last_json_string_field(&tail, "aiTitle"))
            .or_else(|| extract_last_json_string_field(&head, "aiTitle"))
            .or_else(|| {
                let p = extract_first_prompt_from_head(&head);
                (!p.is_empty()).then_some(p)
            })
    };

    let (forked_session_id, lines) = build_fork_lines(
        transcript,
        &content_replacements,
        session_id,
        up_to_message_id,
        title,
        derive_title,
    )?;

    let fork_path = project_dir.join(format!("{forked_session_id}.jsonl"));
    let mut body = lines.join("\n");
    body.push('\n');
    write_new_file_0600(&fork_path, body.as_bytes())?;

    Ok(ForkSessionResult {
        session_id: forked_session_id,
    })
}

/// Core fork transform — remap UUIDs and produce serialized JSONL lines.
///
/// Shared by the filesystem and store-backed paths. `derive_title` is only
/// invoked when no explicit `title` is given.
#[allow(clippy::too_many_lines)]
fn build_fork_lines<F>(
    transcript: Vec<Map<String, Value>>,
    content_replacements: &[Value],
    session_id: &str,
    up_to_message_id: Option<&str>,
    title: Option<&str>,
    derive_title: F,
) -> Result<(String, Vec<String>), ClaudeError>
where
    F: FnOnce() -> Option<String>,
{
    // Filter out sidechains; keep isMeta entries (interleaved in main chain).
    let mut transcript: Vec<Map<String, Value>> = transcript
        .into_iter()
        .filter(|e| {
            !e.get("isSidechain")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .collect();

    if transcript.is_empty() {
        return Err(no_messages_err(session_id));
    }

    if let Some(up) = up_to_message_id {
        let cutoff = transcript
            .iter()
            .position(|e| e.get("uuid").and_then(Value::as_str) == Some(up));
        let Some(cutoff) = cutoff else {
            return Err(ClaudeError::InvalidConfig(format!(
                "Message {up} not found in session {session_id}"
            )));
        };
        transcript.truncate(cutoff + 1);
    }

    // Map every entry uuid (including progress) for parent chain walking.
    let mut uuid_mapping: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for entry in &transcript {
        if let Some(u) = entry.get("uuid").and_then(Value::as_str) {
            uuid_mapping.insert(u.to_string(), Uuid::new_v4().to_string());
        }
    }

    // Progress messages are UI-only chain links, not written to the fork.
    let writable: Vec<&Map<String, Value>> = transcript
        .iter()
        .filter(|e| e.get("type").and_then(Value::as_str) != Some("progress"))
        .collect();
    if writable.is_empty() {
        return Err(no_messages_err(session_id));
    }

    let by_uuid: std::collections::HashMap<&str, &Map<String, Value>> = transcript
        .iter()
        .filter_map(|e| e.get("uuid").and_then(Value::as_str).map(|u| (u, e)))
        .collect();

    let forked_session_id = Uuid::new_v4().to_string();
    let now = iso_now();
    let mut lines: Vec<String> = Vec::new();

    let writable_len = writable.len();
    for (i, original) in writable.iter().enumerate() {
        let original_uuid = original
            .get("uuid")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let new_uuid = uuid_mapping.get(original_uuid).cloned().unwrap_or_default();

        // Resolve parentUuid, skipping progress ancestors.
        let mut new_parent_uuid: Option<String> = None;
        let mut parent_id = original
            .get("parentUuid")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        while let Some(pid) = parent_id.clone() {
            let Some(parent) = by_uuid.get(pid.as_str()) else {
                break;
            };
            if parent.get("type").and_then(Value::as_str) != Some("progress") {
                new_parent_uuid = uuid_mapping.get(&pid).cloned();
                break;
            }
            parent_id = parent
                .get("parentUuid")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string);
        }

        // Only update timestamp on the last message (leaf detection on resume).
        let timestamp = if i == writable_len - 1 {
            now.clone()
        } else {
            original
                .get("timestamp")
                .and_then(Value::as_str)
                .map_or_else(|| now.clone(), str::to_string)
        };

        // Remap logicalParentUuid (compact-boundary backpointer). Mirrors
        // `new_logical_parent = uuid_mapping.get(lp) if lp else lp`:
        //  - a truthy value is looked up in the fork's UUID remap; an unmapped
        //    (out-of-fork) parent resolves to `null`, NOT the stale original.
        //  - a falsy value (absent / empty string) is preserved verbatim.
        let original_logical = original.get("logicalParentUuid");
        let logical_truthy = original_logical
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        let new_logical_parent = match logical_truthy {
            Some(lp) => uuid_mapping
                .get(lp)
                .map_or(Value::Null, |m| Value::String(m.clone())),
            // Falsy: keep whatever was there (null, "", or missing→null).
            None => original_logical.cloned().unwrap_or(Value::Null),
        };

        let mut forked = (*original).clone();
        forked.insert("uuid".into(), Value::String(new_uuid));
        forked.insert(
            "parentUuid".into(),
            new_parent_uuid.map_or(Value::Null, Value::String),
        );
        forked.insert("logicalParentUuid".into(), new_logical_parent);
        forked.insert("sessionId".into(), Value::String(forked_session_id.clone()));
        forked.insert("timestamp".into(), Value::String(timestamp));
        forked.insert("isSidechain".into(), Value::Bool(false));
        forked.insert(
            "forkedFrom".into(),
            json!({ "sessionId": session_id, "messageUuid": original_uuid }),
        );
        for key in ["teamName", "agentName", "slug", "sourceToolAssistantUUID"] {
            forked.remove(key);
        }

        lines.push(compact_line(&Value::Object(forked)));
    }

    // Content-replacement entry (if any) under the fork's sessionId.
    if !content_replacements.is_empty() {
        lines.push(compact_line(&json!({
            "type": "content-replacement",
            "sessionId": forked_session_id,
            "replacements": content_replacements,
            "uuid": Uuid::new_v4().to_string(),
            "timestamp": now,
        })));
    }

    // Title: explicit > derived (customTitle > aiTitle > first prompt) + " (fork)".
    let fork_title = match title.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => t.to_string(),
        None => format!(
            "{} (fork)",
            derive_title().unwrap_or_else(|| "Forked session".to_string())
        ),
    };

    lines.push(compact_line(&json!({
        "type": "custom-title",
        "sessionId": forked_session_id,
        "customTitle": fork_title,
        "uuid": Uuid::new_v4().to_string(),
        "timestamp": now,
    })));

    Ok((forked_session_id, lines))
}

// ---------------------------------------------------------------------------
// SessionStore-backed variants
// ---------------------------------------------------------------------------

/// Rename a session by appending a `custom-title` entry to a [`SessionStore`].
///
/// # Errors
/// [`ClaudeError::InvalidConfig`] for invalid UUID / empty title; adapter
/// errors propagate from `append`.
pub async fn rename_session_via_store(
    session_store: &dyn SessionStore,
    session_id: &str,
    title: &str,
    directory: Option<&Path>,
) -> Result<(), ClaudeError> {
    let uuid = require_uuid(session_id)?;
    let stripped = title.trim();
    if stripped.is_empty() {
        return Err(ClaudeError::InvalidConfig("title must be non-empty".into()));
    }
    let key = SessionKey::new(project_key_for_directory(directory), uuid);
    let entry = json_object(json!({
        "type": "custom-title",
        "customTitle": stripped,
        "sessionId": session_id,
        "uuid": Uuid::new_v4().to_string(),
        "timestamp": iso_now(),
    }));
    session_store.append(&key, vec![entry]).await
}

/// Tag a session by appending a tag entry to a [`SessionStore`] (`None` clears).
///
/// # Errors
/// [`ClaudeError::InvalidConfig`] for invalid UUID / empty tag; adapter errors
/// propagate from `append`.
pub async fn tag_session_via_store(
    session_store: &dyn SessionStore,
    session_id: &str,
    tag: Option<&str>,
    directory: Option<&Path>,
) -> Result<(), ClaudeError> {
    let uuid = require_uuid(session_id)?;
    let stored_tag = match tag {
        Some(t) => {
            let sanitized = sanitize_unicode(t);
            let sanitized = sanitized.trim();
            if sanitized.is_empty() {
                return Err(ClaudeError::InvalidConfig(
                    "tag must be non-empty (use None to clear)".into(),
                ));
            }
            sanitized.to_string()
        }
        None => String::new(),
    };
    let key = SessionKey::new(project_key_for_directory(directory), uuid);
    let entry = json_object(json!({
        "type": "tag",
        "tag": stored_tag,
        "sessionId": session_id,
        "uuid": Uuid::new_v4().to_string(),
        "timestamp": iso_now(),
    }));
    session_store.append(&key, vec![entry]).await
}

/// Delete a session from a [`SessionStore`].
///
/// No-op when the store does not implement `delete` (append-only backends).
///
/// # Errors
/// [`ClaudeError::InvalidConfig`] for an invalid UUID; adapter errors from
/// `delete`.
pub async fn delete_session_via_store(
    session_store: &dyn SessionStore,
    session_id: &str,
    directory: Option<&Path>,
) -> Result<(), ClaudeError> {
    let uuid = require_uuid(session_id)?;
    if !session_store.capabilities().delete {
        return Ok(());
    }
    let key = SessionKey::new(project_key_for_directory(directory), uuid);
    session_store.delete(&key).await
}

/// Fork a session into a new branch with fresh UUIDs via a [`SessionStore`].
///
/// Runs the fork transform directly over the loaded entries (no JSONL
/// round-trip), then re-parses the emitted lines so the store receives the same
/// shape the mirror path would.
///
/// # Errors
/// [`ClaudeError::InvalidConfig`] for invalid UUIDs / empty session;
/// [`ClaudeError::TransportError`] if the source session is absent.
pub async fn fork_session_via_store(
    session_store: &dyn SessionStore,
    session_id: &str,
    directory: Option<&Path>,
    up_to_message_id: Option<&str>,
    title: Option<&str>,
) -> Result<ForkSessionResult, ClaudeError> {
    let uuid = require_uuid(session_id)?;
    if let Some(up) = up_to_message_id {
        if validate_uuid(up).is_none() {
            return Err(ClaudeError::InvalidConfig(format!(
                "Invalid up_to_message_id: {up}"
            )));
        }
    }
    let project_key = project_key_for_directory(directory);
    let src_key = SessionKey::new(project_key.clone(), uuid);
    let loaded = session_store
        .load(&src_key)
        .await?
        .filter(|v| !v.is_empty())
        .ok_or_else(|| ClaudeError::TransportError(format!("Session {session_id} not found")))?;

    // Partition into transcript entries + content-replacement records.
    let mut transcript: Vec<Map<String, Value>> = Vec::new();
    let mut content_replacements: Vec<Value> = Vec::new();
    for entry in &loaded {
        let entry_type = entry.get("type").and_then(Value::as_str);
        let has_uuid = entry.get("uuid").and_then(Value::as_str).is_some();
        match entry_type {
            Some(t) if TRANSCRIPT_TYPES.contains(&t) && has_uuid => {
                transcript.push(entry.clone());
            }
            Some("content-replacement")
                if entry.get("sessionId").and_then(Value::as_str) == Some(session_id) =>
            {
                if let Some(reps) = entry.get("replacements").and_then(Value::as_array) {
                    content_replacements.extend(reps.iter().cloned());
                }
            }
            _ => {}
        }
    }

    let raw_for_title = loaded.clone();
    let derive_title = move || derive_title_from_entries(&raw_for_title);

    let (forked_session_id, lines) = build_fork_lines(
        transcript,
        &content_replacements,
        session_id,
        up_to_message_id,
        title,
        derive_title,
    )?;

    let dst_key = SessionKey::new(project_key, forked_session_id.clone());
    let entries: Vec<SessionStoreEntry> = lines
        .iter()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .map(json_object)
        .collect();
    session_store.append(&dst_key, entries).await?;
    Ok(ForkSessionResult {
        session_id: forked_session_id,
    })
}

/// Mirror the disk path's head/tail title scan over already-parsed entries.
fn derive_title_from_entries(raw: &[SessionStoreEntry]) -> Option<String> {
    let mut custom: Option<String> = None;
    let mut ai: Option<String> = None;
    for e in raw {
        if let Some(ct) = e.get("customTitle").and_then(Value::as_str) {
            if !ct.is_empty() {
                custom = Some(ct.to_string());
            }
        }
        if let Some(at) = e.get("aiTitle").and_then(Value::as_str) {
            if !at.is_empty() {
                ai = Some(at.to_string());
            }
        }
    }
    if let Some(c) = custom {
        return Some(c);
    }
    if let Some(a) = ai {
        return Some(a);
    }
    // First-prompt fallback over a re-serialized JSONL string.
    let mut jsonl = String::new();
    for e in raw {
        jsonl.push_str(&compact_line(&Value::Object(e.clone())));
        jsonl.push('\n');
    }
    let p = extract_first_prompt_from_head(&jsonl);
    (!p.is_empty()).then_some(p)
}

// ---------------------------------------------------------------------------
// Fork parsing helpers
// ---------------------------------------------------------------------------

/// Parse JSONL content into transcript entries + content-replacement records.
fn parse_fork_transcript(
    content: &[u8],
    session_id: &str,
) -> (Vec<Map<String, Value>>, Vec<Value>) {
    let text = String::from_utf8_lossy(content);
    let mut transcript = Vec::new();
    let mut content_replacements = Vec::new();

    for line in text.split('\n') {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(Value::Object(entry)) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let entry_type = entry.get("type").and_then(Value::as_str);
        let has_uuid = entry.get("uuid").and_then(Value::as_str).is_some();
        match entry_type {
            Some(t) if TRANSCRIPT_TYPES.contains(&t) && has_uuid => transcript.push(entry),
            Some("content-replacement")
                if entry.get("sessionId").and_then(Value::as_str) == Some(session_id) =>
            {
                if let Some(reps) = entry.get("replacements").and_then(Value::as_array) {
                    content_replacements.extend(reps.iter().cloned());
                }
            }
            _ => {}
        }
    }
    (transcript, content_replacements)
}

// ---------------------------------------------------------------------------
// Filesystem lookup + append
// ---------------------------------------------------------------------------

fn find_session_file(session_id: &str, directory: Option<&str>) -> Option<PathBuf> {
    find_session_file_with_dir(session_id, directory).map(|(p, _)| p)
}

/// Find a session file and its containing project directory (non-empty file).
fn find_session_file_with_dir(
    session_id: &str,
    directory: Option<&str>,
) -> Option<(PathBuf, PathBuf)> {
    let file_name = format!("{session_id}.jsonl");

    let try_dir = |project_dir: &Path| -> Option<(PathBuf, PathBuf)> {
        let path = project_dir.join(&file_name);
        match std::fs::metadata(&path) {
            Ok(m) if m.len() > 0 => Some((path, project_dir.to_path_buf())),
            _ => None,
        }
    };

    if let Some(dir) = directory.filter(|d| !d.is_empty()) {
        let canonical = canonicalize_path(dir);
        if let Some(project_dir) = find_project_dir(&canonical) {
            if let Some(r) = try_dir(&project_dir) {
                return Some(r);
            }
        }
        for wt in get_worktree_paths(&canonical) {
            if wt == canonical {
                continue;
            }
            if let Some(wt_project_dir) = find_project_dir(&wt) {
                if let Some(r) = try_dir(&wt_project_dir) {
                    return Some(r);
                }
            }
        }
        return None;
    }

    let projects_dir = get_projects_dir(None);
    let rd = std::fs::read_dir(&projects_dir).ok()?;
    for entry in rd.flatten() {
        if let Some(r) = try_dir(&entry.path()) {
            return Some(r);
        }
    }
    None
}

/// Append `data` (a single JSONL line, newline included) to an existing file.
///
/// Uses append-only opens without create so a missing file surfaces as
/// not-found; a 0-byte file is treated as "not here, keep searching".
fn append_to_session(
    session_id: &str,
    data: &str,
    directory: Option<&str>,
) -> Result<(), ClaudeError> {
    let file_name = format!("{session_id}.jsonl");

    if let Some(dir) = directory.filter(|d| !d.is_empty()) {
        let canonical = canonicalize_path(dir);
        if let Some(project_dir) = find_project_dir(&canonical) {
            if try_append(&project_dir.join(&file_name), data)? {
                return Ok(());
            }
        }
        for wt in get_worktree_paths(&canonical) {
            if wt == canonical {
                continue;
            }
            if let Some(wt_project_dir) = find_project_dir(&wt) {
                if try_append(&wt_project_dir.join(&file_name), data)? {
                    return Ok(());
                }
            }
        }
        return Err(ClaudeError::TransportError(format!(
            "Session {session_id} not found in project directory for {dir}"
        )));
    }

    let projects_dir = get_projects_dir(None);
    let rd = std::fs::read_dir(&projects_dir).map_err(|_| {
        ClaudeError::TransportError(format!(
            "Session {session_id} not found (no projects directory)"
        ))
    })?;
    for entry in rd.flatten() {
        if try_append(&entry.path().join(&file_name), data)? {
            return Ok(());
        }
    }
    Err(ClaudeError::TransportError(format!(
        "Session {session_id} not found in any project directory"
    )))
}

/// Try appending to `path`. Returns `Ok(false)` for ENOENT / 0-byte files;
/// real write failures (permissions, disk full) propagate.
fn try_append(path: &Path, data: &str) -> Result<bool, ClaudeError> {
    let mut opts = OpenOptions::new();
    opts.append(true);
    let mut file = match opts.open(path) {
        Ok(f) => f,
        Err(e)
            if matches!(
                e.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
            ) =>
        {
            return Ok(false);
        }
        Err(e) => return Err(ClaudeError::TransportError(e.to_string())),
    };
    let meta = file
        .metadata()
        .map_err(|e| ClaudeError::TransportError(e.to_string()))?;
    if meta.len() == 0 {
        return Ok(false);
    }
    file.write_all(data.as_bytes())
        .map_err(|e| ClaudeError::TransportError(e.to_string()))?;
    Ok(true)
}

/// Write a new file with mode 0600, failing if it already exists.
fn write_new_file_0600(path: &Path, bytes: &[u8]) -> Result<(), ClaudeError> {
    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut file = opts
        .open(path)
        .map_err(|e| ClaudeError::TransportError(e.to_string()))?;
    file.write_all(bytes)
        .map_err(|e| ClaudeError::TransportError(e.to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Unicode sanitization
// ---------------------------------------------------------------------------

/// Sanitize a string by removing dangerous Unicode characters.
///
/// Iteratively applies NFKC normalization and strips format (Cf), private-use
/// (Co), unassigned (Cn) categories and explicit dangerous ranges until stable
/// (max 10 iterations), matching the TS/Python fallback path.
fn sanitize_unicode(value: &str) -> String {
    let mut current = value.to_string();
    for _ in 0..10 {
        let previous = current.clone();
        let normalized: String = previous.nfkc().collect();
        current = normalized
            .chars()
            .filter(|c| !is_dangerous_char(*c))
            .collect();
        if current == previous {
            break;
        }
    }
    current
}

/// Whether `c` should be stripped by [`sanitize_unicode`].
fn is_dangerous_char(c: char) -> bool {
    // Explicit dangerous ranges (mirrors `_UNICODE_STRIP_RE`).
    matches!(c,
        '\u{200b}'..='\u{200f}'   // zero-width, LTR/RTL marks
        | '\u{202a}'..='\u{202e}' // directional formatting
        | '\u{2066}'..='\u{2069}' // directional isolates
        | '\u{feff}'              // BOM
        | '\u{e000}'..='\u{f8ff}' // BMP private use
    ) || is_format_or_unassigned(c)
}

/// Approximate Cf (format) / Co (private use) / Cn (unassigned) detection
/// without a full Unicode database. Covers the ranges the CLI targets.
fn is_format_or_unassigned(c: char) -> bool {
    let cp = c as u32;
    matches!(cp,
        0x00ad                       // soft hyphen (Cf)
        | 0x0600..=0x0605            // Arabic number-format marks (Cf)
        | 0x061c                     // Arabic letter mark (Cf)
        | 0x06dd | 0x070f | 0x08e2   // format marks (Cf)
        | 0x180e                     // Mongolian vowel separator (Cf)
        | 0x200b..=0x200f            // zero-width / bidi (Cf)
        | 0x202a..=0x202e            // bidi embedding/override (Cf)
        | 0x2060..=0x2064            // word joiner / invisible operators (Cf)
        | 0x2066..=0x206f            // bidi isolates / deprecated (Cf)
        | 0xfeff                     // BOM / ZWNBSP (Cf)
        | 0xfff9..=0xfffb            // interlinear annotation (Cf)
        | 0xe0000..=0xe007f          // tag characters (Cf)
        | 0xe000..=0xf8ff            // BMP private use (Co)
        | 0xf0000..=0xffffd          // plane 15 private use (Co)
        | 0x0010_0000..=0x0010_fffd  // plane 16 private use (Co)
    )
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn require_uuid(session_id: &str) -> Result<String, ClaudeError> {
    // Canonical hyphenated UUIDs only — the id is used as a filesystem path
    // component and a store key; non-canonical spellings are rejected to match
    // the official `_validate_uuid`.
    validate_uuid(session_id)
        .map(str::to_string)
        .ok_or_else(|| ClaudeError::InvalidConfig(format!("Invalid session_id: {session_id}")))
}

fn no_messages_err(session_id: &str) -> ClaudeError {
    ClaudeError::InvalidConfig(format!("Session {session_id} has no messages to fork"))
}

fn not_found_message(session_id: &str, directory: Option<&str>) -> String {
    match directory.filter(|d| !d.is_empty()) {
        Some(dir) => format!("Session {session_id} not found in project directory for {dir}"),
        None => format!("Session {session_id} not found"),
    }
}

/// Compact single-line JSON with a trailing newline.
fn compact_line(value: &Value) -> String {
    let mut s = serde_json::to_string(value).unwrap_or_default();
    s.push('\n');
    s
}

/// Coerce a JSON value into a [`SessionStoreEntry`] object (empty if not object).
fn json_object(value: Value) -> SessionStoreEntry {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

/// Current UTC time as an ISO-8601 string with `Z` suffix.
fn iso_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = i64::try_from(now.as_secs()).unwrap_or(0);
    let millis = now.subsec_millis();
    format_iso8601(secs, millis)
}

/// Format epoch seconds + millis as `YYYY-MM-DDTHH:MM:SS.mmmZ`.
fn format_iso8601(epoch_secs: i64, millis: u32) -> String {
    let days = epoch_secs.div_euclid(86400);
    let secs_of_day = epoch_secs.rem_euclid(86400);
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;
    let second = secs_of_day % 60;

    // civil_from_days (Howard Hinnant).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sessions::store::InMemorySessionStore;
    use serde_json::json;

    const SRC: &str = "11111111-1111-4111-8111-111111111111";
    // Message UUIDs. Upstream validates `up_to_message_id` with the CLI UUID
    // regex, so transcript message ids (and any cutoff id) must be real UUIDs.
    const MSG_A: &str = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
    const MSG_B: &str = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
    const MSG_C: &str = "cccccccc-cccc-4ccc-8ccc-cccccccccccc";

    fn entry(v: Value) -> SessionStoreEntry {
        match v {
            Value::Object(m) => m,
            _ => panic!("expected object"),
        }
    }

    // Store-backed mutations derive the project key from the directory via
    // `project_key_for_directory(None)` (the sanitized cwd), exactly as the
    // upstream Python SDK does on both the write and read sides. The fixture
    // must read under the SAME key the mutation functions write under.
    fn key(store_uuid: &str) -> SessionKey {
        SessionKey::new(project_key_for_directory(None), store_uuid)
    }

    // Read a forked session under the same project key the fork wrote to.
    fn forked_key(session_id: impl ToString) -> SessionKey {
        SessionKey::new(project_key_for_directory(None), session_id)
    }

    // ------- unicode sanitization -------

    #[test]
    fn sanitize_unicode_strips_dangerous_chars() {
        let dirty = "hel\u{200b}lo\u{202e}world\u{feff}";
        assert_eq!(sanitize_unicode(dirty), "helloworld");
    }

    #[test]
    fn sanitize_unicode_preserves_normal_text() {
        assert_eq!(sanitize_unicode("normal tag-123"), "normal tag-123");
    }

    // ------- iso8601 roundtrip sanity -------

    #[test]
    fn iso_now_is_well_formed() {
        let s = iso_now();
        assert!(s.ends_with('Z'));
        assert_eq!(s.len(), 24); // YYYY-MM-DDTHH:MM:SS.mmmZ
        assert_eq!(format_iso8601(0, 0), "1970-01-01T00:00:00.000Z");
    }

    // ------- store-backed mutations -------

    #[tokio::test]
    async fn rename_via_store_appends_custom_title() {
        let store = InMemorySessionStore::new();
        rename_session_via_store(&store, SRC, "  New Title  ", None)
            .await
            .unwrap();
        let entries = store.get_entries(&key(SRC));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].get("type").unwrap(), "custom-title");
        assert_eq!(entries[0].get("customTitle").unwrap(), "New Title");
    }

    #[tokio::test]
    async fn rename_via_store_rejects_empty_title() {
        let store = InMemorySessionStore::new();
        let err = rename_session_via_store(&store, SRC, "   ", None)
            .await
            .unwrap_err();
        assert!(matches!(err, ClaudeError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn tag_via_store_sanitizes_and_clear() {
        let store = InMemorySessionStore::new();
        tag_session_via_store(&store, SRC, Some("exp\u{200b}"), None)
            .await
            .unwrap();
        tag_session_via_store(&store, SRC, None, None)
            .await
            .unwrap();
        let entries = store.get_entries(&key(SRC));
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].get("tag").unwrap(), "exp");
        assert_eq!(entries[1].get("tag").unwrap(), ""); // cleared
    }

    #[tokio::test]
    async fn delete_via_store_cascades_subagents() {
        let store = InMemorySessionStore::new();
        let main = key(SRC);
        let sub = SessionKey::with_subpath(
            main.project_key.clone(),
            main.session_id.clone(),
            "subagents/agent-x",
        )
        .unwrap();
        store
            .append(&main, vec![entry(json!({"type":"user","uuid":"a"}))])
            .await
            .unwrap();
        store
            .append(&sub, vec![entry(json!({"type":"user","uuid":"b"}))])
            .await
            .unwrap();
        delete_session_via_store(&store, SRC, None).await.unwrap();
        assert!(store.get_entries(&main).is_empty());
        assert!(store.get_entries(&sub).is_empty());
    }

    // ------- fork transform (via store) -------

    #[tokio::test]
    async fn fork_via_store_remaps_uuids_and_stamps_forked_from() {
        let store = InMemorySessionStore::new();
        store
            .append(
                &key(SRC),
                vec![
                    entry(json!({"type":"user","uuid":MSG_A,"parentUuid":null,"sessionId":SRC,"teamName":"team","message":{"content":"1"}})),
                    entry(json!({"type":"assistant","uuid":MSG_B,"parentUuid":MSG_A,"sessionId":SRC,"message":{"content":"2"}})),
                ],
            )
            .await
            .unwrap();

        let result = fork_session_via_store(&store, SRC, None, None, None)
            .await
            .unwrap();
        assert_ne!(result.session_id.clone(), SRC);

        let forked = store.get_entries(&forked_key(&result.session_id));
        // 2 messages + custom-title.
        assert_eq!(forked.len(), 3);
        // UUIDs remapped (not the originals).
        for e in &forked[..2] {
            let u = e.get("uuid").and_then(Value::as_str).unwrap();
            assert!(u != MSG_A && u != MSG_B);
            assert_eq!(
                e.get("sessionId").unwrap(),
                &json!(result.session_id.clone())
            );
            let ff = e.get("forkedFrom").unwrap();
            assert_eq!(ff.get("sessionId").unwrap(), SRC);
        }
        // teamName stripped from forked output.
        assert!(forked[0].get("teamName").is_none());
        // parent chain preserved: second entry's parentUuid == first entry's uuid.
        let first_uuid = forked[0].get("uuid").unwrap();
        assert_eq!(forked[1].get("parentUuid").unwrap(), first_uuid);
        // title entry ends with " (fork)".
        let title = forked[2]
            .get("customTitle")
            .and_then(Value::as_str)
            .unwrap();
        assert!(title.ends_with("(fork)"));
    }

    #[tokio::test]
    async fn fork_via_store_remaps_or_nulls_logical_parent() {
        // Upstream `new_logical_parent = uuid_mapping.get(lp) if lp else lp`:
        //  - a logicalParentUuid pointing INSIDE the fork is remapped to the
        //    new uuid;
        //  - one pointing OUTSIDE (unmapped) resolves to null, NOT the stale
        //    original;
        //  - a falsy (empty/missing) value is preserved verbatim.
        let store = InMemorySessionStore::new();
        store
            .append(
                &key(SRC),
                vec![
                    // A: no logical parent.
                    entry(json!({
                        "type":"user","uuid":MSG_A,"parentUuid":null,"sessionId":SRC,
                        "message":{"content":"a"}
                    })),
                    // B: logicalParentUuid points at A (in-fork → remapped).
                    entry(json!({
                        "type":"assistant","uuid":MSG_B,"parentUuid":MSG_A,"sessionId":SRC,
                        "logicalParentUuid":MSG_A,"message":{"content":"b"}
                    })),
                    // C: logicalParentUuid points at an id NOT in the transcript
                    //    (out-of-fork → null).
                    entry(json!({
                        "type":"assistant","uuid":MSG_C,"parentUuid":MSG_B,"sessionId":SRC,
                        "logicalParentUuid":"deadbeef-dead-4bee-8bee-deadbeefdead",
                        "message":{"content":"c"}
                    })),
                ],
            )
            .await
            .unwrap();

        let result = fork_session_via_store(&store, SRC, None, None, None)
            .await
            .unwrap();
        let forked = store.get_entries(&forked_key(&result.session_id));
        // 3 messages + custom-title.
        assert_eq!(forked.len(), 4);
        let a_uuid = forked[0].get("uuid").and_then(Value::as_str).unwrap();
        // B's logicalParentUuid remapped to A's new uuid.
        assert_eq!(
            forked[1].get("logicalParentUuid").and_then(Value::as_str),
            Some(a_uuid),
            "in-fork logicalParentUuid remapped to the new uuid"
        );
        // C's out-of-fork logicalParentUuid is null, not the stale original.
        assert_eq!(
            forked[2].get("logicalParentUuid"),
            Some(&Value::Null),
            "unmapped logicalParentUuid resolves to null"
        );
    }

    #[tokio::test]
    async fn fork_via_store_cutoff_and_explicit_title() {
        let store = InMemorySessionStore::new();
        store
            .append(
                &key(SRC),
                vec![
                    entry(json!({"type":"user","uuid":MSG_A,"parentUuid":null,"sessionId":SRC,"message":{"content":"1"}})),
                    entry(json!({"type":"assistant","uuid":MSG_B,"parentUuid":MSG_A,"sessionId":SRC,"message":{"content":"2"}})),
                    entry(json!({"type":"user","uuid":MSG_C,"parentUuid":MSG_B,"sessionId":SRC,"message":{"content":"3"}})),
                ],
            )
            .await
            .unwrap();
        let result = fork_session_via_store(&store, SRC, None, Some(MSG_B), Some("Custom"))
            .await
            .unwrap();
        let forked = store.get_entries(&forked_key(&result.session_id));
        // cutoff at MSG_B => 2 messages + title.
        assert_eq!(forked.len(), 3);
        assert_eq!(
            forked[2].get("customTitle").and_then(Value::as_str),
            Some("Custom")
        );
    }

    #[tokio::test]
    async fn fork_via_store_missing_message_errors() {
        let store = InMemorySessionStore::new();
        store
            .append(
                &key(SRC),
                vec![entry(json!({"type":"user","uuid":MSG_A,"parentUuid":null,"sessionId":SRC,"message":{"content":"1"}}))],
            )
            .await
            .unwrap();
        let err = fork_session_via_store(&store, SRC, None, Some(MSG_C), None)
            .await
            .unwrap_err();
        assert!(matches!(err, ClaudeError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn fork_via_store_not_found() {
        let store = InMemorySessionStore::new();
        let err = fork_session_via_store(&store, SRC, None, None, None)
            .await
            .unwrap_err();
        assert!(matches!(err, ClaudeError::TransportError(_)));
    }

    #[test]
    fn invalid_uuid_rejected_across_apis() {
        assert!(matches!(
            rename_session("bad", "t", None),
            Err(ClaudeError::InvalidConfig(_))
        ));
        assert!(matches!(
            tag_session("bad", Some("t"), None),
            Err(ClaudeError::InvalidConfig(_))
        ));
        assert!(matches!(
            delete_session("bad", None),
            Err(ClaudeError::InvalidConfig(_))
        ));
        assert!(matches!(
            fork_session("bad", None, None, None),
            Err(ClaudeError::InvalidConfig(_))
        ));
    }

    // ------- filesystem fork + delete cascade (env-gated) -------

    fn env_lock() -> tokio::sync::MutexGuard<'static, ()> {
        crate::sessions::filesystem::TEST_ENV_LOCK.blocking_lock()
    }

    #[test]
    fn filesystem_rename_tag_and_fork_and_delete() {
        let _guard = env_lock();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("CLAUDE_CONFIG_DIR", tmp.path());
        let cwd = "/tmp/mutations-project";
        let project_dir = get_projects_dir(None)
            .join(crate::sessions::key::sanitize_path(&canonicalize_path(cwd)));
        std::fs::create_dir_all(&project_dir).unwrap();

        let content = [
            json!({"type":"user","uuid":"a","parentUuid":null,"sessionId":SRC,"cwd":cwd,"message":{"content":"first prompt"}}).to_string(),
            json!({"type":"assistant","uuid":"b","parentUuid":"a","sessionId":SRC,"message":{"content":"reply"}}).to_string(),
        ]
        .join("\n");
        std::fs::write(project_dir.join(format!("{SRC}.jsonl")), content).unwrap();
        // subagent dir to verify delete cascade.
        let sub_dir = project_dir.join(SRC).join("subagents");
        std::fs::create_dir_all(&sub_dir).unwrap();
        std::fs::write(sub_dir.join("agent-x.jsonl"), "{}").unwrap();

        // rename appends a custom-title line.
        rename_session(SRC, "Renamed", Some(cwd)).unwrap();
        let after = std::fs::read_to_string(project_dir.join(format!("{SRC}.jsonl"))).unwrap();
        assert!(after.contains("\"customTitle\":\"Renamed\""));

        // tag appends a tag line.
        tag_session(SRC, Some("mytag"), Some(cwd)).unwrap();
        let after = std::fs::read_to_string(project_dir.join(format!("{SRC}.jsonl"))).unwrap();
        assert!(after.contains("\"type\":\"tag\""));

        // fork writes a new sibling file with remapped ids.
        let fork = fork_session(SRC, Some(cwd), None, None).unwrap();
        let fork_path = project_dir.join(format!("{}.jsonl", fork.session_id));
        assert!(fork_path.exists());
        let fork_content = std::fs::read_to_string(&fork_path).unwrap();
        assert!(fork_content.contains("\"forkedFrom\""));
        assert!(!fork_content.contains("\"uuid\":\"a\""));

        // delete removes main file + subagents dir.
        delete_session(SRC, Some(cwd)).unwrap();
        assert!(!project_dir.join(format!("{SRC}.jsonl")).exists());
        assert!(!project_dir.join(SRC).exists());

        std::env::remove_var("CLAUDE_CONFIG_DIR");
    }
}
