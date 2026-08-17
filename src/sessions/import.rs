//! Replay a local on-disk session transcript into a [`SessionStore`].
//!
//! The inverse of resume materialization: read the local
//! `~/.claude/projects/<dir>/<sessionId>.jsonl` (plus subagent transcripts and
//! `.meta.json` sidecars) and replay each line into `store.append()` in bounded
//! batches. Ports the official Python `_internal/session_import.py`.
//!
//! The destination `project_key` is the on-disk project directory name — the
//! same key `file_path_to_session_key` (and thus the mirror batcher) produces —
//! so an imported session is indistinguishable from a live-mirrored one.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};
use uuid::Uuid;

use crate::error::ClaudeError;
use crate::sessions::filesystem::{resolve_session_file_path, validate_uuid};
use crate::sessions::mirror::{MAX_PENDING_BYTES, MAX_PENDING_ENTRIES};
use crate::sessions::store::{SessionStore, SessionStoreEntry};
use crate::sessions::SessionKey;

/// Replay a local session transcript into a [`SessionStore`].
///
/// Streams the on-disk JSONL line-by-line and calls `store.append(key, batch)`
/// every `batch_size` entries (or [`MAX_PENDING_BYTES`] of line bytes, whichever
/// comes first). Subagent transcripts under `<sessionId>/subagents/**` and their
/// `.meta.json` sidecars are imported too when `include_subagents` is set.
///
/// # Arguments
/// * `batch_size` — max entries per `append` call; `0` falls back to the default
///   [`MAX_PENDING_ENTRIES`] (500).
///
/// # Errors
/// * [`ClaudeError::InvalidConfig`] if `session_id` is not a valid UUID.
/// * [`ClaudeError::TransportError`] if the session JSONL cannot be found or a
///   file read fails.
/// * [`ClaudeError::SerializationError`] for malformed JSONL lines / sidecars.
/// * Adapter errors propagate from `append`.
pub async fn import_session_to_store(
    session_id: &str,
    store: &dyn SessionStore,
    directory: Option<&str>,
    include_subagents: bool,
    batch_size: usize,
) -> Result<(), ClaudeError> {
    let uuid = validate_uuid(session_id)
        .ok_or_else(|| ClaudeError::InvalidConfig(format!("Invalid session_id: {session_id}")))?;
    let session_uuid = Uuid::parse_str(uuid)
        .map_err(|_| ClaudeError::InvalidConfig(format!("Invalid session_id: {session_id}")))?;

    let resolved = resolve_session_file_path(session_id, directory)
        .ok_or_else(|| ClaudeError::TransportError(format!("Session {session_id} not found")))?;

    // Key under the on-disk project directory name — matches
    // file_path_to_session_key() even when the resolver's search or worktree
    // fallback found the file somewhere other than `directory`.
    let project_key = resolved
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .ok_or_else(|| {
            ClaudeError::TransportError(format!(
                "Session {session_id} resolved to a path with no project directory"
            ))
        })?
        .to_string();

    let batch_size = if batch_size == 0 {
        MAX_PENDING_ENTRIES
    } else {
        batch_size
    };

    let main_key = SessionKey::new(project_key.clone(), session_uuid);
    append_jsonl_file_in_batches(&resolved, &main_key, store, batch_size).await?;

    if !include_subagents {
        return Ok(());
    }

    // Subagent transcripts live at <projectDir>/<sessionId>/subagents/**.
    let session_dir = resolved.with_extension("");
    let subagents_dir = session_dir.join("subagents");
    for file_path in collect_jsonl_files(&subagents_dir) {
        // subpath is the path relative to session_dir, '/'-joined, sans .jsonl:
        // e.g. subagents/agent-abc or subagents/workflows/run-1/agent-def.
        let Some(subpath) = relative_subpath(&session_dir, &file_path) else {
            continue;
        };
        let sub_key = SessionKey::with_subpath(project_key.clone(), session_uuid, subpath)?;
        append_jsonl_file_in_batches(&file_path, &sub_key, store, batch_size).await?;

        // The on-disk .jsonl does NOT contain agent_metadata entries — those
        // live only in the .meta.json sidecar. Import it so resume can recreate
        // it and resumed subagents keep their agentType/worktreePath.
        let meta_path = meta_sidecar_path(&file_path);
        match std::fs::read_to_string(&meta_path) {
            Ok(text) => {
                let meta: Value = serde_json::from_str(&text)?;
                let mut meta_entry: SessionStoreEntry = Map::new();
                meta_entry.insert("type".into(), Value::String("agent_metadata".into()));
                // Upstream does `meta_entry.update(meta)`, which raises for a
                // non-object sidecar. Route the same failure rather than
                // silently importing a bare `{"type":"agent_metadata"}`.
                match meta {
                    Value::Object(fields) => {
                        for (k, v) in fields {
                            meta_entry.insert(k, v);
                        }
                    }
                    other => {
                        return Err(ClaudeError::MessageParse {
                            message: format!(
                                "agent metadata sidecar {} is not a JSON object",
                                meta_path.display()
                            ),
                            data: Some(other),
                        });
                    }
                }
                store.append(&sub_key, vec![meta_entry]).await?;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(ClaudeError::TransportError(e.to_string())),
        }
    }

    Ok(())
}

/// Stream-read a JSONL file, flushing to `append` in bounded batches.
///
/// Flushes every `batch_size` entries or [`MAX_PENDING_BYTES`] of line text,
/// whichever comes first. Blank lines are skipped.
async fn append_jsonl_file_in_batches(
    file_path: &Path,
    key: &SessionKey,
    store: &dyn SessionStore,
    batch_size: usize,
) -> Result<(), ClaudeError> {
    let file = File::open(file_path).map_err(|e| ClaudeError::TransportError(e.to_string()))?;
    let reader = BufReader::new(file);

    let mut batch: Vec<SessionStoreEntry> = Vec::new();
    let mut nbytes = 0usize;

    for line in reader.lines() {
        let line = line.map_err(|e| ClaudeError::TransportError(e.to_string()))?;
        let line = line.trim_end_matches('\n');
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)?;
        let entry = match value {
            Value::Object(map) => map,
            // A non-object JSONL line is malformed for the store contract.
            other => {
                return Err(ClaudeError::MessageParse {
                    message: "session transcript line is not a JSON object".into(),
                    data: Some(other),
                });
            }
        };
        batch.push(entry);
        nbytes += line.len();
        if batch.len() >= batch_size || nbytes >= MAX_PENDING_BYTES {
            store.append(key, std::mem::take(&mut batch)).await?;
            nbytes = 0;
        }
    }

    if !batch.is_empty() {
        store.append(key, batch).await?;
    }
    Ok(())
}

/// Recursively collect `*.jsonl` files under `base_dir`, sorted per directory.
///
/// Returns an empty vec if `base_dir` does not exist. Deterministic ordering
/// across platforms.
fn collect_jsonl_files(base_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_jsonl(base_dir, &mut out);
    out
}

fn walk_jsonl(base_dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(base_dir) else {
        return;
    };
    let mut dirents: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    dirents.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    for entry in dirents {
        if entry.is_dir() {
            walk_jsonl(&entry, out);
        } else if entry.is_file() && entry.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            out.push(entry);
        }
    }
}

/// Build the `/`-joined subpath (sans `.jsonl`) of `file_path` under `session_dir`.
fn relative_subpath(session_dir: &Path, file_path: &Path) -> Option<String> {
    let rel = file_path.strip_prefix(session_dir).ok()?;
    let mut parts: Vec<String> = rel
        .components()
        .filter_map(|c| c.as_os_str().to_str().map(str::to_string))
        .collect();
    let last = parts.last_mut()?;
    if let Some(stripped) = last.strip_suffix(".jsonl") {
        *last = stripped.to_string();
    }
    Some(parts.join("/"))
}

/// `<name>.jsonl` -> `<name>.meta.json` alongside the transcript.
fn meta_sidecar_path(file_path: &Path) -> PathBuf {
    let name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let base = name.strip_suffix(".jsonl").unwrap_or(name);
    file_path.with_file_name(format!("{base}.meta.json"))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sessions::key::project_key_for_directory;
    use crate::sessions::store::InMemorySessionStore;
    use serde_json::json;

    const SID: &str = "11111111-1111-4111-8111-111111111111";

    async fn env_lock() -> tokio::sync::MutexGuard<'static, ()> {
        crate::sessions::filesystem::TEST_ENV_LOCK.lock().await
    }

    fn write_lines(path: &Path, lines: &[Value]) {
        let content = lines
            .iter()
            .map(|l| serde_json::to_string(l).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(path, content).unwrap();
    }

    #[tokio::test]
    async fn import_invalid_uuid_errors() {
        let store = InMemorySessionStore::new();
        let err = import_session_to_store("nope", &store, None, true, 0)
            .await
            .unwrap_err();
        assert!(matches!(err, ClaudeError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn import_not_found_errors() {
        let _guard = env_lock().await;
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("CLAUDE_CONFIG_DIR", tmp.path());
        let store = InMemorySessionStore::new();
        let err = import_session_to_store(SID, &store, Some("/tmp/missing-proj"), true, 0)
            .await
            .unwrap_err();
        assert!(matches!(err, ClaudeError::TransportError(_)));
        std::env::remove_var("CLAUDE_CONFIG_DIR");
    }

    #[tokio::test]
    async fn import_main_transcript_key_parity_and_batching() {
        let _guard = env_lock().await;
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("CLAUDE_CONFIG_DIR", tmp.path());
        let cwd = "/tmp/import-project";
        let project_dir = project_dir_for(cwd);
        std::fs::create_dir_all(&project_dir).unwrap();

        // 5 entries, batch_size=2 => 3 append calls; all land under the key.
        let entries: Vec<Value> = (0..5)
            .map(|i| json!({"type":"user","uuid":format!("u{i}"),"message":{"content":i}}))
            .collect();
        write_lines(&project_dir.join(format!("{SID}.jsonl")), &entries);

        let store = InMemorySessionStore::new();
        import_session_to_store(SID, &store, Some(cwd), true, 2)
            .await
            .unwrap();

        let pk = project_key_for_directory(Some(Path::new(&canonical(cwd))));
        let key = SessionKey::new(pk, Uuid::parse_str(SID).unwrap());
        let stored = store.get_entries(&key);
        assert_eq!(stored.len(), 5);
        std::env::remove_var("CLAUDE_CONFIG_DIR");
    }

    #[tokio::test]
    async fn import_non_object_line_fails_visibly() {
        let _guard = env_lock().await;
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("CLAUDE_CONFIG_DIR", tmp.path());
        let cwd = "/tmp/import-bad-line";
        let project_dir = project_dir_for(cwd);
        std::fs::create_dir_all(&project_dir).unwrap();
        // A scalar JSONL line cannot be a store entry (which is a JSON object);
        // it must fail visibly rather than being silently dropped/defaulted.
        write_lines(
            &project_dir.join(format!("{SID}.jsonl")),
            &[json!({"type":"user","uuid":"u0"}), json!(42)],
        );
        let store = InMemorySessionStore::new();
        let err = import_session_to_store(SID, &store, Some(cwd), false, 0)
            .await
            .unwrap_err();
        assert!(
            matches!(err, ClaudeError::MessageParse { .. }),
            "non-object JSONL line must route to MessageParse, got {err:?}"
        );
        std::env::remove_var("CLAUDE_CONFIG_DIR");
    }

    #[tokio::test]
    async fn import_scalar_sidecar_fails_visibly() {
        let _guard = env_lock().await;
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("CLAUDE_CONFIG_DIR", tmp.path());
        let cwd = "/tmp/import-bad-sidecar";
        let project_dir = project_dir_for(cwd);
        let subagents = project_dir.join(SID).join("subagents");
        std::fs::create_dir_all(&subagents).unwrap();
        // Valid main transcript + subagent transcript.
        write_lines(
            &project_dir.join(format!("{SID}.jsonl")),
            &[json!({"type":"user","uuid":"u0"})],
        );
        write_lines(
            &subagents.join("agent-abc.jsonl"),
            &[json!({"type":"assistant","uuid":"a0"})],
        );
        // Scalar sidecar: upstream's `meta_entry.update(meta)` raises for a
        // non-object; we route the same failure instead of importing a bare
        // `{"type":"agent_metadata"}`.
        std::fs::write(subagents.join("agent-abc.meta.json"), "42").unwrap();
        let store = InMemorySessionStore::new();
        let err = import_session_to_store(SID, &store, Some(cwd), true, 0)
            .await
            .unwrap_err();
        assert!(
            matches!(err, ClaudeError::MessageParse { .. }),
            "scalar sidecar must route to MessageParse, got {err:?}"
        );
        std::env::remove_var("CLAUDE_CONFIG_DIR");
    }

    fn project_dir_for(cwd: &str) -> PathBuf {
        crate::sessions::filesystem::get_project_dir(cwd)
    }

    fn canonical(cwd: &str) -> String {
        crate::sessions::key::canonicalize_path(cwd)
    }

    #[tokio::test]
    async fn import_subagents_with_nesting_and_sidecar() {
        let _guard = env_lock().await;
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("CLAUDE_CONFIG_DIR", tmp.path());
        let cwd = "/tmp/import-sub-project";
        let project_dir = project_dir_for(cwd);
        std::fs::create_dir_all(&project_dir).unwrap();
        write_lines(
            &project_dir.join(format!("{SID}.jsonl")),
            &[json!({"type":"user","uuid":"m","message":{"content":"main"}})],
        );

        let session_dir = project_dir.join(SID);
        let flat = session_dir.join("subagents");
        let nested = flat.join("workflows").join("run-1");
        std::fs::create_dir_all(&nested).unwrap();
        write_lines(
            &flat.join("agent-abc.jsonl"),
            &[json!({"type":"user","uuid":"s1","message":{"content":"sub"}})],
        );
        // sidecar metadata for the flat agent.
        std::fs::write(
            flat.join("agent-abc.meta.json"),
            json!({"agentType":"researcher","worktreePath":"/wt"}).to_string(),
        )
        .unwrap();
        write_lines(
            &nested.join("agent-def.jsonl"),
            &[json!({"type":"assistant","uuid":"s2","message":{"content":"deep"}})],
        );

        let store = InMemorySessionStore::new();
        import_session_to_store(SID, &store, Some(cwd), true, 0)
            .await
            .unwrap();

        let pk = project_key_for_directory(Some(Path::new(&canonical(cwd))));
        let sid = Uuid::parse_str(SID).unwrap();

        // flat subagent key + sidecar agent_metadata entry appended.
        let flat_key = SessionKey::with_subpath(pk.clone(), sid, "subagents/agent-abc").unwrap();
        let flat_entries = store.get_entries(&flat_key);
        assert_eq!(flat_entries.len(), 2); // transcript + agent_metadata
        let meta = flat_entries
            .iter()
            .find(|e| e.get("type").and_then(Value::as_str) == Some("agent_metadata"))
            .unwrap();
        assert_eq!(meta.get("agentType").unwrap(), "researcher");
        assert_eq!(meta.get("worktreePath").unwrap(), "/wt");

        // nested subagent key uses '/'-joined subpath.
        let nested_key =
            SessionKey::with_subpath(pk, sid, "subagents/workflows/run-1/agent-def").unwrap();
        assert_eq!(store.get_entries(&nested_key).len(), 1);
        std::env::remove_var("CLAUDE_CONFIG_DIR");
    }

    #[tokio::test]
    async fn import_excludes_subagents_when_flag_false() {
        let _guard = env_lock().await;
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("CLAUDE_CONFIG_DIR", tmp.path());
        let cwd = "/tmp/import-nosub";
        let project_dir = project_dir_for(cwd);
        std::fs::create_dir_all(&project_dir).unwrap();
        write_lines(
            &project_dir.join(format!("{SID}.jsonl")),
            &[json!({"type":"user","uuid":"m","message":{"content":"main"}})],
        );
        let flat = project_dir.join(SID).join("subagents");
        std::fs::create_dir_all(&flat).unwrap();
        write_lines(
            &flat.join("agent-abc.jsonl"),
            &[json!({"type":"user","uuid":"s1","message":{"content":"sub"}})],
        );

        let store = InMemorySessionStore::new();
        import_session_to_store(SID, &store, Some(cwd), false, 0)
            .await
            .unwrap();

        let pk = project_key_for_directory(Some(Path::new(&canonical(cwd))));
        let sid = Uuid::parse_str(SID).unwrap();
        assert_eq!(store.size(), 1); // only main transcript
        assert!(store
            .get_entries(&SessionKey::with_subpath(pk, sid, "subagents/agent-abc").unwrap())
            .is_empty());
        std::env::remove_var("CLAUDE_CONFIG_DIR");
    }

    #[test]
    fn relative_subpath_strips_jsonl_and_joins_slash() {
        let session_dir = Path::new("/proj/sess");
        let file = Path::new("/proj/sess/subagents/workflows/run-1/agent-x.jsonl");
        assert_eq!(
            relative_subpath(session_dir, file).as_deref(),
            Some("subagents/workflows/run-1/agent-x")
        );
    }

    #[test]
    fn meta_sidecar_path_derivation() {
        let file = Path::new("/a/agent-x.jsonl");
        assert_eq!(meta_sidecar_path(file), Path::new("/a/agent-x.meta.json"));
    }
}
