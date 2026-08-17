//! Filesystem-backed session discovery and transcript reading.
//!
//! Ports the official Python `_internal/sessions.py` local-disk paths:
//! project directory scanning, path/worktree discovery, head/tail metadata
//! extraction without a full parse, `parentUuid` chain reconstruction,
//! corrupt-line tolerance, sidechain/metadata filtering, pagination, and the
//! subagent transcript APIs.
//!
//! The `SessionStore`-backed variants live in [`crate::sessions`] modules owned
//! by other components; this module owns the on-disk behavior only.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use crate::sessions::key::{self, MAX_SANITIZED_LENGTH};

/// Size of the head/tail buffer for lite metadata reads (64 KiB).
pub(crate) const LITE_READ_BUF_SIZE: usize = 65536;

/// Transcript entry types that carry `uuid` + `parentUuid` chain links.
const TRANSCRIPT_ENTRY_TYPES: [&str; 5] = ["user", "assistant", "progress", "system", "attachment"];

/// Process-wide lock serializing tests that mutate `CLAUDE_CONFIG_DIR`.
///
/// Env is process-global; the filesystem/mutations/import test modules all set
/// `CLAUDE_CONFIG_DIR`, so they must share one guard to avoid cross-test races.
/// A [`tokio::sync::Mutex`] is used so async tests can hold the guard across
/// `.await` points without tripping `clippy::await_holding_lock`; synchronous
/// tests take it via `blocking_lock()`.
#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::LazyLock<tokio::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

// ---------------------------------------------------------------------------
// SessionMessage — public transcript message shape
// ---------------------------------------------------------------------------

/// A single conversation message reconstructed from a session transcript.
///
/// Mirrors the official `SessionMessage`: user/assistant messages returned by
/// [`get_session_messages`] and the subagent variants, in chronological order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionMessage {
    /// `"user"` or `"assistant"`.
    pub r#type: SessionMessageType,
    /// Message UUID (the transcript entry's `uuid`).
    pub uuid: String,
    /// Session ID the message belongs to (the entry's `sessionId`).
    pub session_id: String,
    /// The raw `message` payload from the transcript entry, if present.
    pub message: Option<Value>,
    /// Parent tool-use id; always `None` for transcript reads (matches upstream).
    pub parent_tool_use_id: Option<String>,
}

/// The role of a [`SessionMessage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionMessageType {
    /// A user-authored message.
    User,
    /// An assistant-authored message.
    Assistant,
}

// ---------------------------------------------------------------------------
// UUID validation
// ---------------------------------------------------------------------------

/// Returns `Some(s)` if `s` is a valid canonical hyphenated UUID, else `None`.
///
/// Mirrors the official `_UUID_RE`
/// (`^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`,
/// case-insensitive). Deliberately stricter than [`uuid::Uuid::parse_str`],
/// which also accepts braced, URN, and unhyphenated spellings — the on-disk
/// and store key layouts only ever use the canonical hyphenated form, and
/// accepting other spellings would let non-canonical ids address the same
/// session under two different keys.
pub(crate) fn validate_uuid(maybe_uuid: &str) -> Option<&str> {
    // Group lengths for the 8-4-4-4-12 canonical layout.
    const GROUPS: [usize; 5] = [8, 4, 4, 4, 12];
    let mut groups = maybe_uuid.split('-');
    for expected in GROUPS {
        let group = groups.next()?;
        if group.len() != expected || !group.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
    }
    if groups.next().is_some() {
        return None;
    }
    Some(maybe_uuid)
}

// ---------------------------------------------------------------------------
// Config directories
// ---------------------------------------------------------------------------

fn nfc(s: &str) -> String {
    s.nfc().collect()
}

fn claude_config_home_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("CLAUDE_CONFIG_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(nfc(&dir));
        }
    }
    let home = home_dir().unwrap_or_else(|| PathBuf::from("."));
    PathBuf::from(nfc(&home.join(".claude").to_string_lossy()))
}

/// Best-effort home directory (`$HOME`, falling back to `%USERPROFILE%`).
fn home_dir() -> Option<PathBuf> {
    if let Ok(h) = std::env::var("HOME") {
        if !h.is_empty() {
            return Some(PathBuf::from(h));
        }
    }
    if let Ok(h) = std::env::var("USERPROFILE") {
        if !h.is_empty() {
            return Some(PathBuf::from(h));
        }
    }
    None
}

/// Returns the projects directory (`<config>/projects`).
///
/// `env_override` is consulted before the process environment so callers that
/// pass `CLAUDE_CONFIG_DIR` via `options.env` resolve the same directory the
/// subprocess will write to.
pub(crate) fn get_projects_dir(env_override: Option<&BTreeMap<String, String>>) -> PathBuf {
    if let Some(env) = env_override {
        if let Some(override_dir) = env.get("CLAUDE_CONFIG_DIR") {
            if !override_dir.is_empty() {
                return PathBuf::from(nfc(override_dir)).join("projects");
            }
        }
    }
    claude_config_home_dir().join("projects")
}

/// Returns the project directory for a project path (`<projects>/<sanitized>`).
pub(crate) fn get_project_dir(project_path: &str) -> PathBuf {
    get_projects_dir(None).join(key::sanitize_path(project_path))
}

/// Finds the on-disk project directory for a path.
///
/// Tolerates hash mismatches for long paths (>200 chars): the CLI (Bun) and the
/// SDK (this crate) may produce different hash suffixes, so we fall back to
/// prefix scanning when the exact directory does not exist.
pub(crate) fn find_project_dir(project_path: &str) -> Option<PathBuf> {
    let exact = get_project_dir(project_path);
    if exact.is_dir() {
        return Some(exact);
    }

    let sanitized = key::sanitize_path(project_path);
    if sanitized.len() <= MAX_SANITIZED_LENGTH {
        return None;
    }

    let prefix = &sanitized[..MAX_SANITIZED_LENGTH];
    let projects_dir = get_projects_dir(None);
    let with_dash = format!("{prefix}-");
    let entries = std::fs::read_dir(&projects_dir).ok()?;
    for entry in entries.flatten() {
        if entry.path().is_dir() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with(&with_dash) {
                    return Some(entry.path());
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// JSON string field extraction — no full parse, works on truncated lines
// ---------------------------------------------------------------------------

/// Unescape a JSON string value extracted as raw text.
fn unescape_json_string(raw: &str) -> String {
    if !raw.contains('\\') {
        return raw.to_string();
    }
    match serde_json::from_str::<String>(&format!("\"{raw}\"")) {
        Ok(s) => s,
        Err(_) => raw.to_string(),
    }
}

/// Extracts the first `"key":"value"` / `"key": "value"` string value.
pub(crate) fn extract_json_string_field(text: &str, key: &str) -> Option<String> {
    let patterns = [format!("\"{key}\":\""), format!("\"{key}\": \"")];
    let bytes = text.as_bytes();
    for pattern in &patterns {
        let Some(idx) = text.find(pattern.as_str()) else {
            continue;
        };
        let value_start = idx + pattern.len();
        let mut i = value_start;
        while i < bytes.len() {
            match bytes[i] {
                b'\\' => i += 2,
                b'"' => return Some(unescape_json_string(&text[value_start..i])),
                _ => i += 1,
            }
        }
    }
    None
}

/// Like [`extract_json_string_field`] but returns the LAST occurrence.
pub(crate) fn extract_last_json_string_field(text: &str, key: &str) -> Option<String> {
    let patterns = [format!("\"{key}\":\""), format!("\"{key}\": \"")];
    let bytes = text.as_bytes();
    let mut last_value: Option<String> = None;
    for pattern in &patterns {
        let mut search_from = 0;
        while let Some(rel) = text[search_from..].find(pattern.as_str()) {
            let idx = search_from + rel;
            let value_start = idx + pattern.len();
            let mut i = value_start;
            while i < bytes.len() {
                match bytes[i] {
                    b'\\' => i += 2,
                    b'"' => {
                        last_value = Some(unescape_json_string(&text[value_start..i]));
                        break;
                    }
                    _ => i += 1,
                }
            }
            search_from = i + 1;
            if search_from > bytes.len() {
                break;
            }
        }
    }
    last_value
}

// ---------------------------------------------------------------------------
// First prompt extraction from head chunk
// ---------------------------------------------------------------------------

/// Returns `true` if `result` marks an auto-generated / system message to skip
/// when looking for the first meaningful user prompt.
///
/// Mirrors the official `_SKIP_FIRST_PROMPT_PATTERN` (anchored at the start of
/// the stripped text):
/// `^(?:<local-command-stdout>|<session-start-hook>|<tick>|<goal>|`
/// `\[Request interrupted by user[^\]]*\]|`
/// `\s*<ide_opened_file>[\s\S]*</ide_opened_file>\s*$|`
/// `\s*<ide_selection>[\s\S]*</ide_selection>\s*$)`
fn is_skip_first_prompt(result: &str) -> bool {
    const PREFIXES: [&str; 4] = [
        "<local-command-stdout>",
        "<session-start-hook>",
        "<tick>",
        "<goal>",
    ];
    if PREFIXES.iter().any(|p| result.starts_with(p)) {
        return true;
    }
    // `[Request interrupted by user...]` — a leading bracketed banner with no
    // embedded `]`.
    if let Some(rest) = result.strip_prefix("[Request interrupted by user") {
        if let Some(close) = rest.find(']') {
            if !rest[..close].contains(']') {
                return true;
            }
        }
    }
    // `<ide_opened_file>...</ide_opened_file>` / `<ide_selection>...` wrapping
    // the ENTIRE (already `\n`->` ` collapsed, but not re-trimmed) result. The
    // upstream pattern allows surrounding whitespace via `\s*...\s*$`; the
    // caller passes the trimmed text, so a bare `.trim()` re-check is a safe
    // superset here.
    for (open, close) in [
        ("<ide_opened_file>", "</ide_opened_file>"),
        ("<ide_selection>", "</ide_selection>"),
    ] {
        let trimmed = result.trim();
        if trimmed.starts_with(open) && trimmed.ends_with(close) {
            return true;
        }
    }
    false
}

/// Extracts the `<command-name>NAME</command-name>` payload, if present.
fn extract_command_name(text: &str) -> Option<String> {
    let start = text.find("<command-name>")? + "<command-name>".len();
    let end_rel = text[start..].find("</command-name>")?;
    Some(text[start..start + end_rel].to_string())
}

/// Extracts the first meaningful user prompt from a JSONL head chunk.
///
/// Skips `tool_result`, `isMeta`, `isCompactSummary`, slash-command messages,
/// and auto-generated patterns. Truncates to 200 chars with an ellipsis.
pub(crate) fn extract_first_prompt_from_head(head: &str) -> String {
    let mut command_fallback = String::new();

    for line in head.split('\n') {
        if !line.contains("\"type\":\"user\"") && !line.contains("\"type\": \"user\"") {
            continue;
        }
        if line.contains("\"tool_result\"") {
            continue;
        }
        if line.contains("\"isMeta\":true") || line.contains("\"isMeta\": true") {
            continue;
        }
        if line.contains("\"isCompactSummary\":true") || line.contains("\"isCompactSummary\": true")
        {
            continue;
        }

        let Ok(entry) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(obj) = entry.as_object() else {
            continue;
        };
        if obj.get("type").and_then(Value::as_str) != Some("user") {
            continue;
        }
        let Some(message) = obj.get("message").and_then(Value::as_object) else {
            continue;
        };

        let mut texts: Vec<String> = Vec::new();
        match message.get("content") {
            Some(Value::String(s)) => texts.push(s.clone()),
            Some(Value::Array(blocks)) => {
                for block in blocks {
                    if let Some(b) = block.as_object() {
                        if b.get("type").and_then(Value::as_str) == Some("text") {
                            if let Some(t) = b.get("text").and_then(Value::as_str) {
                                texts.push(t.to_string());
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        for raw in texts {
            let result = raw.replace('\n', " ");
            let result = result.trim();
            if result.is_empty() {
                continue;
            }

            if let Some(cmd) = extract_command_name(result) {
                if command_fallback.is_empty() {
                    command_fallback = cmd;
                }
                continue;
            }

            if is_skip_first_prompt(result) {
                continue;
            }

            return truncate_prompt(result);
        }
    }

    command_fallback
}

/// Truncate a prompt to 200 chars (by `char`), appending an ellipsis.
fn truncate_prompt(result: &str) -> String {
    if result.chars().count() > 200 {
        let truncated: String = result.chars().take(200).collect();
        format!("{}\u{2026}", truncated.trim_end())
    } else {
        result.to_string()
    }
}

// ---------------------------------------------------------------------------
// File I/O — read head and tail of a file
// ---------------------------------------------------------------------------

/// Result of reading a session file's head, tail, mtime and size.
pub(crate) struct LiteSessionFile {
    pub mtime: i64,
    pub size: i64,
    pub head: String,
    pub tail: String,
}

/// Opens a session file, stats it, and reads head + tail (64 KiB each).
///
/// Returns `None` on any error or if the file is empty.
pub(crate) fn read_session_lite(file_path: &Path) -> Option<LiteSessionFile> {
    let mut f = File::open(file_path).ok()?;
    let meta = f.metadata().ok()?;
    let size = i64::try_from(meta.len()).unwrap_or(i64::MAX);
    let mtime = mtime_ms(&meta);

    let mut head_bytes = vec![0u8; LITE_READ_BUF_SIZE];
    let n = f.read(&mut head_bytes).ok()?;
    if n == 0 {
        return None;
    }
    head_bytes.truncate(n);
    let head = String::from_utf8_lossy(&head_bytes).into_owned();

    let file_len = meta.len();
    let tail_offset = file_len.saturating_sub(LITE_READ_BUF_SIZE as u64);
    let tail = if tail_offset == 0 {
        head.clone()
    } else {
        f.seek(SeekFrom::Start(tail_offset)).ok()?;
        let mut tail_bytes = vec![0u8; LITE_READ_BUF_SIZE];
        let tn = f.read(&mut tail_bytes).ok()?;
        tail_bytes.truncate(tn);
        String::from_utf8_lossy(&tail_bytes).into_owned()
    };

    Some(LiteSessionFile {
        mtime,
        size,
        head,
        tail,
    })
}

/// Modification time of `meta` in Unix epoch milliseconds.
fn mtime_ms(meta: &std::fs::Metadata) -> i64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

// ---------------------------------------------------------------------------
// Git worktree detection
// ---------------------------------------------------------------------------

/// Returns absolute worktree paths for the git repo containing `cwd`.
///
/// Returns an empty vec if git is unavailable or `cwd` is not in a repo.
pub(crate) fn get_worktree_paths(cwd: &str) -> Vec<String> {
    let output = Command::new("git")
        .args(["worktree", "list", "--porcelain"])
        .current_dir(cwd)
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() || output.stdout.is_empty() {
        return Vec::new();
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut paths = Vec::new();
    for line in stdout.split('\n') {
        if let Some(rest) = line.strip_prefix("worktree ") {
            paths.push(nfc(rest));
        }
    }
    paths
}

// ---------------------------------------------------------------------------
// Field extraction — shared by list_sessions and get_session_info
// ---------------------------------------------------------------------------

use crate::sessions::store::SDKSessionInfo;

/// Parses [`SDKSessionInfo`] fields from a lite session read.
///
/// Returns `None` for sidechain sessions or metadata-only sessions with no
/// extractable summary.
pub(crate) fn parse_session_info_from_lite(
    session_id: &str,
    lite: &LiteSessionFile,
    project_path: Option<&str>,
) -> Option<SDKSessionInfo> {
    let head = &lite.head;
    let tail = &lite.tail;

    // Check first line for sidechain sessions.
    let first_line = match head.find('\n') {
        Some(i) => &head[..i],
        None => head.as_str(),
    };
    if first_line.contains("\"isSidechain\":true") || first_line.contains("\"isSidechain\": true") {
        return None;
    }

    // customTitle wins over aiTitle; head fallback covers short sessions.
    // Each extraction is coerced through `nonempty` so an empty string value
    // behaves like Python's trailing `or None` and lets the fallback chain
    // continue (an empty title is not a title).
    let custom_title = nonempty(extract_last_json_string_field(tail, "customTitle"))
        .or_else(|| nonempty(extract_last_json_string_field(head, "customTitle")))
        .or_else(|| nonempty(extract_last_json_string_field(tail, "aiTitle")))
        .or_else(|| nonempty(extract_last_json_string_field(head, "aiTitle")));

    let first_prompt = nonempty(Some(extract_first_prompt_from_head(head)));

    let summary = custom_title
        .clone()
        .or_else(|| nonempty(extract_last_json_string_field(tail, "lastPrompt")))
        .or_else(|| nonempty(extract_last_json_string_field(tail, "summary")))
        .or_else(|| first_prompt.clone());

    // Skip metadata-only sessions (no title, no summary, no prompt). Matches
    // the Python `if not summary` guard — an empty summary drops the session.
    let summary = nonempty(summary)?;

    let git_branch = nonempty(extract_last_json_string_field(tail, "gitBranch"))
        .or_else(|| nonempty(extract_json_string_field(head, "gitBranch")));

    let session_cwd = nonempty(extract_json_string_field(head, "cwd"))
        .or_else(|| project_path.map(str::to_string));

    // Scope tag extraction to `{"type":"tag"` lines — a bare tail scan for
    // "tag" would match tool_use inputs (git tag, Docker tags, cloud resource
    // tags). Matches upstream's `ln.startswith('{"type":"tag"')` byte prefix;
    // `_entries_to_jsonl` hoists `type` first so store-serialized lines carry
    // the same prefix.
    let tag = tail
        .split('\n')
        .rev()
        .find(|ln| ln.starts_with("{\"type\":\"tag\""))
        .and_then(|tag_line| nonempty(extract_last_json_string_field(tag_line, "tag")));

    // created_at from the first ISO timestamp in the head (epoch ms).
    let created_at =
        extract_json_string_field(head, "timestamp").and_then(|ts| parse_iso8601_ms(&ts));

    Some(SDKSessionInfo {
        session_id: session_id.to_string(),
        summary,
        last_modified: lite.mtime,
        file_size: Some(lite.size),
        custom_title,
        first_prompt,
        git_branch,
        cwd: session_cwd,
        tag,
        created_at,
    })
}

/// Maps `Some("")` to `None`, leaving other values intact. Mirrors Python's
/// trailing `... or None` used to collapse empty extracted string fields.
fn nonempty(value: Option<String>) -> Option<String> {
    value.filter(|s| !s.is_empty())
}

/// Parse an ISO-8601 timestamp (with trailing `Z` or offset) to epoch ms.
// `yoe`/`doe`/`doy`/`era` are the canonical variable names from Howard
// Hinnant's `days_from_civil` algorithm; renaming them to satisfy the
// `similar_names` heuristic would obscure the well-known reference.
#[allow(clippy::similar_names)]
pub(crate) fn parse_iso8601_ms(ts: &str) -> Option<i64> {
    // Minimal parser: `YYYY-MM-DDTHH:MM:SS[.fff][Z|+HH:MM]`. Days-from-civil
    // algorithm avoids pulling in a date crate.
    let bytes = ts.as_bytes();
    if bytes.len() < 19 {
        return None;
    }
    // Structural separators must be exact: `YYYY-MM-DD` then `T`/space then
    // `HH:MM:SS`. `datetime.fromisoformat` rejects anything else.
    if bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    if bytes[10] != b'T' && bytes[10] != b' ' {
        return None;
    }
    if bytes[13] != b':' || bytes[16] != b':' {
        return None;
    }
    let year: i64 = ts.get(0..4)?.parse().ok()?;
    let month: i64 = ts.get(5..7)?.parse().ok()?;
    let day: i64 = ts.get(8..10)?.parse().ok()?;
    let hour: i64 = ts.get(11..13)?.parse().ok()?;
    let minute: i64 = ts.get(14..16)?.parse().ok()?;
    let second: i64 = ts.get(17..19)?.parse().ok()?;

    // Calendar validation — reject values `datetime` would refuse (month 13,
    // day 32, Feb 30, hour 24, ...). Matches `fromisoformat`'s ValueError.
    if !(1..=12).contains(&month) {
        return None;
    }
    let max_day = days_in_month(year, month)?;
    if !(1..=max_day).contains(&day) {
        return None;
    }
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }

    let mut millis = 0i64;
    let mut rest = &ts[19..];
    if let Some(frac) = rest.strip_prefix('.') {
        let digits: String = frac.chars().take_while(char::is_ascii_digit).collect();
        rest = &rest[digits.len() + 1..];
        let mut ms_str = digits;
        ms_str.truncate(3);
        while ms_str.len() < 3 {
            ms_str.push('0');
        }
        millis = ms_str.parse().unwrap_or(0);
    }

    // Timezone offset (default UTC). `datetime.fromisoformat` accepts a bare
    // `Z` (Python 3.11+), a `+HH:MM`/`-HH:MM` offset, or no suffix at all — and
    // rejects anything else, including trailing garbage. We must do the same so
    // a malformed timestamp fails visibly (returns `None`) rather than being
    // silently parsed as if the trailing bytes were absent.
    let mut offset_secs = 0i64;
    if rest == "Z" {
        rest = "";
    } else if let Some(off) = rest.strip_prefix('+').or_else(|| rest.strip_prefix('-')) {
        let sign = if rest.starts_with('-') { -1 } else { 1 };
        // Offset must be exactly `HH:MM` or `HH` (with nothing trailing).
        if off.len() != 5 && off.len() != 2 {
            return None;
        }
        let oh: i64 = off.get(0..2)?.parse().ok()?;
        if off.len() == 5 {
            if off.as_bytes()[2] != b':' {
                return None;
            }
            let om: i64 = off.get(3..5)?.parse().ok()?;
            if oh > 23 || om > 59 {
                return None;
            }
            offset_secs = sign * (oh * 3600 + om * 60);
        } else {
            if oh > 23 {
                return None;
            }
            offset_secs = sign * oh * 3600;
        }
        rest = "";
    }
    // Any leftover bytes mean the timestamp is malformed (e.g. trailing
    // garbage after the seconds/fraction/offset).
    if !rest.is_empty() {
        return None;
    }

    // days_from_civil (Howard Hinnant).
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;

    let epoch_secs = days * 86400 + hour * 3600 + minute * 60 + second - offset_secs;
    Some(epoch_secs * 1000 + millis)
}

/// Number of days in a proleptic Gregorian calendar month, or `None` for an
/// out-of-range month. Used for ISO-8601 calendar validation.
fn days_in_month(year: i64, month: i64) -> Option<i64> {
    let is_leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    Some(match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap => 29,
        2 => 28,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Core listing implementation
// ---------------------------------------------------------------------------

/// Reads session files from a single project directory.
fn read_sessions_from_dir(project_dir: &Path, project_path: Option<&str>) -> Vec<SDKSessionInfo> {
    let Ok(entries) = std::fs::read_dir(project_dir) else {
        return Vec::new();
    };

    let mut results = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(stem) = name.strip_suffix(".jsonl") else {
            continue;
        };
        if validate_uuid(stem).is_none() {
            continue;
        }
        let Some(lite) = read_session_lite(&entry.path()) else {
            continue;
        };
        if let Some(info) = parse_session_info_from_lite(stem, &lite, project_path) {
            results.push(info);
        }
    }
    results
}

/// Deduplicates by `session_id`, keeping the newest `last_modified`.
fn deduplicate_by_session_id(sessions: Vec<SDKSessionInfo>) -> Vec<SDKSessionInfo> {
    let mut by_id: HashMap<String, SDKSessionInfo> = HashMap::new();
    for s in sessions {
        match by_id.get(&s.session_id) {
            Some(existing) if existing.last_modified >= s.last_modified => {}
            _ => {
                by_id.insert(s.session_id.clone(), s);
            }
        }
    }
    by_id.into_values().collect()
}

/// Sorts by `last_modified` descending and applies `offset` + `limit`.
pub(crate) fn apply_sort_limit_offset(
    mut sessions: Vec<SDKSessionInfo>,
    limit: Option<usize>,
    offset: usize,
) -> Vec<SDKSessionInfo> {
    sessions.sort_by_key(|s| std::cmp::Reverse(s.last_modified));
    if offset > 0 {
        sessions = if offset < sessions.len() {
            sessions.split_off(offset)
        } else {
            Vec::new()
        };
    }
    if let Some(limit) = limit {
        if limit > 0 && sessions.len() > limit {
            sessions.truncate(limit);
        }
    }
    sessions
}

pub(crate) fn canonicalize_path(d: &str) -> String {
    key::canonicalize_path(d)
}

/// Lists sessions for a specific project directory (and its worktrees).
fn list_sessions_for_project(
    directory: &str,
    limit: Option<usize>,
    offset: usize,
    include_worktrees: bool,
) -> Vec<SDKSessionInfo> {
    let canonical_dir = canonicalize_path(directory);

    let worktree_paths = if include_worktrees {
        get_worktree_paths(&canonical_dir)
    } else {
        Vec::new()
    };

    // No worktrees — scan the single project dir.
    if worktree_paths.len() <= 1 {
        let Some(project_dir) = find_project_dir(&canonical_dir) else {
            return Vec::new();
        };
        let sessions = read_sessions_from_dir(&project_dir, Some(&canonical_dir));
        return apply_sort_limit_offset(sessions, limit, offset);
    }

    let projects_dir = get_projects_dir(None);
    let case_insensitive = cfg!(windows);

    // Sort worktree paths by sanitized-prefix length (longest first).
    let mut indexed: Vec<(String, String)> = worktree_paths
        .iter()
        .map(|wt| {
            let sanitized = key::sanitize_path(wt);
            let prefix = if case_insensitive {
                sanitized.to_lowercase()
            } else {
                sanitized
            };
            (wt.clone(), prefix)
        })
        .collect();
    indexed.sort_by_key(|(_, prefix)| std::cmp::Reverse(prefix.len()));

    let all_dirents: Vec<PathBuf> = if let Ok(rd) = std::fs::read_dir(&projects_dir) {
        rd.flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect()
    } else {
        let Some(matched_project_dir) = find_project_dir(&canonical_dir) else {
            return apply_sort_limit_offset(Vec::new(), limit, offset);
        };
        let sessions = read_sessions_from_dir(&matched_project_dir, Some(&canonical_dir));
        return apply_sort_limit_offset(sessions, limit, offset);
    };

    let mut all_sessions: Vec<SDKSessionInfo> = Vec::new();
    let mut seen_dirs: HashSet<String> = HashSet::new();

    // Always include the user's actual directory.
    if let Some(canonical_project_dir) = find_project_dir(&canonical_dir) {
        if let Some(dir_base) = canonical_project_dir.file_name().and_then(|n| n.to_str()) {
            let key_name = if case_insensitive {
                dir_base.to_lowercase()
            } else {
                dir_base.to_string()
            };
            seen_dirs.insert(key_name);
        }
        all_sessions.extend(read_sessions_from_dir(
            &canonical_project_dir,
            Some(&canonical_dir),
        ));
    }

    for entry in &all_dirents {
        let Some(raw_name) = entry.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let dir_name = if case_insensitive {
            raw_name.to_lowercase()
        } else {
            raw_name.to_string()
        };
        if seen_dirs.contains(&dir_name) {
            continue;
        }

        for (wt_path, prefix) in &indexed {
            let is_match = &dir_name == prefix
                || (prefix.len() >= MAX_SANITIZED_LENGTH
                    && dir_name.starts_with(&format!("{prefix}-")));
            if is_match {
                seen_dirs.insert(dir_name.clone());
                all_sessions.extend(read_sessions_from_dir(entry, Some(wt_path)));
                break;
            }
        }
    }

    let deduped = deduplicate_by_session_id(all_sessions);
    apply_sort_limit_offset(deduped, limit, offset)
}

/// Lists sessions across all project directories.
fn list_all_sessions(limit: Option<usize>, offset: usize) -> Vec<SDKSessionInfo> {
    let projects_dir = get_projects_dir(None);
    let Ok(rd) = std::fs::read_dir(&projects_dir) else {
        return Vec::new();
    };
    let project_dirs: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();

    let mut all_sessions = Vec::new();
    for project_dir in &project_dirs {
        all_sessions.extend(read_sessions_from_dir(project_dir, None));
    }
    let deduped = deduplicate_by_session_id(all_sessions);
    apply_sort_limit_offset(deduped, limit, offset)
}

/// Lists sessions with metadata extracted from stat + head/tail reads.
///
/// When `directory` is provided, returns sessions for that project directory
/// and (optionally) its git worktrees. When omitted, returns sessions across
/// all projects. Results are sorted by `last_modified` descending.
#[must_use]
pub fn list_sessions(
    directory: Option<&str>,
    limit: Option<usize>,
    offset: usize,
    include_worktrees: bool,
) -> Vec<SDKSessionInfo> {
    match directory {
        Some(dir) if !dir.is_empty() => {
            list_sessions_for_project(dir, limit, offset, include_worktrees)
        }
        _ => list_all_sessions(limit, offset),
    }
}

// ---------------------------------------------------------------------------
// get_session_info — single-session metadata lookup
// ---------------------------------------------------------------------------

/// Reads metadata for a single session by ID.
///
/// Directory resolution matches [`get_session_messages`]. Returns `None` when
/// the file is not found, is a sidechain, or has no extractable summary.
#[must_use]
pub fn get_session_info(session_id: &str, directory: Option<&str>) -> Option<SDKSessionInfo> {
    let uuid = validate_uuid(session_id)?;
    let file_name = format!("{uuid}.jsonl");

    if let Some(dir) = directory.filter(|d| !d.is_empty()) {
        let canonical = canonicalize_path(dir);
        if let Some(project_dir) = find_project_dir(&canonical) {
            if let Some(lite) = read_session_lite(&project_dir.join(&file_name)) {
                return parse_session_info_from_lite(uuid, &lite, Some(&canonical));
            }
        }

        for wt in get_worktree_paths(&canonical) {
            if wt == canonical {
                continue;
            }
            if let Some(wt_project_dir) = find_project_dir(&wt) {
                if let Some(lite) = read_session_lite(&wt_project_dir.join(&file_name)) {
                    return parse_session_info_from_lite(uuid, &lite, Some(&wt));
                }
            }
        }
        return None;
    }

    let projects_dir = get_projects_dir(None);
    let rd = std::fs::read_dir(&projects_dir).ok()?;
    for entry in rd.flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        if let Some(lite) = read_session_lite(&entry.path().join(&file_name)) {
            return parse_session_info_from_lite(uuid, &lite, None);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Transcript reading + chain reconstruction
// ---------------------------------------------------------------------------

/// A parsed JSONL transcript entry (loose object).
pub(crate) type TranscriptEntry = serde_json::Map<String, Value>;

fn try_read_session_file(project_dir: &Path, file_name: &str) -> Option<String> {
    std::fs::read_to_string(project_dir.join(file_name)).ok()
}

/// Finds and reads a session JSONL file, following worktree fallbacks.
fn read_session_file(session_id: &str, directory: Option<&str>) -> Option<String> {
    let file_name = format!("{session_id}.jsonl");

    if let Some(dir) = directory.filter(|d| !d.is_empty()) {
        let canonical_dir = canonicalize_path(dir);
        if let Some(project_dir) = find_project_dir(&canonical_dir) {
            if let Some(content) = try_read_session_file(&project_dir, &file_name) {
                if !content.is_empty() {
                    return Some(content);
                }
            }
        }
        for wt in get_worktree_paths(&canonical_dir) {
            if wt == canonical_dir {
                continue;
            }
            if let Some(wt_project_dir) = find_project_dir(&wt) {
                if let Some(content) = try_read_session_file(&wt_project_dir, &file_name) {
                    if !content.is_empty() {
                        return Some(content);
                    }
                }
            }
        }
        return None;
    }

    let projects_dir = get_projects_dir(None);
    let rd = std::fs::read_dir(&projects_dir).ok()?;
    for entry in rd.flatten() {
        if let Some(content) = try_read_session_file(&entry.path(), &file_name) {
            if !content.is_empty() {
                return Some(content);
            }
        }
    }
    None
}

/// Parses JSONL content into transcript entries, tolerating corrupt lines.
///
/// Only keeps entries with a `uuid` and a transcript message type.
pub(crate) fn parse_transcript_entries(content: &str) -> Vec<TranscriptEntry> {
    let mut entries = Vec::new();
    for line in content.split('\n') {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(Value::Object(entry)) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let entry_type = entry.get("type").and_then(Value::as_str);
        let has_uuid = entry.get("uuid").and_then(Value::as_str).is_some();
        if let Some(t) = entry_type {
            if TRANSCRIPT_ENTRY_TYPES.contains(&t) && has_uuid {
                entries.push(entry);
            }
        }
    }
    entries
}

fn entry_uuid(entry: &TranscriptEntry) -> &str {
    entry.get("uuid").and_then(Value::as_str).unwrap_or("")
}

fn entry_type(entry: &TranscriptEntry) -> Option<&str> {
    entry.get("type").and_then(Value::as_str)
}

fn entry_parent(entry: &TranscriptEntry) -> Option<&str> {
    entry
        .get("parentUuid")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

fn entry_bool(entry: &TranscriptEntry, field: &str) -> bool {
    entry.get(field).and_then(Value::as_bool).unwrap_or(false)
}

fn entry_str_present(entry: &TranscriptEntry, field: &str) -> bool {
    entry
        .get(field)
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty())
}

/// Builds the conversation chain by finding the leaf and walking `parentUuid`.
///
/// Returns messages in chronological order (root -> leaf). `logicalParentUuid`
/// is intentionally NOT followed (matches upstream / VS Code behavior).
pub(crate) fn build_conversation_chain(entries: &[TranscriptEntry]) -> Vec<TranscriptEntry> {
    if entries.is_empty() {
        return Vec::new();
    }

    let mut by_uuid: HashMap<&str, usize> = HashMap::new();
    let mut entry_index: HashMap<&str, usize> = HashMap::new();
    for (i, entry) in entries.iter().enumerate() {
        by_uuid.insert(entry_uuid(entry), i);
        entry_index.insert(entry_uuid(entry), i);
    }

    // Terminal messages: uuids that no other entry points to via parentUuid.
    let mut parent_uuids: HashSet<&str> = HashSet::new();
    for entry in entries {
        if let Some(p) = entry_parent(entry) {
            parent_uuids.insert(p);
        }
    }
    let terminals: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| !parent_uuids.contains(entry_uuid(e)))
        .map(|(i, _)| i)
        .collect();

    // From each terminal, walk back to the nearest user/assistant leaf.
    let mut leaves: Vec<usize> = Vec::new();
    for &terminal in &terminals {
        let mut cur = Some(terminal);
        let mut seen: HashSet<&str> = HashSet::new();
        while let Some(idx) = cur {
            let e = &entries[idx];
            let uid = entry_uuid(e);
            if seen.contains(uid) {
                break;
            }
            seen.insert(uid);
            if matches!(entry_type(e), Some("user" | "assistant")) {
                leaves.push(idx);
                break;
            }
            cur = entry_parent(e).and_then(|p| by_uuid.get(p).copied());
        }
    }

    if leaves.is_empty() {
        return Vec::new();
    }

    // Prefer the main-chain leaf (not sidechain/team/meta), highest file pos.
    let main_leaves: Vec<usize> = leaves
        .iter()
        .copied()
        .filter(|&i| {
            let e = &entries[i];
            !entry_bool(e, "isSidechain")
                && !entry_str_present(e, "teamName")
                && !entry_bool(e, "isMeta")
        })
        .collect();

    let pick_best = |candidates: &[usize]| -> usize {
        let mut best = candidates[0];
        let mut best_idx = entry_index
            .get(entry_uuid(&entries[best]))
            .copied()
            .unwrap_or(0);
        for &cur in &candidates[1..] {
            let cur_idx = entry_index
                .get(entry_uuid(&entries[cur]))
                .copied()
                .unwrap_or(0);
            if cur_idx > best_idx {
                best = cur;
                best_idx = cur_idx;
            }
        }
        best
    };

    let leaf = if main_leaves.is_empty() {
        pick_best(&leaves)
    } else {
        pick_best(&main_leaves)
    };

    // Walk from leaf to root via parentUuid.
    let mut chain: Vec<TranscriptEntry> = Vec::new();
    let mut chain_seen: HashSet<String> = HashSet::new();
    let mut cur = Some(leaf);
    while let Some(idx) = cur {
        let e = &entries[idx];
        let uid = entry_uuid(e).to_string();
        if chain_seen.contains(&uid) {
            break;
        }
        chain_seen.insert(uid);
        chain.push(e.clone());
        cur = entry_parent(e).and_then(|p| by_uuid.get(p).copied());
    }

    chain.reverse();
    chain
}

/// Returns `true` if the entry should be included in the returned messages.
fn is_visible_message(entry: &TranscriptEntry) -> bool {
    if !matches!(entry_type(entry), Some("user" | "assistant")) {
        return false;
    }
    if entry_bool(entry, "isMeta") || entry_bool(entry, "isSidechain") {
        return false;
    }
    // isCompactSummary messages ARE included (they hold compacted content).
    !entry_str_present(entry, "teamName")
}

/// Converts a transcript entry into a [`SessionMessage`].
fn to_session_message(entry: &TranscriptEntry) -> SessionMessage {
    let msg_type = if entry_type(entry) == Some("user") {
        SessionMessageType::User
    } else {
        SessionMessageType::Assistant
    };
    SessionMessage {
        r#type: msg_type,
        uuid: entry_uuid(entry).to_string(),
        session_id: entry
            .get("sessionId")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        message: entry.get("message").cloned(),
        parent_tool_use_id: None,
    }
}

/// Builds the conversation chain from parsed entries and applies paging.
pub(crate) fn entries_to_session_messages(
    entries: &[TranscriptEntry],
    limit: Option<usize>,
    offset: usize,
) -> Vec<SessionMessage> {
    let chain = build_conversation_chain(entries);
    let messages: Vec<SessionMessage> = chain
        .iter()
        .filter(|e| is_visible_message(e))
        .map(to_session_message)
        .collect();
    page(messages, limit, offset)
}

/// Applies `offset`/`limit` paging matching upstream slice semantics.
fn page<T>(items: Vec<T>, limit: Option<usize>, offset: usize) -> Vec<T> {
    if let Some(limit) = limit {
        if limit > 0 {
            return items.into_iter().skip(offset).take(limit).collect();
        }
    }
    if offset > 0 {
        return items.into_iter().skip(offset).collect();
    }
    items
}

/// Reads a session's conversation messages from its JSONL transcript file.
///
/// Parses the full JSONL, builds the conversation chain via `parentUuid` links,
/// and returns user/assistant messages in chronological order.
#[must_use]
pub fn get_session_messages(
    session_id: &str,
    directory: Option<&str>,
    limit: Option<usize>,
    offset: usize,
) -> Vec<SessionMessage> {
    if validate_uuid(session_id).is_none() {
        return Vec::new();
    }
    let Some(content) = read_session_file(session_id, directory) else {
        return Vec::new();
    };
    let entries = parse_transcript_entries(&content);
    entries_to_session_messages(&entries, limit, offset)
}

// ---------------------------------------------------------------------------
// Subagent transcript reading
// ---------------------------------------------------------------------------

/// Resolves the on-disk path of a session JSONL file (non-empty match).
pub(crate) fn resolve_session_file_path(
    session_id: &str,
    directory: Option<&str>,
) -> Option<PathBuf> {
    let file_name = format!("{session_id}.jsonl");

    let stat_candidate = |project_dir: &Path| -> Option<PathBuf> {
        let candidate = project_dir.join(&file_name);
        match std::fs::metadata(&candidate) {
            Ok(m) if m.len() > 0 => Some(candidate),
            _ => None,
        }
    };

    if let Some(dir) = directory.filter(|d| !d.is_empty()) {
        let canonical_dir = canonicalize_path(dir);
        if let Some(project_dir) = find_project_dir(&canonical_dir) {
            if let Some(found) = stat_candidate(&project_dir) {
                return Some(found);
            }
        }
        for wt in get_worktree_paths(&canonical_dir) {
            if wt == canonical_dir {
                continue;
            }
            if let Some(wt_project_dir) = find_project_dir(&wt) {
                if let Some(found) = stat_candidate(&wt_project_dir) {
                    return Some(found);
                }
            }
        }
        return None;
    }

    let projects_dir = get_projects_dir(None);
    let rd = std::fs::read_dir(&projects_dir).ok()?;
    for entry in rd.flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        if let Some(found) = stat_candidate(&entry.path()) {
            return Some(found);
        }
    }
    None
}

/// Resolves the `subagents/` directory for a given session.
fn resolve_subagents_dir(session_id: &str, directory: Option<&str>) -> Option<PathBuf> {
    let resolved = resolve_session_file_path(session_id, directory)?;
    // Strip .jsonl -> session dir, then /subagents.
    let session_dir = resolved.with_extension("");
    Some(session_dir.join("subagents"))
}

/// Recursively collects `agent-*.jsonl` files, sorted per directory.
///
/// Returns `(agent_id, file_path)` tuples for deterministic ordering.
fn collect_agent_files(base_dir: &Path) -> Vec<(String, PathBuf)> {
    let mut results = Vec::new();
    walk_agent_files(base_dir, &mut results);
    results
}

fn walk_agent_files(current_dir: &Path, results: &mut Vec<(String, PathBuf)>) {
    let Ok(rd) = std::fs::read_dir(current_dir) else {
        return;
    };
    let mut dirents: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    dirents.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    for entry in dirents {
        let Some(name) = entry.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if let Some(agent_id) = entry
            .is_file()
            .then(|| {
                name.strip_prefix("agent-")
                    .and_then(|n| n.strip_suffix(".jsonl"))
            })
            .flatten()
        {
            results.push((agent_id.to_string(), entry.clone()));
        } else if entry.is_dir() {
            walk_agent_files(&entry, results);
        }
    }
}

/// Builds the (linear) conversation chain for a subagent transcript.
pub(crate) fn build_subagent_chain(entries: &[TranscriptEntry]) -> Vec<TranscriptEntry> {
    if entries.is_empty() {
        return Vec::new();
    }
    let mut by_uuid: HashMap<&str, usize> = HashMap::new();
    for (i, entry) in entries.iter().enumerate() {
        by_uuid.insert(entry_uuid(entry), i);
    }

    // The last user/assistant entry is the leaf.
    let leaf = entries
        .iter()
        .enumerate()
        .rev()
        .find(|(_, e)| matches!(entry_type(e), Some("user" | "assistant")))
        .map(|(i, _)| i);
    let Some(leaf) = leaf else {
        return Vec::new();
    };

    let mut chain: Vec<TranscriptEntry> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut cur = Some(leaf);
    while let Some(idx) = cur {
        let e = &entries[idx];
        let uid = entry_uuid(e).to_string();
        if seen.contains(&uid) {
            break;
        }
        seen.insert(uid);
        chain.push(e.clone());
        cur = entry_parent(e).and_then(|p| by_uuid.get(p).copied());
    }
    chain.reverse();
    chain
}

pub(crate) fn entries_to_subagent_messages(
    entries: &[TranscriptEntry],
    limit: Option<usize>,
    offset: usize,
) -> Vec<SessionMessage> {
    let chain = build_subagent_chain(entries);
    let messages: Vec<SessionMessage> = chain
        .iter()
        .filter(|e| matches!(entry_type(e), Some("user" | "assistant")))
        .map(to_session_message)
        .collect();
    page(messages, limit, offset)
}

/// Lists subagent IDs for a session by scanning the `subagents/` directory.
#[must_use]
pub fn list_subagents(session_id: &str, directory: Option<&str>) -> Vec<String> {
    if validate_uuid(session_id).is_none() {
        return Vec::new();
    }
    let Some(subagents_dir) = resolve_subagents_dir(session_id, directory) else {
        return Vec::new();
    };
    collect_agent_files(&subagents_dir)
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

/// Reads a subagent's conversation messages from its JSONL transcript file.
#[must_use]
pub fn get_subagent_messages(
    session_id: &str,
    agent_id: &str,
    directory: Option<&str>,
    limit: Option<usize>,
    offset: usize,
) -> Vec<SessionMessage> {
    if validate_uuid(session_id).is_none() || agent_id.is_empty() {
        return Vec::new();
    }
    let Some(subagents_dir) = resolve_subagents_dir(session_id, directory) else {
        return Vec::new();
    };

    let mut matched: Option<PathBuf> = None;
    for (found_id, file_path) in collect_agent_files(&subagents_dir) {
        if found_id == agent_id {
            matched = Some(file_path);
            break;
        }
    }
    let Some(matched) = matched else {
        return Vec::new();
    };

    let Ok(content) = std::fs::read_to_string(&matched) else {
        return Vec::new();
    };
    if content.is_empty() {
        return Vec::new();
    }
    let entries = parse_transcript_entries(&content);
    entries_to_subagent_messages(&entries, limit, offset)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    const S1: &str = "11111111-1111-4111-8111-111111111111";
    const S2: &str = "22222222-2222-4222-8222-222222222222";

    fn parse(lines: &[Value]) -> Vec<TranscriptEntry> {
        let content = lines
            .iter()
            .map(|l| serde_json::to_string(l).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        parse_transcript_entries(&content)
    }

    // ------- UUID / field extraction -------

    #[test]
    fn validate_uuid_accepts_valid_rejects_invalid() {
        assert_eq!(validate_uuid(S1), Some(S1));
        assert_eq!(validate_uuid("not-a-uuid"), None);
        assert_eq!(validate_uuid(""), None);
    }

    #[test]
    fn extract_json_string_field_first_and_last() {
        let text = r#"{"cwd":"/a","cwd":"/b"}"#;
        assert_eq!(
            extract_json_string_field(text, "cwd").as_deref(),
            Some("/a")
        );
        assert_eq!(
            extract_last_json_string_field(text, "cwd").as_deref(),
            Some("/b")
        );
    }

    #[test]
    fn extract_json_string_field_handles_spacing_and_escapes() {
        let text = r#"{"gitBranch": "feature\/x"}"#;
        assert_eq!(
            extract_json_string_field(text, "gitBranch").as_deref(),
            Some("feature/x")
        );
    }

    #[test]
    fn extract_json_string_field_missing_returns_none() {
        assert_eq!(extract_json_string_field("{}", "nope"), None);
    }

    // ------- first prompt extraction -------

    #[test]
    fn first_prompt_skips_meta_and_commands() {
        let head = [
            json!({"type":"user","isMeta":true,"message":{"content":"meta"}}).to_string(),
            json!({"type":"user","message":{"content":"<command-name>build</command-name>"}})
                .to_string(),
            json!({"type":"user","message":{"content":"real question"}}).to_string(),
        ]
        .join("\n");
        assert_eq!(extract_first_prompt_from_head(&head), "real question");
    }

    #[test]
    fn first_prompt_command_fallback_when_only_commands() {
        let head =
            json!({"type":"user","message":{"content":"<command-name>deploy</command-name>"}})
                .to_string();
        assert_eq!(extract_first_prompt_from_head(&head), "deploy");
    }

    #[test]
    fn first_prompt_truncates_to_200_chars() {
        let long = "x".repeat(250);
        let head = json!({"type":"user","message":{"content": long}}).to_string();
        let out = extract_first_prompt_from_head(&head);
        assert_eq!(out.chars().count(), 201); // 200 + ellipsis
        assert!(out.ends_with('\u{2026}'));
    }

    #[test]
    fn first_prompt_handles_content_blocks() {
        let head = json!({
            "type":"user",
            "message":{"content":[{"type":"text","text":"hello blocks"}]}
        })
        .to_string();
        assert_eq!(extract_first_prompt_from_head(&head), "hello blocks");
    }

    #[test]
    fn is_skip_first_prompt_covers_every_marker() {
        // Every marker in the official `_SKIP_FIRST_PROMPT_PATTERN` must skip.
        assert!(is_skip_first_prompt("<local-command-stdout>output"));
        assert!(is_skip_first_prompt("<session-start-hook>hi"));
        assert!(is_skip_first_prompt("<tick>"));
        assert!(is_skip_first_prompt("<goal>do the thing"));
        // `[Request interrupted by user...]` — bracketed banner, optional tail.
        assert!(is_skip_first_prompt("[Request interrupted by user]"));
        assert!(is_skip_first_prompt(
            "[Request interrupted by user for tool use]"
        ));
        // Full-wrap IDE markers (allowing surrounding whitespace).
        assert!(is_skip_first_prompt(
            "<ide_opened_file>foo.rs</ide_opened_file>"
        ));
        assert!(is_skip_first_prompt(
            "  <ide_selection>lines 1-5</ide_selection>  "
        ));
        // Genuine prompts are NOT skipped.
        assert!(!is_skip_first_prompt("real question"));
        // A partial IDE marker that does not wrap the whole text is kept.
        assert!(!is_skip_first_prompt(
            "<ide_selection>x</ide_selection> then a real ask"
        ));
        // A non-anchored marker (banner not at the start) is kept.
        assert!(!is_skip_first_prompt("hello <tick>"));
    }

    #[test]
    fn first_prompt_skips_ide_selection_marker() {
        // A first user message that is purely an <ide_selection> wrapper must be
        // skipped in favour of the next real prompt.
        let head = [
            json!({"type":"user","message":{"content":"<ide_selection>foo</ide_selection>"}})
                .to_string(),
            json!({"type":"user","message":{"content":"the real prompt"}}).to_string(),
        ]
        .join("\n");
        assert_eq!(extract_first_prompt_from_head(&head), "the real prompt");
    }

    // ------- chain reconstruction -------

    #[test]
    fn chain_walks_parent_uuid_in_order() {
        let entries = parse(&[
            json!({"type":"user","uuid":"a","parentUuid":null,"sessionId":S1,"message":{"content":"1"}}),
            json!({"type":"assistant","uuid":"b","parentUuid":"a","sessionId":S1,"message":{"content":"2"}}),
            json!({"type":"user","uuid":"c","parentUuid":"b","sessionId":S1,"message":{"content":"3"}}),
        ]);
        let chain = build_conversation_chain(&entries);
        let uuids: Vec<&str> = chain.iter().map(entry_uuid).collect();
        assert_eq!(uuids, vec!["a", "b", "c"]);
    }

    #[test]
    fn chain_prefers_main_over_sidechain_leaf() {
        // main chain a->b (leaf b) and a sidechain leaf s later in file.
        let entries = parse(&[
            json!({"type":"user","uuid":"a","parentUuid":null,"sessionId":S1,"message":{"content":"1"}}),
            json!({"type":"assistant","uuid":"b","parentUuid":"a","sessionId":S1,"message":{"content":"2"}}),
            json!({"type":"assistant","uuid":"s","parentUuid":null,"isSidechain":true,"sessionId":S1,"message":{"content":"side"}}),
        ]);
        let chain = build_conversation_chain(&entries);
        let uuids: Vec<&str> = chain.iter().map(entry_uuid).collect();
        assert_eq!(uuids, vec!["a", "b"]);
    }

    #[test]
    fn chain_breaks_cycles() {
        // pathological self/mutual cycle must not loop forever.
        let entries = parse(&[
            json!({"type":"user","uuid":"a","parentUuid":"b","sessionId":S1,"message":{"content":"1"}}),
            json!({"type":"assistant","uuid":"b","parentUuid":"a","sessionId":S1,"message":{"content":"2"}}),
        ]);
        let chain = build_conversation_chain(&entries);
        assert!(chain.len() <= 2);
    }

    #[test]
    fn messages_filter_meta_sidechain_and_include_compact_summary() {
        let entries = parse(&[
            json!({"type":"user","uuid":"a","parentUuid":null,"sessionId":S1,"message":{"content":"q"}}),
            json!({"type":"assistant","uuid":"b","parentUuid":"a","isCompactSummary":true,"sessionId":S1,"message":{"content":"summary"}}),
            json!({"type":"user","uuid":"c","parentUuid":"b","isMeta":true,"sessionId":S1,"message":{"content":"meta"}}),
            json!({"type":"assistant","uuid":"d","parentUuid":"c","sessionId":S1,"message":{"content":"final"}}),
        ]);
        let msgs = entries_to_session_messages(&entries, None, 0);
        let uuids: Vec<&str> = msgs.iter().map(|m| m.uuid.as_str()).collect();
        // 'c' (isMeta) filtered; 'b' (compact summary) kept.
        assert_eq!(uuids, vec!["a", "b", "d"]);
    }

    #[test]
    fn messages_pagination_offset_and_limit() {
        let entries = parse(&[
            json!({"type":"user","uuid":"a","parentUuid":null,"sessionId":S1,"message":{"content":"1"}}),
            json!({"type":"assistant","uuid":"b","parentUuid":"a","sessionId":S1,"message":{"content":"2"}}),
            json!({"type":"user","uuid":"c","parentUuid":"b","sessionId":S1,"message":{"content":"3"}}),
            json!({"type":"assistant","uuid":"d","parentUuid":"c","sessionId":S1,"message":{"content":"4"}}),
        ]);
        let page = entries_to_session_messages(&entries, Some(2), 1);
        let uuids: Vec<&str> = page.iter().map(|m| m.uuid.as_str()).collect();
        assert_eq!(uuids, vec!["b", "c"]);
    }

    #[test]
    fn corrupt_lines_are_skipped() {
        let content = format!(
            "{}\n{{ this is not json\n{}\n\n",
            json!({"type":"user","uuid":"a","parentUuid":null,"sessionId":S1,"message":{"content":"1"}}),
            json!({"type":"assistant","uuid":"b","parentUuid":"a","sessionId":S1,"message":{"content":"2"}}),
        );
        let entries = parse_transcript_entries(&content);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn entries_without_uuid_or_wrong_type_are_dropped() {
        let entries = parse(&[
            json!({"type":"user","sessionId":S1,"message":{"content":"no uuid"}}),
            json!({"type":"tag","uuid":"t","tag":"x"}),
            json!({"type":"user","uuid":"ok","parentUuid":null,"sessionId":S1,"message":{"content":"ok"}}),
        ]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entry_uuid(&entries[0]), "ok");
    }

    // ------- subagent chain (linear) -------

    #[test]
    fn subagent_chain_is_linear_last_leaf() {
        let entries = parse(&[
            json!({"type":"user","uuid":"a","parentUuid":null,"sessionId":S1,"message":{"content":"1"}}),
            json!({"type":"assistant","uuid":"b","parentUuid":"a","sessionId":S1,"message":{"content":"2"}}),
        ]);
        let msgs = entries_to_subagent_messages(&entries, None, 0);
        let uuids: Vec<&str> = msgs.iter().map(|m| m.uuid.as_str()).collect();
        assert_eq!(uuids, vec!["a", "b"]);
    }

    // ------- metadata parsing from lite -------

    #[test]
    fn parse_info_extracts_metadata_fields() {
        let head = [
            json!({"type":"user","uuid":"a","cwd":"/proj","gitBranch":"main","timestamp":"2024-01-02T03:04:05.678Z","message":{"content":"first question here"}}).to_string(),
        ].join("\n");
        let tail = [
            head.clone(),
            json!({"type":"custom-title","customTitle":"My Title","sessionId":S1}).to_string(),
            format!(r#"{{"type":"tag","tag":"exp","sessionId":"{S1}"}}"#),
        ]
        .join("\n");
        let lite = LiteSessionFile {
            mtime: 1000,
            size: 42,
            head,
            tail,
        };
        let info = parse_session_info_from_lite(S1, &lite, Some("/proj")).unwrap();
        assert_eq!(info.summary, "My Title");
        assert_eq!(info.custom_title.as_deref(), Some("My Title"));
        assert_eq!(info.git_branch.as_deref(), Some("main"));
        assert_eq!(info.cwd.as_deref(), Some("/proj"));
        assert_eq!(info.tag.as_deref(), Some("exp"));
        assert_eq!(info.first_prompt.as_deref(), Some("first question here"));
        assert_eq!(info.file_size, Some(42));
        assert_eq!(info.last_modified, 1000);
        // 2024-01-02T03:04:05.678Z -> epoch ms
        assert_eq!(info.created_at, Some(1_704_164_645_678));
    }

    #[test]
    fn parse_info_rejects_sidechain_first_line() {
        let head = json!({"type":"user","uuid":"a","isSidechain":true,"message":{"content":"x"}})
            .to_string();
        let lite = LiteSessionFile {
            mtime: 1,
            size: 1,
            head: head.clone(),
            tail: head,
        };
        assert!(parse_session_info_from_lite(S1, &lite, None).is_none());
    }

    #[test]
    fn parse_info_rejects_metadata_only_session() {
        let head = json!({"type":"summary","note":"nothing useful"}).to_string();
        let lite = LiteSessionFile {
            mtime: 1,
            size: 1,
            head: head.clone(),
            tail: head,
        };
        assert!(parse_session_info_from_lite(S1, &lite, None).is_none());
    }

    #[test]
    fn parse_info_empty_title_falls_back_to_first_prompt() {
        // An empty-string customTitle must NOT be treated as a title — Python's
        // trailing `or None` lets the fallback chain continue, so the summary
        // comes from the first user prompt instead of an empty string.
        let user =
            json!({"type":"user","uuid":"a","message":{"content":"the real prompt"}}).to_string();
        let custom = json!({"type":"custom-title","customTitle":""}).to_string();
        let head = [custom.clone(), user].join("\n");
        let lite = LiteSessionFile {
            mtime: 1,
            size: 1,
            head: head.clone(),
            tail: head,
        };
        let info = parse_session_info_from_lite(S1, &lite, None).unwrap();
        assert_eq!(info.custom_title, None, "empty customTitle is not a title");
        assert_eq!(
            info.summary, "the real prompt",
            "summary falls back to the first prompt when title is empty"
        );
    }

    #[test]
    fn parse_info_all_empty_metadata_drops_session() {
        // Empty title AND no prompt/summary → nothing extractable → the whole
        // session is dropped (Python's `if not summary` guard).
        let custom = json!({"type":"custom-title","customTitle":""}).to_string();
        let meta = json!({"type":"summary","note":"x"}).to_string();
        let head = [custom, meta].join("\n");
        let lite = LiteSessionFile {
            mtime: 1,
            size: 1,
            head: head.clone(),
            tail: head,
        };
        assert!(
            parse_session_info_from_lite(S1, &lite, None).is_none(),
            "session with only empty metadata is dropped"
        );
    }

    #[test]
    fn parse_info_tag_scoped_to_tag_lines() {
        // A tool_use "tag" must not be mistaken for a session tag.
        let head =
            json!({"type":"user","uuid":"a","message":{"content":"real prompt text"}}).to_string();
        let tail = [
            head.clone(),
            json!({"type":"assistant","uuid":"b","tag":"git-tag-input"}).to_string(),
        ]
        .join("\n");
        let lite = LiteSessionFile {
            mtime: 1,
            size: 1,
            head,
            tail,
        };
        let info = parse_session_info_from_lite(S1, &lite, None).unwrap();
        assert_eq!(info.tag, None);
    }

    // ------- dedup / sort / pagination -------

    #[test]
    fn dedup_keeps_newest_and_sort_desc() {
        let mk = |id: &str, mtime: i64| SDKSessionInfo {
            session_id: id.to_string(),
            summary: "s".into(),
            last_modified: mtime,
            ..Default::default()
        };
        let deduped = deduplicate_by_session_id(vec![mk(S1, 100), mk(S1, 200), mk(S2, 50)]);
        let sorted = apply_sort_limit_offset(deduped, None, 0);
        assert_eq!(sorted.len(), 2);
        assert_eq!(sorted[0].last_modified, 200);
        assert_eq!(sorted[1].last_modified, 50);
    }

    #[test]
    fn sort_limit_offset_paginates() {
        let mk = |mtime: i64| SDKSessionInfo {
            session_id: uuid::Uuid::new_v4().to_string(),
            summary: "s".into(),
            last_modified: mtime,
            ..Default::default()
        };
        let items = vec![mk(1), mk(2), mk(3), mk(4), mk(5)];
        let page = apply_sort_limit_offset(items, Some(2), 1);
        let mtimes: Vec<i64> = page.iter().map(|s| s.last_modified).collect();
        assert_eq!(mtimes, vec![4, 3]);
    }

    // ------- ISO parsing -------

    #[test]
    fn iso_parser_handles_z_and_offset() {
        assert_eq!(parse_iso8601_ms("1970-01-01T00:00:00.000Z"), Some(0));
        assert_eq!(parse_iso8601_ms("1970-01-01T00:00:01Z"), Some(1000));
        // +01:00 offset means the instant is one hour earlier in UTC.
        assert_eq!(parse_iso8601_ms("1970-01-01T01:00:00+01:00"), Some(0));
        // -01:00 offset means the instant is one hour later in UTC.
        assert_eq!(parse_iso8601_ms("1969-12-31T23:00:00-01:00"), Some(0));
        // Space separator (fromisoformat accepts it) and no zone → UTC.
        assert_eq!(parse_iso8601_ms("1970-01-01 00:00:00"), Some(0));
    }

    #[test]
    fn iso_parser_rejects_malformed_and_trailing() {
        // This parser is a fixed-shape reader for the CLI's full
        // `YYYY-MM-DDTHH:MM:SS[.fff][Z|±HH:MM]` timestamps, not a general
        // `datetime.fromisoformat`. It is intentionally at least as strict:
        // date-only or too-short inputs are rejected (the CLI never writes
        // them), and every case below is one both this parser AND
        // `datetime.fromisoformat` reject.
        assert_eq!(parse_iso8601_ms("2024-01-01"), None); // too short (date-only)
        assert_eq!(parse_iso8601_ms(""), None);
        // Wrong date separators.
        assert_eq!(parse_iso8601_ms("2024/01/01T00:00:00"), None);
        // Wrong time separators.
        assert_eq!(parse_iso8601_ms("2024-01-01T00-00-00"), None);
        // Out-of-range calendar/clock values (fromisoformat raises).
        assert_eq!(parse_iso8601_ms("2024-13-01T00:00:00"), None);
        assert_eq!(parse_iso8601_ms("2024-02-30T00:00:00"), None);
        assert_eq!(parse_iso8601_ms("2023-02-29T00:00:00"), None); // 2023 not leap
        assert_eq!(parse_iso8601_ms("2024-01-01T00:60:00"), None);
        // Trailing garbage after a valid instant must fail visibly, not be
        // silently accepted as if the trailing bytes were absent (both this
        // parser and fromisoformat reject these).
        assert_eq!(parse_iso8601_ms("1970-01-01T00:00:00Zextra"), None);
        assert_eq!(parse_iso8601_ms("1970-01-01T00:00:00garbage"), None);
        assert_eq!(parse_iso8601_ms("1970-01-01T00:00:00.5x"), None);
        // Malformed offset (missing zero-padded hour digit).
        assert_eq!(parse_iso8601_ms("1970-01-01T00:00:00+1:00"), None);
        // A leap-day in a leap year is valid.
        assert_eq!(
            parse_iso8601_ms("2024-02-29T00:00:00Z"),
            Some(1_709_164_800_000)
        );
    }

    // ------- filesystem-backed scans (env-gated, serialized) -------

    fn env_lock() -> tokio::sync::MutexGuard<'static, ()> {
        super::TEST_ENV_LOCK.blocking_lock()
    }

    /// Create a project directory under a fake `CLAUDE_CONFIG_DIR` for `cwd`.
    fn setup_project(config: &Path, cwd: &str) -> PathBuf {
        std::env::set_var("CLAUDE_CONFIG_DIR", config);
        let project_dir = get_project_dir(cwd);
        std::fs::create_dir_all(&project_dir).unwrap();
        project_dir
    }

    fn write_session(project_dir: &Path, session_id: &str, entries: &[Value]) {
        let content = entries
            .iter()
            .map(|e| serde_json::to_string(e).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(project_dir.join(format!("{session_id}.jsonl")), content).unwrap();
    }

    #[test]
    fn list_sessions_scans_project_dir() {
        let _guard = env_lock();
        let tmp = tempfile::tempdir().unwrap();
        let cwd = "/tmp/list-project-abc";
        let project_dir = setup_project(tmp.path(), cwd);
        write_session(
            &project_dir,
            S1,
            &[json!({"type":"user","uuid":"a","message":{"content":"hello there"}})],
        );
        write_session(
            &project_dir,
            S2,
            &[json!({"type":"user","uuid":"b","message":{"content":"second session"}})],
        );
        // metadata-only file -> filtered out.
        write_session(
            &project_dir,
            "33333333-3333-4333-8333-333333333333",
            &[json!({"type":"summary"})],
        );
        // sidechain file -> filtered out.
        write_session(
            &project_dir,
            "44444444-4444-4444-8444-444444444444",
            &[json!({"type":"user","uuid":"s","isSidechain":true,"message":{"content":"x"}})],
        );
        // non-uuid filename -> ignored.
        std::fs::write(project_dir.join("notes.jsonl"), "junk").unwrap();

        let sessions = list_sessions(Some(cwd), None, 0, false);
        let ids: HashSet<String> = sessions.iter().map(|s| s.session_id.clone()).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(S1));
        assert!(ids.contains(S2));
        std::env::remove_var("CLAUDE_CONFIG_DIR");
    }

    #[test]
    fn get_session_info_and_messages_roundtrip() {
        let _guard = env_lock();
        let tmp = tempfile::tempdir().unwrap();
        let cwd = "/tmp/info-project-xyz";
        let project_dir = setup_project(tmp.path(), cwd);
        write_session(
            &project_dir,
            S1,
            &[
                json!({"type":"user","uuid":"a","parentUuid":null,"sessionId":S1,"cwd":cwd,"message":{"content":"the first prompt"}}),
                json!({"type":"assistant","uuid":"b","parentUuid":"a","sessionId":S1,"message":{"content":"answer"}}),
            ],
        );
        let info = get_session_info(S1, Some(cwd)).unwrap();
        assert_eq!(info.summary, "the first prompt");
        let msgs = get_session_messages(S1, Some(cwd), None, 0);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].r#type, SessionMessageType::User);
        assert_eq!(msgs[1].r#type, SessionMessageType::Assistant);
        std::env::remove_var("CLAUDE_CONFIG_DIR");
    }

    #[test]
    fn subagent_discovery_nested_and_traversal() {
        let _guard = env_lock();
        let tmp = tempfile::tempdir().unwrap();
        let cwd = "/tmp/subagent-project";
        let project_dir = setup_project(tmp.path(), cwd);
        write_session(
            &project_dir,
            S1,
            &[json!({"type":"user","uuid":"a","message":{"content":"main"}})],
        );
        // nested subagent dir
        let nested = project_dir
            .join(S1)
            .join("subagents")
            .join("workflows")
            .join("run-1");
        std::fs::create_dir_all(&nested).unwrap();
        let flat = project_dir.join(S1).join("subagents");
        std::fs::write(
            flat.join("agent-abc.jsonl"),
            json!({"type":"user","uuid":"x","message":{"content":"sub"}}).to_string(),
        )
        .unwrap();
        std::fs::write(
            nested.join("agent-def.jsonl"),
            json!({"type":"assistant","uuid":"y","message":{"content":"deep"}}).to_string(),
        )
        .unwrap();

        let mut ids = list_subagents(S1, Some(cwd));
        ids.sort();
        assert_eq!(ids, vec!["abc".to_string(), "def".to_string()]);

        let msgs = get_subagent_messages(S1, "def", Some(cwd), None, 0);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].uuid, "y");
        std::env::remove_var("CLAUDE_CONFIG_DIR");
    }
}
