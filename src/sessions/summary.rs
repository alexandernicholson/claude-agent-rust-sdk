//! Incremental session-summary derivation for [`SessionStore`] adapters.
//!
//! [`fold_session_summary`] lets a store maintain a per-session
//! [`SessionSummaryEntry`] sidecar incrementally inside `append()` so a
//! summary listing can fetch all metadata in one call instead of N per-session
//! loads. Every derived field is append-incremental (set-once or last-wins) so
//! adapters never need to re-read previously appended entries.
//!
//! Ported from the official Python Agent SDK
//! (`_internal/session_summary.py`).

use serde_json::{Map, Value};
#[cfg(test)]
use uuid::Uuid;

use crate::sessions::key::SessionKey;
use crate::sessions::store::{SDKSessionInfo, SessionStoreEntry};

/// Incrementally-maintained session summary sidecar.
///
/// Stores obtain this from [`fold_session_summary`] inside
/// [`SessionStore::append`](crate::sessions::SessionStore::append) and persist
/// it verbatim; they return the full set from
/// [`list_session_summaries`](crate::sessions::SessionStore::list_session_summaries).
/// The `data` field is opaque SDK-owned state — stores MUST NOT interpret it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSummaryEntry {
    /// The session identifier (see [`SessionKey::session_id`]).
    pub session_id: String,
    /// Storage write time of the sidecar, in Unix epoch milliseconds. Must use
    /// the same clock as [`list_sessions`](crate::sessions::SessionStore::list_sessions).
    /// Not set by [`fold_session_summary`]; the adapter stamps it after
    /// persisting.
    pub mtime: i64,
    /// Opaque SDK-owned summary state. Persist verbatim; do not interpret.
    pub data: Map<String, Value>,
}

/// JSONL entry keys → summary `data` keys for last-wins string fields. Each
/// appended entry overwrites the previous value when present.
const LAST_WINS_FIELDS: &[(&str, &str)] = &[
    ("customTitle", "custom_title"),
    ("aiTitle", "ai_title"),
    ("lastPrompt", "last_prompt"),
    ("summary", "summary_hint"),
    ("gitBranch", "git_branch"),
];

/// Parses an ISO-8601 timestamp to Unix epoch milliseconds. Returns `None` for
/// non-strings or unparseable values.
///
/// Delegates to the single canonical parser in
/// [`crate::sessions::filesystem::parse_iso8601_ms`] so the store-summary path
/// and the disk lite-parse path share identical, strict semantics (calendar
/// validation, offset handling, and rejection of trailing garbage). Leading /
/// trailing whitespace is stripped first, matching the CLI's lenient framing of
/// the surrounding JSON string.
fn iso_to_epoch_ms(ts: Option<&Value>) -> Option<i64> {
    let s = ts?.as_str()?;
    crate::sessions::filesystem::parse_iso8601_ms(s.trim())
}

/// Extracts text strings from a `type=="user"` entry's message content.
fn entry_text_blocks(entry: &Map<String, Value>) -> Vec<String> {
    let Some(message) = entry.get("message").and_then(Value::as_object) else {
        return Vec::new();
    };
    let mut texts = Vec::new();
    match message.get("content") {
        Some(Value::String(s)) => texts.push(s.clone()),
        Some(Value::Array(blocks)) => {
            for block in blocks {
                if let Some(obj) = block.as_object() {
                    if obj.get("type").and_then(Value::as_str) == Some("text") {
                        if let Some(t) = obj.get("text").and_then(Value::as_str) {
                            texts.push(t.to_string());
                        }
                    }
                }
            }
        }
        _ => {}
    }
    texts
}

/// True if a `user` message content list carries a `tool_result` block.
fn has_tool_result(entry: &Map<String, Value>) -> bool {
    entry
        .get("message")
        .and_then(Value::as_object)
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
        .is_some_and(|blocks| {
            blocks.iter().any(|b| {
                b.as_object()
                    .and_then(|o| o.get("type"))
                    .and_then(Value::as_str)
                    == Some("tool_result")
            })
        })
}

/// Extracts a `<command-name>...</command-name>` inner value, if present.
fn command_name(s: &str) -> Option<&str> {
    let start = s.find("<command-name>")? + "<command-name>".len();
    let rest = &s[start..];
    let end = rest.find("</command-name>")?;
    Some(&rest[..end])
}

/// Replicates the "first meaningful prompt" skip filter (auto-generated
/// wrappers the disk lite-parse also skips).
fn is_skipped_first_prompt(s: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "<local-command-stdout>",
        "<session-start-hook>",
        "<tick>",
        "<goal>",
    ];
    if PREFIXES.iter().any(|p| s.starts_with(p)) {
        return true;
    }
    // `[Request interrupted by user...]` — leading bracketed banner, no
    // embedded `]` before the close.
    if let Some(rest) = s.strip_prefix("[Request interrupted by user") {
        if let Some(close) = rest.find(']') {
            if !rest[..close].contains(']') {
                return true;
            }
        }
    }
    let trimmed = s.trim();
    for (open, close) in [
        ("<ide_opened_file>", "</ide_opened_file>"),
        ("<ide_selection>", "</ide_selection>"),
    ] {
        if trimmed.starts_with(open) && trimmed.ends_with(close) {
            return true;
        }
    }
    false
}

/// Folds first-prompt state for a single parsed entry, mutating `data`:
/// sets `first_prompt` + `first_prompt_locked` on a real match, or stashes a
/// `command_fallback` for slash-command messages.
fn fold_first_prompt(data: &mut Map<String, Value>, entry: &Map<String, Value>) {
    if data.get("first_prompt_locked").and_then(Value::as_bool) == Some(true) {
        return;
    }
    if entry.get("type").and_then(Value::as_str) != Some("user") {
        return;
    }
    if entry.get("isMeta").and_then(Value::as_bool) == Some(true)
        || entry.get("isCompactSummary").and_then(Value::as_bool) == Some(true)
    {
        return;
    }
    if has_tool_result(entry) {
        return;
    }

    for raw in entry_text_blocks(entry) {
        let result = raw.replace('\n', " ");
        let result = result.trim();
        if result.is_empty() {
            continue;
        }
        if let Some(cmd) = command_name(result) {
            if !data.contains_key("command_fallback") {
                data.insert("command_fallback".into(), Value::String(cmd.to_string()));
            }
            continue;
        }
        if is_skipped_first_prompt(result) {
            continue;
        }
        let value = if result.chars().count() > 200 {
            let truncated: String = result.chars().take(200).collect();
            format!("{}\u{2026}", truncated.trim_end())
        } else {
            result.to_string()
        };
        data.insert("first_prompt".into(), Value::String(value));
        data.insert("first_prompt_locked".into(), Value::Bool(true));
        return;
    }
}

/// Folds a batch of appended entries into the running summary for `key`.
///
/// Stores call this from inside `append()` to keep a [`SessionSummaryEntry`]
/// sidecar up to date without re-reading the transcript. `prev` is the previous
/// summary for the same key (or `None` for the first append).
///
/// Do not call this for keys with a `subpath` — subagent transcripts must not
/// contribute to the main session's summary.
///
/// `mtime` is NOT touched by the fold (it preserves `prev.mtime`, or `0` for a
/// new session); the adapter stamps storage write time after persisting.
#[must_use]
pub fn fold_session_summary(
    prev: Option<&SessionSummaryEntry>,
    key: &SessionKey,
    entries: &[SessionStoreEntry],
) -> SessionSummaryEntry {
    let mut summary = match prev {
        Some(p) => SessionSummaryEntry {
            session_id: p.session_id.clone(),
            mtime: p.mtime,
            data: p.data.clone(),
        },
        None => SessionSummaryEntry {
            session_id: key.session_id.clone(),
            mtime: 0,
            data: Map::new(),
        },
    };
    let data = &mut summary.data;

    for entry in entries {
        let ms = iso_to_epoch_ms(entry.get("timestamp"));

        if !data.contains_key("is_sidechain") {
            let is_sidechain = entry.get("isSidechain").and_then(Value::as_bool) == Some(true);
            data.insert("is_sidechain".into(), Value::Bool(is_sidechain));
        }
        if !data.contains_key("created_at") {
            if let Some(ms) = ms {
                data.insert("created_at".into(), Value::from(ms));
            }
        }
        if !data.contains_key("cwd") {
            if let Some(cwd) = entry.get("cwd").and_then(Value::as_str) {
                if !cwd.is_empty() {
                    data.insert("cwd".into(), Value::String(cwd.to_string()));
                }
            }
        }

        fold_first_prompt(data, entry);

        for (src, dst) in LAST_WINS_FIELDS {
            if let Some(val) = entry.get(*src).and_then(Value::as_str) {
                data.insert((*dst).to_string(), Value::String(val.to_string()));
            }
        }

        if entry.get("type").and_then(Value::as_str) == Some("tag") {
            match entry.get("tag").and_then(Value::as_str) {
                Some(tag) if !tag.is_empty() => {
                    data.insert("tag".into(), Value::String(tag.to_string()));
                }
                // Empty string or absent tag clears the tag.
                _ => {
                    data.remove("tag");
                }
            }
        }
    }

    summary
}

/// Converts a [`SessionSummaryEntry`] to [`SDKSessionInfo`].
///
/// Returns `None` for sidechain sessions or sessions with no extractable
/// summary, matching the disk lite-parse filtering.
#[must_use]
pub fn summary_entry_to_sdk_info(
    entry: &SessionSummaryEntry,
    project_path: Option<&str>,
) -> Option<SDKSessionInfo> {
    let data = &entry.data;
    if data.get("is_sidechain").and_then(Value::as_bool) == Some(true) {
        return None;
    }

    let str_field = |k: &str| data.get(k).and_then(Value::as_str).map(str::to_string);

    let first_prompt = if data.get("first_prompt_locked").and_then(Value::as_bool) == Some(true) {
        str_field("first_prompt")
    } else {
        str_field("command_fallback")
    }
    .filter(|s| !s.is_empty());

    let custom_title = str_field("custom_title")
        .or_else(|| str_field("ai_title"))
        .filter(|s| !s.is_empty());

    let summary = custom_title
        .clone()
        .or_else(|| str_field("last_prompt"))
        .or_else(|| str_field("summary_hint"))
        .or_else(|| first_prompt.clone())
        .filter(|s| !s.is_empty())?;

    Some(SDKSessionInfo {
        session_id: entry.session_id.clone(),
        summary,
        last_modified: entry.mtime,
        // file_size is meaningful only for the local-disk path.
        file_size: None,
        custom_title,
        first_prompt,
        git_branch: str_field("git_branch").filter(|s| !s.is_empty()),
        cwd: str_field("cwd")
            .filter(|s| !s.is_empty())
            .or_else(|| project_path.map(str::to_string)),
        tag: str_field("tag").filter(|s| !s.is_empty()),
        created_at: data.get("created_at").and_then(Value::as_i64),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry(v: Value) -> SessionStoreEntry {
        match v {
            Value::Object(m) => m,
            _ => panic!("entry must be a JSON object"),
        }
    }

    fn key() -> SessionKey {
        SessionKey::new("proj", Uuid::from_bytes([7; 16]))
    }

    #[test]
    fn iso_to_epoch_ms_delegates_and_trims() {
        // 2024-01-02T03:04:05Z -> known epoch.
        let z = iso_to_epoch_ms(Some(&json!("2024-01-02T03:04:05Z"))).unwrap();
        assert_eq!(z, 1_704_164_645_000);
        // With milliseconds.
        assert_eq!(
            iso_to_epoch_ms(Some(&json!("2024-01-02T03:04:05.250Z"))),
            Some(1_704_164_645_250)
        );
        // Offset +01:00 shifts back one hour.
        assert_eq!(
            iso_to_epoch_ms(Some(&json!("2024-01-02T04:04:05+01:00"))),
            Some(z)
        );
        // Surrounding whitespace is trimmed before parsing.
        assert_eq!(
            iso_to_epoch_ms(Some(&json!("  2024-01-02T03:04:05Z  "))),
            Some(z)
        );
        // Delegated strictness: garbage and non-strings are None.
        assert!(iso_to_epoch_ms(Some(&json!("nonsense"))).is_none());
        assert!(iso_to_epoch_ms(Some(&json!("2024-01-02T03:04:05Zextra"))).is_none());
        assert!(iso_to_epoch_ms(Some(&json!(42))).is_none());
        assert!(iso_to_epoch_ms(None).is_none());
    }

    #[test]
    fn fold_created_at_set_once_first_prompt_and_last_wins() {
        let mut s = fold_session_summary(
            None,
            &key(),
            &[entry(json!({
                "type": "user",
                "timestamp": "2024-01-02T03:04:05Z",
                "cwd": "/work",
                "message": {"content": "hello world"}
            }))],
        );
        assert_eq!(
            s.data.get("created_at").and_then(Value::as_i64),
            Some(1_704_164_645_000)
        );
        assert_eq!(s.data.get("cwd").and_then(Value::as_str), Some("/work"));
        assert_eq!(
            s.data.get("first_prompt").and_then(Value::as_str),
            Some("hello world")
        );
        assert_eq!(
            s.data.get("first_prompt_locked").and_then(Value::as_bool),
            Some(true)
        );

        // Second batch: created_at/cwd/first_prompt latched; gitBranch last-wins.
        s = fold_session_summary(
            Some(&s),
            &key(),
            &[entry(json!({
                "type": "user",
                "timestamp": "2024-06-06T00:00:00Z",
                "cwd": "/other",
                "gitBranch": "main",
                "message": {"content": "second"}
            }))],
        );
        assert_eq!(
            s.data.get("created_at").and_then(Value::as_i64),
            Some(1_704_164_645_000)
        );
        assert_eq!(s.data.get("cwd").and_then(Value::as_str), Some("/work"));
        assert_eq!(
            s.data.get("first_prompt").and_then(Value::as_str),
            Some("hello world")
        );
        assert_eq!(
            s.data.get("git_branch").and_then(Value::as_str),
            Some("main")
        );
    }

    #[test]
    fn fold_command_fallback_then_real_prompt() {
        let s = fold_session_summary(
            None,
            &key(),
            &[
                entry(
                    json!({"type": "user", "message": {"content": "<command-name>clear</command-name>"}}),
                ),
                entry(json!({"type": "user", "message": {"content": "real question"}})),
            ],
        );
        assert_eq!(
            s.data.get("command_fallback").and_then(Value::as_str),
            Some("clear")
        );
        assert_eq!(
            s.data.get("first_prompt").and_then(Value::as_str),
            Some("real question")
        );
    }

    #[test]
    fn fold_skips_tool_result_and_meta() {
        let s = fold_session_summary(
            None,
            &key(),
            &[
                entry(json!({"type": "user", "isMeta": true, "message": {"content": "meta"}})),
                entry(
                    json!({"type": "user", "message": {"content": [{"type": "tool_result", "content": "x"}]}}),
                ),
                entry(json!({"type": "user", "message": {"content": "actual"}})),
            ],
        );
        assert_eq!(
            s.data.get("first_prompt").and_then(Value::as_str),
            Some("actual")
        );
    }

    #[test]
    fn fold_tag_set_then_cleared() {
        let mut s =
            fold_session_summary(None, &key(), &[entry(json!({"type": "tag", "tag": "bug"}))]);
        assert_eq!(s.data.get("tag").and_then(Value::as_str), Some("bug"));
        s = fold_session_summary(
            Some(&s),
            &key(),
            &[entry(json!({"type": "tag", "tag": ""}))],
        );
        assert!(!s.data.contains_key("tag"));
    }

    #[test]
    fn sidechain_yields_no_info() {
        let s = fold_session_summary(
            None,
            &key(),
            &[entry(
                json!({"type": "user", "isSidechain": true, "message": {"content": "x"}}),
            )],
        );
        assert!(summary_entry_to_sdk_info(&s, None).is_none());
    }

    #[test]
    fn info_summary_precedence_custom_title_wins() {
        let mut s = fold_session_summary(
            None,
            &key(),
            &[entry(json!({
                "type": "user",
                "customTitle": "My Title",
                "lastPrompt": "later",
                "message": {"content": "first prompt here"}
            }))],
        );
        s.mtime = 999;
        let info = summary_entry_to_sdk_info(&s, Some("/proj")).unwrap();
        assert_eq!(info.summary, "My Title");
        assert_eq!(info.custom_title.as_deref(), Some("My Title"));
        assert_eq!(info.first_prompt.as_deref(), Some("first prompt here"));
        assert_eq!(info.last_modified, 999);
    }

    #[test]
    fn info_falls_back_to_project_path_cwd() {
        let s = fold_session_summary(
            None,
            &key(),
            &[entry(
                json!({"type": "user", "message": {"content": "prompt"}}),
            )],
        );
        let info = summary_entry_to_sdk_info(&s, Some("/fallback")).unwrap();
        assert_eq!(info.cwd.as_deref(), Some("/fallback"));
        assert_eq!(info.summary, "prompt");
    }
}
