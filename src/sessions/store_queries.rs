//! [`SessionStore`]-backed session query APIs.
//!
//! Async, store-backed counterparts to the local-disk query functions in
//! [`crate::sessions::filesystem`]. Ported from the official Python Agent SDK
//! `_internal/sessions.py` (`*_from_store` functions): they load transcripts
//! from a [`SessionStore`] and reuse the same lite-parse / chain-building the
//! filesystem path uses, so disk and store paths produce identical results for
//! the same transcript content.
//!
//! - [`list_sessions_from_store`] — summary fast path + bounded gap-fill loads.
//! - [`get_session_info_from_store`] — single-session lite metadata.
//! - [`get_session_messages_from_store`] — conversation chain from entries.
//! - [`list_subagents_from_store`] — subagent id enumeration via `list_subkeys`.
//! - [`get_subagent_messages_from_store`] — subagent chain (subkey scan or
//!   direct path fallback).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde_json::Value;

use super::filesystem::{
    apply_sort_limit_offset, entries_to_session_messages, entries_to_subagent_messages,
    parse_iso8601_ms, parse_session_info_from_lite, validate_uuid, LiteSessionFile, SessionMessage,
    LITE_READ_BUF_SIZE,
};
use super::key::{canonicalize_path, project_key_for_directory, SessionKey, SessionListSubkeysKey};
use super::store::{SDKSessionInfo, SessionStore, SessionStoreEntry};
use super::summary::summary_entry_to_sdk_info;
use crate::error::ClaudeError;

/// Upper bound on concurrent `store.load()` calls issued by
/// [`list_sessions_from_store`]. Keeps large project listings from exhausting
/// adapter connection pools or tripping backend rate limits. Mirrors the
/// official `_STORE_LIST_LOAD_CONCURRENCY`.
const STORE_LIST_LOAD_CONCURRENCY: usize = 16;

/// Transcript entry types that carry `uuid` + `parentUuid` chain links.
const TRANSCRIPT_ENTRY_TYPES: [&str; 5] = ["user", "assistant", "progress", "system", "attachment"];

// ---------------------------------------------------------------------------
// JSONL / lite helpers (object path — no full re-parse where avoidable)
// ---------------------------------------------------------------------------

/// Serialize store entries to a JSONL string (one compact line per entry).
///
/// `SessionStore::load` permits adapters to reorder object keys (e.g. Postgres
/// JSONB), but the lite-parse scans for the `{"type":"tag"` line prefix. Hoist
/// `type` to the front so the store path matches the byte shape the disk path
/// produces. Mirrors the official `_entries_to_jsonl`.
fn entries_to_jsonl(entries: &[SessionStoreEntry]) -> String {
    let mut out = String::new();
    for e in entries {
        let value = if e.contains_key("type") {
            // Rebuild with `type` first, then the rest in original order.
            let mut ordered = serde_json::Map::with_capacity(e.len());
            if let Some(t) = e.get("type") {
                ordered.insert("type".to_string(), t.clone());
            }
            for (k, v) in e {
                if k != "type" {
                    ordered.insert(k.clone(), v.clone());
                }
            }
            Value::Object(ordered)
        } else {
            Value::Object(e.clone())
        };
        // serde's compact serialization matches the disk transcript's
        // `json.dumps(separators=(",", ":"))` byte shape.
        out.push_str(&serde_json::to_string(&value).unwrap_or_default());
        out.push('\n');
    }
    out
}

/// Build the head/tail/size lite shape from an in-memory JSONL string, matching
/// `read_session_lite`'s byte semantics. Mirrors `_jsonl_to_lite`.
fn jsonl_to_lite(jsonl: &str, mtime: i64) -> LiteSessionFile {
    let buf = jsonl.as_bytes();
    let size = buf.len();
    let head = decode_lossy(&buf[..size.min(LITE_READ_BUF_SIZE)]);
    let tail = if size > LITE_READ_BUF_SIZE {
        decode_lossy(&buf[size - LITE_READ_BUF_SIZE..])
    } else {
        head.clone()
    };
    LiteSessionFile {
        mtime,
        size: i64::try_from(size).unwrap_or(i64::MAX),
        head,
        tail,
    }
}

/// UTF-8 decode a byte slice, replacing invalid sequences (mirrors Python's
/// `bytes.decode("utf-8", errors="replace")`).
fn decode_lossy(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Best-effort mtime: parse the last entry's `timestamp` field, falling back to
/// the current wall-clock time. Mirrors `_mtime_from_jsonl_tail`.
fn mtime_from_jsonl_tail(jsonl: &str) -> i64 {
    let trimmed = jsonl.trim_end();
    let last_line = match trimmed.rfind('\n') {
        Some(i) => &trimmed[i + 1..],
        None => trimmed,
    };
    if let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(last_line) {
        if let Some(ts) = obj.get("timestamp").and_then(Value::as_str) {
            if let Some(ms) = parse_iso8601_ms(ts) {
                return ms;
            }
        }
    }
    now_ms()
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// Filter store-loaded entries to transcript message types with a `uuid`.
/// Mirrors `_filter_transcript_entries` so chain-building never sees
/// metadata-only entries (custom-title, tag, `agent_metadata`, ...).
fn filter_transcript_entries(entries: &[SessionStoreEntry]) -> Vec<SessionStoreEntry> {
    entries
        .iter()
        .filter(|e| {
            e.get("type")
                .and_then(Value::as_str)
                .is_some_and(|t| TRANSCRIPT_ENTRY_TYPES.contains(&t))
                && e.get("uuid").and_then(Value::as_str).is_some()
        })
        .cloned()
        .collect()
}

/// Load a session's entries from a store and serialize to a JSONL string.
/// Returns `None` if the session has no entries. Mirrors
/// `_load_store_entries_as_jsonl`.
async fn load_store_entries_as_jsonl(
    store: &dyn SessionStore,
    session_id: &str,
    directory: Option<&str>,
) -> Result<Option<String>, ClaudeError> {
    let project_key = project_key_for_directory(directory.map(std::path::Path::new));
    let key = SessionKey::new(project_key, session_id);
    let entries = store.load(&key).await?;
    match entries {
        Some(entries) if !entries.is_empty() => Ok(Some(entries_to_jsonl(&entries))),
        _ => Ok(None),
    }
}

/// A pagination slot: either a resolved summary or a placeholder awaiting a
/// gap-fill `load()`.
struct Slot {
    mtime: i64,
    session_id: String,
    info: Option<SDKSessionInfo>,
}

/// Derive [`SDKSessionInfo`] for each listing entry via a bounded-concurrency
/// per-session `store.load()` + lite-parse. Adapter errors degrade that row to
/// an empty summary instead of failing the whole list; sidechain and
/// no-summary sessions are dropped. Mirrors `_derive_infos_via_load`.
async fn derive_infos_via_load(
    store: Arc<dyn SessionStore>,
    listing: &[(String, i64)],
    directory: Option<&str>,
    project_path: &str,
) -> Vec<SDKSessionInfo> {
    use futures::future::join_all;
    let limiter = Arc::new(tokio::sync::Semaphore::new(STORE_LIST_LOAD_CONCURRENCY));
    let dir_owned = directory.map(str::to_string);

    let loads = listing.iter().map(|(sid, _mtime)| {
        let store = Arc::clone(&store);
        let limiter = Arc::clone(&limiter);
        let sid = sid.clone();
        let dir_owned = dir_owned.clone();
        async move {
            let _permit = limiter.acquire().await;
            // Ok(None) = no entries (drop); Ok(Some) = jsonl; Err = adapter
            // error (degrade to empty summary).
            load_store_entries_as_jsonl(store.as_ref(), &sid, dir_owned.as_deref()).await
        }
    });
    let settled = join_all(loads).await;

    let mut results = Vec::new();
    for ((sid, mtime), outcome) in listing.iter().zip(settled) {
        match outcome {
            Err(_) => {
                // Adapter is user code; a load failure degrades this row to an
                // empty summary rather than failing the whole list.
                results.push(SDKSessionInfo {
                    session_id: sid.clone(),
                    summary: String::new(),
                    last_modified: *mtime,
                    ..Default::default()
                });
            }
            Ok(None) => {} // no entries — drop
            Ok(Some(jsonl)) => {
                let lite = jsonl_to_lite(&jsonl, *mtime);
                if let Some(mut parsed) =
                    parse_session_info_from_lite(sid, &lite, Some(project_path))
                {
                    parsed.last_modified = *mtime;
                    results.push(parsed);
                }
                // else: sidechain or no extractable summary — drop, matching
                // the filesystem path.
            }
        }
    }
    results
}

async fn build_summary_slots(
    store: &dyn SessionStore,
    project_key: &str,
    project_path: &str,
    has_list_sessions: bool,
) -> Result<Vec<Slot>, ClaudeError> {
    let summaries = store.list_session_summaries(project_key).await?;
    let (listing, known_mtimes): (Vec<(String, i64)>, HashMap<String, i64>) = if has_list_sessions {
        let listing: Vec<(String, i64)> = store
            .list_sessions(project_key)
            .await?
            .into_iter()
            .map(|entry| (entry.session_id, entry.mtime))
            .collect();
        let known = listing.iter().cloned().collect();
        (listing, known)
    } else {
        tracing::debug!(
            "list_session_summaries without list_sessions: gap-fill \
                 skipped; sessions lacking a sidecar will be omitted"
        );
        (Vec::new(), HashMap::new())
    };

    let mut slots = Vec::new();
    let mut fresh_summary_ids = HashSet::new();
    for summary in &summaries {
        let session_id = summary.session_id.clone();
        if has_list_sessions {
            match known_mtimes.get(&session_id) {
                None => continue,
                Some(&known) if summary.mtime < known => continue,
                _ => {}
            }
        }
        match summary_entry_to_sdk_info(summary, Some(project_path)) {
            Some(info) => {
                slots.push(Slot {
                    mtime: summary.mtime,
                    session_id: session_id.clone(),
                    info: Some(info),
                });
                fresh_summary_ids.insert(session_id);
            }
            None => {
                // Sidechain and empty summaries do not consume page slots.
                fresh_summary_ids.insert(session_id);
            }
        }
    }
    if has_list_sessions {
        for (session_id, mtime) in listing {
            if !fresh_summary_ids.contains(&session_id) {
                slots.push(Slot {
                    mtime,
                    session_id,
                    info: None,
                });
            }
        }
    }
    Ok(slots)
}

async fn paginate_and_fill_summary_slots(
    store: Arc<dyn SessionStore>,
    mut slots: Vec<Slot>,
    directory: Option<&str>,
    project_path: &str,
    limit: Option<usize>,
    offset: usize,
) -> Vec<SDKSessionInfo> {
    // Paginate before loading gaps so adapter work is bounded by page size.
    slots.sort_by_key(|slot| std::cmp::Reverse(slot.mtime));
    let mut page: Vec<Slot> = slots.into_iter().skip(offset).collect();
    if let Some(limit) = limit.filter(|limit| *limit > 0) {
        page.truncate(limit);
    }

    let to_fill: Vec<(String, i64)> = page
        .iter()
        .filter(|slot| slot.info.is_none())
        .map(|slot| (slot.session_id.clone(), slot.mtime))
        .collect();
    if !to_fill.is_empty() {
        let filled =
            derive_infos_via_load(Arc::clone(&store), &to_fill, directory, project_path).await;
        let by_session_id: HashMap<String, SDKSessionInfo> = filled
            .into_iter()
            .map(|info| (info.session_id.clone(), info))
            .collect();
        for slot in &mut page {
            if slot.info.is_none() {
                slot.info = by_session_id.get(&slot.session_id).cloned();
            }
        }
    }

    // Gaps that resolve to a sidechain or empty transcript are omitted.
    page.into_iter().filter_map(|slot| slot.info).collect()
}

// ---------------------------------------------------------------------------
// Public store-backed query APIs
// ---------------------------------------------------------------------------

/// List sessions from a [`SessionStore`].
///
/// Async, store-backed counterpart to
/// [`list_sessions`](crate::sessions::list_sessions). If the store maintains
/// incremental summaries, this is one batch summary call plus one cheap
/// `list_sessions()` enumeration to gap-fill sessions missing or with a stale
/// sidecar — zero per-session `load()` calls when sidecars are complete and
/// fresh. Otherwise falls back to one `load()` per session (bounded at 16
/// concurrent).
///
/// `include_worktrees` is a filesystem concept and is not honored here — the
/// store operates on a single `project_key`.
///
/// # Errors
/// Returns [`ClaudeError`] if the store implements neither
/// [`list_session_summaries`](SessionStore::list_session_summaries) nor
/// [`list_sessions`](SessionStore::list_sessions), or an adapter call fails.
pub async fn list_sessions_from_store(
    session_store: Arc<dyn SessionStore>,
    directory: Option<&str>,
    limit: Option<usize>,
    offset: usize,
) -> Result<Vec<SDKSessionInfo>, ClaudeError> {
    let project_path = canonicalize_path(directory.unwrap_or("."));
    let project_key = project_key_for_directory(directory.map(std::path::Path::new));
    let caps = session_store.capabilities();
    let has_list_sessions = caps.list_sessions;

    // Fast path: fresh summary sidecars avoid per-session transcript loads.
    if caps.list_session_summaries {
        let slots = build_summary_slots(
            session_store.as_ref(),
            &project_key,
            &project_path,
            has_list_sessions,
        )
        .await?;
        return Ok(paginate_and_fill_summary_slots(
            session_store,
            slots,
            directory,
            &project_path,
            limit,
            offset,
        )
        .await);
    }

    if !has_list_sessions {
        return Err(ClaudeError::InvalidConfig(
            "session_store implements neither list_session_summaries() nor \
             list_sessions() -- cannot list sessions. Provide a store with at \
             least one of those methods."
                .to_string(),
        ));
    }

    let listing: Vec<(String, i64)> = session_store
        .list_sessions(&project_key)
        .await?
        .into_iter()
        .map(|e| (e.session_id, e.mtime))
        .collect();
    let results = derive_infos_via_load(
        Arc::clone(&session_store),
        &listing,
        directory,
        &project_path,
    )
    .await;
    Ok(apply_sort_limit_offset(results, limit, offset))
}

/// Read metadata for a single session from a [`SessionStore`].
///
/// Async, store-backed counterpart to
/// [`get_session_info`](crate::sessions::get_session_info). Returns `None` if
/// the session is not found, `session_id` is not a valid UUID, the session is a
/// sidechain, or it has no extractable summary.
///
/// # Errors
/// Adapter load failures.
pub async fn get_session_info_from_store(
    session_store: &dyn SessionStore,
    session_id: &str,
    directory: Option<&str>,
) -> Result<Option<SDKSessionInfo>, ClaudeError> {
    if validate_uuid(session_id).is_none() {
        return Ok(None);
    }
    let Some(jsonl) = load_store_entries_as_jsonl(session_store, session_id, directory).await?
    else {
        return Ok(None);
    };
    let lite = jsonl_to_lite(&jsonl, mtime_from_jsonl_tail(&jsonl));
    let project_path = canonicalize_path(directory.unwrap_or("."));
    Ok(parse_session_info_from_lite(
        session_id,
        &lite,
        Some(&project_path),
    ))
}

/// Read a session's conversation messages from a [`SessionStore`].
///
/// Async, store-backed counterpart to
/// [`get_session_messages`](crate::sessions::get_session_messages). Feeds
/// `store.load()` results directly into the chain builder — no JSONL
/// round-trip. Empty result if the session is not found or `session_id` is
/// invalid.
///
/// # Errors
/// Adapter load failures.
pub async fn get_session_messages_from_store(
    session_store: &dyn SessionStore,
    session_id: &str,
    directory: Option<&str>,
    limit: Option<usize>,
    offset: usize,
) -> Result<Vec<SessionMessage>, ClaudeError> {
    if validate_uuid(session_id).is_none() {
        return Ok(Vec::new());
    }
    let project_key = project_key_for_directory(directory.map(std::path::Path::new));
    let key = SessionKey::new(project_key, session_id);
    let entries = match session_store.load(&key).await? {
        Some(entries) if !entries.is_empty() => entries,
        _ => return Ok(Vec::new()),
    };
    let filtered = filter_transcript_entries(&entries);
    Ok(entries_to_session_messages(&filtered, limit, offset))
}

/// List subagent IDs for a session from a [`SessionStore`].
///
/// Async, store-backed counterpart to
/// [`list_subagents`](crate::sessions::list_subagents). Empty result if
/// `session_id` is invalid or the session has no subagents.
///
/// # Errors
/// Returns [`ClaudeError`] if the store does not implement
/// [`list_subkeys`](SessionStore::list_subkeys), or an adapter call fails.
pub async fn list_subagents_from_store(
    session_store: &dyn SessionStore,
    session_id: &str,
    directory: Option<&str>,
) -> Result<Vec<String>, ClaudeError> {
    if validate_uuid(session_id).is_none() {
        return Ok(Vec::new());
    }
    if !session_store.capabilities().list_subkeys {
        return Err(ClaudeError::InvalidConfig(
            "session_store does not implement list_subkeys() -- cannot list \
             subagents. Provide a store with a list_subkeys() method."
                .to_string(),
        ));
    }
    let project_key = project_key_for_directory(directory.map(std::path::Path::new));
    let subkeys = session_store
        .list_subkeys(&SessionListSubkeysKey::new(project_key, session_id))
        .await?;
    let mut seen: HashSet<String> = HashSet::new();
    let mut ids: Vec<String> = Vec::new();
    for subpath in subkeys {
        if !subpath.starts_with("subagents/") {
            continue;
        }
        let last = subpath.rsplit('/').next().unwrap_or(&subpath);
        if let Some(agent_id) = last.strip_prefix("agent-") {
            if seen.insert(agent_id.to_string()) {
                ids.push(agent_id.to_string());
            }
        }
    }
    Ok(ids)
}

/// Read a subagent's conversation messages from a [`SessionStore`].
///
/// Async, store-backed counterpart to
/// [`get_subagent_messages`](crate::sessions::get_subagent_messages).
/// Subagents may live at `subagents/agent-<id>` or nested under
/// `subagents/workflows/<runId>/agent-<id>`. Scans subkeys when the store
/// implements [`list_subkeys`](SessionStore::list_subkeys); otherwise tries the
/// direct path. Empty result if the session/subagent is not found.
///
/// # Errors
/// Adapter load failures.
pub async fn get_subagent_messages_from_store(
    session_store: &dyn SessionStore,
    session_id: &str,
    agent_id: &str,
    directory: Option<&str>,
    limit: Option<usize>,
    offset: usize,
) -> Result<Vec<SessionMessage>, ClaudeError> {
    if validate_uuid(session_id).is_none() || agent_id.is_empty() {
        return Ok(Vec::new());
    }
    let project_key = project_key_for_directory(directory.map(std::path::Path::new));

    let mut subpath = format!("subagents/agent-{agent_id}");
    if session_store.capabilities().list_subkeys {
        let subkeys = session_store
            .list_subkeys(&SessionListSubkeysKey::new(project_key.clone(), session_id))
            .await?;
        let target = format!("agent-{agent_id}");
        let matched = subkeys.into_iter().find(|sk| {
            sk.starts_with("subagents/") && sk.rsplit('/').next() == Some(target.as_str())
        });
        let Some(matched) = matched else {
            return Ok(Vec::new());
        };
        subpath = matched;
    }

    let Ok(key) = SessionKey::with_subpath(project_key, session_id, subpath) else {
        return Ok(Vec::new());
    };
    let entries = match session_store.load(&key).await? {
        Some(entries) if !entries.is_empty() => entries,
        _ => return Ok(Vec::new()),
    };

    // Drop synthetic agent_metadata entries injected by the mirror hook — they
    // describe the .meta.json sidecar, not transcript lines.
    let transcript: Vec<SessionStoreEntry> = entries
        .into_iter()
        .filter(|e| e.get("type").and_then(Value::as_str) != Some("agent_metadata"))
        .collect();
    if transcript.is_empty() {
        return Ok(Vec::new());
    }
    let filtered = filter_transcript_entries(&transcript);
    Ok(entries_to_subagent_messages(&filtered, limit, offset))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sessions::store::InMemorySessionStore;
    use serde_json::json;

    fn entry(v: Value) -> SessionStoreEntry {
        match v {
            Value::Object(m) => m,
            _ => panic!("entry must be a JSON object"),
        }
    }

    const SID: &str = "11111111-1111-4111-8111-111111111111";
    const SID2: &str = "22222222-2222-4222-8222-222222222222";

    async fn store_with_session(directory: &str) -> Arc<InMemorySessionStore> {
        let store = Arc::new(InMemorySessionStore::new());
        let pk = project_key_for_directory(Some(std::path::Path::new(directory)));
        let key = SessionKey::new(pk, SID);
        store
            .append(
                &key,
                vec![
                    entry(json!({
                        "type": "user",
                        "uuid": "u1",
                        "parentUuid": null,
                        "sessionId": SID,
                        "cwd": directory,
                        "timestamp": "2026-08-17T00:00:00Z",
                        "message": {"content": "the first prompt"}
                    })),
                    entry(json!({
                        "type": "assistant",
                        "uuid": "a1",
                        "parentUuid": "u1",
                        "sessionId": SID,
                        "message": {"content": "answer"}
                    })),
                ],
            )
            .await
            .unwrap();
        store
    }

    #[tokio::test]
    async fn get_info_rejects_non_canonical_uuid() {
        let store = InMemorySessionStore::new();
        // Braced form parses under uuid::Uuid but is not canonical.
        let braced = "{11111111-1111-4111-8111-111111111111}";
        let info = get_session_info_from_store(&store, braced, Some("/tmp/x"))
            .await
            .unwrap();
        assert!(info.is_none(), "non-canonical uuid rejected");
    }

    #[tokio::test]
    async fn get_info_returns_summary_from_store() {
        let dir = "/tmp/store-queries-info";
        let store = store_with_session(dir).await;
        let info = get_session_info_from_store(store.as_ref(), SID, Some(dir))
            .await
            .unwrap()
            .expect("session info");
        assert_eq!(info.session_id, SID);
        assert_eq!(info.summary, "the first prompt");
        assert_eq!(info.first_prompt.as_deref(), Some("the first prompt"));
    }

    #[tokio::test]
    async fn get_messages_builds_chain() {
        let dir = "/tmp/store-queries-msgs";
        let store = store_with_session(dir).await;
        let msgs = get_session_messages_from_store(store.as_ref(), SID, Some(dir), None, 0)
            .await
            .unwrap();
        let uuids: Vec<&str> = msgs.iter().map(|m| m.uuid.as_str()).collect();
        assert_eq!(uuids, vec!["u1", "a1"]);
    }

    #[tokio::test]
    async fn list_uses_summary_fast_path() {
        let dir = "/tmp/store-queries-list";
        let store = store_with_session(dir).await;
        let infos = list_sessions_from_store(store.clone(), Some(dir), None, 0)
            .await
            .unwrap();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].session_id, SID);
        assert_eq!(infos[0].summary, "the first prompt");
    }

    #[tokio::test]
    async fn list_paginates_by_mtime_desc() {
        let dir = "/tmp/store-queries-page";
        let store = store_with_session(dir).await;
        // Second session, appended later => higher mtime => sorts first.
        let pk = project_key_for_directory(Some(std::path::Path::new(dir)));
        store
            .append(
                &SessionKey::new(pk, SID2),
                vec![entry(json!({
                    "type": "user",
                    "uuid": "u2",
                    "sessionId": SID2,
                    "timestamp": "2026-08-18T00:00:00Z",
                    "message": {"content": "second session"}
                }))],
            )
            .await
            .unwrap();
        let page = list_sessions_from_store(store.clone(), Some(dir), Some(1), 0)
            .await
            .unwrap();
        assert_eq!(page.len(), 1, "limit=1 returns one");
        assert_eq!(page[0].session_id, SID2, "newest first");
        let page2 = list_sessions_from_store(store.clone(), Some(dir), Some(1), 1)
            .await
            .unwrap();
        assert_eq!(page2.len(), 1);
        assert_eq!(page2[0].session_id, SID, "offset skips newest");
    }

    #[tokio::test]
    async fn list_subagents_dedupes_and_strips_prefix() {
        let dir = "/tmp/store-queries-subs";
        let store = store_with_session(dir).await;
        let pk = project_key_for_directory(Some(std::path::Path::new(dir)));
        store
            .append(
                &SessionKey::with_subpath(pk, SID, "subagents/agent-inv").unwrap(),
                vec![entry(json!({"type": "assistant", "uuid": "s1", "sessionId": SID, "message": {"content": "sub"}}))],
            )
            .await
            .unwrap();
        let ids = list_subagents_from_store(store.as_ref(), SID, Some(dir))
            .await
            .unwrap();
        assert_eq!(ids, vec!["inv"]);
    }

    #[tokio::test]
    async fn get_subagent_messages_drops_metadata() {
        let dir = "/tmp/store-queries-submsg";
        let store = store_with_session(dir).await;
        let pk = project_key_for_directory(Some(std::path::Path::new(dir)));
        store
            .append(
                &SessionKey::with_subpath(pk, SID, "subagents/agent-inv").unwrap(),
                vec![
                    entry(json!({"type": "agent_metadata", "agentType": "investigator"})),
                    entry(json!({"type": "user", "uuid": "su1", "sessionId": SID, "message": {"content": "hi"}})),
                    entry(json!({"type": "assistant", "uuid": "sa1", "parentUuid": "su1", "sessionId": SID, "message": {"content": "yo"}})),
                ],
            )
            .await
            .unwrap();
        let msgs = get_subagent_messages_from_store(store.as_ref(), SID, "inv", Some(dir), None, 0)
            .await
            .unwrap();
        let uuids: Vec<&str> = msgs.iter().map(|m| m.uuid.as_str()).collect();
        assert_eq!(uuids, vec!["su1", "sa1"]);
    }

    #[tokio::test]
    async fn list_requires_a_capability() {
        // A store with neither list_sessions nor list_session_summaries.
        #[derive(Debug)]
        struct Bare;
        #[async_trait::async_trait]
        impl SessionStore for Bare {
            async fn append(
                &self,
                _k: &SessionKey,
                _e: Vec<SessionStoreEntry>,
            ) -> Result<(), ClaudeError> {
                Ok(())
            }
            async fn load(
                &self,
                _k: &SessionKey,
            ) -> Result<Option<Vec<SessionStoreEntry>>, ClaudeError> {
                Ok(None)
            }
        }
        let err = list_sessions_from_store(Arc::new(Bare), Some("/tmp/x"), None, 0)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("neither"));
    }
}
