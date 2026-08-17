//! [`SessionStore`] trait, entry/list/info types, capability probing, flush
//! mode, and the reference [`InMemorySessionStore`] adapter.
//!
//! Ported from the official Python Agent SDK
//! (`types.py::SessionStore` and `_internal/session_store.py`). Adapters mirror
//! session transcripts to external storage; only [`SessionStore::append`] and
//! [`SessionStore::load`] are required, the rest are optional and gated behind
//! [`SessionStore::capabilities`] (Rust's equivalent of Python duck-typing).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
#[cfg(test)]
use uuid::Uuid;

use crate::error::ClaudeError;
use crate::sessions::key::{SessionKey, SessionListSubkeysKey};
use crate::sessions::summary::{fold_session_summary, SessionSummaryEntry};

/// One JSONL transcript line as observed by a [`SessionStore`] adapter.
///
/// The concrete shape is the CLI's on-disk transcript format (a large
/// discriminated union). That union is internal, so this is a permissive JSON
/// object: adapters treat entries as pass-through blobs; the only required
/// invariant is that a round-trip preserves the object (deep-equal, not
/// byte-equal — key order may change).
pub type SessionStoreEntry = Map<String, Value>;

/// Entry returned by [`SessionStore::list_sessions`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionStoreListEntry {
    /// The session identifier (see [`SessionKey::session_id`]).
    pub session_id: String,
    /// Last-modified time in Unix epoch milliseconds. Adapters without native
    /// modification time must maintain their own index.
    pub mtime: i64,
}

/// Session metadata returned by session-listing APIs.
///
/// Contains only data extractable from stat + head/tail reads — no full JSONL
/// parsing required.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SDKSessionInfo {
    /// Unique session identifier.
    pub session_id: String,
    /// Display title — custom title, auto-generated summary, or first prompt.
    pub summary: String,
    /// Last-modified time in milliseconds since epoch.
    pub last_modified: i64,
    /// Session file size in bytes; only populated for local JSONL storage.
    pub file_size: Option<i64>,
    /// User-set custom title or AI-generated title.
    pub custom_title: Option<String>,
    /// First meaningful user prompt in the session.
    pub first_prompt: Option<String>,
    /// Git branch at the end of the session.
    pub git_branch: Option<String>,
    /// Working directory for the session.
    pub cwd: Option<String>,
    /// User-set session tag.
    pub tag: Option<String>,
    /// Creation time in milliseconds since epoch, from the first entry's ISO
    /// timestamp field.
    pub created_at: Option<i64>,
}

/// Controls when transcript-mirror entries are flushed to a [`SessionStore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum SessionStoreFlushMode {
    /// Buffer entries and flush once per turn (on `result`) or when the pending
    /// buffer exceeds 500 entries / 1 MiB.
    #[default]
    Batched,
    /// Trigger a background flush after every `transcript_mirror` frame.
    Eager,
}

/// Advertises which optional [`SessionStore`] methods an adapter implements.
///
/// Call sites probe this before invoking an optional method (Rust's equivalent
/// of the Python SDK's runtime duck-typed method probing). Required methods
/// (`append`/`load`) are always available and not listed here.
// Each bool mirrors one optional `SessionStore` method (the official
// capability-probing shape); a flag per method is the intended API, not an
// enum/bitflag candidate.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SessionStoreCapabilities {
    /// [`SessionStore::list_sessions`] is implemented.
    pub list_sessions: bool,
    /// [`SessionStore::list_session_summaries`] is implemented.
    pub list_session_summaries: bool,
    /// [`SessionStore::delete`] is implemented.
    pub delete: bool,
    /// [`SessionStore::list_subkeys`] is implemented.
    pub list_subkeys: bool,
}

/// Adapter for mirroring session transcripts to external storage.
///
/// The subprocess still writes to local disk; the adapter receives a secondary
/// copy. Only [`append`](SessionStore::append) and [`load`](SessionStore::load)
/// are required. The remaining methods are optional: the default
/// implementations return [`ClaudeError::Unsupported`], and
/// [`capabilities`](SessionStore::capabilities) reports which are actually
/// implemented so call sites can skip absent ones.
#[async_trait::async_trait]
pub trait SessionStore: std::fmt::Debug + Send + Sync {
    /// Mirror a batch of transcript entries, appending in call order.
    ///
    /// Called after the subprocess's local write succeeds.
    ///
    /// # Errors
    /// Adapter-specific persistence failures.
    async fn append(
        &self,
        key: &SessionKey,
        entries: Vec<SessionStoreEntry>,
    ) -> Result<(), ClaudeError>;

    /// Load a full session for resume. Returns `None` for a key that was never
    /// written. Entries must be deep-equal to what was appended.
    ///
    /// # Errors
    /// Adapter-specific load failures.
    async fn load(&self, key: &SessionKey) -> Result<Option<Vec<SessionStoreEntry>>, ClaudeError>;

    /// Reports which optional methods this adapter implements. Default: none.
    fn capabilities(&self) -> SessionStoreCapabilities {
        SessionStoreCapabilities::default()
    }

    /// List sessions for a `project_key` (IDs + modification times). Optional.
    ///
    /// # Errors
    /// [`ClaudeError::Unsupported`] by default; adapter errors otherwise.
    async fn list_sessions(
        &self,
        project_key: &str,
    ) -> Result<Vec<SessionStoreListEntry>, ClaudeError> {
        let _ = project_key;
        Err(ClaudeError::Unsupported(
            "SessionStore::list_sessions".into(),
        ))
    }

    /// Return incrementally-maintained summaries for all sessions in one call.
    /// Optional.
    ///
    /// # Errors
    /// [`ClaudeError::Unsupported`] by default; adapter errors otherwise.
    async fn list_session_summaries(
        &self,
        project_key: &str,
    ) -> Result<Vec<SessionSummaryEntry>, ClaudeError> {
        let _ = project_key;
        Err(ClaudeError::Unsupported(
            "SessionStore::list_session_summaries".into(),
        ))
    }

    /// Delete a session. Deleting a main-transcript key cascades to subkeys.
    /// Optional.
    ///
    /// # Errors
    /// [`ClaudeError::Unsupported`] by default; adapter errors otherwise.
    async fn delete(&self, key: &SessionKey) -> Result<(), ClaudeError> {
        let _ = key;
        Err(ClaudeError::Unsupported("SessionStore::delete".into()))
    }

    /// List all subpath keys under a session (e.g. subagent transcripts).
    /// Optional.
    ///
    /// # Errors
    /// [`ClaudeError::Unsupported`] by default; adapter errors otherwise.
    async fn list_subkeys(&self, key: &SessionListSubkeysKey) -> Result<Vec<String>, ClaudeError> {
        let _ = key;
        Err(ClaudeError::Unsupported(
            "SessionStore::list_subkeys".into(),
        ))
    }
}

// ===========================================================================
// InMemorySessionStore
// ===========================================================================

/// Internal mutable state, guarded by a single mutex so the store is
/// `Send + Sync` and appends serialize in call order.
#[derive(Debug, Default)]
struct MemState {
    /// Composite `project_key/session_id[/subpath]` → entries.
    store: HashMap<String, Vec<SessionStoreEntry>>,
    /// Composite key → storage write time (epoch ms).
    mtimes: HashMap<String, i64>,
    /// `(project_key, session_id-string)` → folded summary sidecar.
    summaries: HashMap<(String, String), SessionSummaryEntry>,
    /// Monotonic clock guard: the last mtime handed out.
    last_mtime: i64,
}

/// In-memory [`SessionStore`] for testing and development. Not durable — data
/// is lost when the process exits. Advertises all optional capabilities.
#[derive(Debug, Default)]
pub struct InMemorySessionStore {
    state: Mutex<MemState>,
}

impl InMemorySessionStore {
    /// Creates an empty in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Test/inspection helper: all entries for a key (empty if absent).
    ///
    /// # Panics
    /// If the internal mutex is poisoned by a panic in another thread.
    #[must_use]
    pub fn get_entries(&self, key: &SessionKey) -> Vec<SessionStoreEntry> {
        let state = self.state.lock().expect("session store mutex poisoned");
        state
            .store
            .get(&key.storage_key())
            .cloned()
            .unwrap_or_default()
    }

    /// Test helper: number of stored main transcripts (keys with no subpath).
    ///
    /// # Panics
    /// If the internal mutex is poisoned by a panic in another thread.
    #[must_use]
    pub fn size(&self) -> usize {
        let state = self.state.lock().expect("session store mutex poisoned");
        state
            .store
            .keys()
            .filter(|k| {
                // A main transcript is "project_key/session_id" — exactly one
                // '/' separator (none in the trailing component).
                k.find('/').is_some_and(|i| !k[i + 1..].contains('/'))
            })
            .count()
    }

    /// Test helper: clear all stored data.
    ///
    /// # Panics
    /// If the internal mutex is poisoned by a panic in another thread.
    pub fn clear(&self) {
        let mut state = self.state.lock().expect("session store mutex poisoned");
        state.store.clear();
        state.mtimes.clear();
        state.summaries.clear();
        state.last_mtime = 0;
    }
}

/// Wall-clock epoch milliseconds; strictly monotonic vs `last` so back-to-back
/// appends always produce distinct mtimes (real backends get this from commit
/// ordering).
fn next_mtime(last: &mut i64) -> i64 {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
    let ms = if now_ms <= *last { *last + 1 } else { now_ms };
    *last = ms;
    ms
}

#[async_trait::async_trait]
impl SessionStore for InMemorySessionStore {
    async fn append(
        &self,
        key: &SessionKey,
        entries: Vec<SessionStoreEntry>,
    ) -> Result<(), ClaudeError> {
        let mut state = self.state.lock().expect("session store mutex poisoned");
        let now_ms = next_mtime(&mut state.last_mtime);
        let k = key.storage_key();

        // Maintain the per-session summary sidecar incrementally so
        // list_session_summaries() never re-reads. Subagent subpaths don't
        // contribute to the main session's summary.
        if key.subpath.is_none() {
            let sk = (key.project_key.clone(), key.session_id.clone());
            let prev = state.summaries.get(&sk);
            let mut folded = fold_session_summary(prev, key, &entries);
            // Stamp the sidecar with this adapter's storage write time — the
            // SAME clock list_sessions() exposes below.
            folded.mtime = now_ms;
            state.summaries.insert(sk, folded);
        }

        state.store.entry(k.clone()).or_default().extend(entries);
        state.mtimes.insert(k, now_ms);
        Ok(())
    }

    async fn load(&self, key: &SessionKey) -> Result<Option<Vec<SessionStoreEntry>>, ClaudeError> {
        let state = self.state.lock().expect("session store mutex poisoned");
        Ok(state.store.get(&key.storage_key()).cloned())
    }

    fn capabilities(&self) -> SessionStoreCapabilities {
        SessionStoreCapabilities {
            list_sessions: true,
            list_session_summaries: true,
            delete: true,
            list_subkeys: true,
        }
    }

    async fn list_sessions(
        &self,
        project_key: &str,
    ) -> Result<Vec<SessionStoreListEntry>, ClaudeError> {
        let state = self.state.lock().expect("session store mutex poisoned");
        let prefix = format!("{project_key}/");
        let mut results = Vec::new();
        for k in state.store.keys() {
            if let Some(rest) = k.strip_prefix(&prefix) {
                // Only main transcripts (no subpath → no second '/').
                if !rest.contains('/') {
                    results.push(SessionStoreListEntry {
                        session_id: rest.to_string(),
                        mtime: state.mtimes.get(k).copied().unwrap_or(0),
                    });
                }
            }
        }
        Ok(results)
    }

    async fn list_session_summaries(
        &self,
        project_key: &str,
    ) -> Result<Vec<SessionSummaryEntry>, ClaudeError> {
        let state = self.state.lock().expect("session store mutex poisoned");
        Ok(state
            .summaries
            .iter()
            .filter(|((pk, _), _)| pk == project_key)
            .map(|(_, s)| s.clone())
            .collect())
    }

    async fn delete(&self, key: &SessionKey) -> Result<(), ClaudeError> {
        let mut state = self.state.lock().expect("session store mutex poisoned");
        let k = key.storage_key();
        state.store.remove(&k);
        state.mtimes.remove(&k);

        // Deleting the main transcript cascades to its subkeys (subagent
        // transcripts, metadata). A targeted delete with an explicit subpath
        // removes only that one entry.
        if key.subpath.is_none() {
            state
                .summaries
                .remove(&(key.project_key.clone(), key.session_id.clone()));
            let prefix = format!("{}/{}/", key.project_key, key.session_id);
            let to_remove: Vec<String> = state
                .store
                .keys()
                .filter(|sk| sk.starts_with(&prefix))
                .cloned()
                .collect();
            for sk in to_remove {
                state.store.remove(&sk);
                state.mtimes.remove(&sk);
            }
        }
        Ok(())
    }

    async fn list_subkeys(&self, key: &SessionListSubkeysKey) -> Result<Vec<String>, ClaudeError> {
        let state = self.state.lock().expect("session store mutex poisoned");
        let prefix = key.prefix();
        Ok(state
            .store
            .keys()
            .filter_map(|k| k.strip_prefix(&prefix).map(str::to_string))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry(v: &Value) -> SessionStoreEntry {
        v.as_object().unwrap().clone()
    }

    fn uuid(n: u8) -> Uuid {
        Uuid::from_bytes([n; 16])
    }

    #[test]
    fn flush_mode_serde_wire_spelling() {
        assert_eq!(
            serde_json::to_string(&SessionStoreFlushMode::Batched).unwrap(),
            "\"batched\""
        );
        assert_eq!(
            serde_json::to_string(&SessionStoreFlushMode::Eager).unwrap(),
            "\"eager\""
        );
        assert_eq!(
            SessionStoreFlushMode::default(),
            SessionStoreFlushMode::Batched
        );
    }

    #[tokio::test]
    async fn append_load_roundtrip_deep_equal() {
        let store = InMemorySessionStore::new();
        let key = SessionKey::new("proj", uuid(1));
        let e1 = entry(&json!({"type": "user", "uuid": "a", "n": 1}));
        let e2 = entry(&json!({"type": "assistant", "uuid": "b", "nested": {"x": [1, 2]}}));
        store.append(&key, vec![e1.clone()]).await.unwrap();
        store.append(&key, vec![e2.clone()]).await.unwrap();
        let loaded = store.load(&key).await.unwrap().unwrap();
        assert_eq!(loaded, vec![e1, e2]);
    }

    #[tokio::test]
    async fn load_absent_returns_none() {
        let store = InMemorySessionStore::new();
        let key = SessionKey::new("proj", uuid(9));
        assert!(store.load(&key).await.unwrap().is_none());
    }
}
