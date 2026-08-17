//! Batching layer between `transcript_mirror` stdout frames and a [`SessionStore`].
//!
//! The CLI subprocess emits
//! `{"type": "transcript_mirror", "filePath": ..., "entries": [...]}` frames
//! interleaved with normal SDK messages. The receive loop peels these off and
//! hands them to [`TranscriptMirrorBatcher::enqueue`], which accumulates them
//! and flushes to [`SessionStore::append`] either when a `result` message
//! arrives (explicit [`flush`](TranscriptMirrorBatcher::flush)) or when the
//! pending buffer exceeds size thresholds (eager background flush). This keeps
//! adapter latency off the hot path during model streaming.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::{Mutex as StdMutex, PoisonError};

use serde_json::Value;
use tokio::sync::Mutex;

use super::key::{file_path_to_session_key, SessionKey};
use super::store::{SessionStore, SessionStoreEntry};

/// Eager-flush entry threshold. Exposed for tests.
pub const MAX_PENDING_ENTRIES: usize = 500;
/// Eager-flush byte threshold (1 MiB). Exposed for tests.
pub const MAX_PENDING_BYTES: usize = 1 << 20;
/// Default per-append send timeout in seconds.
pub const SEND_TIMEOUT_SECONDS: f64 = 60.0;

/// Bounded retry for transient adapter failures. Backoff list length must be
/// `MIRROR_APPEND_MAX_ATTEMPTS - 1` (one delay between each pair of attempts).
pub const MIRROR_APPEND_MAX_ATTEMPTS: usize = 3;
/// Backoff between mirror append attempts, in seconds.
pub const MIRROR_APPEND_BACKOFF_S: [f64; 2] = [0.2, 0.8];

/// Callback invoked when a batch is permanently dropped after exhausting
/// retries. The `SessionKey` is `None` only when the frame's file path could
/// not be resolved to a key (never, in practice, since unresolved frames are
/// dropped before flush). Must never panic; errors inside it are swallowed.
pub type MirrorErrorHandler = Arc<
    dyn Fn(Option<SessionKey>, String) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync,
>;

/// One enqueued `transcript_mirror` frame.
#[derive(Debug)]
struct MirrorEntry {
    file_path: String,
    entries: Vec<SessionStoreEntry>,
}

/// Mutable state guarded so `enqueue` (sync, hot path) and `flush`/`close`
/// (async) can share the pending buffer without a lock on the enqueue path.
#[derive(Debug, Default)]
struct Pending {
    items: Vec<MirrorEntry>,
    entries: usize,
    bytes: usize,
}

/// Accumulates `transcript_mirror` frames and flushes them to a store.
///
/// [`enqueue`](Self::enqueue) is fire-and-forget; [`flush`](Self::flush) is
/// async. The pending queue is bounded — when it exceeds `max_pending_entries`
/// or `max_pending_bytes` an eager flush fires in the background so memory
/// stays flat during long turns where no `result` (and thus no explicit
/// `flush()`) arrives.
///
/// Adapter failures are retried ([`MIRROR_APPEND_MAX_ATTEMPTS`] attempts total)
/// with short backoff; timeouts are not retried since the in-flight call may
/// still land. Only after the final attempt fails is the batch dropped and
/// reported via `on_error`. Failures never propagate out of flush — the
/// local-disk transcript is already durable so the session must continue
/// unaffected. Adapters should dedupe by `entry["uuid"]` when present (some
/// entry types lack a uuid) since a retried batch may partially overlap a prior
/// partial write.
pub struct TranscriptMirrorBatcher {
    store: Arc<dyn SessionStore>,
    projects_dir: PathBuf,
    on_error: MirrorErrorHandler,
    send_timeout: f64,
    max_pending_entries: usize,
    max_pending_bytes: usize,

    /// Buffer of enqueued-but-unflushed frames. Guarded by a fast mutex so the
    /// sync `enqueue` path can push without awaiting.
    pending: StdMutex<Pending>,
    /// Serializes flushes so `store.append` calls preserve enqueue order across
    /// concurrent eager and explicit flushes.
    flush_lock: Mutex<()>,
}

impl std::fmt::Debug for TranscriptMirrorBatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscriptMirrorBatcher")
            .field("projects_dir", &self.projects_dir)
            .field("send_timeout", &self.send_timeout)
            .field("max_pending_entries", &self.max_pending_entries)
            .field("max_pending_bytes", &self.max_pending_bytes)
            .finish_non_exhaustive()
    }
}

impl TranscriptMirrorBatcher {
    /// Construct a batcher with the default thresholds.
    ///
    /// Pass `max_pending_entries = 0` and `max_pending_bytes = 0` for eager
    /// mode (every enqueued frame schedules a background flush); use
    /// [`build`](Self::build) or [`super::resume::build_mirror_batcher`] which
    /// derive these from [`crate::sessions::SessionStoreFlushMode`].
    #[must_use]
    pub fn new(
        store: Arc<dyn SessionStore>,
        projects_dir: PathBuf,
        on_error: MirrorErrorHandler,
    ) -> Self {
        Self::build(
            store,
            projects_dir,
            on_error,
            SEND_TIMEOUT_SECONDS,
            MAX_PENDING_ENTRIES,
            MAX_PENDING_BYTES,
        )
    }

    /// Construct a batcher with explicit thresholds and send timeout.
    #[must_use]
    pub fn build(
        store: Arc<dyn SessionStore>,
        projects_dir: PathBuf,
        on_error: MirrorErrorHandler,
        send_timeout: f64,
        max_pending_entries: usize,
        max_pending_bytes: usize,
    ) -> Self {
        Self {
            store,
            projects_dir,
            on_error,
            send_timeout,
            max_pending_entries,
            max_pending_bytes,
            pending: StdMutex::new(Pending::default()),
            flush_lock: Mutex::new(()),
        }
    }

    /// Buffer a frame. Returns `true` when the pending buffer now exceeds a
    /// threshold and the caller should schedule an eager background flush
    /// (`spawn(batcher.flush())`). The batcher does not own a task runtime, so
    /// spawning is the runtime's responsibility; ordering still holds because
    /// [`flush`](Self::flush) serializes on an internal lock.
    ///
    /// The approximate wire size is one `serde_json` stringify per frame (not
    /// per entry), keeping this cheap relative to the parse the transport
    /// already did.
    #[must_use = "schedule an eager flush when this returns true"]
    pub fn enqueue(&self, file_path: impl Into<String>, entries: Vec<SessionStoreEntry>) -> bool {
        // Match the official batcher's `len(json.dumps(entries))`: Python's
        // default separators are `", "`/`": "` (with spaces) and
        // `ensure_ascii=True` escapes every non-ASCII scalar to `\uXXXX`.
        // serde's compact `to_string` uses no spaces and keeps UTF-8, so it
        // undercounts at the eager byte threshold; compute the Python byte
        // length instead.
        let mut size = 0usize;
        json_dumps_len_list(&entries, &mut size);
        let n = entries.len();
        let mut pending = self.pending.lock().unwrap_or_else(PoisonError::into_inner);
        pending.items.push(MirrorEntry {
            file_path: file_path.into(),
            entries,
        });
        pending.entries += n;
        pending.bytes += size;
        pending.entries > self.max_pending_entries || pending.bytes > self.max_pending_bytes
    }

    /// Flush all pending entries, serialized after any in-flight flush. Never
    /// returns an error — adapter failures are retried and, if still failing,
    /// reported via `on_error` and dropped.
    pub async fn flush(&self) {
        self.drain().await;
    }

    /// Final flush before teardown. Never panics, and — crucially —
    /// **cancellation-immune**: the flush runs on a detached Tokio task, so if
    /// the caller awaiting `close()` is cancelled (its future is dropped, e.g.
    /// a client disconnect / Ctrl+C tearing down the run future), the final
    /// batch still reaches the store.
    ///
    /// This mirrors the official Python batcher, whose `close()` wraps the
    /// flush in `anyio.CancelScope(shield=True)` so the final append survives a
    /// cancelled `__aexit__`. In Tokio there is no ambient cancel scope to
    /// shield; the equivalent is to move the work onto a task that owns an
    /// `Arc<Self>` and is not tied to the caller's future. Awaiting the
    /// `JoinHandle` keeps the happy-path ordering guarantee (the flush is
    /// observed complete before `close()` returns) while dropping that await
    /// under cancellation leaves the spawned flush running to completion.
    pub async fn close(self: &Arc<Self>) {
        let this = Arc::clone(self);
        // Detach the flush so it is not cancelled when the caller's future is
        // dropped. flush() never returns an error and never panics.
        let handle = tokio::spawn(async move { this.flush().await });
        // Await completion on the happy path; a cancelled caller drops this
        // await but the detached flush still finishes.
        let _ = handle.await;
    }

    /// Detach the pending buffer, await any prior flush, then send. Detaching
    /// happens before acquiring the flush lock so `enqueue` can keep
    /// accumulating into a fresh buffer while a prior flush is in flight.
    async fn drain(&self) {
        let items = {
            let mut pending = self.pending.lock().unwrap_or_else(PoisonError::into_inner);
            pending.entries = 0;
            pending.bytes = 0;
            std::mem::take(&mut pending.items)
        };

        let mut errors: Vec<(SessionKey, String)> = Vec::new();
        {
            let _guard = self.flush_lock.lock().await;
            if items.is_empty() {
                return;
            }
            self.do_flush(items, &mut errors).await;
        }
        // Report errors after releasing the lock so a slow on_error callback
        // cannot block subsequent drains (which only need the lock for append
        // ordering).
        for (key, msg) in errors {
            (self.on_error)(Some(key), msg).await;
        }
    }

    async fn do_flush(&self, items: Vec<MirrorEntry>, errors: &mut Vec<(SessionKey, String)>) {
        // Coalesce by file_path so each unique file gets one append per flush
        // instead of one per enqueued frame. Preserve first-seen path order and
        // enqueue order within a path.
        let mut order: Vec<String> = Vec::new();
        let mut by_path: std::collections::HashMap<String, Vec<SessionStoreEntry>> =
            std::collections::HashMap::new();
        for item in items {
            if let Some(bucket) = by_path.get_mut(&item.file_path) {
                bucket.extend(item.entries);
            } else {
                order.push(item.file_path.clone());
                by_path.insert(item.file_path, item.entries);
            }
        }

        for file_path in order {
            let entries = by_path.remove(&file_path).unwrap_or_default();
            if entries.is_empty() {
                // Avoid creating phantom keys in adapters that touch storage on
                // append([]) — nothing to write.
                continue;
            }
            let Some(key) = file_path_to_session_key(Path::new(&file_path), &self.projects_dir)
            else {
                tracing::warn!(
                    file_path = %file_path,
                    projects_dir = %self.projects_dir.display(),
                    "dropping mirror frame: filePath is not under projects_dir \
                     (subprocess CLAUDE_CONFIG_DIR likely differs from parent)"
                );
                continue;
            };

            match self.append_with_retry(&key, &file_path, entries).await {
                Ok(()) => {}
                Err(msg) => errors.push((key, msg)),
            }
        }
    }

    /// Append a coalesced batch, retrying transient failures. Timeouts are not
    /// retried (the in-flight call may still land; a retry would launch a
    /// concurrent duplicate and inflate the worst-case lock hold). Returns the
    /// last error string on permanent failure.
    async fn append_with_retry(
        &self,
        key: &SessionKey,
        file_path: &str,
        entries: Vec<SessionStoreEntry>,
    ) -> Result<(), String> {
        let mut last_err: Option<String> = None;
        for attempt in 0..MIRROR_APPEND_MAX_ATTEMPTS {
            if attempt > 0 {
                let backoff = MIRROR_APPEND_BACKOFF_S[attempt - 1];
                tokio::time::sleep(std::time::Duration::from_secs_f64(backoff)).await;
            }
            let timeout = std::time::Duration::from_secs_f64(self.send_timeout);
            match tokio::time::timeout(timeout, self.store.append(key, entries.clone())).await {
                Ok(Ok(())) => return Ok(()),
                Ok(Err(e)) => {
                    let msg = e.to_string();
                    tracing::debug!(
                        attempt = attempt + 1,
                        max = MIRROR_APPEND_MAX_ATTEMPTS,
                        file_path = %file_path,
                        error = %msg,
                        "mirror append attempt failed"
                    );
                    last_err = Some(msg);
                }
                Err(_elapsed) => {
                    // Timeout: do not retry.
                    tracing::debug!(
                        timeout = self.send_timeout,
                        file_path = %file_path,
                        "mirror append timed out — not retrying"
                    );
                    last_err = Some(format!(
                        "SessionStore::append timed out after {:.1}s",
                        self.send_timeout
                    ));
                    break;
                }
            }
        }
        let msg = last_err.unwrap_or_else(|| "unknown mirror append failure".to_string());
        tracing::error!(file_path = %file_path, error = %msg, "mirror flush failed");
        Err(msg)
    }
}

/// Byte length of `json.dumps(list_of_entries)` under Python's defaults
/// (`separators=(", ", ": ")`, `ensure_ascii=True`), accumulated into `acc`.
fn json_dumps_len_list(entries: &[SessionStoreEntry], acc: &mut usize) {
    *acc += 1; // '['
    for (i, e) in entries.iter().enumerate() {
        if i > 0 {
            *acc += 2; // ", "
        }
        json_dumps_len_object(e, acc);
    }
    *acc += 1; // ']'
}

fn json_dumps_len_object(map: &serde_json::Map<String, Value>, acc: &mut usize) {
    *acc += 1; // '{'
    for (i, (k, v)) in map.iter().enumerate() {
        if i > 0 {
            *acc += 2; // ", "
        }
        json_dumps_len_string(k, acc);
        *acc += 2; // ": "
        json_dumps_len_value(v, acc);
    }
    *acc += 1; // '}'
}

fn json_dumps_len_value(value: &Value, acc: &mut usize) {
    match value {
        Value::Null | Value::Bool(true) => *acc += 4, // null / true
        Value::Bool(false) => *acc += 5,              // false
        Value::Number(n) => *acc += n.to_string().len(),
        Value::String(s) => json_dumps_len_string(s, acc),
        Value::Array(items) => json_dumps_len_array(items, acc),
        Value::Object(m) => json_dumps_len_object(m, acc),
    }
}

fn json_dumps_len_array(items: &[Value], acc: &mut usize) {
    *acc += 1; // '['
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            *acc += 2; // ", "
        }
        json_dumps_len_value(item, acc);
    }
    *acc += 1; // ']'
}

/// Length of a JSON string literal as Python's `json.dumps` emits it with
/// `ensure_ascii=True`: surrounding quotes, C-style escapes for control chars
/// and `"`/`\\`, and `\uXXXX` (12 bytes for astral scalars via a surrogate
/// pair) for every non-ASCII scalar.
fn json_dumps_len_string(s: &str, acc: &mut usize) {
    *acc += 2; // surrounding quotes
    for ch in s.chars() {
        *acc += match ch {
            '"' | '\\' | '\n' | '\r' | '\t' | '\u{08}' | '\u{0c}' => 2,
            c if (c as u32) < 0x20 => 6, // \u00XX
            c if c.is_ascii() => 1,
            c if (c as u32) <= 0xFFFF => 6, // \uXXXX
            _ => 12,                        // surrogate pair \uXXXX\uXXXX
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ClaudeError;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use uuid::Uuid;

    /// Recording store that logs every append call in order and can be
    /// configured to fail or hang on a bounded number of attempts.
    #[derive(Debug, Default)]
    struct RecordingStore {
        appends: StdMutex<Vec<(SessionKey, Vec<SessionStoreEntry>)>>,
        /// Number of leading append attempts that should fail with an error.
        fail_first: AtomicUsize,
        /// Number of leading append attempts that should hang past the timeout.
        hang_first: AtomicUsize,
        attempts: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl SessionStore for RecordingStore {
        async fn append(
            &self,
            key: &SessionKey,
            entries: Vec<SessionStoreEntry>,
        ) -> Result<(), ClaudeError> {
            let n = self.attempts.fetch_add(1, Ordering::SeqCst);
            if n < self.hang_first.load(Ordering::SeqCst) {
                // Hang well past the batcher's (test-shortened) timeout.
                tokio::time::sleep(std::time::Duration::from_hours(1)).await;
            }
            if n < self.fail_first.load(Ordering::SeqCst) {
                return Err(ClaudeError::TransportError("boom".into()));
            }
            self.appends.lock().unwrap().push((key.clone(), entries));
            Ok(())
        }
        async fn load(
            &self,
            _key: &SessionKey,
        ) -> Result<Option<Vec<SessionStoreEntry>>, ClaudeError> {
            Ok(None)
        }
    }

    fn entry(uuid: &str) -> SessionStoreEntry {
        let mut m = SessionStoreEntry::new();
        m.insert("type".into(), json!("user"));
        m.insert("uuid".into(), json!(uuid));
        m
    }

    fn entry_from(v: serde_json::Value) -> SessionStoreEntry {
        let serde_json::Value::Object(m) = v else {
            panic!("entry must be an object");
        };
        m
    }

    /// The eager byte threshold must be computed against Python's
    /// `json.dumps(entries)` default output — `separators=(", ", ": ")` with
    /// spaces and `ensure_ascii=True` escaping every non-ASCII scalar to
    /// `\uXXXX` (a 12-byte surrogate pair for astral scalars). A compact
    /// UTF-8 serde stringify would undercount and let the buffer grow past the
    /// intended threshold. These expected lengths are `len(json.dumps(...))`
    /// captured from `CPython`.
    #[test]
    fn json_dumps_len_matches_python() {
        let check = |v: serde_json::Value, expected: usize| {
            let entries = vec![entry_from(v)];
            let mut size = 0usize;
            json_dumps_len_list(&entries, &mut size);
            assert_eq!(size, expected, "json.dumps byte length parity");
        };
        // Pure ASCII.
        check(json!({"type":"x","uuid":"abc"}), 30);
        // BMP non-ASCII (`é`, `☕`) → 6 bytes each via `\uXXXX` + spaced
        // separators.
        check(json!({"type":"x","msg":"café ☕"}), 42);
        // Astral scalar (`😀`, U+1F600) → 12 bytes via a `\uXXXX\uXXXX`
        // surrogate pair. This is the Unicode threshold that a naive UTF-8
        // byte count (4 bytes) would badly undercount.
        check(json!({"type":"x","msg":"emoji 😀 test"}), 49);
        // C-style escapes: `"` `\` `\n` `\t` → 2 bytes each.
        check(json!({"type":"x","msg":"a\"b\\c\nd\te"}), 39);
        // Other control chars → `\u00XX` (6 bytes); `\b`/`\f` → 2 bytes.
        check(json!({"type":"x","msg":"\u{0001}\u{0008}\u{000c}"}), 36);
    }

    /// Collected `(key, message)` pairs from a recording `on_error` handler.
    type ErrorSink = Arc<StdMutex<Vec<(Option<SessionKey>, String)>>>;

    /// `projects_dir` under which our synthetic file paths resolve.
    fn projects_dir() -> PathBuf {
        PathBuf::from("/tmp/projects")
    }

    fn main_file(project_key: &str, session_id: Uuid) -> String {
        format!("/tmp/projects/{project_key}/{session_id}.jsonl")
    }

    fn noop_on_error() -> MirrorErrorHandler {
        Arc::new(|_key, _msg| Box::pin(async {}))
    }

    fn recording_on_error(sink: ErrorSink) -> MirrorErrorHandler {
        Arc::new(move |key, msg| {
            let sink = sink.clone();
            Box::pin(async move {
                sink.lock().unwrap().push((key, msg));
            })
        })
    }

    #[tokio::test]
    async fn flush_coalesces_entries_per_path_in_order() {
        let store = Arc::new(RecordingStore::default());
        let sid = Uuid::new_v4();
        let path = main_file("proj", sid);
        let batcher = TranscriptMirrorBatcher::new(store.clone(), projects_dir(), noop_on_error());

        assert!(!batcher.enqueue(path.clone(), vec![entry("a")]));
        assert!(!batcher.enqueue(path.clone(), vec![entry("b"), entry("c")]));
        batcher.flush().await;

        let appends = store.appends.lock().unwrap();
        assert_eq!(appends.len(), 1, "one append per unique path");
        let (key, entries) = &appends[0];
        assert_eq!(key.project_key, "proj");
        assert_eq!(key.session_id, sid.to_string());
        assert_eq!(key.subpath, None);
        let uuids: Vec<&str> = entries
            .iter()
            .map(|e| e["uuid"].as_str().unwrap())
            .collect();
        assert_eq!(uuids, vec!["a", "b", "c"], "enqueue order preserved");
    }

    #[tokio::test]
    async fn distinct_paths_get_distinct_appends_in_first_seen_order() {
        let store = Arc::new(RecordingStore::default());
        let s1 = Uuid::new_v4();
        let s2 = Uuid::new_v4();
        let p1 = main_file("proj", s1);
        let p2 = main_file("proj", s2);
        let batcher = TranscriptMirrorBatcher::new(store.clone(), projects_dir(), noop_on_error());

        let _ = batcher.enqueue(p2.clone(), vec![entry("x")]);
        let _ = batcher.enqueue(p1.clone(), vec![entry("y")]);
        let _ = batcher.enqueue(p2.clone(), vec![entry("z")]);
        batcher.flush().await;

        let appends = store.appends.lock().unwrap();
        assert_eq!(appends.len(), 2);
        // p2 seen first.
        assert_eq!(appends[0].0.session_id, s2.to_string());
        assert_eq!(appends[1].0.session_id, s1.to_string());
        let p2_uuids: Vec<&str> = appends[0]
            .1
            .iter()
            .map(|e| e["uuid"].as_str().unwrap())
            .collect();
        assert_eq!(p2_uuids, vec!["x", "z"]);
    }

    #[tokio::test]
    async fn entry_threshold_signals_eager_flush() {
        let store = Arc::new(RecordingStore::default());
        let sid = Uuid::new_v4();
        let path = main_file("proj", sid);
        // Eager: thresholds zero → every non-empty enqueue signals.
        let batcher = TranscriptMirrorBatcher::build(
            store.clone(),
            projects_dir(),
            noop_on_error(),
            SEND_TIMEOUT_SECONDS,
            0,
            0,
        );
        assert!(
            batcher.enqueue(path, vec![entry("a")]),
            "eager signals immediately"
        );
    }

    #[tokio::test]
    async fn batched_threshold_only_signals_after_overflow() {
        let store = Arc::new(RecordingStore::default());
        let sid = Uuid::new_v4();
        let path = main_file("proj", sid);
        let batcher = TranscriptMirrorBatcher::build(
            store.clone(),
            projects_dir(),
            noop_on_error(),
            SEND_TIMEOUT_SECONDS,
            3, // entry threshold
            MAX_PENDING_BYTES,
        );
        assert!(!batcher.enqueue(path.clone(), vec![entry("a"), entry("b")]));
        // 2 <= 3, no signal; add 2 more → 4 > 3 → signal.
        assert!(batcher.enqueue(path, vec![entry("c"), entry("d")]));
    }

    #[tokio::test]
    async fn empty_batch_does_not_append() {
        let store = Arc::new(RecordingStore::default());
        let sid = Uuid::new_v4();
        let path = main_file("proj", sid);
        let batcher = TranscriptMirrorBatcher::new(store.clone(), projects_dir(), noop_on_error());
        let _ = batcher.enqueue(path, Vec::new());
        batcher.flush().await;
        assert!(
            store.appends.lock().unwrap().is_empty(),
            "no phantom append"
        );
    }

    #[tokio::test]
    async fn unresolvable_path_is_dropped_without_error() {
        let sink = Arc::new(StdMutex::new(Vec::new()));
        let store = Arc::new(RecordingStore::default());
        let batcher = TranscriptMirrorBatcher::new(
            store.clone(),
            projects_dir(),
            recording_on_error(sink.clone()),
        );
        // Not under projects_dir → unresolvable → dropped, not reported.
        let _ = batcher.enqueue("/somewhere/else/foo.jsonl", vec![entry("a")]);
        batcher.flush().await;
        assert!(store.appends.lock().unwrap().is_empty());
        assert!(
            sink.lock().unwrap().is_empty(),
            "dropped frames are not on_error"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn transient_failure_retries_then_succeeds() {
        let store = Arc::new(RecordingStore::default());
        store.fail_first.store(2, Ordering::SeqCst); // fail twice, succeed on 3rd
        let sid = Uuid::new_v4();
        let path = main_file("proj", sid);
        let batcher = TranscriptMirrorBatcher::new(store.clone(), projects_dir(), noop_on_error());
        let _ = batcher.enqueue(path, vec![entry("a")]);
        // start_paused auto-advances virtual time through the backoff sleeps
        // whenever the runtime is otherwise idle.
        batcher.flush().await;
        assert_eq!(store.attempts.load(Ordering::SeqCst), 3);
        assert_eq!(store.appends.lock().unwrap().len(), 1, "eventual success");
    }

    #[tokio::test(start_paused = true)]
    async fn permanent_failure_reports_on_error_after_max_attempts() {
        let sink = Arc::new(StdMutex::new(Vec::new()));
        let store = Arc::new(RecordingStore::default());
        store
            .fail_first
            .store(MIRROR_APPEND_MAX_ATTEMPTS, Ordering::SeqCst);
        let sid = Uuid::new_v4();
        let path = main_file("proj", sid);
        let batcher = TranscriptMirrorBatcher::new(
            store.clone(),
            projects_dir(),
            recording_on_error(sink.clone()),
        );
        let _ = batcher.enqueue(path, vec![entry("a")]);
        batcher.flush().await;

        assert_eq!(
            store.attempts.load(Ordering::SeqCst),
            MIRROR_APPEND_MAX_ATTEMPTS
        );
        assert!(store.appends.lock().unwrap().is_empty());
        let errs = sink.lock().unwrap();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].0.is_some(), "key attached to error");
        assert!(errs[0].1.contains("boom"));
    }

    #[tokio::test(start_paused = true)]
    async fn timeout_is_not_retried() {
        let sink = Arc::new(StdMutex::new(Vec::new()));
        let store = Arc::new(RecordingStore::default());
        store.hang_first.store(1, Ordering::SeqCst); // first attempt hangs
        let sid = Uuid::new_v4();
        let path = main_file("proj", sid);
        // Short send_timeout so the hang trips it quickly under virtual time;
        // the hang's 3600s sleep is a later timer, so auto-advance fires the
        // 0.5s timeout first.
        let batcher = TranscriptMirrorBatcher::build(
            store.clone(),
            projects_dir(),
            recording_on_error(sink.clone()),
            0.5,
            MAX_PENDING_ENTRIES,
            MAX_PENDING_BYTES,
        );
        let _ = batcher.enqueue(path, vec![entry("a")]);
        batcher.flush().await;

        // Exactly one attempt — timeout does not retry.
        assert_eq!(store.attempts.load(Ordering::SeqCst), 1);
        let errs = sink.lock().unwrap();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].1.contains("timed out"));
    }

    #[tokio::test]
    async fn final_flush_via_close_writes_pending() {
        let store = Arc::new(RecordingStore::default());
        let sid = Uuid::new_v4();
        let path = main_file("proj", sid);
        let batcher = Arc::new(TranscriptMirrorBatcher::new(
            store.clone(),
            projects_dir(),
            noop_on_error(),
        ));
        let _ = batcher.enqueue(path, vec![entry("final")]);
        batcher.close().await;
        assert_eq!(store.appends.lock().unwrap().len(), 1);
    }

    /// `close()` is cancellation-immune: even when the caller awaiting it is
    /// cancelled (its future dropped mid-flush, as during a client disconnect /
    /// Ctrl+C teardown), the final pending batch still reaches the store. This
    /// is the Tokio analogue of the official Python batcher's
    /// `close()` -> `anyio.CancelScope(shield=True)` around the final flush.
    #[tokio::test(start_paused = true)]
    async fn close_is_cancellation_immune() {
        let store = Arc::new(RecordingStore::default());
        // First append attempt fails transiently, forcing a retry after a
        // backoff sleep. That sleep is the window during which we cancel the
        // caller's `close()` await — the detached flush task must survive the
        // cancellation, wait out the backoff, retry, and land the append.
        store.fail_first.store(1, Ordering::SeqCst);
        let sid = Uuid::new_v4();
        let path = main_file("proj", sid);
        let batcher = Arc::new(TranscriptMirrorBatcher::new(
            store.clone(),
            projects_dir(),
            noop_on_error(),
        ));
        let _ = batcher.enqueue(path, vec![entry("final")]);

        // Drive the close future far enough to spawn the detached flush task
        // and run its first (failing) append, which then parks on the backoff
        // sleep. Then drop the future to simulate the awaiting caller being
        // cancelled during teardown.
        {
            let close_fut = batcher.close();
            tokio::pin!(close_fut);
            // `yield_now` inside the timeout lets the spawned task make
            // progress; poll close a few times so the first append runs and the
            // task reaches its backoff sleep before we cancel.
            for _ in 0..5 {
                let _ =
                    tokio::time::timeout(std::time::Duration::from_millis(1), &mut close_fut).await;
                tokio::task::yield_now().await;
            }
        }

        // The awaiting caller is now gone. Advance virtual time past the retry
        // backoff so the detached flush task retries and succeeds, yielding
        // between advances so the task is scheduled.
        for _ in 0..10 {
            tokio::time::advance(std::time::Duration::from_secs(1)).await;
            tokio::task::yield_now().await;
        }

        assert_eq!(
            store.attempts.load(Ordering::SeqCst),
            2,
            "detached flush retried after the transient failure"
        );
        assert_eq!(
            store.appends.lock().unwrap().len(),
            1,
            "final batch reached the store despite the caller being cancelled"
        );
    }
}
