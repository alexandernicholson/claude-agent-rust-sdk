//! Persistent reader/router runtime for the Claude Code Agent SDK client.
//!
//! This module owns the concurrency-safe machinery behind
//! [`crate::agent::ClaudeAgentClient`]: a detached reader task that routes the
//! bidirectional control protocol, a pending-control-response waiter map,
//! cancellable inbound control handlers, a task ledger that decides when stdin
//! may be closed, and contextual process-error attribution.
//!
//! The design mirrors the official Python `Query` state machine
//! (`_internal/query.py`). The transport is held behind an `Arc` so the reader
//! task, control senders, and consumers can all touch it concurrently; every
//! transport method already takes `&self`.

use crate::agent::{
    AgentMessage, AgentOptions, AgentTransport, SdkMcpServer, SdkMcpServerDescriptor,
};
use crate::error::ClaudeError;
use crate::extensions::{
    CanUseTool, HookCallback, HookContext, HookEvent, HookInput, HookMatcher, PermissionResult,
    SkillSelection, SystemPrompt, ToolPermissionContext,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task::JoinHandle;
use tokio::time::Instant;

/// Task types whose completion runs a follow-up turn, and which therefore may
/// still need the control channel after the turn's result frame. Mirrors
/// `DEFERRING_TASK_TYPES` in the Python SDK.
const DEFERRING_TASK_TYPES: [&str; 2] = ["local_agent", "local_workflow"];

/// Terminal task statuses that clear an in-flight task from the ledger.
const TERMINAL_TASK_STATUSES: [&str; 4] = ["completed", "failed", "stopped", "killed"];

/// Bounded message-buffer capacity, matching the Python memory stream (100).
const MESSAGE_BUFFER: usize = 100;

/// One item delivered to a consumer of the regular-message stream.
enum StreamItem {
    /// A regular (non-control) Agent SDK frame, as raw JSON.
    Message { frame: Value },
    /// The run is over: stdin no longer needs to stay open for control
    /// requests. Follows the frame that ended the run.
    RunEnded,
    /// The reader observed a fatal error; consumers fail with it.
    Error(ClaudeError),
}

/// One event from [`Runtime::receive_event`].
// Transient value moved straight to the caller; boxing every message would
// allocate on the hot path to shrink a short-lived enum.
#[allow(clippy::large_enum_variant)]
pub(crate) enum RuntimeEvent {
    /// A regular Agent SDK message.
    Message(AgentMessage),
    /// The run ended (see [`RunTracker`]).
    RunEnded,
}

/// Shared control-protocol state touched by the reader task and control senders.
struct ControlState {
    /// Pending outbound control requests keyed by request id.
    pending: Mutex<HashMap<String, oneshot::Sender<Result<Value, ClaudeError>>>>,
    /// Cancellation handles for inbound control-request handler tasks.
    inflight: Mutex<HashMap<String, JoinHandle<()>>>,
}

impl ControlState {
    fn new() -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
            inflight: Mutex::new(HashMap::new()),
        }
    }
}

/// Callbacks and servers the reader needs to answer inbound control requests.
///
/// Cloned into each spawned handler task; the fields are all cheap `Arc`/map
/// clones.
struct ControlContext<T: AgentTransport> {
    transport: Arc<T>,
    mcp_servers: Arc<HashMap<String, SdkMcpServer>>,
    can_use_tool: Option<CanUseTool>,
    hook_callbacks: Arc<HashMap<String, HookCallback>>,
    control: Arc<ControlState>,
}

// Manual `Clone`: the derived impl would demand `T: Clone`, but every field is
// a cheap `Arc`/`Option<Arc>` so cloning never touches `T` itself.
impl<T: AgentTransport> Clone for ControlContext<T> {
    fn clone(&self) -> Self {
        Self {
            transport: Arc::clone(&self.transport),
            mcp_servers: Arc::clone(&self.mcp_servers),
            can_use_tool: self.can_use_tool.clone(),
            hook_callbacks: Arc::clone(&self.hook_callbacks),
            control: Arc::clone(&self.control),
        }
    }
}

/// A live Agent SDK session: the reader task plus everything needed to send
/// control requests and receive regular messages.
pub(crate) struct Runtime<T: AgentTransport> {
    transport: Arc<T>,
    control: Arc<ControlState>,
    /// Receiver for regular frames; behind a `Mutex` so `receive` is `&self`.
    receiver: Mutex<mpsc::Receiver<StreamItem>>,
    reader: Mutex<Option<JoinHandle<()>>>,
    request_counter: AtomicU64,
    /// Whether SDK MCP servers, hooks, or a permission callback require
    /// holding stdin open until the run ends (Python
    /// `wait_for_result_and_end_input`).
    defer_end_input: bool,
    initialize_timeout: Duration,
    closed: Arc<std::sync::atomic::AtomicBool>,
    initialization_result: Value,
    /// `SessionStore` mirror batcher, when a session store is configured.
    batcher: Option<Arc<crate::sessions::TranscriptMirrorBatcher>>,
    /// Materialized resume temp state; its config dir is cleaned up after the
    /// transport closes (never before, or the child could read a deleted dir).
    materialized: Mutex<Option<crate::sessions::MaterializedResume>>,
}

impl<T: AgentTransport> fmt::Debug for Runtime<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Runtime")
            .field("defer_end_input", &self.defer_end_input)
            .field("closed", &self.closed.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

impl<T: AgentTransport> Runtime<T> {
    /// Start the reader, perform the initialize handshake, and (for a custom
    /// session id or default) leave input handling to the caller.
    ///
    /// On any handshake failure the transport is closed and the error is
    /// returned so the caller can roll back to a disconnected state.
    pub(crate) async fn connect(
        transport: Arc<T>,
        options: &AgentOptions,
        materialized: Option<crate::sessions::MaterializedResume>,
        mcp_servers: Arc<HashMap<String, SdkMcpServer>>,
        descriptors: &[SdkMcpServerDescriptor],
    ) -> Result<Self, ClaudeError> {
        transport.connect(options, descriptors).await?;

        // Register hook callbacks and build the initialize hooks payload, then
        // assemble the request. A build failure (e.g. an unserializable agent
        // definition) rolls back the transport before the reader ever starts.
        let (hooks_config, hook_callbacks) = build_hooks(options.hooks.as_ref());
        let request = match build_initialize_request(options, hooks_config) {
            Ok(request) => request,
            Err(error) => {
                let _ = transport.close().await;
                if let Some(mat) = materialized {
                    mat.cleanup().await;
                }
                return Err(error);
            }
        };

        let control = Arc::new(ControlState::new());
        let (sender, receiver) = mpsc::channel::<StreamItem>(MESSAGE_BUFFER);
        let closed = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Build the SessionStore mirror batcher when a store is configured. Its
        // `on_error` synthesizes a `system/mirror_error` frame onto the message
        // stream so consumers observe append failures (at-most-once).
        let batcher = build_connect_batcher(options, &sender, materialized.as_ref());

        let ctx = ControlContext {
            transport: Arc::clone(&transport),
            mcp_servers,
            can_use_tool: options.can_use_tool.clone(),
            hook_callbacks: Arc::new(hook_callbacks),
            control: Arc::clone(&control),
        };

        // Whether control requests (SDK MCP servers, hooks, a permission
        // callback) may still need stdin until the run ends.
        let defer_end_input = !ctx.mcp_servers.is_empty()
            || options.hooks.as_ref().is_some_and(|h| !h.is_empty())
            || ctx.can_use_tool.is_some();
        let run_state = Arc::new(RunState::new(RunTracker::new(
            defer_end_input,
            run_end_ceiling_ms(&options.env),
        )));

        let reader = tokio::spawn(read_messages(
            ctx.clone(),
            sender,
            run_state,
            Arc::clone(&closed),
            batcher.clone(),
        ));

        let runtime = Self {
            transport,
            control,
            receiver: Mutex::new(receiver),
            reader: Mutex::new(Some(reader)),
            request_counter: AtomicU64::new(0),
            defer_end_input,
            initialize_timeout: options.initialize_timeout,
            closed,
            initialization_result: Value::Null,
            batcher,
            materialized: Mutex::new(materialized),
        };

        // Send the initialize control request and await its response.
        match runtime
            .send_control_request(request, runtime.initialize_timeout, "initialize")
            .await
        {
            Ok(response) => {
                let mut this = runtime;
                this.initialization_result = response;
                Ok(this)
            }
            Err(error) => {
                // Roll back: tear down the reader and transport, preserve error.
                runtime.close().await;
                Err(error)
            }
        }
    }

    /// Stored `initialize` response (server info / available commands).
    pub(crate) fn server_info(&self) -> &Value {
        &self.initialization_result
    }

    /// Write one raw JSON frame, serializing and appending the NDJSON newline.
    pub(crate) async fn write_frame(&self, frame: &Value) -> Result<(), ClaudeError> {
        write_frame(&self.transport, frame).await
    }

    /// Whether the reader is still alive and the transport is ready.
    pub(crate) fn is_ready(&self) -> bool {
        !self.closed.load(Ordering::SeqCst) && self.transport.is_ready()
    }

    /// Synchronously abort the reader task without awaiting. Used by `Drop`
    /// cleanup where no async context is available; the transport's own drop
    /// releases process resources.
    pub(crate) fn abort_reader(&self) {
        self.closed.store(true, Ordering::SeqCst);
        if let Ok(mut guard) = self.reader.try_lock() {
            if let Some(handle) = guard.take() {
                handle.abort();
            }
        }
    }

    /// Receive the next regular Agent SDK message, or `None` at end of stream.
    ///
    /// Unknown top-level frames and run-end markers are skipped; malformed
    /// known frames surface as [`ClaudeError::MessageParse`].
    pub(crate) async fn receive(&self) -> Result<Option<AgentMessage>, ClaudeError> {
        loop {
            match self.receive_event().await? {
                None => return Ok(None),
                Some(RuntimeEvent::Message(message)) => return Ok(Some(message)),
                Some(RuntimeEvent::RunEnded) => {}
            }
        }
    }

    /// Like [`receive`](Self::receive) but also reports run-end markers: the
    /// point where the run is over and stdin no longer needs to stay open for
    /// control requests. The marker follows the frame that ended the run (the
    /// `result` itself, or the `idle` state frame that followed it, which is
    /// not surfaced when it is `sdk_host_only`). Used by the aggregate `query`
    /// to drain the correct run boundary without a racy shared latch.
    pub(crate) async fn receive_event(&self) -> Result<Option<RuntimeEvent>, ClaudeError> {
        let mut receiver = self.receiver.lock().await;
        loop {
            match receiver.recv().await {
                None => return Ok(None),
                Some(StreamItem::Error(error)) => return Err(error),
                Some(StreamItem::RunEnded) => return Ok(Some(RuntimeEvent::RunEnded)),
                Some(StreamItem::Message { frame }) => {
                    if let Some(message) = AgentMessage::from_value(frame)? {
                        return Ok(Some(RuntimeEvent::Message(message)));
                    }
                }
            }
        }
    }

    /// Send a control request and await its response, with a timeout that
    /// cleans up the pending waiter and reports [`ClaudeError::ControlTimeout`].
    pub(crate) async fn send_control_request(
        &self,
        request: Value,
        timeout: Duration,
        subtype: &str,
    ) -> Result<Value, ClaudeError> {
        let request_id = self.next_request_id();
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.control.pending.lock().await;
            pending.insert(request_id.clone(), tx);
        }

        let frame = json!({
            "type": "control_request",
            "request_id": request_id,
            "request": request,
        });
        if let Err(error) = self.write_frame(&frame).await {
            self.control.pending.lock().await.remove(&request_id);
            return Err(error);
        }

        if let Ok(Ok(result)) = tokio::time::timeout(timeout, rx).await {
            result
        } else {
            // Sender dropped without sending (reader ended) or the timeout
            // elapsed: both are connection failures callers should fail fast on.
            self.control.pending.lock().await.remove(&request_id);
            Err(ClaudeError::ControlTimeout {
                subtype: subtype.to_owned(),
            })
        }
    }

    /// Default 60s control-request timeout used by dynamic control operations.
    pub(crate) async fn control(
        &self,
        request: Value,
        subtype: &str,
    ) -> Result<Value, ClaudeError> {
        self.send_control_request(request, Duration::from_mins(1), subtype)
            .await
    }

    /// Whether SDK MCP servers or hooks require deferring stdin closure until a
    /// run-ending result.
    pub(crate) fn defers_end_input(&self) -> bool {
        self.defer_end_input
    }

    fn next_request_id(&self) -> String {
        let counter = self.request_counter.fetch_add(1, Ordering::SeqCst) + 1;
        // Opaque unique id: counter plus random suffix (matches Python shape).
        format!("req_{counter}_{}", uuid::Uuid::new_v4().simple())
    }

    /// Cancellation-safe, idempotent teardown: cancel the reader and inbound
    /// handlers, then close the transport. Always leaves the runtime closed.
    pub(crate) async fn close(&self) {
        if self.closed.swap(true, Ordering::SeqCst) {
            // Already closed; still ensure the transport close ran at least once.
            let _ = self.transport.close().await;
            return;
        }
        // Cancel in-flight inbound control handlers.
        {
            let mut inflight = self.control.inflight.lock().await;
            for (_, handle) in inflight.drain() {
                handle.abort();
            }
        }
        // Fail any pending outbound control waiters.
        {
            let mut pending = self.control.pending.lock().await;
            for (subtype, tx) in pending.drain() {
                let _ = tx.send(Err(ClaudeError::CliConnection(format!(
                    "Agent SDK connection closed while awaiting control response: {subtype}"
                ))));
            }
        }
        // Cancel the reader task.
        if let Some(handle) = self.reader.lock().await.take() {
            handle.abort();
        }
        // Final mirror flush before teardown: aborting the reader above may
        // skip its own finally-flush, so flush here too (idempotent).
        if let Some(batcher) = &self.batcher {
            batcher.close().await;
        }
        // Close the input half, then the transport, THEN clean up the
        // materialized resume temp config dir — never before, or the child
        // could read a deleted dir. Each step is best-effort so a failure in
        // one never skips the others.
        let _ = self.transport.end_input().await;
        let _ = self.transport.close().await;
        if let Some(mat) = self.materialized.lock().await.take() {
            mat.cleanup().await;
        }
    }
}

/// Serialize a JSON frame to an NDJSON line and write it via the transport.
async fn write_frame<T: AgentTransport>(
    transport: &Arc<T>,
    frame: &Value,
) -> Result<(), ClaudeError> {
    let mut line = serde_json::to_string(frame)?;
    line.push('\n');
    transport.write(&line).await
}

/// Build the `SessionStore` mirror batcher for [`AgentRuntime::connect`] when a
/// store is configured, returning `None` otherwise. The `on_error` handler
/// synthesizes a `system/mirror_error` frame onto the message stream so
/// consumers observe append failures (at-most-once), matching Python's
/// `report_mirror_error` which always includes `key`.
fn build_connect_batcher(
    options: &AgentOptions,
    sender: &mpsc::Sender<StreamItem>,
    materialized: Option<&crate::sessions::MaterializedResume>,
) -> Option<Arc<crate::sessions::TranscriptMirrorBatcher>> {
    options.session_store.as_ref().map(|store| {
        let error_sender = sender.clone();
        let on_error: crate::sessions::MirrorErrorHandler = Arc::new(move |key, message| {
            let error_sender = error_sender.clone();
            Box::pin(async move {
                // Serialize the failing SessionKey into the frame so consumers
                // can attribute the error.
                let (key_value, session_id) = match key.as_ref() {
                    Some(k) => {
                        let mut map = serde_json::Map::new();
                        map.insert("project_key".into(), json!(k.project_key));
                        map.insert("session_id".into(), json!(k.session_id));
                        map.insert("subpath".into(), json!(k.subpath));
                        (Value::Object(map), k.session_id.clone())
                    }
                    None => (Value::Null, String::new()),
                };
                let frame = json!({
                    "type": "system",
                    "subtype": "mirror_error",
                    "error": message,
                    "key": key_value,
                    "session_id": session_id,
                    "uuid": uuid::Uuid::new_v4().to_string(),
                });
                // Non-blocking: drop on a full buffer rather than stalling the
                // batcher.
                let _ = error_sender.try_send(StreamItem::Message { frame });
            })
        });
        Arc::new(crate::sessions::build_mirror_batcher(
            Arc::clone(store),
            materialized,
            Some(&options.env),
            on_error,
            options.session_store_flush,
        ))
    })
}

/// Peel a transcript mirror frame off stdout and enqueue its entries. Official
/// Python indexes `filePath` and `entries` directly, so malformed frames remain
/// visible errors rather than silently losing transcript writes.
fn enqueue_mirror_frame(
    frame: &Value,
    batcher: Option<&Arc<crate::sessions::TranscriptMirrorBatcher>>,
) -> Result<(), ClaudeError> {
    let Some(batcher) = batcher else {
        return Ok(());
    };
    let (file_path, entries) = parse_mirror_frame(frame)?;
    if batcher.enqueue(file_path, entries) {
        let batcher = Arc::clone(batcher);
        tokio::spawn(async move { batcher.flush().await });
    }
    Ok(())
}

/// The detached reader loop: read frames, route control traffic, track the run
/// boundary and error attribution, and forward regular frames to the stream.
async fn read_messages<T: AgentTransport>(
    ctx: ControlContext<T>,
    sender: mpsc::Sender<StreamItem>,
    run_state: Arc<RunState>,
    closed: Arc<std::sync::atomic::AtomicBool>,
    batcher: Option<Arc<crate::sessions::TranscriptMirrorBatcher>>,
) {
    let mut last_error_result: Option<Value> = None;
    let timer = run_state.has_ceiling_timer().then(|| {
        AbortOnDrop(tokio::spawn(run_end_timer(
            Arc::clone(&run_state),
            sender.clone(),
        )))
    });

    loop {
        if closed.load(Ordering::SeqCst) {
            break;
        }
        let frame = match ctx.transport.read().await {
            Ok(Some(frame)) => frame,
            Ok(None) => break, // clean EOF: end sentinel below
            Err(error) => {
                fail_read_error(
                    error,
                    last_error_result.as_ref(),
                    &ctx.control,
                    batcher.as_ref(),
                    &sender,
                )
                .await;
                return;
            }
        };

        let msg_type = frame
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();

        match msg_type {
            "control_response" => {
                route_control_response(&ctx.control, &frame).await;
                continue;
            }
            "control_request" => {
                spawn_control_handler(ctx.clone(), frame).await;
                continue;
            }
            "control_cancel_request" => {
                if let Some(cancel_id) = frame.get("request_id").and_then(Value::as_str) {
                    if let Some(handle) = ctx.control.inflight.lock().await.remove(cancel_id) {
                        handle.abort();
                    }
                }
                continue;
            }
            "transcript_mirror" => {
                if let Err(error) = enqueue_mirror_frame(&frame, batcher.as_ref()) {
                    fail_read_error(
                        error,
                        last_error_result.as_ref(),
                        &ctx.control,
                        batcher.as_ref(),
                        &sender,
                    )
                    .await;
                    return;
                }
                continue;
            }
            _ => {}
        }

        let mut run_ended = false;
        let mut host_only = false;
        if msg_type == "system" {
            run_ended = run_state.update(|tracker| tracker.on_system_frame(&frame));
            // State frames the CLI sent only because the transport asked for
            // them (`CLAUDE_CODE_SDK_READS_SESSION_STATE`); the caller did not
            // opt in, so they drive the run boundary but are not surfaced.
            host_only = is_session_state_frame(&frame, msg_type)
                && frame.get("sdk_host_only").and_then(Value::as_bool) == Some(true);
        } else if is_main_thread_turn_frame(&frame, msg_type) {
            run_state.update(|tracker| {
                tracker.on_turn_activity();
                false
            });
        }

        track_error_result(&frame, msg_type, batcher.as_ref(), &mut last_error_result).await;
        if msg_type == "result" {
            run_ended = run_state.update(RunTracker::on_result);
        }

        if !host_only && sender.send(StreamItem::Message { frame }).await.is_err() {
            // Consumer dropped the receiver: nothing left to deliver.
            break;
        }
        if run_ended && sender.send(StreamItem::RunEnded).await.is_err() {
            break;
        }
    }

    // Clean end of stream: final mirror flush, then drop the timer and sender
    // (channel close signals end to consumers as `None`).
    if let Some(batcher) = &batcher {
        batcher.close().await;
    }
    drop(timer);
    drop(sender);
}

/// Extract `(filePath, entries)` from a `transcript_mirror` frame, mirroring
/// the official Python read loop's direct `message["filePath"]` /
/// `message["entries"]` access.
///
/// A frame missing either key, carrying a non-string `filePath`, a non-array
/// `entries`, or entries that are not JSON objects is malformed: rather than
/// silently dropping it (which would lose a turn's transcript writes without a
/// trace), this returns a [`ClaudeError::MessageParse`] so the read loop can
/// surface the failure visibly, exactly as Python's `KeyError`/`TypeError`
/// would propagate out of the receive loop.
fn parse_mirror_frame(
    frame: &Value,
) -> Result<(String, Vec<serde_json::Map<String, Value>>), ClaudeError> {
    let file_path = frame
        .get("filePath")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ClaudeError::message_parse(
                "transcript_mirror frame missing string filePath",
                Some(frame.clone()),
            )
        })?
        .to_owned();
    let raw_entries = frame.get("entries").ok_or_else(|| {
        ClaudeError::message_parse(
            "transcript_mirror frame missing entries",
            Some(frame.clone()),
        )
    })?;
    let entries: Vec<serde_json::Map<String, Value>> = serde_json::from_value(raw_entries.clone())
        .map_err(|e| {
            ClaudeError::message_parse(
                format!("transcript_mirror frame has invalid entries: {e}"),
                Some(frame.clone()),
            )
        })?;
    Ok((file_path, entries))
}

/// Tear down the reader on a transport read error: fan the error out to pending
/// waiters, flush the mirror batcher, and deliver the error to the stream.
///
/// The CLI emits an `is_error` result then exits non-zero on purpose, so a bare
/// process error is replaced with a [`ClaudeError::ResultError`] carrying that
/// result when one was seen this turn. Mirrors the TS/Python SDK.
async fn fail_read_error(
    error: ClaudeError,
    last_error_result: Option<&Value>,
    control: &ControlState,
    batcher: Option<&Arc<crate::sessions::TranscriptMirrorBatcher>>,
    sender: &mpsc::Sender<StreamItem>,
) {
    let pending_error = match (&error, last_error_result) {
        (ClaudeError::Process { exit_code, .. }, Some(result)) => {
            ClaudeError::result_error(result, *exit_code)
        }
        _ => error,
    };
    fail_pending(control, &pending_error).await;
    // Final mirror flush before teardown so an early transport error doesn't
    // drop entries batched this turn.
    if let Some(batcher) = batcher {
        batcher.close().await;
    }
    let _ = sender.send(StreamItem::Error(pending_error)).await;
}

/// Flush the mirror batcher before a `result` is yielded and track the last
/// error result, so a following process error can be replaced with a
/// [`ClaudeError::ResultError`] carrying that result.
async fn track_error_result(
    frame: &Value,
    msg_type: &str,
    batcher: Option<&Arc<crate::sessions::TranscriptMirrorBatcher>>,
    last_error_result: &mut Option<Value>,
) {
    if msg_type == "result" {
        // Flush pending mirror entries before yielding the result so a
        // consumer observing the result sees an up-to-date SessionStore.
        if let Some(batcher) = batcher {
            batcher.flush().await;
        }
        *last_error_result =
            (frame.get("is_error").and_then(Value::as_bool) == Some(true)).then(|| frame.clone());
    } else if !is_session_state_frame(frame, msg_type) {
        // Any non-marker frame means the conversation moved on; a later
        // process error is a fresh crash, not the expected error-result exit.
        *last_error_result = None;
    }
}

fn is_session_state_frame(frame: &Value, msg_type: &str) -> bool {
    msg_type == "system"
        && frame.get("subtype").and_then(Value::as_str) == Some("session_state_changed")
}

/// Whether a frame is a main-thread turn frame (not subagent output).
fn is_main_thread_turn_frame(frame: &Value, msg_type: &str) -> bool {
    matches!(msg_type, "assistant" | "stream_event")
        && frame.get("parent_tool_use_id").is_none_or(Value::is_null)
}

/// Signal every pending outbound control waiter with a fatal error.
async fn fail_pending(control: &ControlState, error: &ClaudeError) {
    let mut pending = control.pending.lock().await;
    for (_, tx) in pending.drain() {
        let _ = tx.send(Err(clone_error(error)));
    }
}

/// Resolve a pending control waiter from an inbound `control_response` frame.
async fn route_control_response(control: &ControlState, frame: &Value) {
    let Some(response) = frame.get("response") else {
        return;
    };
    let Some(request_id) = response.get("request_id").and_then(Value::as_str) else {
        return;
    };
    let Some(tx) = control.pending.lock().await.remove(request_id) else {
        return; // unknown / already-resolved id: ignore
    };
    let result = if response.get("subtype").and_then(Value::as_str) == Some("error") {
        Err(ClaudeError::CliConnection(
            response
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("Unknown error")
                .to_owned(),
        ))
    } else {
        // A success response with no `response` member (or a non-object one)
        // resolves to an empty object, matching Python's
        // `_send_control_request` default of `{}`.
        let inner = response.get("response").cloned().unwrap_or(Value::Null);
        Ok(if inner.is_object() {
            inner
        } else {
            Value::Object(serde_json::Map::new())
        })
    };
    let _ = tx.send(result);
}

/// Spawn a cancellable handler task for an inbound control request and track it.
///
/// A start gate guarantees the handle is registered in `inflight` before the
/// handler runs, so an immediately-following `control_cancel_request` always
/// finds it and the handler's self-removal never races ahead of registration.
async fn spawn_control_handler<T: AgentTransport>(ctx: ControlContext<T>, frame: Value) {
    let Some(request_id) = frame
        .get("request_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
    else {
        return;
    };
    let control = Arc::clone(&ctx.control);
    let (gate_tx, gate_rx) = oneshot::channel::<()>();
    let req_id = request_id.clone();
    let handle = tokio::spawn(async move {
        // Wait until registration completes; if the gate sender is dropped
        // (registration skipped) proceed anyway so we never wedge.
        let _ = gate_rx.await;
        handle_control_request(&ctx, &req_id, &frame).await;
        // Remove ourselves once done so a late cancel is a no-op.
        ctx.control.inflight.lock().await.remove(&req_id);
    });
    control.inflight.lock().await.insert(request_id, handle);
    // Registration done: release the handler.
    let _ = gate_tx.send(());
}

/// Process one inbound control request and write its response, unless the task
/// is cancelled (in which case the CLI has abandoned it and expects no reply).
async fn handle_control_request<T: AgentTransport>(
    ctx: &ControlContext<T>,
    request_id: &str,
    frame: &Value,
) {
    let request = frame.get("request").cloned().unwrap_or(Value::Null);
    let subtype = request
        .get("subtype")
        .and_then(Value::as_str)
        .unwrap_or_default();

    let response = match subtype {
        "can_use_tool" => handle_permission_request(ctx, &request).await,
        "hook_callback" => handle_hook_callback(ctx, &request).await,
        "mcp_message" => handle_mcp_message(ctx, &request).await,
        other => Err(ClaudeError::CliConnection(format!(
            "Unsupported control request subtype: {other}"
        ))),
    };

    let wire = match response {
        Ok(data) => json!({
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": request_id,
                "response": data,
            }
        }),
        Err(error) => json!({
            "type": "control_response",
            "response": {
                "subtype": "error",
                "request_id": request_id,
                "error": error.to_string(),
            }
        }),
    };
    let _ = write_frame(&ctx.transport, &wire).await;
}

/// Answer a `can_use_tool` request via the configured permission callback.
async fn handle_permission_request<T: AgentTransport>(
    ctx: &ControlContext<T>,
    request: &Value,
) -> Result<Value, ClaudeError> {
    let Some(callback) = &ctx.can_use_tool else {
        return Err(ClaudeError::CliConnection(
            "canUseTool callback is not provided".into(),
        ));
    };
    let tool_name = request
        .get("tool_name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let original_input = request.get("input").cloned().unwrap_or(Value::Null);
    let context = ToolPermissionContext::from_request(request);
    let result: PermissionResult = callback
        .can_use_tool(tool_name, &original_input, &context)
        .await?;
    Ok(result.to_wire(&original_input))
}

/// Answer a `hook_callback` request via the registered hook handler.
async fn handle_hook_callback<T: AgentTransport>(
    ctx: &ControlContext<T>,
    request: &Value,
) -> Result<Value, ClaudeError> {
    let callback_id = request
        .get("callback_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let Some(callback) = ctx.hook_callbacks.get(callback_id) else {
        return Err(ClaudeError::CliConnection(format!(
            "No hook callback found for ID: {callback_id}"
        )));
    };
    // The control protocol may deliver a minimal input object; pass it through
    // to the callback without requiring base fields (matches Python's
    // permissive dispatch).
    let input = HookInput::from_value(request.get("input").cloned().unwrap_or(Value::Null));
    let tool_use_id = request.get("tool_use_id").and_then(Value::as_str);
    let context = HookContext { signal: None };
    let output = callback.call(&input, tool_use_id, &context).await?;
    Ok(output.to_wire())
}

/// Answer an `mcp_message` request by routing to the named in-process server.
async fn handle_mcp_message<T: AgentTransport>(
    ctx: &ControlContext<T>,
    request: &Value,
) -> Result<Value, ClaudeError> {
    let server_name = request.get("server_name").and_then(Value::as_str);
    let message = request.get("message");
    let (Some(server_name), Some(message)) = (server_name, message) else {
        return Err(ClaudeError::CliConnection(
            "Missing server_name or message for MCP request".into(),
        ));
    };
    if server_name.is_empty() || !message.is_object() {
        return Err(ClaudeError::CliConnection(
            "Missing server_name or message for MCP request".into(),
        ));
    }
    let mcp_response = if let Some(server) = ctx.mcp_servers.get(server_name) {
        server.handle(message).await
    } else {
        json!({
            "jsonrpc": "2.0",
            "id": message.get("id").cloned().unwrap_or(Value::Null),
            "error": {
                "code": -32_601,
                "message": format!("Server '{server_name}' not found"),
            }
        })
    };
    Ok(json!({ "mcp_response": mcp_response }))
}

/// Track in-flight delegated tasks from `system` task-lifecycle frames.
fn track_task_lifecycle(frame: &Value, inflight: &mut HashSet<String>) {
    let subtype = frame.get("subtype").and_then(Value::as_str);
    let Some(task_id) = frame.get("task_id").and_then(Value::as_str) else {
        return;
    };
    match subtype {
        Some("task_started") => {
            let task_type = frame.get("task_type").and_then(Value::as_str).unwrap_or("");
            if DEFERRING_TASK_TYPES.contains(&task_type) {
                inflight.insert(task_id.to_owned());
            }
        }
        Some("task_notification") => {
            inflight.remove(task_id);
        }
        Some("task_updated") => {
            let status = frame
                .get("patch")
                .and_then(|patch| patch.get("status"))
                .and_then(Value::as_str)
                .unwrap_or("");
            if TERMINAL_TASK_STATUSES.contains(&status) {
                inflight.remove(task_id);
            }
        }
        _ => {}
    }
}

/// Environment variable bounding the wait for the CLI's `idle`; the CLI reads
/// the same variable for its own background-wait ceiling once stdin closes.
const RUN_END_CEILING_ENV: &str = "CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS";

/// The CLI's default background-wait ceiling (10 minutes).
const DEFAULT_RUN_END_CEILING_MS: u64 = 600_000;

/// Longest ceiling honored (~24.8 days), as in the Python and TypeScript SDKs.
const MAX_RUN_END_CEILING_MS: u64 = 2_147_483_647;

/// Read `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS` as the CLI will see it:
/// `options_env` overrides the inherited environment. `0` means no limit;
/// anything that is not a plain non-negative integer falls back to the CLI's
/// default of 10 minutes.
fn run_end_ceiling_ms(options_env: &BTreeMap<String, String>) -> u64 {
    options_env
        .get(RUN_END_CEILING_ENV)
        .cloned()
        .or_else(|| std::env::var(RUN_END_CEILING_ENV).ok())
        .and_then(|raw| raw.trim().parse::<i64>().ok())
        .and_then(|value| u64::try_from(value).ok())
        .unwrap_or(DEFAULT_RUN_END_CEILING_MS)
}

/// Run-boundary state machine for runs that serve control requests (hooks,
/// `can_use_tool`, SDK MCP servers), mirroring Python `Query` (#1088, #1190).
///
/// A `result` frame ends one turn, not necessarily the run: a background agent
/// that finished just before it still wakes the session for a follow-up turn
/// whose control requests need stdin. A CLI that reports
/// `session_state_changed` stays `running` while such a turn is owed and
/// reports `idle` once it is not, so the run ends at the `idle` that follows a
/// result (or at a result that follows `idle`). A CLI that sends no state
/// falls back to ending the run at the first result with no tracked
/// background agent in flight. The wait for `idle` is bounded by a ceiling
/// that counts only the gap between turns.
///
/// Every transition method returns `true` when the run just ended.
// The four flags are independent run-boundary facts, not a state machine.
#[allow(clippy::struct_excessive_bools)]
struct RunTracker {
    /// Whether control requests may still need stdin (hooks, SDK MCP servers,
    /// or a permission callback).
    needs_stdin: bool,
    /// Bound on the wait for `idle`; `None` waits however long it takes.
    ceiling: Option<Duration>,
    /// Started-but-unfinished delegated tasks (#1088).
    inflight_tasks: HashSet<String>,
    /// Latest `session_state_changed` state; `None` while the CLI sends none.
    session_state: Option<String>,
    result_received: bool,
    /// A main-thread turn is under way, so the ceiling is not armed.
    turn_in_progress: bool,
    run_ended: bool,
    deadline: Option<Instant>,
}

impl RunTracker {
    fn new(needs_stdin: bool, ceiling_ms: u64) -> Self {
        Self {
            needs_stdin,
            ceiling: (ceiling_ms > 0)
                .then(|| Duration::from_millis(ceiling_ms.min(MAX_RUN_END_CEILING_MS))),
            inflight_tasks: HashSet::new(),
            session_state: None,
            result_received: false,
            turn_in_progress: false,
            run_ended: false,
            deadline: None,
        }
    }

    /// Track a `system` frame: task lifecycle and `session_state_changed`.
    fn on_system_frame(&mut self, frame: &Value) -> bool {
        let had_tasks = !self.inflight_tasks.is_empty();
        track_task_lifecycle(frame, &mut self.inflight_tasks);
        if had_tasks && self.inflight_tasks.is_empty() {
            // The ceiling left the last tracked agent alone; the wait between
            // turns starts over now that it settled.
            self.rearm_between_turns();
        }
        if frame.get("subtype").and_then(Value::as_str) == Some("session_state_changed") {
            return self.on_session_state(frame.get("state").and_then(Value::as_str));
        }
        false
    }

    fn on_session_state(&mut self, state: Option<&str>) -> bool {
        self.session_state = state.map(str::to_owned);
        if state == Some("idle") {
            return self.result_received && self.maybe_end_run();
        }
        // Work the CLI took up after the run ended reopens the run until the
        // next `idle`.
        self.run_ended = false;
        if state == Some("requires_action") {
            // The host is answering a request; stdin must outlast it.
            self.deadline = None;
        } else {
            self.rearm_between_turns();
        }
        false
    }

    fn on_result(&mut self) -> bool {
        self.result_received = true;
        self.turn_in_progress = false;
        // Every result is a fresh run-boundary decision.
        self.run_ended = false;
        let state = self.session_state.as_deref();
        if matches!(state, None | Some("idle")) || !self.needs_stdin {
            return self.maybe_end_run();
        }
        if state != Some("requires_action") {
            // While the SDK is still answering a request the ceiling waits for
            // the `running` that follows.
            self.arm_ceiling();
        }
        false
    }

    /// A main-thread `assistant`/`stream_event` frame: a turn is under way, so
    /// the ceiling stops and an ended run reopens.
    fn on_turn_activity(&mut self) {
        self.turn_in_progress = true;
        self.run_ended = false;
        self.deadline = None;
    }

    /// The ceiling timer for `deadline` fired.
    fn on_ceiling(&mut self, deadline: Instant) -> bool {
        if self.deadline != Some(deadline) {
            return false;
        }
        self.deadline = None;
        // A tracked background agent may still need stdin for its hook,
        // permission and SDK MCP requests, so it is not cut off; the ceiling
        // starts over once it settles.
        self.inflight_tasks.is_empty() && self.end_run()
    }

    fn maybe_end_run(&mut self) -> bool {
        // Tracked background agents may still need stdin; a later result or
        // `idle` ends the run once they settle.
        self.inflight_tasks.is_empty() && self.end_run()
    }

    fn end_run(&mut self) -> bool {
        self.deadline = None;
        !std::mem::replace(&mut self.run_ended, true)
    }

    fn arm_ceiling(&mut self) {
        self.deadline = None;
        let Some(ceiling) = self.ceiling else {
            return;
        };
        if self.run_ended || self.turn_in_progress || !self.needs_stdin {
            return;
        }
        self.deadline = Some(Instant::now() + ceiling);
    }

    /// Restart the ceiling when the run is between turns, past a result, with
    /// the CLI still reporting work.
    fn rearm_between_turns(&mut self) {
        if self.result_received
            && !matches!(
                self.session_state.as_deref(),
                None | Some("idle" | "requires_action")
            )
        {
            self.arm_ceiling();
        }
    }
}

/// [`RunTracker`] shared between the reader and the ceiling timer task.
struct RunState {
    tracker: parking_lot::Mutex<RunTracker>,
    /// Signalled when the tracker's deadline changed.
    deadline_changed: tokio::sync::Notify,
}

impl RunState {
    fn new(tracker: RunTracker) -> Self {
        Self {
            tracker: parking_lot::Mutex::new(tracker),
            deadline_changed: tokio::sync::Notify::new(),
        }
    }

    /// Apply `transition`, waking the timer task if the deadline moved.
    /// Returns whether the run just ended.
    fn update(&self, transition: impl FnOnce(&mut RunTracker) -> bool) -> bool {
        let mut tracker = self.tracker.lock();
        let before = tracker.deadline;
        let ended = transition(&mut tracker);
        if tracker.deadline != before {
            self.deadline_changed.notify_one();
        }
        ended
    }

    fn has_ceiling_timer(&self) -> bool {
        let tracker = self.tracker.lock();
        tracker.needs_stdin && tracker.ceiling.is_some()
    }
}

/// Aborts the wrapped task when dropped, so the ceiling timer dies with the
/// reader even when the reader itself is aborted.
struct AbortOnDrop(JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Ends the run when no `idle` arrives within the ceiling after a result.
async fn run_end_timer(state: Arc<RunState>, sender: mpsc::Sender<StreamItem>) {
    loop {
        let notified = state.deadline_changed.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        let deadline = state.tracker.lock().deadline;
        let Some(deadline) = deadline else {
            notified.await;
            continue;
        };
        tokio::select! {
            () = tokio::time::sleep_until(deadline) => {
                if state.update(|tracker| tracker.on_ceiling(deadline))
                    && sender.send(StreamItem::RunEnded).await.is_err()
                {
                    return;
                }
            }
            () = &mut notified => {}
        }
    }
}

/// Build the initialize hooks payload and register callbacks by generated id.
///
/// Returns the wire `hooks` object (or `Value::Null` when empty) and the
/// callback-id → handler map installed before the request is written.
fn build_hooks(
    hooks: Option<&std::collections::BTreeMap<HookEvent, Vec<HookMatcher>>>,
) -> (Value, HashMap<String, HookCallback>) {
    let mut callbacks: HashMap<String, HookCallback> = HashMap::new();
    let mut next_id = 0u64;
    let mut config = serde_json::Map::new();

    if let Some(hooks) = hooks {
        for (event, matchers) in hooks {
            if matchers.is_empty() {
                continue;
            }
            let mut event_matchers = Vec::with_capacity(matchers.len());
            for matcher in matchers {
                let mut callback_ids = Vec::with_capacity(matcher.hooks.len());
                for callback in &matcher.hooks {
                    let callback_id = format!("hook_{next_id}");
                    next_id += 1;
                    callbacks.insert(callback_id.clone(), callback.clone());
                    callback_ids.push(Value::String(callback_id));
                }
                let mut matcher_config = serde_json::Map::new();
                matcher_config.insert(
                    "matcher".into(),
                    matcher.matcher.clone().map_or(Value::Null, Value::String),
                );
                matcher_config.insert("hookCallbackIds".into(), Value::Array(callback_ids));
                if let Some(timeout) = matcher.timeout {
                    if let Some(number) = serde_json::Number::from_f64(timeout) {
                        matcher_config.insert("timeout".into(), Value::Number(number));
                    }
                }
                event_matchers.push(Value::Object(matcher_config));
            }
            config.insert(event.as_wire().to_owned(), Value::Array(event_matchers));
        }
    }

    let hooks_value = if config.is_empty() {
        Value::Null
    } else {
        Value::Object(config)
    };
    (hooks_value, callbacks)
}

/// Assemble the `initialize` control request body from options.
fn build_initialize_request(
    options: &AgentOptions,
    hooks_config: Value,
) -> Result<Value, ClaudeError> {
    let mut request = serde_json::Map::new();
    request.insert("subtype".into(), Value::String("initialize".into()));
    request.insert("hooks".into(), hooks_config);

    if let Some(agents) = options.agents.as_ref() {
        if !agents.is_empty() {
            let mut map = serde_json::Map::with_capacity(agents.len());
            for (name, def) in agents {
                map.insert(name.clone(), def.to_initialize_value()?);
            }
            request.insert("agents".into(), Value::Object(map));
        }
    }

    if let Some(SystemPrompt::Preset {
        exclude_dynamic_sections: Some(exclude),
        ..
    }) = options.system_prompt.as_ref()
    {
        request.insert("excludeDynamicSections".into(), Value::Bool(*exclude));
    }

    if let Some(snapshot) = options
        .system_prompt
        .as_ref()
        .and_then(SystemPrompt::snapshot)
    {
        request.insert("systemPromptSnapshot".into(), Value::Bool(snapshot));
    }

    if options.forward_subagent_text {
        request.insert("forwardSubagentText".into(), Value::Bool(true));
    }

    // 'all' and omitted are equivalent at the wire level; only send an explicit
    // list.
    if let Some(SkillSelection::List(names)) = options.skills.as_ref() {
        request.insert(
            "skills".into(),
            Value::Array(names.iter().cloned().map(Value::String).collect()),
        );
    }

    Ok(Value::Object(request))
}

/// Clone a [`ClaudeError`] for fan-out to multiple pending waiters.
///
/// `ClaudeError` is not `Clone` (it wraps non-`Clone` sources), so this
/// reconstructs an equivalent connection-level error preserving the message.
fn clone_error(error: &ClaudeError) -> ClaudeError {
    match error {
        ClaudeError::Process {
            message,
            exit_code,
            stderr,
        } => ClaudeError::Process {
            message: message.clone(),
            exit_code: *exit_code,
            stderr: stderr.clone(),
        },
        ClaudeError::ResultError(details) => ClaudeError::ResultError(details.clone()),
        ClaudeError::ControlTimeout { subtype } => ClaudeError::ControlTimeout {
            subtype: subtype.clone(),
        },
        other => ClaudeError::CliConnection(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_mirror_frame_accepts_valid() {
        let frame = json!({
            "type": "transcript_mirror",
            "filePath": "/p/proj/sess.jsonl",
            "entries": [{"type": "user", "uuid": "a"}, {"type": "assistant", "uuid": "b"}],
        });
        let (path, entries) = parse_mirror_frame(&frame).unwrap();
        assert_eq!(path, "/p/proj/sess.jsonl");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].get("uuid").unwrap(), "a");
    }

    #[test]
    fn parse_mirror_frame_accepts_empty_entries() {
        let frame = json!({
            "type": "transcript_mirror",
            "filePath": "/p/proj/sess.jsonl",
            "entries": [],
        });
        let (_, entries) = parse_mirror_frame(&frame).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn parse_mirror_frame_missing_file_path_is_message_parse() {
        // Python accesses `message["filePath"]` directly — a missing key raises
        // rather than silently dropping the frame.
        let frame = json!({"type": "transcript_mirror", "entries": []});
        assert!(matches!(
            parse_mirror_frame(&frame),
            Err(ClaudeError::MessageParse { .. })
        ));
    }

    #[test]
    fn parse_mirror_frame_missing_entries_is_message_parse() {
        let frame = json!({"type": "transcript_mirror", "filePath": "/p/s.jsonl"});
        assert!(matches!(
            parse_mirror_frame(&frame),
            Err(ClaudeError::MessageParse { .. })
        ));
    }

    #[test]
    fn parse_mirror_frame_non_string_file_path_is_message_parse() {
        let frame = json!({"type": "transcript_mirror", "filePath": 42, "entries": []});
        assert!(matches!(
            parse_mirror_frame(&frame),
            Err(ClaudeError::MessageParse { .. })
        ));
    }

    #[test]
    fn parse_mirror_frame_non_array_entries_is_message_parse() {
        // `entries` must be an array; a scalar or object is malformed and must
        // fail visibly instead of defaulting to an empty batch.
        let frame = json!({"type": "transcript_mirror", "filePath": "/p/s.jsonl", "entries": 7});
        assert!(matches!(
            parse_mirror_frame(&frame),
            Err(ClaudeError::MessageParse { .. })
        ));
    }

    #[test]
    fn parse_mirror_frame_non_object_entry_is_message_parse() {
        // Entries must be JSON objects; a scalar element is malformed.
        let frame = json!({
            "type": "transcript_mirror",
            "filePath": "/p/s.jsonl",
            "entries": [{"type": "user"}, 5],
        });
        assert!(matches!(
            parse_mirror_frame(&frame),
            Err(ClaudeError::MessageParse { .. })
        ));
    }

    fn state_frame(state: &str) -> Value {
        json!({"type": "system", "subtype": "session_state_changed", "state": state})
    }

    fn task_frame(subtype: &str, task_id: &str) -> Value {
        json!({"type": "system", "subtype": subtype, "task_id": task_id,
               "task_type": "local_agent"})
    }

    #[test]
    fn run_ends_at_idle_after_a_result() {
        let mut tracker = RunTracker::new(true, 0);
        assert!(!tracker.on_system_frame(&state_frame("running")));
        // Idle before any result never ends the run.
        assert!(!tracker.on_system_frame(&state_frame("idle")));
        assert!(!tracker.on_system_frame(&state_frame("running")));
        assert!(!tracker.on_result());
        assert!(tracker.on_system_frame(&state_frame("idle")));
    }

    #[test]
    fn run_ends_at_result_when_idle_came_first_or_state_is_absent() {
        let mut tracker = RunTracker::new(true, 0);
        tracker.on_system_frame(&state_frame("idle"));
        assert!(tracker.on_result());

        let mut stateless = RunTracker::new(true, 0);
        assert!(stateless.on_result());
        // A later turn's result is a fresh decision.
        assert!(stateless.on_result());
    }

    #[test]
    fn tracked_background_agent_holds_the_run_open() {
        let mut tracker = RunTracker::new(true, 0);
        tracker.on_system_frame(&task_frame("task_started", "t1"));
        assert!(!tracker.on_result());
        assert!(!tracker.on_system_frame(&state_frame("idle")));
        tracker.on_system_frame(&task_frame("task_notification", "t1"));
        assert!(tracker.on_result());
    }

    #[test]
    fn ceiling_arms_between_turns_and_stops_for_turn_activity() {
        let mut tracker = RunTracker::new(true, 1_000);
        tracker.on_system_frame(&state_frame("running"));
        assert!(!tracker.on_result());
        let deadline = tracker.deadline.expect("ceiling armed after a result");
        // A main-thread turn stops the ceiling.
        tracker.on_turn_activity();
        assert!(tracker.deadline.is_none());
        assert!(!tracker.on_ceiling(deadline));
        // The next result re-arms it, and firing ends the run once.
        assert!(!tracker.on_result());
        let deadline = tracker.deadline.expect("re-armed");
        assert!(tracker.on_ceiling(deadline));
        assert!(!tracker.on_ceiling(deadline));
    }

    #[test]
    fn ceiling_leaves_tracked_agent_alone_and_restarts_when_it_settles() {
        let mut tracker = RunTracker::new(true, 1_000);
        tracker.on_system_frame(&task_frame("task_started", "t1"));
        tracker.on_system_frame(&state_frame("running"));
        tracker.on_result();
        let deadline = tracker.deadline.unwrap();
        assert!(!tracker.on_ceiling(deadline));
        assert!(tracker.deadline.is_none());
        tracker.on_system_frame(&task_frame("task_notification", "t1"));
        assert!(tracker.deadline.is_some());
    }

    #[test]
    fn requires_action_pauses_the_ceiling() {
        let mut tracker = RunTracker::new(true, 1_000);
        tracker.on_system_frame(&state_frame("requires_action"));
        tracker.on_result();
        assert!(tracker.deadline.is_none());
        tracker.on_system_frame(&state_frame("running"));
        assert!(tracker.deadline.is_some());
    }

    #[test]
    fn run_end_ceiling_reads_options_env_with_cli_default() {
        let env =
            |value: &str| BTreeMap::from([(RUN_END_CEILING_ENV.to_owned(), value.to_owned())]);
        assert_eq!(run_end_ceiling_ms(&env("1500")), 1_500);
        assert_eq!(run_end_ceiling_ms(&env("0")), 0);
        assert_eq!(run_end_ceiling_ms(&env("-5")), DEFAULT_RUN_END_CEILING_MS);
        assert_eq!(run_end_ceiling_ms(&env("soon")), DEFAULT_RUN_END_CEILING_MS);
    }
}
