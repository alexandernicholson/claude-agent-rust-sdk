//! Session persistence: keys, the [`SessionStore`] adapter contract, the
//! incremental summary fold, and the reference in-memory adapter.
//!
//! Ported from the official Python Agent SDK session subsystem. The public
//! surface here is the store/key/summary contract; higher-level filesystem
//! query, mirroring, resume, mutation, and import layers live in sibling
//! modules that build on these types.

pub mod key;
pub mod store;
pub mod summary;

/// Reusable [`SessionStore`] conformance suite for validating adapters.
pub mod conformance;

/// Filesystem-backed session discovery and transcript reading.
pub mod filesystem;
/// Streaming import of local transcripts into a [`SessionStore`].
pub mod import;
/// Transcript-mirror batching from CLI frames into a [`SessionStore`].
pub mod mirror;
/// Portable session mutations: rename, tag, delete, fork.
pub mod mutations;
/// Store-backed resume/continue materialization into a temp config dir.
pub mod resume;
/// Store-backed session query APIs (async counterparts to the disk queries).
pub mod store_queries;
/// Pre-flight validation of `session_store` option combinations.
pub mod validation;

pub use filesystem::{
    get_session_info, get_session_messages, get_subagent_messages, list_sessions, list_subagents,
    SessionMessage, SessionMessageType,
};
pub use import::import_session_to_store;
pub use key::{
    file_path_to_session_key, project_key_for_directory, validate_subpath, SessionKey,
    SessionListSubkeysKey,
};
pub use mirror::{
    MirrorErrorHandler, TranscriptMirrorBatcher, MAX_PENDING_BYTES, MAX_PENDING_ENTRIES,
    MIRROR_APPEND_BACKOFF_S, MIRROR_APPEND_MAX_ATTEMPTS, SEND_TIMEOUT_SECONDS,
};
pub use mutations::{
    delete_session, delete_session_via_store, fork_session, fork_session_via_store, rename_session,
    rename_session_via_store, tag_session, tag_session_via_store, ForkSessionResult,
};
pub use resume::{
    apply_materialized_options, build_mirror_batcher, materialize_resume_session,
    MaterializedResume,
};
pub use store::{
    InMemorySessionStore, SDKSessionInfo, SessionStore, SessionStoreCapabilities,
    SessionStoreEntry, SessionStoreFlushMode, SessionStoreListEntry,
};
pub use store_queries::{
    get_session_info_from_store, get_session_messages_from_store, get_subagent_messages_from_store,
    list_sessions_from_store, list_subagents_from_store,
};
pub use summary::{fold_session_summary, summary_entry_to_sdk_info, SessionSummaryEntry};
pub use validation::validate_session_store_options;
