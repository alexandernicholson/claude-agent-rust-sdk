//! Pre-flight validation for [`AgentOptions::session_store`] combinations.
//!
//! Called before subprocess spawn so misconfiguration fails fast instead of
//! surfacing as a confusing runtime error mid-session. Mirrors the official
//! Python `validate_session_store_options`, adapted to Rust's explicit
//! capability probing (there is no duck typing — a store advertises optional
//! methods via [`SessionStore::capabilities`]).
//!
//! [`AgentOptions::session_store`]: crate::agent::AgentOptions
//! [`SessionStore::capabilities`]: crate::sessions::store::SessionStore::capabilities

use crate::agent::AgentOptions;
use crate::error::ClaudeError;

/// Validate `session_store` option combinations.
///
/// Rules (parity with the official SDK):
/// - `continue_conversation` with a `session_store` requires the store to
///   implement `list_sessions` — unless `resume` is also set, in which case
///   `list_sessions` is provably never called (explicit resume wins over
///   continue), so a minimal store is fine.
/// - `session_store` cannot be combined with `enable_file_checkpointing`
///   (checkpoints are local-disk only and would diverge from the mirrored
///   transcript).
///
/// # Errors
///
/// Returns [`ClaudeError::InvalidConfig`] for any invalid combination. Returns
/// `Ok(())` when no `session_store` is configured.
pub fn validate_session_store_options(options: &AgentOptions) -> Result<(), ClaudeError> {
    let Some(store) = options.session_store.as_ref() else {
        return Ok(());
    };

    if options.continue_conversation
        && options.resume.is_none()
        && !store.capabilities().list_sessions
    {
        return Err(ClaudeError::InvalidConfig(
            "continue_conversation with session_store requires the store to \
             implement list_sessions()"
                .into(),
        ));
    }

    if options.enable_file_checkpointing {
        return Err(ClaudeError::InvalidConfig(
            "session_store cannot be combined with enable_file_checkpointing \
             (checkpoints are local-disk only and would diverge from the \
             mirrored transcript)"
                .into(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sessions::key::SessionKey;
    use crate::sessions::store::{SessionStore, SessionStoreCapabilities, SessionStoreEntry};
    use std::sync::Arc;

    #[derive(Debug)]
    struct CapStore(SessionStoreCapabilities);

    #[async_trait::async_trait]
    impl SessionStore for CapStore {
        async fn append(
            &self,
            _key: &SessionKey,
            _entries: Vec<SessionStoreEntry>,
        ) -> Result<(), ClaudeError> {
            Ok(())
        }
        async fn load(
            &self,
            _key: &SessionKey,
        ) -> Result<Option<Vec<SessionStoreEntry>>, ClaudeError> {
            Ok(None)
        }
        fn capabilities(&self) -> SessionStoreCapabilities {
            self.0
        }
    }

    fn minimal_store() -> Arc<dyn SessionStore> {
        Arc::new(CapStore(SessionStoreCapabilities::default()))
    }

    fn listing_store() -> Arc<dyn SessionStore> {
        Arc::new(CapStore(SessionStoreCapabilities {
            list_sessions: true,
            ..Default::default()
        }))
    }

    #[test]
    fn no_store_is_always_valid() {
        let opts = AgentOptions {
            continue_conversation: true,
            enable_file_checkpointing: true,
            ..Default::default()
        };
        assert!(validate_session_store_options(&opts).is_ok());
    }

    #[test]
    fn continue_without_list_sessions_rejected() {
        let opts = AgentOptions {
            session_store: Some(minimal_store()),
            continue_conversation: true,
            ..Default::default()
        };
        let err = validate_session_store_options(&opts).unwrap_err();
        assert!(matches!(err, ClaudeError::InvalidConfig(_)));
        assert!(err.to_string().contains("list_sessions"));
    }

    #[test]
    fn continue_with_list_sessions_allowed() {
        let opts = AgentOptions {
            session_store: Some(listing_store()),
            continue_conversation: true,
            ..Default::default()
        };
        assert!(validate_session_store_options(&opts).is_ok());
    }

    #[test]
    fn continue_with_explicit_resume_allows_minimal_store() {
        // resume wins over continue → list_sessions never called → minimal ok.
        let opts = AgentOptions {
            session_store: Some(minimal_store()),
            continue_conversation: true,
            resume: Some(uuid::Uuid::new_v4().to_string()),
            ..Default::default()
        };
        assert!(validate_session_store_options(&opts).is_ok());
    }

    #[test]
    fn store_with_file_checkpointing_rejected() {
        let opts = AgentOptions {
            session_store: Some(listing_store()),
            enable_file_checkpointing: true,
            ..Default::default()
        };
        let err = validate_session_store_options(&opts).unwrap_err();
        assert!(matches!(err, ClaudeError::InvalidConfig(_)));
        assert!(err.to_string().contains("enable_file_checkpointing"));
    }

    #[test]
    fn plain_resume_store_valid() {
        let opts = AgentOptions {
            session_store: Some(minimal_store()),
            resume: Some(uuid::Uuid::new_v4().to_string()),
            ..Default::default()
        };
        assert!(validate_session_store_options(&opts).is_ok());
    }
}
