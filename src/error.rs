//! Error types for the Claude SDK.
//!
//! All fallible operations in this crate return [`ClaudeError`]. The variants
//! cover API-level errors, network failures, serialization issues, batch
//! timeouts, configuration mistakes, streaming errors, unsupported transport
//! operations, and transport-specific failures.

use thiserror::Error;

/// Errors returned by the Claude SDK.
#[derive(Debug, Error)]
pub enum ClaudeError {
    /// The API returned a non-success status code.
    #[error("API error (status {status}): [{error_type}] {message}")]
    ApiError {
        status: u16,
        error_type: String,
        message: String,
    },

    /// A network-level error occurred (DNS, connection, timeout, etc.).
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    /// Failed to serialize a request or deserialize a response.
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// A batch job did not complete within the allowed polling window.
    #[error("Batch {batch_id} timed out waiting for completion")]
    BatchTimeout { batch_id: String },

    /// The SDK was configured with invalid parameters.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// An error received inside a streaming event.
    #[error("Stream error: [{error_type}] {message}")]
    StreamError {
        error_type: String,
        message: String,
    },

    /// The transport does not support this operation.
    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    /// A transport-specific error (e.g. CLI process failure).
    #[error("Transport error: {0}")]
    TransportError(String),
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn api_error_display() {
        let err = ClaudeError::ApiError {
            status: 429,
            error_type: "rate_limit_error".into(),
            message: "Too many requests".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("429"));
        assert!(msg.contains("rate_limit_error"));
        assert!(msg.contains("Too many requests"));
    }

    #[test]
    fn batch_timeout_display() {
        let err = ClaudeError::BatchTimeout {
            batch_id: "batch_123".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("batch_123"));
        assert!(msg.contains("timed out"));
    }

    #[test]
    fn invalid_config_display() {
        let err = ClaudeError::InvalidConfig("model is required".into());
        let msg = format!("{}", err);
        assert!(msg.contains("model is required"));
    }

    #[test]
    fn stream_error_display() {
        let err = ClaudeError::StreamError {
            error_type: "overloaded_error".into(),
            message: "Overloaded".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("overloaded_error"));
        assert!(msg.contains("Overloaded"));
    }

    #[test]
    fn serialization_error_from() {
        let serde_err = serde_json::from_str::<String>("not json").unwrap_err();
        let err: ClaudeError = serde_err.into();
        assert!(matches!(err, ClaudeError::SerializationError(_)));
        // Ensure it implements std::error::Error
        let _source = err.source();
    }

    #[test]
    fn error_variants_are_debug() {
        let errs: Vec<ClaudeError> = vec![
            ClaudeError::ApiError {
                status: 400,
                error_type: "invalid_request_error".into(),
                message: "bad".into(),
            },
            ClaudeError::BatchTimeout {
                batch_id: "b1".into(),
            },
            ClaudeError::InvalidConfig("x".into()),
            ClaudeError::StreamError {
                error_type: "e".into(),
                message: "m".into(),
            },
        ];
        for err in errs {
            let debug = format!("{:?}", err);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn api_error_status_codes() {
        let codes: Vec<(u16, &str)> = vec![
            (400, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
            (413, "request_too_large"),
            (429, "rate_limit_error"),
            (500, "api_error"),
            (529, "overloaded_error"),
        ];
        for (status, error_type) in codes {
            let err = ClaudeError::ApiError {
                status,
                error_type: error_type.into(),
                message: "test".into(),
            };
            let msg = format!("{}", err);
            assert!(msg.contains(&status.to_string()));
            assert!(msg.contains(error_type));
        }
    }
}
