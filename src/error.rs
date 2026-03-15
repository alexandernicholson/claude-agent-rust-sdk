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
}
