//! Types for the Claude Message Batches API.

use serde::{Deserialize, Serialize};

use super::CreateMessageRequest;
use super::CreateMessageResponse;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// A single request inside a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequest {
    /// Caller-defined identifier for correlating results.
    pub custom_id: String,
    /// The message-creation parameters (same shape as a regular request).
    pub params: CreateMessageRequest,
}

/// Body of a `POST /v1/messages/batches` request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateBatchRequest {
    pub requests: Vec<BatchRequest>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Status of a batch job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    InProgress,
    Ended,
    Canceling,
    Canceled,
    Expired,
}

/// Per-status request counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequestCounts {
    pub processing: u64,
    pub succeeded: u64,
    pub errored: u64,
    pub canceled: u64,
    pub expired: u64,
}

/// Body of a batch-status response (`GET /v1/messages/batches/{id}`, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResponse {
    pub id: String,
    pub processing_status: BatchStatus,
    pub request_counts: BatchRequestCounts,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<String>,

    pub created_at: String,
    pub expires_at: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub results_url: Option<String>,
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// An error returned for a single request inside a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

/// The outcome of a single request inside a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BatchResultBody {
    Succeeded { message: CreateMessageResponse },
    Errored {
        #[serde(default)]
        error: Option<BatchError>,
    },
    Canceled {},
    Expired {},
}

/// One line of a batch results file (JSONL).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub custom_id: String,
    pub result: BatchResultBody,
}
