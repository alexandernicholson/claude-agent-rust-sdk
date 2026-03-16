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
    Canceling,
    Ended,
}

/// Per-status request counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequestCounts {
    pub processing: u64,
    pub succeeded: u64,
    pub errored: u64,
    pub canceled: u64,
    #[serde(default)]
    pub expired: u64,
}

/// Body of a batch-status response (`GET /v1/messages/batches/{id}`, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResponse {
    pub id: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "type")]
    pub batch_type: Option<String>,

    pub processing_status: BatchStatus,
    pub request_counts: BatchRequestCounts,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<String>,

    pub created_at: String,
    pub expires_at: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub results_url: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancel_initiated_at: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub archived_at: Option<String>,
}

// ---------------------------------------------------------------------------
// List response
// ---------------------------------------------------------------------------

/// Response from `GET /v1/messages/batches` (paginated list).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListBatchesResponse {
    /// The batches on this page.
    pub data: Vec<BatchResponse>,

    /// Whether more results exist in the requested direction.
    pub has_more: bool,

    /// First ID in the `data` list (use as `before_id` for the previous page).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,

    /// Last ID in the `data` list (use as `after_id` for the next page).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
}

/// Query parameters for listing batches.
#[derive(Debug, Clone, Default)]
pub struct ListBatchesParams {
    /// ID of the object to use as cursor (returns page after this object).
    pub after_id: Option<String>,
    /// ID of the object to use as cursor (returns page before this object).
    pub before_id: Option<String>,
    /// Number of items per page (1..1000, default 20).
    pub limit: Option<u32>,
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

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_status_round_trip() {
        let statuses = [
            (BatchStatus::InProgress, "\"in_progress\""),
            (BatchStatus::Canceling, "\"canceling\""),
            (BatchStatus::Ended, "\"ended\""),
        ];
        for (status, expected_json) in statuses {
            let json = serde_json::to_string(&status).unwrap();
            assert_eq!(json, expected_json);
            let back: BatchStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn batch_request_counts_round_trip() {
        let counts = BatchRequestCounts {
            processing: 10,
            succeeded: 5,
            errored: 2,
            canceled: 1,
            expired: 0,
        };
        let json = serde_json::to_value(&counts).unwrap();
        assert_eq!(json["processing"], 10);
        assert_eq!(json["succeeded"], 5);
        let back: BatchRequestCounts = serde_json::from_value(json).unwrap();
        assert_eq!(back.processing, 10);
    }

    #[test]
    fn batch_response_round_trip() {
        let json = serde_json::json!({
            "id": "msgbatch_123",
            "type": "message_batch",
            "processing_status": "in_progress",
            "request_counts": {
                "processing": 10,
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0
            },
            "created_at": "2024-01-01T00:00:00Z",
            "expires_at": "2024-01-02T00:00:00Z",
            "results_url": null
        });
        let resp: BatchResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.id, "msgbatch_123");
        assert_eq!(resp.processing_status, BatchStatus::InProgress);
        assert!(resp.results_url.is_none());
    }

    #[test]
    fn batch_response_ended_with_results() {
        let json = serde_json::json!({
            "id": "msgbatch_456",
            "type": "message_batch",
            "processing_status": "ended",
            "request_counts": {
                "processing": 0,
                "succeeded": 5,
                "errored": 1,
                "canceled": 0,
                "expired": 0
            },
            "created_at": "2024-01-01T00:00:00Z",
            "expires_at": "2024-01-02T00:00:00Z",
            "ended_at": "2024-01-01T01:00:00Z",
            "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_456/results"
        });
        let resp: BatchResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.processing_status, BatchStatus::Ended);
        assert!(resp.results_url.is_some());
        assert!(resp.ended_at.is_some());
    }

    #[test]
    fn list_batches_response_round_trip() {
        let json = serde_json::json!({
            "data": [
                {
                    "id": "msgbatch_1",
                    "processing_status": "ended",
                    "request_counts": {
                        "processing": 0,
                        "succeeded": 3,
                        "errored": 0,
                        "canceled": 0,
                        "expired": 0
                    },
                    "created_at": "2024-01-01T00:00:00Z",
                    "expires_at": "2024-01-02T00:00:00Z"
                }
            ],
            "has_more": true,
            "first_id": "msgbatch_1",
            "last_id": "msgbatch_1"
        });
        let resp: ListBatchesResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.data.len(), 1);
        assert!(resp.has_more);
        assert_eq!(resp.first_id.as_deref(), Some("msgbatch_1"));
        assert_eq!(resp.last_id.as_deref(), Some("msgbatch_1"));
    }

    #[test]
    fn list_batches_response_empty() {
        let json = serde_json::json!({
            "data": [],
            "has_more": false
        });
        let resp: ListBatchesResponse = serde_json::from_value(json).unwrap();
        assert!(resp.data.is_empty());
        assert!(!resp.has_more);
    }

    #[test]
    fn batch_result_succeeded() {
        let json = serde_json::json!({
            "custom_id": "req-1",
            "result": {
                "type": "succeeded",
                "message": {
                    "id": "msg_1",
                    "model": "claude-haiku-4-5",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello!"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5}
                }
            }
        });
        let result: BatchResult = serde_json::from_value(json).unwrap();
        assert_eq!(result.custom_id, "req-1");
        match result.result {
            BatchResultBody::Succeeded { message } => {
                assert_eq!(message.text(), Some("Hello!"));
            }
            _ => panic!("expected Succeeded"),
        }
    }

    #[test]
    fn batch_result_errored() {
        let json = serde_json::json!({
            "custom_id": "req-2",
            "result": {
                "type": "errored",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Bad request"
                }
            }
        });
        let result: BatchResult = serde_json::from_value(json).unwrap();
        match result.result {
            BatchResultBody::Errored { error } => {
                let err = error.unwrap();
                assert_eq!(err.error_type, "invalid_request_error");
                assert_eq!(err.message, "Bad request");
            }
            _ => panic!("expected Errored"),
        }
    }

    #[test]
    fn batch_result_canceled() {
        let json = serde_json::json!({
            "custom_id": "req-3",
            "result": {"type": "canceled"}
        });
        let result: BatchResult = serde_json::from_value(json).unwrap();
        assert!(matches!(result.result, BatchResultBody::Canceled {}));
    }

    #[test]
    fn batch_result_expired() {
        let json = serde_json::json!({
            "custom_id": "req-4",
            "result": {"type": "expired"}
        });
        let result: BatchResult = serde_json::from_value(json).unwrap();
        assert!(matches!(result.result, BatchResultBody::Expired {}));
    }

    #[test]
    fn batch_error_round_trip() {
        let err = BatchError {
            error_type: "api_error".into(),
            message: "Internal error".into(),
        };
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["type"], "api_error");
        assert_eq!(json["message"], "Internal error");
        let back: BatchError = serde_json::from_value(json).unwrap();
        assert_eq!(back.error_type, "api_error");
    }

    #[test]
    fn list_params_default() {
        let params = ListBatchesParams::default();
        assert!(params.after_id.is_none());
        assert!(params.before_id.is_none());
        assert!(params.limit.is_none());
    }
}
