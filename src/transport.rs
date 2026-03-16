//! Transport abstraction for the Claude SDK.
//!
//! The [`Transport`] trait defines how API operations are executed. The default
//! The default HTTP implementation (inside `ClaudeClient`) sends requests to
//! the Claude HTTP API. Custom transports
//! can route operations through alternative backends (e.g. a CLI tool, a mock,
//! or a proxy).
//!
//! # Implementing a custom transport
//!
//! Implement the [`Transport`] trait and pass it to
//! [`ClaudeClient::with_transport`](crate::client::ClaudeClient::with_transport).
//! All methods have default implementations
//! that return [`ClaudeError::Unsupported`], so you only need to implement the
//! operations your transport supports.
//!
//! ```ignore
//! use claude_agent_rust_sdk::transport::Transport;
//!
//! struct MyTransport;
//!
//! #[async_trait::async_trait]
//! impl Transport for MyTransport {
//!     async fn create_message(
//!         &self,
//!         request: &CreateMessageRequest,
//!     ) -> Result<CreateMessageResponse, ClaudeError> {
//!         // custom implementation
//!     }
//! }
//!
//! let client = ClaudeClient::with_transport(MyTransport);
//! ```

use std::time::Duration;

use async_trait::async_trait;

use crate::error::ClaudeError;
use crate::streaming::SseStream;
use crate::types::{
    CountTokensRequest, CountTokensResponse, CreateMessageRequest, CreateMessageResponse,
};
use crate::types::batch::{
    BatchResponse, BatchResult, BatchStatus, CreateBatchRequest, ListBatchesParams,
    ListBatchesResponse,
};

/// Trait abstracting how Claude API operations are executed.
///
/// The default implementations return [`ClaudeError::Unsupported`] for all
/// operations. Implement only the methods your transport supports.
///
/// # Operations
///
/// | Method | HTTP Equivalent |
/// |--------|----------------|
/// | [`create_message`](Transport::create_message) | `POST /v1/messages` |
/// | [`create_message_stream`](Transport::create_message_stream) | `POST /v1/messages` (streaming) |
/// | [`count_tokens`](Transport::count_tokens) | `POST /v1/messages/count_tokens` |
/// | [`create_batch`](Transport::create_batch) | `POST /v1/messages/batches` |
/// | [`retrieve_batch`](Transport::retrieve_batch) | `GET /v1/messages/batches/{id}` |
/// | [`list_batches`](Transport::list_batches) | `GET /v1/messages/batches` |
/// | [`batch_results`](Transport::batch_results) | `GET {results_url}` |
/// | [`cancel_batch`](Transport::cancel_batch) | `POST /v1/messages/batches/{id}/cancel` |
/// | [`poll_batch_until_complete`](Transport::poll_batch_until_complete) | Polling loop over `retrieve_batch` |
#[async_trait]
pub trait Transport: Send + Sync + std::fmt::Debug {
    /// Send a message request and return the full response.
    async fn create_message(
        &self,
        _request: &CreateMessageRequest,
    ) -> Result<CreateMessageResponse, ClaudeError> {
        Err(ClaudeError::Unsupported("create_message".into()))
    }

    /// Send a message request with streaming and return an async event stream.
    async fn create_message_stream(
        &self,
        _request: &CreateMessageRequest,
    ) -> Result<SseStream, ClaudeError> {
        Err(ClaudeError::Unsupported("create_message_stream".into()))
    }

    /// Count the tokens in a message request without sending it.
    async fn count_tokens(
        &self,
        _request: &CountTokensRequest,
    ) -> Result<CountTokensResponse, ClaudeError> {
        Err(ClaudeError::Unsupported("count_tokens".into()))
    }

    /// Create a new message batch.
    async fn create_batch(
        &self,
        _request: &CreateBatchRequest,
    ) -> Result<BatchResponse, ClaudeError> {
        Err(ClaudeError::Unsupported("create_batch".into()))
    }

    /// Retrieve the current status of a batch.
    async fn retrieve_batch(&self, _batch_id: &str) -> Result<BatchResponse, ClaudeError> {
        Err(ClaudeError::Unsupported("retrieve_batch".into()))
    }

    /// List all message batches.
    async fn list_batches(
        &self,
        _params: &ListBatchesParams,
    ) -> Result<ListBatchesResponse, ClaudeError> {
        Err(ClaudeError::Unsupported("list_batches".into()))
    }

    /// Fetch the results of a completed batch.
    async fn batch_results(&self, _batch_id: &str) -> Result<Vec<BatchResult>, ClaudeError> {
        Err(ClaudeError::Unsupported("batch_results".into()))
    }

    /// Cancel a batch that is still in progress.
    async fn cancel_batch(&self, _batch_id: &str) -> Result<BatchResponse, ClaudeError> {
        Err(ClaudeError::Unsupported("cancel_batch".into()))
    }

    /// Poll a batch until it reaches a terminal state.
    ///
    /// The default implementation loops over [`retrieve_batch`](Transport::retrieve_batch)
    /// with the given interval. Transports that process batches synchronously
    /// can override this to return immediately.
    async fn poll_batch_until_complete(
        &self,
        batch_id: &str,
        poll_interval: Duration,
    ) -> Result<BatchResponse, ClaudeError> {
        // The result always fits in u64 since 24h / 1ms = 86_400_000.
        let max_iterations = u64::try_from(
            Duration::from_secs(24 * 60 * 60).as_millis()
                / poll_interval.as_millis().max(1),
        )
        .unwrap_or(u64::MAX);

        for _ in 0..max_iterations {
            let batch = self.retrieve_batch(batch_id).await?;

            if batch.processing_status == BatchStatus::Ended {
                return Ok(batch);
            }

            tokio::time::sleep(poll_interval).await;
        }

        Err(ClaudeError::BatchTimeout {
            batch_id: batch_id.to_string(),
        })
    }
}
