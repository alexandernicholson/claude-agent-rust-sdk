//! Client for the Claude Message Batches API.

use std::time::Duration;

use tracing::debug;

use crate::client::ClaudeClient;
use crate::error::ClaudeError;
use crate::types::batch::{
    BatchResponse, BatchResult, BatchStatus, CreateBatchRequest,
    ListBatchesParams, ListBatchesResponse,
};

/// Client for creating and managing message batches.
///
/// Obtain an instance via [`ClaudeClient::batches`].
#[derive(Debug)]
pub struct BatchClient<'a> {
    client: &'a ClaudeClient,
}

impl<'a> BatchClient<'a> {
    pub(crate) fn new(client: &'a ClaudeClient) -> Self {
        Self { client }
    }

    /// Create a new message batch.
    ///
    /// `POST /v1/messages/batches`
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::ApiError`] if the API returns a non-success status,
    /// [`ClaudeError::NetworkError`] on connection failures, or
    /// [`ClaudeError::SerializationError`] if the response cannot be parsed.
    pub async fn create(
        &self,
        request: &CreateBatchRequest,
    ) -> Result<BatchResponse, ClaudeError> {
        if let Some(transport) = self.client.transport() {
            return transport.create_batch(request).await;
        }

        let url = format!("{}/v1/messages/batches", self.client.base_url());
        let headers = self.client.build_headers();

        debug!(url = %url, "creating batch");

        let response = self
            .client
            .http()
            .post(&url)
            .headers(headers)
            .json(request)
            .send()
            .await?;

        Self::handle_response(response).await
    }

    /// Retrieve the current status of a batch.
    ///
    /// `GET /v1/messages/batches/{batch_id}`
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::ApiError`] if the API returns a non-success status,
    /// [`ClaudeError::NetworkError`] on connection failures, or
    /// [`ClaudeError::SerializationError`] if the response cannot be parsed.
    pub async fn retrieve(&self, batch_id: &str) -> Result<BatchResponse, ClaudeError> {
        if let Some(transport) = self.client.transport() {
            return transport.retrieve_batch(batch_id).await;
        }

        let url = format!(
            "{}/v1/messages/batches/{}",
            self.client.base_url(),
            batch_id
        );
        let headers = self.client.build_headers();

        debug!(url = %url, "retrieving batch");

        let response = self
            .client
            .http()
            .get(&url)
            .headers(headers)
            .send()
            .await?;

        Self::handle_response(response).await
    }

    /// List all message batches in the workspace.
    ///
    /// `GET /v1/messages/batches`
    ///
    /// Returns batches in reverse chronological order (most recent first).
    /// Use `params` for pagination.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::ApiError`] if the API returns a non-success status,
    /// [`ClaudeError::NetworkError`] on connection failures, or
    /// [`ClaudeError::SerializationError`] if the response cannot be parsed.
    pub async fn list(
        &self,
        params: &ListBatchesParams,
    ) -> Result<ListBatchesResponse, ClaudeError> {
        if let Some(transport) = self.client.transport() {
            return transport.list_batches(params).await;
        }

        let mut url = format!("{}/v1/messages/batches", self.client.base_url());
        let headers = self.client.build_headers();

        // Build query parameters
        let mut query_parts: Vec<String> = Vec::new();
        if let Some(ref after_id) = params.after_id {
            query_parts.push(format!("after_id={after_id}"));
        }
        if let Some(ref before_id) = params.before_id {
            query_parts.push(format!("before_id={before_id}"));
        }
        if let Some(limit) = params.limit {
            query_parts.push(format!("limit={limit}"));
        }
        if !query_parts.is_empty() {
            url.push('?');
            url.push_str(&query_parts.join("&"));
        }

        debug!(url = %url, "listing batches");

        let response = self
            .client
            .http()
            .get(&url)
            .headers(headers)
            .send()
            .await?;

        Self::handle_response(response).await
    }

    /// Fetch the results of a completed batch.
    ///
    /// This first retrieves the batch to obtain the `results_url`, then
    /// downloads and parses the JSONL results file.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::InvalidConfig`] if the batch has no `results_url`,
    /// [`ClaudeError::ApiError`] if the API returns a non-success status,
    /// [`ClaudeError::NetworkError`] on connection failures, or
    /// [`ClaudeError::SerializationError`] if the response cannot be parsed.
    pub async fn results(&self, batch_id: &str) -> Result<Vec<BatchResult>, ClaudeError> {
        if let Some(transport) = self.client.transport() {
            return transport.batch_results(batch_id).await;
        }

        let batch = self.retrieve(batch_id).await?;

        let results_url = batch.results_url.ok_or_else(|| {
            ClaudeError::InvalidConfig(format!(
                "batch {batch_id} has no results_url (status: {:?})",
                batch.processing_status
            ))
        })?;

        let headers = self.client.build_headers();

        debug!(url = %results_url, "fetching batch results");

        let response = self
            .client
            .http()
            .get(&results_url)
            .headers(headers)
            .send()
            .await?;

        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let body = response.text().await.unwrap_or_default();
            return Err(ClaudeError::ApiError {
                status,
                error_type: "batch_results_error".into(),
                message: body,
            });
        }

        let body = response.text().await?;
        let mut results = Vec::new();
        for line in body.lines() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<BatchResult>(line) {
                Ok(result) => results.push(result),
                Err(e) => {
                    // Log the raw line for debugging but don't fail the entire batch
                    debug!(
                        error = %e,
                        line = &line[..line.len().min(500)],
                        "failed to parse batch result line, skipping"
                    );
                }
            }
        }

        Ok(results)
    }

    /// Poll a batch until it reaches a terminal state (`Ended`, `Canceled`,
    /// or `Expired`).
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::BatchTimeout`] if the batch has not finished
    /// after ~24 hours of polling (the API's maximum batch lifetime).
    /// Also propagates errors from [`retrieve`](Self::retrieve).
    pub async fn poll_until_complete(
        &self,
        batch_id: &str,
        poll_interval: Duration,
    ) -> Result<BatchResponse, ClaudeError> {
        if let Some(transport) = self.client.transport() {
            return transport.poll_batch_until_complete(batch_id, poll_interval).await;
        }

        // Safety cap: stop polling after 24 hours.
        // The result always fits in u64 since 24h / 1ms = 86_400_000.
        let max_iterations = u64::try_from(
            Duration::from_secs(24 * 60 * 60).as_millis()
                / poll_interval.as_millis().max(1),
        )
        .unwrap_or(u64::MAX);

        for _ in 0..max_iterations {
            let batch = self.retrieve(batch_id).await?;

            debug!(
                batch_id = %batch_id,
                status = ?batch.processing_status,
                "polled batch"
            );

            if batch.processing_status == BatchStatus::Ended {
                return Ok(batch);
            }

            tokio::time::sleep(poll_interval).await;
        }

        Err(ClaudeError::BatchTimeout {
            batch_id: batch_id.to_string(),
        })
    }

    /// Cancel a batch that is still in progress.
    ///
    /// `POST /v1/messages/batches/{batch_id}/cancel`
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::ApiError`] if the API returns a non-success status,
    /// [`ClaudeError::NetworkError`] on connection failures, or
    /// [`ClaudeError::SerializationError`] if the response cannot be parsed.
    pub async fn cancel(&self, batch_id: &str) -> Result<BatchResponse, ClaudeError> {
        if let Some(transport) = self.client.transport() {
            return transport.cancel_batch(batch_id).await;
        }

        let url = format!(
            "{}/v1/messages/batches/{}/cancel",
            self.client.base_url(),
            batch_id
        );
        let headers = self.client.build_headers();

        debug!(url = %url, "canceling batch");

        let response = self
            .client
            .http()
            .post(&url)
            .headers(headers)
            .send()
            .await?;

        Self::handle_response(response).await
    }

    // ----- internal ---------------------------------------------------------

    async fn handle_response<T: serde::de::DeserializeOwned>(
        response: reqwest::Response,
    ) -> Result<T, ClaudeError> {
        let status = response.status().as_u16();
        let body = response.text().await?;

        if !(200..300).contains(&status) {
            debug!(status, body = %body, "batch API returned error");

            if let Ok(api_err) = serde_json::from_str::<crate::types::ApiErrorBody>(&body) {
                return Err(ClaudeError::ApiError {
                    status,
                    error_type: api_err.error.error_type,
                    message: api_err.error.message,
                });
            }
            return Err(ClaudeError::ApiError {
                status,
                error_type: "unknown".into(),
                message: body,
            });
        }

        let value: T = serde_json::from_str(&body)?;
        Ok(value)
    }
}
