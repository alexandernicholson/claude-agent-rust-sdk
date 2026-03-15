//! Client for the Claude Message Batches API.

use std::time::Duration;

use tracing::debug;

use crate::client::ClaudeClient;
use crate::error::ClaudeError;
use crate::types::batch::{
    BatchResponse, BatchResult, BatchStatus, CreateBatchRequest,
};

/// Client for creating and managing message batches.
///
/// Obtain an instance via [`ClaudeClient::batches`].
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
    pub async fn create(
        &self,
        request: &CreateBatchRequest,
    ) -> Result<BatchResponse, ClaudeError> {
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
    pub async fn retrieve(&self, batch_id: &str) -> Result<BatchResponse, ClaudeError> {
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

    /// Fetch the results of a completed batch.
    ///
    /// This first retrieves the batch to obtain the `results_url`, then
    /// downloads and parses the JSONL results file.
    pub async fn results(&self, batch_id: &str) -> Result<Vec<BatchResult>, ClaudeError> {
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
            let result: BatchResult = serde_json::from_str(line)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Poll a batch until it reaches a terminal state (`Ended`, `Canceled`,
    /// or `Expired`).
    ///
    /// Returns [`ClaudeError::BatchTimeout`] if the batch has not finished
    /// after ~24 hours of polling (the API's maximum batch lifetime).
    pub async fn poll_until_complete(
        &self,
        batch_id: &str,
        poll_interval: Duration,
    ) -> Result<BatchResponse, ClaudeError> {
        // Safety cap: stop polling after 24 hours.
        let max_iterations = (Duration::from_secs(24 * 60 * 60).as_millis()
            / poll_interval.as_millis().max(1))
            as u64;

        for _ in 0..max_iterations {
            let batch = self.retrieve(batch_id).await?;

            debug!(
                batch_id = %batch_id,
                status = ?batch.processing_status,
                "polled batch"
            );

            match batch.processing_status {
                BatchStatus::Ended
                | BatchStatus::Canceled
                | BatchStatus::Expired => return Ok(batch),
                _ => {}
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
    pub async fn cancel(&self, batch_id: &str) -> Result<BatchResponse, ClaudeError> {
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
