//! HTTP client for the Claude Messages API.

pub mod builder;

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use tracing::debug;

use crate::error::ClaudeError;
use crate::streaming::SseStream;
use crate::types::{
    ApiErrorBody, CountTokensRequest, CountTokensResponse, CreateMessageRequest,
    CreateMessageResponse,
};
use crate::batch::BatchClient;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";

/// Authentication strategy.
#[derive(Debug, Clone)]
enum AuthMethod {
    /// Classic API key sent via the `x-api-key` header.
    ApiKey(String),
    /// OAuth / service-account token sent via `Authorization: Bearer`.
    BearerToken(String),
}

/// The main entry point for calling the Claude API.
///
/// Create an instance with [`ClaudeClient::new`] (API key) or
/// [`ClaudeClient::with_oauth_token`] (Bearer token), then call
/// [`create_message`](ClaudeClient::create_message) or use the
/// [`MessageBuilder`](builder::MessageBuilder) via [`messages`](ClaudeClient::messages).
#[derive(Debug, Clone)]
pub struct ClaudeClient {
    http: reqwest::Client,
    base_url: String,
    auth: AuthMethod,
    /// Extra beta feature headers (e.g. `"interleaved-thinking-2025-05-14"`).
    beta_features: Vec<String>,
}

impl ClaudeClient {
    // ----- constructors -----------------------------------------------------

    /// Create a client that authenticates with a classic API key.
    pub fn new(api_key: &str) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            auth: AuthMethod::ApiKey(api_key.to_string()),
            beta_features: Vec::new(),
        }
    }

    /// Create a client that authenticates with an OAuth / Bearer token.
    pub fn with_oauth_token(token: &str) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            auth: AuthMethod::BearerToken(token.to_string()),
            beta_features: Vec::new(),
        }
    }

    /// Override the base URL (useful for testing or proxying).
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.trim_end_matches('/').to_string();
        self
    }

    /// Add one or more `anthropic-beta` feature flags.
    ///
    /// Multiple calls accumulate features. They are sent as a
    /// comma-separated value in the `anthropic-beta` header.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let client = ClaudeClient::new("sk-ant-...")
    ///     .with_beta("interleaved-thinking-2025-05-14")
    ///     .with_beta("files-api-2025-04-14");
    /// ```
    pub fn with_beta(mut self, feature: &str) -> Self {
        self.beta_features.push(feature.to_string());
        self
    }

    // ----- sub-clients ------------------------------------------------------

    /// Return a [`MessageBuilder`](builder::MessageBuilder) for ergonomic
    /// request construction.
    pub fn messages(&self) -> builder::MessageBuilder<'_> {
        builder::MessageBuilder::new(self)
    }

    /// Return a [`BatchClient`] for interacting with the Message Batches
    /// API.
    pub fn batches(&self) -> BatchClient<'_> {
        BatchClient::new(self)
    }

    // ----- core request methods ---------------------------------------------

    /// Send a message-creation request and return the full response.
    pub async fn create_message(
        &self,
        request: &CreateMessageRequest,
    ) -> Result<CreateMessageResponse, ClaudeError> {
        let url = format!("{}/v1/messages", self.base_url);
        let headers = self.build_headers();

        debug!(url = %url, "sending create_message request");

        let response = self
            .http
            .post(&url)
            .headers(headers)
            .json(request)
            .send()
            .await?;

        let status = response.status().as_u16();

        if !(200..300).contains(&status) {
            let body = response.text().await.unwrap_or_default();
            debug!(status, body = %body, "API returned error");

            // Try to parse the structured error; fall back to the raw body.
            if let Ok(api_err) = serde_json::from_str::<ApiErrorBody>(&body) {
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

        let body = response.text().await?;
        debug!(body = %body, "received response");

        let msg: CreateMessageResponse = serde_json::from_str(&body)?;
        Ok(msg)
    }

    /// Send a message-creation request with `stream: true` and return an
    /// async stream of SSE events.
    ///
    /// The request's `stream` field is forced to `true`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use futures::stream::StreamExt;
    ///
    /// let request = /* build a CreateMessageRequest */;
    /// let mut stream = client.create_message_stream(&request).await?;
    ///
    /// while let Some(event) = stream.next().await {
    ///     match event? {
    ///         StreamEvent::ContentBlockDelta { delta, .. } => { /* handle delta */ }
    ///         StreamEvent::MessageStop {} => break,
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub async fn create_message_stream(
        &self,
        request: &CreateMessageRequest,
    ) -> Result<SseStream, ClaudeError> {
        let url = format!("{}/v1/messages", self.base_url);
        let headers = self.build_headers();

        // Force stream: true
        let mut req = request.clone();
        req.stream = Some(true);

        debug!(url = %url, "sending streaming create_message request");

        let response = self
            .http
            .post(&url)
            .headers(headers)
            .json(&req)
            .send()
            .await?;

        let status = response.status().as_u16();

        if !(200..300).contains(&status) {
            let body = response.text().await.unwrap_or_default();
            debug!(status, body = %body, "API returned error");

            if let Ok(api_err) = serde_json::from_str::<ApiErrorBody>(&body) {
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

        Ok(SseStream::from_response(response))
    }

    /// Count the tokens in a message request without sending it.
    ///
    /// `POST /v1/messages/count_tokens`
    pub async fn count_tokens(
        &self,
        request: &CountTokensRequest,
    ) -> Result<CountTokensResponse, ClaudeError> {
        let url = format!("{}/v1/messages/count_tokens", self.base_url);
        let headers = self.build_headers();

        debug!(url = %url, "sending count_tokens request");

        let response = self
            .http
            .post(&url)
            .headers(headers)
            .json(request)
            .send()
            .await?;

        let status = response.status().as_u16();

        if !(200..300).contains(&status) {
            let body = response.text().await.unwrap_or_default();
            debug!(status, body = %body, "count_tokens API returned error");

            if let Ok(api_err) = serde_json::from_str::<ApiErrorBody>(&body) {
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

        let body = response.text().await?;
        let resp: CountTokensResponse = serde_json::from_str(&body)?;
        Ok(resp)
    }

    // ----- internal helpers -------------------------------------------------

    /// Construct the header map common to every request.
    pub(crate) fn build_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(API_VERSION),
        );

        match &self.auth {
            AuthMethod::ApiKey(key) => {
                // unwrap is safe: API keys are ASCII.
                headers.insert("x-api-key", HeaderValue::from_str(key).expect("invalid API key characters"));
            }
            AuthMethod::BearerToken(token) => {
                let value = format!("Bearer {token}");
                headers.insert(AUTHORIZATION, HeaderValue::from_str(&value).expect("invalid token characters"));
            }
        }

        // Add beta features header if any are configured
        if !self.beta_features.is_empty() {
            let beta_value = self.beta_features.join(",");
            if let Ok(hv) = HeaderValue::from_str(&beta_value) {
                headers.insert("anthropic-beta", hv);
            }
        }

        headers
    }

    /// The configured base URL (used by sub-clients).
    pub(crate) fn base_url(&self) -> &str {
        &self.base_url
    }

    /// A reference to the inner `reqwest::Client` (used by sub-clients).
    pub(crate) fn http(&self) -> &reqwest::Client {
        &self.http
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_client_uses_api_key_auth() {
        let client = ClaudeClient::new("sk-ant-test");
        let headers = client.build_headers();
        assert_eq!(
            headers.get("x-api-key").unwrap().to_str().unwrap(),
            "sk-ant-test"
        );
        assert!(headers.get("authorization").is_none());
    }

    #[test]
    fn oauth_client_uses_bearer_auth() {
        let client = ClaudeClient::with_oauth_token("eyJhbGciOi");
        let headers = client.build_headers();
        assert!(headers.get("x-api-key").is_none());
        assert_eq!(
            headers.get("authorization").unwrap().to_str().unwrap(),
            "Bearer eyJhbGciOi"
        );
    }

    #[test]
    fn custom_base_url() {
        let client = ClaudeClient::new("key").with_base_url("https://proxy.example.com/");
        assert_eq!(client.base_url(), "https://proxy.example.com");
    }

    #[test]
    fn version_header_present() {
        let client = ClaudeClient::new("key");
        let headers = client.build_headers();
        assert_eq!(
            headers.get("anthropic-version").unwrap().to_str().unwrap(),
            "2023-06-01"
        );
    }

    #[test]
    fn content_type_header() {
        let client = ClaudeClient::new("key");
        let headers = client.build_headers();
        assert_eq!(
            headers.get("content-type").unwrap().to_str().unwrap(),
            "application/json"
        );
    }

    #[test]
    fn beta_features_header() {
        let client = ClaudeClient::new("key")
            .with_beta("interleaved-thinking-2025-05-14")
            .with_beta("files-api-2025-04-14");
        let headers = client.build_headers();
        let beta = headers.get("anthropic-beta").unwrap().to_str().unwrap();
        assert!(beta.contains("interleaved-thinking-2025-05-14"));
        assert!(beta.contains("files-api-2025-04-14"));
        assert!(beta.contains(','));
    }

    #[test]
    fn no_beta_header_when_empty() {
        let client = ClaudeClient::new("key");
        let headers = client.build_headers();
        assert!(headers.get("anthropic-beta").is_none());
    }
}
