//! HTTP client for the Claude Messages API.

pub mod builder;

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use tracing::debug;

use crate::error::ClaudeError;
use crate::types::{ApiErrorBody, CreateMessageRequest, CreateMessageResponse};
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
}

impl ClaudeClient {
    // ----- constructors -----------------------------------------------------

    /// Create a client that authenticates with a classic API key.
    pub fn new(api_key: &str) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            auth: AuthMethod::ApiKey(api_key.to_string()),
        }
    }

    /// Create a client that authenticates with an OAuth / Bearer token.
    pub fn with_oauth_token(token: &str) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            auth: AuthMethod::BearerToken(token.to_string()),
        }
    }

    /// Override the base URL (useful for testing or proxying).
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.trim_end_matches('/').to_string();
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
