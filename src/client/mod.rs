//! HTTP client for the Claude Messages API.
//!
//! This module provides [`ClaudeClient`], the main entry point for the SDK.
//! Use it to create messages, stream responses, count tokens, and access the
//! batch API.
//!
//! # Authentication
//!
//! - [`ClaudeClient::new`] -- authenticate with a classic API key (`x-api-key` header).
//! - [`ClaudeClient::with_oauth_token`] -- authenticate with an OAuth / Bearer token.
//! - [`ClaudeClient::with_transport`] -- bypass HTTP entirely with a custom
//!   [`Transport`] implementation.
//!
//! # Sub-clients
//!
//! - [`ClaudeClient::messages`] returns a [`MessageBuilder`](builder::MessageBuilder)
//!   for fluent request construction.
//! - [`ClaudeClient::batches`] returns a [`BatchClient`]
//!   for the Message Batches API.

pub mod builder;

use std::sync::Arc;

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use tracing::debug;

use crate::batch::BatchClient;
use crate::error::ClaudeError;
use crate::streaming::{RawSseStream, SseStream};
use crate::transport::Transport;
use crate::types::{
    ApiErrorBody, CountTokensRequest, CountTokensResponse, CreateMessageRequest,
    CreateMessageResponse,
};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2026-10-21";

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
///
/// For custom backends, use [`ClaudeClient::with_transport`] to provide a
/// [`Transport`] implementation that intercepts all API operations.
#[derive(Debug, Clone)]
pub struct ClaudeClient {
    http: reqwest::Client,
    base_url: String,
    auth: AuthMethod,
    /// Extra beta feature headers (e.g. `"interleaved-thinking-2025-05-14"`).
    beta_features: Vec<String>,
    /// `anthropic-workspace-id` header sent with every request.
    workspace_id: Option<HeaderValue>,
    /// `anthropic-user-profile-id` header sent with message creation, token
    /// counting, and batch creation.
    user_profile_id: Option<HeaderValue>,
    /// `anthropic-version` header value.
    api_version: HeaderValue,
    /// Additional headers sent with every request.
    extra_headers: HeaderMap,
    /// Optional custom transport. When set, all operations delegate to it.
    transport: Option<Arc<dyn Transport>>,
}

impl ClaudeClient {
    // ----- constructors -----------------------------------------------------

    /// Create a client that authenticates with a classic API key.
    #[must_use]
    pub fn new(api_key: &str) -> Self {
        Self::with_auth(AuthMethod::ApiKey(api_key.to_string()), None)
    }

    /// Create a client that authenticates with an OAuth / Bearer token.
    #[must_use]
    pub fn with_oauth_token(token: &str) -> Self {
        Self::with_auth(AuthMethod::BearerToken(token.to_string()), None)
    }

    /// Create a client backed by a custom [`Transport`].
    ///
    /// All API operations (messages, batches, streaming, token counting)
    /// will be routed through the transport instead of HTTP.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use claude_agent_rust_sdk::client::ClaudeClient;
    /// use my_transport::CliTransport;
    ///
    /// let client = ClaudeClient::with_transport(CliTransport::new());
    /// let response = client.create_message(&request).await?;
    /// ```
    #[must_use]
    pub fn with_transport<T: Transport + 'static>(transport: T) -> Self {
        Self::with_auth(AuthMethod::ApiKey(String::new()), Some(Arc::new(transport)))
    }

    fn with_auth(auth: AuthMethod, transport: Option<Arc<dyn Transport>>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            auth,
            beta_features: Vec::new(),
            workspace_id: None,
            user_profile_id: None,
            api_version: HeaderValue::from_static(API_VERSION),
            extra_headers: HeaderMap::new(),
            transport,
        }
    }

    /// Override the base URL (useful for testing or proxying).
    #[must_use]
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
    #[must_use]
    pub fn with_beta(mut self, feature: &str) -> Self {
        self.beta_features.push(feature.to_string());
        self
    }

    /// Override the `anthropic-version` header (default `2026-10-21`).
    ///
    /// Useful for Anthropic-compatible providers that only accept an older
    /// API version.
    ///
    /// # Panics
    ///
    /// Panics if `version` is not a valid HTTP header value.
    #[must_use]
    pub fn with_api_version(mut self, version: &str) -> Self {
        self.api_version = HeaderValue::from_str(version).expect("invalid API version characters");
        self
    }

    /// Send an additional header with every request, e.g. a gateway's own
    /// routing or authentication header. A later call with the same name
    /// replaces the earlier value; headers the client sets itself take
    /// precedence.
    ///
    /// # Panics
    ///
    /// Panics if `name` or `value` is not a valid HTTP header.
    #[must_use]
    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        let name =
            reqwest::header::HeaderName::from_bytes(name.as_bytes()).expect("invalid header name");
        let value = HeaderValue::from_str(value).expect("invalid header value");
        self.extra_headers.insert(name, value);
        self
    }

    /// Select the Workspace for every request (`anthropic-workspace-id`
    /// header), e.g. `"wrkspc_011CZkZaBF1tNoB5wlCeusgy"`.
    ///
    /// Only needed for credentials that can act on more than one Workspace.
    /// A Workspace-scoped credential may omit it; if sent, it must match.
    ///
    /// # Panics
    ///
    /// Panics if `workspace_id` is not a valid HTTP header value.
    #[must_use]
    pub fn with_workspace_id(mut self, workspace_id: &str) -> Self {
        self.workspace_id =
            Some(HeaderValue::from_str(workspace_id).expect("invalid workspace ID characters"));
        self
    }

    /// Attribute requests to a user profile (`anthropic-user-profile-id`
    /// header) when acting on behalf of a party other than your
    /// organization.
    ///
    /// Sent with message creation (including streaming), token counting,
    /// and batch creation.
    ///
    /// # Panics
    ///
    /// Panics if `user_profile_id` is not a valid HTTP header value.
    #[must_use]
    pub fn with_user_profile_id(mut self, user_profile_id: &str) -> Self {
        self.user_profile_id = Some(
            HeaderValue::from_str(user_profile_id).expect("invalid user profile ID characters"),
        );
        self
    }

    // ----- sub-clients ------------------------------------------------------

    /// Return a [`MessageBuilder`](builder::MessageBuilder) for ergonomic
    /// request construction.
    #[must_use]
    pub fn messages(&self) -> builder::MessageBuilder<'_> {
        builder::MessageBuilder::new(self)
    }

    /// Return a [`BatchClient`] for interacting with the Message Batches
    /// API.
    #[must_use]
    pub fn batches(&self) -> BatchClient<'_> {
        BatchClient::new(self)
    }

    // ----- core request methods ---------------------------------------------

    /// Send a message-creation request and return the full response.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::ApiError`] if the API returns a non-success status,
    /// [`ClaudeError::NetworkError`] on connection failures, or
    /// [`ClaudeError::SerializationError`] if the response cannot be parsed.
    pub async fn create_message(
        &self,
        request: &CreateMessageRequest,
    ) -> Result<CreateMessageResponse, ClaudeError> {
        if let Some(ref transport) = self.transport {
            return transport.create_message(request).await;
        }

        let url = format!("{}/v1/messages", self.base_url);
        let headers = self.build_attributed_headers();

        debug!(url = %url, "sending create_message request");

        let response = self
            .http
            .post(&url)
            .headers(headers)
            .json(request)
            .send()
            .await?;
        let response = Self::check_status(response).await?;

        let body = response.text().await?;
        debug!(body = %body, "received response");

        let msg: CreateMessageResponse = serde_json::from_str(&body)?;
        Ok(msg)
    }

    /// Send an untyped message-creation request and return the untyped
    /// response body.
    ///
    /// Use this for request fields or content block types the typed
    /// [`CreateMessageRequest`] / [`CreateMessageResponse`] do not model yet:
    /// the JSON is sent and returned verbatim, so content such as thinking
    /// signatures or server-tool result blocks round-trips losslessly.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::ApiError`] if the API returns a non-success
    /// status, [`ClaudeError::NetworkError`] on connection failures,
    /// [`ClaudeError::SerializationError`] if the body is not JSON, or
    /// [`ClaudeError::Unsupported`] when the client uses a custom transport.
    pub async fn create_message_raw(
        &self,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, ClaudeError> {
        let response = self.send_raw(body, false).await?;
        let text = response.text().await?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Send an untyped message-creation request with `stream: true` and
    /// return the untyped SSE events.
    ///
    /// The body's `stream` field is forced to `true`. See
    /// [`create_message_raw`](Self::create_message_raw) for when to use the
    /// raw methods.
    ///
    /// # Errors
    ///
    /// Same as [`create_message_raw`](Self::create_message_raw).
    pub async fn create_message_stream_raw(
        &self,
        body: &serde_json::Value,
    ) -> Result<RawSseStream, ClaudeError> {
        let response = self.send_raw(body, true).await?;
        Ok(SseStream::from_response(response))
    }

    async fn send_raw(
        &self,
        body: &serde_json::Value,
        stream: bool,
    ) -> Result<reqwest::Response, ClaudeError> {
        if self.transport.is_some() {
            return Err(ClaudeError::Unsupported(
                "raw message requests are not routed through custom transports".into(),
            ));
        }
        let mut body = body.clone();
        if stream {
            match body.as_object_mut() {
                Some(object) => {
                    object.insert("stream".into(), serde_json::Value::Bool(true));
                }
                None => {
                    return Err(ClaudeError::InvalidConfig(
                        "raw message request body must be a JSON object".into(),
                    ))
                }
            }
        }
        let url = format!("{}/v1/messages", self.base_url);
        debug!(url = %url, stream, "sending raw create_message request");
        let response = self
            .http
            .post(&url)
            .headers(self.build_attributed_headers())
            .json(&body)
            .send()
            .await?;
        Self::check_status(response).await
    }

    /// Map a non-2xx response to [`ClaudeError::ApiError`].
    async fn check_status(response: reqwest::Response) -> Result<reqwest::Response, ClaudeError> {
        let status = response.status().as_u16();
        if (200..300).contains(&status) {
            return Ok(response);
        }
        let body = response.text().await.unwrap_or_default();
        debug!(status, body = %body, "API returned error");
        if let Ok(api_err) = serde_json::from_str::<ApiErrorBody>(&body) {
            return Err(ClaudeError::ApiError {
                status,
                error_type: api_err.error.error_type,
                message: api_err.error.message,
            });
        }
        Err(ClaudeError::ApiError {
            status,
            error_type: "unknown".into(),
            message: body,
        })
    }

    /// Send a message-creation request with `stream: true` and return an
    /// async stream of SSE events.
    ///
    /// The request's `stream` field is forced to `true`.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::ApiError`] if the API returns a non-success status,
    /// [`ClaudeError::NetworkError`] on connection failures, or
    /// [`ClaudeError::SerializationError`] if the response cannot be parsed.
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
        if let Some(ref transport) = self.transport {
            return transport.create_message_stream(request).await;
        }

        let url = format!("{}/v1/messages", self.base_url);
        let headers = self.build_attributed_headers();

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
        let response = Self::check_status(response).await?;

        Ok(SseStream::from_response(response))
    }

    /// Count the tokens in a message request without sending it.
    ///
    /// `POST /v1/messages/count_tokens`
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::ApiError`] if the API returns a non-success status,
    /// [`ClaudeError::NetworkError`] on connection failures, or
    /// [`ClaudeError::SerializationError`] if the response cannot be parsed.
    pub async fn count_tokens(
        &self,
        request: &CountTokensRequest,
    ) -> Result<CountTokensResponse, ClaudeError> {
        if let Some(ref transport) = self.transport {
            return transport.count_tokens(request).await;
        }

        let url = format!("{}/v1/messages/count_tokens", self.base_url);
        let headers = self.build_attributed_headers();

        debug!(url = %url, "sending count_tokens request");

        let response = self
            .http
            .post(&url)
            .headers(headers)
            .json(request)
            .send()
            .await?;
        let response = Self::check_status(response).await?;

        let body = response.text().await?;
        let resp: CountTokensResponse = serde_json::from_str(&body)?;
        Ok(resp)
    }

    // ----- internal helpers -------------------------------------------------

    /// Construct the header map common to every request.
    pub(crate) fn build_headers(&self) -> HeaderMap {
        let mut headers = self.extra_headers.clone();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert("anthropic-version", self.api_version.clone());

        match &self.auth {
            AuthMethod::ApiKey(key) => {
                // unwrap is safe: API keys are ASCII.
                headers.insert(
                    "x-api-key",
                    HeaderValue::from_str(key).expect("invalid API key characters"),
                );
            }
            AuthMethod::BearerToken(token) => {
                let value = format!("Bearer {token}");
                headers.insert(
                    AUTHORIZATION,
                    HeaderValue::from_str(&value).expect("invalid token characters"),
                );
            }
        }

        // Add beta features header if any are configured
        if !self.beta_features.is_empty() {
            let beta_value = self.beta_features.join(",");
            if let Ok(hv) = HeaderValue::from_str(&beta_value) {
                headers.insert("anthropic-beta", hv);
            }
        }

        if let Some(workspace_id) = &self.workspace_id {
            headers.insert("anthropic-workspace-id", workspace_id.clone());
        }

        headers
    }

    /// [`build_headers`](Self::build_headers) plus the
    /// `anthropic-user-profile-id` header, for requests that create
    /// messages, count tokens, or create batches.
    pub(crate) fn build_attributed_headers(&self) -> HeaderMap {
        let mut headers = self.build_headers();
        if let Some(user_profile_id) = &self.user_profile_id {
            headers.insert("anthropic-user-profile-id", user_profile_id.clone());
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

    /// A reference to the custom transport, if any (used by sub-clients).
    pub(crate) fn transport(&self) -> Option<&dyn Transport> {
        self.transport.as_ref().map(AsRef::as_ref)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Serve exactly one HTTP request on a loopback port. The response is
    /// written in the given chunks (each flushed separately) so tests can split
    /// a payload at awkward byte offsets. Returns the base URL and a receiver
    /// for the raw request text.
    async fn serve_once(
        status_line: &'static str,
        content_type: &'static str,
        chunks: Vec<Vec<u8>>,
    ) -> (String, tokio::sync::oneshot::Receiver<String>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buf = [0_u8; 4096];
            loop {
                let n = socket.read(&mut buf).await.unwrap();
                request.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&request).to_string();
                if let Some(header_end) = text.find("\r\n\r\n") {
                    let length = text[..header_end]
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if request.len() >= header_end + 4 + length {
                        break;
                    }
                }
            }
            let total: usize = chunks.iter().map(Vec::len).sum();
            let head = format!(
                "HTTP/1.1 {status_line}\r\ncontent-type: {content_type}\r\ncontent-length: {total}\r\nconnection: close\r\n\r\n"
            );
            socket.write_all(head.as_bytes()).await.unwrap();
            for chunk in chunks {
                socket.write_all(&chunk).await.unwrap();
                socket.flush().await.unwrap();
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            let _ = tx.send(String::from_utf8_lossy(&request).to_string());
        });
        (format!("http://{addr}"), rx)
    }

    #[tokio::test]
    async fn raw_request_round_trips_unmodelled_content_and_custom_headers() {
        let response = br#"{"id":"msg_1","content":[{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[]}]}"#;
        let (base, request) =
            serve_once("200 OK", "application/json", vec![response.to_vec()]).await;
        let client = ClaudeClient::new("key")
            .with_base_url(&base)
            .with_api_version("2023-06-01")
            .with_header("x-gateway-route", "blue");
        let body = serde_json::json!({"model": "m", "max_tokens": 8, "messages": [], "context_management": {"edits": []}});

        let out = client.create_message_raw(&body).await.unwrap();

        assert_eq!(out["content"][0]["type"], "web_search_tool_result");
        let request = request.await.unwrap();
        let lower = request.to_ascii_lowercase();
        assert!(lower.contains("anthropic-version: 2023-06-01"));
        assert!(lower.contains("x-gateway-route: blue"));
        let sent: serde_json::Value =
            serde_json::from_str(&request[request.find("\r\n\r\n").unwrap() + 4..]).unwrap();
        assert_eq!(sent, body, "non-streaming raw body is sent verbatim");
    }

    #[tokio::test]
    async fn raw_stream_forces_stream_and_decodes_utf8_split_across_chunks() {
        let sse = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"h\u{e9}llo\"}}\n\nevent: novel_event\ndata: {\"type\":\"novel_event\"}\n\n";
        let bytes = sse.as_bytes();
        // Split inside the two-byte encoding of 'é'.
        let split = sse.find('\u{e9}').unwrap() + 1;
        let (base, request) = serve_once(
            "200 OK",
            "text/event-stream",
            vec![bytes[..split].to_vec(), bytes[split..].to_vec()],
        )
        .await;
        let client = ClaudeClient::new("key").with_base_url(&base);

        let events: Vec<serde_json::Value> = client
            .create_message_stream_raw(&serde_json::json!({"model": "m", "stream": false}))
            .await
            .unwrap()
            .map(Result::unwrap)
            .collect()
            .await;

        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["delta"]["text"], "h\u{e9}llo");
        assert_eq!(events[1]["type"], "novel_event");
        let request = request.await.unwrap();
        let sent: serde_json::Value =
            serde_json::from_str(&request[request.find("\r\n\r\n").unwrap() + 4..]).unwrap();
        assert_eq!(sent["stream"], true);
    }

    #[tokio::test]
    async fn raw_request_maps_error_status_to_api_error() {
        let body = br#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#;
        let (base, _request) =
            serve_once("529 Overloaded", "application/json", vec![body.to_vec()]).await;
        let client = ClaudeClient::new("key").with_base_url(&base);

        let err = client
            .create_message_raw(&serde_json::json!({}))
            .await
            .unwrap_err();

        match err {
            ClaudeError::ApiError {
                status,
                error_type,
                message,
            } => {
                assert_eq!(status, 529);
                assert_eq!(error_type, "overloaded_error");
                assert_eq!(message, "busy");
            }
            other => panic!("expected ApiError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn raw_stream_rejects_non_object_body() {
        let client = ClaudeClient::new("key").with_base_url("http://127.0.0.1:9");
        let err = client
            .create_message_stream_raw(&serde_json::json!([1, 2]))
            .await
            .unwrap_err();
        assert!(matches!(err, ClaudeError::InvalidConfig(_)));
    }

    #[derive(Debug)]
    struct NoopTransport;

    #[async_trait::async_trait]
    impl Transport for NoopTransport {}

    #[tokio::test]
    async fn raw_requests_are_unsupported_on_custom_transports() {
        let client = ClaudeClient::with_transport(NoopTransport);
        let err = client
            .create_message_raw(&serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(matches!(err, ClaudeError::Unsupported(_)));
    }

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
            "2026-10-21"
        );
    }

    #[test]
    fn workspace_header_on_every_request() {
        let client = ClaudeClient::new("key").with_workspace_id("wrkspc_123");
        for headers in [client.build_headers(), client.build_attributed_headers()] {
            assert_eq!(
                headers
                    .get("anthropic-workspace-id")
                    .unwrap()
                    .to_str()
                    .unwrap(),
                "wrkspc_123"
            );
        }
    }

    #[test]
    fn user_profile_header_only_on_attributed_requests() {
        let client = ClaudeClient::new("key").with_user_profile_id("uprof_123");
        assert!(client
            .build_headers()
            .get("anthropic-user-profile-id")
            .is_none());
        assert_eq!(
            client
                .build_attributed_headers()
                .get("anthropic-user-profile-id")
                .unwrap()
                .to_str()
                .unwrap(),
            "uprof_123"
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
