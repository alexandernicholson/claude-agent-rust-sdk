# Architecture

This document describes the internal design of `claude-agent-rust-sdk`, covering the client layer, type system, authentication, batch processing, prompt caching, and error handling.

---

## System Diagram

```
                        +-----------------+
                        |   User Code     |
                        +--------+--------+
                                 |
                    MessageBuilder / BatchClient
                                 |
                        +--------v--------+
                        |  ClaudeClient   |
                        |  (src/client/)  |
                        +--------+--------+
                                 |
                         reqwest::Client
                                 |
                        +--------v--------+
                        | Claude API      |
                        | api.anthropic   |
                        +-----------------+
```

All HTTP communication flows through a single `ClaudeClient` instance. User code interacts with the client either directly or through higher-level abstractions (`MessageBuilder`, `BatchClient`).

---

## Client Design

### ClaudeClient

`ClaudeClient` is the central type. It owns:

- A `reqwest::Client` configured with connection pooling and default timeouts.
- The base URL (`https://api.anthropic.com`).
- An authentication credential (API key or OAuth token).

```
ClaudeClient
  reqwest_client: reqwest::Client
  base_url: String
  auth: AuthMethod
```

`AuthMethod` is an internal enum:

```rust
enum AuthMethod {
    ApiKey(String),       // sent as x-api-key header
    OAuthToken(String),   // sent as Authorization: Bearer header
}
```

Every outgoing request passes through a single internal method that:

1. Constructs the full URL by joining `base_url` with the endpoint path.
2. Injects the authentication header based on `AuthMethod`.
3. Adds the `anthropic-version: 2023-06-01` header.
4. Sets `Content-Type: application/json`.
5. Serializes the request body with `serde_json`.
6. Sends the request via `reqwest` and deserializes the response.

If the API returns a non-2xx status, the client reads the error body and constructs a `ClaudeError::ApiError` with the status code, error type string, and message.

### Why a single client

Reusing one `reqwest::Client` across all operations shares the underlying connection pool and TLS sessions. Creating separate clients per request would open new connections each time, increasing latency and resource usage. The SDK encourages creating one `ClaudeClient` at startup and passing references where needed.

---

## Authentication Flow

The SDK supports two authentication methods, chosen at construction time:

### API Key (`ClaudeClient::new`)

```
Request header: x-api-key: sk-ant-api03-...
```

This is the standard method for server-side applications. The key is stored in `AuthMethod::ApiKey` and injected into every request.

### OAuth Token (`ClaudeClient::with_oauth_token`)

```
Request header: Authorization: Bearer eyJhbGciOi...
```

This method supports tokens obtained through OAuth flows. It is useful in contexts where an OAuth token is already available (for example, from the `CLAUDE_CODE_OAUTH_TOKEN` environment variable). The token is stored in `AuthMethod::OAuthToken`.

Both methods are functionally equivalent from the API's perspective. The choice depends on how credentials are provisioned in the deployment environment.

---

## Type System Design

All request and response types live in `src/types/`. They are plain Rust structs that derive `serde::Serialize` and `serde::Deserialize`, providing a direct mapping to the Claude API JSON format.

### Request types

```
MessageRequest
  model: String
  max_tokens: u32
  messages: Vec<Message>
  system: Option<SystemPrompt>
  temperature: Option<f64>
  top_p: Option<f64>
  top_k: Option<u32>
  stop_sequences: Option<Vec<String>>
  stream: Option<bool>
```

```
Message
  role: Role           // "user" or "assistant"
  content: Content     // String or Vec<ContentBlock>
```

```
ContentBlock
  type: String         // "text", "image", etc.
  text: Option<String>
  cache_control: Option<CacheControl>
  // ... other variant-specific fields
```

Optional fields use `#[serde(skip_serializing_if = "Option::is_none")]` so they are omitted from the JSON when unset, keeping payloads minimal.

### Response types

```
MessageResponse
  id: String
  type: String          // always "message"
  role: String          // always "assistant"
  content: Vec<ContentBlock>
  model: String
  stop_reason: StopReason
  usage: Usage
```

```
Usage
  input_tokens: u32
  output_tokens: u32
  cache_creation_input_tokens: u32
  cache_read_input_tokens: u32
```

The `Usage` struct always includes cache fields. When caching is not used, these are zero.

### Design rationale

- **Concrete types over generics.** The API surface is small enough that a fixed set of structs is clearer than a generic type hierarchy. Each struct matches one JSON schema.
- **Enums for known values.** `Role`, `StopReason`, and `BatchStatus` are Rust enums with `#[serde(rename_all = "snake_case")]`, providing exhaustive matching.
- **Extensible with `#[serde(deny_unknown_fields)]` off.** The API may add new fields over time. By defaulting to ignoring unknown fields during deserialization, the SDK remains forward-compatible without code changes.

---

## Builder Pattern

`MessageBuilder` provides a fluent API for constructing `MessageRequest` values. It enforces required fields through its constructor signature:

```rust
MessageBuilder::new(model: &str, max_tokens: u32)
```

The builder accumulates messages through `.user()` and `.assistant()` calls, which append to an internal `Vec<Message>`. Optional parameters like `.temperature()` and `.system()` set fields on the builder.

`.build()` consumes the builder and returns a `MessageRequest`. `.send(&client)` is a convenience that calls `.build()` followed by `client.send_message()`.

### Cache-aware system prompts

`.system_with_cache(text, cache_control)` wraps the system prompt in a `ContentBlock` with the `cache_control` field set:

```json
{
  "system": [
    {
      "type": "text",
      "text": "...",
      "cache_control": { "type": "ephemeral" }
    }
  ]
}
```

This is the content-block form of the system parameter, which the API requires when cache control is present.

---

## Batch Processing Lifecycle

The `BatchClient` wraps `ClaudeClient` to interact with the `/v1/messages/batches` endpoints. A batch job follows this lifecycle:

```
  create() -----> [in_progress] -----> [ended]
                      ^                   |
                      |                   v
                poll_until_complete()   get_results()
```

### Step 1: Create

`BatchClient::create(requests)` POSTs to `/v1/messages/batches` with a JSON body containing a `requests` array. Each request has:

- `custom_id` -- a user-defined string for correlating results.
- `params` -- a standard `MessageRequest`.

The API returns a `BatchResponse` with the batch `id` and an initial `processing_status` of `in_progress`.

### Step 2: Poll

`BatchClient::poll_until_complete(batch_id, interval)` repeatedly GETs `/v1/messages/batches/{batch_id}` at the given interval. It checks the `processing_status` field:

| Status | Meaning |
|--------|---------|
| `in_progress` | Still processing; keep polling. |
| `ended` | All requests have completed, errored, or expired. |
| `canceling` | Cancellation requested; may still transition to `ended`. |

If the batch does not reach a terminal status within a timeout window, the method returns `ClaudeError::BatchTimeout`.

### Step 3: Retrieve results

`BatchClient::get_results(batch_id)` GETs `/v1/messages/batches/{batch_id}/results`, which returns a JSONL stream. Each line is parsed into a `BatchResult`:

```
BatchResult
  custom_id: String
  result: BatchResultType
```

```
BatchResultType
  Succeeded { message: MessageResponse }
  Errored { error: ApiErrorDetail }
  Expired
```

Results are available for 29 days after batch creation.

### Pricing

All batch usage is billed at 50% of standard API prices. This applies to both input and output tokens.

### Batch limits

- Maximum 100,000 requests or 256 MB per batch (whichever is reached first).
- Batches expire after 24 hours if processing has not completed.
- Batches are scoped to the Workspace of the API key.

---

## Prompt Caching Strategy

### Cache control placement

The `cache_control` field can be placed on content blocks within the `system`, `messages`, or `tools` arrays. The API caches the entire prompt prefix up to and including the marked block.

In this SDK, caching is exposed through:

1. `MessageBuilder::system_with_cache(text, cache_control)` -- caches the system prompt.
2. Direct construction of `ContentBlock` values with `cache_control` set -- for caching user messages or tool definitions.

### CacheControl type

```rust
pub struct CacheControl {
    pub r#type: String,   // always "ephemeral"
    pub ttl: Option<String>,  // None = 5 min, Some("1h") = 1 hour
}
```

Convenience constructors:

- `CacheControl::ephemeral()` -- 5-minute TTL (default).
- `CacheControl::ephemeral_1h()` -- 1-hour TTL.

### How the cache hierarchy works

The Claude API evaluates cache hits by comparing the request prefix against stored cache entries. The prefix is defined by this hierarchy:

```
tools -> system -> messages
```

A change at any level invalidates that level and everything below it. For example, modifying the system prompt invalidates the system and message caches, but not the tools cache.

### Cost model

| Token type | Cost relative to base input |
|------------|---------------------------|
| Uncached input | 1.0x |
| Cache write (5 min) | 1.25x |
| Cache write (1 hour) | 2.0x |
| Cache read | 0.1x |

The break-even point for 5-minute caching is 2 requests (write cost is 1.25x, but the second request costs only 0.1x, for a total of 1.35x across two calls versus 2.0x without caching). For most real-world patterns, caching pays for itself almost immediately.

### Minimum token requirements

The cached prefix must meet a minimum token count, which varies by model:

| Model | Minimum tokens |
|-------|---------------|
| Claude Opus 4.6, 4.5 | 4,096 |
| Claude Sonnet 4.6 | 2,048 |
| Claude Sonnet 4.5, 4 | 1,024 |
| Claude Haiku 4.5 | 4,096 |

If the prefix is shorter than the minimum, the `cache_control` marker is silently ignored and the tokens are billed at the standard input rate.

---

## Error Handling Philosophy

The SDK uses a single error enum, `ClaudeError`, for all fallible operations. The design principles are:

### 1. No panics in library code

Every error condition returns a `Result`. The SDK never calls `unwrap()`, `expect()`, or `panic!()` on user-facing paths.

### 2. Errors carry enough context to act on

- `ApiError` includes the HTTP status code, the Anthropic error type string, and the human-readable message. Callers can match on `status` to implement retry logic (for example, retrying on 429 with backoff).
- `BatchTimeout` includes the batch ID so callers can decide whether to continue polling manually.
- `NetworkError` wraps the underlying `reqwest::Error`, preserving the full error chain.

### 3. Error conversion via From

`ClaudeError` implements `From<reqwest::Error>` and `From<serde_json::Error>`, so `?` propagation works naturally inside the SDK and in user code.

### 4. Non-exhaustive for forward compatibility

The `#[non_exhaustive]` attribute on `ClaudeError` means new variants can be added in minor versions without breaking downstream `match` statements (callers must include a wildcard arm).

### Example: retry on rate limit

```rust
use std::time::Duration;
use tokio::time::sleep;

async fn send_with_retry(
    client: &ClaudeClient,
    request: MessageRequest,
    max_retries: u32,
) -> Result<MessageResponse, ClaudeError> {
    let mut attempts = 0;
    loop {
        match client.send_message(&request).await {
            Ok(response) => return Ok(response),
            Err(ClaudeError::ApiError { status: 429, .. }) if attempts < max_retries => {
                attempts += 1;
                sleep(Duration::from_secs(2u64.pow(attempts))).await;
            }
            Err(e) => return Err(e),
        }
    }
}
```

This pattern is not built into the SDK intentionally. Retry policies (exponential backoff, jitter, max attempts) vary by application, so the SDK provides the error information and lets callers implement their own strategy.
