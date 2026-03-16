# Architecture

This document describes the internal design of `claude-agent-rust-sdk`, covering the client layer, type system, authentication, streaming, extended thinking, tool use, batch processing, prompt caching, and error handling.

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
                  +--------------+---------------+
                  |              |               |
            create_message  create_message_  count_tokens
            (non-stream)    stream (SSE)     (/count_tokens)
                  |              |               |
                  v              v               v
              reqwest       reqwest          reqwest
              POST          POST+stream      POST
                  |              |               |
                  v              v               v
             CreateMessage   SseStream      CountTokens
             Response        (Stream<       Response
                              StreamEvent>)
```

All HTTP communication flows through a single `ClaudeClient` instance. User code interacts with the client either directly or through higher-level abstractions (`MessageBuilder`, `BatchClient`).

---

## Client Design

### ClaudeClient

`ClaudeClient` is the central type. It owns:

- A `reqwest::Client` configured with connection pooling and default timeouts.
- The base URL (`https://api.anthropic.com`).
- An authentication credential (API key or OAuth token).
- A list of beta feature flags.

```
ClaudeClient
  reqwest_client: reqwest::Client
  base_url: String
  auth: AuthMethod
  beta_features: Vec<String>
```

Every outgoing request passes through `build_headers()` which:

1. Sets `Content-Type: application/json`.
2. Sets `anthropic-version: 2023-06-01`.
3. Injects the authentication header based on `AuthMethod`.
4. If beta features are configured, sets `anthropic-beta` as a comma-separated value.

### Core Methods

| Method | Endpoint | Returns |
|--------|----------|---------|
| `create_message` | `POST /v1/messages` | `CreateMessageResponse` |
| `create_message_stream` | `POST /v1/messages` (stream: true) | `SseStream` |
| `count_tokens` | `POST /v1/messages/count_tokens` | `CountTokensResponse` |

---

## Authentication Flow

### API Key (`ClaudeClient::new`)

```
Request header: x-api-key: sk-ant-api03-...
```

### OAuth Token (`ClaudeClient::with_oauth_token`)

```
Request header: Authorization: Bearer eyJhbGciOi...
```

### Beta Features (`ClaudeClient::with_beta`)

```
Request header: anthropic-beta: interleaved-thinking-2025-05-14,files-api-2025-04-14
```

---

## Type System Design

All request and response types live in `src/types/`. They derive `serde::Serialize` and `serde::Deserialize`.

### Content Block Types

Request-side `ContentBlock` is a tagged enum:

```
ContentBlock
  Text { text, cache_control }
  Image { source: ImageSource, cache_control }
  Document { source: DocumentSource, cache_control, citations, context, title }
  ToolUse { id, name, input }
  ToolResult { tool_use_id, content, is_error, cache_control }
  Thinking { thinking, signature }
  RedactedThinking { data }
```

Response-side `ResponseContentBlock`:

```
ResponseContentBlock
  Text { text, citations }
  ToolUse { id, name, input }
  Thinking { thinking, signature }
  RedactedThinking { data }
```

### Image and Document Sources

```
ImageSource
  Base64 { media_type, data }
  Url { url }
  File { file_id }

DocumentSource
  Base64 { media_type, data }
  Text { media_type, data }
  Url { url }
  Content { content }
```

### Tool Definitions

```
ToolDefinition (untagged)
  Custom(Tool)          -- user-defined tools with input_schema
  Server(ServerTool)    -- server-side tools (web search, code exec, etc.)
```

### Citation Types

```
Citation (tagged by type)
  CharLocation { cited_text, document_index, start_char_index, end_char_index }
  PageLocation { cited_text, document_index, start_page_number, end_page_number }
  ContentBlockLocation { cited_text, document_index, start_block_index, end_block_index }
  WebSearchResultLocation { cited_text, url, title }
  SearchResultLocation { cited_text, title, source, indices }
```

---

## Streaming Design

### How It Works

When `stream: true` is set, the API returns an SSE (Server-Sent Events) stream. The SDK processes this in layers:

1. **`reqwest::Response`** provides a byte stream via `.chunk()`.
2. **`StreamState`** buffers bytes, splits on newlines, and tracks position.
3. **`parse_sse_line`** converts individual `data: {...}` lines into `StreamEvent` values.
4. **`SseStream`** wraps the above in a `futures::Stream` implementation.

### Event Flow

```
message_start
  content_block_start (index 0)
    content_block_delta (text_delta / thinking_delta / input_json_delta)
    content_block_delta ...
    content_block_delta (signature_delta, for thinking blocks)
  content_block_stop (index 0)
  content_block_start (index 1)
    ...
  content_block_stop (index 1)
message_delta (stop_reason, usage)
message_stop
```

Interspersed `ping` events are also possible. `error` events may arrive at any point.

### Delta Types

```
ContentDelta (tagged by type)
  TextDelta { text }
  InputJsonDelta { partial_json }
  ThinkingDelta { thinking }
  SignatureDelta { signature }
```

### Error Handling in Streams

If the API returns an `error` event, it is yielded as `StreamEvent::Error`. The caller can convert it to `ClaudeError::StreamError` if desired.

---

## Extended Thinking

### Configuration

```
ThinkingConfig (tagged by type)
  Enabled { budget_tokens: u32 }    -- manual thinking with fixed budget
  Disabled {}                       -- no thinking (default)
  Adaptive { budget_tokens: Option } -- model decides (recommended for Opus 4.6)
```

### Response

When thinking is enabled, responses include `Thinking` content blocks before `Text` blocks:

```json
{
  "content": [
    {"type": "thinking", "thinking": "Let me analyze...", "signature": "EqQB..."},
    {"type": "text", "text": "The answer is..."}
  ]
}
```

The `CreateMessageResponse::thinking()` helper extracts the first thinking block.

### Streaming

During streaming, thinking blocks produce `ThinkingDelta` and `SignatureDelta` events.

---

## Builder Pattern

`MessageBuilder` provides a fluent API. All fields except `model`, `max_tokens`, and at least one message are optional.

Key builder methods by category:

| Category | Methods |
|----------|---------|
| Required | `model()`, `max_tokens()`, `user()` |
| Messages | `user()`, `user_blocks()`, `assistant()`, `assistant_blocks()`, `message()` |
| System | `system()`, `system_with_cache()` |
| Sampling | `temperature()`, `top_p()`, `top_k()`, `stop_sequences()` |
| Tools | `tool()`, `tools()`, `custom_tools()`, `tool_choice()` |
| Thinking | `thinking()`, `thinking_adaptive()`, `thinking_config()` |
| Output | `effort()`, `json_schema()` |
| Other | `stream()`, `cache_control()`, `metadata()`, `service_tier()` |
| Send | `build()`, `send()`, `send_stream()` |

---

## Batch Processing Lifecycle

```
  create() -----> [in_progress] -----> [ended]
                      ^                   |
                      |                   v
                poll_until_complete()   results()
```

### List Batches

`GET /v1/messages/batches` with pagination:

```
ListBatchesParams
  after_id: Option<String>
  before_id: Option<String>
  limit: Option<u32>
```

Returns:

```
ListBatchesResponse
  data: Vec<BatchResponse>
  has_more: bool
  first_id: Option<String>
  last_id: Option<String>
```

---

## Prompt Caching Strategy

### CacheControl type

```rust
pub struct CacheControl {
    pub cache_type: String,    // always "ephemeral"
    pub ttl: Option<String>,   // None = "5m", Some("1h") = 1 hour
}
```

Convenience constructors: `ephemeral()`, `ephemeral_5m()`, `ephemeral_1h()`.

---

## Error Handling

```
ClaudeError
  ApiError { status, error_type, message }
  NetworkError(reqwest::Error)
  SerializationError(serde_json::Error)
  BatchTimeout { batch_id }
  InvalidConfig(String)
  StreamError { error_type, message }
```

### HTTP Error Codes

| Status | Error Type |
|--------|-----------|
| 400 | `invalid_request_error` |
| 401 | `authentication_error` |
| 403 | `permission_error` |
| 404 | `not_found_error` |
| 413 | `request_too_large` |
| 429 | `rate_limit_error` |
| 500 | `api_error` |
| 529 | `overloaded_error` |

---

## Token Counting

The `count_tokens` endpoint allows pre-flight token counting:

```
POST /v1/messages/count_tokens

CountTokensRequest { model, messages, system, tools, thinking, tool_choice }
CountTokensResponse { input_tokens }
```

---

## Model Constants

The `models` module provides `&str` constants for all current model IDs:

```rust
pub const CLAUDE_OPUS_4_6: &str = "claude-opus-4-6";
pub const CLAUDE_SONNET_4_6: &str = "claude-sonnet-4-6";
pub const CLAUDE_HAIKU_4_5: &str = "claude-haiku-4-5";
// ... and date-pinned variants
```
