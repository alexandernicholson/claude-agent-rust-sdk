# claude-agent-rust-sdk

> **Unofficial** Rust SDK for the [Claude API](https://platform.claude.com/docs/en/api/messages) by Anthropic.
>
> This is a community-maintained project and is not affiliated with or endorsed by Anthropic.

[![Crates.io](https://img.shields.io/crates/v/claude-agent-rust-sdk.svg)](https://crates.io/crates/claude-agent-rust-sdk)
[![Documentation](https://docs.rs/claude-agent-rust-sdk/badge.svg)](https://docs.rs/claude-agent-rust-sdk)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org)

---

## Overview

A typed, ergonomic Rust client for the Claude Messages API. Built on `reqwest` and `serde`, it provides:

- **Messages API** -- create single-turn and multi-turn conversations
- **Streaming** -- async stream of SSE events with typed deltas
- **Extended Thinking** -- enable Claude's internal reasoning with configurable token budgets
- **Tool Use** -- define custom tools and control tool selection
- **Server Tools** -- built-in `web_fetch` and `web_search` tools that execute server-side
- **Vision** -- send images (base64, URL, or Files API) and documents (PDF, text)
- **Prompt Caching** -- cache system prompts and message prefixes for up to 90% cost reduction
- **Batch Processing** -- submit thousands of requests asynchronously at 50% pricing
- **Token Counting** -- count tokens before sending a request
- **Model Constants** -- typed model IDs for all current Claude models
- **Structured Output** -- JSON schema validation for model responses
- **Builder Pattern** -- construct requests fluently with `MessageBuilder`
- **Strong Types** -- every API request and response is a concrete Rust type with serde mappings
- **Citations** -- character, page, and content block citation types
- **Transport Trait** -- pluggable backend for routing API operations through custom transports (CLI tools, mocks, proxies)

---

## Installation

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
claude-agent-rust-sdk = "0.1"
```

Or reference the Git repository directly:

```toml
[dependencies]
claude-agent-rust-sdk = { git = "https://github.com/alexandernicholson/claude-agent-rust-sdk" }
```

The crate pulls in `reqwest`, `serde`, `tokio`, and `thiserror` transitively. Your project needs a Tokio runtime.

---

## Quick Start

### Create a Message

```rust
use claude_agent_rust_sdk::client::ClaudeClient;
use claude_agent_rust_sdk::models;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClaudeClient::new("sk-ant-...");

    let response = client
        .messages()
        .model(models::CLAUDE_SONNET_4_6)
        .max_tokens(1024)
        .user("Explain ownership in Rust in two sentences.")
        .send()
        .await?;

    println!("{}", response.text().unwrap_or("(no text)"));
    Ok(())
}
```

### Streaming

```rust
use futures::stream::StreamExt;
use claude_agent_rust_sdk::client::ClaudeClient;
use claude_agent_rust_sdk::types::{StreamEvent, ContentDelta};
use claude_agent_rust_sdk::models;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClaudeClient::new("sk-ant-...");

    let mut stream = client
        .messages()
        .model(models::CLAUDE_SONNET_4_6)
        .max_tokens(1024)
        .user("Write a poem about Rust.")
        .send_stream()
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ContentBlockDelta {
                delta: ContentDelta::TextDelta { text },
                ..
            } => print!("{}", text),
            StreamEvent::MessageStop {} => break,
            _ => {}
        }
    }
    Ok(())
}
```

### Batch Processing

```rust
use std::time::Duration;
use claude_agent_rust_sdk::client::ClaudeClient;
use claude_agent_rust_sdk::types::batch::{CreateBatchRequest, BatchRequest};
use claude_agent_rust_sdk::types::{CreateMessageRequest, Message, MessageContent, Role};
use claude_agent_rust_sdk::models;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClaudeClient::new("sk-ant-...");

    // Create a batch of requests
    let batch = client
        .batches()
        .create(&CreateBatchRequest {
            requests: vec![
                BatchRequest {
                    custom_id: "req-1".into(),
                    params: CreateMessageRequest {
                        model: models::CLAUDE_HAIKU_4_5.into(),
                        max_tokens: 1024,
                        messages: vec![Message {
                            role: Role::User,
                            content: MessageContent::Text("Hello!".into()),
                        }],
                        system: None, temperature: None, top_p: None, top_k: None,
                        stop_sequences: None, stream: None, tools: None,
                        tool_choice: None, metadata: None, cache_control: None,
                        output_config: None, thinking: None, service_tier: None,
                    },
                },
            ],
        })
        .await?;

    // Poll until complete, then fetch results
    let _completed = client
        .batches()
        .poll_until_complete(&batch.id, Duration::from_secs(30))
        .await?;

    let results = client.batches().results(&batch.id).await?;
    println!("Got {} results", results.len());
    Ok(())
}
```

### Using the Transport Trait

Route all API operations through a custom backend instead of HTTP:

```rust
use async_trait::async_trait;
use claude_agent_rust_sdk::client::ClaudeClient;
use claude_agent_rust_sdk::error::ClaudeError;
use claude_agent_rust_sdk::transport::Transport;
use claude_agent_rust_sdk::types::{
    CreateMessageRequest, CreateMessageResponse, ResponseContentBlock,
    Role, Usage,
};

#[derive(Debug)]
struct MockTransport;

#[async_trait]
impl Transport for MockTransport {
    async fn create_message(
        &self,
        _request: &CreateMessageRequest,
    ) -> Result<CreateMessageResponse, ClaudeError> {
        Ok(CreateMessageResponse {
            id: "msg_mock".into(),
            response_type: Some("message".into()),
            model: "mock".into(),
            role: Role::Assistant,
            content: vec![ResponseContentBlock::Text {
                text: "Hello from the mock!".into(),
                citations: None,
            }],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: Usage::default(),
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClaudeClient::with_transport(MockTransport);
    let response = client
        .messages()
        .model("any-model")
        .max_tokens(1024)
        .user("Hi")
        .send()
        .await?;
    println!("{}", response.text().unwrap_or("(no text)"));
    Ok(())
}
```

---

## Authentication

### API Key

Uses the `x-api-key` header. Standard method for server-side applications.

```rust
let client = ClaudeClient::new("sk-ant-api03-...");
```

### OAuth Token

Uses the `Authorization: Bearer` header. Useful for OAuth flows.

```rust
let client = ClaudeClient::with_oauth_token("eyJhbGciOi...");
```

### Beta Features

Add `anthropic-beta` headers for beta features:

```rust
let client = ClaudeClient::new("sk-ant-...")
    .with_beta("interleaved-thinking-2025-05-14")
    .with_beta("files-api-2025-04-14");
```

---

## Features

### Messages API

```rust
let response = client
    .messages()
    .model(models::CLAUDE_SONNET_4_6)
    .max_tokens(1024)
    .system("You are a concise technical writer.")
    .user("Explain TCP in one paragraph.")
    .temperature(0.7)
    .send()
    .await?;

println!("{}", response.text().unwrap_or("(no text)"));
```

### Streaming

Stream responses as server-sent events:

```rust
use futures::stream::StreamExt;
use claude_agent_rust_sdk::types::{StreamEvent, ContentDelta};

let mut stream = client
    .messages()
    .model(models::CLAUDE_SONNET_4_6)
    .max_tokens(1024)
    .user("Write a poem about Rust.")
    .send_stream()
    .await?;

while let Some(event) = stream.next().await {
    match event? {
        StreamEvent::ContentBlockDelta {
            delta: ContentDelta::TextDelta { text },
            ..
        } => print!("{}", text),
        StreamEvent::MessageStop {} => break,
        _ => {}
    }
}
```

### Extended Thinking

Enable Claude's internal reasoning:

```rust
let response = client
    .messages()
    .model(models::CLAUDE_SONNET_4_6)
    .max_tokens(16000)
    .thinking(10000) // budget in tokens
    .user("Prove that there are infinitely many primes.")
    .send()
    .await?;

if let Some(thinking) = response.thinking() {
    println!("Thinking: {}", thinking);
}
println!("Answer: {}", response.text().unwrap_or("(no text)"));
```

For Claude Opus 4.6, use adaptive thinking:

```rust
let response = client
    .messages()
    .model(models::CLAUDE_OPUS_4_6)
    .max_tokens(16000)
    .thinking_adaptive(None) // model decides
    .user("Solve this complex problem...")
    .send()
    .await?;
```

### Tool Use

Define tools the model can call:

```rust
use claude_agent_rust_sdk::types::{Tool, ToolChoice};

let response = client
    .messages()
    .model(models::CLAUDE_SONNET_4_6)
    .max_tokens(1024)
    .tool(Tool {
        name: "get_weather".into(),
        description: "Get the current weather".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }),
        cache_control: None,
    })
    .tool_choice(ToolChoice::Auto)
    .user("What's the weather in Tokyo?")
    .send()
    .await?;

for (id, name, input) in response.tool_uses() {
    println!("Tool call: {} ({}) -> {}", name, id, input);
}
```

### Server Tools

Server tools (`web_fetch`, `web_search`) execute on Anthropic's servers and do not require the client to handle tool results. The model decides when to use them.

```rust
use claude_agent_rust_sdk::types::ServerTool;

// Fetch and summarize a URL
let response = client
    .messages()
    .model(models::CLAUDE_SONNET_4_6)
    .max_tokens(4096)
    .server_tool(ServerTool::web_fetch().with_max_uses(3))
    .user("Summarize https://example.com")
    .send()
    .await?;

// Web search with domain restrictions
let response = client
    .messages()
    .model(models::CLAUDE_SONNET_4_6)
    .max_tokens(4096)
    .server_tool(
        ServerTool::web_search()
            .with_max_uses(5)
            .with_allowed_domains(vec!["rust-lang.org".into(), "docs.rs".into()])
    )
    .user("What's new in Rust 1.80?")
    .send()
    .await?;
```

### Vision

Send images with messages:

```rust
use claude_agent_rust_sdk::types::{ContentBlock, ImageSource};

let response = client
    .messages()
    .model(models::CLAUDE_SONNET_4_6)
    .max_tokens(1024)
    .user_blocks(vec![
        ContentBlock::Image {
            source: ImageSource::Url {
                url: "https://example.com/photo.jpg".into(),
            },
            cache_control: None,
        },
        ContentBlock::Text {
            text: "Describe this image.".into(),
            cache_control: None,
        },
    ])
    .send()
    .await?;
```

### Documents (PDF)

Send PDF documents:

```rust
use claude_agent_rust_sdk::types::{ContentBlock, DocumentSource, CitationConfig};

let response = client
    .messages()
    .model(models::CLAUDE_SONNET_4_6)
    .max_tokens(4096)
    .user_blocks(vec![
        ContentBlock::Document {
            source: DocumentSource::Base64 {
                media_type: "application/pdf".into(),
                data: base64_pdf_data,
            },
            cache_control: None,
            citations: Some(CitationConfig { enabled: true }),
            context: None,
            title: Some("report.pdf".into()),
        },
        ContentBlock::Text {
            text: "Summarize this document.".into(),
            cache_control: None,
        },
    ])
    .send()
    .await?;
```

### Structured Output

Require JSON output matching a schema:

```rust
let response = client
    .messages()
    .model(models::CLAUDE_SONNET_4_6)
    .max_tokens(1024)
    .json_schema(serde_json::json!({
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["answer", "confidence"]
    }))
    .user("What is 2+2?")
    .send()
    .await?;
```

### Token Counting

Count tokens before sending:

```rust
use claude_agent_rust_sdk::types::CountTokensRequest;

let count = client
    .count_tokens(&CountTokensRequest {
        model: models::CLAUDE_SONNET_4_6.into(),
        messages: vec![/* ... */],
        system: None,
        tools: None,
        thinking: None,
        tool_choice: None,
    })
    .await?;

println!("Input tokens: {}", count.input_tokens);
```

### Prompt Caching

Cache prompt prefixes for cost savings:

```rust
use claude_agent_rust_sdk::types::CacheControl;

let response = client
    .messages()
    .model(models::CLAUDE_SONNET_4_6)
    .max_tokens(1024)
    .system_with_cache(
        "You are an expert Rust developer. [long instructions...]",
        CacheControl::ephemeral(),  // 5-minute TTL
    )
    .user("How do I implement Iterator?")
    .send()
    .await?;

// Check cache performance:
if let Some(cached) = response.usage.cache_read_input_tokens {
    println!("Cache read tokens: {}", cached);
}
```

### Batch Processing

Submit batches at 50% pricing:

```rust
use claude_agent_rust_sdk::types::batch::{CreateBatchRequest, BatchRequest, ListBatchesParams};

// Create a batch
let batch = client
    .batches()
    .create(&CreateBatchRequest { requests: vec![/* ... */] })
    .await?;

// List batches
let list = client
    .batches()
    .list(&ListBatchesParams::default())
    .await?;

// Poll until complete
let completed = client
    .batches()
    .poll_until_complete(&batch.id, Duration::from_secs(30))
    .await?;

// Get results
let results = client.batches().results(&batch.id).await?;
```

---

## Architecture

```
                        +-----------------+
                        |   User Code     |
                        +--------+--------+
                                 |
                    MessageBuilder / BatchClient
                                 |
                        +--------v--------+
                        |  ClaudeClient   |
                        +--------+--------+
                                 |
                    +------------+------------+
                    |                         |
             Has transport?            Has transport?
              Yes: delegate             No: use HTTP
                    |                         |
             +------v------+          +-------v--------+
             |  Transport   |          |   HttpTransport |
             |  (custom)    |          |   (reqwest)     |
             +-------------+          +----------------+
```

The SDK is organized around these core components:

- **`ClaudeClient`** (`src/client/mod.rs`) -- the main entry point. Holds authentication, base URL, beta features, and an optional custom `Transport`. Provides `create_message`, `create_message_stream`, and `count_tokens` methods.

- **`MessageBuilder`** (`src/client/builder.rs`) -- a fluent builder for constructing `CreateMessageRequest` values. Accessed via `client.messages()`. Supports all API parameters including system prompts, tools, thinking, structured output, and caching.

- **`BatchClient`** (`src/batch/mod.rs`) -- client for the Message Batches API. Accessed via `client.batches()`. Supports create, retrieve, list, poll, cancel, and results.

- **`Transport` trait** (`src/transport.rs`) -- abstraction over how API operations are executed. The default path sends HTTP requests via `reqwest`. Custom transports can route operations through CLI tools, mocks, or proxies. All methods have default implementations returning `Unsupported`, so you only need to implement the operations you use.

- **`SseStream`** (`src/streaming.rs`) -- an async `Stream<Item = Result<StreamEvent, ClaudeError>>` that parses SSE events from a streaming response. Also supports construction from arbitrary streams via `SseStream::from_stream` for custom transports.

- **Type system** (`src/types/mod.rs`, `src/types/batch.rs`) -- strongly-typed request and response types with serde mappings. Includes content blocks, tool definitions (custom + server), citations, thinking config, and streaming events.

- **Error handling** (`src/error.rs`) -- a single `ClaudeError` enum covering API errors, network errors, serialization errors, batch timeouts, invalid config, stream errors, unsupported operations, and transport errors.

- **Model constants** (`src/models.rs`) -- `&str` constants for all current Claude model IDs (Opus, Sonnet, Haiku families with date-pinned variants).

---

## Model Constants

The `models` module provides constants for all current model IDs:

```rust
use claude_agent_rust_sdk::models;

models::CLAUDE_OPUS_4_6       // "claude-opus-4-6"
models::CLAUDE_SONNET_4_6     // "claude-sonnet-4-6"
models::CLAUDE_HAIKU_4_5      // "claude-haiku-4-5"
models::CLAUDE_OPUS_4_5       // "claude-opus-4-5-20251101"
models::CLAUDE_SONNET_4_5     // "claude-sonnet-4-5-20250929"
// ... and more
```

You can also pass any model ID string directly.

---

## Supported Models

| Constant | Model ID | Description |
|----------|----------|-------------|
| `CLAUDE_OPUS_4_6` | `claude-opus-4-6` | Most intelligent model -- best for agents and complex coding |
| `CLAUDE_SONNET_4_6` | `claude-sonnet-4-6` | Best balance of speed and intelligence |
| `CLAUDE_HAIKU_4_5` | `claude-haiku-4-5` | Fastest model with near-frontier intelligence |
| `CLAUDE_OPUS_4_5` | `claude-opus-4-5-20251101` | Opus 4.5, date-pinned |
| `CLAUDE_OPUS_4_1` | `claude-opus-4-1-20250805` | Opus 4.1, date-pinned |
| `CLAUDE_OPUS_4_0` | `claude-opus-4-20250514` | Opus 4.0, date-pinned |
| `CLAUDE_SONNET_4_5` | `claude-sonnet-4-5-20250929` | Sonnet 4.5, date-pinned |
| `CLAUDE_SONNET_4_0` | `claude-sonnet-4-20250514` | Sonnet 4.0, date-pinned |
| `CLAUDE_HAIKU_4_5_PINNED` | `claude-haiku-4-5-20251001` | Haiku 4.5, date-pinned |
| `CLAUDE_3_HAIKU` | `claude-3-haiku-20240307` | Claude 3 Haiku (legacy) |

---

## API Coverage

| Feature | Status |
|---------|--------|
| Messages (create) | Implemented |
| Messages (streaming) | Implemented |
| Extended thinking (enabled, disabled, adaptive) | Implemented |
| Tool use / function calling | Implemented |
| Tool choice (auto, any, tool, none) | Implemented |
| Server tools (web_fetch, web_search) | Implemented |
| Vision (base64, URL, file) | Implemented |
| Documents (PDF, text, URL) | Implemented |
| Citations (char, page, block, web, search) | Implemented |
| Structured output (JSON schema) | Implemented |
| Prompt caching (5m, 1h TTL) | Implemented |
| Token counting | Implemented |
| Model constants | Implemented |
| Beta feature headers | Implemented |
| Transport trait (pluggable backends) | Implemented |
| Message Batches (create, retrieve, poll, results) | Implemented |
| Message Batches (list with pagination) | Implemented |
| Message Batches (cancel) | Implemented |

---

## Error Handling

All operations return `Result<T, ClaudeError>`:

```rust
use claude_agent_rust_sdk::error::ClaudeError;

match result {
    Err(ClaudeError::ApiError { status, error_type, message }) => {
        // 400 invalid_request_error, 401 authentication_error,
        // 403 permission_error, 404 not_found_error,
        // 413 request_too_large, 429 rate_limit_error,
        // 500 api_error, 529 overloaded_error
    }
    Err(ClaudeError::StreamError { error_type, message }) => {
        // Error received inside an SSE stream
    }
    Err(ClaudeError::NetworkError(e)) => { /* reqwest error */ }
    Err(ClaudeError::SerializationError(e)) => { /* serde error */ }
    Err(ClaudeError::BatchTimeout { batch_id }) => { /* polling timeout */ }
    Err(ClaudeError::InvalidConfig(msg)) => { /* SDK misconfiguration */ }
    Err(ClaudeError::Unsupported(op)) => { /* transport doesn't support this */ }
    Err(ClaudeError::TransportError(msg)) => { /* transport-specific failure */ }
    Ok(response) => { /* success */ }
}
```

---

## Contributing

Contributions are welcome. To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes and add tests
4. Run the checks:
   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo test
   ```
5. Open a pull request against `main`

Please follow the existing code style, add doc comments to public items, and include tests for new functionality.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Disclaimer

This is an **unofficial**, community-maintained SDK. It is **not** affiliated with, endorsed by, or supported by Anthropic. The Claude name and API are trademarks of Anthropic, PBC.

Use of this SDK is subject to Anthropic's [Terms of Service](https://www.anthropic.com/terms) and [API usage policies](https://platform.claude.com/docs/en/docs/usage-policy). You are responsible for complying with all applicable terms when using the Claude API through this SDK.
