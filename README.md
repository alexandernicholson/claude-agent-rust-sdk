# claude-agent-rust-sdk

> **Unofficial** Rust SDK for the [Claude API](https://platform.claude.com/docs/en/api/messages) and Claude Code agents by Anthropic.
>
> This is a community-maintained project and is not affiliated with or endorsed by Anthropic.

[![Crates.io](https://img.shields.io/crates/v/claude-agent-rust-sdk.svg)](https://crates.io/crates/claude-agent-rust-sdk)
[![Documentation](https://docs.rs/claude-agent-rust-sdk/badge.svg)](https://docs.rs/claude-agent-rust-sdk)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org)

---

## Overview

A typed Rust client for both the Claude Messages API and the bidirectional Claude Code Agent SDK protocol. Built on `reqwest`, `serde`, and Tokio, it provides:

- **Claude Code agents** -- typed NDJSON messages, streaming turns, control requests, session resume, hooks, skills, plugins, subagents, and in-process MCP tools
- **Session persistence** -- filesystem-compatible query, import, mutation, fork, resume, and pluggable `SessionStore` APIs

- **Messages API** -- create single-turn and multi-turn conversations
- **Streaming** -- async stream of SSE events with typed deltas
- **Extended Thinking** -- adaptive, between-tools, and manual thinking with `display` control and xhigh effort
- **Tool Use** -- define custom tools and control tool selection
- **Server Tools** -- built-in `web_fetch`, `web_search`, and `code_execution` tools (any version via `ServerTool::new`) that execute server-side, with web fetch `url_sources`
- **Toolsets** -- computer-use and browser toolsets
- **Speed** -- fast mode via `speed("fast")`
- **Cache Diagnostics** -- find out why a prompt-cache miss happened
- **Stop Details** -- refusal category and explanation
- **System Messages** -- mid-conversation `Role::System` messages
- **Vision** -- send images (base64, URL, or Files API, with oversized-image transformations) and documents (PDF, text, file)
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
claude-agent-rust-sdk = "0.5"
```

Or reference the Git repository directly:

```toml
[dependencies]
claude-agent-rust-sdk = { git = "https://github.com/alexandernicholson/claude-agent-rust-sdk" }
```

The crate pulls in `reqwest`, `serde`, `tokio`, and `thiserror` transitively. Your project needs a Tokio runtime.

---

## Quick Start

### Run a Claude Code agent

Agent clients are transport-independent. Pair the SDK with
[`claude-cli-transport`](https://github.com/alexandernicholson/claude-cli-transport)
to run against Claude Code 2.0 or newer:

```rust
use claude_agent_rust_sdk::agent::{AgentOptions, ClaudeAgentClient};
use claude_cli_transport::ClaudeCliAgentTransport;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = AgentOptions {
        max_turns: Some(12),
        max_budget_usd: Some(1.0),
        ..AgentOptions::default()
    };
    let mut client = ClaudeAgentClient::new(ClaudeCliAgentTransport::new(), options);
    client.connect().await?;
    let result = client.query("Diagnose the failing CI job.").await?;
    client.close().await?;
    println!("{}", result.text);
    Ok(())
}
```

The Agent SDK surface also supports asynchronous message receipt, control
operations, typed hooks, permission callbacks, MCP servers, and durable session
stores. See the public `agent`, `extensions`, and `sessions` modules for the
complete contracts.

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
                        system: None,
                        stop_sequences: None, stream: None, tools: None,
                        tool_choice: None, metadata: None, cache_control: None,
                        output_config: None, thinking: None, service_tier: None,
                        speed: None, diagnostics: None,
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
            stop_details: None,
            usage: Usage::default(),
            diagnostics: None,
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

### Workspace and User Profile

```rust
let client = ClaudeClient::new("sk-ant-...")
    .with_workspace_id("wrkspc_...")        // anthropic-workspace-id on every request
    .with_user_profile_id("uprof_...");     // anthropic-user-profile-id on messages, streaming, token counting, batch create
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
    .stop_sequences(vec!["END".into()])
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

Adaptive thinking is the recommended mode for current models; the model decides when and how much to think:

```rust
let response = client
    .messages()
    .model(models::CLAUDE_OPUS_5_5)
    .max_tokens(16000)
    .thinking_adaptive(None) // model decides
    .user("Prove that there are infinitely many primes.")
    .send()
    .await?;

if let Some(thinking) = response.thinking() {
    println!("Thinking: {}", thinking);
}
println!("Answer: {}", response.text().unwrap_or("(no text)"));
```

Other modes: `thinking_between_tools()` enables thinking between tool calls, and `thinking_config(...)` accepts a `ThinkingConfig` directly, including `display` (`"summarized"` or `"omitted"`) on the `Enabled` and `Adaptive` variants. Manual `thinking(budget_tokens)` is deprecated on Opus 4.6 and Sonnet 4.6 and rejected by later models.

```rust
// Deprecated: manual budget (Opus 4.6 / Sonnet 4.6 only)
let response = client
    .messages()
    .model(models::CLAUDE_SONNET_4_6)
    .max_tokens(16000)
    .thinking(10000)
    .user("Prove that there are infinitely many primes.")
    .send()
    .await?;
```

Thinking token usage is reported in `response.usage.output_tokens_details`.

### Effort

Effort is a behavioral signal that controls token use across text, tool calls,
and adaptive thinking. `High` is equivalent to omitting the parameter. Use
`max_tokens` and application-level spend limits for hard bounds.

```rust
use claude_agent_rust_sdk::EffortLevel;

let response = client
    .messages()
    .model(models::CLAUDE_SONNET_5)
    .max_tokens(16_000)
    .effort(EffortLevel::Medium)
    .user("Diagnose this CI failure.")
    .send()
    .await?;
```

### Speed (Fast Mode)

```rust
// Some models require the `fast-mode-2026-02-01` beta header for any `speed`
// value; add it with `.with_beta("fast-mode-2026-02-01")` when needed.
let client = ClaudeClient::new("sk-ant-...");
let response = client
    .messages()
    .model(models::CLAUDE_OPUS_5_5)
    .max_tokens(1024)
    .speed("fast") // "standard" | "fast"; some models need the beta above
    .user("Quick question...")
    .send()
    .await?;
```

### Mid-Conversation System Messages and Stop Details

`Role::System` messages (text only) can appear inside `messages` via `system_message`. The top-level `system()` is unchanged. When a response stops with `refusal`, `stop_details` carries the category and explanation:

```rust
let response = client
    .messages()
    .model(models::CLAUDE_SONNET_4_6)
    .max_tokens(1024)
    .user("Hello")
    .assistant("Hi!")
    .system_message("From now on, answer in French.")
    .user("What is Rust?")
    .send()
    .await?;

if let Some(details) = &response.stop_details {
    println!("Stopped: {} {:?} {:?}", details.details_type, details.category, details.explanation);
}
```

The `stop_reason` can also be `model_context_window_exceeded`.

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

Server tools (`web_fetch`, `web_search`, `code_execution`) execute on Anthropic's servers and do not require the client to handle tool results. The model decides when to use them.

```rust
use claude_agent_rust_sdk::types::{ServerTool, UrlSource, WebFetchUrlSources};

// Fetch and summarize a URL
let response = client
    .messages()
    .model(models::CLAUDE_SONNET_4_6)
    .max_tokens(4096)
    .server_tool(ServerTool::web_fetch().with_max_uses(3))
    .user("Summarize https://example.com")
    .send()
    .await?;

// Newer tool versions via the generic constructor, with URL source controls
let response = client
    .messages()
    .model(models::CLAUDE_OPUS_5_5)
    .max_tokens(4096)
    .server_tool(
        ServerTool::new("web_fetch_20260318", "web_fetch")
            .with_url_sources(WebFetchUrlSources {
                user_input: Some(UrlSource::All {}),
                client_tool_results: Some(UrlSource::only(&["my_client_tool"])),
                server_tool_results: Some(UrlSource::None {}),
            })
            .with_response_inclusion("excluded")
            .with_use_cache(true),
    )
    .server_tool(ServerTool::new("code_execution_20260521", "code_execution"))
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

### Toolsets

Computer-use and browser toolsets are attached with `.toolset(...)`. Tool-use and tool-result blocks carry an optional `toolset_name` identifying the originating toolset.

```rust
use claude_agent_rust_sdk::types::Toolset;

let response = client
    .messages()
    .model(models::CLAUDE_OPUS_5_5)
    .max_tokens(4096)
    .toolset(Toolset::computer().with_member_config("zoom", serde_json::json!({"enabled": false})))
    .toolset(Toolset::browser())
    .user("Open the docs site and find the pricing page.")
    .send()
    .await?;
```

Browser state is reported back with `ContentBlock::BrowserState`.

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
            // Optionally downsize oversized images instead of erroring:
            // transformations: Some(ImageTransformations { oversized_image: Some("downsize".into()) }),
            transformations: None,
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

Documents can also reference an uploaded file with `DocumentSource::File { file_id }`.

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
        speed: None,
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

To find out why a cache miss happened, opt in to cache diagnostics. Pass `None` on the first turn and the previous response ID afterwards. Setting `max_tokens(0)` pre-warms the cache without generating output.

```rust
let response = client
    .messages()
    .model(models::CLAUDE_SONNET_4_6)
    .max_tokens(1024)
    .system_with_cache("[long instructions...]", CacheControl::ephemeral())
    .diagnostics(Some(&previous_response.id))
    .user("Next question")
    .send()
    .await?;

if let Some(reason) = response.diagnostics.and_then(|d| d.cache_miss_reason) {
    println!("Cache miss: {} ({:?} tokens)", reason.reason_type, reason.cache_missed_input_tokens);
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

- **`ClaudeClient`** (`src/client/mod.rs`) -- the main entry point. Holds authentication, base URL, beta features, optional workspace/user-profile IDs, and an optional custom `Transport`. Provides `create_message`, `create_message_stream`, and `count_tokens` methods.

- **`MessageBuilder`** (`src/client/builder.rs`) -- a fluent builder for constructing `CreateMessageRequest` values. Accessed via `client.messages()`. Supports all API parameters including system prompts, tools, thinking, structured output, and caching.

- **`BatchClient`** (`src/batch/mod.rs`) -- client for the Message Batches API. Accessed via `client.batches()`. Supports create, retrieve, list, poll, cancel, and results.

- **`Transport` trait** (`src/transport.rs`) -- abstraction over how API operations are executed. The default path sends HTTP requests via `reqwest`. Custom transports can route operations through CLI tools, mocks, or proxies. All methods have default implementations returning `Unsupported`, so you only need to implement the operations you use.

- **`SseStream`** (`src/streaming.rs`) -- an async `Stream<Item = Result<StreamEvent, ClaudeError>>` that parses SSE events from a streaming response. Also supports construction from arbitrary streams via `SseStream::from_stream` for custom transports.

- **Type system** (`src/types/mod.rs`, `src/types/batch.rs`) -- strongly-typed request and response types with serde mappings. Includes content blocks, tool definitions (custom, server, toolset), citations, thinking config, and streaming events.

- **Error handling** (`src/error.rs`) -- a single `ClaudeError` enum covering API errors, network errors, serialization errors, batch timeouts, invalid config, stream errors, unsupported operations, and transport errors.

- **Model constants** (`src/models.rs`) -- `&str` constants for all current Claude model IDs (Fable, Mythos, Opus, Sonnet, Haiku families); retiring models are marked `#[deprecated]`.

---

## Model Constants

The `models` module provides constants for current model IDs:

```rust
use claude_agent_rust_sdk::models;

models::CLAUDE_FABLE_5_1      // "claude-fable-5-1"
models::CLAUDE_MYTHOS_5_1     // "claude-mythos-5-1"
models::CLAUDE_OPUS_5_5       // "claude-opus-5-5"
models::CLAUDE_SONNET_5_5     // "claude-sonnet-5-5"
models::CLAUDE_HAIKU_5_5      // "claude-haiku-5-5"
models::CLAUDE_OPUS_4_6       // "claude-opus-4-6"
models::CLAUDE_SONNET_4_6     // "claude-sonnet-4-6"
models::CLAUDE_HAIKU_4_5      // "claude-haiku-4-5"
// ... and more
```

You can also pass any model ID string directly.

---

## Supported Models

| Constant | Model ID | Notes |
|----------|----------|-------|
| `CLAUDE_FABLE_5_1` | `claude-fable-5-1` | Fable 5.1 |
| `CLAUDE_FABLE_5` | `claude-fable-5` | Fable 5 |
| `CLAUDE_MYTHOS_5_1` | `claude-mythos-5-1` | Mythos 5.1 |
| `CLAUDE_MYTHOS_5` | `claude-mythos-5` | Mythos 5 |
| `CLAUDE_MYTHOS_PREVIEW` | `claude-mythos-preview` | **Deprecated**, end-of-life 2026-06-30; use `CLAUDE_MYTHOS_5` |
| `CLAUDE_OPUS_5_5` | `claude-opus-5-5` | Opus 5.5 |
| `CLAUDE_OPUS_5` | `claude-opus-5` | Opus 5 |
| `CLAUDE_OPUS_4_8` | `claude-opus-4-8` | Opus 4.8 |
| `CLAUDE_OPUS_4_7` | `claude-opus-4-7` | Opus 4.7 |
| `CLAUDE_OPUS_4_6` | `claude-opus-4-6` | Opus 4.6 |
| `CLAUDE_OPUS_4_5` | `claude-opus-4-5-20251101` | Opus 4.5, date-pinned |
| `CLAUDE_SONNET_5_5` | `claude-sonnet-5-5` | Sonnet 5.5 |
| `CLAUDE_SONNET_5` | `claude-sonnet-5` | Sonnet 5 |
| `CLAUDE_SONNET_4_6` | `claude-sonnet-4-6` | Sonnet 4.6 |
| `CLAUDE_SONNET_4_5` | `claude-sonnet-4-5-20250929` | **Deprecated**, end-of-life 2026-11-30; use `CLAUDE_SONNET_5_5` |
| `CLAUDE_HAIKU_5_5` | `claude-haiku-5-5` | Haiku 5.5 |
| `CLAUDE_HAIKU_4_5` | `claude-haiku-4-5` | Haiku 4.5 |
| `CLAUDE_HAIKU_4_5_PINNED` | `claude-haiku-4-5-20251001` | Haiku 4.5, date-pinned |

---

## API Coverage

| Feature | Status |
|---------|--------|
| Claude Code Agent SDK streaming protocol | Implemented |
| Agent options, messages, results, and control requests | Implemented |
| Hooks and tool-permission callbacks | Implemented |
| In-process, stdio, HTTP, SSE, and proxy MCP servers | Implemented |
| Skills, plugins, subagents, and sandbox configuration | Implemented |
| Session query, mutation, import, fork, and resume APIs | Implemented |
| Pluggable session stores and transcript mirroring | Implemented |
| Messages (create) | Implemented |
| Messages (streaming) | Implemented |
| Mid-conversation system messages (`Role::System`) | Implemented |
| Extended thinking (adaptive, between_tools, enabled, disabled, `display`) | Implemented |
| Effort (low, medium, high, xhigh, max) | Implemented |
| Speed / fast mode | Implemented |
| Cache diagnostics (`diagnostics`, `cache_miss_reason`) | Implemented |
| Stop details (refusal category/explanation) | Implemented |
| Tool use / function calling (incl. `toolset_name`) | Implemented |
| Tool choice (auto, any, tool, none) | Implemented |
| Server tools (web_fetch, web_search, code_execution via `ServerTool::new`) | Implemented |
| Web fetch `url_sources`, `response_inclusion`, `use_cache` | Implemented |
| Toolsets (computer, browser) | Implemented |
| Vision (base64, URL, file, image transformations) | Implemented |
| Documents (PDF, text, URL, file) | Implemented |
| Citations (char, page, block, web, search) | Implemented |
| Structured output (JSON schema) | Implemented |
| Prompt caching (5m, 1h TTL, `max_tokens: 0` pre-warm) | Implemented |
| Token counting | Implemented |
| Model constants | Implemented |
| Beta feature headers | Implemented |
| Workspace / user-profile headers | Implemented |
| Transport trait (pluggable backends) | Implemented |
| Message Batches (create, retrieve, poll, results) | Implemented |
| Message Batches (list with pagination) | Implemented |
| Message Batches (cancel) | Implemented |

---

## Migrating from 0.4

Version 0.5 targets API version `2026-10-21` and ports the official Python Agent SDK changes through v0.2.176. See the [official API migration guide](https://platform.claude.com/docs/en/api/migration-2026-10-21). Breaking changes:

- The `anthropic-version` header is now `2026-10-21` (was `2023-06-01`).
- `temperature`, `top_p`, and `top_k` are removed from `CreateMessageRequest` and `MessageBuilder` (`temperature()`, `top_p()`, `top_k()`). The new API version rejects them with a 400.
- Retired model constants are removed: `CLAUDE_OPUS_4_1`, `CLAUDE_OPUS_4_0`, `CLAUDE_SONNET_4_0`, `CLAUDE_3_HAIKU`. `CLAUDE_SONNET_4_5` and `CLAUDE_MYTHOS_PREVIEW` are deprecated.
- New fields on structs and enum variants that you construct with literals or match exhaustively: `CreateMessageRequest { speed, diagnostics }`, `CreateMessageResponse { stop_details, diagnostics }`, `CountTokensRequest { speed }`, `ContentBlock::Image { transformations }`, `ContentBlock::ToolUse`/`ToolResult { toolset_name }`, `ResponseContentBlock::ToolUse`/`StreamContentBlock::ToolUse { toolset_name }`, `MessageDeltaData { stop_details }`, `Usage { output_tokens_details, speed }`, `ServerTool { url_sources, response_inclusion, use_cache }` (prefer `ServerTool::new`), `AgentOptions { forward_subagent_text, verbatim_prompts }` (use `..AgentOptions::default()`), `SystemPrompt::Preset { snapshot }`, `SessionMessage { parent_agent_id }`.
- `ThinkingConfig` variants changed: `Enabled { budget_tokens, display }`, `Adaptive { budget_tokens: Option<u32>, display }`, plus a new `BetweenTools {}` variant.
- New enum variants: `ToolDefinition::Toolset`, `ContentBlock::BrowserState`, `Role::System`, `DocumentSource::File`, `SystemPrompt::Custom`, `ClaudeError::ResultError`.
- A CLI that exits non-zero after an error `result` now surfaces `ClaudeError::ResultError(ResultErrorDetails)` instead of `ClaudeError::Process`. Use `ClaudeError::is_process()` / `exit_code()` to handle both.
- `SandboxIgnoreViolations` is now a command-pattern map (`BTreeMap<String, Vec<String>>`, e.g. `{"*": ["..."]}`) instead of a `{ file, network }` struct, matching the CLI schema.
- `BatchResultBody::Succeeded { message }` now holds a `Box<CreateMessageResponse>`.
- Manual `thinking(budget)` is deprecated on Opus 4.6 and Sonnet 4.6 and rejected by later models. Use `thinking_adaptive(None)`.

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
    Err(ClaudeError::ResultError(details)) => {
        // The Claude Code CLI exited after an error result:
        // details.subtype, details.terminal_reason, details.errors, details.api_error_status
    }
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

## Release notes

### 0.5.0

- Messages API: API version `2026-10-21`; removed `temperature`/`top_p`/`top_k`;
  added `speed`, cache `diagnostics`, `stop_details`, mid-conversation system
  messages, `between_tools` thinking and `display`, usage
  `output_tokens_details`/`speed`, image `transformations`, file document
  sources, toolsets with `toolset_name` and browser state, `ServerTool::new`
  with web fetch `url_sources`, workspace and user-profile headers, and the
  current model catalog.
- Agent SDK (official Python SDK v0.2.140–v0.2.176): `can_use_tool` and
  in-process MCP runs keep stdin open until the CLI reports idle (bounded by
  `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS`); `forward_subagent_text`;
  `verbatim_prompts`; `SystemPrompt::Custom` and `snapshot`; `ResultError`
  with the error result's payload; typed `SystemInitData`; hook-specific output
  types and new hook input fields; `ignoreViolations` as a command-pattern map;
  MCP tool cancellation and snake_case annotation hints; subagent transcripts
  carry `parent_tool_use_id`/`parent_agent_id`; session messages include every
  parallel tool call and result.

### 0.4.0

- Added the Claude Code Agent SDK protocol with typed messages, control
  requests, hooks, permissions, skills, plugins, subagents, and MCP servers.
- Added filesystem-compatible and store-backed session listing, import,
  mutation, fork, resume, transcript mirroring, and summary APIs.
- Matched the official Python SDK's option serialization, message parsing,
  cancellation, and session-store conformance behavior.
- Hardened resume materialization, UUID validation, transcript parsing, and
  cancellation-safe teardown.

---


## License

This project is licensed under the [MIT License](LICENSE).

---

## Disclaimer

This is an **unofficial**, community-maintained SDK. It is **not** affiliated with, endorsed by, or supported by Anthropic. The Claude name and API are trademarks of Anthropic, PBC.

Use of this SDK is subject to Anthropic's [Terms of Service](https://www.anthropic.com/terms) and [API usage policies](https://platform.claude.com/docs/en/docs/usage-policy). You are responsible for complying with all applicable terms when using the Claude API through this SDK.
