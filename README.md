# claude-agent-rust-sdk

> **Unofficial** Rust SDK for the [Claude API](https://docs.anthropic.com/en/api/messages) by Anthropic.
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
- **Prompt Caching** -- cache system prompts and message prefixes for up to 90% cost reduction on reads
- **Batch Processing** -- submit thousands of requests asynchronously at 50% of standard pricing
- **Builder Pattern** -- construct requests fluently with `MessageBuilder`
- **Strong Types** -- every API request and response is a concrete Rust type with serde mappings

---

## Installation

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
claude-agent-rust-sdk = "0.1"
```

The crate pulls in `reqwest`, `serde`, `tokio`, and `thiserror` transitively. Your project needs a Tokio runtime.

---

## Quick Start

```rust
use claude_agent_rust_sdk::{ClaudeClient, MessageBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClaudeClient::new("sk-ant-...");

    let response = MessageBuilder::new("claude-sonnet-4-6", 1024)
        .user("Explain ownership in Rust in two sentences.")
        .build()
        .send(&client)
        .await?;

    println!("{}", response.content[0].text);
    Ok(())
}
```

---

## Authentication

The SDK supports two authentication methods. Both are passed as headers on every request.

### API Key

Uses the `x-api-key` header. This is the standard method for server-side applications.

```rust
let client = ClaudeClient::new("sk-ant-api03-...");
```

### OAuth Token

Uses the `Authorization: Bearer` header. Useful when working with tokens from OAuth flows (for example, `CLAUDE_CODE_OAUTH_TOKEN`).

```rust
let client = ClaudeClient::with_oauth_token("eyJhbGciOi...");
```

Both methods target the same base URL (`https://api.anthropic.com`) and send the `anthropic-version: 2023-06-01` header automatically.

---

## Features

### Messages API

Create a simple message:

```rust
let response = MessageBuilder::new("claude-sonnet-4-6", 1024)
    .user("What is the capital of France?")
    .build()
    .send(&client)
    .await?;
```

Multi-turn conversation:

```rust
let response = MessageBuilder::new("claude-sonnet-4-6", 1024)
    .user("My name is Alex.")
    .assistant("Nice to meet you, Alex!")
    .user("What is my name?")
    .build()
    .send(&client)
    .await?;
```

Add a system prompt:

```rust
let response = MessageBuilder::new("claude-sonnet-4-6", 1024)
    .system("You are a concise technical writer.")
    .user("Explain TCP in one paragraph.")
    .build()
    .send(&client)
    .await?;
```

Configure optional parameters:

```rust
let response = MessageBuilder::new("claude-sonnet-4-6", 2048)
    .system("You are a creative writing assistant.")
    .user("Write a haiku about Rust.")
    .temperature(0.9)
    .top_p(0.95)
    .stop_sequences(vec!["END".into()])
    .build()
    .send(&client)
    .await?;
```

---

### Prompt Caching

Prompt caching lets you cache prompt prefixes across API calls. Cached reads cost **10% of the base input token price** -- a 90% reduction.

#### How it works

Attach a `CacheControl` marker to any content block (system prompt, user message, tool definition). The API caches everything up to and including that block. Subsequent requests that share the same prefix hit the cache instead of reprocessing.

#### TTL options

| TTL | Cache write cost | Best for |
|-----|-----------------|----------|
| **5 minutes** (default) | 1.25x base input | High-frequency reuse (chatbots, agents) |
| **1 hour** | 2.0x base input | Lower-frequency reuse (batch jobs, periodic tasks) |

Cache reads always cost **0.1x base input** regardless of TTL.

#### Minimum token requirements

Not all prompts can be cached. The prefix must meet a minimum token count:

| Model | Minimum tokens |
|-------|---------------|
| Claude Opus 4.6, 4.5 | 4,096 |
| Claude Sonnet 4.6 | 2,048 |
| Claude Sonnet 4.5, 4 | 1,024 |
| Claude Haiku 4.5 | 4,096 |

#### Example: cache a system prompt

```rust
use claude_agent_rust_sdk::types::CacheControl;

let response = MessageBuilder::new("claude-sonnet-4-6", 1024)
    .system_with_cache(
        "You are an expert Rust developer. [... long instructions ...]",
        CacheControl::ephemeral(),         // 5-minute TTL
    )
    .user("How do I implement Iterator?")
    .build()
    .send(&client)
    .await?;

// Check cache performance in the response:
println!("Cache read tokens:  {}", response.usage.cache_read_input_tokens);
println!("Cache write tokens: {}", response.usage.cache_creation_input_tokens);
```

Use the 1-hour TTL for less frequent access patterns:

```rust
let response = MessageBuilder::new("claude-sonnet-4-6", 1024)
    .system_with_cache(
        "You are an expert Rust developer. [... long instructions ...]",
        CacheControl::ephemeral_1h(),      // 1-hour TTL
    )
    .user("Explain lifetimes.")
    .build()
    .send(&client)
    .await?;
```

You can set up to **4 cache breakpoints** per request.

---

### Batch Processing

The Batch API lets you submit up to 100,000 message requests in a single batch. All usage is charged at **50% of standard API prices**. Most batches complete within 1 hour.

#### Create a batch

```rust
use claude_agent_rust_sdk::batch::{BatchClient, BatchRequest};

let batch_client = BatchClient::new(&client);

let requests = vec![
    BatchRequest {
        custom_id: "summary-1".into(),
        params: MessageBuilder::new("claude-haiku-4-5", 1024)
            .user("Summarize: Rust is a systems programming language...")
            .build(),
    },
    BatchRequest {
        custom_id: "summary-2".into(),
        params: MessageBuilder::new("claude-haiku-4-5", 1024)
            .user("Summarize: Python is a high-level language...")
            .build(),
    },
];

let batch = batch_client.create(requests).await?;
println!("Batch ID: {}", batch.id);
```

#### Poll for completion

```rust
use std::time::Duration;

let completed = batch_client
    .poll_until_complete(&batch.id, Duration::from_secs(30))
    .await?;

println!("Status: {:?}", completed.processing_status);
```

#### Retrieve results

```rust
let results = batch_client.get_results(&batch.id).await?;

for result in &results {
    match &result.result {
        BatchResultType::Succeeded { message } => {
            println!("[{}] {}", result.custom_id, message.content[0].text);
        }
        BatchResultType::Errored { error } => {
            eprintln!("[{}] Error: {}", result.custom_id, error.message);
        }
        BatchResultType::Expired => {
            eprintln!("[{}] Expired", result.custom_id);
        }
    }
}
```

#### Combine caching with batching

For maximum savings, use cached prompts inside batch requests. A shared system prompt is cached once and read by every request in the batch:

```rust
let shared_system = "You are a senior code reviewer. [... detailed instructions ...]";

let requests: Vec<BatchRequest> = pull_requests
    .iter()
    .map(|pr| BatchRequest {
        custom_id: pr.id.clone(),
        params: MessageBuilder::new("claude-sonnet-4-6", 2048)
            .system_with_cache(shared_system, CacheControl::ephemeral_1h())
            .user(&format!("Review this diff:\n{}", pr.diff))
            .build(),
    })
    .collect();

let batch = batch_client.create(requests).await?;
// 50% batch discount + 90% cache read discount on the system prompt
```

---

### Builder Pattern

`MessageBuilder` provides a fluent interface for constructing requests:

```rust
let request = MessageBuilder::new("claude-sonnet-4-6", 1024)
    .system("You are a helpful assistant.")
    .user("Hello!")
    .assistant("Hi there!")
    .user("What can you do?")
    .temperature(0.7)
    .top_p(0.9)
    .stop_sequences(vec!["DONE".into()])
    .build();
```

The builder validates required fields at compile time through the type system. `model` and `max_tokens` are set in `new()`, and at least one user message is required before `build()` succeeds.

---

### Error Handling

All fallible operations return `Result<T, ClaudeError>`. The error type covers every failure mode:

```rust
use claude_agent_rust_sdk::ClaudeError;

match client.send_message(request).await {
    Ok(response) => println!("{}", response.content[0].text),

    Err(ClaudeError::ApiError { status, error_type, message }) => {
        // The API returned an error response (4xx/5xx).
        // status: HTTP status code (e.g. 429 for rate limiting)
        // error_type: Anthropic error type (e.g. "rate_limit_error")
        // message: Human-readable description
        eprintln!("API error {status} [{error_type}]: {message}");
    }

    Err(ClaudeError::NetworkError(e)) => {
        // Connection, DNS, or timeout failure.
        eprintln!("Network issue: {e}");
    }

    Err(ClaudeError::SerializationError(e)) => {
        // Failed to serialize the request or deserialize the response.
        eprintln!("Serde error: {e}");
    }

    Err(ClaudeError::BatchTimeout { batch_id }) => {
        // A batch poll exceeded the allowed duration.
        eprintln!("Batch {batch_id} timed out");
    }

    Err(ClaudeError::InvalidConfig(msg)) => {
        // Invalid SDK configuration (e.g. empty API key).
        eprintln!("Config error: {msg}");
    }
}
```

---

## Supported Models

| Model ID | Description |
|----------|-------------|
| `claude-opus-4-6` | Most intelligent model -- best for agents and complex coding |
| `claude-sonnet-4-6` | Best balance of speed and intelligence |
| `claude-haiku-4-5` | Fastest model with near-frontier intelligence |
| `claude-haiku-4-5-20251001` | Date-pinned Haiku 4.5 snapshot |

Pass any model ID string to `MessageBuilder::new()`. The SDK does not restrict which models you use, so newer model IDs will work without an SDK update.

---

## API Coverage

| Feature | Status |
|---------|--------|
| Messages (create) | Implemented |
| Messages (streaming) | Planned |
| Prompt caching | Implemented |
| Message Batches (create, poll, results) | Implemented |
| Message Batches (cancel, list) | Planned |
| Tool use / function calling | Planned |
| Vision (image inputs) | Planned |
| Extended thinking | Planned |

---

## Cost Optimization Guide

Four strategies for reducing your Claude API spend, from simplest to most aggressive:

### 1. Choose the right model

Use `claude-haiku-4-5` for high-volume, cost-sensitive tasks. It is the cheapest model while still delivering near-frontier intelligence.

### 2. Cache repeated prompts

If your system prompt or few-shot examples are reused across requests, add a cache breakpoint. Reads cost **0.1x** the base input price.

```rust
// Without caching: 100 requests * 10,000 tokens * $3/MTok = $3.00
// With caching:    1 write ($3.75/MTok) + 99 reads ($0.30/MTok) = $0.07
//                  Savings: ~98%
```

### 3. Batch non-urgent work

Submit evaluation runs, content generation, and analysis jobs through the Batch API for a flat **50% discount** on all token costs.

### 4. Combine caching and batching

Use cached system prompts inside batch requests. The discounts stack:

| Strategy | Input cost multiplier |
|----------|----------------------|
| Standard | 1.0x |
| Cache read only | 0.1x |
| Batch only | 0.5x |
| Cache read + batch | 0.05x |

A 10,000-token system prompt across 1,000 batch requests drops from $30 (standard) to $1.50 (cache read + batch).

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

Use of this SDK is subject to Anthropic's [Terms of Service](https://www.anthropic.com/terms) and [API usage policies](https://docs.anthropic.com/en/docs/usage-policy). You are responsible for complying with all applicable terms when using the Claude API through this SDK.
