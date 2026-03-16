# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

claude-agent-rust-sdk is an unofficial Rust SDK for the Claude API by Anthropic. It wraps the Messages API, streaming, extended thinking, tool use, vision, documents, prompt caching, token counting, and batch processing behind typed Rust interfaces built on `reqwest` and `serde`.

## Architecture

```
src/
  lib.rs              -- Crate root. Re-exports public types from all modules.
  error.rs            -- ClaudeError enum (ApiError, NetworkError, SerializationError,
                         BatchTimeout, InvalidConfig, StreamError). Uses thiserror for Display/From.
  models.rs           -- Model ID constants (CLAUDE_OPUS_4_6, CLAUDE_SONNET_4_6,
                         CLAUDE_HAIKU_4_5, etc.) for all current Claude models.
  streaming.rs        -- SSE event parsing (parse_sse_line) and SseStream async Stream type.
                         Handles chunked SSE responses from reqwest byte streams.
  types/
    mod.rs            -- API request/response types: CreateMessageRequest, CreateMessageResponse,
                         ContentBlock, ResponseContentBlock, Usage, CacheControl, StreamEvent,
                         ContentDelta, ThinkingConfig, ToolChoice, Tool, ToolDefinition,
                         Citation, DocumentSource, ImageSource, CountTokensRequest/Response,
                         OutputConfig, OutputFormat, Metadata. All derive Serialize/Deserialize.
    batch.rs          -- Batch-specific types: BatchRequest, BatchResponse, BatchResult,
                         BatchResultBody, BatchStatus, ListBatchesResponse, ListBatchesParams.
                         Maps to /v1/messages/batches JSON.
  client/
    mod.rs            -- ClaudeClient: holds a reqwest::Client, base URL, auth config, and
                         beta feature list. Provides create_message(), create_message_stream(),
                         and count_tokens(). Handles header injection (x-api-key or
                         Authorization: Bearer, anthropic-version, anthropic-beta, content-type).
    builder.rs        -- MessageBuilder: fluent builder for CreateMessageRequest. Methods:
                         model(), max_tokens(), system(), system_with_cache(),
                         user(), user_blocks(), assistant(), assistant_blocks(),
                         temperature(), top_p(), top_k(), stop_sequences(), stream(),
                         tool(), tools(), custom_tools(), tool_choice(),
                         thinking(), thinking_adaptive(), thinking_config(),
                         effort(), json_schema(), service_tier(), cache_control(),
                         metadata(), build(), send(), send_stream().
  batch/
    mod.rs            -- BatchClient: wraps ClaudeClient for batch operations.
                         Methods: create(), retrieve(), list(), results(),
                         poll_until_complete(), cancel().
                         Targets /v1/messages/batches endpoints.
```

## Development

### Prerequisites

- Rust 2021 edition (check `Cargo.toml` for MSRV)
- No external system dependencies beyond a C linker

### Commands

```bash
cargo build              # Compile the library
cargo test               # Run all tests (unit + integration)
cargo clippy             # Lint (treat warnings as errors in CI)
cargo fmt --check        # Check formatting
cargo doc --open         # Build and view rustdoc
```

### Dependencies

| Crate | Purpose |
|-------|---------|
| `reqwest` (0.12, json+stream) | HTTP client |
| `serde` (1, derive) | Serialization |
| `serde_json` (1) | JSON parsing |
| `tokio` (1, full) | Async runtime |
| `thiserror` (2) | Error derive macro |
| `tracing` (0.1) | Structured logging |
| `futures` (0.3) | Stream utilities |
| `uuid` (1, v4) | Batch request ID generation |
| `async-trait` (0.1) | Async trait support |

### Dev dependencies

| Crate | Purpose |
|-------|---------|
| `tokio-test` (0.4) | Async test utilities |
| `tracing-subscriber` (0.3) | Logging for examples |

## API Details

| Detail | Value |
|--------|-------|
| Base URL | `https://api.anthropic.com` |
| Version header | `anthropic-version: 2023-06-01` |
| Auth (API key) | `x-api-key: <key>` |
| Auth (OAuth) | `Authorization: Bearer <token>` |
| Beta features | `anthropic-beta: <comma-separated>` |
| Messages endpoint | `POST /v1/messages` |
| Streaming | `POST /v1/messages` with `stream: true` |
| Token counting | `POST /v1/messages/count_tokens` |
| Create batch | `POST /v1/messages/batches` |
| Get batch | `GET /v1/messages/batches/{batch_id}` |
| List batches | `GET /v1/messages/batches` |
| Batch results | `GET /v1/messages/batches/{batch_id}/results` |
| Cancel batch | `POST /v1/messages/batches/{batch_id}/cancel` |

## Key Design Decisions

- **Single `ClaudeClient`** -- one client instance is shared across message and batch operations. It owns the `reqwest::Client` and connection pool.
- **Builder validates at send time** -- `MessageBuilder` validates required fields (model, max_tokens, messages) when `build()` or `send()` is called, returning `ClaudeError::InvalidConfig`.
- **Streaming via `SseStream`** -- wraps a `reqwest::Response` byte stream, parses SSE lines, and yields `StreamEvent` values through the `futures::Stream` trait.
- **Extended thinking types** -- `ThinkingConfig` is a tagged enum with `Enabled { budget_tokens }`, `Disabled {}`, and `Adaptive { budget_tokens }` variants.
- **Tool definitions** -- `ToolDefinition` is an untagged enum supporting both custom tools (`Tool`) and server tools (`ServerTool`).
- **Errors are non-exhaustive** -- `ClaudeError` uses `#[non_exhaustive]` so new variants can be added without breaking downstream.
- **Batch polling** -- `poll_until_complete` takes a `Duration` interval and loops until the batch status is terminal or a timeout is hit.

## Testing Strategy

- **Unit tests** live in each module file under `#[cfg(test)]`.
- Tests cover serialization/deserialization round-trips for all types, builder method behavior, SSE line parsing, model constants, and error formatting.
- Tests do not call the live API. Any test requiring a real API key should be gated behind `#[ignore]` and documented.
- **135 tests** currently pass across all modules.

## Common Tasks

### Adding a new API parameter

1. Add the field to the relevant struct in `src/types/mod.rs` (use `#[serde(skip_serializing_if = "Option::is_none")]` for optional fields).
2. Add a builder method in `src/client/builder.rs`.
3. Add a unit test asserting correct serialization.
4. Update the README feature table if this is user-facing.

### Adding a new error variant

1. Add the variant to `ClaudeError` in `src/error.rs`.
2. Add a `#[error("...")]` format string.
3. If it wraps another error, add `#[from]`.
4. Add a test that constructs and displays the error.

### Adding a new streaming event type

1. Add the variant to `StreamEvent` in `src/types/mod.rs`.
2. Add a deserialization test.
3. Add a `parse_sse_line` test in `src/streaming.rs`.
