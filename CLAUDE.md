# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

claude-agent-rust-sdk is an unofficial Rust SDK for the Claude API by Anthropic. It wraps the Messages API, prompt caching, and batch processing behind typed Rust interfaces built on `reqwest` and `serde`.

## Architecture

```
src/
  lib.rs              -- Crate root. Re-exports public types from all modules.
  error.rs            -- ClaudeError enum (ApiError, NetworkError, SerializationError,
                         BatchTimeout, InvalidConfig). Uses thiserror for Display/From.
  types/
    mod.rs            -- API request/response types: MessageRequest, MessageResponse,
                         Content, ContentBlock, Usage, CacheControl, StopReason.
                         All derive Serialize/Deserialize.
    batch.rs          -- Batch-specific types: BatchRequest, BatchResponse,
                         BatchResultType, BatchStatus. Maps to /v1/messages/batches JSON.
  client/
    mod.rs            -- ClaudeClient: holds a reqwest::Client, base URL, and auth config.
                         Provides send_message() which POSTs to /v1/messages.
                         Handles header injection (x-api-key or Authorization: Bearer,
                         anthropic-version, content-type).
    builder.rs        -- MessageBuilder: fluent builder for MessageRequest. Methods:
                         new(model, max_tokens), system(), system_with_cache(),
                         user(), assistant(), temperature(), top_p(),
                         stop_sequences(), build(), send().
  batch/
    mod.rs            -- BatchClient: wraps ClaudeClient for batch operations.
                         Methods: create(), get_status(), poll_until_complete(),
                         get_results(). Targets /v1/messages/batches endpoints.
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
| `wiremock` (0.6) | HTTP mocking for integration tests |

## API Details

| Detail | Value |
|--------|-------|
| Base URL | `https://api.anthropic.com` |
| Version header | `anthropic-version: 2023-06-01` |
| Auth (API key) | `x-api-key: <key>` |
| Auth (OAuth) | `Authorization: Bearer <token>` |
| Messages endpoint | `POST /v1/messages` |
| Create batch | `POST /v1/messages/batches` |
| Get batch | `GET /v1/messages/batches/{batch_id}` |
| Batch results | `GET /v1/messages/batches/{batch_id}/results` |

## Key Design Decisions

- **Single `ClaudeClient`** -- one client instance is shared across message and batch operations. It owns the `reqwest::Client` and connection pool.
- **Builder validates at compile time** -- `MessageBuilder::new()` requires `model` and `max_tokens`. The builder does not compile without at least one message.
- **Errors are non-exhaustive** -- `ClaudeError` uses `#[non_exhaustive]` so new variants can be added without breaking downstream.
- **No streaming yet** -- the `stream` feature on `reqwest` is included in dependencies but streaming is not yet exposed in the public API.
- **Batch polling** -- `poll_until_complete` takes a `Duration` interval and loops until the batch status is terminal or a timeout is hit, returning `ClaudeError::BatchTimeout` on expiry.

## Testing Strategy

- **Unit tests** live in each module file under `#[cfg(test)]`.
- **Integration tests** use `wiremock` to stand up a local HTTP server and assert correct request headers, body serialization, and response deserialization.
- Tests do not call the live API. Any test requiring a real API key should be gated behind `#[ignore]` and documented.

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
