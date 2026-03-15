//! Unofficial Rust SDK for the Claude API.
//!
//! # Quick start
//!
//! ```ignore
//! use claude_agent_rust_sdk::{client::ClaudeClient, types::CacheControl};
//!
//! let client = ClaudeClient::new("sk-ant-...");
//!
//! let response = client
//!     .messages()
//!     .model("claude-haiku-4-5")
//!     .max_tokens(1024)
//!     .system("You are a helpful assistant.")
//!     .user("What is the capital of France?")
//!     .send()
//!     .await?;
//!
//! println!("{}", response.text().unwrap_or("(no text)"));
//! ```

pub mod batch;
pub mod client;
pub mod error;
pub mod types;
