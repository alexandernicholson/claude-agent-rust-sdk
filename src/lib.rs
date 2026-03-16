//! Unofficial Rust SDK for the Claude API.
//!
//! # Quick start
//!
//! ```ignore
//! use claude_agent_rust_sdk::{client::ClaudeClient, types::CacheControl, models};
//!
//! let client = ClaudeClient::new("sk-ant-...");
//!
//! let response = client
//!     .messages()
//!     .model(models::CLAUDE_HAIKU_4_5)
//!     .max_tokens(1024)
//!     .system("You are a helpful assistant.")
//!     .user("What is the capital of France?")
//!     .send()
//!     .await?;
//!
//! println!("{}", response.text().unwrap_or("(no text)"));
//! ```
//!
//! # Streaming
//!
//! ```ignore
//! use futures::stream::StreamExt;
//! use claude_agent_rust_sdk::types::StreamEvent;
//!
//! let mut stream = client
//!     .messages()
//!     .model(models::CLAUDE_SONNET_4_6)
//!     .max_tokens(1024)
//!     .user("Hello!")
//!     .send_stream()
//!     .await?;
//!
//! while let Some(event) = stream.next().await {
//!     match event? {
//!         StreamEvent::ContentBlockDelta { delta, .. } => { /* handle delta */ }
//!         StreamEvent::MessageStop {} => break,
//!         _ => {}
//!     }
//! }
//! ```
//!
//! # Extended Thinking
//!
//! ```ignore
//! let response = client
//!     .messages()
//!     .model(models::CLAUDE_SONNET_4_6)
//!     .max_tokens(16000)
//!     .thinking(10000)
//!     .user("Solve this complex problem...")
//!     .send()
//!     .await?;
//!
//! if let Some(thinking) = response.thinking() {
//!     println!("Thinking: {}", thinking);
//! }
//! println!("Answer: {}", response.text().unwrap_or("(no text)"));
//! ```

pub mod batch;
pub mod client;
pub mod error;
pub mod models;
pub mod streaming;
pub mod types;
