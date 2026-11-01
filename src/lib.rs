#![deny(
    clippy::all,
    clippy::pedantic,
    missing_debug_implementations,
    unused_must_use
)]

//! Unofficial Rust SDK for the [Claude API](https://platform.claude.com/docs/en/api/messages)
//! by Anthropic.
//!
//! This crate provides a typed, ergonomic Rust client for the Claude Messages API,
//! including streaming, extended thinking, tool use, server tools, prompt caching,
//! batch processing, token counting, vision, documents, citations, and structured
//! output.
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
//! use claude_agent_rust_sdk::types::{StreamEvent, ContentDelta};
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
//!         StreamEvent::ContentBlockDelta {
//!             delta: ContentDelta::TextDelta { text }, ..
//!         } => print!("{}", text),
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
//!     .model(models::CLAUDE_OPUS_5_5)
//!     .max_tokens(16000)
//!     .thinking_adaptive(None) // the model decides how much to think
//!     .effort(EffortLevel::High)
//!     .user("Solve this complex problem...")
//!     .send()
//!     .await?;
//!
//! if let Some(thinking) = response.thinking() {
//!     println!("Thinking: {}", thinking);
//! }
//! println!("Answer: {}", response.text().unwrap_or("(no text)"));
//! ```
//!
//! # Server Tools
//!
//! Server tools like `web_fetch` and `web_search` execute on Anthropic's servers:
//!
//! ```ignore
//! use claude_agent_rust_sdk::types::ServerTool;
//!
//! let response = client
//!     .messages()
//!     .model(models::CLAUDE_SONNET_4_6)
//!     .max_tokens(4096)
//!     .server_tool(ServerTool::web_fetch().with_max_uses(3))
//!     .user("Summarize https://example.com")
//!     .send()
//!     .await?;
//! ```
//!
//! # Batch Processing
//!
//! Submit batches at 50% pricing via the [`batch::BatchClient`]:
//!
//! ```ignore
//! use std::time::Duration;
//!
//! let batch = client.batches().create(&request).await?;
//! let completed = client.batches()
//!     .poll_until_complete(&batch.id, Duration::from_secs(30))
//!     .await?;
//! let results = client.batches().results(&batch.id).await?;
//! ```
//!
//! # Custom Transport
//!
//! Route all operations through a custom backend via the [`transport::Transport`]
//! trait. This is useful for testing, CLI wrappers, or proxy services:
//!
//! ```ignore
//! use claude_agent_rust_sdk::transport::Transport;
//!
//! #[derive(Debug)]
//! struct MyTransport;
//!
//! #[async_trait::async_trait]
//! impl Transport for MyTransport {
//!     async fn create_message(
//!         &self,
//!         request: &CreateMessageRequest,
//!     ) -> Result<CreateMessageResponse, ClaudeError> {
//!         Err(ClaudeError::Unsupported(
//!             "MyTransport does not implement message creation".into(),
//!         ))
//!     }
//! }
//!
//! let client = ClaudeClient::with_transport(MyTransport);
//! ```
//!
//! # Modules
//!
//! - [`client`] -- [`ClaudeClient`](client::ClaudeClient) and
//!   [`MessageBuilder`](client::builder::MessageBuilder).
//! - [`batch`] -- [`BatchClient`](batch::BatchClient) for the Message Batches API.
//! - [`transport`] -- [`Transport`](transport::Transport) trait for pluggable backends.
//! - [`streaming`] -- [`SseStream`](streaming::SseStream) and SSE parsing.
//! - [`types`] -- All request/response types, content blocks, tools, citations,
//!   and streaming events.
//! - [`error`] -- [`ClaudeError`] enum.
//! - [`models`] -- Model ID constants for all Claude model families.

pub mod agent;
mod agent_runtime;
pub mod batch;
pub mod client;
pub mod error;
pub mod extensions;
pub mod models;
pub mod sessions;
pub mod streaming;
pub mod transport;
pub mod types;

pub use error::{ClaudeError, ResultErrorDetails};
pub use extensions::{
    AgentDefinition, AgentEffort, AgentMemory, BackgroundTaskSummary, BaseHookInput, CanUseTool,
    CanUseToolShadowedWarning, ContextUsageCategory, ContextUsageResponse,
    CwdChangedHookSpecificOutput, ElicitationHookSpecificOutput,
    ElicitationResultHookSpecificOutput, FileChangedHookSpecificOutput, HookCallback, HookContext,
    HookEffort, HookEvent, HookHandler, HookInput, HookInputKind, HookJSONOutput, HookMatcher,
    McpClaudeAiProxyServerConfig, McpHttpServerConfig, McpSdkServerConfig,
    McpSdkServerConfigStatus, McpServerConfig, McpServerConnectionStatus, McpServerInfo,
    McpServerStatus, McpServerStatusConfig, McpSseServerConfig, McpStatusResponse,
    McpStdioServerConfig, McpToolAnnotations, McpToolInfo, MessageDisplayHookSpecificOutput,
    ModelUsage, NotificationHookInput, NotificationHookSpecificOutput, OutputFormat,
    PermissionBehavior, PermissionDeniedHookSpecificOutput, PermissionMode,
    PermissionRequestHookInput, PermissionRequestHookSpecificOutput, PermissionResult,
    PermissionRuleValue, PermissionUpdate, PermissionUpdateDestination, PermissionUpdateType,
    PostModelSwitchHookSpecificOutput, PostToolBatchHookSpecificOutput,
    PostToolUseFailureHookInput, PostToolUseFailureHookSpecificOutput, PostToolUseHookInput,
    PostToolUseHookSpecificOutput, PreCompactHookInput, PreCompactTrigger,
    PreModelSwitchHookSpecificOutput, PreToolUseHookInput, PreToolUseHookSpecificOutput,
    SandboxIgnoreViolations, SandboxNetworkConfig, SandboxSettings, SdkBeta, SdkPluginConfig,
    SdkPluginKind, SessionCronSummary, SessionStartHookSpecificOutput, SettingSource,
    SkillSelection, StopHookInput, StopHookSpecificOutput, SubagentStartHookInput,
    SubagentStartHookSpecificOutput, SubagentStopHookInput, SubagentStopHookSpecificOutput,
    SyncHookJSONOutput, SystemPrompt, TaskBudget, TaskUsage, ThinkingConfig as AgentThinkingConfig,
    ThinkingDisplay, ToolPermissionCallback, ToolPermissionContext, ToolsSpec,
    UserPromptExpansionHookSpecificOutput, UserPromptSubmitHookInput,
    UserPromptSubmitHookSpecificOutput, WorktreeCreateHookSpecificOutput,
};
pub use types::EffortLevel;
