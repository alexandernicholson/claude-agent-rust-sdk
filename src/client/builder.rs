//! Ergonomic builder for [`CreateMessageRequest`].
//!
//! [`MessageBuilder`] provides a fluent API for constructing message requests.
//! It supports all API parameters including system prompts, tools (custom and
//! server), extended thinking, structured output, prompt caching, and sampling
//! controls.
//!
//! Obtain a builder via [`ClaudeClient::messages`](crate::client::ClaudeClient::messages).
//!
//! # Example
//!
//! ```ignore
//! let response = client
//!     .messages()
//!     .model(models::CLAUDE_SONNET_4_6)
//!     .max_tokens(1024)
//!     .system("You are a helpful assistant.")
//!     .user("Hello!")
//!     .send()
//!     .await?;
//! ```

use crate::client::ClaudeClient;
use crate::error::ClaudeError;
use crate::streaming::SseStream;
use crate::types::{
    CacheControl, ContentBlock, CreateMessageRequest, CreateMessageResponse, Diagnostics,
    EffortLevel, Message, MessageContent, Metadata, OutputConfig, OutputFormat, Role, ServerTool,
    SystemPrompt, ThinkingConfig, Tool, ToolChoice, ToolDefinition, Toolset,
};

/// A fluent builder for constructing and sending a message request.
///
/// ```ignore
/// let response = client
///     .messages()
///     .model("claude-haiku-4-5")
///     .max_tokens(1024)
///     .system("You are a helpful assistant.")
///     .user("Hello!")
///     .send()
///     .await?;
/// ```
#[derive(Debug)]
pub struct MessageBuilder<'a> {
    client: &'a ClaudeClient,
    model: Option<String>,
    max_tokens: Option<u32>,
    messages: Vec<Message>,
    system: Option<SystemPrompt>,
    stop_sequences: Option<Vec<String>>,
    stream: Option<bool>,
    tools: Option<Vec<ToolDefinition>>,
    tool_choice: Option<ToolChoice>,
    metadata: Option<Metadata>,
    cache_control: Option<CacheControl>,
    output_config: Option<OutputConfig>,
    thinking: Option<ThinkingConfig>,
    service_tier: Option<String>,
    speed: Option<String>,
    diagnostics: Option<Diagnostics>,
}

impl<'a> MessageBuilder<'a> {
    pub(crate) fn new(client: &'a ClaudeClient) -> Self {
        Self {
            client,
            model: None,
            max_tokens: None,
            messages: Vec::new(),
            system: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            cache_control: None,
            output_config: None,
            thinking: None,
            service_tier: None,
            speed: None,
            diagnostics: None,
        }
    }

    // ----- required fields --------------------------------------------------

    /// Set the model identifier (e.g. `"claude-haiku-4-5"`).
    #[must_use]
    pub fn model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    /// Set the maximum number of tokens to generate.
    #[must_use]
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    // ----- messages ---------------------------------------------------------

    /// Append a user message with plain text content.
    #[must_use]
    pub fn user(mut self, text: &str) -> Self {
        self.messages.push(Message {
            role: Role::User,
            content: MessageContent::Text(text.to_string()),
        });
        self
    }

    /// Append a user message made of content blocks.
    #[must_use]
    pub fn user_blocks(mut self, blocks: Vec<ContentBlock>) -> Self {
        self.messages.push(Message {
            role: Role::User,
            content: MessageContent::Blocks(blocks),
        });
        self
    }

    /// Append an assistant message with plain text content.
    #[must_use]
    pub fn assistant(mut self, text: &str) -> Self {
        self.messages.push(Message {
            role: Role::Assistant,
            content: MessageContent::Text(text.to_string()),
        });
        self
    }

    /// Append an assistant message made of content blocks.
    #[must_use]
    pub fn assistant_blocks(mut self, blocks: Vec<ContentBlock>) -> Self {
        self.messages.push(Message {
            role: Role::Assistant,
            content: MessageContent::Blocks(blocks),
        });
        self
    }

    /// Append an arbitrary pre-built [`Message`].
    #[must_use]
    pub fn message(mut self, msg: Message) -> Self {
        self.messages.push(msg);
        self
    }

    /// Append a mid-conversation system message (role `system`).
    ///
    /// For the initial system prompt use [`system`](Self::system).
    #[must_use]
    pub fn system_message(mut self, text: &str) -> Self {
        self.messages.push(Message {
            role: Role::System,
            content: MessageContent::Text(text.to_string()),
        });
        self
    }

    // ----- system prompt ----------------------------------------------------

    /// Set a plain-text system prompt.
    #[must_use]
    pub fn system(mut self, text: &str) -> Self {
        self.system = Some(SystemPrompt::Text(text.to_string()));
        self
    }

    /// Set a system prompt with a cache-control directive.
    ///
    /// The text is wrapped in a `ContentBlock::Text` with the given
    /// [`CacheControl`] so the API can cache the prompt prefix.
    #[must_use]
    pub fn system_with_cache(mut self, text: &str, cache: CacheControl) -> Self {
        let block = ContentBlock::Text {
            text: text.to_string(),
            cache_control: Some(cache),
        };

        match &mut self.system {
            Some(SystemPrompt::Blocks(blocks)) => {
                blocks.push(block);
            }
            _ => {
                self.system = Some(SystemPrompt::Blocks(vec![block]));
            }
        }

        self
    }

    // ----- optional parameters ----------------------------------------------

    /// Set one or more stop sequences.
    #[must_use]
    pub fn stop_sequences(mut self, seqs: Vec<String>) -> Self {
        self.stop_sequences = Some(seqs);
        self
    }

    /// Enable or disable streaming (default: non-streaming).
    ///
    /// Note: for streaming, prefer using [`send_stream`](Self::send_stream)
    /// which returns an async stream of events.
    #[must_use]
    pub fn stream(mut self, enabled: bool) -> Self {
        self.stream = Some(enabled);
        self
    }

    /// Add a single custom tool definition.
    #[must_use]
    pub fn tool(mut self, tool: Tool) -> Self {
        let tools = self.tools.get_or_insert_with(Vec::new);
        tools.push(ToolDefinition::Custom(tool));
        self
    }

    /// Add a single server tool definition (`web_fetch`, `web_search`, etc.).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let req = client
    ///     .messages()
    ///     .model("claude-haiku-4-5")
    ///     .max_tokens(1024)
    ///     .user("Summarize https://example.com")
    ///     .server_tool(ServerTool::web_fetch().with_max_uses(1))
    ///     .send()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn server_tool(mut self, tool: ServerTool) -> Self {
        let tools = self.tools.get_or_insert_with(Vec::new);
        tools.push(ToolDefinition::Server(tool));
        self
    }

    /// Add a toolset entry (computer or browser tool family).
    #[must_use]
    pub fn toolset(mut self, toolset: Toolset) -> Self {
        let tools = self.tools.get_or_insert_with(Vec::new);
        tools.push(ToolDefinition::Toolset(toolset));
        self
    }

    /// Provide multiple tool definitions (custom and/or server tools).
    #[must_use]
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Provide custom tool definitions.
    #[must_use]
    pub fn custom_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools.into_iter().map(ToolDefinition::Custom).collect());
        self
    }

    /// Set the tool-choice strategy.
    #[must_use]
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Enable manual extended thinking with a specific token budget.
    ///
    /// The budget must be >= 1024 and less than `max_tokens`. Manual thinking
    /// is deprecated on Claude Opus 4.6 and Sonnet 4.6 and rejected by later
    /// models; use [`thinking_adaptive`](Self::thinking_adaptive) there.
    #[must_use]
    pub fn thinking(mut self, budget_tokens: u32) -> Self {
        self.thinking = Some(ThinkingConfig::Enabled {
            budget_tokens,
            display: None,
        });
        self
    }

    /// Enable adaptive thinking: the model decides when and how much to
    /// think.
    #[must_use]
    pub fn thinking_adaptive(mut self, budget_tokens: Option<u32>) -> Self {
        self.thinking = Some(ThinkingConfig::Adaptive {
            budget_tokens,
            display: None,
        });
        self
    }

    /// Turn off up-front thinking; the model thinks only between tool calls.
    #[must_use]
    pub fn thinking_between_tools(mut self) -> Self {
        self.thinking = Some(ThinkingConfig::BetweenTools {});
        self
    }

    /// Set the full thinking configuration (e.g. to set `display`).
    #[must_use]
    pub fn thinking_config(mut self, config: ThinkingConfig) -> Self {
        self.thinking = Some(config);
        self
    }

    /// Attach request-level metadata.
    #[must_use]
    pub fn metadata(mut self, meta: Metadata) -> Self {
        self.metadata = Some(meta);
        self
    }

    /// Attach top-level cache control.
    #[must_use]
    pub fn cache_control(mut self, cc: CacheControl) -> Self {
        self.cache_control = Some(cc);
        self
    }

    /// Set how much effort Claude spends on output, tool calls, and thinking.
    #[must_use]
    pub fn effort(mut self, effort: EffortLevel) -> Self {
        let config = self.output_config.get_or_insert(OutputConfig {
            effort: None,
            format: None,
        });
        config.effort = Some(effort);
        self
    }

    /// Set a JSON schema for structured output.
    #[must_use]
    pub fn json_schema(mut self, schema: serde_json::Value) -> Self {
        let config = self.output_config.get_or_insert(OutputConfig {
            effort: None,
            format: None,
        });
        config.format = Some(OutputFormat::JsonSchema { schema });
        self
    }

    /// Set the service tier (`"auto"` or `"standard_only"`).
    #[must_use]
    pub fn service_tier(mut self, tier: &str) -> Self {
        self.service_tier = Some(tier.to_string());
        self
    }

    /// Set the inference speed mode (`"standard"` or `"fast"`).
    ///
    /// Some models require the `fast-mode-2026-02-01` beta header for any
    /// `speed` value.
    #[must_use]
    pub fn speed(mut self, speed: &str) -> Self {
        self.speed = Some(speed.to_string());
        self
    }

    /// Request prompt-cache diagnostics against a previous response.
    ///
    /// Pass the previous response's `id` to get
    /// `diagnostics.cache_miss_reason` back, or `None` on the first turn to
    /// opt in with nothing to compare against.
    #[must_use]
    pub fn diagnostics(mut self, previous_message_id: Option<&str>) -> Self {
        self.diagnostics = Some(Diagnostics {
            previous_message_id: previous_message_id.map(str::to_string),
        });
        self
    }

    // ----- build ------------------------------------------------------------

    /// Build the [`CreateMessageRequest`] without sending it.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::InvalidConfig`] if `model` or `max_tokens` have
    /// not been set, or if no messages have been added.
    pub fn build(self) -> Result<CreateMessageRequest, ClaudeError> {
        let model = self
            .model
            .ok_or_else(|| ClaudeError::InvalidConfig("model is required".into()))?;
        let max_tokens = self
            .max_tokens
            .ok_or_else(|| ClaudeError::InvalidConfig("max_tokens is required".into()))?;

        if self.messages.is_empty() {
            return Err(ClaudeError::InvalidConfig(
                "at least one message is required".into(),
            ));
        }

        Ok(CreateMessageRequest {
            model,
            max_tokens,
            messages: self.messages,
            system: self.system,
            stop_sequences: self.stop_sequences,
            stream: self.stream,
            tools: self.tools,
            tool_choice: self.tool_choice,
            metadata: self.metadata,
            cache_control: self.cache_control,
            output_config: self.output_config,
            thinking: self.thinking,
            service_tier: self.service_tier,
            speed: self.speed,
            diagnostics: self.diagnostics,
        })
    }

    // ----- send -------------------------------------------------------------

    /// Build the request, send it, and return the response.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::InvalidConfig`] if `model` or `max_tokens` have
    /// not been set, [`ClaudeError::ApiError`] if the API returns a non-success
    /// status, or [`ClaudeError::NetworkError`] on connection failures.
    pub async fn send(self) -> Result<CreateMessageResponse, ClaudeError> {
        let client = self.client;
        let request = self.build()?;
        client.create_message(&request).await
    }

    /// Build the request and send it as a stream, returning an async stream
    /// of SSE events.
    ///
    /// The request's `stream` field is forced to `true`.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::InvalidConfig`] if `model` or `max_tokens` have
    /// not been set, [`ClaudeError::ApiError`] if the API returns a non-success
    /// status, or [`ClaudeError::NetworkError`] on connection failures.
    pub async fn send_stream(self) -> Result<SseStream, ClaudeError> {
        let client = self.client;
        let request = self.build()?;
        client.create_message_stream(&request).await
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::ClaudeClient;

    fn test_client() -> ClaudeClient {
        ClaudeClient::new("test-key")
    }

    #[test]
    fn build_minimal_request() {
        let client = test_client();
        let req = client
            .messages()
            .model("claude-haiku-4-5")
            .max_tokens(1024)
            .user("Hello")
            .build()
            .unwrap();

        assert_eq!(req.model, "claude-haiku-4-5");
        assert_eq!(req.max_tokens, 1024);
        assert_eq!(req.messages.len(), 1);
        assert!(req.system.is_none());
        assert!(req.thinking.is_none());
        assert!(req.tools.is_none());
    }

    #[test]
    fn build_fails_without_model() {
        let client = test_client();
        let result = client.messages().max_tokens(1024).user("Hi").build();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ClaudeError::InvalidConfig(_)));
    }

    #[test]
    fn build_fails_without_max_tokens() {
        let client = test_client();
        let result = client
            .messages()
            .model("claude-haiku-4-5")
            .user("Hi")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn build_fails_without_messages() {
        let client = test_client();
        let result = client
            .messages()
            .model("claude-haiku-4-5")
            .max_tokens(1024)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn build_with_system_text() {
        let client = test_client();
        let req = client
            .messages()
            .model("claude-haiku-4-5")
            .max_tokens(1024)
            .system("You are helpful.")
            .user("Hi")
            .build()
            .unwrap();

        match req.system {
            Some(SystemPrompt::Text(t)) => assert_eq!(t, "You are helpful."),
            _ => panic!("expected Text system prompt"),
        }
    }

    #[test]
    fn build_with_system_cache() {
        let client = test_client();
        let req = client
            .messages()
            .model("claude-haiku-4-5")
            .max_tokens(1024)
            .system_with_cache("Long prompt", CacheControl::ephemeral())
            .user("Hi")
            .build()
            .unwrap();

        match req.system {
            Some(SystemPrompt::Blocks(blocks)) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    ContentBlock::Text { cache_control, .. } => {
                        assert!(cache_control.is_some());
                    }
                    _ => panic!("expected Text block"),
                }
            }
            _ => panic!("expected Blocks system prompt"),
        }
    }

    #[test]
    fn sampling_parameters_are_never_sent() {
        let req = test_client()
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .build()
            .unwrap();
        let json = serde_json::to_value(&req).unwrap();
        for removed in ["temperature", "top_p", "top_k"] {
            assert!(json.get(removed).is_none(), "{removed} must not be sent");
        }
    }

    #[test]
    fn build_with_speed() {
        let req = test_client()
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .speed("fast")
            .build()
            .unwrap();
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["speed"], "fast");
    }

    #[test]
    fn diagnostics_first_turn_sends_explicit_null() {
        let req = test_client()
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .diagnostics(None)
            .build()
            .unwrap();
        let json = serde_json::to_value(&req).unwrap();
        let diagnostics = json["diagnostics"].as_object().unwrap();
        assert!(diagnostics.contains_key("previous_message_id"));
        assert!(diagnostics["previous_message_id"].is_null());
    }

    #[test]
    fn diagnostics_with_previous_message_id() {
        let req = test_client()
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .diagnostics(Some("msg_prev"))
            .build()
            .unwrap();
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["diagnostics"]["previous_message_id"], "msg_prev");
    }

    #[test]
    fn system_message_appends_system_role_in_order() {
        let req = test_client()
            .messages()
            .model("m")
            .max_tokens(1)
            .user("first")
            .assistant("reply")
            .system_message("Switch to French.")
            .user("second")
            .build()
            .unwrap();
        let json = serde_json::to_value(&req).unwrap();
        let roles: Vec<&str> = json["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["user", "assistant", "system", "user"]);
        assert_eq!(json["messages"][2]["content"], "Switch to French.");
        // The top-level system prompt is untouched.
        assert!(json.get("system").is_none());
    }

    #[test]
    fn build_with_between_tools_thinking() {
        let req = test_client()
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .thinking_between_tools()
            .build()
            .unwrap();
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(
            json["thinking"],
            serde_json::json!({"type": "between_tools"})
        );
    }

    #[test]
    fn build_with_stop_sequences() {
        let client = test_client();
        let req = client
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .stop_sequences(vec!["END".into(), "STOP".into()])
            .build()
            .unwrap();
        assert_eq!(req.stop_sequences.unwrap().len(), 2);
    }

    #[test]
    fn build_with_thinking() {
        let client = test_client();
        let req = client
            .messages()
            .model("claude-sonnet-4-6")
            .max_tokens(16000)
            .user("Think hard")
            .thinking(10000)
            .build()
            .unwrap();

        match req.thinking {
            Some(ThinkingConfig::Enabled { budget_tokens, .. }) => {
                assert_eq!(budget_tokens, 10000);
            }
            _ => panic!("expected Enabled thinking"),
        }
    }

    #[test]
    fn build_with_adaptive_thinking() {
        let client = test_client();
        let req = client
            .messages()
            .model("claude-opus-4-6")
            .max_tokens(16000)
            .user("Think adaptively")
            .thinking_adaptive(Some(5000))
            .build()
            .unwrap();

        match req.thinking {
            Some(ThinkingConfig::Adaptive { budget_tokens, .. }) => {
                assert_eq!(budget_tokens, Some(5000));
            }
            _ => panic!("expected Adaptive thinking"),
        }
    }

    #[test]
    fn build_with_single_tool() {
        let client = test_client();
        let req = client
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .tool(Tool {
                name: "calc".into(),
                description: "Calculator".into(),
                input_schema: serde_json::json!({"type": "object"}),
                cache_control: None,
            })
            .build()
            .unwrap();

        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
    }

    #[test]
    fn build_with_tool_choice() {
        let client = test_client();
        let req = client
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .tool(Tool {
                name: "calc".into(),
                description: "Calculator".into(),
                input_schema: serde_json::json!({"type": "object"}),
                cache_control: None,
            })
            .tool_choice(ToolChoice::Any)
            .build()
            .unwrap();

        let json = serde_json::to_value(&req.tool_choice).unwrap();
        assert_eq!(json["type"], "any");
    }

    #[test]
    fn build_with_effort() {
        let client = test_client();
        let req = client
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .effort(EffortLevel::High)
            .build()
            .unwrap();

        assert_eq!(req.output_config.unwrap().effort, Some(EffortLevel::High));
    }

    #[test]
    fn build_with_json_schema() {
        let client = test_client();
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}}
        });
        let req = client
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .json_schema(schema.clone())
            .build()
            .unwrap();

        match req.output_config.unwrap().format {
            Some(OutputFormat::JsonSchema { schema: s }) => {
                assert_eq!(s["type"], "object");
            }
            _ => panic!("expected JsonSchema format"),
        }
    }

    #[test]
    fn build_with_service_tier() {
        let client = test_client();
        let req = client
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .service_tier("standard_only")
            .build()
            .unwrap();
        assert_eq!(req.service_tier.as_deref(), Some("standard_only"));
    }

    #[test]
    fn build_with_server_tool() {
        let client = test_client();
        let req = client
            .messages()
            .model("m")
            .max_tokens(1)
            .user("Summarize https://example.com")
            .server_tool(
                ServerTool::web_fetch()
                    .with_max_uses(1)
                    .with_max_content_tokens(5000),
            )
            .build()
            .unwrap();

        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        let json = serde_json::to_value(&tools[0]).unwrap();
        assert_eq!(json["type"], "web_fetch_20250910");
        assert_eq!(json["name"], "web_fetch");
        assert_eq!(json["max_uses"], 1);
        assert_eq!(json["max_content_tokens"], 5000);
    }

    #[test]
    fn build_with_mixed_tools() {
        let client = test_client();
        let req = client
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .tool(Tool {
                name: "calc".into(),
                description: "Calculator".into(),
                input_schema: serde_json::json!({"type": "object"}),
                cache_control: None,
            })
            .server_tool(ServerTool::web_fetch().with_max_uses(1))
            .build()
            .unwrap();

        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn build_with_stream_flag() {
        let client = test_client();
        let req = client
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .stream(true)
            .build()
            .unwrap();
        assert_eq!(req.stream, Some(true));
    }

    #[test]
    fn build_multi_turn() {
        let client = test_client();
        let req = client
            .messages()
            .model("m")
            .max_tokens(1)
            .user("Hello")
            .assistant("Hi!")
            .user("How are you?")
            .build()
            .unwrap();

        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[0].role, Role::User);
        assert_eq!(req.messages[1].role, Role::Assistant);
        assert_eq!(req.messages[2].role, Role::User);
    }

    #[test]
    fn build_with_user_blocks() {
        let client = test_client();
        let blocks = vec![
            ContentBlock::Text {
                text: "Describe this:".into(),
                cache_control: None,
            },
            ContentBlock::Image {
                source: crate::types::ImageSource::Url {
                    url: "https://example.com/img.jpg".into(),
                },
                cache_control: None,
                transformations: None,
            },
        ];
        let req = client
            .messages()
            .model("m")
            .max_tokens(1)
            .user_blocks(blocks)
            .build()
            .unwrap();

        match &req.messages[0].content {
            MessageContent::Blocks(b) => assert_eq!(b.len(), 2),
            MessageContent::Text(_) => panic!("expected Blocks"),
        }
    }

    #[test]
    fn build_serializes_correctly() {
        let client = test_client();
        let req = client
            .messages()
            .model("claude-haiku-4-5")
            .max_tokens(1024)
            .system("Be helpful.")
            .user("Hi")
            .thinking(5000)
            .build()
            .unwrap();

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "claude-haiku-4-5");
        assert_eq!(json["max_tokens"], 1024);
        assert_eq!(json["system"], "Be helpful.");
        assert_eq!(json["thinking"]["type"], "enabled");
        assert_eq!(json["thinking"]["budget_tokens"], 5000);
        assert!(json["thinking"].get("display").is_none()); // Optional, not set
        assert!(json.get("stream").is_none()); // Optional, not set
    }
}
