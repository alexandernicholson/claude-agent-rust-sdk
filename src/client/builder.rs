//! Ergonomic builder for [`CreateMessageRequest`].

use crate::client::ClaudeClient;
use crate::error::ClaudeError;
use crate::streaming::SseStream;
use crate::types::{
    CacheControl, ContentBlock, CreateMessageRequest, CreateMessageResponse, Message,
    MessageContent, Metadata, OutputConfig, OutputFormat, Role, ServerTool, SystemPrompt,
    ThinkingConfig, Tool, ToolChoice, ToolDefinition,
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
///     .temperature(0.7)
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
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    stop_sequences: Option<Vec<String>>,
    stream: Option<bool>,
    tools: Option<Vec<ToolDefinition>>,
    tool_choice: Option<ToolChoice>,
    metadata: Option<Metadata>,
    cache_control: Option<CacheControl>,
    output_config: Option<OutputConfig>,
    thinking: Option<ThinkingConfig>,
    service_tier: Option<String>,
}

impl<'a> MessageBuilder<'a> {
    pub(crate) fn new(client: &'a ClaudeClient) -> Self {
        Self {
            client,
            model: None,
            max_tokens: None,
            messages: Vec::new(),
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            cache_control: None,
            output_config: None,
            thinking: None,
            service_tier: None,
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

    /// Set the sampling temperature (0.0 to 1.0).
    #[must_use]
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Set top-p (nucleus) sampling.
    #[must_use]
    pub fn top_p(mut self, p: f64) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Set top-k sampling (only sample from top K options per token).
    #[must_use]
    pub fn top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k);
        self
    }

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

    /// Enable extended thinking with a specific token budget.
    ///
    /// The budget must be >= 1024 and less than `max_tokens`.
    #[must_use]
    pub fn thinking(mut self, budget_tokens: u32) -> Self {
        self.thinking = Some(ThinkingConfig::Enabled { budget_tokens });
        self
    }

    /// Enable adaptive thinking (recommended for Claude Opus 4.6).
    ///
    /// The model decides whether and how much to think.
    #[must_use]
    pub fn thinking_adaptive(mut self, budget_tokens: Option<u32>) -> Self {
        self.thinking = Some(ThinkingConfig::Adaptive { budget_tokens });
        self
    }

    /// Set the full thinking configuration.
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

    /// Set the reasoning effort level (`"low"`, `"medium"`, `"high"`, `"max"`).
    #[must_use]
    pub fn effort(mut self, effort: &str) -> Self {
        let config = self.output_config.get_or_insert(OutputConfig {
            effort: None,
            format: None,
        });
        config.effort = Some(effort.to_string());
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
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            stop_sequences: self.stop_sequences,
            stream: self.stream,
            tools: self.tools,
            tool_choice: self.tool_choice,
            metadata: self.metadata,
            cache_control: self.cache_control,
            output_config: self.output_config,
            thinking: self.thinking,
            service_tier: self.service_tier,
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
        assert!(req.temperature.is_none());
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
        let result = client.messages().model("claude-haiku-4-5").user("Hi").build();
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
    fn build_with_temperature() {
        let client = test_client();
        let req = client
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .temperature(0.7)
            .build()
            .unwrap();
        assert_eq!(req.temperature, Some(0.7));
    }

    #[test]
    fn build_with_top_p() {
        let client = test_client();
        let req = client
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .top_p(0.9)
            .build()
            .unwrap();
        assert_eq!(req.top_p, Some(0.9));
    }

    #[test]
    fn build_with_top_k() {
        let client = test_client();
        let req = client
            .messages()
            .model("m")
            .max_tokens(1)
            .user("x")
            .top_k(40)
            .build()
            .unwrap();
        assert_eq!(req.top_k, Some(40));
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
            Some(ThinkingConfig::Enabled { budget_tokens }) => {
                assert_eq!(budget_tokens, 10000)
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
            Some(ThinkingConfig::Adaptive { budget_tokens }) => {
                assert_eq!(budget_tokens, Some(5000))
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
            .effort("high")
            .build()
            .unwrap();

        assert_eq!(req.output_config.unwrap().effort.as_deref(), Some("high"));
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
            .server_tool(ServerTool::web_fetch().with_max_uses(1).with_max_content_tokens(5000))
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
            _ => panic!("expected Blocks"),
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
            .temperature(0.5)
            .thinking(5000)
            .build()
            .unwrap();

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "claude-haiku-4-5");
        assert_eq!(json["max_tokens"], 1024);
        assert_eq!(json["system"], "Be helpful.");
        assert_eq!(json["temperature"], 0.5);
        assert_eq!(json["thinking"]["type"], "enabled");
        assert_eq!(json["thinking"]["budget_tokens"], 5000);
        assert!(json.get("top_k").is_none()); // Optional, not set
        assert!(json.get("stream").is_none()); // Optional, not set
    }
}
