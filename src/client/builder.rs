//! Ergonomic builder for [`CreateMessageRequest`].

use crate::client::ClaudeClient;
use crate::error::ClaudeError;
use crate::types::{
    CacheControl, ContentBlock, CreateMessageRequest, CreateMessageResponse, Message,
    MessageContent, Metadata, Role, SystemPrompt, Tool, ToolChoice,
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
pub struct MessageBuilder<'a> {
    client: &'a ClaudeClient,
    model: Option<String>,
    max_tokens: Option<u32>,
    messages: Vec<Message>,
    system: Option<SystemPrompt>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    stop_sequences: Option<Vec<String>>,
    stream: Option<bool>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
    metadata: Option<Metadata>,
    cache_control: Option<CacheControl>,
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
            stop_sequences: None,
            stream: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            cache_control: None,
        }
    }

    // ----- required fields --------------------------------------------------

    /// Set the model identifier (e.g. `"claude-haiku-4-5"`).
    pub fn model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    /// Set the maximum number of tokens to generate.
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    // ----- messages ---------------------------------------------------------

    /// Append a user message with plain text content.
    pub fn user(mut self, text: &str) -> Self {
        self.messages.push(Message {
            role: Role::User,
            content: MessageContent::Text(text.to_string()),
        });
        self
    }

    /// Append a user message made of content blocks.
    pub fn user_blocks(mut self, blocks: Vec<ContentBlock>) -> Self {
        self.messages.push(Message {
            role: Role::User,
            content: MessageContent::Blocks(blocks),
        });
        self
    }

    /// Append an assistant message with plain text content.
    pub fn assistant(mut self, text: &str) -> Self {
        self.messages.push(Message {
            role: Role::Assistant,
            content: MessageContent::Text(text.to_string()),
        });
        self
    }

    /// Append an assistant message made of content blocks.
    pub fn assistant_blocks(mut self, blocks: Vec<ContentBlock>) -> Self {
        self.messages.push(Message {
            role: Role::Assistant,
            content: MessageContent::Blocks(blocks),
        });
        self
    }

    /// Append an arbitrary pre-built [`Message`].
    pub fn message(mut self, msg: Message) -> Self {
        self.messages.push(msg);
        self
    }

    // ----- system prompt ----------------------------------------------------

    /// Set a plain-text system prompt.
    pub fn system(mut self, text: &str) -> Self {
        self.system = Some(SystemPrompt::Text(text.to_string()));
        self
    }

    /// Set a system prompt with a cache-control directive.
    ///
    /// The text is wrapped in a `ContentBlock::Text` with the given
    /// [`CacheControl`] so the API can cache the prompt prefix.
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

    /// Set the sampling temperature.
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Set top-p (nucleus) sampling.
    pub fn top_p(mut self, p: f64) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Set one or more stop sequences.
    pub fn stop_sequences(mut self, seqs: Vec<String>) -> Self {
        self.stop_sequences = Some(seqs);
        self
    }

    /// Enable or disable streaming (default: non-streaming).
    pub fn stream(mut self, enabled: bool) -> Self {
        self.stream = Some(enabled);
        self
    }

    /// Provide tool definitions the model may call.
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the tool-choice strategy.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Attach request-level metadata.
    pub fn metadata(mut self, meta: Metadata) -> Self {
        self.metadata = Some(meta);
        self
    }

    /// Attach top-level cache control.
    pub fn cache_control(mut self, cc: CacheControl) -> Self {
        self.cache_control = Some(cc);
        self
    }

    // ----- send -------------------------------------------------------------

    /// Build the request, send it, and return the response.
    ///
    /// Returns [`ClaudeError::InvalidConfig`] if `model` or `max_tokens` have
    /// not been set.
    pub async fn send(self) -> Result<CreateMessageResponse, ClaudeError> {
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

        let request = CreateMessageRequest {
            model,
            max_tokens,
            messages: self.messages,
            system: self.system,
            temperature: self.temperature,
            top_p: self.top_p,
            stop_sequences: self.stop_sequences,
            stream: self.stream,
            tools: self.tools,
            tool_choice: self.tool_choice,
            metadata: self.metadata,
            cache_control: self.cache_control,
        };

        self.client.create_message(&request).await
    }
}
