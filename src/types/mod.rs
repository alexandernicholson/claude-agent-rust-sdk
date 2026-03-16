//! Request and response types for the Claude Messages API.

pub mod batch;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

// ---------------------------------------------------------------------------
// Role
// ---------------------------------------------------------------------------

/// The role of a message participant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

// ---------------------------------------------------------------------------
// CacheControl
// ---------------------------------------------------------------------------

/// Prompt-caching directive attached to content blocks, tools, or system
/// prompts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    /// Always `"ephemeral"`.
    #[serde(rename = "type")]
    pub cache_type: String,

    /// Optional TTL such as `"5m"` or `"1h"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

impl CacheControl {
    /// Ephemeral cache with a 5-minute TTL.
    pub fn ephemeral_5m() -> Self {
        Self {
            cache_type: "ephemeral".into(),
            ttl: Some("5m".into()),
        }
    }

    /// Ephemeral cache with a 1-hour TTL.
    pub fn ephemeral_1h() -> Self {
        Self {
            cache_type: "ephemeral".into(),
            ttl: Some("1h".into()),
        }
    }

    /// Ephemeral cache with no explicit TTL (server default).
    pub fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral".into(),
            ttl: None,
        }
    }
}

// ---------------------------------------------------------------------------
// ImageSource
// ---------------------------------------------------------------------------

/// Source data for an image content block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Base-64 encoded image data.
    Base64 {
        media_type: String,
        data: String,
    },
    /// A publicly-accessible URL.
    Url {
        url: String,
    },
    /// A file uploaded via the Files API.
    File {
        file_id: String,
    },
}

// ---------------------------------------------------------------------------
// DocumentSource
// ---------------------------------------------------------------------------

/// Source data for a document content block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    /// Base-64 encoded document data (e.g. PDF).
    Base64 {
        media_type: String,
        data: String,
    },
    /// Plain text content.
    Text {
        media_type: String,
        data: String,
    },
    /// A publicly-accessible URL.
    Url {
        url: String,
    },
    /// Content block array.
    Content {
        content: DocumentContentData,
    },
}

/// The content field inside a `DocumentSource::Content` variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DocumentContentData {
    /// A simple text string.
    Text(String),
    /// An array of content blocks.
    Blocks(Vec<ContentBlock>),
}

// ---------------------------------------------------------------------------
// CitationConfig
// ---------------------------------------------------------------------------

/// Configuration to enable/disable citations on a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationConfig {
    pub enabled: bool,
}

// ---------------------------------------------------------------------------
// ContentBlock (request-side)
// ---------------------------------------------------------------------------

/// A single block inside a message's `content` array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text.
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// An image (base-64, URL, or file).
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// A document (PDF, plain text, URL, or content blocks).
    Document {
        source: DocumentSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<CitationConfig>,
        #[serde(skip_serializing_if = "Option::is_none")]
        context: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    /// A tool invocation produced by the model.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// The result of running a tool, sent back to the model.
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// A thinking block from extended thinking (passed back in multi-turn).
    Thinking {
        thinking: String,
        signature: String,
    },
    /// A redacted thinking block (opaque, passed back in multi-turn).
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        data: String,
    },
}

/// The content of a tool result -- either a plain string or an array of blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    /// A simple text string.
    Text(String),
    /// An array of content blocks (e.g. images, text).
    Blocks(Vec<ContentBlock>),
}

// ---------------------------------------------------------------------------
// MessageContent
// ---------------------------------------------------------------------------

/// The content of a message -- either a plain string or an array of blocks.
///
/// Serialized without a tag so it round-trips as the API expects: a bare
/// string _or_ a JSON array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// A simple text string.
    Text(String),
    /// An array of content blocks.
    Blocks(Vec<ContentBlock>),
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

/// A single message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
}

// ---------------------------------------------------------------------------
// SystemPrompt
// ---------------------------------------------------------------------------

/// The system prompt -- either a plain string or an array of content blocks
/// (which allows attaching `cache_control`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    /// A simple text string.
    Text(String),
    /// An array of content blocks (useful for caching).
    Blocks(Vec<ContentBlock>),
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

/// Definition of a tool the model may call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

// ---------------------------------------------------------------------------
// ServerTool
// ---------------------------------------------------------------------------

/// A server-side tool definition (web search, code execution, etc.).
///
/// Server tools are identified by their `type` field and have tool-specific
/// configuration. Use [`ToolDefinition`] to mix custom and server tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerTool {
    /// The server tool type, e.g. `"web_search_20250305"`.
    #[serde(rename = "type")]
    pub tool_type: String,

    /// Tool name (e.g. `"web_search"`).
    pub name: String,

    /// Maximum number of times the tool can be used in a single request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,

    /// Additional tool-specific configuration (flattened into the JSON).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// A tool definition that can be either a custom tool or a server tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolDefinition {
    /// A custom tool with a user-defined input schema.
    Custom(Tool),
    /// A server-side tool (web search, code execution, etc.).
    Server(ServerTool),
}

// ---------------------------------------------------------------------------
// ToolChoice
// ---------------------------------------------------------------------------

/// Controls how the model selects tools.
#[derive(Debug, Clone)]
pub enum ToolChoice {
    /// Let the model decide.
    Auto,
    /// The model must call at least one tool.
    Any,
    /// The model must call this specific tool.
    Tool { name: String },
    /// The model must not call any tool.
    None,
}

impl Serialize for ToolChoice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        match self {
            ToolChoice::Auto => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("type", "auto")?;
                map.end()
            }
            ToolChoice::Any => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("type", "any")?;
                map.end()
            }
            ToolChoice::Tool { name } => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "tool")?;
                map.serialize_entry("name", name)?;
                map.end()
            }
            ToolChoice::None => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("type", "none")?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(rename = "type")]
            kind: String,
            name: Option<String>,
        }
        let raw = Raw::deserialize(deserializer)?;
        match raw.kind.as_str() {
            "auto" => Ok(ToolChoice::Auto),
            "any" => Ok(ToolChoice::Any),
            "none" => Ok(ToolChoice::None),
            "tool" => {
                let name = raw.name.ok_or_else(|| {
                    serde::de::Error::missing_field("name")
                })?;
                Ok(ToolChoice::Tool { name })
            }
            other => Err(serde::de::Error::unknown_variant(
                other,
                &["auto", "any", "tool", "none"],
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// ThinkingConfig
// ---------------------------------------------------------------------------

/// Configuration for extended thinking.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingConfig {
    /// Extended thinking is enabled with a specific budget.
    Enabled {
        /// Maximum tokens the model can use for internal reasoning.
        /// Must be >= 1024 and less than `max_tokens`.
        budget_tokens: u32,
    },
    /// Extended thinking is disabled (default).
    Disabled {},
    /// Adaptive thinking -- the model decides whether to think.
    /// Recommended for Claude Opus 4.6.
    Adaptive {
        /// Optional budget; if omitted the model decides.
        #[serde(skip_serializing_if = "Option::is_none")]
        budget_tokens: Option<u32>,
    },
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Optional metadata attached to a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

// ---------------------------------------------------------------------------
// OutputConfig
// ---------------------------------------------------------------------------

/// Configuration for output format and effort level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Reasoning effort: `"low"`, `"medium"`, `"high"`, or `"max"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,

    /// Structured output format (e.g. JSON schema).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<OutputFormat>,
}

/// Output format configuration for structured outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputFormat {
    /// Require the model to produce valid JSON matching a schema.
    JsonSchema {
        schema: serde_json::Value,
    },
}

// ---------------------------------------------------------------------------
// CreateMessageRequest
// ---------------------------------------------------------------------------

/// Body of a `POST /v1/messages` request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateMessageRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<Message>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// A content block inside a response message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentBlock {
    /// Text produced by the model.
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<Citation>>,
    },
    /// A tool call the model wants to make.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Extended thinking content.
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Redacted thinking content (opaque).
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        data: String,
    },
}

// ---------------------------------------------------------------------------
// Citation types
// ---------------------------------------------------------------------------

/// A citation in a response text block referencing source material.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Citation {
    /// Character-level location in a plain text document.
    CharLocation {
        cited_text: String,
        document_index: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        document_title: Option<String>,
        start_char_index: u32,
        end_char_index: u32,
    },
    /// Page-level location in a PDF document.
    PageLocation {
        cited_text: String,
        document_index: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        document_title: Option<String>,
        start_page_number: u32,
        end_page_number: u32,
    },
    /// Block-level location in structured content.
    ContentBlockLocation {
        cited_text: String,
        document_index: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        document_title: Option<String>,
        start_block_index: u32,
        end_block_index: u32,
    },
    /// Citation from a web search result.
    WebSearchResultLocation {
        cited_text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_index: Option<String>,
    },
    /// Citation from a search result block.
    SearchResultLocation {
        cited_text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        start_block_index: u32,
        end_block_index: u32,
        search_result_index: u32,
    },
}

/// Token-usage statistics returned with every response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
}

/// Body of a successful `POST /v1/messages` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateMessageResponse {
    pub id: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "type")]
    pub response_type: Option<String>,

    pub model: String,
    pub role: Role,
    pub content: Vec<ResponseContentBlock>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,

    pub usage: Usage,
}

impl CreateMessageResponse {
    /// Return the text of the first `Text` content block, if any.
    pub fn text(&self) -> Option<&str> {
        self.content.iter().find_map(|block| match block {
            ResponseContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
    }

    /// Return the thinking content of the first `Thinking` block, if any.
    pub fn thinking(&self) -> Option<&str> {
        self.content.iter().find_map(|block| match block {
            ResponseContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
            _ => None,
        })
    }

    /// Return all tool use blocks in the response.
    pub fn tool_uses(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.content
            .iter()
            .filter_map(|block| match block {
                ResponseContentBlock::ToolUse { id, name, input } => {
                    Some((id.as_str(), name.as_str(), input))
                }
                _ => None,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Token counting
// ---------------------------------------------------------------------------

/// Body of a `POST /v1/messages/count_tokens` request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountTokensRequest {
    pub model: String,
    pub messages: Vec<Message>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

/// Body of a `POST /v1/messages/count_tokens` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountTokensResponse {
    pub input_tokens: u64,
}

// ---------------------------------------------------------------------------
// API error envelope (used internally by the client)
// ---------------------------------------------------------------------------

/// Shape of an error body returned by the Claude API.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub(crate) struct ApiErrorBody {
    #[serde(rename = "type")]
    pub error_type: String,
    pub error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ApiErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Streaming event types
// ---------------------------------------------------------------------------

/// A server-sent event from the Claude streaming API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// The start of a message, containing the initial `Message` object.
    MessageStart {
        message: CreateMessageResponse,
    },
    /// The start of a content block.
    ContentBlockStart {
        index: u32,
        content_block: StreamContentBlock,
    },
    /// A delta update to a content block.
    ContentBlockDelta {
        index: u32,
        delta: ContentDelta,
    },
    /// The end of a content block.
    ContentBlockStop {
        index: u32,
    },
    /// A delta update to the top-level message (stop_reason, usage).
    MessageDelta {
        delta: MessageDeltaData,
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },
    /// The end of the message stream.
    MessageStop {},
    /// A keepalive ping.
    Ping {},
    /// An error event in the stream.
    Error {
        error: StreamError,
    },
}

/// A content block stub at the start of streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamContentBlock {
    /// A text content block (starts empty).
    Text {
        text: String,
    },
    /// A tool use content block.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// A thinking content block.
    Thinking {
        thinking: String,
    },
}

/// Delta types for streaming content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    /// Incremental text.
    TextDelta {
        text: String,
    },
    /// Incremental JSON for tool input.
    InputJsonDelta {
        partial_json: String,
    },
    /// Incremental thinking text.
    ThinkingDelta {
        thinking: String,
    },
    /// A signature for a thinking block.
    SignatureDelta {
        signature: String,
    },
}

/// Top-level message delta (stop_reason, stop_sequence).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDeltaData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

/// An error inside the event stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Role --

    #[test]
    fn role_serialization_round_trip() {
        let user = Role::User;
        let json = serde_json::to_string(&user).unwrap();
        assert_eq!(json, "\"user\"");
        let back: Role = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Role::User);

        let assistant = Role::Assistant;
        let json = serde_json::to_string(&assistant).unwrap();
        assert_eq!(json, "\"assistant\"");
        let back: Role = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Role::Assistant);
    }

    // -- CacheControl --

    #[test]
    fn cache_control_ephemeral() {
        let cc = CacheControl::ephemeral();
        let json = serde_json::to_value(&cc).unwrap();
        assert_eq!(json["type"], "ephemeral");
        assert!(json.get("ttl").is_none());
    }

    #[test]
    fn cache_control_5m() {
        let cc = CacheControl::ephemeral_5m();
        let json = serde_json::to_value(&cc).unwrap();
        assert_eq!(json["type"], "ephemeral");
        assert_eq!(json["ttl"], "5m");
    }

    #[test]
    fn cache_control_1h() {
        let cc = CacheControl::ephemeral_1h();
        let json = serde_json::to_value(&cc).unwrap();
        assert_eq!(json["type"], "ephemeral");
        assert_eq!(json["ttl"], "1h");
    }

    #[test]
    fn cache_control_round_trip() {
        let cc = CacheControl::ephemeral_1h();
        let json = serde_json::to_string(&cc).unwrap();
        let back: CacheControl = serde_json::from_str(&json).unwrap();
        assert_eq!(back.cache_type, "ephemeral");
        assert_eq!(back.ttl.as_deref(), Some("1h"));
    }

    // -- ImageSource --

    #[test]
    fn image_source_base64_round_trip() {
        let src = ImageSource::Base64 {
            media_type: "image/png".into(),
            data: "iVBORw0KGgo=".into(),
        };
        let json = serde_json::to_value(&src).unwrap();
        assert_eq!(json["type"], "base64");
        assert_eq!(json["media_type"], "image/png");
        let back: ImageSource = serde_json::from_value(json).unwrap();
        match back {
            ImageSource::Base64 { media_type, data } => {
                assert_eq!(media_type, "image/png");
                assert_eq!(data, "iVBORw0KGgo=");
            }
            _ => panic!("expected Base64"),
        }
    }

    #[test]
    fn image_source_url_round_trip() {
        let src = ImageSource::Url {
            url: "https://example.com/img.jpg".into(),
        };
        let json = serde_json::to_value(&src).unwrap();
        assert_eq!(json["type"], "url");
        let back: ImageSource = serde_json::from_value(json).unwrap();
        match back {
            ImageSource::Url { url } => assert_eq!(url, "https://example.com/img.jpg"),
            _ => panic!("expected Url"),
        }
    }

    #[test]
    fn image_source_file_round_trip() {
        let src = ImageSource::File {
            file_id: "file_abc123".into(),
        };
        let json = serde_json::to_value(&src).unwrap();
        assert_eq!(json["type"], "file");
        assert_eq!(json["file_id"], "file_abc123");
        let back: ImageSource = serde_json::from_value(json).unwrap();
        match back {
            ImageSource::File { file_id } => assert_eq!(file_id, "file_abc123"),
            _ => panic!("expected File"),
        }
    }

    // -- DocumentSource --

    #[test]
    fn document_source_base64_round_trip() {
        let src = DocumentSource::Base64 {
            media_type: "application/pdf".into(),
            data: "JVBERi0=".into(),
        };
        let json = serde_json::to_value(&src).unwrap();
        assert_eq!(json["type"], "base64");
        assert_eq!(json["media_type"], "application/pdf");
        let back: DocumentSource = serde_json::from_value(json).unwrap();
        match back {
            DocumentSource::Base64 { media_type, .. } => {
                assert_eq!(media_type, "application/pdf")
            }
            _ => panic!("expected Base64"),
        }
    }

    #[test]
    fn document_source_url_round_trip() {
        let src = DocumentSource::Url {
            url: "https://example.com/doc.pdf".into(),
        };
        let json = serde_json::to_value(&src).unwrap();
        assert_eq!(json["type"], "url");
        let back: DocumentSource = serde_json::from_value(json).unwrap();
        match back {
            DocumentSource::Url { url } => assert_eq!(url, "https://example.com/doc.pdf"),
            _ => panic!("expected Url"),
        }
    }

    // -- ContentBlock --

    #[test]
    fn content_block_text_round_trip() {
        let block = ContentBlock::Text {
            text: "Hello".into(),
            cache_control: None,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "Hello");
        assert!(json.get("cache_control").is_none());
        let back: ContentBlock = serde_json::from_value(json).unwrap();
        match back {
            ContentBlock::Text { text, cache_control } => {
                assert_eq!(text, "Hello");
                assert!(cache_control.is_none());
            }
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn content_block_text_with_cache() {
        let block = ContentBlock::Text {
            text: "Hello".into(),
            cache_control: Some(CacheControl::ephemeral()),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn content_block_image_round_trip() {
        let block = ContentBlock::Image {
            source: ImageSource::Url {
                url: "https://example.com/img.jpg".into(),
            },
            cache_control: None,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "image");
        assert_eq!(json["source"]["type"], "url");
        let back: ContentBlock = serde_json::from_value(json).unwrap();
        match back {
            ContentBlock::Image { source, .. } => match source {
                ImageSource::Url { url } => assert_eq!(url, "https://example.com/img.jpg"),
                _ => panic!("expected Url"),
            },
            _ => panic!("expected Image"),
        }
    }

    #[test]
    fn content_block_document_round_trip() {
        let block = ContentBlock::Document {
            source: DocumentSource::Base64 {
                media_type: "application/pdf".into(),
                data: "JVBERi0=".into(),
            },
            cache_control: None,
            citations: Some(CitationConfig { enabled: true }),
            context: None,
            title: Some("test.pdf".into()),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "document");
        assert_eq!(json["title"], "test.pdf");
        assert_eq!(json["citations"]["enabled"], true);
        let back: ContentBlock = serde_json::from_value(json).unwrap();
        match back {
            ContentBlock::Document { title, citations, .. } => {
                assert_eq!(title.as_deref(), Some("test.pdf"));
                assert!(citations.unwrap().enabled);
            }
            _ => panic!("expected Document"),
        }
    }

    #[test]
    fn content_block_tool_use_round_trip() {
        let block = ContentBlock::ToolUse {
            id: "toolu_123".into(),
            name: "get_weather".into(),
            input: serde_json::json!({"location": "SF"}),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_use");
        assert_eq!(json["name"], "get_weather");
        let back: ContentBlock = serde_json::from_value(json).unwrap();
        match back {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_123");
                assert_eq!(name, "get_weather");
                assert_eq!(input["location"], "SF");
            }
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn content_block_tool_result_string() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "toolu_123".into(),
            content: ToolResultContent::Text("72F".into()),
            is_error: None,
            cache_control: None,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_result");
        assert_eq!(json["content"], "72F");
    }

    #[test]
    fn content_block_thinking_round_trip() {
        let block = ContentBlock::Thinking {
            thinking: "Let me think...".into(),
            signature: "sig123".into(),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "thinking");
        assert_eq!(json["thinking"], "Let me think...");
        assert_eq!(json["signature"], "sig123");
        let back: ContentBlock = serde_json::from_value(json).unwrap();
        match back {
            ContentBlock::Thinking { thinking, signature } => {
                assert_eq!(thinking, "Let me think...");
                assert_eq!(signature, "sig123");
            }
            _ => panic!("expected Thinking"),
        }
    }

    #[test]
    fn content_block_redacted_thinking_round_trip() {
        let block = ContentBlock::RedactedThinking {
            data: "redacted_data".into(),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "redacted_thinking");
        assert_eq!(json["data"], "redacted_data");
        let back: ContentBlock = serde_json::from_value(json).unwrap();
        match back {
            ContentBlock::RedactedThinking { data } => assert_eq!(data, "redacted_data"),
            _ => panic!("expected RedactedThinking"),
        }
    }

    // -- ToolChoice --

    #[test]
    fn tool_choice_auto_round_trip() {
        let tc = ToolChoice::Auto;
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "auto");
        let back: ToolChoice = serde_json::from_value(json).unwrap();
        assert!(matches!(back, ToolChoice::Auto));
    }

    #[test]
    fn tool_choice_any_round_trip() {
        let tc = ToolChoice::Any;
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "any");
        let back: ToolChoice = serde_json::from_value(json).unwrap();
        assert!(matches!(back, ToolChoice::Any));
    }

    #[test]
    fn tool_choice_tool_round_trip() {
        let tc = ToolChoice::Tool {
            name: "get_weather".into(),
        };
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "tool");
        assert_eq!(json["name"], "get_weather");
        let back: ToolChoice = serde_json::from_value(json).unwrap();
        match back {
            ToolChoice::Tool { name } => assert_eq!(name, "get_weather"),
            _ => panic!("expected Tool"),
        }
    }

    #[test]
    fn tool_choice_none_round_trip() {
        let tc = ToolChoice::None;
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "none");
        let back: ToolChoice = serde_json::from_value(json).unwrap();
        assert!(matches!(back, ToolChoice::None));
    }

    // -- ThinkingConfig --

    #[test]
    fn thinking_config_enabled_round_trip() {
        let tc = ThinkingConfig::Enabled {
            budget_tokens: 10000,
        };
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "enabled");
        assert_eq!(json["budget_tokens"], 10000);
        let back: ThinkingConfig = serde_json::from_value(json).unwrap();
        match back {
            ThinkingConfig::Enabled { budget_tokens } => assert_eq!(budget_tokens, 10000),
            _ => panic!("expected Enabled"),
        }
    }

    #[test]
    fn thinking_config_disabled_round_trip() {
        let tc = ThinkingConfig::Disabled {};
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "disabled");
        let back: ThinkingConfig = serde_json::from_value(json).unwrap();
        assert!(matches!(back, ThinkingConfig::Disabled {}));
    }

    #[test]
    fn thinking_config_adaptive_round_trip() {
        let tc = ThinkingConfig::Adaptive {
            budget_tokens: Some(5000),
        };
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "adaptive");
        assert_eq!(json["budget_tokens"], 5000);
        let back: ThinkingConfig = serde_json::from_value(json).unwrap();
        match back {
            ThinkingConfig::Adaptive { budget_tokens } => {
                assert_eq!(budget_tokens, Some(5000))
            }
            _ => panic!("expected Adaptive"),
        }
    }

    #[test]
    fn thinking_config_adaptive_no_budget() {
        let tc = ThinkingConfig::Adaptive {
            budget_tokens: None,
        };
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "adaptive");
        assert!(json.get("budget_tokens").is_none());
    }

    // -- OutputConfig & OutputFormat --

    #[test]
    fn output_config_effort_only() {
        let oc = OutputConfig {
            effort: Some("high".into()),
            format: None,
        };
        let json = serde_json::to_value(&oc).unwrap();
        assert_eq!(json["effort"], "high");
        assert!(json.get("format").is_none());
    }

    #[test]
    fn output_config_json_schema() {
        let oc = OutputConfig {
            effort: None,
            format: Some(OutputFormat::JsonSchema {
                schema: serde_json::json!({"type": "object"}),
            }),
        };
        let json = serde_json::to_value(&oc).unwrap();
        assert_eq!(json["format"]["type"], "json_schema");
        assert_eq!(json["format"]["schema"]["type"], "object");
    }

    // -- MessageContent --

    #[test]
    fn message_content_text_round_trip() {
        let mc = MessageContent::Text("Hello".into());
        let json = serde_json::to_string(&mc).unwrap();
        assert_eq!(json, "\"Hello\"");
        let back: MessageContent = serde_json::from_str(&json).unwrap();
        match back {
            MessageContent::Text(t) => assert_eq!(t, "Hello"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn message_content_blocks_round_trip() {
        let mc = MessageContent::Blocks(vec![ContentBlock::Text {
            text: "Hello".into(),
            cache_control: None,
        }]);
        let json = serde_json::to_value(&mc).unwrap();
        assert!(json.is_array());
        let back: MessageContent = serde_json::from_value(json).unwrap();
        match back {
            MessageContent::Blocks(blocks) => assert_eq!(blocks.len(), 1),
            _ => panic!("expected Blocks"),
        }
    }

    // -- Message --

    #[test]
    fn message_round_trip() {
        let msg = Message {
            role: Role::User,
            content: MessageContent::Text("Hello".into()),
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "Hello");
        let back: Message = serde_json::from_value(json).unwrap();
        assert_eq!(back.role, Role::User);
    }

    // -- SystemPrompt --

    #[test]
    fn system_prompt_text_round_trip() {
        let sp = SystemPrompt::Text("You are helpful.".into());
        let json = serde_json::to_string(&sp).unwrap();
        assert_eq!(json, "\"You are helpful.\"");
    }

    #[test]
    fn system_prompt_blocks_round_trip() {
        let sp = SystemPrompt::Blocks(vec![ContentBlock::Text {
            text: "You are helpful.".into(),
            cache_control: Some(CacheControl::ephemeral()),
        }]);
        let json = serde_json::to_value(&sp).unwrap();
        assert!(json.is_array());
    }

    // -- Tool --

    #[test]
    fn tool_round_trip() {
        let tool = Tool {
            name: "get_weather".into(),
            description: "Get weather".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }),
            cache_control: None,
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["name"], "get_weather");
        assert_eq!(json["input_schema"]["type"], "object");
        assert!(json.get("cache_control").is_none());
        let back: Tool = serde_json::from_value(json).unwrap();
        assert_eq!(back.name, "get_weather");
    }

    #[test]
    fn tool_with_cache() {
        let tool = Tool {
            name: "analyze".into(),
            description: "Analyze text".into(),
            input_schema: serde_json::json!({"type": "object"}),
            cache_control: Some(CacheControl::ephemeral()),
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["cache_control"]["type"], "ephemeral");
    }

    // -- ToolDefinition --

    #[test]
    fn tool_definition_custom_round_trip() {
        let td = ToolDefinition::Custom(Tool {
            name: "calc".into(),
            description: "Calculator".into(),
            input_schema: serde_json::json!({"type": "object"}),
            cache_control: None,
        });
        let json = serde_json::to_value(&td).unwrap();
        assert_eq!(json["name"], "calc");
    }

    // -- Metadata --

    #[test]
    fn metadata_round_trip() {
        let m = Metadata {
            user_id: Some("user_123".into()),
        };
        let json = serde_json::to_value(&m).unwrap();
        assert_eq!(json["user_id"], "user_123");
    }

    #[test]
    fn metadata_none_user_id() {
        let m = Metadata { user_id: None };
        let json = serde_json::to_value(&m).unwrap();
        assert!(json.get("user_id").is_none());
    }

    // -- CreateMessageRequest --

    #[test]
    fn create_message_request_minimal() {
        let req = CreateMessageRequest {
            model: "claude-haiku-4-5".into(),
            max_tokens: 1024,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("Hello".into()),
            }],
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
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "claude-haiku-4-5");
        assert_eq!(json["max_tokens"], 1024);
        // Optional fields should be absent
        assert!(json.get("system").is_none());
        assert!(json.get("temperature").is_none());
        assert!(json.get("thinking").is_none());
        assert!(json.get("service_tier").is_none());
    }

    #[test]
    fn create_message_request_with_thinking() {
        let req = CreateMessageRequest {
            model: "claude-sonnet-4-6".into(),
            max_tokens: 16000,
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("Think hard.".into()),
            }],
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
            thinking: Some(ThinkingConfig::Enabled {
                budget_tokens: 10000,
            }),
            service_tier: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["thinking"]["type"], "enabled");
        assert_eq!(json["thinking"]["budget_tokens"], 10000);
    }

    // -- ResponseContentBlock --

    #[test]
    fn response_content_block_text() {
        let json = serde_json::json!({
            "type": "text",
            "text": "Hello!"
        });
        let block: ResponseContentBlock = serde_json::from_value(json).unwrap();
        match block {
            ResponseContentBlock::Text { text, .. } => assert_eq!(text, "Hello!"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn response_content_block_tool_use() {
        let json = serde_json::json!({
            "type": "tool_use",
            "id": "toolu_123",
            "name": "calc",
            "input": {"expr": "2+2"}
        });
        let block: ResponseContentBlock = serde_json::from_value(json).unwrap();
        match block {
            ResponseContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_123");
                assert_eq!(name, "calc");
                assert_eq!(input["expr"], "2+2");
            }
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn response_content_block_thinking() {
        let json = serde_json::json!({
            "type": "thinking",
            "thinking": "Let me analyze...",
            "signature": "sig_abc"
        });
        let block: ResponseContentBlock = serde_json::from_value(json).unwrap();
        match block {
            ResponseContentBlock::Thinking {
                thinking,
                signature,
            } => {
                assert_eq!(thinking, "Let me analyze...");
                assert_eq!(signature.as_deref(), Some("sig_abc"));
            }
            _ => panic!("expected Thinking"),
        }
    }

    #[test]
    fn response_content_block_redacted_thinking() {
        let json = serde_json::json!({
            "type": "redacted_thinking",
            "data": "redacted_blob"
        });
        let block: ResponseContentBlock = serde_json::from_value(json).unwrap();
        match block {
            ResponseContentBlock::RedactedThinking { data } => {
                assert_eq!(data, "redacted_blob")
            }
            _ => panic!("expected RedactedThinking"),
        }
    }

    // -- Citation --

    #[test]
    fn citation_char_location_round_trip() {
        let c = Citation::CharLocation {
            cited_text: "important fact".into(),
            document_index: 0,
            document_title: Some("doc.txt".into()),
            start_char_index: 100,
            end_char_index: 114,
        };
        let json = serde_json::to_value(&c).unwrap();
        assert_eq!(json["type"], "char_location");
        assert_eq!(json["start_char_index"], 100);
        let back: Citation = serde_json::from_value(json).unwrap();
        match back {
            Citation::CharLocation {
                start_char_index, ..
            } => assert_eq!(start_char_index, 100),
            _ => panic!("expected CharLocation"),
        }
    }

    #[test]
    fn citation_page_location_round_trip() {
        let c = Citation::PageLocation {
            cited_text: "on page 5".into(),
            document_index: 1,
            document_title: None,
            start_page_number: 5,
            end_page_number: 7,
        };
        let json = serde_json::to_value(&c).unwrap();
        assert_eq!(json["type"], "page_location");
        assert_eq!(json["start_page_number"], 5);
    }

    #[test]
    fn citation_web_search_result_round_trip() {
        let c = Citation::WebSearchResultLocation {
            cited_text: "found this".into(),
            title: Some("Result".into()),
            url: "https://example.com".into(),
            encrypted_index: None,
        };
        let json = serde_json::to_value(&c).unwrap();
        assert_eq!(json["type"], "web_search_result_location");
        assert_eq!(json["url"], "https://example.com");
    }

    // -- Usage --

    #[test]
    fn usage_round_trip() {
        let u = Usage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: Some(200),
            cache_read_input_tokens: None,
        };
        let json = serde_json::to_value(&u).unwrap();
        assert_eq!(json["input_tokens"], 100);
        assert_eq!(json["output_tokens"], 50);
        assert_eq!(json["cache_creation_input_tokens"], 200);
        assert!(json.get("cache_read_input_tokens").is_none());
    }

    #[test]
    fn usage_default() {
        let u = Usage::default();
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.output_tokens, 0);
    }

    // -- CreateMessageResponse --

    #[test]
    fn create_message_response_text_helper() {
        let resp = CreateMessageResponse {
            id: "msg_123".into(),
            response_type: Some("message".into()),
            model: "claude-haiku-4-5".into(),
            role: Role::Assistant,
            content: vec![
                ResponseContentBlock::Thinking {
                    thinking: "hmm".into(),
                    signature: None,
                },
                ResponseContentBlock::Text {
                    text: "Hello!".into(),
                    citations: None,
                },
            ],
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: Usage::default(),
        };
        assert_eq!(resp.text(), Some("Hello!"));
        assert_eq!(resp.thinking(), Some("hmm"));
        assert_eq!(resp.tool_uses().len(), 0);
    }

    #[test]
    fn create_message_response_tool_uses_helper() {
        let resp = CreateMessageResponse {
            id: "msg_456".into(),
            response_type: None,
            model: "claude-sonnet-4-6".into(),
            role: Role::Assistant,
            content: vec![ResponseContentBlock::ToolUse {
                id: "toolu_1".into(),
                name: "calc".into(),
                input: serde_json::json!({"x": 1}),
            }],
            stop_reason: Some("tool_use".into()),
            stop_sequence: None,
            usage: Usage::default(),
        };
        let uses = resp.tool_uses();
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].0, "toolu_1");
        assert_eq!(uses[0].1, "calc");
    }

    #[test]
    fn create_message_response_deserialize_full() {
        let json = serde_json::json!({
            "id": "msg_01D7",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-4-6",
            "content": [
                {"type": "text", "text": "The answer is 42."}
            ],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 25,
                "output_tokens": 10,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0
            }
        });
        let resp: CreateMessageResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.id, "msg_01D7");
        assert_eq!(resp.text(), Some("The answer is 42."));
        assert_eq!(resp.usage.input_tokens, 25);
    }

    // -- CountTokensRequest / Response --

    #[test]
    fn count_tokens_request_round_trip() {
        let req = CountTokensRequest {
            model: "claude-haiku-4-5".into(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("Hi".into()),
            }],
            system: None,
            tools: None,
            thinking: None,
            tool_choice: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "claude-haiku-4-5");
    }

    #[test]
    fn count_tokens_response_round_trip() {
        let json = serde_json::json!({"input_tokens": 42});
        let resp: CountTokensResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.input_tokens, 42);
    }

    // -- StreamEvent --

    #[test]
    fn stream_event_message_start() {
        let json = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-opus-4-6",
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 25, "output_tokens": 1}
            }
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamEvent::MessageStart { message } => {
                assert_eq!(message.id, "msg_1");
                assert_eq!(message.usage.input_tokens, 25);
            }
            _ => panic!("expected MessageStart"),
        }
    }

    #[test]
    fn stream_event_content_block_start_text() {
        let json = serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                assert_eq!(index, 0);
                match content_block {
                    StreamContentBlock::Text { text } => assert_eq!(text, ""),
                    _ => panic!("expected Text block"),
                }
            }
            _ => panic!("expected ContentBlockStart"),
        }
    }

    #[test]
    fn stream_event_content_block_start_tool_use() {
        let json = serde_json::json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {}
            }
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamEvent::ContentBlockStart {
                content_block: StreamContentBlock::ToolUse { id, name, .. },
                ..
            } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "get_weather");
            }
            _ => panic!("expected ContentBlockStart with ToolUse"),
        }
    }

    #[test]
    fn stream_event_content_block_start_thinking() {
        let json = serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""}
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamEvent::ContentBlockStart {
                content_block: StreamContentBlock::Thinking { thinking },
                ..
            } => assert_eq!(thinking, ""),
            _ => panic!("expected Thinking block start"),
        }
    }

    #[test]
    fn stream_event_text_delta() {
        let json = serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"}
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamEvent::ContentBlockDelta {
                index,
                delta: ContentDelta::TextDelta { text },
            } => {
                assert_eq!(index, 0);
                assert_eq!(text, "Hello");
            }
            _ => panic!("expected TextDelta"),
        }
    }

    #[test]
    fn stream_event_input_json_delta() {
        let json = serde_json::json!({
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "{\"location\": \"SF\"}"}
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamEvent::ContentBlockDelta {
                delta: ContentDelta::InputJsonDelta { partial_json },
                ..
            } => assert_eq!(partial_json, "{\"location\": \"SF\"}"),
            _ => panic!("expected InputJsonDelta"),
        }
    }

    #[test]
    fn stream_event_thinking_delta() {
        let json = serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "I need to..."}
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamEvent::ContentBlockDelta {
                delta: ContentDelta::ThinkingDelta { thinking },
                ..
            } => assert_eq!(thinking, "I need to..."),
            _ => panic!("expected ThinkingDelta"),
        }
    }

    #[test]
    fn stream_event_signature_delta() {
        let json = serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "EqQBCg=="}
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamEvent::ContentBlockDelta {
                delta: ContentDelta::SignatureDelta { signature },
                ..
            } => assert_eq!(signature, "EqQBCg=="),
            _ => panic!("expected SignatureDelta"),
        }
    }

    #[test]
    fn stream_event_content_block_stop() {
        let json = serde_json::json!({
            "type": "content_block_stop",
            "index": 0
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        assert!(matches!(
            event,
            StreamEvent::ContentBlockStop { index: 0 }
        ));
    }

    #[test]
    fn stream_event_message_delta() {
        let json = serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 15}
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamEvent::MessageDelta { delta, usage } => {
                assert_eq!(delta.stop_reason.as_deref(), Some("end_turn"));
                assert_eq!(usage.unwrap().output_tokens, 15);
            }
            _ => panic!("expected MessageDelta"),
        }
    }

    #[test]
    fn stream_event_message_stop() {
        let json = serde_json::json!({"type": "message_stop"});
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        assert!(matches!(event, StreamEvent::MessageStop {}));
    }

    #[test]
    fn stream_event_ping() {
        let json = serde_json::json!({"type": "ping"});
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        assert!(matches!(event, StreamEvent::Ping {}));
    }

    #[test]
    fn stream_event_error() {
        let json = serde_json::json!({
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"}
        });
        let event: StreamEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamEvent::Error { error } => {
                assert_eq!(error.error_type, "overloaded_error");
                assert_eq!(error.message, "Overloaded");
            }
            _ => panic!("expected Error"),
        }
    }
}
