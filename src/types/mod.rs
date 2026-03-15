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
    /// An image (base-64 or URL).
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
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
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

// ---------------------------------------------------------------------------
// MessageContent
// ---------------------------------------------------------------------------

/// The content of a message — either a plain string or an array of blocks.
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

/// The system prompt — either a plain string or an array of content blocks
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
// Metadata
// ---------------------------------------------------------------------------

/// Optional metadata attached to a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
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
    pub stop_sequences: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
}

/// Configuration for output format and effort level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Reasoning effort: `"low"`, `"medium"`, `"high"`, or `"max"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// A content block inside a response message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentBlock {
    /// Text produced by the model.
    Text { text: String },
    /// A tool call the model wants to make.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

/// Token-usage statistics returned with every response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
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
            ResponseContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
    }
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
