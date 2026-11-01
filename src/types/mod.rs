//! Request and response types for the Claude Messages API.

pub mod batch;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::str::FromStr;

// ---------------------------------------------------------------------------
// Role
// ---------------------------------------------------------------------------

/// The role of a message participant.
///
/// [`Role::System`] marks a mid-conversation system message inside
/// `messages`; its content must be text. The initial system prompt still goes
/// in the top-level `system` parameter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
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
    #[must_use]
    pub fn ephemeral_5m() -> Self {
        Self {
            cache_type: "ephemeral".into(),
            ttl: Some("5m".into()),
        }
    }

    /// Ephemeral cache with a 1-hour TTL.
    #[must_use]
    pub fn ephemeral_1h() -> Self {
        Self {
            cache_type: "ephemeral".into(),
            ttl: Some("1h".into()),
        }
    }

    /// Ephemeral cache with no explicit TTL (server default).
    #[must_use]
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
    Base64 { media_type: String, data: String },
    /// A publicly-accessible URL.
    Url { url: String },
    /// A file uploaded via the Files API.
    File { file_id: String },
}

/// Transformations the server applies to an image before the model observes
/// it. Omitted fields keep their default behavior.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImageTransformations {
    /// What to do when the image exceeds the model's maximum image size:
    /// `"downsize"` (default) scales it down to fit; `"error"` rejects the
    /// request with a 400 error naming the image's dimensions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oversized_image: Option<String>,
}

// ---------------------------------------------------------------------------
// DocumentSource
// ---------------------------------------------------------------------------

/// Source data for a document content block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    /// Base-64 encoded document data (e.g. PDF).
    Base64 { media_type: String, data: String },
    /// Plain text content.
    Text { media_type: String, data: String },
    /// A publicly-accessible URL.
    Url { url: String },
    /// Content block array.
    Content { content: DocumentContentData },
    /// A file uploaded via the Files API.
    File { file_id: String },
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
        /// Server-side transformations applied before the model sees the image.
        #[serde(skip_serializing_if = "Option::is_none")]
        transformations: Option<ImageTransformations>,
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
        /// For a toolset member call, the toolset family it belongs to.
        #[serde(skip_serializing_if = "Option::is_none")]
        toolset_name: Option<String>,
    },
    /// The result of running a tool, sent back to the model.
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
        /// For a toolset member result, the `toolset_name` of the paired
        /// `tool_use`.
        #[serde(skip_serializing_if = "Option::is_none")]
        toolset_name: Option<String>,
    },
    /// A thinking block from extended thinking (passed back in multi-turn).
    Thinking { thinking: String, signature: String },
    /// A redacted thinking block (opaque, passed back in multi-turn).
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    /// The caller's browser state after a browser toolset member call. Send
    /// at most one, inside the `tool_result` that answers that call.
    BrowserState {
        /// Every tab open after the call (the full inventory, not a delta).
        tabs: Vec<BrowserTab>,
        /// Tabs opened and download state changes during the call. Omit
        /// when there is nothing to report.
        #[serde(skip_serializing_if = "Option::is_none")]
        state_changes: Option<Vec<BrowserStateChange>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

/// One open tab in a [`ContentBlock::BrowserState`] inventory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserTab {
    /// Caller-assigned identifier, unique within the inventory.
    pub tab_id: String,
    /// Page title. May be empty.
    pub title: String,
    /// Page URL. May be empty.
    pub url: String,
    /// Whether this is the active tab. Exactly one entry is active whenever
    /// the inventory is non-empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active: Option<bool>,
}

/// A side effect of a browser toolset member call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BrowserStateChange {
    /// A tab the call opened that is still open; its entry is in `tabs`.
    TabOpened { tab_id: String },
    /// A download that started during the call.
    DownloadStarted { download_id: String, url: String },
    /// A download that finished during the call.
    DownloadCompleted {
        download_id: String,
        url: String,
        /// Where the file was saved, when another tool can read it there.
        #[serde(skip_serializing_if = "Option::is_none")]
        path: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        size_bytes: Option<u64>,
    },
    /// A download that failed or was cancelled during the call.
    DownloadFailed {
        download_id: String,
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
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

/// Configuration to enable/disable citations on a server tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationsConfig {
    pub enabled: bool,
}

/// A server-side tool definition (web search, web fetch, code execution, etc.).
///
/// Server tools are identified by their `type` field and have tool-specific
/// configuration. Use [`ToolDefinition`] to mix custom and server tools.
///
/// # Example
///
/// ```
/// use claude_agent_rust_sdk::types::ServerTool;
///
/// let tool = ServerTool::web_fetch()
///     .with_max_uses(1)
///     .with_max_content_tokens(5000);
///
/// // Any other tool type or version:
/// let code = ServerTool::new("code_execution_20260521", "code_execution");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerTool {
    /// The server tool type, e.g. `"web_search_20250305"`, `"web_fetch_20250910"`.
    #[serde(rename = "type")]
    pub tool_type: String,

    /// Tool name (e.g. `"web_search"`, `"web_fetch"`).
    pub name: String,

    /// Maximum number of times the tool can be used in a single request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,

    /// Maximum number of content tokens to return from the tool (e.g. `web_fetch`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_content_tokens: Option<u32>,

    /// List of allowed domains for web tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,

    /// List of blocked domains for web tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,

    /// Citations configuration for the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<CitationsConfig>,

    /// List of allowed callers (`"direct"` or a code execution tool type such
    /// as `"code_execution_20260521"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_callers: Option<Vec<String>>,

    /// Which sources contribute the URLs `web_fetch` may fetch. Omitted means
    /// every source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url_sources: Option<WebFetchUrlSources>,

    /// How results consumed by a completed `code_execution` call appear in
    /// the response: `"full"` (default) or `"excluded"`. Supported by
    /// `web_search_20260318` and `web_fetch_20260318`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_inclusion: Option<String>,

    /// Set to `false` to bypass cached content (`web_fetch_20260309` and
    /// later).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_cache: Option<bool>,

    /// Additional tool-specific configuration (flattened into the JSON).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl ServerTool {
    /// Create a server tool of any `type` and `name`, e.g.
    /// `ServerTool::new("web_fetch_20260318", "web_fetch")`.
    #[must_use]
    pub fn new(tool_type: &str, name: &str) -> Self {
        Self {
            tool_type: tool_type.to_string(),
            name: name.to_string(),
            max_uses: None,
            max_content_tokens: None,
            allowed_domains: None,
            blocked_domains: None,
            citations: None,
            allowed_callers: None,
            url_sources: None,
            response_inclusion: None,
            use_cache: None,
            extra: serde_json::Map::new(),
        }
    }

    /// Create a `web_fetch` server tool (`web_fetch_20250910`) with default
    /// configuration. Use [`ServerTool::new`] for later versions such as
    /// `web_fetch_20260318`.
    ///
    /// The `web_fetch` tool allows Claude to fetch the content of URLs
    /// that appear in the conversation. It executes server-side and
    /// does not require the client to handle tool results.
    #[must_use]
    pub fn web_fetch() -> Self {
        Self::new("web_fetch_20250910", "web_fetch")
    }

    /// Create a `web_search` server tool (`web_search_20250305`) with default
    /// configuration. Use [`ServerTool::new`] for later versions such as
    /// `web_search_20260318`.
    #[must_use]
    pub fn web_search() -> Self {
        Self::new("web_search_20250305", "web_search")
    }

    /// Set the maximum number of times the tool can be used per request.
    #[must_use]
    pub fn with_max_uses(mut self, n: u32) -> Self {
        self.max_uses = Some(n);
        self
    }

    /// Set the maximum number of content tokens returned by the tool.
    ///
    /// For `web_fetch`, this limits the fetched page to approximately
    /// this many tokens, keeping costs low.
    #[must_use]
    pub fn with_max_content_tokens(mut self, n: u32) -> Self {
        self.max_content_tokens = Some(n);
        self
    }

    /// Restrict the tool to only fetch from these domains.
    #[must_use]
    pub fn with_allowed_domains(mut self, domains: Vec<String>) -> Self {
        self.allowed_domains = Some(domains);
        self
    }

    /// Block the tool from fetching these domains.
    #[must_use]
    pub fn with_blocked_domains(mut self, domains: Vec<String>) -> Self {
        self.blocked_domains = Some(domains);
        self
    }

    /// Enable or disable citations for the tool.
    #[must_use]
    pub fn with_citations(mut self, enabled: bool) -> Self {
        self.citations = Some(CitationsConfig { enabled });
        self
    }

    /// Restrict which other tools can invoke this tool.
    #[must_use]
    pub fn with_allowed_callers(mut self, callers: Vec<String>) -> Self {
        self.allowed_callers = Some(callers);
        self
    }

    /// Restrict which sources contribute the URLs `web_fetch` may fetch.
    #[must_use]
    pub fn with_url_sources(mut self, sources: WebFetchUrlSources) -> Self {
        self.url_sources = Some(sources);
        self
    }

    /// Set how results consumed by a completed `code_execution` call appear
    /// in the response (`"full"` or `"excluded"`).
    #[must_use]
    pub fn with_response_inclusion(mut self, inclusion: &str) -> Self {
        self.response_inclusion = Some(inclusion.to_string());
        self
    }

    /// Set whether `web_fetch` may serve cached content.
    #[must_use]
    pub fn with_use_cache(mut self, use_cache: bool) -> Self {
        self.use_cache = Some(use_cache);
        self
    }
}

/// Which sources contribute the URLs `web_fetch` may fetch.
///
/// Omitted fields contribute everything. A tool named in an `only` or
/// `except` filter must be declared in the request's `tools`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WebFetchUrlSources {
    /// Whether URLs in user messages are fetchable ([`UrlSource::All`] or
    /// [`UrlSource::None`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_input: Option<UrlSource>,

    /// Which client tools' results contribute fetchable URLs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_tool_results: Option<UrlSource>,

    /// Which server tools' results contribute fetchable URLs. Only
    /// `web_search` and `web_fetch` results ever contribute.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_tool_results: Option<UrlSource>,
}

/// One source filter inside [`WebFetchUrlSources`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UrlSource {
    /// The source contributes in full.
    All {},
    /// The source contributes nothing.
    None {},
    /// Only the named tools' results contribute (tool filters only).
    Only { tools: Vec<UrlSourceTool> },
    /// Every result except the named tools' contributes (tool filters only).
    Except { tools: Vec<UrlSourceTool> },
}

impl UrlSource {
    /// Only results from the named tools contribute.
    #[must_use]
    pub fn only(tool_names: &[&str]) -> Self {
        Self::Only {
            tools: UrlSourceTool::references(tool_names),
        }
    }

    /// Every result except the named tools' contributes.
    #[must_use]
    pub fn except(tool_names: &[&str]) -> Self {
        Self::Except {
            tools: UrlSourceTool::references(tool_names),
        }
    }
}

/// A tool named in a [`UrlSource::Only`] or [`UrlSource::Except`] filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UrlSourceTool {
    /// Names a tool declared in the request's `tools`.
    ToolReference { name: String },
}

impl UrlSourceTool {
    fn references(tool_names: &[&str]) -> Vec<Self> {
        tool_names
            .iter()
            .map(|name| Self::ToolReference {
                name: (*name).to_string(),
            })
            .collect()
    }
}

/// A toolset definition: one `tools` entry, with no `name`, that declares a
/// whole tool family (`computer_toolset_20260801` or
/// `browser_toolset_20260801`).
///
/// The model calls the family's members with `tool_use` blocks whose
/// `toolset_name` names the family; answer them with `tool_result` blocks
/// carrying the same `toolset_name`.
///
/// # Example
///
/// ```
/// use claude_agent_rust_sdk::types::Toolset;
///
/// let computer = Toolset::computer()
///     .with_member_config("zoom", serde_json::json!({"enabled": false}));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Toolset {
    /// The toolset type, e.g. `"computer_toolset_20260801"`.
    #[serde(rename = "type")]
    pub toolset_type: String,

    /// Sparse per-member overrides keyed by member name. Members without an
    /// entry keep their defaults; every member is enabled by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configs: Option<serde_json::Map<String, serde_json::Value>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl Toolset {
    /// Create a toolset entry of any type.
    #[must_use]
    pub fn new(toolset_type: &str) -> Self {
        Self {
            toolset_type: toolset_type.to_string(),
            configs: None,
            cache_control: None,
        }
    }

    /// The computer toolset (`computer_toolset_20260801`).
    #[must_use]
    pub fn computer() -> Self {
        Self::new("computer_toolset_20260801")
    }

    /// The browser toolset (`browser_toolset_20260801`).
    #[must_use]
    pub fn browser() -> Self {
        Self::new("browser_toolset_20260801")
    }

    /// Override one member's configuration, e.g.
    /// `("zoom", json!({"enabled": false}))`.
    #[must_use]
    pub fn with_member_config(mut self, member: &str, config: serde_json::Value) -> Self {
        self.configs
            .get_or_insert_with(serde_json::Map::new)
            .insert(member.to_string(), config);
        self
    }

    /// Attach a cache-control breakpoint to this entry.
    #[must_use]
    pub fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

/// A tool definition: a custom tool, a server tool, or a toolset.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolDefinition {
    /// A custom tool with a user-defined input schema.
    Custom(Tool),
    /// A server-side tool (web search, code execution, etc.).
    Server(ServerTool),
    /// A toolset entry (computer or browser tool family).
    Toolset(Toolset),
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
                let name = raw
                    .name
                    .ok_or_else(|| serde::de::Error::missing_field("name"))?;
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

/// Configuration for Claude's thinking.
///
/// Which variants a model accepts, and what omitting `thinking` means, depend
/// on the model. Manual [`ThinkingConfig::Enabled`] is deprecated on Claude
/// Opus 4.6 and Sonnet 4.6 and rejected by later models, which use
/// [`ThinkingConfig::Adaptive`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingConfig {
    /// Manual extended thinking with a fixed budget.
    Enabled {
        /// Maximum tokens the model can use for internal reasoning.
        /// Must be >= 1024 and less than `max_tokens`.
        budget_tokens: u32,
        /// How thinking appears in the response: `"summarized"` or
        /// `"omitted"` (content redacted, signature kept for multi-turn
        /// continuity). The default depends on the model.
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<String>,
    },
    /// Thinking is turned off. Some models reject this.
    Disabled {},
    /// No up-front thinking: the model thinks only between tool calls.
    BetweenTools {},
    /// Adaptive thinking -- the model decides when and how much to think.
    Adaptive {
        /// Optional budget; if omitted the model decides.
        #[serde(skip_serializing_if = "Option::is_none")]
        budget_tokens: Option<u32>,
        /// How thinking appears in the response: `"summarized"` or
        /// `"omitted"`. The default depends on the model.
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<String>,
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

/// Controls how many tokens Claude spends on a response.
///
/// Effort affects visible output, tool calls, and adaptive thinking. It is a
/// behavioral signal rather than a hard token or monetary budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EffortLevel {
    /// Lowest latency and token use.
    Low,
    /// Balanced capability, latency, and token use.
    Medium,
    /// Deep reasoning and the API default when effort is omitted.
    High,
    /// Extended effort for supported long-horizon agentic workloads.
    #[serde(rename = "xhigh")]
    XHigh,
    /// Maximum capability without an effort-based token constraint.
    Max,
}

impl EffortLevel {
    /// Return the Claude API and CLI wire value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

impl FromStr for EffortLevel {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::XHigh),
            "max" => Ok(Self::Max),
            _ => Err("expected low, medium, high, xhigh, or max"),
        }
    }
}

/// Configuration for output format and effort level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Reasoning effort for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<EffortLevel>,

    /// Structured output format (e.g. JSON schema).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<OutputFormat>,
}

/// Output format configuration for structured outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputFormat {
    /// Require the model to produce valid JSON matching a schema.
    JsonSchema { schema: serde_json::Value },
}

// ---------------------------------------------------------------------------
// CreateMessageRequest
// ---------------------------------------------------------------------------

/// Request-level diagnostics (the `diagnostics` request parameter).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Diagnostics {
    /// The `id` (`msg_...`) of this client's previous response. The server
    /// compares that request's prompt with this one and reports why the
    /// prompt cache could not reuse its prefix. `None` is sent as `null`,
    /// which opts in on a first turn with nothing to compare against.
    pub previous_message_id: Option<String>,
}

/// Body of a `POST /v1/messages` request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateMessageRequest {
    pub model: String,
    /// Maximum tokens to generate. `0` populates the prompt cache without
    /// generating a response.
    pub max_tokens: u32,
    pub messages: Vec<Message>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,

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

    /// Inference speed mode: `"standard"` or `"fast"` (high output tokens
    /// per second). Some models require the `fast-mode-2026-02-01` beta
    /// header for any `speed` value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<String>,

    /// Request-level diagnostics. Supply the previous response's `id` to get
    /// `diagnostics.cache_miss_reason` back.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Diagnostics>,
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
        /// For a toolset member call (e.g. a computer or browser action), the
        /// toolset family. Echo it in the matching `tool_result`.
        #[serde(skip_serializing_if = "Option::is_none")]
        toolset_name: Option<String>,
    },
    /// A server tool invocation (`web_fetch`, `web_search`, etc.).
    ///
    /// Server tools are executed automatically by the API -- the client
    /// does not need to supply results.
    ServerToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// The result of a server tool invocation (e.g. `web_fetch` content).
    WebFetchToolResult {
        tool_use_id: String,
        content: serde_json::Value,
    },
    /// Extended thinking content.
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Redacted thinking content (opaque).
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
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

/// Breakdown of output tokens by category.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// Output tokens generated as internal reasoning (the raw reasoning, not
    /// the possibly shorter summarized text). Always <= `output_tokens`.
    pub thinking_tokens: u64,
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

    /// Breakdown of `output_tokens`, which remains the billed total.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,

    /// The inference speed mode used: `"standard"` or `"fast"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<String>,
}

/// Structured information about why output stopped (`stop_details`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopDetails {
    /// The kind of detail; currently `"refusal"`.
    #[serde(rename = "type")]
    pub details_type: String,

    /// For a refusal, the policy category: `"cyber"`, `"bio"`,
    /// `"frontier_llm"`, `"reasoning_extraction"`, or `"general_harms"`.
    /// `None` when the refusal maps to no named category.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,

    /// Human-readable explanation. The text is not guaranteed to be stable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
}

/// Response-side diagnostics, present when the request supplied
/// [`Diagnostics`] and the prompt cache diverged from the previous request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseDiagnostics {
    /// Why the prompt cache could not fully reuse the previous request's
    /// prefix. `None` means the diagnosis was still pending.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_miss_reason: Option<CacheMissReason>,
}

/// Why the prompt cache missed, reported in [`ResponseDiagnostics`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMissReason {
    /// `"model_changed"`, `"system_changed"`, `"tools_changed"`,
    /// `"messages_changed"`, `"previous_message_not_found"`, or
    /// `"unavailable"`.
    #[serde(rename = "type")]
    pub reason_type: String,

    /// Approximate input tokens that would have been read from cache had the
    /// prefix matched. Absent for `previous_message_not_found` and
    /// `unavailable`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_missed_input_tokens: Option<u64>,
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

    /// `"end_turn"`, `"max_tokens"`, `"stop_sequence"`, `"tool_use"`,
    /// `"pause_turn"`, `"refusal"`, or `"model_context_window_exceeded"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,

    /// Structured detail for the `stop_reason` (e.g. a refusal category).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_details: Option<StopDetails>,

    pub usage: Usage,

    /// Prompt-cache diagnostics, when the request supplied `diagnostics`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<ResponseDiagnostics>,
}

impl CreateMessageResponse {
    /// Return the text of the last `Text` content block, if any.
    ///
    /// When the response includes server tool use (e.g. `web_fetch`), the model
    /// may emit an intermediate text block before the tool call and the actual
    /// answer in a text block after the tool result. Returning the *last* text
    /// block ensures callers get the final answer.
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        self.content.iter().rev().find_map(|block| match block {
            ResponseContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
    }

    /// Return the text of the first `Text` content block, if any.
    #[must_use]
    pub fn first_text(&self) -> Option<&str> {
        self.content.iter().find_map(|block| match block {
            ResponseContentBlock::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
    }

    /// Return all text from every `Text` content block, concatenated with newlines.
    #[must_use]
    pub fn all_text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ResponseContentBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Return the thinking content of the first `Thinking` block, if any.
    #[must_use]
    pub fn thinking(&self) -> Option<&str> {
        self.content.iter().find_map(|block| match block {
            ResponseContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
            _ => None,
        })
    }

    /// Return all tool use blocks in the response.
    #[must_use]
    pub fn tool_uses(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.content
            .iter()
            .filter_map(|block| match block {
                ResponseContentBlock::ToolUse {
                    id, name, input, ..
                } => Some((id.as_str(), name.as_str(), input)),
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

    /// Inference speed mode: `"standard"` or `"fast"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<String>,
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
pub(crate) struct ApiErrorBody {
    #[serde(rename = "type")]
    pub _error_type: String,
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
    MessageStart { message: CreateMessageResponse },
    /// The start of a content block.
    ContentBlockStart {
        index: u32,
        content_block: StreamContentBlock,
    },
    /// A delta update to a content block.
    ContentBlockDelta { index: u32, delta: ContentDelta },
    /// The end of a content block.
    ContentBlockStop { index: u32 },
    /// A delta update to the top-level message (`stop_reason`, usage).
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
    Error { error: StreamError },
}

/// A content block stub at the start of streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamContentBlock {
    /// A text content block (starts empty).
    Text { text: String },
    /// A tool use content block.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        /// For a toolset member call, the toolset family.
        #[serde(skip_serializing_if = "Option::is_none")]
        toolset_name: Option<String>,
    },
    /// A thinking content block.
    Thinking { thinking: String },
}

/// Delta types for streaming content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    /// Incremental text.
    TextDelta { text: String },
    /// Incremental JSON for tool input.
    InputJsonDelta { partial_json: String },
    /// Incremental thinking text.
    ThinkingDelta { thinking: String },
    /// A signature for a thinking block.
    SignatureDelta { signature: String },
}

/// Top-level message delta (`stop_reason`, `stop_sequence`, `stop_details`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDeltaData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_details: Option<StopDetails>,
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
                assert_eq!(media_type, "application/pdf");
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
            ContentBlock::Text {
                text,
                cache_control,
            } => {
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
            transformations: None,
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
            ContentBlock::Document {
                title, citations, ..
            } => {
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
            toolset_name: None,
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "tool_use");
        assert_eq!(json["name"], "get_weather");
        let back: ContentBlock = serde_json::from_value(json).unwrap();
        match back {
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
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
            toolset_name: None,
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
            ContentBlock::Thinking {
                thinking,
                signature,
            } => {
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
            display: None,
        };
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "enabled");
        assert_eq!(json["budget_tokens"], 10000);
        let back: ThinkingConfig = serde_json::from_value(json).unwrap();
        match back {
            ThinkingConfig::Enabled { budget_tokens, .. } => assert_eq!(budget_tokens, 10000),
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
    fn thinking_config_between_tools_round_trip() {
        let back: ThinkingConfig =
            serde_json::from_value(serde_json::json!({"type": "between_tools"})).unwrap();
        assert!(matches!(back, ThinkingConfig::BetweenTools {}));
    }

    #[test]
    fn thinking_config_adaptive_display_omitted() {
        let tc = ThinkingConfig::Adaptive {
            budget_tokens: None,
            display: Some("omitted".into()),
        };
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"type": "adaptive", "display": "omitted"})
        );
    }

    #[test]
    fn thinking_config_adaptive_round_trip() {
        let tc = ThinkingConfig::Adaptive {
            budget_tokens: Some(5000),
            display: None,
        };
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "adaptive");
        assert_eq!(json["budget_tokens"], 5000);
        let back: ThinkingConfig = serde_json::from_value(json).unwrap();
        match back {
            ThinkingConfig::Adaptive { budget_tokens, .. } => {
                assert_eq!(budget_tokens, Some(5000));
            }
            _ => panic!("expected Adaptive"),
        }
    }

    #[test]
    fn thinking_config_adaptive_no_budget() {
        let tc = ThinkingConfig::Adaptive {
            budget_tokens: None,
            display: None,
        };
        let json = serde_json::to_value(&tc).unwrap();
        assert_eq!(json["type"], "adaptive");
        assert!(json.get("budget_tokens").is_none());
    }

    // -- OutputConfig & OutputFormat --

    #[test]
    fn output_config_effort_only() {
        let oc = OutputConfig {
            effort: Some(EffortLevel::High),
            format: None,
        };
        let json = serde_json::to_value(&oc).unwrap();
        assert_eq!(json["effort"], "high");
        assert!(json.get("format").is_none());
    }

    #[test]
    fn effort_levels_round_trip_wire_values() {
        for (wire, level) in [
            ("low", EffortLevel::Low),
            ("medium", EffortLevel::Medium),
            ("high", EffortLevel::High),
            ("xhigh", EffortLevel::XHigh),
            ("max", EffortLevel::Max),
        ] {
            assert_eq!(wire.parse::<EffortLevel>().unwrap(), level);
            assert_eq!(serde_json::to_value(level).unwrap(), wire);
            assert_eq!(level.as_str(), wire);
        }
        assert!("extreme".parse::<EffortLevel>().is_err());
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
            MessageContent::Blocks(_) => panic!("expected Text"),
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
            MessageContent::Text(_) => panic!("expected Blocks"),
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

    // -- ServerTool --

    #[test]
    fn server_tool_web_fetch_constructor() {
        let tool = ServerTool::web_fetch();
        assert_eq!(tool.tool_type, "web_fetch_20250910");
        assert_eq!(tool.name, "web_fetch");
        assert!(tool.max_uses.is_none());
        assert!(tool.max_content_tokens.is_none());
    }

    #[test]
    fn server_tool_web_search_constructor() {
        let tool = ServerTool::web_search();
        assert_eq!(tool.tool_type, "web_search_20250305");
        assert_eq!(tool.name, "web_search");
    }

    #[test]
    fn server_tool_with_max_uses() {
        let tool = ServerTool::web_fetch().with_max_uses(3);
        assert_eq!(tool.max_uses, Some(3));
    }

    #[test]
    fn server_tool_with_max_content_tokens() {
        let tool = ServerTool::web_fetch().with_max_content_tokens(5000);
        assert_eq!(tool.max_content_tokens, Some(5000));
    }

    #[test]
    fn server_tool_with_allowed_domains() {
        let tool = ServerTool::web_fetch().with_allowed_domains(vec!["example.com".into()]);
        assert_eq!(
            tool.allowed_domains.as_ref().unwrap(),
            &vec!["example.com".to_string()]
        );
    }

    #[test]
    fn server_tool_with_blocked_domains() {
        let tool = ServerTool::web_fetch().with_blocked_domains(vec!["private.example.com".into()]);
        assert_eq!(
            tool.blocked_domains.as_ref().unwrap(),
            &vec!["private.example.com".to_string()]
        );
    }

    #[test]
    fn server_tool_with_citations() {
        let tool = ServerTool::web_fetch().with_citations(true);
        assert!(tool.citations.as_ref().unwrap().enabled);
    }

    #[test]
    fn server_tool_with_allowed_callers() {
        let tool = ServerTool::web_fetch().with_allowed_callers(vec!["summarize".into()]);
        assert_eq!(
            tool.allowed_callers.as_ref().unwrap(),
            &vec!["summarize".to_string()]
        );
    }

    #[test]
    fn server_tool_web_fetch_serialization() {
        let tool = ServerTool::web_fetch()
            .with_max_uses(1)
            .with_max_content_tokens(5000);
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "web_fetch_20250910");
        assert_eq!(json["name"], "web_fetch");
        assert_eq!(json["max_uses"], 1);
        assert_eq!(json["max_content_tokens"], 5000);
        // Optional fields should be absent
        assert!(json.get("allowed_domains").is_none());
        assert!(json.get("blocked_domains").is_none());
        assert!(json.get("citations").is_none());
        assert!(json.get("allowed_callers").is_none());
    }

    #[test]
    fn server_tool_web_fetch_full_serialization() {
        let tool = ServerTool::web_fetch()
            .with_max_uses(1)
            .with_max_content_tokens(5000)
            .with_allowed_domains(vec!["example.com".into()])
            .with_blocked_domains(vec!["private.example.com".into()])
            .with_citations(true);
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "web_fetch_20250910");
        assert_eq!(json["name"], "web_fetch");
        assert_eq!(json["max_uses"], 1);
        assert_eq!(json["max_content_tokens"], 5000);
        assert_eq!(json["allowed_domains"][0], "example.com");
        assert_eq!(json["blocked_domains"][0], "private.example.com");
        assert_eq!(json["citations"]["enabled"], true);
    }

    #[test]
    fn server_tool_round_trip() {
        let tool = ServerTool::web_fetch()
            .with_max_uses(2)
            .with_max_content_tokens(3000);
        let json = serde_json::to_value(&tool).unwrap();
        let back: ServerTool = serde_json::from_value(json).unwrap();
        assert_eq!(back.tool_type, "web_fetch_20250910");
        assert_eq!(back.name, "web_fetch");
        assert_eq!(back.max_uses, Some(2));
        assert_eq!(back.max_content_tokens, Some(3000));
    }

    #[test]
    fn tool_definition_server_serialization() {
        let td = ToolDefinition::Server(ServerTool::web_fetch().with_max_uses(1));
        let json = serde_json::to_value(&td).unwrap();
        assert_eq!(json["type"], "web_fetch_20250910");
        assert_eq!(json["name"], "web_fetch");
        assert_eq!(json["max_uses"], 1);
        // Should NOT have "description" or "input_schema" (those are for Custom tools)
        assert!(json.get("description").is_none());
        assert!(json.get("input_schema").is_none());
    }

    #[test]
    fn tool_definition_mixed_custom_and_server() {
        let tools = vec![
            ToolDefinition::Custom(Tool {
                name: "calc".into(),
                description: "Calculator".into(),
                input_schema: serde_json::json!({"type": "object"}),
                cache_control: None,
            }),
            ToolDefinition::Server(ServerTool::web_fetch().with_max_uses(1)),
        ];
        let json = serde_json::to_value(&tools).unwrap();
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        // First: custom tool (has "description")
        assert_eq!(arr[0]["name"], "calc");
        assert_eq!(arr[0]["description"], "Calculator");
        // Second: server tool (has "type" as tool type)
        assert_eq!(arr[1]["type"], "web_fetch_20250910");
        assert_eq!(arr[1]["name"], "web_fetch");
    }

    #[test]
    fn server_tool_no_optional_fields_in_json() {
        let tool = ServerTool::web_fetch();
        let json = serde_json::to_value(&tool).unwrap();
        // Only type and name should be present
        assert_eq!(json["type"], "web_fetch_20250910");
        assert_eq!(json["name"], "web_fetch");
        assert!(json.get("max_uses").is_none());
        assert!(json.get("max_content_tokens").is_none());
        assert!(json.get("allowed_domains").is_none());
        assert!(json.get("blocked_domains").is_none());
        assert!(json.get("citations").is_none());
        assert!(json.get("allowed_callers").is_none());
    }

    #[test]
    fn web_fetch_url_sources_serialize_nested_tags() {
        let tool = ServerTool::new("web_fetch_20260318", "web_fetch")
            .with_url_sources(WebFetchUrlSources {
                user_input: Some(UrlSource::None {}),
                client_tool_results: Some(UrlSource::only(&["search_docs"])),
                server_tool_results: Some(UrlSource::except(&["web_search"])),
            })
            .with_response_inclusion("excluded")
            .with_use_cache(false);
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "web_fetch_20260318");
        assert_eq!(
            json["url_sources"],
            serde_json::json!({
                "user_input": {"type": "none"},
                "client_tool_results": {
                    "type": "only",
                    "tools": [{"type": "tool_reference", "name": "search_docs"}]
                },
                "server_tool_results": {
                    "type": "except",
                    "tools": [{"type": "tool_reference", "name": "web_search"}]
                }
            })
        );
        assert_eq!(json["response_inclusion"], "excluded");
        assert_eq!(json["use_cache"], false);

        let back: ServerTool = serde_json::from_value(json).unwrap();
        assert!(
            back.extra.is_empty(),
            "typed fields must not leak into extra"
        );
        assert!(matches!(
            back.url_sources.unwrap().user_input,
            Some(UrlSource::None {})
        ));
    }

    #[test]
    fn toolset_serializes_without_name() {
        let td = ToolDefinition::Toolset(
            Toolset::computer().with_member_config("zoom", serde_json::json!({"enabled": false})),
        );
        let json = serde_json::to_value(&td).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "type": "computer_toolset_20260801",
                "configs": {"zoom": {"enabled": false}}
            })
        );
    }

    #[test]
    fn tool_definition_deserializes_each_kind() {
        let tools: Vec<ToolDefinition> = serde_json::from_value(serde_json::json!([
            {"name": "calc", "description": "Calculator", "input_schema": {"type": "object"}},
            {"type": "web_search_20260318", "name": "web_search"},
            {"type": "browser_toolset_20260801"}
        ]))
        .unwrap();
        assert!(matches!(tools[0], ToolDefinition::Custom(_)));
        assert!(matches!(tools[1], ToolDefinition::Server(_)));
        match &tools[2] {
            ToolDefinition::Toolset(t) => assert_eq!(t.toolset_type, "browser_toolset_20260801"),
            other => panic!("expected Toolset, got {other:?}"),
        }
    }

    #[test]
    fn toolset_tool_result_with_browser_state() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "toolu_7".into(),
            content: ToolResultContent::Blocks(vec![
                ContentBlock::Text {
                    text: "Clicked.".into(),
                    cache_control: None,
                },
                ContentBlock::BrowserState {
                    tabs: vec![BrowserTab {
                        tab_id: "t1".into(),
                        title: String::new(),
                        url: String::new(),
                        active: Some(true),
                    }],
                    state_changes: Some(vec![
                        BrowserStateChange::TabOpened {
                            tab_id: "t1".into(),
                        },
                        BrowserStateChange::DownloadCompleted {
                            download_id: "d1".into(),
                            url: "https://example.com/a.pdf".into(),
                            path: None,
                            size_bytes: Some(2048),
                        },
                    ]),
                    cache_control: None,
                },
            ]),
            is_error: None,
            cache_control: None,
            toolset_name: Some("browser".into()),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["toolset_name"], "browser");
        let state = &json["content"][1];
        assert_eq!(state["type"], "browser_state");
        assert_eq!(state["tabs"][0]["active"], true);
        assert_eq!(
            state["state_changes"][0],
            serde_json::json!({"type": "tab_opened", "tab_id": "t1"})
        );
        assert_eq!(state["state_changes"][1]["type"], "download_completed");
        assert!(state["state_changes"][1].get("path").is_none());
    }

    #[test]
    fn image_transformations_and_file_document_source() {
        let blocks = vec![
            ContentBlock::Image {
                source: ImageSource::File {
                    file_id: "file_1".into(),
                },
                cache_control: None,
                transformations: Some(ImageTransformations {
                    oversized_image: Some("error".into()),
                }),
            },
            ContentBlock::Document {
                source: DocumentSource::File {
                    file_id: "file_2".into(),
                },
                cache_control: None,
                citations: None,
                context: None,
                title: None,
            },
        ];
        let json = serde_json::to_value(&blocks).unwrap();
        assert_eq!(json[0]["transformations"]["oversized_image"], "error");
        assert_eq!(
            json[1]["source"],
            serde_json::json!({"type": "file", "file_id": "file_2"})
        );
    }

    // -- CitationsConfig --

    #[test]
    fn citations_config_round_trip() {
        let cc = CitationsConfig { enabled: true };
        let json = serde_json::to_value(&cc).unwrap();
        assert_eq!(json["enabled"], true);
        let back: CitationsConfig = serde_json::from_value(json).unwrap();
        assert!(back.enabled);
    }

    // -- ResponseContentBlock: server tool variants --

    #[test]
    fn response_content_block_server_tool_use() {
        let json = serde_json::json!({
            "type": "server_tool_use",
            "id": "srvtoolu_123",
            "name": "web_fetch",
            "input": {"url": "https://example.com"}
        });
        let block: ResponseContentBlock = serde_json::from_value(json).unwrap();
        match block {
            ResponseContentBlock::ServerToolUse { id, name, input } => {
                assert_eq!(id, "srvtoolu_123");
                assert_eq!(name, "web_fetch");
                assert_eq!(input["url"], "https://example.com");
            }
            _ => panic!("expected ServerToolUse"),
        }
    }

    #[test]
    fn response_content_block_web_fetch_tool_result() {
        let json = serde_json::json!({
            "type": "web_fetch_tool_result",
            "tool_use_id": "srvtoolu_123",
            "content": "Page content here"
        });
        let block: ResponseContentBlock = serde_json::from_value(json).unwrap();
        match block {
            ResponseContentBlock::WebFetchToolResult {
                tool_use_id,
                content,
            } => {
                assert_eq!(tool_use_id, "srvtoolu_123");
                assert_eq!(content, "Page content here");
            }
            _ => panic!("expected WebFetchToolResult"),
        }
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
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "claude-haiku-4-5");
        assert_eq!(json["max_tokens"], 1024);
        // Optional fields should be absent
        assert!(json.get("system").is_none());
        assert!(json.get("thinking").is_none());
        assert!(json.get("service_tier").is_none());
        assert!(json.get("speed").is_none());
        assert!(json.get("diagnostics").is_none());
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
            stop_sequences: None,
            stream: None,
            tools: None,
            tool_choice: None,
            metadata: None,
            cache_control: None,
            output_config: None,
            thinking: Some(ThinkingConfig::Enabled {
                budget_tokens: 10000,
                display: None,
            }),
            service_tier: None,
            speed: None,
            diagnostics: None,
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
            ResponseContentBlock::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "toolu_123");
                assert_eq!(name, "calc");
                assert_eq!(input["expr"], "2+2");
            }
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn response_content_block_toolset_tool_use() {
        let json = serde_json::json!({
            "type": "tool_use",
            "id": "toolu_9",
            "name": "left_click",
            "input": {"coordinate": [10, 20]},
            "toolset_name": "computer"
        });
        let block: ResponseContentBlock = serde_json::from_value(json).unwrap();
        match block {
            ResponseContentBlock::ToolUse { toolset_name, .. } => {
                assert_eq!(toolset_name.as_deref(), Some("computer"));
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
                assert_eq!(data, "redacted_blob");
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
            output_tokens_details: None,
            speed: None,
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
            stop_details: None,
            usage: Usage::default(),
            diagnostics: None,
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
                toolset_name: None,
            }],
            stop_reason: Some("tool_use".into()),
            stop_sequence: None,
            stop_details: None,
            usage: Usage::default(),
            diagnostics: None,
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

    #[test]
    fn create_message_response_refusal_details_usage_and_diagnostics() {
        let json = serde_json::json!({
            "id": "msg_02",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [],
            "stop_reason": "refusal",
            "stop_sequence": null,
            "stop_details": {
                "type": "refusal",
                "category": "cyber",
                "explanation": "Could enable malware development."
            },
            "usage": {
                "input_tokens": 30,
                "output_tokens": 12,
                "output_tokens_details": {"thinking_tokens": 8},
                "speed": "fast"
            },
            "diagnostics": {
                "cache_miss_reason": {"type": "tools_changed", "cache_missed_input_tokens": 1200}
            }
        });
        let resp: CreateMessageResponse = serde_json::from_value(json).unwrap();
        let details = resp.stop_details.unwrap();
        assert_eq!(details.details_type, "refusal");
        assert_eq!(details.category.as_deref(), Some("cyber"));
        assert_eq!(resp.usage.output_tokens_details.unwrap().thinking_tokens, 8);
        assert_eq!(resp.usage.speed.as_deref(), Some("fast"));
        let reason = resp.diagnostics.unwrap().cache_miss_reason.unwrap();
        assert_eq!(reason.reason_type, "tools_changed");
        assert_eq!(reason.cache_missed_input_tokens, Some(1200));
    }

    #[test]
    fn create_message_response_pending_and_tokenless_cache_miss_reasons() {
        let pending: ResponseDiagnostics =
            serde_json::from_value(serde_json::json!({"cache_miss_reason": null})).unwrap();
        assert!(pending.cache_miss_reason.is_none());

        let not_found: CacheMissReason =
            serde_json::from_value(serde_json::json!({"type": "previous_message_not_found"}))
                .unwrap();
        assert_eq!(not_found.reason_type, "previous_message_not_found");
        assert!(not_found.cache_missed_input_tokens.is_none());
    }

    #[test]
    fn create_message_response_null_stop_details() {
        let json = serde_json::json!({
            "id": "msg_03",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [],
            "stop_reason": "model_context_window_exceeded",
            "stop_details": null,
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let resp: CreateMessageResponse = serde_json::from_value(json).unwrap();
        assert_eq!(
            resp.stop_reason.as_deref(),
            Some("model_context_window_exceeded")
        );
        assert!(resp.stop_details.is_none());
    }

    #[test]
    fn message_delta_carries_stop_details() {
        let json = serde_json::json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "refusal",
                "stop_sequence": null,
                "stop_details": {"type": "refusal", "category": null, "explanation": null}
            },
            "usage": {"output_tokens": 3}
        });
        match serde_json::from_value::<StreamEvent>(json).unwrap() {
            StreamEvent::MessageDelta { delta, .. } => {
                let details = delta.stop_details.unwrap();
                assert_eq!(details.details_type, "refusal");
                assert!(details.category.is_none());
            }
            _ => panic!("expected MessageDelta"),
        }
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
            speed: None,
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
        assert!(matches!(event, StreamEvent::ContentBlockStop { index: 0 }));
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
