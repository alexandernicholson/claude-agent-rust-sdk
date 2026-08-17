//! Claude Code Agent SDK protocol primitives.
//!
//! This module implements the bidirectional NDJSON control protocol used by
//! Claude Code. Transport implementations own process or network I/O; this
//! crate owns typed messages, the initialization handshake, permission gates,
//! and in-process MCP tool dispatch.

use crate::error::ClaudeError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use uuid::Uuid;

/// MCP protocol version advertised by in-process SDK servers.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Permission behavior passed to Claude Code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PermissionMode {
    /// Use Claude Code's normal permission behavior.
    Default,
    /// Automatically accept file edits.
    AcceptEdits,
    /// Planning only; do not execute tools.
    Plan,
    /// Deny tools that would otherwise require an interactive prompt.
    #[default]
    DontAsk,
    /// Bypass every Claude Code permission check.
    ///
    /// Hosts should avoid this for unattended workloads and enforce their own
    /// deterministic tool gate when it is unavoidable.
    BypassPermissions,
}

impl PermissionMode {
    /// Return the Claude Code command-line representation.
    #[must_use]
    pub const fn as_cli_value(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::AcceptEdits => "acceptEdits",
            Self::Plan => "plan",
            Self::DontAsk => "dontAsk",
            Self::BypassPermissions => "bypassPermissions",
        }
    }
}

/// Filesystem setting layer loaded by Claude Code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingSource {
    /// User-wide settings.
    User,
    /// Project settings.
    Project,
    /// Project-local settings.
    Local,
}

impl SettingSource {
    /// Return the Claude Code command-line representation.
    #[must_use]
    pub const fn as_cli_value(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Project => "project",
            Self::Local => "local",
        }
    }
}

/// Options shared by Agent SDK transports.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct AgentOptions {
    /// System prompt. `None` means an empty prompt, isolating the host from
    /// Claude Code's interactive default prompt.
    pub system_prompt: Option<String>,
    /// Built-in tools exposed to the model. `None` uses Claude Code defaults;
    /// `Some(Vec::new())` disables every built-in tool.
    pub tools: Option<Vec<String>>,
    /// Tools approved without an interactive prompt.
    pub allowed_tools: Vec<String>,
    /// Tools removed from the model's context.
    pub disallowed_tools: Vec<String>,
    /// Permission behavior for this process.
    pub permission_mode: PermissionMode,
    /// Primary model or model alias.
    pub model: Option<String>,
    /// Fallback model or model alias.
    pub fallback_model: Option<String>,
    /// Maximum agent turns for one invocation.
    pub max_turns: Option<u32>,
    /// Maximum API spend for one invocation.
    pub max_budget_usd: Option<f64>,
    /// Working directory visible to Claude Code.
    pub cwd: Option<PathBuf>,
    /// Explicit Claude Code executable path.
    pub cli_path: Option<PathBuf>,
    /// Additional directories exposed to Claude Code.
    pub add_dirs: Vec<PathBuf>,
    /// Environment overrides for the Claude Code child process.
    pub env: BTreeMap<String, String>,
    /// Inherited environment variables removed from the child process before
    /// applying `env` overrides.
    pub env_remove: Vec<String>,
    /// Existing Claude session to resume.
    pub resume: Option<Uuid>,
    /// Explicit session identifier for a new conversation.
    pub session_id: Option<Uuid>,
    /// Resume into a new session instead of mutating the original session.
    pub fork_session: bool,
    /// Ignore MCP configuration outside the supplied SDK servers.
    pub strict_mcp_config: bool,
    /// Filesystem setting layers to load. `Some(Vec::new())` provides SDK
    /// isolation; `None` uses Claude Code defaults.
    pub setting_sources: Option<Vec<SettingSource>>,
    /// Include partial assistant events in the message stream.
    pub include_partial_messages: bool,
    /// Include hook lifecycle events in the message stream.
    pub include_hook_events: bool,
    /// Maximum accepted bytes in one NDJSON frame.
    pub max_buffer_size: usize,
    /// Timeout for the initialize control request.
    pub initialize_timeout: std::time::Duration,
    /// Additional CLI flags. Keys exclude the leading `--`; `None` is a
    /// boolean flag.
    pub extra_args: BTreeMap<String, Option<String>>,
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            system_prompt: None,
            tools: Some(Vec::new()),
            allowed_tools: Vec::new(),
            disallowed_tools: Vec::new(),
            permission_mode: PermissionMode::DontAsk,
            model: None,
            fallback_model: None,
            max_turns: Some(24),
            max_budget_usd: None,
            cwd: None,
            cli_path: None,
            add_dirs: Vec::new(),
            env: BTreeMap::new(),
            env_remove: Vec::new(),
            resume: None,
            session_id: None,
            fork_session: false,
            strict_mcp_config: true,
            setting_sources: Some(Vec::new()),
            include_partial_messages: false,
            include_hook_events: false,
            max_buffer_size: 8 * 1024 * 1024,
            initialize_timeout: std::time::Duration::from_mins(1),
            extra_args: BTreeMap::new(),
        }
    }
}

impl AgentOptions {
    /// Validate invariants before a transport starts a process.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::InvalidConfig`] when any option violates the
    /// Claude Code CLI protocol invariants.
    pub fn validate(&self) -> Result<(), ClaudeError> {
        if self.max_turns == Some(0) {
            return Err(ClaudeError::InvalidConfig(
                "agent max_turns must be greater than zero".into(),
            ));
        }
        if self
            .max_budget_usd
            .is_some_and(|budget| !budget.is_finite() || budget <= 0.0)
        {
            return Err(ClaudeError::InvalidConfig(
                "agent max_budget_usd must be finite and greater than zero".into(),
            ));
        }
        if self.max_buffer_size == 0 {
            return Err(ClaudeError::InvalidConfig(
                "agent max_buffer_size must be greater than zero".into(),
            ));
        }
        if self.resume.is_some() && self.session_id.is_some() && !self.fork_session {
            return Err(ClaudeError::InvalidConfig(
                "agent session_id requires fork_session when resume is set".into(),
            ));
        }
        for (key, value) in &self.env {
            if invalid_environment_name(key) || value.contains('\0') {
                return Err(ClaudeError::InvalidConfig(format!(
                    "invalid agent environment override: {key}"
                )));
            }
        }
        for key in &self.env_remove {
            if invalid_environment_name(key) || self.env.contains_key(key) {
                return Err(ClaudeError::InvalidConfig(format!(
                    "invalid or conflicting removed environment variable: {key}"
                )));
            }
        }
        for key in self.extra_args.keys() {
            if key.is_empty() || key.starts_with('-') || key.contains('=') {
                return Err(ClaudeError::InvalidConfig(format!(
                    "invalid agent CLI flag name: {key}"
                )));
            }
        }
        Ok(())
    }
}

fn invalid_environment_name(name: &str) -> bool {
    name.is_empty() || name.contains('=') || name.contains('\0')
}

/// Metadata passed to a transport so it can expose an SDK MCP server to Claude
/// Code without owning the in-process implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdkMcpServerDescriptor {
    /// Name used in the Claude Code MCP configuration.
    pub name: String,
    /// Informational server version.
    pub version: String,
}

/// Bidirectional transport for the Claude Code Agent SDK protocol.
#[async_trait]
pub trait AgentTransport: fmt::Debug + Send {
    /// Establish the transport and prepare stdin/stdout framing.
    async fn connect(
        &mut self,
        options: &AgentOptions,
        mcp_servers: &[SdkMcpServerDescriptor],
    ) -> Result<(), ClaudeError>;

    /// Write one JSON frame. The implementation adds NDJSON framing.
    async fn write(&mut self, frame: &Value) -> Result<(), ClaudeError>;

    /// Read one JSON frame, or `None` after clean end-of-stream.
    async fn read(&mut self) -> Result<Option<Value>, ClaudeError>;

    /// Close the input half while continuing to drain output.
    async fn end_input(&mut self) -> Result<(), ClaudeError>;

    /// Terminate the transport and release all resources.
    async fn close(&mut self) -> Result<(), ClaudeError>;
}

/// Host decision for a Claude Code permission request.
#[derive(Debug, Clone, PartialEq)]
pub enum PermissionDecision {
    /// Permit execution, optionally replacing the proposed input.
    Allow { updated_input: Option<Value> },
    /// Deny execution. `interrupt` stops the current agent turn.
    Deny { message: String, interrupt: bool },
}

/// Host policy for tools that reach Claude Code's interactive permission path.
#[async_trait]
pub trait ToolPermissionHandler: fmt::Debug + Send + Sync {
    /// Decide whether Claude Code may use `tool_name` with `input`.
    async fn can_use_tool(
        &self,
        tool_name: &str,
        input: &Value,
    ) -> Result<PermissionDecision, ClaudeError>;
}

/// MCP tool safety hints shown to the model and host.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolAnnotations {
    /// Tool only reads external state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_only_hint: Option<bool>,
    /// Tool can perform a destructive operation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub destructive_hint: Option<bool>,
    /// Repeating an identical invocation is safe.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idempotent_hint: Option<bool>,
    /// Tool can communicate with arbitrary external systems.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub open_world_hint: Option<bool>,
}

/// Text content returned by an SDK MCP tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolContent {
    /// MCP content discriminator.
    #[serde(rename = "type")]
    pub kind: String,
    /// Text delivered to the model.
    pub text: String,
}

impl ToolContent {
    /// Create an MCP text content item.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            kind: "text".into(),
            text: text.into(),
        }
    }
}

/// Result returned by an SDK MCP tool implementation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallResult {
    /// Content delivered to the model.
    pub content: Vec<ToolContent>,
    /// Whether this is an expected tool-level error.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
}

impl ToolCallResult {
    /// Create a successful text result.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ToolContent::text(text)],
            is_error: false,
        }
    }

    /// Serialize a value as compact JSON in a successful text result.
    ///
    /// # Errors
    ///
    /// Returns an error when `value` cannot be serialized as JSON.
    pub fn json(value: &Value) -> Result<Self, ClaudeError> {
        Ok(Self::text(serde_json::to_string(value)?))
    }

    /// Create an expected tool-level error without failing the MCP bridge.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![ToolContent::text(message)],
            is_error: true,
        }
    }
}

type ToolFuture = Pin<Box<dyn Future<Output = Result<ToolCallResult, ClaudeError>> + Send>>;
type ToolHandler = dyn Fn(Value) -> ToolFuture + Send + Sync;

/// In-process MCP tool definition.
#[derive(Clone)]
pub struct SdkMcpTool {
    name: String,
    description: String,
    input_schema: Value,
    annotations: ToolAnnotations,
    handler: Arc<ToolHandler>,
}

impl fmt::Debug for SdkMcpTool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SdkMcpTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("input_schema", &self.input_schema)
            .field("annotations", &self.annotations)
            .finish_non_exhaustive()
    }
}

impl SdkMcpTool {
    /// Define an in-process MCP tool.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::InvalidConfig`] when the tool name is invalid or
    /// `input_schema` is not a JSON object.
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        annotations: ToolAnnotations,
        handler: F,
    ) -> Result<Self, ClaudeError>
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolCallResult, ClaudeError>> + Send + 'static,
    {
        let name = name.into();
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(ClaudeError::InvalidConfig(format!(
                "invalid MCP tool name: {name}"
            )));
        }
        if !input_schema.is_object() {
            return Err(ClaudeError::InvalidConfig(format!(
                "MCP tool {name} input schema must be a JSON object"
            )));
        }
        Ok(Self {
            name,
            description: description.into(),
            input_schema,
            annotations,
            handler: Arc::new(move |input| Box::pin(handler(input))),
        })
    }
}

/// In-process MCP server exposed through the Agent SDK control channel.
#[derive(Debug, Clone)]
pub struct SdkMcpServer {
    name: String,
    version: String,
    tools: BTreeMap<String, SdkMcpTool>,
}

impl SdkMcpServer {
    /// Create an empty SDK MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::InvalidConfig`] when `name` is not a valid MCP
    /// server identifier.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Result<Self, ClaudeError> {
        let name = name.into();
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(ClaudeError::InvalidConfig(format!(
                "invalid SDK MCP server name: {name}"
            )));
        }
        Ok(Self {
            name,
            version: version.into(),
            tools: BTreeMap::new(),
        })
    }

    /// Add a tool. Duplicate names are rejected.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::InvalidConfig`] when a tool with the same name
    /// is already registered.
    pub fn add_tool(&mut self, tool: SdkMcpTool) -> Result<(), ClaudeError> {
        if self.tools.contains_key(&tool.name) {
            return Err(ClaudeError::InvalidConfig(format!(
                "duplicate MCP tool: {}",
                tool.name
            )));
        }
        self.tools.insert(tool.name.clone(), tool);
        Ok(())
    }

    /// Return transport-visible metadata.
    #[must_use]
    pub fn descriptor(&self) -> SdkMcpServerDescriptor {
        SdkMcpServerDescriptor {
            name: self.name.clone(),
            version: self.version.clone(),
        }
    }

    async fn handle(&self, message: &Value) -> Value {
        let id = message.get("id").cloned().unwrap_or(Value::Null);
        let method = message.get("method").and_then(Value::as_str).unwrap_or("");
        match method {
            "initialize" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self.name, "version": self.version}
                }
            }),
            "notifications/initialized" => json!({"jsonrpc": "2.0", "result": {}}),
            "tools/list" => {
                let tools = self
                    .tools
                    .values()
                    .map(|tool| {
                        json!({
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.input_schema,
                            "annotations": tool.annotations
                        })
                    })
                    .collect::<Vec<_>>();
                json!({"jsonrpc": "2.0", "id": id, "result": {"tools": tools}})
            }
            "tools/call" => {
                let params = message.get("params").and_then(Value::as_object);
                let name = params
                    .and_then(|value| value.get("name"))
                    .and_then(Value::as_str);
                let arguments = params
                    .and_then(|value| value.get("arguments"))
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let Some(name) = name else {
                    return mcp_error(&id, -32_602, "tools/call is missing params.name");
                };
                let Some(tool) = self.tools.get(name) else {
                    return mcp_error(&id, -32_601, &format!("tool '{name}' not found"));
                };
                match (tool.handler)(arguments).await {
                    Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
                    Err(error) => json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": ToolCallResult::error(error.to_string())
                    }),
                }
            }
            _ => mcp_error(&id, -32_601, &format!("method '{method}' not found")),
        }
    }
}

fn mcp_error(id: &Value, code: i32, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message}
    })
}

/// One content block in an assistant message.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentContentBlock {
    /// User-visible assistant text.
    Text(String),
    /// Extended-thinking content and signature.
    Thinking { thinking: String, signature: String },
    /// Client-side tool invocation.
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// Any forward-compatible content block.
    Unknown(Value),
}

impl AgentContentBlock {
    fn from_value(value: Value) -> Self {
        match value.get("type").and_then(Value::as_str) {
            Some("text") => value.get("text").and_then(Value::as_str).map_or_else(
                || Self::Unknown(value.clone()),
                |text| Self::Text(text.into()),
            ),
            Some("thinking") => {
                let thinking = value.get("thinking").and_then(Value::as_str);
                let signature = value.get("signature").and_then(Value::as_str);
                match (thinking, signature) {
                    (Some(thinking), Some(signature)) => Self::Thinking {
                        thinking: thinking.into(),
                        signature: signature.into(),
                    },
                    _ => Self::Unknown(value),
                }
            }
            Some("tool_use") => {
                let id = value.get("id").and_then(Value::as_str);
                let name = value.get("name").and_then(Value::as_str);
                match (id, name) {
                    (Some(id), Some(name)) => Self::ToolUse {
                        id: id.into(),
                        name: name.into(),
                        input: value.get("input").cloned().unwrap_or(Value::Null),
                    },
                    _ => Self::Unknown(value),
                }
            }
            _ => Self::Unknown(value),
        }
    }
}

/// User message emitted or replayed by Claude Code.
#[derive(Debug, Clone, PartialEq)]
pub struct UserAgentMessage {
    /// Message content in Claude wire format.
    pub content: Value,
    /// Stable transcript entry identifier when present.
    pub uuid: Option<String>,
    /// Parent tool invocation for nested messages.
    pub parent_tool_use_id: Option<String>,
    /// Message provenance supplied by Claude Code.
    pub origin: Option<Value>,
}

/// Assistant message emitted by Claude Code.
#[derive(Debug, Clone, PartialEq)]
pub struct AssistantAgentMessage {
    /// Typed assistant content blocks.
    pub content: Vec<AgentContentBlock>,
    /// Model that produced the message.
    pub model: String,
    /// Claude session identifier.
    pub session_id: Option<String>,
    /// Stable transcript entry identifier.
    pub uuid: Option<String>,
    /// API stop reason.
    pub stop_reason: Option<String>,
    /// Raw usage counters.
    pub usage: Option<Value>,
}

/// System event emitted by Claude Code.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemAgentMessage {
    /// System event discriminator.
    pub subtype: String,
    /// Complete event payload for forward compatibility.
    pub data: Value,
}

/// Terminal result for one agent turn.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ResultAgentMessage {
    /// Result subtype, such as `success` or `error_max_turns`.
    pub subtype: String,
    /// Wall-clock duration.
    #[serde(default)]
    pub duration_ms: u64,
    /// Time spent in API calls.
    #[serde(default)]
    pub duration_api_ms: u64,
    /// Whether Claude Code considers this result an error.
    #[serde(default)]
    pub is_error: bool,
    /// Number of agent turns consumed.
    #[serde(default)]
    pub num_turns: u32,
    /// Claude session identifier used for resume.
    pub session_id: String,
    /// User-visible terminal response when present.
    #[serde(default)]
    pub result: Option<String>,
    /// Structured output when a JSON schema was configured.
    #[serde(default)]
    pub structured_output: Option<Value>,
    /// Total API cost for this session.
    #[serde(default)]
    pub total_cost_usd: Option<f64>,
    /// API or agent errors.
    #[serde(default)]
    pub errors: Vec<String>,
    /// Why the query loop stopped.
    #[serde(default)]
    pub terminal_reason: Option<String>,
    /// HTTP status of a failed API request.
    #[serde(default)]
    pub api_error_status: Option<u16>,
    /// Raw usage counters.
    #[serde(default)]
    pub usage: Option<Value>,
}

/// Partial API stream event emitted when enabled.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamAgentMessage {
    /// Stable event identifier.
    pub uuid: String,
    /// Claude session identifier.
    pub session_id: String,
    /// Raw Anthropic API stream event.
    pub event: Value,
}

/// Typed message delivered by the Agent SDK.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentMessage {
    /// User message or replay.
    User(UserAgentMessage),
    /// Assistant response.
    Assistant(AssistantAgentMessage),
    /// System lifecycle event.
    System(SystemAgentMessage),
    /// Terminal turn result.
    Result(ResultAgentMessage),
    /// Partial assistant event.
    StreamEvent(StreamAgentMessage),
    /// Forward-compatible frame not modeled by this SDK version.
    Unknown(Value),
}

impl AgentMessage {
    /// Parse one non-control Agent SDK frame.
    ///
    /// # Errors
    ///
    /// Returns a serialization error when a recognized frame has malformed
    /// fields.
    pub fn from_value(value: Value) -> Result<Self, ClaudeError> {
        match value.get("type").and_then(Value::as_str) {
            Some("user") => {
                let content = value
                    .get("message")
                    .and_then(|message| message.get("content"))
                    .cloned()
                    .unwrap_or(Value::Null);
                Ok(Self::User(UserAgentMessage {
                    content,
                    uuid: string_field(&value, "uuid"),
                    parent_tool_use_id: string_field(&value, "parent_tool_use_id"),
                    origin: value.get("origin").cloned(),
                }))
            }
            Some("assistant") => {
                let message = value.get("message").ok_or_else(|| {
                    ClaudeError::TransportError(
                        "assistant Agent SDK frame is missing message".into(),
                    )
                })?;
                let content = message
                    .get("content")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        ClaudeError::TransportError(
                            "assistant Agent SDK frame is missing message.content".into(),
                        )
                    })?
                    .iter()
                    .cloned()
                    .map(AgentContentBlock::from_value)
                    .collect();
                Ok(Self::Assistant(AssistantAgentMessage {
                    content,
                    model: message
                        .get("model")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                        .into(),
                    session_id: string_field(&value, "session_id"),
                    uuid: string_field(&value, "uuid"),
                    stop_reason: message
                        .get("stop_reason")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    usage: message.get("usage").cloned(),
                }))
            }
            Some("system") => Ok(Self::System(SystemAgentMessage {
                subtype: value
                    .get("subtype")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .into(),
                data: value,
            })),
            Some("result") => Ok(Self::Result(serde_json::from_value(value)?)),
            Some("stream_event") => Ok(Self::StreamEvent(StreamAgentMessage {
                uuid: required_string(&value, "uuid")?,
                session_id: required_string(&value, "session_id")?,
                event: value.get("event").cloned().unwrap_or(Value::Null),
            })),
            _ => Ok(Self::Unknown(value)),
        }
    }
}

fn string_field(value: &Value, field: &str) -> Option<String> {
    value.get(field).and_then(Value::as_str).map(str::to_owned)
}

fn required_string(value: &Value, field: &str) -> Result<String, ClaudeError> {
    string_field(value, field)
        .ok_or_else(|| ClaudeError::TransportError(format!("Agent SDK frame is missing {field}")))
}

/// Final output returned by [`ClaudeAgentClient::query`].
#[derive(Debug, Clone, PartialEq)]
pub struct AgentRunResult {
    /// Claude session identifier; persist this to resume a Slack thread.
    pub session_id: String,
    /// User-visible response, preferring the terminal result text.
    pub text: String,
    /// Structured output when configured.
    pub structured_output: Option<Value>,
    /// Whether Claude Code reported an error.
    pub is_error: bool,
    /// Result subtype.
    pub subtype: String,
    /// Number of turns consumed.
    pub num_turns: u32,
    /// Total API cost when supplied by Claude Code.
    pub total_cost_usd: Option<f64>,
    /// Errors supplied by Claude Code.
    pub errors: Vec<String>,
    /// Why the agent stopped.
    pub terminal_reason: Option<String>,
}

/// Stateful Agent SDK client over a bidirectional transport.
#[derive(Debug)]
pub struct ClaudeAgentClient<T: AgentTransport> {
    transport: T,
    options: AgentOptions,
    mcp_servers: BTreeMap<String, SdkMcpServer>,
    permission_handler: Option<Arc<dyn ToolPermissionHandler>>,
    queued_messages: VecDeque<AgentMessage>,
    request_counter: u64,
    connected: bool,
}

impl<T: AgentTransport> ClaudeAgentClient<T> {
    /// Construct a disconnected client.
    #[must_use]
    pub fn new(transport: T, options: AgentOptions) -> Self {
        Self {
            transport,
            options,
            mcp_servers: BTreeMap::new(),
            permission_handler: None,
            queued_messages: VecDeque::new(),
            request_counter: 0,
            connected: false,
        }
    }

    /// Register an in-process MCP server before connecting.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::InvalidConfig`] after connection or when a
    /// server with the same name is already registered.
    pub fn add_mcp_server(&mut self, server: SdkMcpServer) -> Result<(), ClaudeError> {
        if self.connected {
            return Err(ClaudeError::InvalidConfig(
                "MCP servers must be registered before Agent SDK connect".into(),
            ));
        }
        if self.mcp_servers.contains_key(&server.name) {
            return Err(ClaudeError::InvalidConfig(format!(
                "duplicate SDK MCP server: {}",
                server.name
            )));
        }
        self.mcp_servers.insert(server.name.clone(), server);
        Ok(())
    }

    /// Set the host permission handler before connecting.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::InvalidConfig`] when called after connection.
    pub fn set_permission_handler(
        &mut self,
        handler: Arc<dyn ToolPermissionHandler>,
    ) -> Result<(), ClaudeError> {
        if self.connected {
            return Err(ClaudeError::InvalidConfig(
                "permission handler must be configured before Agent SDK connect".into(),
            ));
        }
        self.permission_handler = Some(handler);
        Ok(())
    }

    /// Connect and complete the Agent SDK initialize handshake.
    ///
    /// # Errors
    ///
    /// Returns an option validation, transport, protocol, or initialization
    /// timeout error when the handshake cannot complete.
    pub async fn connect(&mut self) -> Result<(), ClaudeError> {
        if self.connected {
            return Ok(());
        }
        self.options.validate()?;
        let descriptors = self
            .mcp_servers
            .values()
            .map(SdkMcpServer::descriptor)
            .collect::<Vec<_>>();
        self.transport.connect(&self.options, &descriptors).await?;
        self.connected = true;

        let request_id = self.next_request_id();
        self.transport
            .write(&json!({
                "type": "control_request",
                "request_id": request_id,
                "request": {"subtype": "initialize", "hooks": Value::Null}
            }))
            .await?;
        self.wait_for_control_response(&request_id).await?;
        Ok(())
    }

    /// Send one prompt and wait for its terminal result.
    ///
    /// # Errors
    ///
    /// Returns an error when the client is disconnected, transport I/O fails,
    /// or the stream ends before a terminal result.
    pub async fn query(
        &mut self,
        prompt: impl Into<String>,
    ) -> Result<AgentRunResult, ClaudeError> {
        if !self.connected {
            return Err(ClaudeError::InvalidConfig(
                "Agent SDK client is not connected".into(),
            ));
        }
        let prompt = prompt.into();
        let prompt_id = Uuid::new_v4();
        self.transport
            .write(&json!({
                "type": "user",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": Value::Null,
                "session_id": "default",
                "uuid": prompt_id.to_string(),
                "origin": {"kind": "human"}
            }))
            .await?;

        let mut assistant_text = String::new();
        loop {
            let message = self.next_message().await?.ok_or_else(|| {
                ClaudeError::TransportError(
                    "Claude Code ended before emitting an Agent SDK result".into(),
                )
            })?;
            match message {
                AgentMessage::Assistant(assistant) => {
                    for block in assistant.content {
                        if let AgentContentBlock::Text(text) = block {
                            assistant_text.push_str(&text);
                        }
                    }
                }
                AgentMessage::Result(result) => {
                    return Ok(AgentRunResult {
                        session_id: result.session_id,
                        text: result.result.unwrap_or(assistant_text),
                        structured_output: result.structured_output,
                        is_error: result.is_error,
                        subtype: result.subtype,
                        num_turns: result.num_turns,
                        total_cost_usd: result.total_cost_usd,
                        errors: result.errors,
                        terminal_reason: result.terminal_reason,
                    });
                }
                _ => {}
            }
        }
    }

    /// Receive the next non-control message.
    ///
    /// # Errors
    ///
    /// Returns a transport, protocol, MCP tool, or permission-handler error.
    pub async fn next_message(&mut self) -> Result<Option<AgentMessage>, ClaudeError> {
        if let Some(message) = self.queued_messages.pop_front() {
            return Ok(Some(message));
        }
        loop {
            let Some(frame) = self.transport.read().await? else {
                return Ok(None);
            };
            match frame.get("type").and_then(Value::as_str) {
                Some("control_request") => self.handle_control_request(&frame).await?,
                Some("control_cancel_request" | "control_response" | "transcript_mirror") => {}
                _ => return AgentMessage::from_value(frame).map(Some),
            }
        }
    }

    /// Close stdin and terminate the underlying transport.
    ///
    /// # Errors
    ///
    /// Returns a transport error when ending input or closing the transport
    /// fails.
    pub async fn close(&mut self) -> Result<(), ClaudeError> {
        if !self.connected {
            return Ok(());
        }
        let end_result = self.transport.end_input().await;
        let close_result = self.transport.close().await;
        self.connected = false;
        end_result.and(close_result)
    }

    async fn wait_for_control_response(&mut self, request_id: &str) -> Result<Value, ClaudeError> {
        let timeout = self.options.initialize_timeout;
        tokio::time::timeout(timeout, async {
            loop {
                let frame = self.transport.read().await?.ok_or_else(|| {
                    ClaudeError::TransportError(
                        "Claude Code ended during Agent SDK initialization".into(),
                    )
                })?;
                match frame.get("type").and_then(Value::as_str) {
                    Some("control_response") => {
                        let response = frame.get("response").ok_or_else(|| {
                            ClaudeError::TransportError(
                                "Agent SDK control response is missing response".into(),
                            )
                        })?;
                        if response.get("request_id").and_then(Value::as_str) != Some(request_id) {
                            continue;
                        }
                        if response.get("subtype").and_then(Value::as_str) == Some("error") {
                            return Err(ClaudeError::TransportError(
                                response
                                    .get("error")
                                    .and_then(Value::as_str)
                                    .unwrap_or("Agent SDK initialization failed")
                                    .into(),
                            ));
                        }
                        return Ok(response.get("response").cloned().unwrap_or(Value::Null));
                    }
                    Some("control_request") => self.handle_control_request(&frame).await?,
                    Some("control_cancel_request" | "transcript_mirror") => {}
                    _ => self
                        .queued_messages
                        .push_back(AgentMessage::from_value(frame)?),
                }
            }
        })
        .await
        .map_err(|_| {
            ClaudeError::TransportError(format!(
                "Agent SDK initialize timed out after {} seconds",
                timeout.as_secs()
            ))
        })?
    }

    async fn handle_control_request(&mut self, frame: &Value) -> Result<(), ClaudeError> {
        let request_id = required_string(frame, "request_id")?;
        let request = frame.get("request").ok_or_else(|| {
            ClaudeError::TransportError("Agent SDK control request is missing request".into())
        })?;
        let subtype = request.get("subtype").and_then(Value::as_str).unwrap_or("");
        let response = match subtype {
            "mcp_message" => {
                let server_name = request
                    .get("server_name")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let mcp_message = request.get("message").cloned().unwrap_or(Value::Null);
                let mcp_response = if let Some(server) = self.mcp_servers.get(server_name) {
                    server.handle(&mcp_message).await
                } else {
                    mcp_error(
                        &mcp_message.get("id").cloned().unwrap_or(Value::Null),
                        -32_601,
                        &format!("SDK MCP server '{server_name}' not found"),
                    )
                };
                Ok(json!({"mcp_response": mcp_response}))
            }
            "can_use_tool" => self.handle_permission_request(request).await,
            _ => Err(ClaudeError::TransportError(format!(
                "unsupported Agent SDK control request: {subtype}"
            ))),
        };

        let wire_response = match response {
            Ok(response) => json!({
                "type": "control_response",
                "response": {
                    "subtype": "success",
                    "request_id": request_id,
                    "response": response
                }
            }),
            Err(error) => json!({
                "type": "control_response",
                "response": {
                    "subtype": "error",
                    "request_id": request_id,
                    "error": error.to_string()
                }
            }),
        };
        self.transport.write(&wire_response).await
    }

    async fn handle_permission_request(&self, request: &Value) -> Result<Value, ClaudeError> {
        let tool_name = request
            .get("tool_name")
            .and_then(Value::as_str)
            .unwrap_or("");
        let input = request.get("input").cloned().unwrap_or(Value::Null);
        let decision = if let Some(handler) = &self.permission_handler {
            handler.can_use_tool(tool_name, &input).await?
        } else {
            PermissionDecision::Deny {
                message: "tool was not pre-authorized by the Agent SDK host".into(),
                interrupt: false,
            }
        };
        Ok(match decision {
            PermissionDecision::Allow { updated_input } => json!({
                "behavior": "allow",
                "updatedInput": updated_input.unwrap_or(input)
            }),
            PermissionDecision::Deny { message, interrupt } => json!({
                "behavior": "deny",
                "message": message,
                "interrupt": interrupt
            }),
        })
    }

    fn next_request_id(&mut self) -> String {
        self.request_counter += 1;
        format!("req_{}_{}", self.request_counter, Uuid::new_v4().simple())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default)]
    struct MockTransport {
        incoming: VecDeque<Value>,
        outgoing: Vec<Value>,
        connected: bool,
    }

    #[async_trait]
    impl AgentTransport for MockTransport {
        async fn connect(
            &mut self,
            _options: &AgentOptions,
            _mcp_servers: &[SdkMcpServerDescriptor],
        ) -> Result<(), ClaudeError> {
            self.connected = true;
            Ok(())
        }

        async fn write(&mut self, frame: &Value) -> Result<(), ClaudeError> {
            self.outgoing.push(frame.clone());
            if frame
                .get("request")
                .and_then(|request| request.get("subtype"))
                .and_then(Value::as_str)
                == Some("initialize")
            {
                self.incoming.push_back(json!({
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": frame["request_id"],
                        "response": {"commands": []}
                    }
                }));
            }
            Ok(())
        }

        async fn read(&mut self) -> Result<Option<Value>, ClaudeError> {
            Ok(self.incoming.pop_front())
        }

        async fn end_input(&mut self) -> Result<(), ClaudeError> {
            Ok(())
        }

        async fn close(&mut self) -> Result<(), ClaudeError> {
            self.connected = false;
            Ok(())
        }
    }

    #[test]
    fn validates_bounded_options() {
        let options = AgentOptions {
            max_turns: Some(0),
            ..AgentOptions::default()
        };
        assert!(matches!(
            options.validate(),
            Err(ClaudeError::InvalidConfig(_))
        ));
    }

    #[test]
    fn validates_environment_isolation() {
        let options = AgentOptions {
            env_remove: vec!["SERVICE_SECRET".into()],
            env: BTreeMap::from([("SERVICE_SECRET".into(), "leak".into())]),
            ..AgentOptions::default()
        };
        assert!(matches!(
            options.validate(),
            Err(ClaudeError::InvalidConfig(_))
        ));
    }

    #[tokio::test]
    async fn dispatches_sdk_mcp_tool() {
        let tool = SdkMcpTool::new(
            "echo",
            "Echo input",
            json!({"type": "object"}),
            ToolAnnotations::default(),
            |input| async move { ToolCallResult::json(&input) },
        )
        .unwrap();
        let mut server = SdkMcpServer::new("test", "1.0.0").unwrap();
        server.add_tool(tool).unwrap();

        let response = server
            .handle(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"value": 42}}
            }))
            .await;
        assert_eq!(response["result"]["isError"], Value::Null);
        assert_eq!(response["result"]["content"][0]["text"], "{\"value\":42}");
    }

    #[tokio::test]
    async fn runs_prompt_to_typed_result() {
        let mut transport = MockTransport::default();
        transport.incoming.push_back(json!({
            "type": "assistant",
            "session_id": "session-1",
            "message": {
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "diagnosis"}]
            }
        }));
        transport.incoming.push_back(json!({
            "type": "result",
            "subtype": "success",
            "duration_ms": 20,
            "duration_api_ms": 10,
            "is_error": false,
            "num_turns": 1,
            "session_id": "session-1"
        }));

        let mut client = ClaudeAgentClient::new(transport, AgentOptions::default());
        client.connect().await.unwrap();
        let result = client.query("diagnose").await.unwrap();
        assert_eq!(result.session_id, "session-1");
        assert_eq!(result.text, "diagnosis");
        assert!(!result.is_error);
    }
}
