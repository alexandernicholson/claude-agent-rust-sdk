//! Claude Code Agent SDK protocol primitives.
//!
//! This module implements the bidirectional NDJSON control protocol used by
//! Claude Code. Transport implementations own process or network I/O; this
//! crate owns typed messages, the initialization handshake, permission gates,
//! and in-process MCP tool dispatch.

use crate::error::ClaudeError;
use crate::extensions::{
    AgentDefinition, CanUseTool, CanUseToolShadowedWarning, HookEvent, HookMatcher,
    McpServerConfig, OutputFormat, PermissionMode, SandboxSettings, SdkBeta, SdkPluginConfig,
    SettingSource, SkillSelection, SystemPrompt, TaskBudget, ThinkingConfig, ToolsSpec,
};
use crate::sessions::{SessionStore, SessionStoreFlushMode};
use crate::types::EffortLevel;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

/// MCP protocol version advertised by in-process SDK servers.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Callback invoked with each stderr line emitted by the Claude Code
/// subprocess.
pub type StderrCallback = Arc<dyn Fn(&str) + Send + Sync>;

/// MCP server configurations for an [`AgentOptions`].
///
/// Mirrors the official `mcp_servers: dict[str, McpServerConfig] | str | Path`
/// union: a named map of server configs, a raw `--mcp-config` JSON string, or a
/// filesystem path to an MCP config file. The transport encodes the map form
/// per server (stripping the SDK `instance` marker and merging in-process SDK
/// descriptors), and passes the string/path forms straight through to
/// `--mcp-config`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpServers {
    /// A map of server name to configuration.
    Map(BTreeMap<String, McpServerConfig>),
    /// A raw `--mcp-config` JSON configuration string.
    ConfigString(String),
    /// A filesystem path to an MCP configuration file.
    ConfigPath(PathBuf),
}

impl Default for McpServers {
    fn default() -> Self {
        Self::Map(BTreeMap::new())
    }
}

impl McpServers {
    /// Whether this configuration contributes no servers (an empty map).
    ///
    /// String and path forms are never considered empty: they always reach the
    /// CLI verbatim.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Map(map) if map.is_empty())
    }

    /// Borrow the server map when this is the map form.
    #[must_use]
    pub const fn as_map(&self) -> Option<&BTreeMap<String, McpServerConfig>> {
        match self {
            Self::Map(map) => Some(map),
            _ => None,
        }
    }
}

impl From<BTreeMap<String, McpServerConfig>> for McpServers {
    fn from(map: BTreeMap<String, McpServerConfig>) -> Self {
        Self::Map(map)
    }
}

/// Options shared by Agent SDK transports.
///
/// Mirrors the official Python `ClaudeAgentOptions`. Optional fields default to
/// absent (`None`); collections default to empty and remain distinguishable
/// from absent. Explicit empty collections carry meaning (for example
/// `Some(ToolsSpec::List(vec![]))` disables every built-in tool while `None`
/// keeps Claude Code defaults).
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone)]
pub struct AgentOptions {
    /// System prompt configuration. `None` uses Claude Code's default; a
    /// custom string, a preset, or a file are all expressible.
    pub system_prompt: Option<SystemPrompt>,
    /// Base set of built-in tools. `None` uses Claude Code defaults;
    /// `Some(ToolsSpec::List(vec![]))` disables every built-in tool.
    pub tools: Option<ToolsSpec>,
    /// Tools auto-approved without an interactive prompt.
    pub allowed_tools: Vec<String>,
    /// Tools removed from the model's context.
    pub disallowed_tools: Vec<String>,
    /// MCP server configurations: a named map, a raw `--mcp-config` JSON string,
    /// or a config file path. Mirrors the official
    /// `dict[str, McpServerConfig] | str | Path`.
    pub mcp_servers: McpServers,
    /// Restrict MCP configuration to the supplied servers only.
    pub strict_mcp_config: bool,
    /// Permission behavior for this process. `None` uses the CLI default.
    pub permission_mode: Option<PermissionMode>,
    /// Continue the most recent conversation in the working directory.
    pub continue_conversation: bool,
    /// Session identifier or title to resume.
    pub resume: Option<String>,
    /// Explicit session identifier for a new conversation.
    pub session_id: Option<String>,
    /// Maximum agent turns for one invocation. `None` uses the CLI default.
    pub max_turns: Option<u32>,
    /// Maximum API spend for one invocation.
    pub max_budget_usd: Option<f64>,
    /// Primary model or model alias.
    pub model: Option<String>,
    /// Fallback model or model alias.
    pub fallback_model: Option<String>,
    /// Beta features to enable.
    pub betas: Vec<SdkBeta>,
    /// MCP tool name used to route permission prompts.
    pub permission_prompt_tool_name: Option<String>,
    /// Working directory visible to Claude Code.
    pub cwd: Option<PathBuf>,
    /// Explicit Claude Code executable path.
    pub cli_path: Option<PathBuf>,
    /// Additional settings JSON file path or inline JSON.
    pub settings: Option<String>,
    /// Additional directories exposed to Claude Code.
    pub add_dirs: Vec<PathBuf>,
    /// Environment overrides for the Claude Code child process.
    pub env: BTreeMap<String, String>,
    /// Inherited environment variables removed before applying `env` overrides.
    pub env_remove: Vec<String>,
    /// Additional CLI flags. Keys exclude the leading `--`; `None` is a
    /// boolean flag.
    pub extra_args: BTreeMap<String, Option<String>>,
    /// Maximum accepted bytes in one NDJSON frame. `None` uses the 1 MiB
    /// default; `Some(0)` is valid and rejects every non-empty frame, matching
    /// the official `max_buffer_size: int | None`.
    pub max_buffer_size: Option<usize>,
    /// Deprecated and no longer read by the transport; retained for parity with
    /// the official `debug_stderr` field. Use the [`stderr`](Self::stderr)
    /// callback instead.
    pub debug_stderr: Value,
    /// Callback for stderr lines from the Claude Code subprocess.
    pub stderr: Option<StderrCallback>,
    /// Custom permission handler for tool calls that would otherwise prompt.
    pub can_use_tool: Option<CanUseTool>,
    /// Hook callbacks registered per hook event.
    pub hooks: Option<BTreeMap<HookEvent, Vec<HookMatcher>>>,
    /// Optional OS user identifier for the subprocess.
    pub user: Option<String>,
    /// Include partial assistant events in the message stream.
    pub include_partial_messages: bool,
    /// Include hook lifecycle events in the message stream.
    pub include_hook_events: bool,
    /// Fork resumed sessions to a new session identifier.
    pub fork_session: bool,
    /// When resuming, load only up to and including this transcript UUID.
    pub resume_session_at: Option<String>,
    /// UUID of the user prompt whose turn a truncating resume discards.
    pub resume_drops_turn: Option<String>,
    /// Programmatically defined subagents, keyed by name.
    pub agents: Option<BTreeMap<String, AgentDefinition>>,
    /// Filesystem setting layers to load. `None` uses CLI defaults; `Some(vec![])`
    /// disables filesystem settings (SDK isolation).
    pub setting_sources: Option<Vec<SettingSource>>,
    /// Skills to enable for the main session.
    pub skills: Option<SkillSelection>,
    /// Sandbox settings for command execution isolation.
    pub sandbox: Option<SandboxSettings>,
    /// Local plugins to load for this session.
    pub plugins: Vec<SdkPluginConfig>,
    /// Deprecated maximum thinking-token budget; superseded by `thinking`.
    pub max_thinking_tokens: Option<u32>,
    /// Thinking/reasoning behavior configuration.
    pub thinking: Option<ThinkingConfig>,
    /// Effort applied to model output, tool calls, and adaptive thinking.
    pub effort: Option<EffortLevel>,
    /// Structured output format configuration.
    pub output_format: Option<OutputFormat>,
    /// Enable file checkpointing for rewind support.
    pub enable_file_checkpointing: bool,
    /// External store for mirroring session transcripts.
    pub session_store: Option<Arc<dyn SessionStore>>,
    /// Flush policy for mirrored transcript entries.
    pub session_store_flush: SessionStoreFlushMode,
    /// Timeout in milliseconds for each session-store load/list call.
    pub load_timeout_ms: u64,
    /// API-side task budget in tokens.
    pub task_budget: Option<TaskBudget>,
    /// Timeout for the initialize control request.
    pub initialize_timeout: Duration,
}

impl fmt::Debug for AgentOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AgentOptions")
            .field("system_prompt", &self.system_prompt)
            .field("tools", &self.tools)
            .field("allowed_tools", &self.allowed_tools)
            .field("disallowed_tools", &self.disallowed_tools)
            .field("mcp_servers", &self.mcp_servers)
            .field("strict_mcp_config", &self.strict_mcp_config)
            .field("permission_mode", &self.permission_mode)
            .field("continue_conversation", &self.continue_conversation)
            .field("resume", &self.resume)
            .field("session_id", &self.session_id)
            .field("max_turns", &self.max_turns)
            .field("max_budget_usd", &self.max_budget_usd)
            .field("model", &self.model)
            .field("fallback_model", &self.fallback_model)
            .field("betas", &self.betas)
            .field(
                "permission_prompt_tool_name",
                &self.permission_prompt_tool_name,
            )
            .field("cwd", &self.cwd)
            .field("cli_path", &self.cli_path)
            .field("settings", &self.settings)
            .field("add_dirs", &self.add_dirs)
            .field("env", &self.env)
            .field("env_remove", &self.env_remove)
            .field("extra_args", &self.extra_args)
            .field("max_buffer_size", &self.max_buffer_size)
            .field("debug_stderr", &self.debug_stderr)
            .field("stderr", &self.stderr.as_ref().map(|_| "<callback>"))
            .field(
                "can_use_tool",
                &self.can_use_tool.as_ref().map(|_| "<callback>"),
            )
            .field("hooks", &self.hooks)
            .field("user", &self.user)
            .field("include_partial_messages", &self.include_partial_messages)
            .field("include_hook_events", &self.include_hook_events)
            .field("fork_session", &self.fork_session)
            .field("resume_session_at", &self.resume_session_at)
            .field("resume_drops_turn", &self.resume_drops_turn)
            .field("agents", &self.agents)
            .field("setting_sources", &self.setting_sources)
            .field("skills", &self.skills)
            .field("sandbox", &self.sandbox)
            .field("plugins", &self.plugins)
            .field("max_thinking_tokens", &self.max_thinking_tokens)
            .field("thinking", &self.thinking)
            .field("effort", &self.effort)
            .field("output_format", &self.output_format)
            .field("enable_file_checkpointing", &self.enable_file_checkpointing)
            .field("session_store", &self.session_store)
            .field("session_store_flush", &self.session_store_flush)
            .field("load_timeout_ms", &self.load_timeout_ms)
            .field("task_budget", &self.task_budget)
            .field("initialize_timeout", &self.initialize_timeout)
            .finish()
    }
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            system_prompt: None,
            tools: None,
            allowed_tools: Vec::new(),
            disallowed_tools: Vec::new(),
            mcp_servers: McpServers::default(),
            strict_mcp_config: false,
            permission_mode: None,
            continue_conversation: false,
            resume: None,
            session_id: None,
            max_turns: None,
            max_budget_usd: None,
            model: None,
            fallback_model: None,
            betas: Vec::new(),
            permission_prompt_tool_name: None,
            cwd: None,
            cli_path: None,
            settings: None,
            add_dirs: Vec::new(),
            env: BTreeMap::new(),
            env_remove: Vec::new(),
            extra_args: BTreeMap::new(),
            max_buffer_size: None,
            debug_stderr: Value::Null,
            stderr: None,
            can_use_tool: None,
            hooks: None,
            user: None,
            include_partial_messages: false,
            include_hook_events: false,
            fork_session: false,
            resume_session_at: None,
            resume_drops_turn: None,
            agents: None,
            setting_sources: None,
            skills: None,
            sandbox: None,
            plugins: Vec::new(),
            max_thinking_tokens: None,
            thinking: None,
            effort: None,
            output_format: None,
            enable_file_checkpointing: false,
            session_store: None,
            session_store_flush: SessionStoreFlushMode::default(),
            load_timeout_ms: 60_000,
            task_budget: None,
            initialize_timeout: Duration::from_mins(1),
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
        // max_turns, max_budget_usd, and max_buffer_size are forwarded to the
        // CLI/transport verbatim (they own their validation), matching the
        // official Python SDK. In particular an explicit `Some(0)` buffer is
        // accepted here and enforced later by the transport frame guard.
        if let Some(SkillSelection::List(names)) = &self.skills {
            for name in names {
                if name.is_empty() || name.trim() != name || name.contains(['*', ',']) {
                    return Err(ClaudeError::InvalidConfig(format!(
                        "invalid skill name: {name:?}"
                    )));
                }
            }
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

    /// Validate a `can_use_tool` callback against incompatible options and
    /// derive the effective options used for the control protocol.
    ///
    /// Mirrors the official `_process_query_inner`/`_connect_inner` setup: a
    /// permission callback is mutually exclusive with an explicit
    /// `permission_prompt_tool_name`, and when a callback is present the
    /// effective `permission_prompt_tool_name` is forced to `"stdio"` so the
    /// CLI routes prompts back over the control channel. Returns an owned
    /// options clone when normalization is required, or borrows `self`
    /// otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::InvalidConfig`] when a `can_use_tool` callback is
    /// combined with an explicit `permission_prompt_tool_name`.
    fn resolve_permission_options(&self) -> Result<Cow<'_, Self>, ClaudeError> {
        if self.can_use_tool.is_none() {
            return Ok(Cow::Borrowed(self));
        }
        if self.permission_prompt_tool_name.is_some() {
            return Err(ClaudeError::InvalidConfig(
                "can_use_tool callback cannot be used with permission_prompt_tool_name. \
                 Please use one or the other."
                    .into(),
            ));
        }
        let mut effective = self.clone();
        effective.permission_prompt_tool_name = Some("stdio".to_string());
        Ok(Cow::Owned(effective))
    }

    /// Return the advisory warning emitted when a `can_use_tool` callback is
    /// shadowed by auto-approval settings, or `None` when it is not shadowed.
    ///
    /// Mirrors the official `_get_can_use_tool_shadowed_warning`: a
    /// `bypassPermissions` mode or an `allowed_tools` entry that allows a whole
    /// tool auto-approves calls before the callback runs. `skills="all"`
    /// contributes a bare `Skill` allow, matching the transport.
    #[must_use]
    pub fn can_use_tool_shadow_warning(&self) -> Option<CanUseToolShadowedWarning> {
        self.can_use_tool.as_ref()?;
        if self.permission_mode == Some(PermissionMode::BypassPermissions) {
            return Some(CanUseToolShadowedWarning::new(
                "can_use_tool will not be invoked: permission_mode 'bypassPermissions' \
                 auto-approves every tool call (except explicit deny rules) before the \
                 callback is consulted. To gate every tool call, use a PreToolUse hook instead.",
            ));
        }
        // skills="all" makes the transport append a bare "Skill" allow, which
        // shadows the callback just like a hand-written entry.
        let mut allowed: Vec<&str> = self.allowed_tools.iter().map(String::as_str).collect();
        if matches!(self.skills, Some(SkillSelection::All)) && !allowed.contains(&"Skill") {
            allowed.push("Skill");
        }
        // Dedupe while preserving order: ["Read", "Read()"] resolve to the same
        // tool and must not be reported twice.
        let mut shadowed: Vec<String> = Vec::new();
        for entry in allowed {
            if let Some(tool) = whole_tool_allowed(entry) {
                if !shadowed.iter().any(|existing| existing == tool) {
                    shadowed.push(tool.to_string());
                }
            }
        }
        if shadowed.is_empty() {
            return None;
        }
        Some(CanUseToolShadowedWarning::new(format!(
            "can_use_tool will not be invoked for: {}. An allowed_tools entry that allows a \
             whole tool auto-approves it before the callback is consulted. To gate every tool \
             call, use a PreToolUse hook; or narrow the entry so calls fall through to \
             can_use_tool. Allow rules from settings files can also shadow the callback but \
             are not visible here.",
            shadowed.join(", ")
        )))
    }
}

fn emit_can_use_tool_shadow_warning(warning: &CanUseToolShadowedWarning) {
    static EMITTED: LazyLock<parking_lot::Mutex<HashSet<String>>> =
        LazyLock::new(|| parking_lot::Mutex::new(HashSet::new()));
    let message = warning.message();
    if EMITTED.lock().insert(message.to_string()) {
        tracing::warn!(
            target: "claude_agent_sdk::can_use_tool",
            code = "CLAUDE_SDK_CAN_USE_TOOL_SHADOWED",
            warning_type = "CanUseToolShadowedWarning",
            warning = message
        );
    }
}

fn invalid_environment_name(name: &str) -> bool {
    name.is_empty() || name.contains('=') || name.contains('\0')
}

/// Return the tool an `allowed_tools` entry allows outright, else `None`.
///
/// Mirrors the CLI's rule parser (official `_whole_tool_allowed`): an entry
/// allows a whole tool when it has no `(...)` specifier (`"Read"`), or when the
/// specifier is empty or a lone wildcard (`"Read()"`, `"Read(*)"`). A real
/// specifier (`"Bash(ls:*)"`) only allows matching invocations.
fn whole_tool_allowed(entry: &str) -> Option<&str> {
    if entry.trim().is_empty() {
        return None;
    }
    let Some(open) = entry.find('(') else {
        return Some(entry);
    };
    if open == 0 || !entry.ends_with(')') {
        return None;
    }
    let spec = &entry[open + 1..entry.len() - 1];
    if spec.is_empty() || spec == "*" {
        Some(&entry[..open])
    } else {
        None
    }
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
///
/// Implementations own process or network I/O behind interior synchronization.
/// Every method takes `&self` so a single transport can be shared (e.g. behind
/// an [`Arc`]) between a persistent reader task and concurrent writers. The SDK
/// owns frame serialization: [`AgentTransport::write`] accepts a raw NDJSON line
/// and the implementation writes it verbatim (appending the newline).
#[async_trait]
pub trait AgentTransport: fmt::Debug + Send + Sync + 'static {
    /// Establish the transport and prepare stdin/stdout framing.
    async fn connect(
        &self,
        options: &AgentOptions,
        mcp_servers: &[SdkMcpServerDescriptor],
    ) -> Result<(), ClaudeError>;

    /// Write one raw NDJSON line. The SDK has already serialized the frame; the
    /// implementation writes the bytes and terminates the line.
    async fn write(&self, raw: &str) -> Result<(), ClaudeError>;

    /// Read one JSON frame, or `None` after clean end-of-stream.
    async fn read(&self) -> Result<Option<Value>, ClaudeError>;

    /// Report whether the transport is connected and ready for I/O.
    fn is_ready(&self) -> bool;

    /// Close the input half while continuing to drain output.
    async fn end_input(&self) -> Result<(), ClaudeError>;

    /// Terminate the transport and release all resources.
    async fn close(&self) -> Result<(), ClaudeError>;
}

/// MCP tool safety hints shown to the model and host.
///
/// Mirrors the official `mcp.types.ToolAnnotations` plus the Anthropic-specific
/// `maxResultSizeChars` hint. The size hint is not a standard annotation field:
/// the MCP schema strips unknown annotation keys, so it is emitted separately
/// under the tool's `_meta` as `anthropic/maxResultSizeChars`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolAnnotations {
    /// Human-readable title for the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
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
    /// Anthropic-specific tool-result spill threshold (characters). Emitted in
    /// `_meta` (never as a wire annotation) as `anthropic/maxResultSizeChars`.
    #[serde(skip)]
    pub max_result_size_chars: Option<u64>,
}

/// One content item returned by an SDK MCP tool.
///
/// Mirrors the MCP content union produced by the official SDK bridge: text,
/// image, resource link, and embedded resource. The `type` discriminator uses
/// the official wire spellings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolContent {
    /// Plain text content.
    Text {
        /// Text delivered to the model.
        text: String,
    },
    /// Image content: base64 data plus MIME type.
    Image {
        /// Base64-encoded image bytes.
        data: String,
        /// Image MIME type (for example `image/png`).
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    /// A link to an external resource.
    ResourceLink {
        /// Resource URI.
        uri: String,
        /// Optional display name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Optional description.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        /// Optional MIME type.
        #[serde(default, rename = "mimeType", skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
    /// An embedded resource carrying inline text or binary contents.
    Resource {
        /// The embedded resource object (`{uri, text?, blob?, mimeType?}`).
        resource: Value,
    },
}

impl ToolContent {
    /// Create an MCP text content item.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create an MCP image content item.
    #[must_use]
    pub fn image(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Image {
            data: data.into(),
            mime_type: mime_type.into(),
        }
    }

    /// Convert this content item to the MCP `CallToolResult` wire form,
    /// applying the official bridge's conversions.
    ///
    /// Mirrors `create_sdk_mcp_server`'s `call_tool`: text and image pass
    /// through; a resource link is rendered to text (name/uri/description
    /// joined by newlines); an embedded resource with inline text becomes text;
    /// a binary embedded resource cannot be converted and is dropped
    /// (`None`).
    #[must_use]
    pub fn to_mcp_wire(&self) -> Option<Value> {
        match self {
            Self::Text { text } => Some(json!({ "type": "text", "text": text })),
            Self::Image { data, mime_type } => Some(json!({
                "type": "image",
                "data": data,
                "mimeType": mime_type,
            })),
            Self::ResourceLink {
                uri,
                name,
                description,
                ..
            } => {
                let mut parts: Vec<&str> = Vec::new();
                if let Some(name) = name.as_deref() {
                    if !name.is_empty() {
                        parts.push(name);
                    }
                }
                if !uri.is_empty() {
                    parts.push(uri);
                }
                if let Some(description) = description.as_deref() {
                    if !description.is_empty() {
                        parts.push(description);
                    }
                }
                let text = if parts.is_empty() {
                    "Resource link".to_string()
                } else {
                    parts.join("\n")
                };
                Some(json!({ "type": "text", "text": text }))
            }
            Self::Resource { resource } => resource
                .get("text")
                .and_then(Value::as_str)
                .map(|text| json!({ "type": "text", "text": text })),
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

    /// Render the MCP `CallToolResult` wire object for this result, applying
    /// the official content conversions and dropping content that cannot be
    /// represented as MCP content (binary embedded resources).
    fn to_mcp_result(&self) -> Value {
        let content: Vec<Value> = self
            .content
            .iter()
            .filter_map(ToolContent::to_mcp_wire)
            .collect();
        json!({ "content": content, "isError": self.is_error })
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
    annotations: Option<ToolAnnotations>,
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
    /// Annotations are optional, matching the official
    /// `SdkMcpTool.annotations` (`ToolAnnotations | None`); when absent they
    /// are omitted from `tools/list`.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::InvalidConfig`] when the tool name is invalid or
    /// `input_schema` is not a JSON object.
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        annotations: impl Into<Option<ToolAnnotations>>,
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
            annotations: annotations.into(),
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

    /// Dispatch one inbound MCP JSON-RPC message and return the JSON-RPC reply.
    pub(crate) async fn handle(&self, message: &Value) -> Value {
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
                        let mut entry = serde_json::Map::new();
                        entry.insert("name".into(), json!(tool.name));
                        entry.insert("description".into(), json!(tool.description));
                        entry.insert("inputSchema".into(), tool.input_schema.clone());
                        // Annotations are optional and omitted when absent,
                        // matching the official bridge; the size hint is not a
                        // wire annotation so it never appears here.
                        if let Some(annotations) = &tool.annotations {
                            entry.insert(
                                "annotations".into(),
                                serde_json::to_value(annotations).unwrap_or(Value::Null),
                            );
                        }
                        // maxResultSizeChars rides in _meta under a namespaced
                        // key (the MCP schema strips unknown annotation fields).
                        if let Some(max) = tool
                            .annotations
                            .as_ref()
                            .and_then(|a| a.max_result_size_chars)
                        {
                            entry.insert(
                                "_meta".into(),
                                json!({ "anthropic/maxResultSizeChars": max }),
                            );
                        }
                        Value::Object(entry)
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
                    Ok(result) => {
                        json!({"jsonrpc": "2.0", "id": id, "result": result.to_mcp_result()})
                    }
                    Err(error) => json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": ToolCallResult::error(error.to_string()).to_mcp_result()
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

/// One content block in a user or assistant message.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentContentBlock {
    /// User-visible text.
    Text { text: String },
    /// Extended-thinking content and signature.
    Thinking { thinking: String, signature: String },
    /// Client-side tool invocation.
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// Result returned for a client-side tool call.
    ToolResult {
        tool_use_id: String,
        /// Result content: a string, a list of blocks, or absent.
        content: Option<Value>,
        /// Whether the tool reported an error.
        is_error: Option<bool>,
    },
    /// Server-side tool invocation (advisor, `web_search`, `web_fetch`, ...).
    ServerToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// Result returned for a server-side tool call.
    ServerToolResult {
        tool_use_id: String,
        /// Raw result content dict, opaque to this layer.
        content: Value,
    },
}

impl AgentContentBlock {
    /// Parse one content block, enforcing required fields for known types.
    ///
    /// Returns `Ok(None)` when the block type is unrecognized (forward
    /// compatibility) and `Err` when a known block type is malformed.
    fn from_value(value: &Value) -> Result<Option<Self>, ClaudeError> {
        let Some(block) = value.as_object() else {
            return Err(message_parse(
                "content block must be an object",
                value.clone(),
            ));
        };
        let block_type = block.get("type").and_then(Value::as_str);
        match block_type {
            Some("text") => Ok(Some(Self::Text {
                text: field_str(block, "text", value)?,
            })),
            Some("thinking") => Ok(Some(Self::Thinking {
                thinking: field_str(block, "thinking", value)?,
                signature: field_str(block, "signature", value)?,
            })),
            Some("tool_use") => Ok(Some(Self::ToolUse {
                id: field_str(block, "id", value)?,
                name: field_str(block, "name", value)?,
                input: field_value(block, "input", value)?,
            })),
            Some("tool_result") => Ok(Some(Self::ToolResult {
                tool_use_id: field_str(block, "tool_use_id", value)?,
                content: block.get("content").cloned(),
                is_error: block.get("is_error").and_then(Value::as_bool),
            })),
            Some("server_tool_use") => Ok(Some(Self::ServerToolUse {
                id: field_str(block, "id", value)?,
                name: field_str(block, "name", value)?,
                input: field_value(block, "input", value)?,
            })),
            Some("advisor_tool_result") => Ok(Some(Self::ServerToolResult {
                tool_use_id: field_str(block, "tool_use_id", value)?,
                content: field_value(block, "content", value)?,
            })),
            // A block with no `type` field is malformed: the official parser
            // indexes `block["type"]` and raises on a missing discriminator.
            // A present-but-unrecognized type is skipped for forward
            // compatibility.
            _ if !block.contains_key("type") => Err(message_parse(
                "Invalid content block (missing 'type' field)",
                value.clone(),
            )),
            _ => Ok(None),
        }
    }
}

/// Message provenance object attached to user messages and results.
pub type MessageOrigin = Value;

/// User message emitted or replayed by Claude Code.
#[derive(Debug, Clone, PartialEq)]
pub struct UserAgentMessage {
    /// Message content: typed blocks when a list was provided, otherwise the
    /// raw value (for example a plain string).
    pub content: UserContent,
    /// Stable transcript entry identifier when present.
    pub uuid: Option<String>,
    /// Parent tool invocation for nested messages.
    pub parent_tool_use_id: Option<String>,
    /// Metadata about a tool execution result attached to this message.
    pub tool_use_result: Option<Value>,
    /// Message provenance supplied by Claude Code.
    pub origin: Option<MessageOrigin>,
}

/// User message content in either typed-block or raw form.
#[derive(Debug, Clone, PartialEq)]
pub enum UserContent {
    /// Content supplied as a list of typed blocks.
    Blocks(Vec<AgentContentBlock>),
    /// Content supplied in any other form (typically a plain string).
    Raw(Value),
}

/// Assistant message emitted by Claude Code.
#[derive(Debug, Clone, PartialEq)]
pub struct AssistantAgentMessage {
    /// Typed assistant content blocks.
    pub content: Vec<AgentContentBlock>,
    /// Model that produced the message.
    pub model: String,
    /// Parent tool invocation for nested subagent messages.
    pub parent_tool_use_id: Option<String>,
    /// Assistant error classification, when the turn failed.
    pub error: Option<String>,
    /// Raw usage counters.
    pub usage: Option<Value>,
    /// Anthropic API message identifier.
    pub message_id: Option<String>,
    /// API stop reason.
    pub stop_reason: Option<String>,
    /// Claude session identifier.
    pub session_id: Option<String>,
    /// Stable transcript entry identifier.
    pub uuid: Option<String>,
}

/// System event emitted by Claude Code.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemAgentMessage {
    /// System event discriminator.
    pub subtype: String,
    /// Complete event payload for forward compatibility.
    pub data: Value,
}

/// Task lifecycle: a task started.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskStartedAgentMessage {
    /// Complete raw payload.
    pub data: Value,
    pub task_id: String,
    pub description: String,
    pub uuid: String,
    pub session_id: String,
    pub tool_use_id: Option<String>,
    pub task_type: Option<String>,
}

/// Task lifecycle: progress update.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskProgressAgentMessage {
    /// Complete raw payload.
    pub data: Value,
    pub task_id: String,
    pub description: String,
    /// Raw usage counters (`total_tokens`, `tool_uses`, `duration_ms`).
    pub usage: Value,
    pub uuid: String,
    pub session_id: String,
    pub tool_use_id: Option<String>,
    pub last_tool_name: Option<String>,
}

/// Task lifecycle: terminal notification.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskNotificationAgentMessage {
    /// Complete raw payload.
    pub data: Value,
    pub task_id: String,
    /// One of `completed`, `failed`, `stopped`.
    pub status: String,
    pub output_file: String,
    pub summary: String,
    pub uuid: String,
    pub session_id: String,
    pub tool_use_id: Option<String>,
    pub usage: Option<Value>,
}

/// Task lifecycle: state patch (may carry terminal status).
#[derive(Debug, Clone, PartialEq)]
pub struct TaskUpdatedAgentMessage {
    /// Complete raw payload.
    pub data: Value,
    pub task_id: String,
    /// Changed fields; `status` (when present) drives terminal-ness.
    pub patch: Value,
    /// One of `pending`, `running`, `paused`, `completed`, `failed`, `killed`.
    pub status: Option<String>,
    pub session_id: Option<String>,
    pub uuid: Option<String>,
}

/// SDK-synthesized error emitted when a session-store append fails.
#[derive(Debug, Clone, PartialEq)]
pub struct MirrorErrorAgentMessage {
    /// Complete raw payload.
    pub data: Value,
    /// Session key that failed to mirror, when known.
    pub key: Option<Value>,
    /// Error description.
    pub error: String,
}

/// Hook lifecycle event emitted when `include_hook_events` is enabled.
#[derive(Debug, Clone, PartialEq)]
pub struct HookEventAgentMessage {
    /// Lifecycle phase: `hook_started` or `hook_response`.
    pub subtype: String,
    /// Hook event name (for example `PreToolUse`).
    pub hook_event_name: String,
    /// Complete raw payload.
    pub data: Value,
    pub session_id: Option<String>,
    pub uuid: Option<String>,
}

/// Terminal result for one agent turn.
#[derive(Debug, Clone, PartialEq)]
pub struct ResultAgentMessage {
    /// Result subtype, such as `success` or `error_max_turns`.
    pub subtype: String,
    /// Wall-clock duration.
    pub duration_ms: u64,
    /// Time spent in API calls.
    pub duration_api_ms: u64,
    /// Whether Claude Code considers this result an error.
    pub is_error: bool,
    /// Number of agent turns consumed.
    pub num_turns: u32,
    /// Claude session identifier used for resume.
    pub session_id: String,
    /// API stop reason.
    pub stop_reason: Option<String>,
    /// Total API cost for this session.
    pub total_cost_usd: Option<f64>,
    /// Raw usage counters.
    pub usage: Option<Value>,
    /// User-visible terminal response when present.
    pub result: Option<String>,
    /// Structured output when a JSON schema was configured.
    pub structured_output: Option<Value>,
    /// Per-model usage and cost breakdown (`modelUsage`).
    pub model_usage: Option<Value>,
    /// Permission denials recorded during the turn.
    pub permission_denials: Option<Value>,
    /// Tool call deferred by a `PreToolUse` hook.
    pub deferred_tool_use: Option<DeferredToolUse>,
    /// API or agent errors.
    pub errors: Option<Vec<String>>,
    /// HTTP status of a failed API request.
    pub api_error_status: Option<u16>,
    /// Stable transcript entry identifier.
    pub uuid: Option<String>,
    /// Why the query loop stopped.
    pub terminal_reason: Option<String>,
    /// Provenance of the user message that triggered this turn.
    pub origin: Option<MessageOrigin>,
}

/// Tool use deferred by a `PreToolUse` hook returning `defer`.
#[derive(Debug, Clone, PartialEq)]
pub struct DeferredToolUse {
    pub id: String,
    pub name: String,
    pub input: Value,
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
    /// Parent tool invocation for nested subagent streams.
    pub parent_tool_use_id: Option<String>,
}

/// Rate limit event emitted when rate limit state changes.
#[derive(Debug, Clone, PartialEq)]
pub struct RateLimitAgentMessage {
    /// Parsed rate limit info.
    pub rate_limit_info: RateLimitInfo,
    pub uuid: String,
    pub session_id: String,
}

/// Rate limit status snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct RateLimitInfo {
    /// One of `allowed`, `allowed_warning`, `rejected`.
    pub status: String,
    pub resets_at: Option<i64>,
    pub rate_limit_type: Option<String>,
    pub utilization: Option<f64>,
    pub overage_status: Option<String>,
    pub overage_resets_at: Option<i64>,
    pub overage_disabled_reason: Option<String>,
    /// Full raw dict, including fields not modeled above.
    pub raw: Value,
}

/// Conversation reset emitted when the transcript is replaced mid-session.
#[derive(Debug, Clone, PartialEq)]
pub struct ConversationResetAgentMessage {
    pub new_conversation_id: String,
    pub uuid: String,
    pub session_id: String,
}

/// Typed message delivered by the Agent SDK.
// `Result` wraps a large official message struct; boxing it would change the
// public `AgentMessage` variant shape, so accept the size disparity.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum AgentMessage {
    /// User message or replay.
    User(UserAgentMessage),
    /// Assistant response.
    Assistant(AssistantAgentMessage),
    /// System lifecycle event with an unrecognized subtype.
    System(SystemAgentMessage),
    /// Task started.
    TaskStarted(TaskStartedAgentMessage),
    /// Task progress.
    TaskProgress(TaskProgressAgentMessage),
    /// Task terminal notification.
    TaskNotification(TaskNotificationAgentMessage),
    /// Task state patch.
    TaskUpdated(TaskUpdatedAgentMessage),
    /// Session-store mirror error.
    MirrorError(MirrorErrorAgentMessage),
    /// Hook lifecycle event.
    HookEvent(HookEventAgentMessage),
    /// Terminal turn result.
    Result(ResultAgentMessage),
    /// Partial assistant event.
    StreamEvent(StreamAgentMessage),
    /// Rate limit state change.
    RateLimit(RateLimitAgentMessage),
    /// Conversation reset.
    ConversationReset(ConversationResetAgentMessage),
}

impl AgentMessage {
    /// Parse one non-control Agent SDK frame.
    ///
    /// Returns `Ok(None)` when the top-level message type is unrecognized
    /// (forward-compatible skip). Returns `Err(ClaudeError::MessageParse)` with
    /// the raw frame preserved when a recognized frame has malformed or missing
    /// required fields.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::MessageParse`] for malformed known frames.
    pub fn from_value(value: Value) -> Result<Option<Self>, ClaudeError> {
        // A non-object frame is malformed: the official parser rejects any
        // top-level value that is not a dict.
        if !value.is_object() {
            return Err(message_parse(
                "Invalid message data type (expected object)",
                value.clone(),
            ));
        }
        // A missing or empty top-level `type` is malformed (Python raises
        // "Message missing 'type' field" for any falsy type).
        match value.get("type").and_then(Value::as_str) {
            Some(t) if !t.is_empty() => {}
            _ => {
                return Err(message_parse("Message missing 'type' field", value.clone()));
            }
        }

        if value.get("type").and_then(Value::as_str) == Some("system") {
            if let Some(subtype @ ("hook_started" | "hook_response")) =
                value.get("subtype").and_then(Value::as_str)
            {
                // Truthiness fallback: an explicitly empty `hook_event` still
                // falls through to `hook_name`/`hook_event_name`, matching
                // Python's `data.get(...) or ...` chain.
                let hook_event_name = ["hook_event", "hook_name", "hook_event_name"]
                    .iter()
                    .find_map(|key| {
                        value
                            .get(*key)
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                    })
                    .unwrap_or("")
                    .to_owned();
                return Ok(Some(Self::HookEvent(HookEventAgentMessage {
                    subtype: subtype.to_owned(),
                    hook_event_name,
                    session_id: string_field(&value, "session_id"),
                    uuid: string_field(&value, "uuid"),
                    data: value,
                })));
            }
        }

        match value.get("type").and_then(Value::as_str) {
            Some("user") => Self::parse_user(&value).map(Some),
            Some("assistant") => Self::parse_assistant(&value).map(Some),
            Some("system") => Self::parse_system(value).map(Some),
            Some("result") => Self::parse_result(&value).map(Some),
            Some("stream_event") => Self::parse_stream_event(&value).map(Some),
            Some("rate_limit_event") => Self::parse_rate_limit(&value).map(Some),
            Some("conversation_reset") => Self::parse_conversation_reset(&value).map(Some),
            // Unknown top-level type: skip for forward compatibility.
            _ => Ok(None),
        }
    }

    fn parse_user(value: &Value) -> Result<Self, ClaudeError> {
        let raw_content = value
            .get("message")
            .and_then(|message| message.get("content"))
            .ok_or_else(|| {
                message_parse(
                    "Missing required field in user message: content",
                    value.clone(),
                )
            })?;
        let content = if let Some(blocks) = raw_content.as_array() {
            let mut parsed = Vec::with_capacity(blocks.len());
            for block in blocks {
                if let Some(parsed_block) = AgentContentBlock::from_value(block)? {
                    parsed.push(parsed_block);
                }
            }
            UserContent::Blocks(parsed)
        } else {
            UserContent::Raw(raw_content.clone())
        };
        Ok(Self::User(UserAgentMessage {
            content,
            uuid: string_field(value, "uuid"),
            parent_tool_use_id: string_field(value, "parent_tool_use_id"),
            tool_use_result: value.get("tool_use_result").cloned(),
            origin: parse_origin(value),
        }))
    }

    fn parse_assistant(value: &Value) -> Result<Self, ClaudeError> {
        let message = value.get("message").ok_or_else(|| {
            message_parse(
                "Missing required field in assistant message: message",
                value.clone(),
            )
        })?;
        let raw_content = message.get("content").ok_or_else(|| {
            message_parse(
                "Missing required field in assistant message: content",
                value.clone(),
            )
        })?;
        let blocks = raw_content.as_array().ok_or_else(|| {
            message_parse("Invalid assistant content (expected list)", value.clone())
        })?;
        let mut content = Vec::with_capacity(blocks.len());
        for block in blocks {
            if let Some(parsed_block) = AgentContentBlock::from_value(block)? {
                content.push(parsed_block);
            }
        }
        let model = message
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                message_parse(
                    "Missing required field in assistant message: model",
                    value.clone(),
                )
            })?
            .to_owned();
        Ok(Self::Assistant(AssistantAgentMessage {
            content,
            model,
            parent_tool_use_id: string_field(value, "parent_tool_use_id"),
            error: string_field(value, "error"),
            usage: message.get("usage").cloned(),
            message_id: string_field(message, "id"),
            stop_reason: string_field(message, "stop_reason"),
            session_id: string_field(value, "session_id"),
            uuid: string_field(value, "uuid"),
        }))
    }

    fn parse_system(value: Value) -> Result<Self, ClaudeError> {
        let subtype = value
            .get("subtype")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                message_parse(
                    "Missing required field in system message: subtype",
                    value.clone(),
                )
            })?;
        match subtype {
            "task_started" => Ok(Self::TaskStarted(TaskStartedAgentMessage {
                task_id: required_field_str(&value, "task_id")?,
                description: required_field_str(&value, "description")?,
                uuid: required_field_str(&value, "uuid")?,
                session_id: required_field_str(&value, "session_id")?,
                tool_use_id: string_field(&value, "tool_use_id"),
                task_type: string_field(&value, "task_type"),
                data: value,
            })),
            "task_progress" => Ok(Self::TaskProgress(TaskProgressAgentMessage {
                task_id: required_field_str(&value, "task_id")?,
                description: required_field_str(&value, "description")?,
                usage: required_field_value(&value, "usage")?,
                uuid: required_field_str(&value, "uuid")?,
                session_id: required_field_str(&value, "session_id")?,
                tool_use_id: string_field(&value, "tool_use_id"),
                last_tool_name: string_field(&value, "last_tool_name"),
                data: value,
            })),
            "task_notification" => Ok(Self::TaskNotification(TaskNotificationAgentMessage {
                task_id: required_field_str(&value, "task_id")?,
                status: required_field_str(&value, "status")?,
                output_file: required_field_str(&value, "output_file")?,
                summary: required_field_str(&value, "summary")?,
                uuid: required_field_str(&value, "uuid")?,
                session_id: required_field_str(&value, "session_id")?,
                tool_use_id: string_field(&value, "tool_use_id"),
                usage: value.get("usage").cloned(),
                data: value,
            })),
            "task_updated" => {
                // Parsed defensively: a lifecycle patch may omit uuid/session_id
                // and parsing must never raise on a lifecycle event.
                let patch = value
                    .get("patch")
                    .filter(|patch| patch.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let status = patch
                    .get("status")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                Ok(Self::TaskUpdated(TaskUpdatedAgentMessage {
                    task_id: string_field(&value, "task_id").unwrap_or_default(),
                    patch,
                    status,
                    session_id: string_field(&value, "session_id"),
                    uuid: string_field(&value, "uuid"),
                    data: value,
                }))
            }
            "mirror_error" => Ok(Self::MirrorError(MirrorErrorAgentMessage {
                key: value.get("key").cloned(),
                error: string_field(&value, "error").unwrap_or_default(),
                data: value,
            })),
            _ => Ok(Self::System(SystemAgentMessage {
                subtype: subtype.to_owned(),
                data: value,
            })),
        }
    }

    fn parse_result(value: &Value) -> Result<Self, ClaudeError> {
        // Python guards with `if deferred:` — a falsy value (null, {}, empty)
        // yields None. Only a non-empty object is parsed.
        let deferred_tool_use = match value.get("deferred_tool_use") {
            Some(Value::Object(deferred)) if !deferred.is_empty() => Some(DeferredToolUse {
                id: deferred
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        message_parse(
                            "Missing required field in result message: deferred_tool_use.id",
                            value.clone(),
                        )
                    })?,
                name: deferred
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        message_parse(
                            "Missing required field in result message: deferred_tool_use.name",
                            value.clone(),
                        )
                    })?,
                input: deferred.get("input").cloned().ok_or_else(|| {
                    message_parse(
                        "Missing required field in result message: deferred_tool_use.input",
                        value.clone(),
                    )
                })?,
            }),
            _ => None,
        };
        let errors = value.get("errors").and_then(Value::as_array).map(|errors| {
            errors
                .iter()
                .filter_map(|error| error.as_str().map(str::to_owned))
                .collect()
        });
        Ok(Self::Result(ResultAgentMessage {
            subtype: required_field_str(value, "subtype")?,
            duration_ms: required_field_u64(value, "duration_ms")?,
            duration_api_ms: required_field_u64(value, "duration_api_ms")?,
            is_error: value
                .get("is_error")
                .and_then(Value::as_bool)
                .ok_or_else(|| {
                    message_parse(
                        "Missing required field in result message: is_error",
                        value.clone(),
                    )
                })?,
            num_turns: u32::try_from(required_field_u64(value, "num_turns")?).unwrap_or(u32::MAX),
            session_id: required_field_str(value, "session_id")?,
            stop_reason: string_field(value, "stop_reason"),
            total_cost_usd: value.get("total_cost_usd").and_then(Value::as_f64),
            usage: value.get("usage").cloned(),
            result: string_field(value, "result"),
            structured_output: value.get("structured_output").cloned(),
            model_usage: value.get("modelUsage").cloned(),
            permission_denials: value.get("permission_denials").cloned(),
            deferred_tool_use,
            errors,
            api_error_status: value
                .get("api_error_status")
                .and_then(Value::as_u64)
                .and_then(|status| u16::try_from(status).ok()),
            uuid: string_field(value, "uuid"),
            terminal_reason: string_field(value, "terminal_reason"),
            origin: parse_origin(value),
        }))
    }

    fn parse_stream_event(value: &Value) -> Result<Self, ClaudeError> {
        Ok(Self::StreamEvent(StreamAgentMessage {
            uuid: required_field_str(value, "uuid")?,
            session_id: required_field_str(value, "session_id")?,
            event: required_field_value(value, "event")?,
            parent_tool_use_id: string_field(value, "parent_tool_use_id"),
        }))
    }

    fn parse_rate_limit(value: &Value) -> Result<Self, ClaudeError> {
        let info = value.get("rate_limit_info").ok_or_else(|| {
            message_parse(
                "Missing required field in rate_limit_event message: rate_limit_info",
                value.clone(),
            )
        })?;
        let status = info.get("status").and_then(Value::as_str).ok_or_else(|| {
            message_parse(
                "Missing required field in rate_limit_event message: status",
                value.clone(),
            )
        })?;
        let rate_limit_info = RateLimitInfo {
            status: status.to_owned(),
            resets_at: info.get("resetsAt").and_then(Value::as_i64),
            rate_limit_type: string_field(info, "rateLimitType"),
            utilization: info.get("utilization").and_then(Value::as_f64),
            overage_status: string_field(info, "overageStatus"),
            overage_resets_at: info.get("overageResetsAt").and_then(Value::as_i64),
            overage_disabled_reason: string_field(info, "overageDisabledReason"),
            raw: info.clone(),
        };
        Ok(Self::RateLimit(RateLimitAgentMessage {
            rate_limit_info,
            uuid: required_field_str(value, "uuid")?,
            session_id: required_field_str(value, "session_id")?,
        }))
    }

    fn parse_conversation_reset(value: &Value) -> Result<Self, ClaudeError> {
        Ok(Self::ConversationReset(ConversationResetAgentMessage {
            new_conversation_id: required_field_str(value, "new_conversation_id")?,
            uuid: required_field_str(value, "uuid")?,
            session_id: required_field_str(value, "session_id")?,
        }))
    }
}

fn string_field(value: &Value, field: &str) -> Option<String> {
    value.get(field).and_then(Value::as_str).map(str::to_owned)
}

/// Return `data["origin"]` when it is a well-formed origin object (has a string
/// `kind`), matching the official `_parse_origin` guard.
fn parse_origin(value: &Value) -> Option<MessageOrigin> {
    let origin = value.get("origin")?;
    if origin.is_object() && origin.get("kind").and_then(Value::as_str).is_some() {
        Some(origin.clone())
    } else {
        None
    }
}

fn message_parse(message: &str, data: Value) -> ClaudeError {
    ClaudeError::MessageParse {
        message: message.to_owned(),
        data: Some(data),
    }
}

fn field_str(
    block: &serde_json::Map<String, Value>,
    field: &str,
    raw: &Value,
) -> Result<String, ClaudeError> {
    block
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            message_parse(
                &format!("Missing required content-block field: {field}"),
                raw.clone(),
            )
        })
}

fn field_value(
    block: &serde_json::Map<String, Value>,
    field: &str,
    raw: &Value,
) -> Result<Value, ClaudeError> {
    block.get(field).cloned().ok_or_else(|| {
        message_parse(
            &format!("Missing required content-block field: {field}"),
            raw.clone(),
        )
    })
}

fn required_field_str(value: &Value, field: &str) -> Result<String, ClaudeError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| message_parse(&format!("Missing required field: {field}"), value.clone()))
}

fn required_field_value(value: &Value, field: &str) -> Result<Value, ClaudeError> {
    value
        .get(field)
        .cloned()
        .ok_or_else(|| message_parse(&format!("Missing required field: {field}"), value.clone()))
}

fn required_field_u64(value: &Value, field: &str) -> Result<u64, ClaudeError> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| message_parse(&format!("Missing required field: {field}"), value.clone()))
}

/// Final output returned by [`ClaudeAgentClient::query`].
#[derive(Debug, Clone, PartialEq)]
pub struct AgentRunResult {
    /// Claude session identifier; persist this to resume the conversation.
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
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// Time spent in API calls in milliseconds.
    pub duration_api_ms: u64,
    /// Total API cost when supplied by Claude Code.
    pub total_cost_usd: Option<f64>,
    /// Aggregate usage counters when supplied.
    pub usage: Option<Value>,
    /// Per-model usage and cost breakdown (`modelUsage`).
    pub model_usage: Option<Value>,
    /// Permission denials recorded during the turn.
    pub permission_denials: Option<Value>,
    /// Tool call deferred by a `PreToolUse` hook.
    pub deferred_tool_use: Option<DeferredToolUse>,
    /// Errors supplied by Claude Code.
    pub errors: Vec<String>,
    /// HTTP status of a failed API request.
    pub api_error_status: Option<u16>,
    /// API stop reason.
    pub stop_reason: Option<String>,
    /// Stable transcript entry identifier for the result.
    pub uuid: Option<String>,
    /// Why the agent stopped.
    pub terminal_reason: Option<String>,
}

#[cfg(test)]
mod agent_parser_tests {
    use super::*;
    use serde_json::json;

    fn parse(value: Value) -> Option<AgentMessage> {
        AgentMessage::from_value(value).expect("parse should succeed")
    }

    #[test]
    fn default_options_match_python_absent_semantics() {
        let options = AgentOptions::default();
        assert!(options.system_prompt.is_none());
        assert!(options.tools.is_none());
        assert!(options.allowed_tools.is_empty());
        assert!(options.disallowed_tools.is_empty());
        assert!(options.mcp_servers.is_empty());
        assert!(!options.strict_mcp_config);
        assert!(options.permission_mode.is_none());
        assert!(!options.continue_conversation);
        assert!(options.max_turns.is_none());
        assert!(options.setting_sources.is_none());
        assert!(options.betas.is_empty());
        assert!(options.plugins.is_empty());
        assert!(!options.include_partial_messages);
        assert!(!options.include_hook_events);
        assert!(!options.fork_session);
        assert!(!options.enable_file_checkpointing);
        assert_eq!(options.max_buffer_size, None);
        assert_eq!(options.load_timeout_ms, 60_000);
        assert_eq!(
            options.session_store_flush,
            SessionStoreFlushMode::default()
        );
    }

    #[test]
    fn default_options_validate() {
        AgentOptions::default()
            .validate()
            .expect("default options are valid");
    }

    #[test]
    fn parse_user_message_string_content() {
        let message = parse(json!({
            "type": "user",
            "message": {"content": "Simple string content"},
            "tool_use_result": {"filePath": "/a.py", "userModified": true},
        }));
        let Some(AgentMessage::User(user)) = message else {
            panic!("expected user message");
        };
        assert_eq!(
            user.content,
            UserContent::Raw(json!("Simple string content"))
        );
        assert_eq!(
            user.tool_use_result,
            Some(json!({"filePath": "/a.py", "userModified": true}))
        );
    }

    #[test]
    fn parse_user_message_block_content_and_tool_result() {
        let message = parse(json!({
            "type": "user",
            "message": {"content": [
                {"type": "text", "text": "hi"},
                {"type": "tool_result", "tool_use_id": "t1", "content": "done", "is_error": true},
            ]},
            "parent_tool_use_id": "toolu_parent",
            "uuid": "u1",
        }));
        let Some(AgentMessage::User(user)) = message else {
            panic!("expected user message");
        };
        assert_eq!(user.parent_tool_use_id.as_deref(), Some("toolu_parent"));
        assert_eq!(user.uuid.as_deref(), Some("u1"));
        let UserContent::Blocks(blocks) = user.content else {
            panic!("expected typed blocks");
        };
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0], AgentContentBlock::Text { text: "hi".into() });
        assert_eq!(
            blocks[1],
            AgentContentBlock::ToolResult {
                tool_use_id: "t1".into(),
                content: Some(json!("done")),
                is_error: Some(true),
            }
        );
    }

    #[test]
    fn parse_user_origin_requires_kind() {
        let peer = json!({"kind": "peer", "from": "a"});
        let with_kind = parse(json!({
            "type": "user",
            "message": {"content": "hi"},
            "origin": peer.clone(),
        }));
        let Some(AgentMessage::User(user)) = with_kind else {
            panic!("expected user");
        };
        assert_eq!(user.origin, Some(peer));

        // origin without a string kind is treated as absent.
        let no_kind = parse(json!({
            "type": "user",
            "message": {"content": "hi"},
            "origin": {"server": "x"},
        }));
        let Some(AgentMessage::User(user)) = no_kind else {
            panic!("expected user");
        };
        assert!(user.origin.is_none());
    }

    #[test]
    fn parse_assistant_preserves_metadata_and_blocks() {
        let message = parse(json!({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "tool_use", "id": "tu1", "name": "Read", "input": {"path": "/x"}},
                    {"type": "server_tool_use", "id": "st1", "name": "advisor", "input": {}},
                    {"type": "advisor_tool_result", "tool_use_id": "st1", "content": {"type": "advisor_result", "text": "ok"}},
                ],
                "model": "claude-opus-4-5",
                "id": "msg_01",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5},
            },
            "parent_tool_use_id": "p1",
            "session_id": "s1",
            "uuid": "u1",
            "error": "rate_limit",
        }));
        let Some(AgentMessage::Assistant(assistant)) = message else {
            panic!("expected assistant");
        };
        assert_eq!(assistant.model, "claude-opus-4-5");
        assert_eq!(assistant.message_id.as_deref(), Some("msg_01"));
        assert_eq!(assistant.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(assistant.parent_tool_use_id.as_deref(), Some("p1"));
        assert_eq!(assistant.session_id.as_deref(), Some("s1"));
        assert_eq!(assistant.error.as_deref(), Some("rate_limit"));
        assert_eq!(assistant.usage, Some(json!({"input_tokens": 5})));
        assert_eq!(assistant.content.len(), 5);
        assert_eq!(
            assistant.content[3],
            AgentContentBlock::ServerToolUse {
                id: "st1".into(),
                name: "advisor".into(),
                input: json!({}),
            }
        );
        assert_eq!(
            assistant.content[4],
            AgentContentBlock::ServerToolResult {
                tool_use_id: "st1".into(),
                content: json!({"type": "advisor_result", "text": "ok"}),
            }
        );
    }

    #[test]
    fn parse_assistant_skips_unknown_block_types() {
        let message = parse(json!({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "keep"},
                    {"type": "future_block", "payload": 1},
                ],
                "model": "m",
            },
        }));
        let Some(AgentMessage::Assistant(assistant)) = message else {
            panic!("expected assistant");
        };
        assert_eq!(
            assistant.content,
            vec![AgentContentBlock::Text {
                text: "keep".into()
            }]
        );
    }

    #[test]
    fn malformed_assistant_missing_message_yields_message_parse() {
        let raw = json!({"type": "assistant"});
        let error = AgentMessage::from_value(raw.clone()).expect_err("should error");
        match error {
            ClaudeError::MessageParse { data, .. } => assert_eq!(data, Some(raw)),
            other => panic!("expected MessageParse, got {other:?}"),
        }
    }

    #[test]
    fn malformed_assistant_string_content_yields_message_parse() {
        let raw = json!({"type": "assistant", "message": {"model": "m", "content": "hi"}});
        let error = AgentMessage::from_value(raw.clone()).expect_err("should error");
        assert!(matches!(error, ClaudeError::MessageParse { .. }));
    }

    #[test]
    fn parse_system_known_and_unknown_subtypes() {
        let generic = parse(json!({"type": "system", "subtype": "start", "foo": "bar"}));
        let Some(AgentMessage::System(system)) = generic else {
            panic!("expected system");
        };
        assert_eq!(system.subtype, "start");

        let missing = AgentMessage::from_value(json!({"type": "system"}));
        assert!(matches!(missing, Err(ClaudeError::MessageParse { .. })));
    }

    #[test]
    fn parse_task_lifecycle_messages() {
        let started = parse(json!({
            "type": "system", "subtype": "task_started",
            "task_id": "t", "description": "d", "uuid": "u", "session_id": "s",
            "tool_use_id": "tu",
        }));
        assert!(matches!(started, Some(AgentMessage::TaskStarted(_))));

        let progress = parse(json!({
            "type": "system", "subtype": "task_progress",
            "task_id": "t", "description": "d", "uuid": "u", "session_id": "s",
            "usage": {"total_tokens": 10, "tool_uses": 1, "duration_ms": 5},
        }));
        let Some(AgentMessage::TaskProgress(p)) = progress else {
            panic!("expected task_progress");
        };
        assert_eq!(
            p.usage,
            json!({"total_tokens": 10, "tool_uses": 1, "duration_ms": 5})
        );

        let notification = parse(json!({
            "type": "system", "subtype": "task_notification",
            "task_id": "t", "status": "failed", "output_file": "o", "summary": "sum",
            "uuid": "u", "session_id": "s",
        }));
        let Some(AgentMessage::TaskNotification(n)) = notification else {
            panic!("expected task_notification");
        };
        assert_eq!(n.status, "failed");
        assert!(n.usage.is_none());
    }

    #[test]
    fn parse_task_updated_defensive() {
        let updated = parse(json!({
            "type": "system", "subtype": "task_updated",
            "task_id": "t", "patch": {"status": "completed", "end_time": 1},
        }));
        let Some(AgentMessage::TaskUpdated(u)) = updated else {
            panic!("expected task_updated");
        };
        assert_eq!(u.status.as_deref(), Some("completed"));

        // No patch, no uuid/session: must not raise; status is None.
        let bare = parse(json!({"type": "system", "subtype": "task_updated", "task_id": "t"}));
        let Some(AgentMessage::TaskUpdated(u)) = bare else {
            panic!("expected task_updated");
        };
        assert_eq!(u.patch, json!({}));
        assert!(u.status.is_none());
    }

    #[test]
    fn parse_mirror_error() {
        let mirror = parse(json!({
            "type": "system", "subtype": "mirror_error",
            "key": {"project_key": "p"}, "error": "boom",
        }));
        let Some(AgentMessage::MirrorError(m)) = mirror else {
            panic!("expected mirror_error");
        };
        assert_eq!(m.error, "boom");
        assert_eq!(m.key, Some(json!({"project_key": "p"})));
    }

    #[test]
    fn parse_hook_event_from_system_subtype() {
        let hook = parse(json!({
            "type": "system", "subtype": "hook_started",
            "hook_event": "PreToolUse", "session_id": "s", "uuid": "u",
        }));
        let Some(AgentMessage::HookEvent(h)) = hook else {
            panic!("expected hook event");
        };
        assert_eq!(h.subtype, "hook_started");
        assert_eq!(h.hook_event_name, "PreToolUse");

        // hook_name fallback and absent optionals still parse.
        let hook2 =
            parse(json!({"type": "system", "subtype": "hook_response", "hook_name": "Stop"}));
        let Some(AgentMessage::HookEvent(h)) = hook2 else {
            panic!("expected hook event");
        };
        assert_eq!(h.hook_event_name, "Stop");
        assert!(h.session_id.is_none());
    }

    #[test]
    fn parse_result_full_metadata() {
        let result = parse(json!({
            "type": "result", "subtype": "success",
            "duration_ms": 3000, "duration_api_ms": 2000,
            "is_error": false, "num_turns": 4, "session_id": "s",
            "total_cost_usd": 0.5,
            "modelUsage": {"claude": {"costUSD": 0.01}},
            "permission_denials": [],
            "uuid": "ru",
            "deferred_tool_use": {"id": "d1", "name": "Bash", "input": {"command": "ls"}},
            "errors": ["bad"],
            "api_error_status": 429,
            "terminal_reason": "completed",
            "origin": {"kind": "human"},
        }));
        let Some(AgentMessage::Result(r)) = result else {
            panic!("expected result");
        };
        assert_eq!(r.duration_ms, 3000);
        assert_eq!(r.num_turns, 4);
        assert_eq!(r.model_usage, Some(json!({"claude": {"costUSD": 0.01}})));
        assert_eq!(r.permission_denials, Some(json!([])));
        assert_eq!(r.uuid.as_deref(), Some("ru"));
        assert_eq!(r.api_error_status, Some(429));
        assert_eq!(r.terminal_reason.as_deref(), Some("completed"));
        assert_eq!(r.origin, Some(json!({"kind": "human"})));
        let deferred = r.deferred_tool_use.expect("deferred present");
        assert_eq!(deferred.id, "d1");
        assert_eq!(deferred.name, "Bash");
        assert_eq!(deferred.input, json!({"command": "ls"}));
        assert_eq!(r.errors.as_deref(), Some(["bad".to_owned()].as_slice()));
    }

    #[test]
    fn parse_result_optional_fields_absent() {
        let result = parse(json!({
            "type": "result", "subtype": "success",
            "duration_ms": 1, "duration_api_ms": 1,
            "is_error": false, "num_turns": 1, "session_id": "s",
        }));
        let Some(AgentMessage::Result(r)) = result else {
            panic!("expected result");
        };
        assert!(r.model_usage.is_none());
        assert!(r.permission_denials.is_none());
        assert!(r.deferred_tool_use.is_none());
        assert!(r.errors.is_none());
        assert!(r.api_error_status.is_none());
        assert!(r.uuid.is_none());
    }

    #[test]
    fn malformed_result_missing_field_yields_message_parse() {
        let raw = json!({"type": "result", "subtype": "success"});
        let error = AgentMessage::from_value(raw.clone()).expect_err("should error");
        match error {
            ClaudeError::MessageParse { data, .. } => assert_eq!(data, Some(raw)),
            other => panic!("expected MessageParse, got {other:?}"),
        }
    }

    #[test]
    fn parse_stream_event_with_parent() {
        let stream = parse(json!({
            "type": "stream_event", "uuid": "u", "session_id": "s",
            "event": {"type": "content_block_delta"},
            "parent_tool_use_id": "p",
        }));
        let Some(AgentMessage::StreamEvent(e)) = stream else {
            panic!("expected stream event");
        };
        assert_eq!(e.parent_tool_use_id.as_deref(), Some("p"));
        assert_eq!(e.event, json!({"type": "content_block_delta"}));
    }

    #[test]
    fn parse_mirror_error_retains_full_key() {
        // The runtime's mirror on_error serializes the failing SessionKey into
        // `key` (project_key/session_id/subpath), matching Python's
        // report_mirror_error; consumers must observe it intact.
        let frame = json!({
            "type": "system",
            "subtype": "mirror_error",
            "error": "append failed",
            "key": {
                "project_key": "proj",
                "session_id": "11111111-1111-1111-1111-111111111111",
                "subpath": null,
            },
            "session_id": "11111111-1111-1111-1111-111111111111",
            "uuid": "u",
        });
        let Some(AgentMessage::MirrorError(m)) = parse(frame) else {
            panic!("expected mirror_error");
        };
        assert_eq!(m.error, "append failed");
        let key = m.key.expect("mirror error must retain key");
        assert_eq!(key["project_key"], "proj");
        assert_eq!(key["session_id"], "11111111-1111-1111-1111-111111111111");
        assert_eq!(key["subpath"], Value::Null);
    }

    #[test]
    fn parse_rate_limit_event() {
        let event = parse(json!({
            "type": "rate_limit_event",
            "rate_limit_info": {
                "status": "allowed_warning",
                "resetsAt": 1_700_000_000,
                "rateLimitType": "five_hour",
                "utilization": 0.9,
            },
            "uuid": "u", "session_id": "s",
        }));
        let Some(AgentMessage::RateLimit(r)) = event else {
            panic!("expected rate limit");
        };
        assert_eq!(r.rate_limit_info.status, "allowed_warning");
        assert_eq!(r.rate_limit_info.resets_at, Some(1_700_000_000));
        assert_eq!(
            r.rate_limit_info.rate_limit_type.as_deref(),
            Some("five_hour")
        );
        assert_eq!(r.rate_limit_info.utilization, Some(0.9));
        assert!(r.rate_limit_info.raw.get("status").is_some());
    }

    #[test]
    fn parse_conversation_reset() {
        let reset = parse(json!({
            "type": "conversation_reset",
            "new_conversation_id": "nc", "uuid": "u", "session_id": "s",
        }));
        let Some(AgentMessage::ConversationReset(c)) = reset else {
            panic!("expected conversation reset");
        };
        assert_eq!(c.new_conversation_id, "nc");

        let missing = AgentMessage::from_value(json!({
            "type": "conversation_reset", "uuid": "u", "session_id": "s",
        }));
        assert!(matches!(missing, Err(ClaudeError::MessageParse { .. })));
    }

    #[test]
    fn unknown_top_level_type_skipped() {
        // A present-but-unrecognized type is skipped for forward compatibility.
        assert!(AgentMessage::from_value(json!({"type": "unknown_type"}))
            .unwrap()
            .is_none());
    }

    #[test]
    fn missing_top_level_type_errors() {
        // An object with no `type` field is malformed (Python raises
        // "Message missing 'type' field").
        let raw = json!({"foo": "bar"});
        assert!(matches!(
            AgentMessage::from_value(raw),
            Err(ClaudeError::MessageParse { .. })
        ));
        // An explicitly empty type is also falsy and rejected.
        assert!(matches!(
            AgentMessage::from_value(json!({"type": ""})),
            Err(ClaudeError::MessageParse { .. })
        ));
    }

    #[test]
    fn non_object_frame_errors() {
        for raw in [json!(null), json!([1, 2, 3]), json!("hi"), json!(42)] {
            assert!(
                matches!(
                    AgentMessage::from_value(raw.clone()),
                    Err(ClaudeError::MessageParse { .. })
                ),
                "expected parse error for {raw:?}"
            );
        }
    }

    #[test]
    fn content_block_missing_type_errors() {
        // A user/assistant content block with no `type` is malformed.
        let raw = json!({
            "type": "assistant",
            "message": {"model": "m", "content": [{"text": "hi"}]},
        });
        assert!(matches!(
            AgentMessage::from_value(raw),
            Err(ClaudeError::MessageParse { .. })
        ));
        // A present-but-unknown block type is skipped, not an error.
        let raw = json!({
            "type": "assistant",
            "message": {"model": "m", "content": [{"type": "future_block"}]},
        });
        let Some(AgentMessage::Assistant(a)) = AgentMessage::from_value(raw).unwrap() else {
            panic!("expected assistant");
        };
        assert!(a.content.is_empty());
    }
}

/// Stateful Agent SDK client over a bidirectional transport.
///
/// The client owns the transport behind an [`Arc`] and, once connected, runs a
/// persistent reader task that routes the bidirectional control protocol. Send
/// and receive surfaces take `&self` so a single connected client can be driven
/// concurrently (e.g. one task streaming input while another drains results).
#[derive(Debug)]
pub struct ClaudeAgentClient<T: AgentTransport> {
    transport: Arc<T>,
    options: AgentOptions,
    mcp_servers: BTreeMap<String, SdkMcpServer>,
    /// Live session; `None` until [`connect`](Self::connect) succeeds and after
    /// [`close`](Self::close). Behind a `Mutex` so `&self` operations can take
    /// and release it without exterior `&mut`.
    runtime: tokio::sync::Mutex<Option<Arc<crate::agent_runtime::Runtime<T>>>>,
}

impl<T: AgentTransport> ClaudeAgentClient<T> {
    /// Construct a disconnected client.
    #[must_use]
    pub fn new(transport: T, options: AgentOptions) -> Self {
        Self {
            transport: Arc::new(transport),
            options,
            mcp_servers: BTreeMap::new(),
            runtime: tokio::sync::Mutex::new(None),
        }
    }

    /// Register an in-process MCP server before connecting.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::InvalidConfig`] after connection or when a
    /// server with the same name is already registered.
    pub fn add_mcp_server(&mut self, server: SdkMcpServer) -> Result<(), ClaudeError> {
        if self.runtime.get_mut().is_some() {
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

    /// Connect and complete the Agent SDK initialize handshake.
    ///
    /// Connection is transactional: the client is only marked connected after a
    /// successful handshake. Any transport, protocol, or timeout failure rolls
    /// back (closes the transport, drops the reader) and leaves the client
    /// disconnected so a later call performs a fresh handshake. A duplicate
    /// connect on an already-live client is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an option validation, transport, protocol, or initialization
    /// timeout error when the handshake cannot complete.
    pub async fn connect(&mut self) -> Result<(), ClaudeError> {
        {
            let guard = self.runtime.get_mut();
            if guard.as_ref().is_some_and(|rt| rt.is_ready()) {
                return Ok(());
            }
            // A stale, closed runtime is replaced by the fresh handshake below.
            *guard = None;
        }
        self.options.validate()?;

        // Normalize the permission callback setup: reject a callback combined
        // with an explicit permission_prompt_tool_name and force the effective
        // permission_prompt_tool_name to "stdio", matching the official
        // control-protocol setup. Emit the advisory shadowing diagnostic once.
        let resolved = self.options.resolve_permission_options()?;
        if let Some(warning) = resolved.can_use_tool_shadow_warning() {
            emit_can_use_tool_shadow_warning(&warning);
        }

        // Every SDK server referenced by an options `Sdk` config must have a
        // registered in-process instance, otherwise its tool calls would route
        // to a nonexistent handler. Reject such configs before spawning.
        if let Some(map) = resolved.mcp_servers.as_map() {
            for (name, config) in map {
                if let McpServerConfig::Sdk(sdk) = config {
                    if !self.mcp_servers.contains_key(&sdk.name) {
                        return Err(ClaudeError::InvalidConfig(format!(
                            "SDK MCP server {:?} (config key {name:?}) has no registered \
                             in-process instance; register it with add_mcp_server before connect",
                            sdk.name
                        )));
                    }
                }
            }
        }

        // Fail fast on invalid session-store combinations before spawning the
        // subprocess, then materialize a resume session into a temp config dir
        // when the store holds the requested conversation. The materialized
        // options repoint the subprocess at that temp dir; its cleanup is
        // deferred until after the transport closes (owned by the runtime).
        crate::sessions::validate_session_store_options(&resolved)?;
        let materialized = crate::sessions::materialize_resume_session(&resolved).await?;
        let effective_options = match materialized.as_ref() {
            Some(mat) => crate::sessions::apply_materialized_options(&resolved, mat),
            None => resolved.into_owned(),
        };

        let descriptors = self
            .mcp_servers
            .values()
            .map(SdkMcpServer::descriptor)
            .collect::<Vec<_>>();
        let servers: HashMap<String, SdkMcpServer> = self
            .mcp_servers
            .iter()
            .map(|(name, server)| (name.clone(), server.clone()))
            .collect();

        let runtime = crate::agent_runtime::Runtime::connect(
            Arc::clone(&self.transport),
            &effective_options,
            materialized,
            Arc::new(servers),
            &descriptors,
        )
        .await?;
        *self.runtime.get_mut() = Some(Arc::new(runtime));
        Ok(())
    }

    /// Whether the client is connected and its reader/transport are ready.
    pub async fn is_ready(&self) -> bool {
        self.runtime
            .lock()
            .await
            .as_ref()
            .is_some_and(|rt| rt.is_ready())
    }

    /// Server initialization info (available commands, output styles) captured
    /// during the handshake.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when called before a successful
    /// [`connect`](Self::connect), matching Python's `get_server_info()` which
    /// raises `CLIConnectionError` when not connected.
    pub async fn server_info(&self) -> Result<Value, ClaudeError> {
        self.runtime
            .lock()
            .await
            .as_ref()
            .map(|rt| rt.server_info().clone())
            .ok_or_else(|| {
                ClaudeError::CliConnection(
                    "Not connected. Call connect() first before getting server info.".into(),
                )
            })
    }

    async fn runtime(&self) -> Result<Arc<crate::agent_runtime::Runtime<T>>, ClaudeError> {
        self.runtime.lock().await.clone().ok_or_else(|| {
            ClaudeError::CliConnection("Not connected. Call connect() first.".into())
        })
    }

    /// Send one prompt as a streamed user frame without waiting for a result.
    ///
    /// Mirrors the stateful Python `ClaudeSDKClient.query`: the frame is written
    /// and stdin is left open for the rest of the conversation. Consume the turn
    /// via [`receive_response`](Self::receive_response).
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when disconnected, or a transport
    /// error when the write fails.
    pub async fn send(&self, prompt: impl Into<String>) -> Result<(), ClaudeError> {
        self.send_with_session(prompt, "default").await
    }

    /// Send one prompt as a streamed user frame under an explicit session id.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when disconnected, or a transport
    /// error when the write fails.
    pub async fn send_with_session(
        &self,
        prompt: impl Into<String>,
        session_id: &str,
    ) -> Result<(), ClaudeError> {
        let runtime = self.runtime().await?;
        let frame = json!({
            "type": "user",
            "message": {"role": "user", "content": prompt.into()},
            "parent_tool_use_id": Value::Null,
            "session_id": session_id,
        });
        runtime.write_frame(&frame).await
    }

    /// Stream a sequence of pre-built user frames, filling in a session id where
    /// a frame omits one, matching the stateful Python async-iterable path.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when disconnected, or a transport
    /// error when a write fails.
    pub async fn send_stream(
        &self,
        frames: impl IntoIterator<Item = Value>,
        session_id: &str,
    ) -> Result<(), ClaudeError> {
        let runtime = self.runtime().await?;
        for mut frame in frames {
            if let Value::Object(map) = &mut frame {
                map.entry("session_id".to_string())
                    .or_insert_with(|| Value::String(session_id.to_owned()));
            }
            runtime.write_frame(&frame).await?;
        }
        Ok(())
    }

    /// Receive the next regular (non-control) message, or `None` at end of
    /// stream.
    ///
    /// Unknown top-level frames are skipped; malformed known frames surface as
    /// [`ClaudeError::MessageParse`]; a reader-observed process error surfaces
    /// with its structured text.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when disconnected, or a transport,
    /// parse, or process error observed by the reader.
    pub async fn receive_message(&self) -> Result<Option<AgentMessage>, ClaudeError> {
        self.runtime().await?.receive().await
    }

    /// Drain messages up to and including the next terminal
    /// [`AgentMessage::Result`], returning them in order.
    ///
    /// Mirrors the Python `receive_response`: the result is included and the
    /// iterator stops after it, leaving any trailing frames for the next turn.
    ///
    /// # Errors
    ///
    /// Propagates receive errors; a clean end of stream before a result returns
    /// the messages seen so far.
    pub async fn receive_response(&self) -> Result<Vec<AgentMessage>, ClaudeError> {
        let runtime = self.runtime().await?;
        let mut messages = Vec::new();
        while let Some(message) = runtime.receive().await? {
            let is_result = matches!(message, AgentMessage::Result(_));
            messages.push(message);
            if is_result {
                break;
            }
        }
        Ok(messages)
    }

    /// Send one prompt and drain its run to the terminal result, aggregating
    /// assistant text into an [`AgentRunResult`].
    ///
    /// This is the convenience one-shot surface used by LADAA. It writes the
    /// prompt, then drains the correct run boundary: for a plain session it
    /// stops at the first result; with hooks or SDK MCP servers configured it
    /// continues through intermediate results (delegated-task turns) until a
    /// run-ending result arrives, matching the deferred stdin-closure contract.
    ///
    /// # Errors
    ///
    /// Returns an error when disconnected, when transport I/O fails, or when the
    /// stream ends before a terminal result.
    pub async fn query(&self, prompt: impl Into<String>) -> Result<AgentRunResult, ClaudeError> {
        let runtime = self.runtime().await?;
        self.send(prompt).await?;

        // With hooks/MCP the run may span multiple result frames (delegated
        // tasks keep stdin open); otherwise the first result ends the run.
        let defer = runtime.defers_end_input();
        let mut assistant_text = String::new();
        loop {
            let (message, run_ended) = runtime.receive_annotated().await?.ok_or_else(|| {
                ClaudeError::CliConnection(
                    "Claude Code ended before emitting an Agent SDK result".into(),
                )
            })?;
            match message {
                AgentMessage::Assistant(assistant) => {
                    for block in assistant.content {
                        if let AgentContentBlock::Text { text } = block {
                            assistant_text.push_str(&text);
                        }
                    }
                }
                AgentMessage::Result(result)
                    // Without deferral, the first result ends the run. With
                    // hooks/MCP a result only ends the run once the reader's
                    // task ledger is empty (`run_ended`); intermediate results
                    // are aggregated and draining continues until the
                    // run-ending one. The flag is computed by the reader at the
                    // moment it emitted this exact result, so it is race-free.
                    if (!defer || run_ended) => {
                        return Ok(build_run_result(result, assistant_text));
                    }
                _ => {}
            }
        }
    }

    // -- Dynamic control operations -----------------------------------------

    /// Interrupt the current agent turn.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when disconnected, or a control
    /// error/timeout.
    pub async fn interrupt(&self) -> Result<(), ClaudeError> {
        self.runtime()
            .await?
            .control(json!({"subtype": "interrupt"}), "interrupt")
            .await
            .map(drop)
    }

    /// Change the permission mode for the rest of the conversation.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when disconnected, or a control
    /// error/timeout.
    pub async fn set_permission_mode(
        &self,
        mode: crate::extensions::PermissionMode,
    ) -> Result<(), ClaudeError> {
        self.runtime()
            .await?
            .control(
                json!({"subtype": "set_permission_mode", "mode": mode.as_cli_value()}),
                "set_permission_mode",
            )
            .await
            .map(drop)
    }

    /// Change the model, or reset to the default when `model` is `None`.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when disconnected, or a control
    /// error/timeout.
    pub async fn set_model(&self, model: Option<&str>) -> Result<(), ClaudeError> {
        self.runtime()
            .await?
            .control(json!({"subtype": "set_model", "model": model}), "set_model")
            .await
            .map(drop)
    }

    /// Rewind tracked files to their state at a specific user message.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when disconnected, or a control
    /// error/timeout.
    pub async fn rewind_files(&self, user_message_id: &str) -> Result<(), ClaudeError> {
        self.runtime()
            .await?
            .control(
                json!({"subtype": "rewind_files", "user_message_id": user_message_id}),
                "rewind_files",
            )
            .await
            .map(drop)
    }

    /// Reconnect a disconnected or failed MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when disconnected, or a control
    /// error/timeout.
    pub async fn reconnect_mcp_server(&self, server_name: &str) -> Result<(), ClaudeError> {
        self.runtime()
            .await?
            .control(
                json!({"subtype": "mcp_reconnect", "serverName": server_name}),
                "mcp_reconnect",
            )
            .await
            .map(drop)
    }

    /// Enable or disable an MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when disconnected, or a control
    /// error/timeout.
    pub async fn toggle_mcp_server(
        &self,
        server_name: &str,
        enabled: bool,
    ) -> Result<(), ClaudeError> {
        self.runtime()
            .await?
            .control(
                json!({"subtype": "mcp_toggle", "serverName": server_name, "enabled": enabled}),
                "mcp_toggle",
            )
            .await
            .map(drop)
    }

    /// Stop a running task by id.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when disconnected, or a control
    /// error/timeout.
    pub async fn stop_task(&self, task_id: &str) -> Result<(), ClaudeError> {
        self.runtime()
            .await?
            .control(
                json!({"subtype": "stop_task", "task_id": task_id}),
                "stop_task",
            )
            .await
            .map(drop)
    }

    /// Query live MCP server connection status.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when disconnected, or a control
    /// error/timeout.
    pub async fn get_mcp_status(&self) -> Result<Value, ClaudeError> {
        self.runtime()
            .await?
            .control(json!({"subtype": "mcp_status"}), "mcp_status")
            .await
    }

    /// Query a breakdown of current context-window usage by category.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when disconnected, or a control
    /// error/timeout.
    pub async fn get_context_usage(&self) -> Result<Value, ClaudeError> {
        self.runtime()
            .await?
            .control(json!({"subtype": "get_context_usage"}), "get_context_usage")
            .await
    }

    /// Query live MCP server connection status as the typed
    /// [`McpStatusResponse`](crate::extensions::McpStatusResponse).
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when disconnected, a control
    /// error/timeout, or [`ClaudeError::MessageParse`] when the response does
    /// not match the typed shape.
    pub async fn get_mcp_status_typed(
        &self,
    ) -> Result<crate::extensions::McpStatusResponse, ClaudeError> {
        let value = self.get_mcp_status().await?;
        serde_json::from_value(value.clone()).map_err(|source| ClaudeError::MessageParse {
            message: format!("invalid mcp_status response: {source}"),
            data: Some(value),
        })
    }

    /// Query the current context-window usage as the typed
    /// [`ContextUsageResponse`](crate::extensions::ContextUsageResponse).
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::CliConnection`] when disconnected, a control
    /// error/timeout, or [`ClaudeError::MessageParse`] when the response does
    /// not match the typed shape.
    pub async fn get_context_usage_typed(
        &self,
    ) -> Result<crate::extensions::ContextUsageResponse, ClaudeError> {
        let value = self.get_context_usage().await?;
        serde_json::from_value(value.clone()).map_err(|source| ClaudeError::MessageParse {
            message: format!("invalid get_context_usage response: {source}"),
            data: Some(value),
        })
    }

    /// Close stdin at the correct run boundary and terminate the transport.
    ///
    /// Idempotent and cancellation-immune: repeated calls are harmless, and the
    /// teardown (reader cancel, transport close, materialized cleanup) runs to
    /// completion on a detached task even if the caller's future is cancelled
    /// mid-await. After close the client is disconnected until a fresh
    /// [`connect`](Self::connect).
    ///
    /// # Errors
    ///
    /// Never returns an error; process-exit status from the transport is
    /// swallowed here (the reader already surfaces process errors on the
    /// message stream).
    pub async fn close(&self) -> Result<(), ClaudeError> {
        let runtime = { self.runtime.lock().await.take() };
        if let Some(runtime) = runtime {
            // Cancellation-immune teardown: spawn the close on a detached task
            // so it runs to completion (cancel handlers -> abort reader ->
            // transport.close -> materialized cleanup) even if the caller's
            // await is cancelled mid-flight. `close` never blocks on a
            // run-ending result — the deferred stdin-close contract is honored
            // by the streaming `query` draining to the run boundary before the
            // caller closes.
            let handle = tokio::spawn(async move { runtime.close().await });
            // Await completion on the happy path; a cancelled caller drops this
            // await but the spawned teardown still finishes.
            let _ = handle.await;
        }
        Ok(())
    }
}

impl<T: AgentTransport> Drop for ClaudeAgentClient<T> {
    fn drop(&mut self) {
        // Best-effort cleanup for a client dropped without an explicit close:
        // abort the reader synchronously so the child process is not leaked.
        if let Some(runtime) = self.runtime.get_mut().take() {
            runtime.abort_reader();
        }
    }
}

/// Aggregate a terminal [`ResultAgentMessage`] and collected assistant text
/// into the convenience [`AgentRunResult`].
fn build_run_result(result: ResultAgentMessage, assistant_text: String) -> AgentRunResult {
    AgentRunResult {
        session_id: result.session_id,
        text: result.result.unwrap_or(assistant_text),
        structured_output: result.structured_output,
        is_error: result.is_error,
        subtype: result.subtype,
        num_turns: result.num_turns,
        duration_ms: result.duration_ms,
        duration_api_ms: result.duration_api_ms,
        total_cost_usd: result.total_cost_usd,
        usage: result.usage,
        model_usage: result.model_usage,
        permission_denials: result.permission_denials,
        deferred_tool_use: result.deferred_tool_use,
        errors: result.errors.unwrap_or_default(),
        api_error_status: result.api_error_status,
        stop_reason: result.stop_reason,
        uuid: result.uuid,
        terminal_reason: result.terminal_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extensions::{PermissionResult, ToolPermissionCallback, ToolPermissionContext};
    use parking_lot::Mutex as StdMutex;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    /// A scripted, concurrency-safe mock transport.
    ///
    /// `incoming` frames are delivered in order by `read`; when `auto_init` is
    /// set, the first `initialize` control request written triggers a synthetic
    /// success response so `connect` completes. Additional frames may be queued
    /// dynamically to model responses that depend on written control requests.
    #[derive(Debug)]
    struct MockTransport {
        incoming: StdMutex<VecDeque<Frame>>,
        outgoing: StdMutex<Vec<Value>>,
        ready: std::sync::atomic::AtomicBool,
        closed: std::sync::atomic::AtomicBool,
        end_input_calls: AtomicUsize,
        close_calls: AtomicUsize,
        auto_init: bool,
        connect_result: StdMutex<Option<ClaudeError>>,
        read_notify: Arc<Notify>,
    }

    use tokio::sync::Notify;

    /// One scripted read: a frame now, or a signal that the stream is at EOF or
    /// should surface an error.
    #[derive(Debug)]
    enum Frame {
        Value(Value),
        Eof,
        Error(String),
    }

    impl MockTransport {
        fn new(auto_init: bool) -> Arc<Self> {
            Arc::new(Self {
                incoming: StdMutex::new(VecDeque::new()),
                outgoing: StdMutex::new(Vec::new()),
                ready: std::sync::atomic::AtomicBool::new(false),
                closed: std::sync::atomic::AtomicBool::new(false),
                end_input_calls: AtomicUsize::new(0),
                close_calls: AtomicUsize::new(0),
                auto_init,
                connect_result: StdMutex::new(None),
                read_notify: Arc::new(Notify::new()),
            })
        }

        fn push(&self, frame: Value) {
            self.incoming.lock().push_back(Frame::Value(frame));
            self.read_notify.notify_waiters();
        }

        fn push_eof(&self) {
            self.incoming.lock().push_back(Frame::Eof);
            self.read_notify.notify_waiters();
        }

        fn push_error(&self, message: &str) {
            self.incoming
                .lock()
                .push_back(Frame::Error(message.to_owned()));
            self.read_notify.notify_waiters();
        }

        fn outgoing(&self) -> Vec<Value> {
            self.outgoing.lock().clone()
        }
    }

    #[async_trait]
    impl AgentTransport for MockTransport {
        async fn connect(
            &self,
            _options: &AgentOptions,
            _mcp_servers: &[SdkMcpServerDescriptor],
        ) -> Result<(), ClaudeError> {
            if let Some(error) = self.connect_result.lock().take() {
                return Err(error);
            }
            self.ready.store(true, Ordering::SeqCst);
            Ok(())
        }

        async fn write(&self, raw: &str) -> Result<(), ClaudeError> {
            let frame: Value = serde_json::from_str(raw.trim_end())
                .map_err(|e| ClaudeError::TransportError(e.to_string()))?;
            let is_initialize = frame
                .get("request")
                .and_then(|request| request.get("subtype"))
                .and_then(Value::as_str)
                == Some("initialize");
            let request_id = frame.get("request_id").cloned();
            self.outgoing.lock().push(frame);
            if self.auto_init && is_initialize {
                if let Some(request_id) = request_id {
                    self.push(json!({
                        "type": "control_response",
                        "response": {
                            "subtype": "success",
                            "request_id": request_id,
                            "response": {"commands": []}
                        }
                    }));
                }
            }
            Ok(())
        }

        async fn read(&self) -> Result<Option<Value>, ClaudeError> {
            loop {
                // Register the notified future BEFORE inspecting the queue so a
                // push that races between the pop and the await is not missed.
                let notified = self.read_notify.notified();
                let popped = self.incoming.lock().pop_front();
                match popped {
                    Some(Frame::Value(value)) => return Ok(Some(value)),
                    Some(Frame::Eof) => {
                        self.ready.store(false, Ordering::SeqCst);
                        return Ok(None);
                    }
                    Some(Frame::Error(message)) => {
                        self.ready.store(false, Ordering::SeqCst);
                        return Err(ClaudeError::Process {
                            message,
                            exit_code: Some(1),
                            stderr: None,
                        });
                    }
                    None => {
                        // Await more scripted input rather than busy-looping.
                        notified.await;
                    }
                }
            }
        }

        fn is_ready(&self) -> bool {
            self.ready.load(Ordering::SeqCst) && !self.closed.load(Ordering::SeqCst)
        }

        async fn end_input(&self) -> Result<(), ClaudeError> {
            self.end_input_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn close(&self) -> Result<(), ClaudeError> {
            self.close_calls.fetch_add(1, Ordering::SeqCst);
            self.closed.store(true, Ordering::SeqCst);
            self.ready.store(false, Ordering::SeqCst);
            self.read_notify.notify_waiters();
            Ok(())
        }
    }

    /// A permission callback that records its calls and returns a fixed result.
    #[derive(Debug)]
    struct RecordingCallback {
        calls: AtomicUsize,
        result: StdMutex<PermissionResult>,
    }

    #[async_trait]
    impl ToolPermissionCallback for RecordingCallback {
        async fn can_use_tool(
            &self,
            _tool_name: &str,
            _input: &Value,
            _context: &ToolPermissionContext,
        ) -> Result<PermissionResult, ClaudeError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.result.lock().clone())
        }
    }

    fn assistant(text: &str) -> Value {
        json!({
            "type": "assistant",
            "session_id": "session-1",
            "message": {
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": text}]
            }
        })
    }

    fn result_frame(session: &str) -> Value {
        json!({
            "type": "result",
            "subtype": "success",
            "duration_ms": 20,
            "duration_api_ms": 10,
            "is_error": false,
            "num_turns": 1,
            "session_id": session
        })
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
        assert_eq!(response["result"]["isError"], json!(false));
        assert_eq!(response["result"]["content"][0]["type"], "text");
        assert_eq!(response["result"]["content"][0]["text"], "{\"value\":42}");
    }

    #[tokio::test]
    async fn tools_call_preserves_text_and_image() {
        // An SDK tool returning both a text and image item reaches tools/call
        // with both content items and no data loss.
        let tool = SdkMcpTool::new(
            "render",
            "render",
            json!({"type": "object"}),
            None,
            |_input| async move {
                Ok(ToolCallResult {
                    content: vec![
                        ToolContent::text("caption"),
                        ToolContent::image("aGVsbG8=", "image/png"),
                    ],
                    is_error: false,
                })
            },
        )
        .unwrap();
        let mut server = SdkMcpServer::new("art", "1.0.0").unwrap();
        server.add_tool(tool).unwrap();

        let response = server
            .handle(&json!({
                "jsonrpc": "2.0", "id": 7, "method": "tools/call",
                "params": {"name": "render", "arguments": {}}
            }))
            .await;
        let content = response["result"]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0], json!({"type": "text", "text": "caption"}));
        assert_eq!(
            content[1],
            json!({"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"})
        );
    }

    #[tokio::test]
    async fn tools_call_converts_resource_link_and_resource() {
        // resource_link renders to joined text; an embedded text resource
        // becomes text; a binary embedded resource is dropped, matching the
        // official bridge conversions.
        let tool = SdkMcpTool::new(
            "docs",
            "docs",
            json!({"type": "object"}),
            None,
            |_input| async move {
                Ok(ToolCallResult {
                    content: vec![
                        ToolContent::ResourceLink {
                            uri: "file:///a".into(),
                            name: Some("A".into()),
                            description: Some("desc".into()),
                            mime_type: None,
                        },
                        ToolContent::Resource {
                            resource: json!({"uri": "file:///b", "text": "inline"}),
                        },
                        ToolContent::Resource {
                            resource: json!({"uri": "file:///c", "blob": "AAAA"}),
                        },
                    ],
                    is_error: false,
                })
            },
        )
        .unwrap();
        let mut server = SdkMcpServer::new("docsrv", "1.0.0").unwrap();
        server.add_tool(tool).unwrap();
        let response = server
            .handle(&json!({
                "jsonrpc": "2.0", "id": 8, "method": "tools/call",
                "params": {"name": "docs", "arguments": {}}
            }))
            .await;
        let content = response["result"]["content"].as_array().unwrap();
        // Binary resource dropped, so 2 items remain.
        assert_eq!(content.len(), 2);
        assert_eq!(
            content[0],
            json!({"type": "text", "text": "A\nfile:///a\ndesc"})
        );
        assert_eq!(content[1], json!({"type": "text", "text": "inline"}));
    }

    #[tokio::test]
    async fn tools_list_omits_absent_annotations_and_emits_meta() {
        // Absent annotations are omitted from tools/list; maxResultSizeChars
        // rides in _meta under the namespaced key, never as an annotation.
        let plain = SdkMcpTool::new("plain", "d", json!({"type": "object"}), None, |_| async {
            Ok(ToolCallResult::text("ok"))
        })
        .unwrap();
        let sized = SdkMcpTool::new(
            "sized",
            "d",
            json!({"type": "object"}),
            ToolAnnotations {
                read_only_hint: Some(true),
                max_result_size_chars: Some(2048),
                ..Default::default()
            },
            |_| async { Ok(ToolCallResult::text("ok")) },
        )
        .unwrap();
        let mut server = SdkMcpServer::new("srv", "1.0.0").unwrap();
        server.add_tool(plain).unwrap();
        server.add_tool(sized).unwrap();
        let response = server
            .handle(&json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}))
            .await;
        let tools = response["result"]["tools"].as_array().unwrap();
        let plain = tools.iter().find(|t| t["name"] == "plain").unwrap();
        assert!(plain.get("annotations").is_none());
        assert!(plain.get("_meta").is_none());
        let sized = tools.iter().find(|t| t["name"] == "sized").unwrap();
        assert_eq!(sized["annotations"]["readOnlyHint"], json!(true));
        // maxResultSizeChars must not appear as an annotation.
        assert!(sized["annotations"].get("maxResultSizeChars").is_none());
        assert_eq!(sized["_meta"]["anthropic/maxResultSizeChars"], json!(2048));
    }

    #[tokio::test]
    async fn runs_prompt_to_typed_result() {
        let transport = MockTransport::new(true);
        transport.push(assistant("diagnosis"));
        transport.push(result_frame("session-1"));

        let mut client = ClaudeAgentClient::new(
            TransportHandle(Arc::clone(&transport)),
            AgentOptions::default(),
        );
        client.connect().await.unwrap();
        let result = client.query("diagnose").await.unwrap();
        assert_eq!(result.session_id, "session-1");
        assert_eq!(result.text, "diagnosis");
        assert!(!result.is_error);
        client.close().await.unwrap();
    }

    /// A cloneable transport handle so the test can hold a reference to the
    /// shared mock while the client owns its `Arc`.
    #[derive(Debug)]
    struct TransportHandle(Arc<MockTransport>);

    #[async_trait]
    impl AgentTransport for TransportHandle {
        async fn connect(
            &self,
            options: &AgentOptions,
            servers: &[SdkMcpServerDescriptor],
        ) -> Result<(), ClaudeError> {
            self.0.connect(options, servers).await
        }
        async fn write(&self, raw: &str) -> Result<(), ClaudeError> {
            self.0.write(raw).await
        }
        async fn read(&self) -> Result<Option<Value>, ClaudeError> {
            self.0.read().await
        }
        fn is_ready(&self) -> bool {
            self.0.is_ready()
        }
        async fn end_input(&self) -> Result<(), ClaudeError> {
            self.0.end_input().await
        }
        async fn close(&self) -> Result<(), ClaudeError> {
            self.0.close().await
        }
    }

    #[tokio::test]
    async fn deferred_run_skips_intermediate_result_with_delegated_task() {
        // An SDK MCP server enables deferred stdin closure, so a result that
        // arrives while a delegated agent task is in flight is intermediate:
        // the aggregate query must keep draining until the run-ending result.
        let server = SdkMcpServer::new("deferrer", "1.0.0").unwrap();
        let transport = MockTransport::new(true);
        let mut client = ClaudeAgentClient::new(
            TransportHandle(Arc::clone(&transport)),
            AgentOptions::default(),
        );
        client.add_mcp_server(server).unwrap();
        client.connect().await.unwrap();

        let query = {
            let transport = Arc::clone(&transport);
            tokio::spawn(async move {
                // A delegated agent task starts, then an intermediate result
                // arrives with the task still in flight (must not end the run).
                transport.push(json!({
                    "type": "system",
                    "subtype": "task_started",
                    "task_id": "t1",
                    "task_type": "local_agent",
                    "description": "delegated agent work",
                    "uuid": "u1",
                    "session_id": "mid",
                    "tool_use_id": "tu1"
                }));
                transport.push(assistant("intermediate"));
                transport.push(json!({
                    "type": "result",
                    "subtype": "success",
                    "duration_ms": 1,
                    "duration_api_ms": 1,
                    "is_error": false,
                    "num_turns": 1,
                    "session_id": "mid"
                }));
                // The task settles, then the run-ending result arrives.
                transport.push(json!({
                    "type": "system",
                    "subtype": "task_updated",
                    "task_id": "t1",
                    "patch": {"status": "completed"}
                }));
                transport.push(assistant("final"));
                transport.push(result_frame("done"));
            })
        };

        let result = client.query("go").await.unwrap();
        // The run-ending result (session "done") is returned, not "mid".
        assert_eq!(result.session_id, "done");
        query.await.unwrap();
        client.close().await.unwrap();
    }

    #[tokio::test]
    async fn failed_handshake_rolls_back_and_retry_reconnects() {
        let transport = MockTransport::new(false); // no auto-init: initialize will time out
                                                   // First connect: initialize never gets a response.
        let mut client = ClaudeAgentClient::new(
            TransportHandle(Arc::clone(&transport)),
            AgentOptions {
                initialize_timeout: Duration::from_millis(50),
                ..AgentOptions::default()
            },
        );
        let error = client.connect().await.unwrap_err();
        assert!(matches!(error, ClaudeError::ControlTimeout { .. }));
        assert!(!client.is_ready().await);
        // Transport was closed on rollback.
        assert_eq!(transport.close_calls.load(Ordering::SeqCst), 1);

        // Retry with auto-init behaviour: push a success response after the
        // client writes initialize. Model auto-init by watching outgoing.
        let retry_transport = MockTransport::new(true);
        let mut client = ClaudeAgentClient::new(
            TransportHandle(Arc::clone(&retry_transport)),
            AgentOptions::default(),
        );
        client.connect().await.unwrap();
        assert!(client.is_ready().await);
        // A fresh handshake wrote a new initialize request.
        assert!(retry_transport.outgoing().iter().any(|f| f
            .get("request")
            .and_then(|r| r.get("subtype"))
            .and_then(Value::as_str)
            == Some("initialize")));
        client.close().await.unwrap();
    }

    #[tokio::test]
    async fn receive_response_stops_at_first_result() {
        let transport = MockTransport::new(true);
        transport.push(assistant("Answer"));
        transport.push(result_frame("test"));
        transport.push(assistant("should not be consumed by this turn"));

        let mut client = ClaudeAgentClient::new(
            TransportHandle(Arc::clone(&transport)),
            AgentOptions::default(),
        );
        client.connect().await.unwrap();
        client.send("q").await.unwrap();
        let turn = client.receive_response().await.unwrap();
        assert_eq!(turn.len(), 2);
        assert!(matches!(turn[0], AgentMessage::Assistant(_)));
        assert!(matches!(turn[1], AgentMessage::Result(_)));
        // The trailing assistant is still available for the next receive.
        let next = client.receive_message().await.unwrap();
        assert!(matches!(next, Some(AgentMessage::Assistant(_))));
        client.close().await.unwrap();
    }

    #[tokio::test]
    async fn concurrent_send_and_receive() {
        let transport = MockTransport::new(true);
        let mut client = ClaudeAgentClient::new(
            TransportHandle(Arc::clone(&transport)),
            AgentOptions::default(),
        );
        client.connect().await.unwrap();
        let client = Arc::new(client);

        // Receiver waits for a frame that a concurrent sender triggers.
        let receiver = {
            let client = Arc::clone(&client);
            tokio::spawn(async move { client.receive_message().await })
        };
        // Give the receiver a moment to park on the empty channel.
        tokio::time::sleep(Duration::from_millis(20)).await;
        client.send("question").await.unwrap();
        transport.push(assistant("Response 1"));

        let message = receiver.await.unwrap().unwrap();
        assert!(matches!(message, Some(AgentMessage::Assistant(_))));
        client.close().await.unwrap();
    }

    #[tokio::test]
    async fn routes_control_response_by_id() {
        let transport = MockTransport::new(true);
        let mut client = ClaudeAgentClient::new(
            TransportHandle(Arc::clone(&transport)),
            AgentOptions::default(),
        );
        client.connect().await.unwrap();
        let client = Arc::new(client);

        // Fire two control requests concurrently; respond out of order by id.
        let a = {
            let client = Arc::clone(&client);
            tokio::spawn(async move { client.get_mcp_status().await })
        };
        let b = {
            let client = Arc::clone(&client);
            tokio::spawn(async move { client.get_context_usage().await })
        };
        // Wait until both requests were written, then answer by matching id.
        loop {
            let out = transport.outgoing();
            let ids: Vec<(String, String)> = out
                .iter()
                .filter_map(|f| {
                    let id = f.get("request_id").and_then(Value::as_str)?.to_owned();
                    let sub = f
                        .get("request")
                        .and_then(|r| r.get("subtype"))
                        .and_then(Value::as_str)?
                        .to_owned();
                    Some((sub, id))
                })
                .collect();
            if ids.iter().filter(|(s, _)| s != "initialize").count() >= 2 {
                // Respond to context usage first (out of order), then status.
                for (sub, id) in ids.into_iter().filter(|(s, _)| s != "initialize") {
                    let payload = if sub == "get_context_usage" {
                        json!({"percentage": 42})
                    } else {
                        json!({"mcpServers": []})
                    };
                    transport.push(json!({
                        "type": "control_response",
                        "response": {"subtype": "success", "request_id": id, "response": payload}
                    }));
                }
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        let status = a.await.unwrap().unwrap();
        let usage = b.await.unwrap().unwrap();
        assert_eq!(status["mcpServers"], json!([]));
        assert_eq!(usage["percentage"], 42);
        client.close().await.unwrap();
    }

    #[tokio::test]
    async fn cancels_slow_inbound_control_handler() {
        // A slow permission callback that never returns until cancelled.
        #[derive(Debug)]
        struct SlowCallback {
            started: Arc<Notify>,
        }
        #[async_trait]
        impl ToolPermissionCallback for SlowCallback {
            async fn can_use_tool(
                &self,
                _tool_name: &str,
                _input: &Value,
                _context: &ToolPermissionContext,
            ) -> Result<PermissionResult, ClaudeError> {
                self.started.notify_waiters();
                // Sleep long enough that the cancel arrives first.
                tokio::time::sleep(Duration::from_secs(30)).await;
                Ok(PermissionResult::allow())
            }
        }

        let started = Arc::new(Notify::new());
        let transport = MockTransport::new(true);
        let options = AgentOptions {
            can_use_tool: Some(Arc::new(SlowCallback {
                started: Arc::clone(&started),
            })),
            ..AgentOptions::default()
        };
        let mut client = ClaudeAgentClient::new(TransportHandle(Arc::clone(&transport)), options);
        client.connect().await.unwrap();

        // Inbound can_use_tool request, then a cancel for the same id.
        transport.push(json!({
            "type": "control_request",
            "request_id": "perm_1",
            "request": {"subtype": "can_use_tool", "tool_name": "Bash", "input": {}}
        }));
        // Wait until the handler started, then cancel it.
        started.notified().await;
        transport.push(json!({
            "type": "control_cancel_request",
            "request_id": "perm_1"
        }));
        // Give the reader time to process the cancel.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // A cancelled handler writes no control response.
        let has_response = transport.outgoing().iter().any(|f| {
            f.get("type").and_then(Value::as_str) == Some("control_response")
                && f.get("response")
                    .and_then(|r| r.get("request_id"))
                    .and_then(Value::as_str)
                    == Some("perm_1")
        });
        assert!(!has_response, "cancelled request must not write a response");
        client.close().await.unwrap();
    }

    #[tokio::test]
    async fn permission_callback_allows_with_updated_input() {
        let callback = Arc::new(RecordingCallback {
            calls: AtomicUsize::new(0),
            result: StdMutex::new(PermissionResult::Allow {
                updated_input: Some(json!({"safe": true})),
                updated_permissions: None,
            }),
        });
        let transport = MockTransport::new(true);
        let options = AgentOptions {
            can_use_tool: Some(callback.clone()),
            ..AgentOptions::default()
        };
        let mut client = ClaudeAgentClient::new(TransportHandle(Arc::clone(&transport)), options);
        client.connect().await.unwrap();

        transport.push(json!({
            "type": "control_request",
            "request_id": "perm_9",
            "request": {"subtype": "can_use_tool", "tool_name": "Write", "input": {"orig": 1}}
        }));
        // Drain until the response is written.
        for _ in 0..50 {
            if transport.outgoing().iter().any(|f| {
                f.get("response")
                    .and_then(|r| r.get("request_id"))
                    .and_then(Value::as_str)
                    == Some("perm_9")
            }) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(callback.calls.load(Ordering::SeqCst), 1);
        let response = transport
            .outgoing()
            .into_iter()
            .find(|f| {
                f.get("response")
                    .and_then(|r| r.get("request_id"))
                    .and_then(Value::as_str)
                    == Some("perm_9")
            })
            .unwrap();
        assert_eq!(response["response"]["response"]["behavior"], "allow");
        assert_eq!(
            response["response"]["response"]["updatedInput"],
            json!({"safe": true})
        );
        client.close().await.unwrap();
    }

    #[tokio::test]
    async fn hook_callback_invokes_with_minimal_input() {
        // A hook_callback control request carrying a minimal input object (no
        // base fields) still invokes the callback and writes a success
        // response, matching the official permissive dispatch.
        use crate::extensions::{
            HookContext, HookEvent, HookHandler, HookInput, HookJSONOutput, HookMatcher,
            SyncHookJSONOutput,
        };

        #[derive(Debug)]
        struct RecordingHook {
            seen: Arc<StdMutex<Option<Value>>>,
        }
        #[async_trait]
        impl HookHandler for RecordingHook {
            async fn call(
                &self,
                input: &HookInput,
                _tool_use_id: Option<&str>,
                _context: &HookContext,
            ) -> Result<HookJSONOutput, ClaudeError> {
                *self.seen.lock() = Some(input.raw().clone());
                Ok(HookJSONOutput::Sync(SyncHookJSONOutput {
                    system_message: Some("ok".into()),
                    ..Default::default()
                }))
            }
        }

        let seen = Arc::new(StdMutex::new(None));
        let hook: crate::extensions::HookCallback = Arc::new(RecordingHook {
            seen: Arc::clone(&seen),
        });
        let mut hooks = std::collections::BTreeMap::new();
        hooks.insert(
            HookEvent::PreToolUse,
            vec![HookMatcher::new(None, vec![hook])],
        );
        let transport = MockTransport::new(true);
        let options = AgentOptions {
            hooks: Some(hooks),
            ..AgentOptions::default()
        };
        let mut client = ClaudeAgentClient::new(TransportHandle(Arc::clone(&transport)), options);
        client.connect().await.unwrap();

        // The first registered callback id is deterministic (`hook_0`).
        transport.push(json!({
            "type": "control_request",
            "request_id": "hook_req_1",
            "request": {
                "subtype": "hook_callback",
                "callback_id": "hook_0",
                "input": {"test": "data"},
                "tool_use_id": "tool-123"
            }
        }));
        for _ in 0..50 {
            if seen.lock().is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(seen.lock().clone(), Some(json!({"test": "data"})));
        let response = transport.outgoing().into_iter().find(|f| {
            f.get("response")
                .and_then(|r| r.get("request_id"))
                .and_then(Value::as_str)
                == Some("hook_req_1")
        });
        let response = response.expect("hook callback must write a success response");
        assert_eq!(response["response"]["subtype"], "success");
        client.close().await.unwrap();
    }

    #[tokio::test]
    async fn process_error_after_error_result_uses_result_text() {
        let transport = MockTransport::new(true);
        let mut client = ClaudeAgentClient::new(
            TransportHandle(Arc::clone(&transport)),
            AgentOptions::default(),
        );
        client.connect().await.unwrap();
        client.send("q").await.unwrap();
        // Frames arrive only after the initialize handshake completes.
        transport.push(json!({
            "type": "result",
            "subtype": "error_max_turns",
            "duration_ms": 1,
            "duration_api_ms": 1,
            "is_error": true,
            "num_turns": 1,
            "session_id": "s",
            "errors": ["boom", "bang"]
        }));
        transport.push_error("exit code 1");
        // First message is the error result (delivered, not raised).
        let first = client.receive_message().await.unwrap().unwrap();
        assert!(matches!(first, AgentMessage::Result(_)));
        // The trailing process error is replaced by the result's error text.
        let err = client.receive_message().await.unwrap_err();
        match err {
            ClaudeError::Process { message, .. } => {
                assert!(message.contains("Claude Code returned an error result"));
                assert!(message.contains("boom; bang"));
            }
            other => panic!("expected Process error, got {other:?}"),
        }
        client.close().await.unwrap();
    }

    #[tokio::test]
    async fn process_error_without_error_result_is_unchanged() {
        let transport = MockTransport::new(true);
        let mut client = ClaudeAgentClient::new(
            TransportHandle(Arc::clone(&transport)),
            AgentOptions::default(),
        );
        client.connect().await.unwrap();
        client.send("q").await.unwrap();
        transport.push(assistant("hi"));
        transport.push_error("exit code 1");
        let _ = client.receive_message().await.unwrap();
        let err = client.receive_message().await.unwrap_err();
        match err {
            ClaudeError::Process { message, .. } => {
                assert_eq!(message, "exit code 1");
            }
            other => panic!("expected Process error, got {other:?}"),
        }
        client.close().await.unwrap();
    }

    #[tokio::test]
    async fn close_is_idempotent_and_reconnect_has_no_stale_frames() {
        let transport = MockTransport::new(true);
        transport.push(assistant("first"));
        transport.push(result_frame("s1"));

        let mut client = ClaudeAgentClient::new(
            TransportHandle(Arc::clone(&transport)),
            AgentOptions::default(),
        );
        client.connect().await.unwrap();
        let r1 = client.query("one").await.unwrap();
        assert_eq!(r1.text, "first");
        client.close().await.unwrap();
        client.close().await.unwrap(); // idempotent
        assert!(transport.close_calls.load(Ordering::SeqCst) >= 1);

        // Reconnect on a fresh transport: no stale queued frames leak in.
        let transport2 = MockTransport::new(true);
        transport2.push(assistant("second"));
        transport2.push(result_frame("s2"));
        let mut client2 = ClaudeAgentClient::new(
            TransportHandle(Arc::clone(&transport2)),
            AgentOptions::default(),
        );
        client2.connect().await.unwrap();
        let r2 = client2.query("two").await.unwrap();
        assert_eq!(r2.text, "second");
        assert_eq!(r2.session_id, "s2");
        client2.close().await.unwrap();
    }

    #[tokio::test]
    async fn disconnected_calls_error() {
        let transport = MockTransport::new(true);
        let client = ClaudeAgentClient::new(
            TransportHandle(Arc::clone(&transport)),
            AgentOptions::default(),
        );
        assert!(matches!(
            client.send("x").await,
            Err(ClaudeError::CliConnection(_))
        ));
        assert!(matches!(
            client.interrupt().await,
            Err(ClaudeError::CliConnection(_))
        ));
        // close on a never-connected client is a harmless no-op.
        client.close().await.unwrap();
    }

    #[tokio::test]
    async fn clean_eof_before_result_ends_stream_and_query_errors() {
        let transport = MockTransport::new(true);
        let mut client = ClaudeAgentClient::new(
            TransportHandle(Arc::clone(&transport)),
            AgentOptions::default(),
        );
        client.connect().await.unwrap();
        client.send("q").await.unwrap();
        transport.push(assistant("partial"));
        transport.push_eof();

        // receive_message drains the assistant then reports end of stream.
        let first = client.receive_message().await.unwrap();
        assert!(matches!(first, Some(AgentMessage::Assistant(_))));
        let end = client.receive_message().await.unwrap();
        assert!(end.is_none(), "clean EOF yields None, not an error");
        client.close().await.unwrap();

        // A one-shot query over a stream that ends without a result errors.
        let transport2 = MockTransport::new(true);
        let mut client2 = ClaudeAgentClient::new(
            TransportHandle(Arc::clone(&transport2)),
            AgentOptions::default(),
        );
        client2.connect().await.unwrap();
        // Drive the query concurrently; feed an assistant then EOF.
        let query = {
            let transport2 = Arc::clone(&transport2);
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(10)).await;
                transport2.push(assistant("no result follows"));
                transport2.push_eof();
            })
        };
        let err = client2.query("go").await.unwrap_err();
        assert!(matches!(err, ClaudeError::CliConnection(_)));
        query.await.unwrap();
        client2.close().await.unwrap();
    }

    #[tokio::test]
    async fn can_use_tool_conflicts_with_permission_prompt_tool_name() {
        let options = AgentOptions {
            can_use_tool: Some(Arc::new(RecordingCallback {
                calls: AtomicUsize::new(0),
                result: StdMutex::new(PermissionResult::allow()),
            })),
            permission_prompt_tool_name: Some("mcp__x".into()),
            ..Default::default()
        };
        let transport = MockTransport::new(true);
        let mut client = ClaudeAgentClient::new(TransportHandle(transport), options);
        let err = client.connect().await.unwrap_err();
        assert!(
            matches!(&err, ClaudeError::InvalidConfig(m) if m.contains("permission_prompt_tool_name")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn can_use_tool_forces_stdio_permission_prompt_tool() {
        let options = AgentOptions {
            can_use_tool: Some(Arc::new(RecordingCallback {
                calls: AtomicUsize::new(0),
                result: StdMutex::new(PermissionResult::allow()),
            })),
            ..Default::default()
        };
        let resolved = options.resolve_permission_options().unwrap();
        assert_eq!(
            resolved.permission_prompt_tool_name.as_deref(),
            Some("stdio")
        );
    }

    #[test]
    fn can_use_tool_shadow_warning_reports_allowed_tool() {
        // "Read" and "Read()" resolve to the same tool: reported once.
        let options = AgentOptions {
            can_use_tool: Some(Arc::new(RecordingCallback {
                calls: AtomicUsize::new(0),
                result: StdMutex::new(PermissionResult::allow()),
            })),
            allowed_tools: vec!["Read".into(), "Read()".into(), "Bash(ls:*)".into()],
            ..Default::default()
        };
        let warning = options.can_use_tool_shadow_warning().unwrap();
        let diagnostic = warning.message();
        assert!(diagnostic.contains("Read"), "diagnostic: {diagnostic}");
        assert_eq!(
            diagnostic.matches("Read").count(),
            1,
            "deduped: {diagnostic}"
        );
        // A narrowed Bash rule does not shadow.
        assert!(!diagnostic.contains("Bash"), "diagnostic: {diagnostic}");
    }

    #[test]
    fn can_use_tool_shadow_warning_bypass_mode() {
        let options = AgentOptions {
            can_use_tool: Some(Arc::new(RecordingCallback {
                calls: AtomicUsize::new(0),
                result: StdMutex::new(PermissionResult::allow()),
            })),
            permission_mode: Some(PermissionMode::BypassPermissions),
            ..Default::default()
        };
        let warning = options.can_use_tool_shadow_warning().unwrap();
        assert!(
            warning.message().contains("bypassPermissions"),
            "warning: {warning}"
        );
    }

    #[test]
    fn no_shadow_warning_without_callback() {
        let options = AgentOptions {
            allowed_tools: vec!["Read".into()],
            ..Default::default()
        };
        assert!(options.can_use_tool_shadow_warning().is_none());
    }

    #[test]
    fn mcp_server_config_accepts_omitted_stdio_type() {
        let config: McpServerConfig =
            serde_json::from_value(json!({"command": "/srv", "args": ["--x"]})).unwrap();
        match config {
            McpServerConfig::Stdio(stdio) => {
                assert_eq!(stdio.command, "/srv");
                assert_eq!(stdio.args.as_deref(), Some(&["--x".to_string()][..]));
            }
            other => panic!("expected stdio, got {other:?}"),
        }
        // Explicit type still parses.
        let explicit: McpServerConfig =
            serde_json::from_value(json!({"type": "stdio", "command": "c"})).unwrap();
        assert!(matches!(explicit, McpServerConfig::Stdio(_)));
    }

    #[test]
    fn mcp_servers_config_string_and_path_forms() {
        let string = McpServers::ConfigString("{\"mcpServers\":{}}".into());
        assert!(!string.is_empty());
        assert!(string.as_map().is_none());
        let path = McpServers::ConfigPath(PathBuf::from("/etc/mcp.json"));
        assert!(!path.is_empty());
        let empty: McpServers = BTreeMap::new().into();
        assert!(empty.is_empty());
    }

    #[tokio::test]
    async fn options_sdk_server_without_instance_is_rejected() {
        let mut map = BTreeMap::new();
        map.insert(
            "calc".to_string(),
            McpServerConfig::Sdk(crate::extensions::McpSdkServerConfig {
                name: "calc".into(),
            }),
        );
        let options = AgentOptions {
            mcp_servers: McpServers::Map(map),
            ..Default::default()
        };
        let transport = MockTransport::new(true);
        let mut client = ClaudeAgentClient::new(TransportHandle(transport), options);
        let err = client.connect().await.unwrap_err();
        assert!(
            matches!(&err, ClaudeError::InvalidConfig(m) if m.contains("no registered")),
            "unexpected: {err:?}"
        );
    }

    #[tokio::test]
    async fn options_sdk_server_with_registered_instance_connects() {
        let mut map = BTreeMap::new();
        map.insert(
            "calc".to_string(),
            McpServerConfig::Sdk(crate::extensions::McpSdkServerConfig {
                name: "calc".into(),
            }),
        );
        let options = AgentOptions {
            mcp_servers: McpServers::Map(map),
            ..Default::default()
        };
        let transport = MockTransport::new(true);
        let mut client = ClaudeAgentClient::new(TransportHandle(transport), options);
        client
            .add_mcp_server(SdkMcpServer::new("calc", "1.0.0").unwrap())
            .unwrap();
        client.connect().await.unwrap();
        assert!(client.is_ready().await);
        client.close().await.unwrap();
    }

    #[tokio::test]
    async fn server_info_errors_before_connect() {
        let transport = MockTransport::new(true);
        let client = ClaudeAgentClient::new(TransportHandle(transport), AgentOptions::default());
        let err = client.server_info().await.unwrap_err();
        assert!(err.is_cli_connection(), "unexpected: {err:?}");
    }

    #[test]
    fn hook_event_name_empty_falls_back() {
        // An explicitly empty hook_event still falls through to hook_name.
        let msg = AgentMessage::from_value(json!({
            "type": "system",
            "subtype": "hook_started",
            "hook_event": "",
            "hook_name": "PreToolUse",
        }))
        .unwrap()
        .unwrap();
        let AgentMessage::HookEvent(e) = msg else {
            panic!("expected hook event");
        };
        assert_eq!(e.hook_event_name, "PreToolUse");
    }
}
