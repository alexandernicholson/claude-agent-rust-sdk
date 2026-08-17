//! Public extension and configuration schema for the Claude Code Agent SDK.
//!
//! This module mirrors the official Python Agent SDK's public option surface
//! (`types.py` / `__init__.py`): permission modes and updates, hook events and
//! typed hook inputs/outputs, agent definitions, skills, plugins, MCP server
//! configuration and status, sandbox settings, system-prompt and tools
//! presets, thinking/output/task-budget configuration, and the callback
//! contracts consumed by [`AgentOptions`](crate::agent::AgentOptions).
//!
//! Wire serialization matches the CLI exactly: fields use official camelCase or
//! CLI spelling, optional fields are omitted when absent (`None`) and remain
//! distinguishable from explicit empty collections.

use crate::error::ClaudeError;
use crate::types::EffortLevel;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::sync::Arc;

/// Advisory warning that `can_use_tool` is shadowed by auto-approval rules.
///
/// The official Python SDK exposes the same condition as a dedicated
/// `UserWarning`, while the TypeScript SDK emits process warning code
/// `CLAUDE_SDK_CAN_USE_TOOL_SHADOWED`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanUseToolShadowedWarning {
    message: String,
}

impl CanUseToolShadowedWarning {
    /// Construct a warning with the rendered diagnostic.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Return the rendered diagnostic.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for CanUseToolShadowedWarning {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CanUseToolShadowedWarning {}

// ---------------------------------------------------------------------------
// Permission mode and setting sources
// ---------------------------------------------------------------------------

/// Permission behavior passed to Claude Code.
///
/// Mirrors the official `PermissionMode` literal
/// (`"default" | "acceptEdits" | "plan" | "bypassPermissions" | "dontAsk" |
/// "auto"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionMode {
    /// Standard permission behavior; prompts for dangerous operations.
    #[serde(rename = "default")]
    Default,
    /// Automatically accept file edits.
    #[serde(rename = "acceptEdits")]
    AcceptEdits,
    /// Planning only; do not execute tools.
    #[serde(rename = "plan")]
    Plan,
    /// Bypass every Claude Code permission check.
    ///
    /// Hosts should avoid this for unattended workloads and enforce their own
    /// deterministic tool gate when it is unavoidable.
    #[serde(rename = "bypassPermissions")]
    BypassPermissions,
    /// Deny tools that would otherwise require an interactive prompt.
    #[serde(rename = "dontAsk")]
    DontAsk,
    /// Automatically decide whether to prompt.
    #[serde(rename = "auto")]
    Auto,
}

impl PermissionMode {
    /// Return the Claude Code command-line representation.
    #[must_use]
    pub const fn as_cli_value(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::AcceptEdits => "acceptEdits",
            Self::Plan => "plan",
            Self::BypassPermissions => "bypassPermissions",
            Self::DontAsk => "dontAsk",
            Self::Auto => "auto",
        }
    }
}

/// Filesystem setting layer loaded by Claude Code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SettingSource {
    /// User-wide settings (`~/.claude/settings.json`).
    #[serde(rename = "user")]
    User,
    /// Project settings (`.claude/settings.json`).
    #[serde(rename = "project")]
    Project,
    /// Project-local settings (`.claude/settings.local.json`).
    #[serde(rename = "local")]
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

// ---------------------------------------------------------------------------
// SDK beta features
// ---------------------------------------------------------------------------

/// SDK beta feature flags.
///
/// See <https://docs.anthropic.com/en/api/beta-headers>.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SdkBeta {
    /// Enable the 1M token context window (Sonnet 4/4.5 only).
    #[serde(rename = "context-1m-2025-08-07")]
    ContextOneM,
}

impl SdkBeta {
    /// Return the beta header wire value.
    #[must_use]
    pub const fn as_wire(self) -> &'static str {
        match self {
            Self::ContextOneM => "context-1m-2025-08-07",
        }
    }
}

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

/// System prompt configuration.
///
/// Mirrors the official `str | SystemPromptPreset | SystemPromptFile` union.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemPrompt {
    /// A custom system prompt string.
    Text(String),
    /// Claude Code's default (`claude_code`) preset prompt.
    Preset {
        /// Instructions appended to the preset prompt.
        append: Option<String>,
        /// Strip per-user dynamic sections so the prompt stays cacheable.
        ///
        /// Sent in the initialize control request, not as a CLI flag.
        exclude_dynamic_sections: Option<bool>,
    },
    /// Load the system prompt from a file path.
    File {
        /// Path to the system-prompt file.
        path: String,
    },
}

impl SystemPrompt {
    /// Convenience constructor for a plain string prompt.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Convenience constructor for the `claude_code` preset.
    #[must_use]
    pub const fn preset() -> Self {
        Self::Preset {
            append: None,
            exclude_dynamic_sections: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tools preset
// ---------------------------------------------------------------------------

/// Base set of built-in tools exposed to the model.
///
/// Mirrors the official `list[str] | ToolsPreset` union. `None` on
/// [`AgentOptions`](crate::agent::AgentOptions) omits the flag and uses CLI
/// defaults; `Some(List(vec![]))` disables every built-in tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolsSpec {
    /// An explicit list of tool names. Empty disables all built-in tools.
    List(Vec<String>),
    /// The `claude_code` preset (all default Claude Code tools).
    Preset,
}

// ---------------------------------------------------------------------------
// Task budget, thinking, output format
// ---------------------------------------------------------------------------

/// API-side task budget in tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskBudget {
    /// Total token budget made known to the model.
    pub total: i64,
}

/// Controls whether thinking text is returned summarized or omitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingDisplay {
    /// Return summarized thinking text.
    Summarized,
    /// Return signature-only thinking (no text).
    Omitted,
}

impl ThinkingDisplay {
    /// Return the CLI wire value.
    #[must_use]
    pub const fn as_wire(self) -> &'static str {
        match self {
            Self::Summarized => "summarized",
            Self::Omitted => "omitted",
        }
    }
}

/// Controls Claude's thinking/reasoning behavior for an agent session.
///
/// Distinct from [`crate::types::ThinkingConfig`] (the Messages API type): this
/// is the agent-facing configuration with a `display` field and the
/// `adaptive` variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ThinkingConfig {
    /// Claude decides when and how much to think (Opus 4.6+).
    Adaptive {
        /// Optional thinking display mode.
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<ThinkingDisplay>,
    },
    /// Fixed thinking token budget (older models).
    Enabled {
        /// Thinking token budget.
        budget_tokens: i64,
        /// Optional thinking display mode.
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<ThinkingDisplay>,
    },
    /// No extended thinking.
    Disabled,
}

/// Output format configuration for structured responses.
///
/// Mirrors the Messages API shape, e.g.
/// `{"type": "json_schema", "schema": {...}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputFormat {
    /// Structured output validated against a JSON Schema.
    JsonSchema {
        /// The JSON Schema the output must match.
        schema: Value,
    },
}

impl OutputFormat {
    /// Return the JSON Schema when this is a `json_schema` output format.
    #[must_use]
    pub const fn schema(&self) -> &Value {
        match self {
            Self::JsonSchema { schema } => schema,
        }
    }
}

// ---------------------------------------------------------------------------
// Skills
// ---------------------------------------------------------------------------

/// Skills enabled for the main session.
///
/// Mirrors the official `list[str] | Literal["all"] | None` union. `None` on
/// [`AgentOptions`](crate::agent::AgentOptions) means no SDK auto-configuration;
/// [`SkillSelection::List`] with an empty vector suppresses every skill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillSelection {
    /// Enable every discovered skill.
    All,
    /// Enable only the listed skills (`name` or `plugin:skill`).
    List(Vec<String>),
}

impl SkillSelection {
    /// Whether this selection enables all discovered skills.
    #[must_use]
    pub const fn is_all(&self) -> bool {
        matches!(self, Self::All)
    }

    /// Return the explicit skill names when this is a list.
    #[must_use]
    pub fn names(&self) -> Option<&[String]> {
        match self {
            Self::All => None,
            Self::List(names) => Some(names),
        }
    }

    /// Validate every listed skill name against CLI hazards.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::InvalidConfig`] when a name is empty, has
    /// surrounding whitespace, or contains a wildcard, control, or path
    /// separator character.
    pub fn validate(&self) -> Result<(), ClaudeError> {
        let Self::List(names) = self else {
            return Ok(());
        };
        for name in names {
            validate_skill_name(name)?;
        }
        Ok(())
    }
}

/// Validate one skill name against tokenizer/control/wildcard/slash hazards.
///
/// # Errors
///
/// Returns [`ClaudeError::InvalidConfig`] when the name is unsafe.
pub fn validate_skill_name(name: &str) -> Result<(), ClaudeError> {
    if name.is_empty() {
        return Err(ClaudeError::InvalidConfig(
            "skill name must not be empty".into(),
        ));
    }
    if name != name.trim() {
        return Err(ClaudeError::InvalidConfig(format!(
            "skill name must not have surrounding whitespace: {name:?}"
        )));
    }
    for ch in name.chars() {
        if ch.is_control()
            || ch.is_whitespace()
            || matches!(ch, '*' | '?' | '/' | '\\' | '(' | ')' | ',')
        {
            return Err(ClaudeError::InvalidConfig(format!(
                "invalid character in skill name: {name:?}"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Plugins
// ---------------------------------------------------------------------------

/// SDK plugin configuration. Only local plugins are currently supported.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdkPluginConfig {
    /// Plugin kind. Always `"local"`.
    #[serde(rename = "type")]
    pub kind: SdkPluginKind,
    /// Absolute path to the local plugin directory.
    pub path: String,
}

/// Supported plugin kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SdkPluginKind {
    /// A local plugin loaded from a filesystem path.
    #[serde(rename = "local")]
    Local,
}

impl SdkPluginConfig {
    /// Construct a local plugin configuration.
    #[must_use]
    pub fn local(path: impl Into<String>) -> Self {
        Self {
            kind: SdkPluginKind::Local,
            path: path.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Agent definitions
// ---------------------------------------------------------------------------

/// Memory scope for an agent definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentMemory {
    /// User-scoped memory.
    #[serde(rename = "user")]
    User,
    /// Project-scoped memory.
    #[serde(rename = "project")]
    Project,
    /// Local-scoped memory.
    #[serde(rename = "local")]
    Local,
}

/// Effort for an agent definition: an [`EffortLevel`] or a raw token count.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AgentEffort {
    /// A named effort level.
    Level(EffortLevel),
    /// A raw token budget.
    Tokens(i64),
}

/// Programmatic subagent definition sent in the initialize control request.
///
/// Field names use the official camelCase wire spelling; every optional field
/// is omitted when `None`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentDefinition {
    /// Human-readable description shown to the model.
    pub description: String,
    /// System prompt for the subagent.
    pub prompt: String,
    /// Built-in tools available to the subagent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    /// Tools removed from the subagent's context.
    #[serde(
        rename = "disallowedTools",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub disallowed_tools: Option<Vec<String>>,
    /// Model alias (`sonnet`, `opus`, `haiku`, `inherit`) or full model ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Skills enabled for the subagent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<String>>,
    /// Memory scope for the subagent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<AgentMemory>,
    /// MCP servers: each entry is a server name or an inline `{name: config}`.
    #[serde(
        rename = "mcpServers",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub mcp_servers: Option<Vec<Value>>,
    /// Prompt injected as the subagent's first turn.
    #[serde(
        rename = "initialPrompt",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub initial_prompt: Option<String>,
    /// Maximum turns for the subagent.
    #[serde(rename = "maxTurns", default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    /// Whether the subagent runs in the background.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    /// Effort applied to the subagent's output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<AgentEffort>,
    /// Permission mode for the subagent.
    #[serde(
        rename = "permissionMode",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub permission_mode: Option<PermissionMode>,
}

impl AgentDefinition {
    /// Construct a minimal definition with only description and prompt.
    #[must_use]
    pub fn new(description: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            prompt: prompt.into(),
            tools: None,
            disallowed_tools: None,
            model: None,
            skills: None,
            memory: None,
            mcp_servers: None,
            initial_prompt: None,
            max_turns: None,
            background: None,
            effort: None,
            permission_mode: None,
        }
    }

    /// Serialize this definition for the `initialize.agents` map.
    ///
    /// Equivalent to Python's `asdict` with `None` fields omitted.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::MessageParse`] if serialization fails.
    pub fn to_initialize_value(&self) -> Result<Value, ClaudeError> {
        serde_json::to_value(self).map_err(|source| ClaudeError::MessageParse {
            message: format!("failed to serialize agent definition: {source}"),
            data: None,
        })
    }
}

// ---------------------------------------------------------------------------
// MCP server configuration
// ---------------------------------------------------------------------------

/// MCP stdio server configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpStdioServerConfig {
    /// Command to spawn.
    pub command: String,
    /// Command arguments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<String>>,
    /// Environment overrides for the spawned process.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<BTreeMap<String, String>>,
}

/// MCP SSE server configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpSseServerConfig {
    /// Server URL.
    pub url: String,
    /// Optional request headers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
}

/// MCP HTTP server configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpHttpServerConfig {
    /// Server URL.
    pub url: String,
    /// Optional request headers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
}

/// In-process SDK MCP server reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpSdkServerConfig {
    /// Server name used to route tool calls.
    pub name: String,
}

/// MCP server configuration union.
///
/// Serializes with an internal `type` discriminator matching the CLI wire
/// format. The stdio variant is emitted with an explicit `"stdio"` type. On
/// deserialization an object with no `type` is treated as stdio, matching the
/// official Python `McpStdioServerConfig` where `type` is `NotRequired` for
/// backwards compatibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum McpServerConfig {
    /// Stdio transport.
    Stdio(McpStdioServerConfig),
    /// Server-sent events transport.
    Sse(McpSseServerConfig),
    /// HTTP transport.
    Http(McpHttpServerConfig),
    /// In-process SDK server.
    Sdk(McpSdkServerConfig),
}

impl<'de> Deserialize<'de> for McpServerConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Read into a generic value so a missing `type` can default to stdio,
        // matching Python's optional stdio discriminator.
        let value = Value::deserialize(deserializer)?;
        let Some(object) = value.as_object() else {
            return Err(serde::de::Error::custom(
                "MCP server configuration must be a JSON object",
            ));
        };
        let kind = object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("stdio");
        match kind {
            "stdio" => serde_json::from_value(value)
                .map(Self::Stdio)
                .map_err(serde::de::Error::custom),
            "sse" => serde_json::from_value(value)
                .map(Self::Sse)
                .map_err(serde::de::Error::custom),
            "http" => serde_json::from_value(value)
                .map(Self::Http)
                .map_err(serde::de::Error::custom),
            "sdk" => serde_json::from_value(value)
                .map(Self::Sdk)
                .map_err(serde::de::Error::custom),
            other => Err(serde::de::Error::custom(format!(
                "unknown MCP server type: {other}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// MCP server status (returned by get_mcp_status)
// ---------------------------------------------------------------------------

/// SDK MCP server config as returned in status responses (serializable only).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpSdkServerConfigStatus {
    /// Server name.
    pub name: String,
}

/// Claude.ai proxy MCP server config (output-only).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpClaudeAiProxyServerConfig {
    /// Proxy URL.
    pub url: String,
    /// Proxy identifier.
    pub id: String,
}

/// Broader MCP config union for status responses (includes `claudeai-proxy`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum McpServerStatusConfig {
    /// Stdio transport.
    Stdio(McpStdioServerConfig),
    /// Server-sent events transport.
    Sse(McpSseServerConfig),
    /// HTTP transport.
    Http(McpHttpServerConfig),
    /// In-process SDK server (serializable form).
    Sdk(McpSdkServerConfigStatus),
    /// Claude.ai proxy server.
    #[serde(rename = "claudeai-proxy")]
    ClaudeAiProxy(McpClaudeAiProxyServerConfig),
}

/// Tool annotations as returned in MCP server status (wire camelCase).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpToolAnnotations {
    /// Tool only reads state.
    #[serde(rename = "readOnly", default, skip_serializing_if = "Option::is_none")]
    pub read_only: Option<bool>,
    /// Tool performs destructive actions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub destructive: Option<bool>,
    /// Tool interacts with an open-ended external world.
    #[serde(rename = "openWorld", default, skip_serializing_if = "Option::is_none")]
    pub open_world: Option<bool>,
}

/// Information about a tool provided by an MCP server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpToolInfo {
    /// Tool name.
    pub name: String,
    /// Optional tool description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Optional tool annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<McpToolAnnotations>,
}

/// Server info from the MCP initialize handshake.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerInfo {
    /// Server name.
    pub name: String,
    /// Server version.
    pub version: String,
}

/// Connection status for an MCP server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum McpServerConnectionStatus {
    /// Connected and ready.
    Connected,
    /// Connection failed.
    Failed,
    /// Requires authentication.
    NeedsAuth,
    /// Connection is pending.
    Pending,
    /// Server is disabled.
    Disabled,
}

/// Status information for an MCP server connection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerStatus {
    /// Server name as configured.
    pub name: String,
    /// Current connection status.
    pub status: McpServerConnectionStatus,
    /// Server info from the MCP handshake (when connected).
    #[serde(
        rename = "serverInfo",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub server_info: Option<McpServerInfo>,
    /// Error message (when `status` is `failed`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Server configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<McpServerStatusConfig>,
    /// Configuration scope (e.g. `project`, `user`, `local`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Tools provided by this server (when connected).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<McpToolInfo>>,
}

/// Response from `get_mcp_status()`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpStatusResponse {
    /// Per-server status entries.
    #[serde(rename = "mcpServers")]
    pub mcp_servers: Vec<McpServerStatus>,
}

// ---------------------------------------------------------------------------
// Context usage (returned by get_context_usage)
// ---------------------------------------------------------------------------

/// A single context-usage category (system prompt, tools, messages, etc.).
///
/// Mirrors the official `ContextUsageCategory`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextUsageCategory {
    /// Category display name.
    pub name: String,
    /// Tokens attributed to this category.
    pub tokens: u64,
    /// Display color for the category.
    pub color: String,
    /// Whether this category is deferred.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_deferred: Option<bool>,
}

/// Typed response from `get_context_usage()`.
///
/// Mirrors the official `ContextUsageResponse`. Fields not modeled here remain
/// available on the raw [`Value`] returned by
/// [`crate::agent::ClaudeAgentClient::get_context_usage`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextUsageResponse {
    /// Token usage broken down by category.
    pub categories: Vec<ContextUsageCategory>,
    /// Total tokens currently in the context window.
    pub total_tokens: u64,
    /// Effective maximum tokens (may be reduced by the autocompact buffer).
    pub max_tokens: u64,
    /// Raw model context-window size.
    pub raw_max_tokens: u64,
    /// Percentage of the context window used (0-100).
    pub percentage: f64,
    /// Model the usage is calculated for.
    pub model: String,
    /// Whether autocompact is enabled for this session.
    pub is_auto_compact_enabled: bool,
    /// Loaded memory files (path, type, token counts).
    #[serde(default)]
    pub memory_files: Vec<Value>,
    /// MCP tools (name, serverName, tokens, isLoaded).
    #[serde(default)]
    pub mcp_tools: Vec<Value>,
}

/// Usage statistics reported in task progress/notification messages.
///
/// Mirrors the official `TaskUsage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskUsage {
    /// Total tokens used by the task.
    pub total_tokens: u64,
    /// Number of tool invocations.
    pub tool_uses: u64,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// Per-model token usage and cost breakdown (the `modelUsage` field).
///
/// Mirrors the official `ModelUsage`; camelCase keys are passed through
/// verbatim from the CLI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelUsage {
    /// Input tokens.
    pub input_tokens: u64,
    /// Output tokens.
    pub output_tokens: u64,
    /// Cache-read input tokens.
    pub cache_read_input_tokens: u64,
    /// Cache-creation input tokens.
    pub cache_creation_input_tokens: u64,
    /// Web-search requests.
    pub web_search_requests: u64,
    /// Cost in USD.
    #[serde(rename = "costUSD")]
    pub cost_usd: f64,
    /// Model context-window size.
    pub context_window: u64,
    /// Maximum output tokens.
    pub max_output_tokens: u64,
    /// Canonical model id used for pricing lookup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical_model: Option<String>,
    /// API provider that served this model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

// ---------------------------------------------------------------------------
// Sandbox settings
// ---------------------------------------------------------------------------

/// Network configuration for the sandbox.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SandboxNetworkConfig {
    /// Domains sandboxed processes can access.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
    /// Domains always blocked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub denied_domains: Option<Vec<String>>,
    /// When true, only managed-settings allowed domains are respected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_managed_domains_only: Option<bool>,
    /// Unix socket paths accessible in the sandbox.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_unix_sockets: Option<Vec<String>>,
    /// Allow all Unix sockets (less secure).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_all_unix_sockets: Option<bool>,
    /// Allow binding to localhost ports (macOS only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_local_binding: Option<bool>,
    /// macOS XPC/Mach service names to allow.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_mach_lookup: Option<Vec<String>>,
    /// HTTP proxy port for a bring-your-own proxy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_proxy_port: Option<i64>,
    /// SOCKS5 proxy port for a bring-your-own proxy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub socks_proxy_port: Option<i64>,
}

/// Violations to ignore in the sandbox.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SandboxIgnoreViolations {
    /// File paths for which violations are ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<Vec<String>>,
    /// Network hosts for which violations are ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network: Option<Vec<String>>,
}

/// Sandbox settings controlling filesystem/network isolation of bash commands.
///
/// Merged under the `sandbox` key of the CLI settings JSON.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SandboxSettings {
    /// Enable bash sandboxing (macOS/Linux only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Auto-approve bash commands when sandboxed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_allow_bash_if_sandboxed: Option<bool>,
    /// Commands that run outside the sandbox.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub excluded_commands: Option<Vec<String>>,
    /// Allow commands to bypass the sandbox via `dangerouslyDisableSandbox`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_unsandboxed_commands: Option<bool>,
    /// Network configuration for the sandbox.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network: Option<SandboxNetworkConfig>,
    /// Violations to ignore.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore_violations: Option<SandboxIgnoreViolations>,
    /// Enable a weaker nested sandbox (unprivileged Docker; reduces security).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_weaker_nested_sandbox: Option<bool>,
}

impl SandboxSettings {
    /// Serialize to the camelCase JSON value merged under `settings["sandbox"]`.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::MessageParse`] if serialization fails.
    pub fn to_json_value(&self) -> Result<Value, ClaudeError> {
        serde_json::to_value(self).map_err(|source| ClaudeError::MessageParse {
            message: format!("failed to serialize sandbox settings: {source}"),
            data: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Permission updates
// ---------------------------------------------------------------------------

/// Where a permission update is persisted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionUpdateDestination {
    /// User settings.
    #[serde(rename = "userSettings")]
    UserSettings,
    /// Project settings.
    #[serde(rename = "projectSettings")]
    ProjectSettings,
    /// Local settings.
    #[serde(rename = "localSettings")]
    LocalSettings,
    /// Session-only (not persisted).
    #[serde(rename = "session")]
    Session,
}

/// Permission behavior applied by a rule update.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionBehavior {
    /// Allow the tool.
    Allow,
    /// Deny the tool.
    Deny,
    /// Prompt for the tool.
    Ask,
}

/// A single permission rule value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionRuleValue {
    /// Tool name the rule applies to.
    pub tool_name: String,
    /// Optional rule content (e.g. a path or command pattern).
    pub rule_content: Option<String>,
}

/// The variant type of a [`PermissionUpdate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionUpdateType {
    /// Add permission rules.
    #[serde(rename = "addRules")]
    AddRules,
    /// Replace permission rules.
    #[serde(rename = "replaceRules")]
    ReplaceRules,
    /// Remove permission rules.
    #[serde(rename = "removeRules")]
    RemoveRules,
    /// Set the permission mode.
    #[serde(rename = "setMode")]
    SetMode,
    /// Add allowed directories.
    #[serde(rename = "addDirectories")]
    AddDirectories,
    /// Remove allowed directories.
    #[serde(rename = "removeDirectories")]
    RemoveDirectories,
}

impl PermissionUpdateType {
    fn as_wire(self) -> &'static str {
        match self {
            Self::AddRules => "addRules",
            Self::ReplaceRules => "replaceRules",
            Self::RemoveRules => "removeRules",
            Self::SetMode => "setMode",
            Self::AddDirectories => "addDirectories",
            Self::RemoveDirectories => "removeDirectories",
        }
    }

    fn from_wire(value: &str) -> Result<Self, ClaudeError> {
        Ok(match value {
            "addRules" => Self::AddRules,
            "replaceRules" => Self::ReplaceRules,
            "removeRules" => Self::RemoveRules,
            "setMode" => Self::SetMode,
            "addDirectories" => Self::AddDirectories,
            "removeDirectories" => Self::RemoveDirectories,
            other => {
                return Err(ClaudeError::MessageParse {
                    message: format!("unknown permission update type: {other}"),
                    data: None,
                });
            }
        })
    }
}

/// A permission update suggested by the CLI or returned by a callback.
///
/// Mirrors the official `PermissionUpdate` dataclass and its
/// `to_dict`/`from_dict` control-protocol wire form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionUpdate {
    /// The update variant.
    pub update_type: PermissionUpdateType,
    /// Rules for rule-based variants.
    pub rules: Option<Vec<PermissionRuleValue>>,
    /// Behavior for rule-based variants.
    pub behavior: Option<PermissionBehavior>,
    /// Mode for the `setMode` variant.
    pub mode: Option<PermissionMode>,
    /// Directories for directory variants.
    pub directories: Option<Vec<String>>,
    /// Persistence destination for all variants.
    pub destination: Option<PermissionUpdateDestination>,
}

impl PermissionUpdate {
    /// Serialize to the control-protocol dict, matching Python's `to_dict`.
    #[must_use]
    pub fn to_wire(&self) -> Value {
        let mut result = Map::new();
        result.insert(
            "type".to_string(),
            Value::String(self.update_type.as_wire().to_string()),
        );
        if let Some(destination) = self.destination {
            result.insert(
                "destination".to_string(),
                serde_json::to_value(destination).unwrap_or(Value::Null),
            );
        }
        match self.update_type {
            PermissionUpdateType::AddRules
            | PermissionUpdateType::ReplaceRules
            | PermissionUpdateType::RemoveRules => {
                if let Some(rules) = &self.rules {
                    let rules_wire: Vec<Value> = rules
                        .iter()
                        .map(|rule| {
                            json!({
                                "toolName": rule.tool_name,
                                "ruleContent": rule.rule_content,
                            })
                        })
                        .collect();
                    result.insert("rules".to_string(), Value::Array(rules_wire));
                }
                if let Some(behavior) = self.behavior {
                    result.insert(
                        "behavior".to_string(),
                        serde_json::to_value(behavior).unwrap_or(Value::Null),
                    );
                }
            }
            PermissionUpdateType::SetMode => {
                if let Some(mode) = self.mode {
                    result.insert(
                        "mode".to_string(),
                        serde_json::to_value(mode).unwrap_or(Value::Null),
                    );
                }
            }
            PermissionUpdateType::AddDirectories | PermissionUpdateType::RemoveDirectories => {
                if let Some(directories) = &self.directories {
                    result.insert(
                        "directories".to_string(),
                        Value::Array(
                            directories
                                .iter()
                                .map(|dir| Value::String(dir.clone()))
                                .collect(),
                        ),
                    );
                }
            }
        }
        Value::Object(result)
    }

    /// Parse from the control-protocol dict, matching Python's `from_dict`.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError::MessageParse`] when `type` is missing or unknown,
    /// or when a nested rule is malformed.
    pub fn from_wire(value: &Value) -> Result<Self, ClaudeError> {
        let update_type_str =
            value
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| ClaudeError::MessageParse {
                    message: "permission update missing type".into(),
                    data: Some(value.clone()),
                })?;
        let update_type = PermissionUpdateType::from_wire(update_type_str)?;

        let rules = match value.get("rules") {
            Some(Value::Array(items)) => {
                let mut parsed = Vec::with_capacity(items.len());
                for item in items {
                    let tool_name = item
                        .get("toolName")
                        .and_then(Value::as_str)
                        .ok_or_else(|| ClaudeError::MessageParse {
                            message: "permission rule missing toolName".into(),
                            data: Some(item.clone()),
                        })?
                        .to_string();
                    let rule_content = item
                        .get("ruleContent")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    parsed.push(PermissionRuleValue {
                        tool_name,
                        rule_content,
                    });
                }
                Some(parsed)
            }
            _ => None,
        };

        let behavior = match value.get("behavior") {
            Some(behavior) if !behavior.is_null() => {
                Some(serde_json::from_value(behavior.clone()).map_err(|source| {
                    ClaudeError::MessageParse {
                        message: format!("invalid permission behavior: {source}"),
                        data: Some(value.clone()),
                    }
                })?)
            }
            _ => None,
        };

        let mode = match value.get("mode") {
            Some(mode) if !mode.is_null() => {
                Some(serde_json::from_value(mode.clone()).map_err(|source| {
                    ClaudeError::MessageParse {
                        message: format!("invalid permission mode: {source}"),
                        data: Some(value.clone()),
                    }
                })?)
            }
            _ => None,
        };

        let directories = value
            .get("directories")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|dir| dir.as_str().map(str::to_string))
                    .collect()
            });

        let destination = match value.get("destination") {
            Some(destination) if !destination.is_null() => Some(
                serde_json::from_value(destination.clone()).map_err(|source| {
                    ClaudeError::MessageParse {
                        message: format!("invalid permission destination: {source}"),
                        data: Some(value.clone()),
                    }
                })?,
            ),
            _ => None,
        };

        Ok(Self {
            update_type,
            rules,
            behavior,
            mode,
            directories,
            destination,
        })
    }
}

// ---------------------------------------------------------------------------
// Tool permission callback
// ---------------------------------------------------------------------------

/// Context passed to a [`ToolPermissionCallback`].
///
/// Mirrors the official `ToolPermissionContext`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolPermissionContext {
    /// Reserved for future abort-signal support.
    pub signal: Option<Value>,
    /// Permission update suggestions from the CLI.
    pub suggestions: Vec<PermissionUpdate>,
    /// Identifier of this tool call within the assistant message.
    pub tool_use_id: Option<String>,
    /// Sub-agent identifier when running inside a sub-agent.
    pub agent_id: Option<String>,
    /// Path that triggered the permission request, if applicable.
    pub blocked_path: Option<String>,
    /// Reason this permission request was triggered.
    pub decision_reason: Option<String>,
    /// Full permission prompt sentence.
    pub title: Option<String>,
    /// Short noun phrase for the tool action.
    pub display_name: Option<String>,
    /// Human-readable subtitle for the permission UI.
    pub description: Option<String>,
}

impl ToolPermissionContext {
    /// Parse the context fields from a `can_use_tool` control request.
    ///
    /// Unknown or absent fields are left at their defaults; malformed
    /// suggestion entries are skipped so a single bad suggestion cannot fail
    /// the whole callback.
    #[must_use]
    pub fn from_request(request: &Value) -> Self {
        let suggestions = request
            .get("permission_suggestions")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| PermissionUpdate::from_wire(item).ok())
                    .collect()
            })
            .unwrap_or_default();
        Self {
            signal: None,
            suggestions,
            tool_use_id: string_field(request, "tool_use_id"),
            agent_id: string_field(request, "agent_id"),
            blocked_path: string_field(request, "blocked_path"),
            decision_reason: string_field(request, "decision_reason"),
            title: string_field(request, "title"),
            display_name: string_field(request, "display_name"),
            description: string_field(request, "description"),
        }
    }
}

/// The result of a tool permission decision.
///
/// Mirrors `PermissionResultAllow | PermissionResultDeny`.
#[derive(Debug, Clone, PartialEq)]
pub enum PermissionResult {
    /// Permit execution, optionally replacing input and persisting updates.
    Allow {
        /// Replacement tool input; the original is used when `None`.
        updated_input: Option<Value>,
        /// Permission updates to persist.
        updated_permissions: Option<Vec<PermissionUpdate>>,
    },
    /// Deny execution.
    Deny {
        /// Message explaining the denial.
        message: String,
        /// Whether to interrupt the current agent turn.
        interrupt: bool,
    },
}

impl PermissionResult {
    /// Construct a plain allow result.
    #[must_use]
    pub const fn allow() -> Self {
        Self::Allow {
            updated_input: None,
            updated_permissions: None,
        }
    }

    /// Construct a plain deny result.
    #[must_use]
    pub fn deny(message: impl Into<String>) -> Self {
        Self::Deny {
            message: message.into(),
            interrupt: false,
        }
    }

    /// Serialize to the control-response `response` payload.
    ///
    /// Allow defaults `updatedInput` to `original_input` when the callback did
    /// not replace it. Deny only emits `interrupt` when it is `true`, matching
    /// the official wire form.
    #[must_use]
    pub fn to_wire(&self, original_input: &Value) -> Value {
        match self {
            Self::Allow {
                updated_input,
                updated_permissions,
            } => {
                let mut result = Map::new();
                result.insert("behavior".to_string(), Value::String("allow".to_string()));
                result.insert(
                    "updatedInput".to_string(),
                    updated_input
                        .clone()
                        .unwrap_or_else(|| original_input.clone()),
                );
                if let Some(updates) = updated_permissions {
                    result.insert(
                        "updatedPermissions".to_string(),
                        Value::Array(updates.iter().map(PermissionUpdate::to_wire).collect()),
                    );
                }
                Value::Object(result)
            }
            Self::Deny { message, interrupt } => {
                let mut result = Map::new();
                result.insert("behavior".to_string(), Value::String("deny".to_string()));
                result.insert("message".to_string(), Value::String(message.clone()));
                if *interrupt {
                    result.insert("interrupt".to_string(), Value::Bool(true));
                }
                Value::Object(result)
            }
        }
    }
}

/// Callback invoked when a tool call would otherwise prompt the user.
///
/// Mirrors the official `CanUseTool` callable.
#[async_trait]
pub trait ToolPermissionCallback: fmt::Debug + Send + Sync {
    /// Decide whether `tool_name` may run with `input`.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError`] when the callback cannot produce a decision.
    async fn can_use_tool(
        &self,
        tool_name: &str,
        input: &Value,
        context: &ToolPermissionContext,
    ) -> Result<PermissionResult, ClaudeError>;
}

/// A shareable tool permission callback.
pub type CanUseTool = Arc<dyn ToolPermissionCallback>;

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

/// Lifecycle events a hook can subscribe to.
///
/// The wire spelling is exact `PascalCase`. `Ord` allows use as a `BTreeMap` key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HookEvent {
    /// Before a tool runs.
    PreToolUse,
    /// After a tool runs.
    PostToolUse,
    /// After a tool run fails.
    PostToolUseFailure,
    /// When the user submits a prompt.
    UserPromptSubmit,
    /// When the main agent stops.
    Stop,
    /// When a sub-agent stops.
    SubagentStop,
    /// Before conversation compaction.
    PreCompact,
    /// When a notification is emitted.
    Notification,
    /// When a sub-agent starts.
    SubagentStart,
    /// When a permission request is raised.
    PermissionRequest,
}

impl HookEvent {
    /// Return the exact `PascalCase` wire value.
    #[must_use]
    pub const fn as_wire(self) -> &'static str {
        match self {
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
            Self::PostToolUseFailure => "PostToolUseFailure",
            Self::UserPromptSubmit => "UserPromptSubmit",
            Self::Stop => "Stop",
            Self::SubagentStop => "SubagentStop",
            Self::PreCompact => "PreCompact",
            Self::Notification => "Notification",
            Self::SubagentStart => "SubagentStart",
            Self::PermissionRequest => "PermissionRequest",
        }
    }
}

/// Base fields present across many hook events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseHookInput {
    /// Session identifier.
    pub session_id: String,
    /// Path to the transcript file.
    pub transcript_path: String,
    /// Working directory.
    pub cwd: String,
    /// Current permission mode, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permission_mode: Option<String>,
}

/// Input for `PreToolUse` events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreToolUseHookInput {
    /// Tool name.
    pub tool_name: String,
    /// Tool input payload.
    pub tool_input: Value,
    /// Identifier of this tool call.
    pub tool_use_id: String,
    /// Sub-agent identifier, when firing inside a sub-agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Agent type name, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_type: Option<String>,
}

/// Input for `PostToolUse` events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostToolUseHookInput {
    /// Tool name.
    pub tool_name: String,
    /// Tool input payload.
    pub tool_input: Value,
    /// Tool response payload.
    pub tool_response: Value,
    /// Identifier of this tool call.
    pub tool_use_id: String,
    /// Sub-agent identifier, when firing inside a sub-agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Agent type name, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_type: Option<String>,
}

/// Input for `PostToolUseFailure` events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostToolUseFailureHookInput {
    /// Tool name.
    pub tool_name: String,
    /// Tool input payload.
    pub tool_input: Value,
    /// Identifier of this tool call.
    pub tool_use_id: String,
    /// Error message.
    pub error: String,
    /// Whether the failure was an interrupt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_interrupt: Option<bool>,
    /// Sub-agent identifier, when firing inside a sub-agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Agent type name, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_type: Option<String>,
}

/// Input for `UserPromptSubmit` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserPromptSubmitHookInput {
    /// The submitted prompt.
    pub prompt: String,
}

/// Input for `Stop` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StopHookInput {
    /// Whether the stop hook is already active.
    pub stop_hook_active: bool,
}

/// Input for `SubagentStop` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentStopHookInput {
    /// Whether the stop hook is already active.
    pub stop_hook_active: bool,
    /// Sub-agent identifier.
    pub agent_id: String,
    /// Path to the sub-agent transcript.
    pub agent_transcript_path: String,
    /// Agent type name.
    pub agent_type: String,
}

/// The trigger for a `PreCompact` event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PreCompactTrigger {
    /// Compaction triggered manually.
    Manual,
    /// Compaction triggered automatically.
    Auto,
}

/// Input for `PreCompact` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreCompactHookInput {
    /// What triggered the compaction.
    pub trigger: PreCompactTrigger,
    /// Custom compaction instructions, if any.
    pub custom_instructions: Option<String>,
}

/// Input for `Notification` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotificationHookInput {
    /// Notification message.
    pub message: String,
    /// Notification title, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Notification type.
    pub notification_type: String,
}

/// Input for `SubagentStart` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentStartHookInput {
    /// Sub-agent identifier.
    pub agent_id: String,
    /// Agent type name.
    pub agent_type: String,
}

/// Input for `PermissionRequest` events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PermissionRequestHookInput {
    /// Tool name.
    pub tool_name: String,
    /// Tool input payload.
    pub tool_input: Value,
    /// Permission suggestions from the CLI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permission_suggestions: Option<Vec<Value>>,
    /// Sub-agent identifier, when firing inside a sub-agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Agent type name, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_type: Option<String>,
}

/// Event-specific typed hook input, plus base fields and the raw frame.
///
/// Callbacks may inspect the typed [`HookInput::event`] payload or fall back to
/// [`HookInput::raw`] for forward compatibility.
#[derive(Debug, Clone, PartialEq)]
pub struct HookInput {
    /// Base fields shared across events, when present and well-formed.
    ///
    /// The official control protocol may deliver a minimal input object with
    /// none of the base fields (see upstream
    /// `tests/test_tool_callbacks.py::test_hook_callback`), so this is best
    /// effort and never blocks callback dispatch.
    pub base: Option<BaseHookInput>,
    /// Event-specific typed payload.
    pub event: HookInputKind,
    raw: Value,
}

/// Event-specific typed hook input payload.
#[derive(Debug, Clone, PartialEq)]
pub enum HookInputKind {
    /// `PreToolUse` input.
    PreToolUse(PreToolUseHookInput),
    /// `PostToolUse` input.
    PostToolUse(PostToolUseHookInput),
    /// `PostToolUseFailure` input.
    PostToolUseFailure(PostToolUseFailureHookInput),
    /// `UserPromptSubmit` input.
    UserPromptSubmit(UserPromptSubmitHookInput),
    /// `Stop` input.
    Stop(StopHookInput),
    /// `SubagentStop` input.
    SubagentStop(SubagentStopHookInput),
    /// `PreCompact` input.
    PreCompact(PreCompactHookInput),
    /// `Notification` input.
    Notification(NotificationHookInput),
    /// `SubagentStart` input.
    SubagentStart(SubagentStartHookInput),
    /// `PermissionRequest` input.
    PermissionRequest(PermissionRequestHookInput),
    /// An unrecognized event name, or a recognized event whose typed payload
    /// could not be decoded; only the raw frame is available.
    Unknown(String),
}

impl HookInput {
    /// Parse a hook input frame, dispatching on `hook_event_name`.
    ///
    /// This is permissive and infallible: the official control protocol passes
    /// the raw `input` object straight to the callback without validating base
    /// fields (upstream `_internal/query.py::_handle_control_request`), so a
    /// minimal object such as `{"test": "data"}` still yields a usable
    /// [`HookInput`]. Base fields and the typed event payload are decoded on a
    /// best-effort basis; when either cannot be decoded the raw frame is always
    /// preserved via [`HookInput::raw`].
    #[must_use]
    pub fn from_value(value: Value) -> Self {
        let base: Option<BaseHookInput> = serde_json::from_value(value.clone()).ok();
        let event_name = value
            .get("hook_event_name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        // Decode the typed payload best-effort; a recognized event with a
        // malformed payload falls back to `Unknown` rather than failing the
        // callback, matching the permissive upstream dispatch.
        let event = match event_name.as_str() {
            "PreToolUse" => parse_hook_kind(&value).map(HookInputKind::PreToolUse),
            "PostToolUse" => parse_hook_kind(&value).map(HookInputKind::PostToolUse),
            "PostToolUseFailure" => parse_hook_kind(&value).map(HookInputKind::PostToolUseFailure),
            "UserPromptSubmit" => parse_hook_kind(&value).map(HookInputKind::UserPromptSubmit),
            "Stop" => parse_hook_kind(&value).map(HookInputKind::Stop),
            "SubagentStop" => parse_hook_kind(&value).map(HookInputKind::SubagentStop),
            "PreCompact" => parse_hook_kind(&value).map(HookInputKind::PreCompact),
            "Notification" => parse_hook_kind(&value).map(HookInputKind::Notification),
            "SubagentStart" => parse_hook_kind(&value).map(HookInputKind::SubagentStart),
            "PermissionRequest" => parse_hook_kind(&value).map(HookInputKind::PermissionRequest),
            other => Ok(HookInputKind::Unknown(other.to_string())),
        }
        .unwrap_or(HookInputKind::Unknown(event_name));

        Self {
            base,
            event,
            raw: value,
        }
    }

    /// Return the raw frame for forward compatibility.
    #[must_use]
    pub const fn raw(&self) -> &Value {
        &self.raw
    }
}

fn parse_hook_kind<T: for<'de> Deserialize<'de>>(value: &Value) -> Result<T, serde_json::Error> {
    serde_json::from_value(value.clone())
}

/// Hook-specific output for `PreToolUse` events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreToolUseHookSpecificOutput {
    /// Event discriminator (`"PreToolUse"`).
    #[serde(rename = "hookEventName")]
    pub hook_event_name: String,
    /// Permission decision (`allow`/`deny`/`ask`/`defer`).
    #[serde(
        rename = "permissionDecision",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub permission_decision: Option<String>,
    /// Reason for the permission decision.
    #[serde(
        rename = "permissionDecisionReason",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub permission_decision_reason: Option<String>,
    /// Replacement tool input.
    #[serde(
        rename = "updatedInput",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub updated_input: Option<Value>,
    /// Additional context injected for the model.
    #[serde(
        rename = "additionalContext",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_context: Option<String>,
}

/// Hook-specific output for `PostToolUse` events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostToolUseHookSpecificOutput {
    /// Event discriminator (`"PostToolUse"`).
    #[serde(rename = "hookEventName")]
    pub hook_event_name: String,
    /// Additional context injected for the model.
    #[serde(
        rename = "additionalContext",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_context: Option<String>,
    /// Replacement tool output (all tools).
    #[serde(
        rename = "updatedToolOutput",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub updated_tool_output: Option<Value>,
    /// Replacement tool output (MCP tools only).
    #[serde(
        rename = "updatedMCPToolOutput",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub updated_mcp_tool_output: Option<Value>,
}

/// Hook-specific output for `PostToolUseFailure` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostToolUseFailureHookSpecificOutput {
    /// Event discriminator (`"PostToolUseFailure"`).
    #[serde(rename = "hookEventName")]
    pub hook_event_name: String,
    /// Additional context injected for the model.
    #[serde(
        rename = "additionalContext",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_context: Option<String>,
}

/// Hook-specific output for `UserPromptSubmit` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserPromptSubmitHookSpecificOutput {
    /// Event discriminator (`"UserPromptSubmit"`).
    #[serde(rename = "hookEventName")]
    pub hook_event_name: String,
    /// Additional context injected for the model.
    #[serde(
        rename = "additionalContext",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_context: Option<String>,
}

/// Hook-specific output for `SessionStart` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionStartHookSpecificOutput {
    /// Event discriminator (`"SessionStart"`).
    #[serde(rename = "hookEventName")]
    pub hook_event_name: String,
    /// Additional context injected for the model.
    #[serde(
        rename = "additionalContext",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_context: Option<String>,
}

/// Hook-specific output for `Notification` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotificationHookSpecificOutput {
    /// Event discriminator (`"Notification"`).
    #[serde(rename = "hookEventName")]
    pub hook_event_name: String,
    /// Additional context injected for the model.
    #[serde(
        rename = "additionalContext",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_context: Option<String>,
}

/// Hook-specific output for `SubagentStart` events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentStartHookSpecificOutput {
    /// Event discriminator (`"SubagentStart"`).
    #[serde(rename = "hookEventName")]
    pub hook_event_name: String,
    /// Additional context injected for the model.
    #[serde(
        rename = "additionalContext",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_context: Option<String>,
}

/// Hook-specific output for `PermissionRequest` events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PermissionRequestHookSpecificOutput {
    /// Event discriminator (`"PermissionRequest"`).
    #[serde(rename = "hookEventName")]
    pub hook_event_name: String,
    /// The permission decision payload.
    pub decision: Value,
}

/// Synchronous hook output with control and decision fields.
///
/// The Rust field names avoid keyword conflicts (`continue_`); [`HookJSONOutput::to_wire`]
/// renames them to the CLI's `continue`/`async` spellings.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SyncHookJSONOutput {
    /// Whether Claude proceeds after the hook (default `true`).
    pub continue_: Option<bool>,
    /// Hide stdout from transcript mode.
    pub suppress_output: Option<bool>,
    /// Message shown when `continue_` is `false`.
    pub stop_reason: Option<String>,
    /// Set to `"block"` for blocking behavior.
    pub decision: Option<String>,
    /// Warning message shown to the user.
    pub system_message: Option<String>,
    /// Feedback message for Claude about the decision.
    pub reason: Option<String>,
    /// Event-specific controls.
    pub hook_specific_output: Option<Value>,
}

/// Hook callback output.
///
/// Mirrors `AsyncHookJSONOutput | SyncHookJSONOutput`.
#[derive(Debug, Clone, PartialEq)]
pub enum HookJSONOutput {
    /// Defer hook execution.
    Async {
        /// Optional async timeout in milliseconds.
        async_timeout: Option<i64>,
    },
    /// Synchronous control/decision output.
    Sync(SyncHookJSONOutput),
}

impl HookJSONOutput {
    /// Serialize to the control-response payload, renaming `async_`/`continue_`
    /// to the CLI's `async`/`continue`.
    #[must_use]
    pub fn to_wire(&self) -> Value {
        match self {
            Self::Async { async_timeout } => {
                let mut result = Map::new();
                result.insert("async".to_string(), Value::Bool(true));
                if let Some(timeout) = async_timeout {
                    result.insert("asyncTimeout".to_string(), Value::from(*timeout));
                }
                Value::Object(result)
            }
            Self::Sync(sync) => {
                let mut result = Map::new();
                if let Some(continue_) = sync.continue_ {
                    result.insert("continue".to_string(), Value::Bool(continue_));
                }
                if let Some(suppress) = sync.suppress_output {
                    result.insert("suppressOutput".to_string(), Value::Bool(suppress));
                }
                if let Some(stop_reason) = &sync.stop_reason {
                    result.insert("stopReason".to_string(), Value::String(stop_reason.clone()));
                }
                if let Some(decision) = &sync.decision {
                    result.insert("decision".to_string(), Value::String(decision.clone()));
                }
                if let Some(system_message) = &sync.system_message {
                    result.insert(
                        "systemMessage".to_string(),
                        Value::String(system_message.clone()),
                    );
                }
                if let Some(reason) = &sync.reason {
                    result.insert("reason".to_string(), Value::String(reason.clone()));
                }
                if let Some(hook_specific) = &sync.hook_specific_output {
                    result.insert("hookSpecificOutput".to_string(), hook_specific.clone());
                }
                Value::Object(result)
            }
        }
    }
}

/// Context passed to a [`HookHandler`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HookContext {
    /// Reserved for future abort-signal support.
    pub signal: Option<Value>,
}

/// Callback invoked for a matched hook event.
///
/// Mirrors the official `HookCallback` callable
/// `(HookInput, str | None, HookContext) -> Awaitable[HookJSONOutput]`.
#[async_trait]
pub trait HookHandler: fmt::Debug + Send + Sync {
    /// Handle one hook invocation.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeError`] when the hook cannot produce output.
    async fn call(
        &self,
        input: &HookInput,
        tool_use_id: Option<&str>,
        context: &HookContext,
    ) -> Result<HookJSONOutput, ClaudeError>;
}

/// A shareable hook callback.
pub type HookCallback = Arc<dyn HookHandler>;

/// Hook matcher configuration.
///
/// Mirrors the official `HookMatcher` dataclass. Serialized into
/// `initialize.hooks[event]` as `{matcher, hookCallbackIds, timeout}`.
#[derive(Clone)]
pub struct HookMatcher {
    /// Matcher string (e.g. a tool name or `"Write|Edit"`); `None` matches all.
    pub matcher: Option<String>,
    /// Callbacks invoked when the matcher fires.
    pub hooks: Vec<HookCallback>,
    /// Timeout in seconds for all hooks in this matcher.
    pub timeout: Option<f64>,
}

impl HookMatcher {
    /// Construct a matcher with the given matcher string and callbacks.
    #[must_use]
    pub fn new(matcher: Option<String>, hooks: Vec<HookCallback>) -> Self {
        Self {
            matcher,
            hooks,
            timeout: None,
        }
    }
}

impl fmt::Debug for HookMatcher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HookMatcher")
            .field("matcher", &self.matcher)
            .field("hooks", &self.hooks.len())
            .field("timeout", &self.timeout)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// SDK MCP helper constructors
// ---------------------------------------------------------------------------

/// Build an in-process MCP tool, equivalent to Python's `tool(...)` helper.
///
/// Wraps [`crate::agent::SdkMcpTool::new`] with the same argument order as the
/// Python decorator (`name`, `description`, `input_schema`, then handler).
/// Annotations are optional, matching the official `SdkMcpTool.annotations`
/// (`ToolAnnotations | None`); pass `None` to omit them from `tools/list`.
///
/// # Errors
///
/// Returns [`ClaudeError::InvalidConfig`] when the tool name is empty.
pub fn tool<F, Fut>(
    name: impl Into<String>,
    description: impl Into<String>,
    input_schema: Value,
    annotations: impl Into<Option<crate::agent::ToolAnnotations>>,
    handler: F,
) -> Result<crate::agent::SdkMcpTool, ClaudeError>
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<crate::agent::ToolCallResult, ClaudeError>> + Send + 'static,
{
    crate::agent::SdkMcpTool::new(name, description, input_schema, annotations, handler)
}

/// Create an in-process MCP server, equivalent to Python's
/// `create_sdk_mcp_server(name, version="1.0.0", tools=None)`.
///
/// Registers each tool on a fresh [`crate::agent::SdkMcpServer`]. `version`
/// defaults to `"1.0.0"` when `None`; `tools` defaults to empty when `None`,
/// matching the official defaults.
///
/// # Errors
///
/// Returns [`ClaudeError::InvalidConfig`] when the server name is empty or a
/// duplicate tool name is supplied.
pub fn create_sdk_mcp_server(
    name: impl Into<String>,
    version: impl Into<Option<String>>,
    tools: impl Into<Option<Vec<crate::agent::SdkMcpTool>>>,
) -> Result<crate::agent::SdkMcpServer, ClaudeError> {
    let version = version.into().unwrap_or_else(|| "1.0.0".to_string());
    let mut server = crate::agent::SdkMcpServer::new(name, version)?;
    for tool in tools.into().unwrap_or_default() {
        server.add_tool(tool)?;
    }
    Ok(server)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn string_field(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_mode_wire_values() {
        assert_eq!(
            serde_json::to_value(PermissionMode::Auto).unwrap(),
            json!("auto")
        );
        assert_eq!(
            serde_json::to_value(PermissionMode::BypassPermissions).unwrap(),
            json!("bypassPermissions")
        );
        assert_eq!(PermissionMode::DontAsk.as_cli_value(), "dontAsk");
        let parsed: PermissionMode = serde_json::from_value(json!("acceptEdits")).unwrap();
        assert_eq!(parsed, PermissionMode::AcceptEdits);
    }

    #[test]
    fn setting_source_wire_values() {
        assert_eq!(SettingSource::Project.as_cli_value(), "project");
        assert_eq!(
            serde_json::to_value(SettingSource::Local).unwrap(),
            json!("local")
        );
    }

    #[test]
    fn sdk_beta_wire() {
        assert_eq!(SdkBeta::ContextOneM.as_wire(), "context-1m-2025-08-07");
        assert_eq!(
            serde_json::to_value(SdkBeta::ContextOneM).unwrap(),
            json!("context-1m-2025-08-07")
        );
    }

    #[test]
    fn thinking_config_serialization() {
        assert_eq!(
            serde_json::to_value(ThinkingConfig::Adaptive { display: None }).unwrap(),
            json!({"type": "adaptive"})
        );
        assert_eq!(
            serde_json::to_value(ThinkingConfig::Adaptive {
                display: Some(ThinkingDisplay::Summarized)
            })
            .unwrap(),
            json!({"type": "adaptive", "display": "summarized"})
        );
        assert_eq!(
            serde_json::to_value(ThinkingConfig::Enabled {
                budget_tokens: 4096,
                display: None
            })
            .unwrap(),
            json!({"type": "enabled", "budget_tokens": 4096})
        );
        assert_eq!(
            serde_json::to_value(ThinkingConfig::Disabled).unwrap(),
            json!({"type": "disabled"})
        );
    }

    #[test]
    fn output_format_serialization() {
        let format = OutputFormat::JsonSchema {
            schema: json!({"type": "object"}),
        };
        assert_eq!(
            serde_json::to_value(&format).unwrap(),
            json!({"type": "json_schema", "schema": {"type": "object"}})
        );
    }

    #[test]
    fn task_budget_serialization() {
        assert_eq!(
            serde_json::to_value(TaskBudget { total: 12345 }).unwrap(),
            json!({"total": 12345})
        );
    }

    #[test]
    fn plugin_config_serialization() {
        assert_eq!(
            serde_json::to_value(SdkPluginConfig::local("/tmp/plugin")).unwrap(),
            json!({"type": "local", "path": "/tmp/plugin"})
        );
    }

    #[test]
    fn agent_definition_omits_none_fields() {
        let def = AgentDefinition::new("reviewer", "Review code carefully.");
        let value = def.to_initialize_value().unwrap();
        assert_eq!(
            value,
            json!({"description": "reviewer", "prompt": "Review code carefully."})
        );
    }

    #[test]
    fn agent_definition_camel_case_and_effort() {
        let def = AgentDefinition {
            description: "d".into(),
            prompt: "p".into(),
            tools: Some(vec!["Read".into()]),
            disallowed_tools: Some(vec!["Bash".into()]),
            model: Some("sonnet".into()),
            skills: Some(vec!["plugin:skill".into()]),
            memory: Some(AgentMemory::Project),
            mcp_servers: Some(vec![json!("shared")]),
            initial_prompt: Some("start".into()),
            max_turns: Some(3),
            background: Some(true),
            effort: Some(AgentEffort::Level(EffortLevel::High)),
            permission_mode: Some(PermissionMode::AcceptEdits),
        };
        let value = def.to_initialize_value().unwrap();
        assert_eq!(value["disallowedTools"], json!(["Bash"]));
        assert_eq!(value["mcpServers"], json!(["shared"]));
        assert_eq!(value["initialPrompt"], json!("start"));
        assert_eq!(value["maxTurns"], json!(3));
        assert_eq!(value["permissionMode"], json!("acceptEdits"));
        assert_eq!(value["memory"], json!("project"));
        assert_eq!(value["effort"], json!("high"));
    }

    #[test]
    fn agent_effort_untagged() {
        assert_eq!(
            serde_json::to_value(AgentEffort::Tokens(2048)).unwrap(),
            json!(2048)
        );
        assert_eq!(
            serde_json::to_value(AgentEffort::Level(EffortLevel::Max)).unwrap(),
            json!("max")
        );
    }

    #[test]
    fn mcp_server_config_discriminators() {
        let stdio = McpServerConfig::Stdio(McpStdioServerConfig {
            command: "node".into(),
            args: Some(vec!["server.js".into()]),
            env: None,
        });
        assert_eq!(
            serde_json::to_value(&stdio).unwrap(),
            json!({"type": "stdio", "command": "node", "args": ["server.js"]})
        );

        let sse = McpServerConfig::Sse(McpSseServerConfig {
            url: "https://example/sse".into(),
            headers: None,
        });
        assert_eq!(
            serde_json::to_value(&sse).unwrap(),
            json!({"type": "sse", "url": "https://example/sse"})
        );

        let http = McpServerConfig::Http(McpHttpServerConfig {
            url: "https://example/mcp".into(),
            headers: None,
        });
        assert_eq!(
            serde_json::to_value(&http).unwrap(),
            json!({"type": "http", "url": "https://example/mcp"})
        );

        let sdk = McpServerConfig::Sdk(McpSdkServerConfig {
            name: "calc".into(),
        });
        assert_eq!(
            serde_json::to_value(&sdk).unwrap(),
            json!({"type": "sdk", "name": "calc"})
        );
    }

    #[test]
    fn mcp_status_config_claudeai_proxy() {
        let config = McpServerStatusConfig::ClaudeAiProxy(McpClaudeAiProxyServerConfig {
            url: "https://claude.ai".into(),
            id: "abc".into(),
        });
        assert_eq!(
            serde_json::to_value(&config).unwrap(),
            json!({"type": "claudeai-proxy", "url": "https://claude.ai", "id": "abc"})
        );
    }

    #[test]
    fn mcp_connection_status_kebab() {
        assert_eq!(
            serde_json::to_value(McpServerConnectionStatus::NeedsAuth).unwrap(),
            json!("needs-auth")
        );
    }

    #[test]
    fn mcp_tool_annotations_camel() {
        let annotations = McpToolAnnotations {
            read_only: Some(true),
            destructive: None,
            open_world: Some(false),
        };
        assert_eq!(
            serde_json::to_value(annotations).unwrap(),
            json!({"readOnly": true, "openWorld": false})
        );
    }

    #[test]
    fn sandbox_settings_camel_and_omission() {
        let settings = SandboxSettings {
            enabled: Some(true),
            excluded_commands: Some(vec!["docker".into()]),
            network: Some(SandboxNetworkConfig {
                allow_local_binding: Some(true),
                http_proxy_port: Some(8080),
                ..SandboxNetworkConfig::default()
            }),
            ..SandboxSettings::default()
        };
        let value = settings.to_json_value().unwrap();
        assert_eq!(
            value,
            json!({
                "enabled": true,
                "excludedCommands": ["docker"],
                "network": {"allowLocalBinding": true, "httpProxyPort": 8080}
            })
        );
    }

    #[test]
    fn sandbox_settings_empty_is_empty_object() {
        assert_eq!(
            SandboxSettings::default().to_json_value().unwrap(),
            json!({})
        );
    }

    #[test]
    fn permission_update_add_rules_wire() {
        let update = PermissionUpdate {
            update_type: PermissionUpdateType::AddRules,
            rules: Some(vec![PermissionRuleValue {
                tool_name: "Bash".into(),
                rule_content: Some("ls".into()),
            }]),
            behavior: Some(PermissionBehavior::Allow),
            mode: None,
            directories: None,
            destination: Some(PermissionUpdateDestination::Session),
        };
        assert_eq!(
            update.to_wire(),
            json!({
                "type": "addRules",
                "destination": "session",
                "rules": [{"toolName": "Bash", "ruleContent": "ls"}],
                "behavior": "allow"
            })
        );
    }

    #[test]
    fn permission_update_set_mode_wire() {
        let update = PermissionUpdate {
            update_type: PermissionUpdateType::SetMode,
            rules: None,
            behavior: None,
            mode: Some(PermissionMode::Plan),
            directories: None,
            destination: None,
        };
        assert_eq!(update.to_wire(), json!({"type": "setMode", "mode": "plan"}));
    }

    #[test]
    fn permission_update_directories_wire() {
        let update = PermissionUpdate {
            update_type: PermissionUpdateType::AddDirectories,
            rules: None,
            behavior: None,
            mode: None,
            directories: Some(vec!["/a".into(), "/b".into()]),
            destination: None,
        };
        assert_eq!(
            update.to_wire(),
            json!({"type": "addDirectories", "directories": ["/a", "/b"]})
        );
    }

    #[test]
    fn permission_update_round_trip() {
        let update = PermissionUpdate {
            update_type: PermissionUpdateType::AddRules,
            rules: Some(vec![PermissionRuleValue {
                tool_name: "Read".into(),
                rule_content: None,
            }]),
            behavior: Some(PermissionBehavior::Deny),
            mode: None,
            directories: None,
            destination: Some(PermissionUpdateDestination::ProjectSettings),
        };
        let wire = update.to_wire();
        let parsed = PermissionUpdate::from_wire(&wire).unwrap();
        assert_eq!(parsed, update);
    }

    #[test]
    fn permission_result_allow_defaults_updated_input() {
        let allow = PermissionResult::allow();
        let original = json!({"path": "foo.txt"});
        assert_eq!(
            allow.to_wire(&original),
            json!({"behavior": "allow", "updatedInput": {"path": "foo.txt"}})
        );
    }

    #[test]
    fn permission_result_allow_replaces_input_and_updates() {
        let allow = PermissionResult::Allow {
            updated_input: Some(json!({"path": "bar.txt"})),
            updated_permissions: Some(vec![PermissionUpdate {
                update_type: PermissionUpdateType::SetMode,
                rules: None,
                behavior: None,
                mode: Some(PermissionMode::AcceptEdits),
                directories: None,
                destination: None,
            }]),
        };
        assert_eq!(
            allow.to_wire(&json!({"path": "foo.txt"})),
            json!({
                "behavior": "allow",
                "updatedInput": {"path": "bar.txt"},
                "updatedPermissions": [{"type": "setMode", "mode": "acceptEdits"}]
            })
        );
    }

    #[test]
    fn permission_result_deny_omits_false_interrupt() {
        assert_eq!(
            PermissionResult::deny("nope").to_wire(&json!({})),
            json!({"behavior": "deny", "message": "nope"})
        );
        let interrupting = PermissionResult::Deny {
            message: "stop".into(),
            interrupt: true,
        };
        assert_eq!(
            interrupting.to_wire(&json!({})),
            json!({"behavior": "deny", "message": "stop", "interrupt": true})
        );
    }

    #[test]
    fn tool_permission_context_from_request() {
        let request = json!({
            "tool_use_id": "tu_1",
            "agent_id": "agent_9",
            "blocked_path": "/etc/passwd",
            "decision_reason": "outside allowed dir",
            "title": "Claude wants to read foo",
            "display_name": "Read file",
            "description": "Read a file",
            "permission_suggestions": [
                {"type": "addDirectories", "directories": ["/tmp"]}
            ]
        });
        let context = ToolPermissionContext::from_request(&request);
        assert_eq!(context.tool_use_id.as_deref(), Some("tu_1"));
        assert_eq!(context.agent_id.as_deref(), Some("agent_9"));
        assert_eq!(context.blocked_path.as_deref(), Some("/etc/passwd"));
        assert_eq!(context.title.as_deref(), Some("Claude wants to read foo"));
        assert_eq!(context.suggestions.len(), 1);
        assert_eq!(
            context.suggestions[0].update_type,
            PermissionUpdateType::AddDirectories
        );
    }

    #[test]
    fn hook_event_wire() {
        assert_eq!(HookEvent::PreToolUse.as_wire(), "PreToolUse");
        assert_eq!(
            serde_json::to_value(HookEvent::PostToolUseFailure).unwrap(),
            json!("PostToolUseFailure")
        );
    }

    #[test]
    fn hook_input_pretooluse_typed() {
        let value = json!({
            "hook_event_name": "PreToolUse",
            "session_id": "s1",
            "transcript_path": "/t",
            "cwd": "/w",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_use_id": "tu_1",
            "agent_id": "a1"
        });
        let input = HookInput::from_value(value.clone());
        assert_eq!(input.base.as_ref().unwrap().session_id, "s1");
        match &input.event {
            HookInputKind::PreToolUse(pre) => {
                assert_eq!(pre.tool_name, "Bash");
                assert_eq!(pre.tool_use_id, "tu_1");
                assert_eq!(pre.agent_id.as_deref(), Some("a1"));
            }
            other => panic!("expected PreToolUse, got {other:?}"),
        }
        assert_eq!(input.raw(), &value);
    }

    #[test]
    fn hook_input_unknown_event_preserves_raw() {
        let value = json!({
            "hook_event_name": "SomethingNew",
            "session_id": "s1",
            "transcript_path": "/t",
            "cwd": "/w"
        });
        let input = HookInput::from_value(value.clone());
        assert!(matches!(&input.event, HookInputKind::Unknown(name) if name == "SomethingNew"));
        assert_eq!(input.raw(), &value);
    }

    #[test]
    fn hook_output_async_renames() {
        let output = HookJSONOutput::Async {
            async_timeout: Some(5000),
        };
        assert_eq!(
            output.to_wire(),
            json!({"async": true, "asyncTimeout": 5000})
        );
    }

    #[test]
    fn hook_output_sync_renames_continue() {
        let output = HookJSONOutput::Sync(SyncHookJSONOutput {
            continue_: Some(false),
            stop_reason: Some("done".into()),
            decision: Some("block".into()),
            ..SyncHookJSONOutput::default()
        });
        assert_eq!(
            output.to_wire(),
            json!({"continue": false, "stopReason": "done", "decision": "block"})
        );
    }

    #[test]
    fn hook_output_sync_empty() {
        let output = HookJSONOutput::Sync(SyncHookJSONOutput::default());
        assert_eq!(output.to_wire(), json!({}));
    }

    #[test]
    fn skill_selection_validation() {
        assert!(SkillSelection::All.validate().is_ok());
        assert!(
            SkillSelection::List(vec!["good-skill".into(), "plugin:skill".into()])
                .validate()
                .is_ok()
        );
        assert!(SkillSelection::List(vec!["bad*".into()])
            .validate()
            .is_err());
        assert!(SkillSelection::List(vec![" spaced".into()])
            .validate()
            .is_err());
        assert!(SkillSelection::List(vec!["a/b".into()]).validate().is_err());
        assert!(SkillSelection::List(vec![String::new()])
            .validate()
            .is_err());
    }

    #[test]
    fn skill_selection_accessors() {
        assert!(SkillSelection::All.is_all());
        assert_eq!(SkillSelection::All.names(), None);
        let list = SkillSelection::List(vec!["x".into()]);
        assert!(!list.is_all());
        assert_eq!(list.names(), Some(&["x".to_string()][..]));
    }

    #[test]
    fn pre_tool_use_specific_output_camel() {
        let output = PreToolUseHookSpecificOutput {
            hook_event_name: "PreToolUse".into(),
            permission_decision: Some("deny".into()),
            permission_decision_reason: Some("blocked".into()),
            updated_input: None,
            additional_context: None,
        };
        assert_eq!(
            serde_json::to_value(&output).unwrap(),
            json!({
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "blocked"
            })
        );
    }
}
