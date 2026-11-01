//! Model ID constants for the Claude API.
//!
//! These constants provide current identifiers across the Fable, Mythos, Opus,
//! Sonnet, and Haiku families. You can also use any model ID string directly;
//! the SDK does not restrict which models you use.
//!
//! Starting with Claude 4.6, dateless model IDs such as
//! [`CLAUDE_OPUS_5_5`] are pinned snapshots, not moving aliases. Older
//! convenience aliases such as [`CLAUDE_HAIKU_4_5`] may resolve to a dated
//! snapshot.
//!
//! Deprecated models are marked `#[deprecated]` with their end-of-life date.
//! Retired models are removed because requests naming them fail.
//!
//! # Example
//!
//! ```ignore
//! use claude_agent_rust_sdk::models;
//!
//! let response = client
//!     .messages()
//!     .model(models::CLAUDE_OPUS_5_5)
//!     .max_tokens(1024)
//!     .user("Hello!")
//!     .send()
//!     .await?;
//! ```

// ---------------------------------------------------------------------------
// Claude Fable family
// ---------------------------------------------------------------------------

/// Claude Fable 5.1 -- frontier intelligence for ambitious tasks across
/// coding, scientific discovery, and enterprise workflows.
pub const CLAUDE_FABLE_5_1: &str = "claude-fable-5-1";

/// Claude Fable 5 -- for the hardest knowledge work and coding problems.
pub const CLAUDE_FABLE_5: &str = "claude-fable-5";

// ---------------------------------------------------------------------------
// Claude Mythos family
// ---------------------------------------------------------------------------

/// Claude Mythos 5.1 -- most capable model for cybersecurity and biology
/// research, available through trusted access programs.
pub const CLAUDE_MYTHOS_5_1: &str = "claude-mythos-5-1";

/// Claude Mythos 5 -- limited-availability defensive cybersecurity model.
pub const CLAUDE_MYTHOS_5: &str = "claude-mythos-5";

/// Invitation-only Claude Mythos Preview model.
#[deprecated(note = "end-of-life June 30, 2026; migrate to CLAUDE_MYTHOS_5")]
pub const CLAUDE_MYTHOS_PREVIEW: &str = "claude-mythos-preview";

// ---------------------------------------------------------------------------
// Claude Opus family
// ---------------------------------------------------------------------------

/// Claude Opus 5.5 -- powerful intelligence for coding, knowledge work, and
/// long-running agents.
pub const CLAUDE_OPUS_5_5: &str = "claude-opus-5-5";

/// Claude Opus 5 -- complex agentic coding and enterprise work.
pub const CLAUDE_OPUS_5: &str = "claude-opus-5";

/// Claude Opus 4.8 -- legacy adaptive-thinking Opus snapshot.
pub const CLAUDE_OPUS_4_8: &str = "claude-opus-4-8";

/// Claude Opus 4.7 -- legacy adaptive-thinking Opus snapshot.
pub const CLAUDE_OPUS_4_7: &str = "claude-opus-4-7";

/// Claude Opus 4.6 -- legacy adaptive-thinking Opus snapshot.
pub const CLAUDE_OPUS_4_6: &str = "claude-opus-4-6";

/// Claude Opus 4.5 (date-pinned).
pub const CLAUDE_OPUS_4_5: &str = "claude-opus-4-5-20251101";

// ---------------------------------------------------------------------------
// Claude Sonnet family
// ---------------------------------------------------------------------------

/// Claude Sonnet 5.5 -- the best combination of speed and intelligence.
pub const CLAUDE_SONNET_5_5: &str = "claude-sonnet-5-5";

/// Claude Sonnet 5.
pub const CLAUDE_SONNET_5: &str = "claude-sonnet-5";

/// Claude Sonnet 4.6 -- legacy adaptive-thinking Sonnet snapshot.
pub const CLAUDE_SONNET_4_6: &str = "claude-sonnet-4-6";

/// Claude Sonnet 4.5 (date-pinned).
#[deprecated(note = "end-of-life November 30, 2026; migrate to CLAUDE_SONNET_5_5")]
pub const CLAUDE_SONNET_4_5: &str = "claude-sonnet-4-5-20250929";

// ---------------------------------------------------------------------------
// Claude Haiku family
// ---------------------------------------------------------------------------

/// Claude Haiku 5.5 -- fastest model, for high-volume, latency-sensitive
/// tasks.
pub const CLAUDE_HAIKU_5_5: &str = "claude-haiku-5-5";

/// Claude Haiku 4.5.
pub const CLAUDE_HAIKU_4_5: &str = "claude-haiku-4-5";

/// Claude Haiku 4.5 (date-pinned).
pub const CLAUDE_HAIKU_4_5_PINNED: &str = "claude-haiku-4-5-20251001";
