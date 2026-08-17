//! Model ID constants for the Claude API.
//!
//! These constants provide current identifiers across the Fable, Mythos, Opus,
//! Sonnet, and Haiku families. You can also use any model ID string directly;
//! the SDK does not restrict which models you use.
//!
//! Starting with Claude 4.6, dateless model IDs such as
//! [`CLAUDE_OPUS_5`] are pinned snapshots, not moving aliases. Older
//! convenience aliases such as [`CLAUDE_HAIKU_4_5`] may resolve to a dated
//! snapshot.
//!
//! # Example
//!
//! ```ignore
//! use claude_agent_rust_sdk::models;
//!
//! let response = client
//!     .messages()
//!     .model(models::CLAUDE_SONNET_5)
//!     .max_tokens(1024)
//!     .user("Hello!")
//!     .send()
//!     .await?;
//! ```

// ---------------------------------------------------------------------------
// Claude 5 frontier models
// ---------------------------------------------------------------------------

/// Claude Fable 5 -- most capable widely released model for long-running agents.
pub const CLAUDE_FABLE_5: &str = "claude-fable-5";

/// Claude Mythos 5 -- limited-availability defensive cybersecurity model.
pub const CLAUDE_MYTHOS_5: &str = "claude-mythos-5";

/// Invitation-only Claude Mythos Preview model.
pub const CLAUDE_MYTHOS_PREVIEW: &str = "claude-mythos-preview";

// ---------------------------------------------------------------------------
// Claude Opus family
// ---------------------------------------------------------------------------
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

/// Claude Opus 4.1 (date-pinned).
pub const CLAUDE_OPUS_4_1: &str = "claude-opus-4-1-20250805";

/// Claude Opus 4.0 (date-pinned).
pub const CLAUDE_OPUS_4_0: &str = "claude-opus-4-20250514";

// ---------------------------------------------------------------------------
// Claude Sonnet family
// ---------------------------------------------------------------------------
/// Claude Sonnet 5 -- current speed and intelligence balance.
pub const CLAUDE_SONNET_5: &str = "claude-sonnet-5";

/// Claude Sonnet 4.6 -- legacy adaptive-thinking Sonnet snapshot.
pub const CLAUDE_SONNET_4_6: &str = "claude-sonnet-4-6";

/// Claude Sonnet 4.5 (date-pinned).
pub const CLAUDE_SONNET_4_5: &str = "claude-sonnet-4-5-20250929";

/// Claude Sonnet 4.0 (date-pinned).
pub const CLAUDE_SONNET_4_0: &str = "claude-sonnet-4-20250514";

// ---------------------------------------------------------------------------
// Claude Haiku family
// ---------------------------------------------------------------------------

/// Claude Haiku 4.5 -- fastest model with near-frontier intelligence.
pub const CLAUDE_HAIKU_4_5: &str = "claude-haiku-4-5";

/// Claude Haiku 4.5 (date-pinned).
pub const CLAUDE_HAIKU_4_5_PINNED: &str = "claude-haiku-4-5-20251001";

/// Claude 3 Haiku (legacy, date-pinned).
pub const CLAUDE_3_HAIKU: &str = "claude-3-haiku-20240307";

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_constants_are_valid_strings() {
        // Verify all constants are non-empty and contain "claude"
        let models = [
            CLAUDE_FABLE_5,
            CLAUDE_MYTHOS_5,
            CLAUDE_MYTHOS_PREVIEW,
            CLAUDE_OPUS_5,
            CLAUDE_OPUS_4_8,
            CLAUDE_OPUS_4_7,
            CLAUDE_OPUS_4_6,
            CLAUDE_OPUS_4_5,
            CLAUDE_OPUS_4_1,
            CLAUDE_OPUS_4_0,
            CLAUDE_SONNET_5,
            CLAUDE_SONNET_4_6,
            CLAUDE_SONNET_4_5,
            CLAUDE_SONNET_4_0,
            CLAUDE_HAIKU_4_5,
            CLAUDE_HAIKU_4_5_PINNED,
            CLAUDE_3_HAIKU,
        ];
        for model in models {
            assert!(!model.is_empty(), "Model constant should not be empty");
            assert!(
                model.contains("claude"),
                "Model constant should contain 'claude': {model}"
            );
        }
    }

    #[test]
    fn current_model_ids_match_api_contract() {
        assert_eq!(CLAUDE_FABLE_5, "claude-fable-5");
        assert_eq!(CLAUDE_MYTHOS_5, "claude-mythos-5");
        assert_eq!(CLAUDE_MYTHOS_PREVIEW, "claude-mythos-preview");
        assert_eq!(CLAUDE_OPUS_5, "claude-opus-5");
        assert_eq!(CLAUDE_OPUS_4_8, "claude-opus-4-8");
        assert_eq!(CLAUDE_OPUS_4_7, "claude-opus-4-7");
        assert_eq!(CLAUDE_SONNET_5, "claude-sonnet-5");
    }

    #[test]
    fn opus_models_contain_opus() {
        assert!(CLAUDE_OPUS_5.contains("opus"));
        assert!(CLAUDE_OPUS_4_8.contains("opus"));
        assert!(CLAUDE_OPUS_4_7.contains("opus"));
        assert!(CLAUDE_OPUS_4_6.contains("opus"));
        assert!(CLAUDE_OPUS_4_5.contains("opus"));
        assert!(CLAUDE_OPUS_4_1.contains("opus"));
        assert!(CLAUDE_OPUS_4_0.contains("opus"));
    }

    #[test]
    fn sonnet_models_contain_sonnet() {
        assert!(CLAUDE_SONNET_5.contains("sonnet"));
        assert!(CLAUDE_SONNET_4_6.contains("sonnet"));
        assert!(CLAUDE_SONNET_4_5.contains("sonnet"));
        assert!(CLAUDE_SONNET_4_0.contains("sonnet"));
    }

    #[test]
    fn haiku_models_contain_haiku() {
        assert!(CLAUDE_HAIKU_4_5.contains("haiku"));
        assert!(CLAUDE_HAIKU_4_5_PINNED.contains("haiku"));
        assert!(CLAUDE_3_HAIKU.contains("haiku"));
    }

    #[test]
    fn pinned_models_have_date() {
        // Pinned models should have a date suffix (YYYYMMDD)
        assert!(CLAUDE_OPUS_4_5.chars().last().unwrap().is_ascii_digit());
        assert!(CLAUDE_OPUS_4_1.chars().last().unwrap().is_ascii_digit());
        assert!(CLAUDE_OPUS_4_0.chars().last().unwrap().is_ascii_digit());
        assert!(CLAUDE_SONNET_4_5.chars().last().unwrap().is_ascii_digit());
        assert!(CLAUDE_SONNET_4_0.chars().last().unwrap().is_ascii_digit());
        assert!(CLAUDE_HAIKU_4_5_PINNED
            .chars()
            .last()
            .unwrap()
            .is_ascii_digit());
        assert!(CLAUDE_3_HAIKU.chars().last().unwrap().is_ascii_digit());
    }
}
