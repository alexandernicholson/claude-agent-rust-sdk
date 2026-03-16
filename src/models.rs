//! Model ID constants for the Claude API.
//!
//! These constants provide the current model identifiers for the Opus, Sonnet,
//! and Haiku families. You can also use any model ID string directly -- the SDK
//! does not restrict which models you use.
//!
//! Constants without a date suffix (e.g. [`CLAUDE_OPUS_4_6`]) track the latest
//! release and may change over time. Date-pinned constants (e.g.
//! [`CLAUDE_OPUS_4_5`]) always point to a specific snapshot.
//!
//! # Example
//!
//! ```ignore
//! use claude_agent_rust_sdk::models;
//!
//! let response = client
//!     .messages()
//!     .model(models::CLAUDE_SONNET_4_6)
//!     .max_tokens(1024)
//!     .user("Hello!")
//!     .send()
//!     .await?;
//! ```

// ---------------------------------------------------------------------------
// Claude Opus family
// ---------------------------------------------------------------------------

/// Claude Opus 4.6 -- most intelligent model, best for agents and complex coding.
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

/// Claude Sonnet 4.6 -- best balance of speed and intelligence.
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
            CLAUDE_OPUS_4_6,
            CLAUDE_OPUS_4_5,
            CLAUDE_OPUS_4_1,
            CLAUDE_OPUS_4_0,
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
                "Model constant should contain 'claude': {}",
                model
            );
        }
    }

    #[test]
    fn opus_models_contain_opus() {
        assert!(CLAUDE_OPUS_4_6.contains("opus"));
        assert!(CLAUDE_OPUS_4_5.contains("opus"));
        assert!(CLAUDE_OPUS_4_1.contains("opus"));
        assert!(CLAUDE_OPUS_4_0.contains("opus"));
    }

    #[test]
    fn sonnet_models_contain_sonnet() {
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
