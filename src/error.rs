//! Error types for the Claude SDK.
//!
//! All fallible operations in this crate return [`ClaudeError`]. The variants
//! cover two overlapping surfaces:
//!
//! * **Messages API errors** -- API status failures, network errors,
//!   (de)serialization failures, batch polling timeouts, configuration
//!   mistakes, streaming errors, and unsupported transport operations.
//! * **Agent SDK errors** -- a structured taxonomy mirroring the official
//!   Python `claude_agent_sdk` error classes: CLI connection failures, missing
//!   CLI, process exits, malformed CLI JSON, typed message-parse failures, and
//!   control-request timeouts.
//!
//! Agent SDK variants preserve their causes (via [`std::error::Error::source`])
//! and expose the same inspectable fields as the Python originals so downstream
//! callers can pattern-match and recover structured context.

use thiserror::Error;

/// Errors returned by the Claude SDK.
///
/// # Messages API variants
///
/// [`ApiError`](ClaudeError::ApiError), [`NetworkError`](ClaudeError::NetworkError),
/// [`SerializationError`](ClaudeError::SerializationError),
/// [`BatchTimeout`](ClaudeError::BatchTimeout),
/// [`InvalidConfig`](ClaudeError::InvalidConfig),
/// [`StreamError`](ClaudeError::StreamError),
/// [`Unsupported`](ClaudeError::Unsupported), and
/// [`TransportError`](ClaudeError::TransportError).
///
/// # Agent SDK variants
///
/// [`CliConnection`](ClaudeError::CliConnection),
/// [`CliNotFound`](ClaudeError::CliNotFound),
/// [`Process`](ClaudeError::Process),
/// [`ResultError`](ClaudeError::ResultError),
/// [`CliJsonDecode`](ClaudeError::CliJsonDecode),
/// [`MessageParse`](ClaudeError::MessageParse), and
/// [`ControlTimeout`](ClaudeError::ControlTimeout).
///
/// The Agent SDK variants correspond directly to the official Python
/// `claude_agent_sdk` exception hierarchy:
///
/// | Rust variant                    | Python exception       |
/// | ------------------------------- | ---------------------- |
/// | [`CliConnection`](ClaudeError::CliConnection) | `CLIConnectionError`   |
/// | [`CliNotFound`](ClaudeError::CliNotFound)     | `CLINotFoundError`     |
/// | [`Process`](ClaudeError::Process)             | `ProcessError`         |
/// | [`ResultError`](ClaudeError::ResultError)     | `ResultError`          |
/// | [`CliJsonDecode`](ClaudeError::CliJsonDecode) | `CLIJSONDecodeError`   |
/// | [`MessageParse`](ClaudeError::MessageParse)   | `MessageParseError`    |
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ClaudeError {
    /// The API returned a non-success status code.
    #[error("API error (status {status}): [{error_type}] {message}")]
    ApiError {
        /// HTTP status code returned by the API.
        status: u16,
        /// Machine-readable error type (e.g. `rate_limit_error`).
        error_type: String,
        /// Human-readable error message.
        message: String,
    },

    /// A network-level error occurred (DNS, connection, timeout, etc.).
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    /// Failed to serialize a request or deserialize a response.
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// A batch job did not complete within the allowed polling window.
    #[error("Batch {batch_id} timed out waiting for completion")]
    BatchTimeout {
        /// Identifier of the batch that timed out.
        batch_id: String,
    },

    /// The SDK was configured with invalid parameters.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// An error received inside a streaming event.
    #[error("Stream error: [{error_type}] {message}")]
    StreamError {
        /// Machine-readable error type reported by the stream.
        error_type: String,
        /// Human-readable error message reported by the stream.
        message: String,
    },

    /// The transport does not support this operation.
    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    /// A transport-specific error (e.g. CLI process failure) without a more
    /// specific structured variant.
    #[error("Transport error: {0}")]
    TransportError(String),

    // -----------------------------------------------------------------------
    // Agent SDK taxonomy
    // -----------------------------------------------------------------------
    /// Unable to connect to the Claude Code CLI.
    ///
    /// Mirrors Python `CLIConnectionError`.
    #[error("{0}")]
    CliConnection(String),

    /// The Claude Code CLI could not be found or is not installed.
    ///
    /// Mirrors Python `CLINotFoundError` (a subclass of `CLIConnectionError`).
    /// Construct with [`ClaudeError::cli_not_found`] to reproduce the Python
    /// `"{message}: {cli_path}"` formatting.
    #[error("{0}")]
    CliNotFound(String),

    /// The CLI process failed.
    ///
    /// Mirrors Python `ProcessError`, retaining the optional exit code and
    /// captured stderr for inspection. The [`Display`](std::fmt::Display)
    /// output matches Python: the exit code is appended as
    /// `" (exit code: N)"` and non-empty stderr as `"\nError output: ..."`.
    #[error(fmt = fmt_process)]
    Process {
        /// Base failure message.
        message: String,
        /// Process exit code, when known.
        exit_code: Option<i32>,
        /// Captured stderr output, when available.
        stderr: Option<String>,
    },

    /// The CLI exited after reporting a terminal error `result`.
    ///
    /// Mirrors Python `ResultError`, a subclass of `ProcessError`: the CLI ends
    /// a failed run by emitting a `result` frame with `is_error: true` (also
    /// delivered as an [`AgentMessage::Result`](crate::agent::AgentMessage))
    /// and then exiting non-zero. The SDK surfaces this variant instead of the
    /// bare "exit code N" [`Process`](ClaudeError::Process) error so callers
    /// can branch on [`ResultErrorDetails::terminal_reason`] /
    /// [`ResultErrorDetails::subtype`] without string matching. Use
    /// [`ClaudeError::is_process`] and [`ClaudeError::exit_code`] to treat it
    /// like a process error.
    #[error("{0}")]
    ResultError(Box<ResultErrorDetails>),

    /// Unable to decode a JSON line emitted by the CLI.
    ///
    /// Mirrors Python `CLIJSONDecodeError`, retaining the raw `line` and the
    /// underlying decode error as the [`source`](std::error::Error::source).
    /// The [`Display`](std::fmt::Display) output truncates the line to the
    /// first 100 characters, matching Python.
    #[error(fmt = fmt_cli_json_decode)]
    CliJsonDecode {
        /// The raw line that failed to decode.
        line: String,
        /// The underlying JSON decode error.
        #[source]
        source: serde_json::Error,
    },

    /// Unable to parse a typed message from CLI output.
    ///
    /// Mirrors Python `MessageParseError`, retaining the raw `data` that
    /// failed to parse for inspection.
    #[error("{message}")]
    MessageParse {
        /// Human-readable description of the parse failure.
        message: String,
        /// The raw frame data that failed to parse, when available.
        data: Option<serde_json::Value>,
    },

    /// A control request did not receive a response within the timeout.
    ///
    /// Mirrors Python's `Control request timeout: <subtype>` error.
    #[error("Control request timeout: {subtype}")]
    ControlTimeout {
        /// The control request subtype that timed out.
        subtype: String,
    },
}

impl ClaudeError {
    /// Construct a [`CliNotFound`](ClaudeError::CliNotFound) error, mirroring
    /// Python `CLINotFoundError(message, cli_path)`.
    ///
    /// When `cli_path` is provided, the message becomes `"{message}: {cli_path}"`.
    #[must_use]
    pub fn cli_not_found(message: impl Into<String>, cli_path: Option<&str>) -> Self {
        let mut message = message.into();
        if let Some(path) = cli_path {
            message = format!("{message}: {path}");
        }
        Self::CliNotFound(message)
    }

    /// Construct a [`Process`](ClaudeError::Process) error, mirroring Python
    /// `ProcessError(message, exit_code, stderr)`.
    #[must_use]
    pub fn process(
        message: impl Into<String>,
        exit_code: Option<i32>,
        stderr: Option<String>,
    ) -> Self {
        Self::Process {
            message: message.into(),
            exit_code,
            stderr,
        }
    }

    /// Construct a [`MessageParse`](ClaudeError::MessageParse) error, mirroring
    /// Python `MessageParseError(message, data)`.
    #[must_use]
    pub fn message_parse(message: impl Into<String>, data: Option<serde_json::Value>) -> Self {
        Self::MessageParse {
            message: message.into(),
            data,
        }
    }

    /// Construct a [`ResultError`](ClaudeError::ResultError) from a raw
    /// `result` frame with `is_error: true`, mirroring Python
    /// `ResultError(message, data, exit_code)`.
    #[must_use]
    pub fn result_error(data: &serde_json::Value, exit_code: Option<i32>) -> Self {
        Self::ResultError(Box::new(ResultErrorDetails::from_result_frame(
            data, exit_code,
        )))
    }

    /// Whether this error is a CLI process failure, treating
    /// [`ResultError`](ClaudeError::ResultError) as a subtype.
    ///
    /// Mirrors the Python hierarchy where `ResultError` subclasses
    /// `ProcessError`, so `except ProcessError` also catches it.
    #[must_use]
    pub const fn is_process(&self) -> bool {
        matches!(self, Self::Process { .. } | Self::ResultError(_))
    }

    /// The CLI exit code carried by a [`Process`](ClaudeError::Process) or
    /// [`ResultError`](ClaudeError::ResultError) error.
    #[must_use]
    pub fn exit_code(&self) -> Option<i32> {
        match self {
            Self::Process { exit_code, .. } => *exit_code,
            Self::ResultError(details) => details.exit_code,
            _ => None,
        }
    }

    /// Whether this error is a CLI connection failure, treating
    /// [`CliNotFound`](ClaudeError::CliNotFound) as a subtype.
    ///
    /// Mirrors the Python exception hierarchy where `CLINotFoundError`
    /// subclasses `CLIConnectionError`, so `except CLIConnectionError` also
    /// catches a missing-CLI failure.
    #[must_use]
    pub const fn is_cli_connection(&self) -> bool {
        matches!(self, Self::CliConnection(_) | Self::CliNotFound(_))
    }
}

/// Payload of [`ClaudeError::ResultError`]: the terminal error `result` frame
/// the CLI reported before exiting non-zero.
///
/// Boxed inside the error enum to keep [`ClaudeError`] small.
#[derive(Debug, Clone, PartialEq)]
pub struct ResultErrorDetails {
    /// `"Claude Code returned an error result: <text>"`, where `<text>` is the
    /// most informative of `errors`, `result`, a non-`success` `subtype`, or
    /// the HTTP status.
    pub message: String,
    /// CLI process exit code, when known.
    pub exit_code: Option<i32>,
    /// Result subtype (`error_max_turns`, `error_during_execution`, ... or
    /// `success` when the loop completed but the last turn was an API error).
    pub subtype: Option<String>,
    /// Error strings reported by the CLI, trimmed with blank and non-string
    /// entries dropped (a bare string is accepted as a one-element list).
    pub errors: Vec<String>,
    /// Result text, if any. For API failures this holds the `API Error: ...`
    /// prose.
    pub result: Option<String>,
    /// HTTP status of the failing API call, if any.
    pub api_error_status: Option<u16>,
    /// Why the run ended (e.g. `api_error`, `max_turns`), if reported.
    pub terminal_reason: Option<String>,
    /// Session the result belongs to, if reported.
    pub session_id: Option<String>,
    /// The raw `result` frame as emitted by the CLI.
    pub data: serde_json::Value,
}

impl ResultErrorDetails {
    /// Build the details from a raw `result` frame (`is_error: true`) and the
    /// CLI exit code, normalizing fields like Python `ResultError`.
    #[must_use]
    pub fn from_result_frame(data: &serde_json::Value, exit_code: Option<i32>) -> Self {
        let text_field = |key: &str| data.get(key).and_then(|v| v.as_str()).map(str::to_owned);
        let errors = normalize_result_errors(data.get("errors"));
        let result = text_field("result");
        let subtype = text_field("subtype");
        let api_error_status = data
            .get("api_error_status")
            .and_then(serde_json::Value::as_u64)
            .and_then(|status| u16::try_from(status).ok());
        let text = if !errors.is_empty() {
            errors.join("; ")
        } else if let Some(text) = result.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
            text.to_owned()
        } else if let Some(kind) = subtype
            .as_deref()
            .filter(|t| !t.is_empty() && *t != "success")
        {
            kind.to_owned()
        } else if let Some(status) = data.get("api_error_status").filter(|v| !v.is_null()) {
            format!("API error (HTTP {status})")
        } else {
            "unknown error".to_owned()
        };
        Self {
            message: format!("Claude Code returned an error result: {text}"),
            exit_code,
            subtype,
            errors,
            result,
            api_error_status,
            terminal_reason: text_field("terminal_reason"),
            session_id: text_field("session_id"),
            data: data.clone(),
        }
    }
}

impl std::fmt::Display for ResultErrorDetails {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)?;
        if let Some(code) = self.exit_code {
            write!(f, " (exit code: {code})")?;
        }
        Ok(())
    }
}

/// Normalize a `result` frame's `errors` to clean strings (Python
/// `_normalize_result_errors`): accept a bare string, drop non-string and
/// blank entries, trim the rest.
fn normalize_result_errors(raw: Option<&serde_json::Value>) -> Vec<String> {
    match raw {
        Some(serde_json::Value::String(text)) => {
            let text = text.trim();
            if text.is_empty() {
                Vec::new()
            } else {
                vec![text.to_owned()]
            }
        }
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.as_str())
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

/// Display for [`ClaudeError::Process`], matching Python `ProcessError`.
///
/// The `&Option<_>` parameters mirror the field references that `thiserror`'s
/// `#[error(fmt = ...)]` passes in, so `ref_option` and
/// `trivially_copy_pass_by_ref` are narrowly allowed rather than changed: the
/// derive dictates the argument shapes.
#[allow(clippy::ref_option, clippy::trivially_copy_pass_by_ref)]
fn fmt_process(
    message: &str,
    exit_code: &Option<i32>,
    stderr: &Option<String>,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    f.write_str(message)?;
    if let Some(code) = exit_code {
        write!(f, " (exit code: {code})")?;
    }
    if let Some(stderr) = stderr {
        if !stderr.is_empty() {
            write!(f, "\nError output: {stderr}")?;
        }
    }
    Ok(())
}

/// Display for [`ClaudeError::CliJsonDecode`], matching Python
/// `CLIJSONDecodeError` (`Failed to decode JSON: {line[:100]}...`).
fn fmt_cli_json_decode(
    line: &str,
    _source: &serde_json::Error,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    let truncated: String = line.chars().take(100).collect();
    write!(f, "Failed to decode JSON: {truncated}...")
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::error::Error;

    #[test]
    fn api_error_display() {
        let err = ClaudeError::ApiError {
            status: 429,
            error_type: "rate_limit_error".into(),
            message: "Too many requests".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("429"));
        assert!(msg.contains("rate_limit_error"));
        assert!(msg.contains("Too many requests"));
    }

    #[test]
    fn batch_timeout_display() {
        let err = ClaudeError::BatchTimeout {
            batch_id: "batch_123".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("batch_123"));
        assert!(msg.contains("timed out"));
    }

    #[test]
    fn invalid_config_display() {
        let err = ClaudeError::InvalidConfig("model is required".into());
        let msg = format!("{err}");
        assert!(msg.contains("model is required"));
    }

    #[test]
    fn stream_error_display() {
        let err = ClaudeError::StreamError {
            error_type: "overloaded_error".into(),
            message: "Overloaded".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("overloaded_error"));
        assert!(msg.contains("Overloaded"));
    }

    #[test]
    fn serialization_error_from() {
        let serde_err = serde_json::from_str::<String>("not json").unwrap_err();
        let err: ClaudeError = serde_err.into();
        assert!(matches!(err, ClaudeError::SerializationError(_)));
        // Ensure it implements std::error::Error
        let _source = err.source();
    }

    #[test]
    fn error_variants_are_debug() {
        let errs: Vec<ClaudeError> = vec![
            ClaudeError::ApiError {
                status: 400,
                error_type: "invalid_request_error".into(),
                message: "bad".into(),
            },
            ClaudeError::BatchTimeout {
                batch_id: "b1".into(),
            },
            ClaudeError::InvalidConfig("x".into()),
            ClaudeError::StreamError {
                error_type: "e".into(),
                message: "m".into(),
            },
        ];
        for err in errs {
            let debug = format!("{err:?}");
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn api_error_status_codes() {
        let codes: Vec<(u16, &str)> = vec![
            (400, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
            (413, "request_too_large"),
            (429, "rate_limit_error"),
            (500, "api_error"),
            (529, "overloaded_error"),
        ];
        for (status, error_type) in codes {
            let err = ClaudeError::ApiError {
                status,
                error_type: error_type.into(),
                message: "test".into(),
            };
            let msg = format!("{err}");
            assert!(msg.contains(&status.to_string()));
            assert!(msg.contains(error_type));
        }
    }

    // -----------------------------------------------------------------------
    // Agent SDK taxonomy
    // -----------------------------------------------------------------------

    #[test]
    fn cli_connection_matchable_and_displays() {
        let err = ClaudeError::CliConnection("Failed to connect to CLI".into());
        assert!(matches!(err, ClaudeError::CliConnection(_)));
        assert_eq!(format!("{err}"), "Failed to connect to CLI");
        assert!(err.source().is_none());
    }

    #[test]
    fn cli_not_found_without_path() {
        let err = ClaudeError::cli_not_found("Claude Code not found", None);
        assert!(matches!(err, ClaudeError::CliNotFound(_)));
        assert_eq!(format!("{err}"), "Claude Code not found");
    }

    #[test]
    fn cli_not_found_appends_path() {
        let err = ClaudeError::cli_not_found("Claude Code not found", Some("/usr/bin/claude"));
        let msg = format!("{err}");
        assert_eq!(msg, "Claude Code not found: /usr/bin/claude");
        assert!(matches!(err, ClaudeError::CliNotFound(_)));
    }

    #[test]
    fn process_error_preserves_fields() {
        let err = ClaudeError::process("Process failed", Some(1), Some("Command not found".into()));
        let ClaudeError::Process {
            exit_code, stderr, ..
        } = &err
        else {
            panic!("expected Process variant");
        };
        assert_eq!(*exit_code, Some(1));
        assert_eq!(stderr.as_deref(), Some("Command not found"));
    }

    #[test]
    fn process_error_display_matches_python() {
        let err = ClaudeError::process("Process failed", Some(1), Some("Command not found".into()));
        let msg = format!("{err}");
        assert!(msg.contains("Process failed"));
        assert!(msg.contains("exit code: 1"));
        assert!(msg.contains("Command not found"));
        assert!(msg.contains("Error output: Command not found"));
    }

    #[test]
    fn process_error_display_omits_absent_fields() {
        let err = ClaudeError::process("Process failed", None, None);
        assert_eq!(format!("{err}"), "Process failed");

        // Empty stderr is not rendered, matching Python's truthiness check.
        let err = ClaudeError::process("Process failed", Some(2), Some(String::new()));
        assert_eq!(format!("{err}"), "Process failed (exit code: 2)");
    }

    #[test]
    fn cli_json_decode_preserves_line_and_source() {
        let bad = "{invalid json}";
        let serde_err = serde_json::from_str::<serde_json::Value>(bad).unwrap_err();
        let err = ClaudeError::CliJsonDecode {
            line: bad.into(),
            source: serde_err,
        };
        let ClaudeError::CliJsonDecode { line, .. } = &err else {
            panic!("expected CliJsonDecode variant");
        };
        assert_eq!(line, bad);
        assert!(format!("{err}").contains("Failed to decode JSON"));
        // The underlying decode error is retained as the source.
        assert!(err.source().is_some());
    }

    #[test]
    fn cli_json_decode_display_truncates_to_100_chars() {
        let long_line = "x".repeat(500);
        let serde_err = serde_json::from_str::<serde_json::Value>("nope").unwrap_err();
        let err = ClaudeError::CliJsonDecode {
            line: long_line,
            source: serde_err,
        };
        let msg = format!("{err}");
        assert!(msg.starts_with("Failed to decode JSON: "));
        assert!(msg.ends_with("..."));
        // 100 rendered characters from the line, no more.
        let rendered = msg
            .strip_prefix("Failed to decode JSON: ")
            .and_then(|s| s.strip_suffix("..."))
            .unwrap();
        assert_eq!(rendered.chars().count(), 100);
    }

    #[test]
    fn message_parse_preserves_data() {
        let data = json!({"type": "weird", "payload": 42});
        let err = ClaudeError::message_parse("unexpected message type", Some(data.clone()));
        let ClaudeError::MessageParse {
            message,
            data: retained,
        } = &err
        else {
            panic!("expected MessageParse variant");
        };
        assert_eq!(message, "unexpected message type");
        assert_eq!(retained.as_ref(), Some(&data));
        assert_eq!(format!("{err}"), "unexpected message type");
    }

    #[test]
    fn message_parse_allows_absent_data() {
        let err = ClaudeError::message_parse("no data", None);
        assert!(matches!(err, ClaudeError::MessageParse { data: None, .. }));
    }

    #[test]
    fn control_timeout_display() {
        let err = ClaudeError::ControlTimeout {
            subtype: "initialize".into(),
        };
        assert!(matches!(err, ClaudeError::ControlTimeout { .. }));
        assert_eq!(format!("{err}"), "Control request timeout: initialize");
    }

    #[test]
    fn agent_variants_are_pattern_matchable() {
        // Exhaustive smoke test that every named structured variant can be
        // constructed and matched by downstream code.
        let variants: Vec<ClaudeError> = vec![
            ClaudeError::CliConnection("c".into()),
            ClaudeError::cli_not_found("n", Some("/p")),
            ClaudeError::process("p", Some(1), Some("e".into())),
            ClaudeError::CliJsonDecode {
                line: "l".into(),
                source: serde_json::from_str::<serde_json::Value>("bad").unwrap_err(),
            },
            ClaudeError::message_parse("m", Some(json!({"a": 1}))),
            ClaudeError::ControlTimeout {
                subtype: "s".into(),
            },
        ];
        for err in variants {
            let matched = matches!(
                err,
                ClaudeError::CliConnection(_)
                    | ClaudeError::CliNotFound(_)
                    | ClaudeError::Process { .. }
                    | ClaudeError::CliJsonDecode { .. }
                    | ClaudeError::MessageParse { .. }
                    | ClaudeError::ControlTimeout { .. }
            );
            assert!(matched);
            // Debug is available for all.
            assert!(!format!("{err:?}").is_empty());
        }
    }

    #[test]
    fn result_error_normalizes_frame_and_counts_as_process() {
        let frame = json!({
            "type": "result",
            "is_error": true,
            "subtype": "error_max_turns",
            "errors": ["  first ", "", 7, "second"],
            "terminal_reason": "max_turns",
            "api_error_status": 529,
            "session_id": "s1",
        });
        let err = ClaudeError::result_error(&frame, Some(1));
        assert!(err.is_process());
        assert_eq!(err.exit_code(), Some(1));
        assert_eq!(
            err.to_string(),
            "Claude Code returned an error result: first; second (exit code: 1)"
        );
        let ClaudeError::ResultError(details) = err else {
            panic!("expected ResultError");
        };
        assert_eq!(details.errors, ["first", "second"]);
        assert_eq!(details.subtype.as_deref(), Some("error_max_turns"));
        assert_eq!(details.terminal_reason.as_deref(), Some("max_turns"));
        assert_eq!(details.api_error_status, Some(529));
        assert_eq!(details.session_id.as_deref(), Some("s1"));
        assert_eq!(details.data, frame);
    }

    #[test]
    fn result_error_text_falls_back_to_result_then_subtype_then_status() {
        let api = json!({"subtype": "success", "result": " API Error: overloaded ", "errors": []});
        assert!(ClaudeError::result_error(&api, None)
            .to_string()
            .ends_with(": API Error: overloaded"));
        let bare = json!({"subtype": "error_during_execution"});
        assert!(ClaudeError::result_error(&bare, None)
            .to_string()
            .ends_with(": error_during_execution"));
        let status = json!({"subtype": "success", "api_error_status": 500});
        assert!(ClaudeError::result_error(&status, None)
            .to_string()
            .ends_with(": API error (HTTP 500)"));
        let string_errors = json!({"errors": "boom"});
        let ClaudeError::ResultError(details) = ClaudeError::result_error(&string_errors, None)
        else {
            panic!("expected ResultError");
        };
        assert_eq!(details.errors, ["boom"]);
        assert!(!ClaudeError::process("p", None, None).to_string().is_empty());
    }
}
