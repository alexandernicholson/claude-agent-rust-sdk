//! Server-Sent Events (SSE) parsing and async stream support for the Claude
//! streaming API.
//!
//! When a request is made with `stream: true`, the Claude API returns a stream
//! of SSE events. This module provides:
//!
//! - [`parse_sse_line`] -- parse a single SSE data line into a [`StreamEvent`].
//! - [`SseStream`] -- an async `Stream` of [`StreamEvent`] values built on top
//!   of a `reqwest::Response` byte stream.

use futures::stream::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::error::ClaudeError;
use crate::types::StreamEvent;

/// Parse an SSE `data:` line into a [`StreamEvent`].
///
/// Returns `None` if the line is empty, a comment, or doesn't start with
/// `data: `. Returns `Some(Err(...))` if the JSON is malformed.
pub fn parse_sse_line(line: &str) -> Option<Result<StreamEvent, ClaudeError>> {
    let line = line.trim();

    // Skip empty lines and comments
    if line.is_empty() || line.starts_with(':') {
        return None;
    }

    // We only care about data lines
    if let Some(data) = line.strip_prefix("data: ") {
        // The [DONE] sentinel is not used by Claude, but handle it gracefully
        if data == "[DONE]" {
            return None;
        }
        Some(
            serde_json::from_str::<StreamEvent>(data)
                .map_err(ClaudeError::SerializationError),
        )
    } else {
        // event: lines, id: lines, retry: lines -- skip them
        None
    }
}

/// An async stream of [`StreamEvent`] values parsed from an SSE response.
///
/// Constructed by [`ClaudeClient::create_message_stream`](crate::client::ClaudeClient::create_message_stream).
pub struct SseStream {
    /// Internal state: buffered lines from the response.
    inner: Pin<Box<dyn Stream<Item = Result<StreamEvent, ClaudeError>> + Send>>,
}

impl std::fmt::Debug for SseStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SseStream").finish_non_exhaustive()
    }
}

impl SseStream {
    /// Create a new `SseStream` from a `reqwest::Response`.
    ///
    /// The response must have been initiated with `stream: true`.
    pub(crate) fn from_response(response: reqwest::Response) -> Self {
        let stream = async_stream(response);
        Self {
            inner: Box::pin(stream),
        }
    }

    /// Create an `SseStream` from any async stream of [`StreamEvent`] values.
    ///
    /// This is useful for custom [`Transport`](crate::transport::Transport)
    /// implementations that produce stream events from non-HTTP sources.
    pub fn from_stream<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<StreamEvent, ClaudeError>> + Send + 'static,
    {
        Self {
            inner: Box::pin(stream),
        }
    }
}

fn async_stream(
    response: reqwest::Response,
) -> impl Stream<Item = Result<StreamEvent, ClaudeError>> + Send {
    futures::stream::unfold(
        StreamState::new(response),
        |mut state| async move {
            loop {
                // If we have buffered lines, process the next one
                if let Some(line) = state.next_buffered_line() {
                    if let Some(result) = parse_sse_line(&line) {
                        return Some((result, state));
                    }
                    // Line was empty/comment/event-type, skip
                    continue;
                }

                // Read more bytes from the response
                match state.read_chunk().await {
                    Ok(true) => {},   // Got more data, loop again
                    Ok(false) => return None, // Stream ended
                    Err(e) => return Some((Err(e), state)),
                }
            }
        },
    )
}

/// Internal state for the SSE stream unfold.
struct StreamState {
    response: reqwest::Response,
    buffer: String,
    lines: Vec<String>,
    line_index: usize,
}

impl StreamState {
    fn new(response: reqwest::Response) -> Self {
        Self {
            response,
            buffer: String::new(),
            lines: Vec::new(),
            line_index: 0,
        }
    }

    fn next_buffered_line(&mut self) -> Option<String> {
        if self.line_index < self.lines.len() {
            let line = self.lines[self.line_index].clone();
            self.line_index += 1;
            Some(line)
        } else {
            None
        }
    }

    async fn read_chunk(&mut self) -> Result<bool, ClaudeError> {
        match self.response.chunk().await? {
            Some(bytes) => {
                let text = String::from_utf8_lossy(&bytes);
                self.buffer.push_str(&text);

                // Split on double newlines (SSE event boundaries) or single newlines
                let lines: Vec<String> = self
                    .buffer
                    .split('\n')
                    .map(ToString::to_string)
                    .collect();

                // The last element might be an incomplete line
                if self.buffer.ends_with('\n') {
                    self.lines = lines;
                    self.buffer.clear();
                } else {
                    // Keep the last incomplete line in the buffer
                    let last = lines.last().cloned().unwrap_or_default();
                    self.lines = lines[..lines.len().saturating_sub(1)].to_vec();
                    self.buffer = last;
                }

                self.line_index = 0;
                Ok(true)
            }
            None => {
                // Process any remaining buffer
                if self.buffer.is_empty() {
                    Ok(false)
                } else {
                    let remaining = std::mem::take(&mut self.buffer);
                    self.lines = remaining.split('\n').map(ToString::to_string).collect();
                    self.line_index = 0;
                    Ok(true)
                }
            }
        }
    }
}

impl Stream for SseStream {
    type Item = Result<StreamEvent, ClaudeError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentDelta, StreamContentBlock};

    #[test]
    fn parse_sse_line_empty() {
        assert!(parse_sse_line("").is_none());
        assert!(parse_sse_line("  ").is_none());
    }

    #[test]
    fn parse_sse_line_comment() {
        assert!(parse_sse_line(": keep-alive").is_none());
    }

    #[test]
    fn parse_sse_line_event_line() {
        // event: lines are skipped
        assert!(parse_sse_line("event: message_start").is_none());
    }

    #[test]
    fn parse_sse_line_done_sentinel() {
        assert!(parse_sse_line("data: [DONE]").is_none());
    }

    #[test]
    fn parse_sse_line_ping() {
        let result = parse_sse_line("data: {\"type\": \"ping\"}");
        assert!(result.is_some());
        let event = result.unwrap().unwrap();
        assert!(matches!(event, StreamEvent::Ping {}));
    }

    #[test]
    fn parse_sse_line_message_start() {
        let line = r#"data: {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": [], "model": "claude-opus-4-6", "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 25, "output_tokens": 1}}}"#;
        let result = parse_sse_line(line);
        assert!(result.is_some());
        let event = result.unwrap().unwrap();
        match event {
            StreamEvent::MessageStart { message } => {
                assert_eq!(message.id, "msg_1");
            }
            _ => panic!("expected MessageStart"),
        }
    }

    #[test]
    fn parse_sse_line_content_block_start() {
        let line = r#"data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}"#;
        let result = parse_sse_line(line).unwrap().unwrap();
        match result {
            StreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                assert_eq!(index, 0);
                assert!(matches!(content_block, StreamContentBlock::Text { .. }));
            }
            _ => panic!("expected ContentBlockStart"),
        }
    }

    #[test]
    fn parse_sse_line_text_delta() {
        let line =
            r#"data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}"#;
        let result = parse_sse_line(line).unwrap().unwrap();
        match result {
            StreamEvent::ContentBlockDelta {
                delta: ContentDelta::TextDelta { text },
                ..
            } => assert_eq!(text, "Hello"),
            _ => panic!("expected TextDelta"),
        }
    }

    #[test]
    fn parse_sse_line_input_json_delta() {
        let line = r#"data: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"loc\": \"SF\"}"}}"#;
        let result = parse_sse_line(line).unwrap().unwrap();
        match result {
            StreamEvent::ContentBlockDelta {
                delta: ContentDelta::InputJsonDelta { partial_json },
                ..
            } => assert!(partial_json.contains("SF")),
            _ => panic!("expected InputJsonDelta"),
        }
    }

    #[test]
    fn parse_sse_line_thinking_delta() {
        let line = r#"data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me think..."}}"#;
        let result = parse_sse_line(line).unwrap().unwrap();
        match result {
            StreamEvent::ContentBlockDelta {
                delta: ContentDelta::ThinkingDelta { thinking },
                ..
            } => assert_eq!(thinking, "Let me think..."),
            _ => panic!("expected ThinkingDelta"),
        }
    }

    #[test]
    fn parse_sse_line_signature_delta() {
        let line = r#"data: {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "EqQB"}}"#;
        let result = parse_sse_line(line).unwrap().unwrap();
        match result {
            StreamEvent::ContentBlockDelta {
                delta: ContentDelta::SignatureDelta { signature },
                ..
            } => assert_eq!(signature, "EqQB"),
            _ => panic!("expected SignatureDelta"),
        }
    }

    #[test]
    fn parse_sse_line_content_block_stop() {
        let line = r#"data: {"type": "content_block_stop", "index": 0}"#;
        let result = parse_sse_line(line).unwrap().unwrap();
        assert!(matches!(
            result,
            StreamEvent::ContentBlockStop { index: 0 }
        ));
    }

    #[test]
    fn parse_sse_line_message_delta() {
        let line = r#"data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 15}}"#;
        let result = parse_sse_line(line).unwrap().unwrap();
        match result {
            StreamEvent::MessageDelta { delta, usage } => {
                assert_eq!(delta.stop_reason.as_deref(), Some("end_turn"));
                assert_eq!(usage.unwrap().output_tokens, 15);
            }
            _ => panic!("expected MessageDelta"),
        }
    }

    #[test]
    fn parse_sse_line_message_stop() {
        let line = r#"data: {"type": "message_stop"}"#;
        let result = parse_sse_line(line).unwrap().unwrap();
        assert!(matches!(result, StreamEvent::MessageStop {}));
    }

    #[test]
    fn parse_sse_line_error_event() {
        let line =
            r#"data: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}"#;
        let result = parse_sse_line(line).unwrap().unwrap();
        match result {
            StreamEvent::Error { error } => {
                assert_eq!(error.error_type, "overloaded_error");
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn parse_sse_line_malformed_json() {
        let line = r#"data: {not valid json}"#;
        let result = parse_sse_line(line);
        assert!(result.is_some());
        assert!(result.unwrap().is_err());
    }
}
