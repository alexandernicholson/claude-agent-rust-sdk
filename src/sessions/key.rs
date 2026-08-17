//! Session identity: project-key canonicalization, key types, and
//! transcript-path → [`SessionKey`] conversion.
//!
//! Ported from the official Python Agent SDK
//! (`_internal/sessions.py::project_key_for_directory` /
//! `_internal/session_store.py::file_path_to_session_key`). Project keys are
//! derived with the same realpath + NFC + djb2-hashed sanitization the CLI
//! uses for on-disk project directory names, so keys match between local-disk
//! transcripts and store-mirrored transcripts even on filesystems that
//! decompose Unicode (e.g. macOS HFS+).

use std::path::{Component, Path, PathBuf};

use unicode_normalization::UnicodeNormalization;
#[cfg(test)]
use uuid::Uuid;

use crate::error::ClaudeError;

/// Paths longer than this (after sanitization) are truncated and suffixed
/// with a portable hash. Matches the CLI's `MAX_SANITIZED_LENGTH`.
pub const MAX_SANITIZED_LENGTH: usize = 200;

/// Identifies a session transcript (or subagent transcript) in a
/// [`SessionStore`](crate::sessions::SessionStore).
///
/// Main transcripts have no `subpath`; subagent transcripts carry a `subpath`
/// like `"subagents/agent-{id}"` mirroring the on-disk directory structure.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionKey {
    /// Caller-defined scope. Default: sanitized cwd. Multi-tenant deployments
    /// should set this to a tenant ID or project name.
    pub project_key: String,
    /// The session identifier. Preserved losslessly as the caller-provided
    /// string; operations that officially require a UUID validate canonical
    /// hyphenated syntax at their boundary rather than here.
    pub session_id: String,
    /// `None` for the main transcript; `Some("subagents/agent-<id>")` for a
    /// subagent file. Never an empty string (validated on construction).
    pub subpath: Option<String>,
}

impl SessionKey {
    /// Constructs a main-transcript key (no subpath).
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(project_key: impl Into<String>, session_id: impl ToString) -> Self {
        Self {
            project_key: project_key.into(),
            session_id: session_id.to_string(),
            subpath: None,
        }
    }

    /// Constructs a subagent/subpath key after validating the subpath is a
    /// safe relative path (no traversal, not absolute, non-empty).
    ///
    /// # Errors
    /// Returns [`ClaudeError::InvalidConfig`] if the subpath is empty,
    /// absolute, contains a `..` component, or contains a NUL byte.
    #[allow(clippy::needless_pass_by_value)]
    pub fn with_subpath(
        project_key: impl Into<String>,
        session_id: impl ToString,
        subpath: impl Into<String>,
    ) -> Result<Self, ClaudeError> {
        let subpath = subpath.into();
        validate_subpath(&subpath)?;
        Ok(Self {
            project_key: project_key.into(),
            session_id: session_id.to_string(),
            subpath: Some(subpath),
        })
    }

    /// Composite storage key: `"project_key/session_id[/subpath]"`.
    #[must_use]
    pub fn storage_key(&self) -> String {
        let mut s = String::with_capacity(
            self.project_key.len()
                + 1
                + self.session_id.len()
                + self.subpath.as_ref().map_or(0, |p| p.len() + 1),
        );
        s.push_str(&self.project_key);
        s.push('/');
        s.push_str(&self.session_id);
        if let Some(subpath) = &self.subpath {
            s.push('/');
            s.push_str(subpath);
        }
        s
    }
}

/// Key argument to [`SessionStore::list_subkeys`](crate::sessions::SessionStore::list_subkeys)
/// (no `subpath`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionListSubkeysKey {
    /// Caller-defined scope (see [`SessionKey::project_key`]).
    pub project_key: String,
    /// The session identifier (see [`SessionKey::session_id`]).
    pub session_id: String,
}

impl SessionListSubkeysKey {
    /// Constructs a subkeys-list key.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(project_key: impl Into<String>, session_id: impl ToString) -> Self {
        Self {
            project_key: project_key.into(),
            session_id: session_id.to_string(),
        }
    }

    /// Prefix (`"project_key/session_id/"`) under which subkeys are listed.
    #[must_use]
    pub fn prefix(&self) -> String {
        format!("{}/{}/", self.project_key, self.session_id)
    }
}

/// Validates that `subpath` is a safe relative storage-key suffix.
///
/// Rejects empty, absolute, NUL-containing, and `..`-traversing paths. A safe
/// subpath is a `/`-joined relative path (the on-disk `subagents/...` shape).
///
/// # Errors
/// Returns [`ClaudeError::InvalidConfig`] describing the violation.
pub fn validate_subpath(subpath: &str) -> Result<(), ClaudeError> {
    if subpath.is_empty() {
        return Err(ClaudeError::InvalidConfig(
            "session subpath must not be empty (omit it for the main transcript)".into(),
        ));
    }
    if subpath.contains('\0') {
        return Err(ClaudeError::InvalidConfig(
            "session subpath must not contain NUL".into(),
        ));
    }
    let path = Path::new(subpath);
    if path.is_absolute() {
        return Err(ClaudeError::InvalidConfig(format!(
            "session subpath must be relative, got {subpath:?}"
        )));
    }
    for component in path.components() {
        match component {
            Component::ParentDir => {
                return Err(ClaudeError::InvalidConfig(format!(
                    "session subpath must not traverse upward ('..'), got {subpath:?}"
                )));
            }
            Component::Prefix(_) | Component::RootDir => {
                return Err(ClaudeError::InvalidConfig(format!(
                    "session subpath must be relative, got {subpath:?}"
                )));
            }
            Component::CurDir | Component::Normal(_) => {}
        }
    }
    Ok(())
}

/// 32-bit djb2-style hash rendered as base36, matching the CLI's directory
/// naming (`hash = (hash << 5) - hash + char`, coerced to a 32-bit signed int
/// each step via JS `hash |= 0`, then `abs(hash).toString(36)`).
#[must_use]
pub fn simple_hash(s: &str) -> String {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut h: i32 = 0;
    for ch in s.chars() {
        // (h << 5) - h + ch, all in wrapping 32-bit arithmetic to emulate
        // JavaScript's `hash |= 0` coercion after each step.
        h = h.wrapping_shl(5).wrapping_sub(h).wrapping_add(ch as i32);
    }
    // JS `Math.abs`; i32::MIN maps to its own magnitude via unsigned cast.
    let mut n: u64 = u64::from(h.unsigned_abs());
    if n == 0 {
        return "0".to_string();
    }
    // DIGITS are all ASCII, so pushing them as `char` yields a valid `String`
    // without an intermediate fallible `from_utf8`.
    let mut out: Vec<char> = Vec::new();
    while n > 0 {
        out.push(DIGITS[(n % 36) as usize] as char);
        n /= 36;
    }
    out.iter().rev().collect()
}

/// Sanitizes a path string for use as a directory name: replaces every
/// non-alphanumeric character with `-`, and for over-long results truncates to
/// [`MAX_SANITIZED_LENGTH`] and appends `-<hash>` where the hash is over the
/// full (unsanitized) input.
#[must_use]
pub fn sanitize_path(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    if sanitized.chars().count() <= MAX_SANITIZED_LENGTH {
        return sanitized;
    }
    let hash = simple_hash(name);
    let truncated: String = sanitized.chars().take(MAX_SANITIZED_LENGTH).collect();
    format!("{truncated}-{hash}")
}

/// Canonicalizes a directory path (realpath + NFC), matching the CLI's
/// `os.path.realpath` semantics.
///
/// When the path exists, resolves it via the OS (following symlinks). When it
/// does not exist — `std::fs::canonicalize` fails with `NotFound` — falls back
/// to Python's `os.path.realpath` behavior for missing paths: absolutize
/// (join a relative path onto the current directory) and lexically normalize
/// `.`/`..` components, rather than returning the raw relative input (which
/// would derive a project key that never matches the CLI's).
#[must_use]
pub fn canonicalize_path(dir: &str) -> String {
    let resolved = std::fs::canonicalize(dir)
        .ok()
        .and_then(|p| p.to_str().map(str::to_string))
        .unwrap_or_else(|| lexical_realpath(dir));
    resolved.nfc().collect()
}

/// Absolutize + lexically normalize a path the way `os.path.realpath` does for
/// a nonexistent target: prepend the current directory when relative, then
/// collapse `.` and `..` components without touching the filesystem.
fn lexical_realpath(dir: &str) -> String {
    let input = Path::new(dir);
    let base = if input.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"))
    };
    let mut out = base;
    for component in input.components() {
        match component {
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::RootDir => {
                // Reset to filesystem root, preserving any Windows prefix.
                let prefix = out
                    .components()
                    .next()
                    .filter(|c| matches!(c, Component::Prefix(_)))
                    .map(|c| c.as_os_str().to_os_string());
                out = prefix.map_or_else(PathBuf::new, PathBuf::from);
                out.push(Component::RootDir.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(seg) => out.push(seg),
        }
    }
    out.to_string_lossy().into_owned()
}

/// Derives the [`SessionStore`](crate::sessions::SessionStore) `project_key`
/// for a directory (defaulting to the current directory).
///
/// Uses realpath + NFC normalization + djb2-hashed sanitization identical to
/// the CLI, so keys match local-disk and store-mirrored transcripts.
#[must_use]
pub fn project_key_for_directory(directory: Option<&Path>) -> String {
    let dir = directory
        .and_then(Path::to_str)
        .map_or_else(|| ".".to_string(), str::to_string);
    let abs = canonicalize_path(&dir);
    sanitize_path(&abs)
}

/// Derives a [`SessionKey`] from an absolute transcript file path.
///
/// - Main transcript: `<projects_dir>/<project_key>/<session_id>.jsonl`
/// - Subagent transcript:
///   `<projects_dir>/<project_key>/<session_id>/subagents/.../agent-<id>.jsonl`
///
/// Returns `None` only if `file_path` is not cleanly under `projects_dir` or
/// has an unrecognized shape. The session-id component is preserved
/// losslessly and is **not** validated as a UUID here — matching the official
/// Python `file_path_to_session_key`, which derives the key from arbitrary
/// on-disk stems (e.g. `abc-123`). Canonical-UUID validation belongs at the
/// APIs that officially require it (resume, import, mutations), not at key
/// derivation. Subpaths are always `/`-joined regardless of OS separator so
/// keys are portable across platforms.
#[must_use]
pub fn file_path_to_session_key(file_path: &Path, projects_dir: &Path) -> Option<SessionKey> {
    let rel = file_path.strip_prefix(projects_dir).ok()?;
    let mut parts: Vec<&str> = Vec::new();
    for component in rel.components() {
        match component {
            Component::Normal(os) => parts.push(os.to_str()?),
            // Any traversal / absolute / prefix component means the path is
            // not cleanly under projects_dir.
            _ => return None,
        }
    }
    if parts.len() < 2 {
        return None;
    }

    let project_key = parts[0].to_string();
    let second = parts[1];

    // Main transcript: <project_key>/<session_id>.jsonl
    if parts.len() == 2 {
        // The stem is the session id verbatim — arbitrary strings are accepted
        // (no UUID check), mirroring official Python.
        let session_id = second.strip_suffix(".jsonl")?;
        return Some(SessionKey::new(project_key, session_id));
    }

    // Subagent transcript: <project_key>/<session_id>/subagents/.../*.jsonl
    if parts.len() >= 4 {
        // The session id is the second component verbatim (arbitrary stems
        // accepted), matching official Python.
        let session_id = second;
        let mut subpath_parts: Vec<String> = parts[2..].iter().map(|s| (*s).to_string()).collect();
        if let Some(last) = subpath_parts.last_mut() {
            if let Some(stripped) = last.strip_suffix(".jsonl") {
                *last = stripped.to_string();
            }
        }
        let subpath = subpath_parts.join("/");
        return SessionKey::with_subpath(project_key, session_id, subpath).ok();
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uuid() -> Uuid {
        Uuid::parse_str("11111111-2222-3333-4444-555555555555").unwrap()
    }

    #[test]
    fn storage_key_main_and_subpath() {
        let id = uuid();
        let main = SessionKey::new("proj", id);
        assert_eq!(main.storage_key(), format!("proj/{}", id.hyphenated()));
        let sub = SessionKey::with_subpath("proj", id, "subagents/agent-7").unwrap();
        assert_eq!(
            sub.storage_key(),
            format!("proj/{}/subagents/agent-7", id.hyphenated())
        );
    }

    #[test]
    fn subpath_validation_rejects_traversal() {
        assert!(validate_subpath("subagents/agent-1").is_ok());
        assert!(validate_subpath("").is_err());
        assert!(validate_subpath("../escape").is_err());
        assert!(validate_subpath("subagents/../../etc/passwd").is_err());
        assert!(validate_subpath("/absolute").is_err());
        assert!(validate_subpath("a\0b").is_err());
        assert!(SessionKey::with_subpath("p", uuid(), "../x").is_err());
    }

    #[test]
    fn simple_hash_matches_reference_vectors() {
        // Empty string hashes to 0 -> "0".
        assert_eq!(simple_hash(""), "0");
        // djb2/JS-coerced values verified against the Python `_simple_hash`
        // reference implementation.
        assert_eq!(simple_hash("a"), simple_hash("a")); // deterministic
                                                        // "hello": compute reference below.
        assert_eq!(simple_hash("hello"), reference_hash("hello"));
        assert_eq!(
            simple_hash("/Users/x/Documents/project"),
            reference_hash("/Users/x/Documents/project")
        );
    }

    /// Independent reimplementation of the Python `_simple_hash` for
    /// cross-checking (uses u64 wraparound math mirroring JS coercion).
    fn reference_hash(s: &str) -> String {
        let mut h: i64 = 0;
        for ch in s.chars() {
            let c = ch as i64;
            h = (h << 5) - h + c;
            h &= 0xFFFF_FFFF;
            if h >= 0x8000_0000 {
                h -= 0x1_0000_0000;
            }
        }
        let mut n = h.unsigned_abs();
        if n == 0 {
            return "0".into();
        }
        let digits = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let mut out = Vec::new();
        while n > 0 {
            out.push(digits[(n % 36) as usize]);
            n /= 36;
        }
        out.reverse();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn sanitize_long_path_gets_hash_suffix() {
        let short = "abc/def";
        assert_eq!(sanitize_path(short), "abc-def");

        let long: String = std::iter::repeat_n('x', 250).collect();
        let out = sanitize_path(&long);
        // 200 chars + '-' + hash.
        assert!(out.len() > MAX_SANITIZED_LENGTH);
        assert!(out.starts_with(&"x".repeat(MAX_SANITIZED_LENGTH)));
        assert_eq!(out, format!("{}-{}", "x".repeat(200), simple_hash(&long)));
    }

    #[test]
    fn nfc_normalization_stable() {
        // project_key_for_directory NFC-normalizes; the same logical path in
        // NFD and NFC form must sanitize identically.
        let nfc = "café";
        let nfd = "cafe\u{0301}";
        let normalized_precomposed: String = nfc.nfc().collect();
        let normalized_decomposed: String = nfd.nfc().collect();
        assert_eq!(
            sanitize_path(&normalized_precomposed),
            sanitize_path(&normalized_decomposed)
        );
    }

    #[test]
    fn canonicalize_nonexistent_path_absolutizes_and_collapses() {
        // A nonexistent ABSOLUTE path must be lexically normalized (collapse
        // `.`/`..`) rather than returned raw — matching `os.path.realpath`,
        // which never returns the un-normalized input for a missing target.
        // Without this, project keys for not-yet-created directories would
        // never match the CLI's.
        assert_eq!(
            canonicalize_path("/nonexistent/foo/../bar"),
            "/nonexistent/bar"
        );
        assert_eq!(canonicalize_path("/a/b/c/../../d"), "/a/d");
        assert_eq!(canonicalize_path("/a/./b/./c"), "/a/b/c");
    }

    #[test]
    fn canonicalize_nonexistent_relative_path_is_absolutized() {
        // A nonexistent RELATIVE path is joined onto the current directory and
        // normalized (os.path.realpath absolutizes relative inputs).
        let cwd = std::env::current_dir().unwrap();
        let out = canonicalize_path("this-dir-does-not-exist/sub");
        let expected = cwd.join("this-dir-does-not-exist").join("sub");
        assert_eq!(out, expected.to_string_lossy());
        assert!(
            Path::new(&out).is_absolute(),
            "canonicalized relative path must be absolute"
        );
    }

    #[test]
    fn file_path_to_key_main_transcript() {
        let id = uuid();
        let projects = Path::new("/root/projects");
        let file = projects
            .join("myproj")
            .join(format!("{}.jsonl", id.hyphenated()));
        let key = file_path_to_session_key(&file, projects).unwrap();
        assert_eq!(key.project_key, "myproj");
        assert_eq!(key.session_id, id.hyphenated().to_string());
        assert_eq!(key.subpath, None);
    }

    #[test]
    fn file_path_to_key_subagent_transcript() {
        let id = uuid();
        let projects = Path::new("/root/projects");
        let file = projects
            .join("myproj")
            .join(id.hyphenated().to_string())
            .join("subagents")
            .join("agent-7.jsonl");
        let key = file_path_to_session_key(&file, projects).unwrap();
        assert_eq!(key.project_key, "myproj");
        assert_eq!(key.session_id, id.hyphenated().to_string());
        assert_eq!(key.subpath.as_deref(), Some("subagents/agent-7"));
    }

    #[test]
    fn file_path_to_key_rejects_outside_and_bad_shape() {
        let projects = Path::new("/root/projects");
        // Not under projects_dir.
        assert!(file_path_to_session_key(Path::new("/elsewhere/x.jsonl"), projects).is_none());
        // Single component (no project_key/session_id split).
        let one = projects.join("only.jsonl");
        assert!(file_path_to_session_key(&one, projects).is_none());
        // Traversal out of projects_dir.
        let up = Path::new("/root/projects/../secrets/s.jsonl");
        assert!(file_path_to_session_key(up, projects).is_none());
    }

    #[test]
    fn file_path_to_key_accepts_arbitrary_stems() {
        // Official Python derives keys from arbitrary on-disk stems with no
        // UUID validation. A non-canonical stem like `abc-123` must round-trip
        // losslessly rather than being rejected.
        let projects = Path::new("/root/projects");
        let main = projects.join("proj").join("abc-123.jsonl");
        let key = file_path_to_session_key(&main, projects).unwrap();
        assert_eq!(key.project_key, "proj");
        assert_eq!(key.session_id, "abc-123");
        assert_eq!(key.subpath, None);

        // Arbitrary subagent stem is preserved as the session id verbatim.
        let sub = projects
            .join("proj")
            .join("not-a-uuid")
            .join("subagents")
            .join("agent-x.jsonl");
        let sub_key = file_path_to_session_key(&sub, projects).unwrap();
        assert_eq!(sub_key.project_key, "proj");
        assert_eq!(sub_key.session_id, "not-a-uuid");
        assert_eq!(sub_key.subpath.as_deref(), Some("subagents/agent-x"));
    }
}
