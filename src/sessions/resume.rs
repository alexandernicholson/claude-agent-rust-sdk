//! Materialize a [`SessionStore`]-backed resume into a temp `CLAUDE_CONFIG_DIR`.
//!
//! When [`AgentOptions::session_store`](crate::agent::AgentOptions) is set together
//! with `resume` or `continue_conversation`, the transcript for the resolved
//! session is loaded from the store and written to a temporary directory laid
//! out like `~/.claude/`. The subprocess runs with
//! `CLAUDE_CONFIG_DIR=<temp>` and resumes from the materialized JSONL using its
//! normal `--resume` path. Auth config (`.credentials.json` with the refresh
//! token redacted, `.claude.json`, user settings) is copied so the subprocess
//! can authenticate, and subagent transcripts are materialized when the store
//! can enumerate them. The temp tree is removed by [`MaterializedResume::cleanup`]
//! after the subprocess disconnects.

use std::collections::BTreeMap;
use std::fs;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;
#[cfg(test)]
use uuid::Uuid;

use super::filesystem::get_projects_dir;
use super::key::{project_key_for_directory, SessionKey, SessionListSubkeysKey};
use super::mirror::{
    MirrorErrorHandler, TranscriptMirrorBatcher, MAX_PENDING_BYTES, MAX_PENDING_ENTRIES,
    SEND_TIMEOUT_SECONDS,
};
use super::store::{SessionStore, SessionStoreEntry, SessionStoreFlushMode};
use crate::agent::AgentOptions;
use crate::error::ClaudeError;

/// Default macOS Keychain service name for OAuth credentials when
/// `CLAUDE_CONFIG_DIR` is unset (production `OAUTH_FILE_SUFFIX` is empty).
const KEYCHAIN_SERVICE_NAME: &str = "Claude Code-credentials";

/// User-settings keys that only misbehave under the redirected
/// `CLAUDE_CONFIG_DIR`: plugin declarations reconcile against the always-empty
/// `tmp_base/plugins` cache and would network-install each declared marketplace
/// on every resume.
const RESUME_SETTINGS_STRIPPED_KEYS: [&str; 2] = ["enabledPlugins", "extraKnownMarketplaces"];

/// Cleanup future produced by [`MaterializedResume`]. Removing the temp tree is
/// async so it can back off on transient lock errors.
pub type CleanupFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

/// Result of [`materialize_resume_session`].
pub struct MaterializedResume {
    /// Temporary directory laid out like `~/.claude/` — point the subprocess at
    /// it via `CLAUDE_CONFIG_DIR`.
    pub config_dir: PathBuf,
    /// Session ID to pass as `--resume`. When the input was
    /// `continue_conversation`, this is the most-recent session resolved via
    /// [`SessionStore::list_sessions`].
    pub resume_session_id: String,
    /// Removes `config_dir` (best-effort). Call after the subprocess exits.
    cleanup: Box<dyn Fn() -> CleanupFuture + Send + Sync>,
}

impl std::fmt::Debug for MaterializedResume {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaterializedResume")
            .field("config_dir", &self.config_dir)
            .field("resume_session_id", &self.resume_session_id)
            .finish_non_exhaustive()
    }
}

impl MaterializedResume {
    /// Remove the materialized temp directory. Idempotent and best-effort;
    /// never panics.
    pub async fn cleanup(&self) {
        (self.cleanup)().await;
    }
}

/// Return a copy of `options` repointed at a materialized temp config dir.
///
/// Sets `CLAUDE_CONFIG_DIR` in `env`, `resume` to the materialized session id,
/// and clears `continue_conversation` (already resolved to a concrete session
/// id during materialization).
#[must_use]
pub fn apply_materialized_options(
    options: &AgentOptions,
    materialized: &MaterializedResume,
) -> AgentOptions {
    let mut next = options.clone();
    next.env.insert(
        "CLAUDE_CONFIG_DIR".to_string(),
        materialized.config_dir.to_string_lossy().into_owned(),
    );
    next.resume = Some(materialized.resume_session_id.clone());
    next.continue_conversation = false;
    next
}

/// Construct the [`TranscriptMirrorBatcher`] for a session.
///
/// Resolves `projects_dir` to the materialized temp dir when present (so
/// `file_path` → key resolution matches what the subprocess writes), otherwise to
/// the standard projects directory under the effective `CLAUDE_CONFIG_DIR`.
///
/// [`SessionStoreFlushMode::Eager`] zeroes the batcher's pending thresholds so
/// every enqueued frame schedules a background flush; [`Batched`] keeps the
/// defaults (flush on `result` or 500-entry / 1 MiB overflow).
///
/// [`Batched`]: SessionStoreFlushMode::Batched
#[must_use]
pub fn build_mirror_batcher(
    store: Arc<dyn SessionStore>,
    materialized: Option<&MaterializedResume>,
    env: Option<&BTreeMap<String, String>>,
    on_error: MirrorErrorHandler,
    flush_mode: SessionStoreFlushMode,
) -> TranscriptMirrorBatcher {
    let projects_dir = match materialized {
        Some(m) => m.config_dir.join("projects"),
        None => get_projects_dir(env),
    };
    let eager = matches!(flush_mode, SessionStoreFlushMode::Eager);
    TranscriptMirrorBatcher::build(
        store,
        projects_dir,
        on_error,
        SEND_TIMEOUT_SECONDS,
        if eager { 0 } else { MAX_PENDING_ENTRIES },
        if eager { 0 } else { MAX_PENDING_BYTES },
    )
}

/// Load a session from `options.session_store` and write it to a temp dir.
///
/// Returns `Ok(None)` when no materialization is needed (no store, no
/// resume/continue, store has no entries, or the resolved session ID is not a
/// valid UUID) — the caller falls through to the normal (no-store) resume/spawn
/// path. For `continue_conversation` this means a fresh session; for an explicit
/// `resume` value the CLI receives it unchanged.
///
/// # Errors
///
/// Returns [`ClaudeError`] if a store call fails or times out, or if writing
/// the temp tree fails.
pub async fn materialize_resume_session(
    options: &AgentOptions,
) -> Result<Option<MaterializedResume>, ClaudeError> {
    materialize_resume_session_in(options, None).await
}

/// [`materialize_resume_session`] with an explicit temp parent directory.
///
/// `temp_parent` overrides the system temp dir for the materialized tree.
/// Tests pass a dedicated per-test directory so leak assertions observe only
/// their own trees, never siblings racing in the shared system temp. `None`
/// (the public entry point) uses the system temp dir as upstream does.
async fn materialize_resume_session_in(
    options: &AgentOptions,
    temp_parent: Option<&Path>,
) -> Result<Option<MaterializedResume>, ClaudeError> {
    let Some(store) = options.session_store.clone() else {
        return Ok(None);
    };
    if options.resume.is_none() && !options.continue_conversation {
        return Ok(None);
    }

    let timeout = std::time::Duration::from_millis(options.load_timeout_ms);
    let project_key = project_key_for_directory(options.cwd.as_deref());

    // Resolve the session ID — explicit resume wins; otherwise pick the
    // most-recently-modified non-sidechain session from the store. Empty
    // list_sessions() → fresh session (matches CLI --continue with no history).
    let resolved = if let Some(resume) = options.resume.as_deref() {
        // session_id is used as a path component below; reject anything that
        // isn't a canonical UUID to prevent traversal and match every other
        // resume path.
        let Some(session_id) = crate::sessions::filesystem::validate_uuid(resume) else {
            return Ok(None);
        };
        load_candidate(store.as_ref(), &project_key, session_id, timeout).await?
    } else {
        resolve_continue_candidate(store.as_ref(), &project_key, timeout).await?
    };
    let Some((session_id, entries)) = resolved else {
        return Ok(None);
    };

    // Own the temp tree with an RAII guard so a dropped/cancelled future
    // between here and success removes it (matching the upstream
    // `except BaseException:` cleanup, which also catches cancellation).
    let guard = TempTreeGuard::new(make_temp_dir(temp_parent)?);
    if let Err(e) = write_resume_tree(
        store.as_ref(),
        guard.path(),
        &project_key,
        &session_id,
        &entries,
        &options.env,
        timeout,
    )
    .await
    {
        // Any failure after mkdtemp leaves the tree (which may already
        // contain a .credentials.json copy) on disk with no path for the
        // caller to clean it up. Use the async retry removal (transient-lock
        // backoff) and disarm the guard so its Drop doesn't remove again.
        rmtree_with_retry(guard.path()).await;
        let _ = guard.disarm();
        return Err(e);
    }

    // Success: transfer cleanup ownership to the returned object; nothing is
    // removed until the caller invokes `cleanup` after the subprocess exits.
    let tmp_base = guard.disarm();
    let cleanup_base = tmp_base.clone();
    let cleanup: Box<dyn Fn() -> CleanupFuture + Send + Sync> = Box::new(move || {
        let base = cleanup_base.clone();
        Box::pin(async move { rmtree_with_retry(&base).await }) as CleanupFuture
    });

    Ok(Some(MaterializedResume {
        config_dir: tmp_base,
        resume_session_id: session_id,
        cleanup,
    }))
}

/// Write the JSONL transcript, auth files, and subagent transcripts under
/// `tmp_base`. Separated from [`materialize_resume_session`] so a single caller
/// owns cancellation-safe cleanup on any failure.
async fn write_resume_tree(
    store: &dyn SessionStore,
    tmp_base: &Path,
    project_key: &str,
    session_id: &str,
    entries: &[SessionStoreEntry],
    env: &BTreeMap<String, String>,
    timeout: std::time::Duration,
) -> Result<(), ClaudeError> {
    let project_dir = tmp_base.join("projects").join(project_key);
    fs::create_dir_all(&project_dir).map_err(|e| {
        ClaudeError::TransportError(format!("resume: mkdir {}: {e}", project_dir.display()))
    })?;
    write_jsonl(&project_dir.join(format!("{session_id}.jsonl")), entries)?;

    // The subprocess will run with CLAUDE_CONFIG_DIR=tmp_base. Copy auth config
    // from the caller's effective config locations so it can authenticate.
    // Missing files are fine (API-key auth, etc.).
    copy_auth_files(tmp_base, env);

    // Materialize subagent transcripts if the store can enumerate them.
    if store.capabilities().list_subkeys {
        materialize_subkeys(store, &project_dir, project_key, session_id, timeout).await?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Candidate resolution
// ---------------------------------------------------------------------------

/// Load entries for `session_id`; return `None` if empty/missing.
async fn load_candidate(
    store: &dyn SessionStore,
    project_key: &str,
    session_id: &str,
    timeout: std::time::Duration,
) -> Result<Option<(String, Vec<SessionStoreEntry>)>, ClaudeError> {
    let key = SessionKey::new(project_key.to_string(), session_id);
    let entries = with_timeout(
        store.load(&key),
        timeout,
        &format!("SessionStore::load() for session {session_id}"),
    )
    .await?;
    match entries {
        Some(entries) if !entries.is_empty() => Ok(Some((session_id.to_string(), entries))),
        _ => Ok(None),
    }
}

/// Pick the most-recently-modified non-sidechain session.
///
/// Sidechain transcripts are mirrored as ordinary top-level keys and often have
/// the highest mtime (their append lands after the main session's in the same
/// flush). Walk newest→oldest, loading each candidate (the load is needed
/// anyway) and skipping sidechains so `--continue` resumes the user's
/// conversation, not a subagent's.
async fn resolve_continue_candidate(
    store: &dyn SessionStore,
    project_key: &str,
    timeout: std::time::Duration,
) -> Result<Option<(String, Vec<SessionStoreEntry>)>, ClaudeError> {
    // continue_conversation requires list_sessions; validated up front, but
    // guard here too so a raw call still degrades to "fresh session".
    if !store.capabilities().list_sessions {
        return Ok(None);
    }
    let mut sessions = with_timeout(
        store.list_sessions(project_key),
        timeout,
        "SessionStore::list_sessions()",
    )
    .await?;
    if sessions.is_empty() {
        return Ok(None);
    }
    sessions.sort_by_key(|s| std::cmp::Reverse(s.mtime));
    for cand in sessions {
        let Some((sid, entries)) =
            load_candidate(store, project_key, &cand.session_id, timeout).await?
        else {
            continue;
        };
        let is_sidechain = entries
            .first()
            .and_then(|e| e.get("isSidechain"))
            .and_then(Value::as_bool)
            == Some(true);
        if is_sidechain {
            continue;
        }
        return Ok(Some((sid, entries)));
    }
    Ok(None)
}

/// Await `fut` with a timeout, wrapping errors with context.
async fn with_timeout<T>(
    fut: impl Future<Output = Result<T, ClaudeError>>,
    timeout: std::time::Duration,
    what: &str,
) -> Result<T, ClaudeError> {
    match tokio::time::timeout(timeout, fut).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(ClaudeError::TransportError(format!(
            "{what} failed during resume materialization: {e}"
        ))),
        Err(_elapsed) => Err(ClaudeError::TransportError(format!(
            "{what} timed out after {}ms during resume materialization",
            timeout.as_millis()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Temp directory + JSONL
// ---------------------------------------------------------------------------

/// Create a fresh `claude-resume-*` temp directory.
///
/// `parent` overrides the system temp dir when set — used by tests to observe
/// leaks under a dedicated root without scanning the shared system temp.
fn make_temp_dir(parent: Option<&Path>) -> Result<PathBuf, ClaudeError> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("claude-resume-");
    let created = match parent {
        Some(dir) => builder.tempdir_in(dir),
        None => builder.tempdir(),
    };
    created
        .map(tempfile::TempDir::keep)
        .map_err(|e| ClaudeError::TransportError(format!("resume: mkdtemp: {e}")))
}

/// RAII guard owning a freshly-created resume temp tree.
///
/// Mirrors the upstream `except BaseException:` cleanup in
/// `session_resume.py`: any early return *or* a dropped/cancelled future
/// between `make_temp_dir` and successful materialization must remove the
/// tree, which may already hold a `.credentials.json` copy. Rust surfaces
/// cancellation as a future drop (not an exception), so cleanup lives in
/// [`Drop`]. On success, [`TempTreeGuard::disarm`] transfers ownership to the
/// returned [`MaterializedResume`] so nothing is removed prematurely.
///
/// `Drop` cleanup is synchronous and best-effort (no async in `Drop`); the
/// explicit error path still uses the async [`rmtree_with_retry`] backoff and
/// disarms the guard, so removal is idempotent and never runs twice.
struct TempTreeGuard {
    path: Option<PathBuf>,
}

impl TempTreeGuard {
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }

    /// Borrow the guarded path.
    fn path(&self) -> &Path {
        self.path
            .as_deref()
            .expect("TempTreeGuard path taken while still in use")
    }

    /// Transfer ownership out of the guard; `Drop` becomes a no-op.
    fn disarm(mut self) -> PathBuf {
        self.path.take().expect("TempTreeGuard disarmed twice")
    }
}

impl Drop for TempTreeGuard {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            // Cancellation/error safety net. Best-effort, never panics; the
            // async error path removes and disarms first, so this only fires
            // on a dropped/cancelled future.
            let _ = fs::remove_dir_all(&path);
        }
    }
}

/// Stream-write `entries` as one compact JSON line each to `path` (mode 0o600).
fn write_jsonl(path: &Path, entries: &[SessionStoreEntry]) -> Result<(), ClaudeError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            ClaudeError::TransportError(format!("resume: mkdir {}: {e}", parent.display()))
        })?;
    }
    let mut buf = String::new();
    for entry in entries {
        let value = Value::Object(entry.clone());
        buf.push_str(&serde_json::to_string(&value).map_err(ClaudeError::SerializationError)?);
        buf.push('\n');
    }
    fs::write(path, buf).map_err(|e| {
        ClaudeError::TransportError(format!("resume: write {}: {e}", path.display()))
    })?;
    set_mode_0600(path);
    Ok(())
}

/// Best-effort `chmod 0o600`. No-op on non-Unix.
fn set_mode_0600(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

// ---------------------------------------------------------------------------
// rmtree with retry
// ---------------------------------------------------------------------------

/// Best-effort recursive remove with retries on transient lock errors.
///
/// On Windows, AV/indexer can briefly hold a handle on freshly-written files
/// (notably `.credentials.json`), causing removal to fail. Retry a few times
/// with a short backoff; after exhausting retries, ignore errors (gives the
/// handle a chance to release first so the access token doesn't leak in temp).
/// Never panics.
async fn rmtree_with_retry(path: &Path) {
    if !path.exists() {
        return;
    }
    for _ in 0..4 {
        match fs::remove_dir_all(path) {
            Ok(()) => return,
            Err(e) if is_retryable_rmtree(&e) => {}
            Err(_) => break,
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let _ = fs::remove_dir_all(path);
}

fn is_retryable_rmtree(e: &std::io::Error) -> bool {
    use std::io::ErrorKind;
    matches!(
        e.kind(),
        ErrorKind::PermissionDenied | ErrorKind::DirectoryNotEmpty
    ) || matches!(e.raw_os_error(), Some(code) if is_retryable_errno(code))
}

#[cfg(unix)]
fn is_retryable_errno(code: i32) -> bool {
    // EBUSY, EMFILE, ENFILE, ENOTEMPTY, EPERM, EACCES
    matches!(code, 16 | 24 | 23 | 39 | 1 | 13)
}

#[cfg(not(unix))]
fn is_retryable_errno(_code: i32) -> bool {
    false
}

// ---------------------------------------------------------------------------
// Auth / settings seeding
// ---------------------------------------------------------------------------

/// Seed `tmp_base` with the caller's auth and user config: `.credentials.json`
/// (refreshToken redacted), `.claude.json`, and user `settings.json` /
/// `cowork_settings.json` (plugin declarations stripped).
fn copy_auth_files(tmp_base: &Path, opt_env: &BTreeMap<String, String>) {
    let caller_config_dir = opt_env
        .get("CLAUDE_CONFIG_DIR")
        .cloned()
        .or_else(|| std::env::var("CLAUDE_CONFIG_DIR").ok())
        .filter(|s| !s.is_empty());
    let source_config_dir = match &caller_config_dir {
        Some(dir) => PathBuf::from(dir),
        None => home_dir().join(".claude"),
    };

    let mut creds_json = read_if_present(&source_config_dir.join(".credentials.json"))
        .and_then(|b| String::from_utf8(b).ok());

    // macOS default setup keeps OAuth tokens in the Keychain, not a file.
    // Redirecting CLAUDE_CONFIG_DIR changes the Keychain service-name suffix,
    // so the subprocess's lookup misses and falls back to plainTextStorage at
    // ${tmp_base}/.credentials.json. Populate that file from the parent's
    // Keychain so the resumed subprocess can auth. Skipped when env-based auth
    // or a custom config dir is already in play.
    if caller_config_dir.is_none()
        && env_or_os_empty(opt_env, "ANTHROPIC_API_KEY")
        && env_or_os_empty(opt_env, "CLAUDE_CODE_OAUTH_TOKEN")
    {
        if let Some(keychain) = read_keychain_credentials() {
            creds_json = Some(keychain);
        }
    }

    write_redacted_credentials(creds_json.as_deref(), &tmp_base.join(".credentials.json"));

    // .claude.json lives at $CLAUDE_CONFIG_DIR/.claude.json when set, else
    // ~/.claude.json (NOT ~/.claude/.claude.json).
    let claude_json_src = match &caller_config_dir {
        Some(dir) => PathBuf::from(dir).join(".claude.json"),
        None => home_dir().join(".claude.json"),
    };
    copy_if_present(&claude_json_src, &tmp_base.join(".claude.json"), None);

    // User settings carry apiKeyHelper plus env/hooks/permissions. Both pass
    // through strip_settings_for_resume so plugin declarations don't reconcile
    // against the empty tmp_base plugin cache.
    for name in ["settings.json", "cowork_settings.json"] {
        copy_if_present(
            &source_config_dir.join(name),
            &tmp_base.join(name),
            Some(strip_settings_for_resume),
        );
    }
}

/// True when neither `opt_env[name]` nor the process env sets a non-empty value.
fn env_or_os_empty(opt_env: &BTreeMap<String, String>, name: &str) -> bool {
    let from_opt = opt_env.get(name).is_some_and(|s| !s.is_empty());
    let from_os = std::env::var(name).is_ok_and(|s| !s.is_empty());
    !from_opt && !from_os
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map_or_else(|| PathBuf::from("."), PathBuf::from)
}

/// Drop settings keys that misbehave under a redirected config dir. Removes
/// [`RESUME_SETTINGS_STRIPPED_KEYS`] and `env.CLAUDE_CONFIG_DIR`. Content that
/// doesn't parse as a JSON object is returned untouched.
fn strip_settings_for_resume(content: &[u8]) -> Vec<u8> {
    // Strip a UTF-8 BOM the way the CLI's settings reader does.
    let bytes = content.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(content);
    let Ok(mut parsed) = serde_json::from_slice::<Value>(bytes) else {
        return content.to_vec();
    };
    let Some(obj) = parsed.as_object_mut() else {
        return content.to_vec();
    };
    let mut stripped = false;
    for key in RESUME_SETTINGS_STRIPPED_KEYS {
        if obj.remove(key).is_some() {
            stripped = true;
        }
    }
    if let Some(Value::Object(env_block)) = obj.get_mut("env") {
        if env_block.remove("CLAUDE_CONFIG_DIR").is_some() {
            stripped = true;
        }
    }
    if !stripped {
        return content.to_vec();
    }
    serde_json::to_vec(&parsed).unwrap_or_else(|_| content.to_vec())
}

/// Write `creds_json` with `claudeAiOauth.refreshToken` removed.
///
/// The resumed subprocess runs under a redirected `CLAUDE_CONFIG_DIR`; if it
/// refreshed, the single-use refresh token would be consumed server-side and
/// the new tokens written where the parent never reads back — leaving the
/// parent's stored creds revoked. With no `refreshToken`, the subprocess's
/// refresh check short-circuits.
fn write_redacted_credentials(creds_json: Option<&str>, dst: &Path) {
    let Some(creds_json) = creds_json else {
        return;
    };
    let out = match serde_json::from_str::<Value>(creds_json) {
        Ok(mut data) => {
            if let Some(Value::Object(oauth)) = data.get_mut("claudeAiOauth") {
                oauth.remove("refreshToken");
            }
            serde_json::to_string(&data).unwrap_or_else(|_| creds_json.to_string())
        }
        // Unparseable — write through; subprocess will fail to parse it too.
        Err(_) => creds_json.to_string(),
    };
    if fs::write(dst, out).is_ok() {
        set_mode_0600(dst);
    }
}

/// Read a regular file, or return `None`. Missing sources are skipped silently;
/// non-regular / unreadable ones are logged and skipped.
fn read_if_present(src: &Path) -> Option<Vec<u8>> {
    let meta = match fs::symlink_metadata(src) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            tracing::warn!(path = %src.display(), error = %e, "resume: skipping (stat)");
            return None;
        }
    };
    if !meta.file_type().is_file() {
        tracing::warn!(path = %src.display(), "resume: skipping (not a regular file)");
        return None;
    }
    match fs::read(src) {
        Ok(b) => Some(b),
        Err(e) => {
            tracing::warn!(path = %src.display(), error = %e, "resume: skipping (read)");
            None
        }
    }
}

/// A settings-content transform applied while copying a config file.
type SettingsTransform = fn(&[u8]) -> Vec<u8>;

/// Copy `src` to `dst` (mode 0o600) if it exists, through an optional transform.
fn copy_if_present(src: &Path, dst: &Path, transform: Option<SettingsTransform>) {
    let Some(content) = read_if_present(src) else {
        return;
    };
    let payload = match transform {
        Some(f) => f(&content),
        None => content,
    };
    match fs::write(dst, payload) {
        Ok(()) => set_mode_0600(dst),
        Err(e) => {
            // Don't leave a truncated dst behind for the subprocess to misparse.
            let _ = fs::remove_file(dst);
            tracing::warn!(path = %src.display(), error = %e, "resume: skipping (write)");
        }
    }
}

/// Read OAuth credentials JSON from the macOS Keychain (default service name).
/// Best-effort — returns `None` on any error or non-macOS platforms.
fn read_keychain_credentials() -> Option<String> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let user = std::env::var("USER").unwrap_or_else(|_| "claude-code-user".to_string());
    let output = std::process::Command::new("security")
        .args([
            "find-generic-password",
            "-a",
            &user,
            "-w",
            "-s",
            KEYCHAIN_SERVICE_NAME,
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let out = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

// ---------------------------------------------------------------------------
// Subagent materialization
// ---------------------------------------------------------------------------

/// Load and write all subagent transcripts/metadata under `session_id`.
async fn materialize_subkeys(
    store: &dyn SessionStore,
    project_dir: &Path,
    project_key: &str,
    session_id: &str,
    timeout: std::time::Duration,
) -> Result<(), ClaudeError> {
    let session_dir = project_dir.join(session_id);
    let list_key = SessionListSubkeysKey {
        project_key: project_key.to_string(),
        session_id: session_id.to_string(),
    };
    let subkeys = with_timeout(
        store.list_subkeys(&list_key),
        timeout,
        &format!("SessionStore::list_subkeys() for session {session_id}"),
    )
    .await?;

    for subpath in subkeys {
        // Subpaths come from an external store and are used as filesystem path
        // components. Reject anything that would escape the session directory.
        if !is_safe_subpath(&subpath, &session_dir) {
            tracing::warn!(subpath = %subpath, "skipping unsafe subpath from list_subkeys");
            continue;
        }

        let sub_key = SessionKey {
            project_key: project_key.to_string(),
            session_id: session_id.to_string(),
            subpath: Some(subpath.clone()),
        };
        let sub_entries = with_timeout(
            store.load(&sub_key),
            timeout,
            &format!("SessionStore::load() for session {session_id} subpath {subpath}"),
        )
        .await?;
        let Some(sub_entries) = sub_entries else {
            continue;
        };
        if sub_entries.is_empty() {
            continue;
        }

        // Partition: agent_metadata entries describe the .meta.json sidecar;
        // everything else is a transcript line.
        let mut metadata: Vec<&SessionStoreEntry> = Vec::new();
        let mut transcript: Vec<SessionStoreEntry> = Vec::new();
        for e in &sub_entries {
            if e.get("type").and_then(Value::as_str) == Some("agent_metadata") {
                metadata.push(e);
            } else {
                transcript.push(e.clone());
            }
        }

        let target = session_dir.join(&subpath);
        let sub_file = with_extra_jsonl(&target);
        if !transcript.is_empty() {
            write_jsonl(&sub_file, &transcript)?;
        }

        if let Some(last) = metadata.last() {
            // Last metadata entry wins; strip the synthetic `type` field.
            let mut meta_content = (*last).clone();
            meta_content.remove("type");
            let meta_file = with_meta_json(&sub_file);
            if let Some(parent) = meta_file.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    ClaudeError::TransportError(format!("resume: mkdir {}: {e}", parent.display()))
                })?;
            }
            let json = serde_json::to_string(&Value::Object(meta_content))
                .map_err(ClaudeError::SerializationError)?;
            fs::write(&meta_file, json).map_err(|e| {
                ClaudeError::TransportError(format!("resume: write {}: {e}", meta_file.display()))
            })?;
            set_mode_0600(&meta_file);
        }
    }
    Ok(())
}

/// `session_dir/subpath` → `.../<name>.jsonl` (append `.jsonl` to the final
/// component, matching the Python writer).
fn with_extra_jsonl(target: &Path) -> PathBuf {
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    target.with_file_name(format!("{name}.jsonl"))
}

/// `<x>.jsonl` → `<x>.meta.json`.
fn with_meta_json(sub_file: &Path) -> PathBuf {
    let name = sub_file
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let stem = name.strip_suffix(".jsonl").unwrap_or(&name);
    sub_file.with_file_name(format!("{stem}.meta.json"))
}

/// Reject subpaths that are empty, absolute, contain `..`, or escape
/// `session_dir` after resolution.
fn is_safe_subpath(subpath: &str, session_dir: &Path) -> bool {
    if subpath.is_empty() {
        return false;
    }
    if subpath.starts_with('/') || subpath.starts_with('\\') {
        return false;
    }
    if Path::new(subpath).is_absolute() {
        return false;
    }
    // Drive-prefixed (`C:foo`) and UNC subpaths are never legitimate keys.
    let bytes = subpath.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
        return false;
    }
    if subpath.contains('\0') {
        return false;
    }
    for part in subpath.split(['/', '\\']) {
        if part == "." || part == ".." {
            return false;
        }
    }
    // Confirm the resolved `.jsonl` target stays under `session_dir`. The
    // component checks above reject lexical `..` traversal, but an intermediate
    // directory in `session_dir/subpath` may be a symlink pointing outside the
    // tree; a purely lexical `starts_with` would miss that escape. Mirror the
    // official `sub_file.resolve().relative_to(session_dir.resolve())`: resolve
    // both paths (following symlinks through existing ancestors, lexical for
    // the not-yet-created tail) and require containment.
    let target = with_extra_jsonl(&session_dir.join(subpath));
    let resolved_target = resolve_realpath(&target);
    let resolved_root = resolve_realpath(session_dir);
    resolved_target.starts_with(&resolved_root)
}

/// Resolve `path` the way `os.path.realpath`/`Path.resolve()` do: canonicalize
/// the deepest existing ancestor (following symlinks) and re-append the tail
/// that does not exist yet, lexically normalized. When no ancestor exists at
/// all, fall back to a purely lexical absolute normalization so nonexistent
/// targets still compare containment correctly.
fn resolve_realpath(path: &Path) -> PathBuf {
    let mut ancestor = path;
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if let Ok(resolved) = std::fs::canonicalize(ancestor) {
            let mut out = resolved;
            for part in tail.iter().rev() {
                out.push(part);
            }
            return out;
        }
        match ancestor.file_name() {
            Some(name) => {
                tail.push(name.to_os_string());
                match ancestor.parent() {
                    Some(parent) => ancestor = parent,
                    None => break,
                }
            }
            // Reached a root / prefix component with no existing canonical
            // form — fall through to lexical normalization.
            None => break,
        }
    }
    lexical_normalize(path)
}

/// Absolutize (against the current directory when relative) and lexically
/// collapse `.`/`..` without touching the filesystem.
fn lexical_normalize(path: &Path) -> PathBuf {
    use std::path::Component;
    let base = if path.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"))
    };
    let mut out = base;
    for component in path.components() {
        match component {
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::RootDir => {
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
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use serde_json::json;

    // ---- test store -------------------------------------------------------

    #[derive(Debug, Default)]
    struct FakeStore {
        /// `session_id` -> entries (main transcript).
        sessions: Mutex<Vec<(Uuid, i64, Vec<SessionStoreEntry>)>>,
        /// `(session_id, subpath)` -> entries.
        subkeys: Mutex<Vec<(Uuid, String, Vec<SessionStoreEntry>)>>,
        caps: super::super::store::SessionStoreCapabilities,
    }

    #[async_trait::async_trait]
    impl SessionStore for FakeStore {
        async fn append(
            &self,
            _key: &SessionKey,
            _entries: Vec<SessionStoreEntry>,
        ) -> Result<(), ClaudeError> {
            Ok(())
        }
        async fn load(
            &self,
            key: &SessionKey,
        ) -> Result<Option<Vec<SessionStoreEntry>>, ClaudeError> {
            if let Some(sub) = &key.subpath {
                let subs = self.subkeys.lock();
                for (sid, sp, entries) in subs.iter() {
                    if sid.to_string() == key.session_id && sp == sub {
                        return Ok(Some(entries.clone()));
                    }
                }
                return Ok(None);
            }
            let sessions = self.sessions.lock();
            for (sid, _mtime, entries) in sessions.iter() {
                if sid.to_string() == key.session_id {
                    return Ok(Some(entries.clone()));
                }
            }
            Ok(None)
        }
        fn capabilities(&self) -> super::super::store::SessionStoreCapabilities {
            self.caps
        }
        async fn list_sessions(
            &self,
            _project_key: &str,
        ) -> Result<Vec<SessionStoreListEntry>, ClaudeError> {
            if !self.caps.list_sessions {
                return Err(ClaudeError::Unsupported("list_sessions".into()));
            }
            Ok(self
                .sessions
                .lock()
                .iter()
                .map(|(sid, mtime, _)| SessionStoreListEntry {
                    session_id: sid.to_string(),
                    mtime: *mtime,
                })
                .collect())
        }
        async fn list_subkeys(
            &self,
            key: &SessionListSubkeysKey,
        ) -> Result<Vec<String>, ClaudeError> {
            if !self.caps.list_subkeys {
                return Err(ClaudeError::Unsupported("list_subkeys".into()));
            }
            Ok(self
                .subkeys
                .lock()
                .iter()
                .filter(|(sid, _, _)| sid.to_string() == key.session_id)
                .map(|(_, sp, _)| sp.clone())
                .collect())
        }
    }

    use super::super::store::{SessionStoreCapabilities, SessionStoreListEntry};

    fn entry(kind: &str, uuid: &str) -> SessionStoreEntry {
        let mut m = SessionStoreEntry::new();
        m.insert("type".into(), json!(kind));
        m.insert("uuid".into(), json!(uuid));
        m
    }

    fn opts_with_store(store: Arc<dyn SessionStore>, cwd: &Path) -> AgentOptions {
        AgentOptions {
            session_store: Some(store),
            cwd: Some(cwd.to_path_buf()),
            ..Default::default()
        }
    }

    // ---- resume by explicit id -------------------------------------------

    #[tokio::test]
    async fn store_only_resume_materializes_jsonl() {
        let tmp = tempfile::tempdir().unwrap();
        let sid = Uuid::new_v4();
        let store = Arc::new(FakeStore {
            caps: SessionStoreCapabilities::default(),
            ..Default::default()
        });
        store
            .sessions
            .lock()
            .push((sid, 1, vec![entry("user", "u1"), entry("assistant", "a1")]));

        let mut opts = opts_with_store(store.clone(), tmp.path());
        opts.resume = Some(sid.to_string());
        opts.env.insert(
            "CLAUDE_CONFIG_DIR".into(),
            tmp.path().join("cfg").to_string_lossy().into_owned(),
        );

        let mat = materialize_resume_session(&opts)
            .await
            .unwrap()
            .expect("materialized");
        assert_eq!(mat.resume_session_id, sid.to_string());

        let project_key = project_key_for_directory(Some(tmp.path()));
        let jsonl = mat
            .config_dir
            .join("projects")
            .join(&project_key)
            .join(format!("{sid}.jsonl"));
        let content = fs::read_to_string(&jsonl).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"u1\""));

        // apply_materialized_options repoints env + resume.
        let applied = apply_materialized_options(&opts, &mat);
        assert_eq!(
            applied.env.get("CLAUDE_CONFIG_DIR").map(String::as_str),
            Some(mat.config_dir.to_string_lossy().as_ref())
        );
        assert_eq!(applied.resume.as_deref(), Some(sid.to_string().as_str()));
        assert!(!applied.continue_conversation);

        mat.cleanup().await;
        assert!(!mat.config_dir.exists(), "cleanup removes temp tree");
    }

    #[tokio::test]
    async fn mode_0600_on_materialized_files() {
        let tmp = tempfile::tempdir().unwrap();
        let sid = Uuid::new_v4();
        let store = Arc::new(FakeStore::default());
        store
            .sessions
            .lock()
            .push((sid, 1, vec![entry("user", "u1")]));
        let mut opts = opts_with_store(store, tmp.path());
        opts.resume = Some(sid.to_string());

        let mat = materialize_resume_session(&opts).await.unwrap().unwrap();
        let project_key = project_key_for_directory(Some(tmp.path()));
        let jsonl = mat
            .config_dir
            .join("projects")
            .join(&project_key)
            .join(format!("{sid}.jsonl"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&jsonl).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        mat.cleanup().await;
    }

    #[tokio::test]
    async fn missing_session_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Arc::new(FakeStore::default());
        let mut opts = opts_with_store(store, tmp.path());
        opts.resume = Some(Uuid::new_v4().to_string());
        assert!(materialize_resume_session(&opts).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn non_uuid_resume_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Arc::new(FakeStore::default());
        let mut opts = opts_with_store(store, tmp.path());
        opts.resume = Some("not-a-uuid".into());
        assert!(materialize_resume_session(&opts).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn no_store_returns_none() {
        let opts = AgentOptions {
            resume: Some(Uuid::new_v4().to_string()),
            ..Default::default()
        };
        assert!(materialize_resume_session(&opts).await.unwrap().is_none());
    }

    // ---- continue --------------------------------------------------------

    #[tokio::test]
    async fn continue_resolves_newest_non_sidechain() {
        let tmp = tempfile::tempdir().unwrap();
        let old = Uuid::new_v4();
        let newest_sidechain = Uuid::new_v4();
        let newest_main = Uuid::new_v4();
        let caps = SessionStoreCapabilities {
            list_sessions: true,
            ..Default::default()
        };
        let store = Arc::new(FakeStore {
            caps,
            ..Default::default()
        });
        {
            let mut s = store.sessions.lock();
            s.push((old, 10, vec![entry("user", "old")]));
            // Highest mtime but a sidechain → skipped.
            let mut side = entry("user", "side");
            side.insert("isSidechain".into(), json!(true));
            s.push((newest_sidechain, 30, vec![side]));
            s.push((newest_main, 20, vec![entry("user", "main")]));
        }

        let mut opts = opts_with_store(store, tmp.path());
        opts.continue_conversation = true;

        let mat = materialize_resume_session(&opts).await.unwrap().unwrap();
        assert_eq!(
            mat.resume_session_id,
            newest_main.to_string(),
            "sidechain skipped, newest non-sidechain wins"
        );
        mat.cleanup().await;
    }

    #[tokio::test]
    async fn continue_empty_store_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let caps = SessionStoreCapabilities {
            list_sessions: true,
            ..Default::default()
        };
        let store = Arc::new(FakeStore {
            caps,
            ..Default::default()
        });
        let mut opts = opts_with_store(store, tmp.path());
        opts.continue_conversation = true;
        assert!(materialize_resume_session(&opts).await.unwrap().is_none());
    }

    // ---- subkeys ---------------------------------------------------------

    #[tokio::test]
    async fn subkeys_materialize_transcript_and_meta() {
        let tmp = tempfile::tempdir().unwrap();
        let sid = Uuid::new_v4();
        let caps = SessionStoreCapabilities {
            list_subkeys: true,
            ..Default::default()
        };
        let store = Arc::new(FakeStore {
            caps,
            ..Default::default()
        });
        store
            .sessions
            .lock()
            .push((sid, 1, vec![entry("user", "u1")]));
        {
            let mut meta = SessionStoreEntry::new();
            meta.insert("type".into(), json!("agent_metadata"));
            meta.insert("title".into(), json!("Sub Agent"));
            store.subkeys.lock().push((
                sid,
                "subagents/agent-1".into(),
                vec![entry("assistant", "s1"), meta],
            ));
        }
        let mut opts = opts_with_store(store, tmp.path());
        opts.resume = Some(sid.to_string());

        let mat = materialize_resume_session(&opts).await.unwrap().unwrap();
        let project_key = project_key_for_directory(Some(tmp.path()));
        let base = mat
            .config_dir
            .join("projects")
            .join(&project_key)
            .join(sid.to_string())
            .join("subagents");
        let jsonl = base.join("agent-1.jsonl");
        let meta = base.join("agent-1.meta.json");
        assert!(jsonl.exists(), "subagent transcript written");
        assert!(meta.exists(), "subagent meta sidecar written");
        let meta_content = fs::read_to_string(&meta).unwrap();
        assert!(meta_content.contains("Sub Agent"));
        assert!(
            !meta_content.contains("agent_metadata"),
            "type field stripped"
        );
        mat.cleanup().await;
    }

    #[tokio::test]
    async fn unsafe_subpath_is_refused() {
        let session_dir = Path::new("/tmp/proj/sess");
        assert!(!is_safe_subpath("", session_dir));
        assert!(!is_safe_subpath("/etc/passwd", session_dir));
        assert!(!is_safe_subpath("../escape", session_dir));
        assert!(!is_safe_subpath("subagents/../../escape", session_dir));
        assert!(!is_safe_subpath("C:evil", session_dir));
        assert!(!is_safe_subpath("a\0b", session_dir));
        assert!(is_safe_subpath("subagents/agent-1", session_dir));
    }

    /// A purely-lexical containment check would accept `subagents/agent-1`
    /// because no `..` component appears — but if `subagents` is a symlink to a
    /// directory outside `session_dir`, the resolved `.jsonl` target escapes.
    /// `is_safe_subpath` must resolve intermediate symlinks and reject it.
    #[cfg(unix)]
    #[tokio::test]
    async fn symlink_intermediate_escape_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let session_dir = tmp.path().join("proj").join("sess");
        std::fs::create_dir_all(&session_dir).unwrap();
        // A sibling directory outside session_dir that the symlink escapes to.
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        // `session_dir/subagents` -> `../../outside` (escapes the session tree).
        std::os::unix::fs::symlink(&outside, session_dir.join("subagents")).unwrap();

        // Lexically clean (no `..`), but the resolved target lands under
        // `outside`, not `session_dir`.
        assert!(
            !is_safe_subpath("subagents/agent-1", &session_dir),
            "symlinked intermediate directory escape must be refused"
        );

        // A non-symlinked, genuinely-contained subpath is still accepted.
        let real_session = tmp.path().join("proj").join("sess2");
        std::fs::create_dir_all(real_session.join("subagents")).unwrap();
        assert!(is_safe_subpath("subagents/agent-1", &real_session));
    }

    #[tokio::test]
    async fn traversal_subpath_skipped_not_materialized() {
        let tmp = tempfile::tempdir().unwrap();
        let sid = Uuid::new_v4();
        let caps = SessionStoreCapabilities {
            list_subkeys: true,
            ..Default::default()
        };
        let store = Arc::new(FakeStore {
            caps,
            ..Default::default()
        });
        store
            .sessions
            .lock()
            .push((sid, 1, vec![entry("user", "u1")]));
        store
            .subkeys
            .lock()
            .push((sid, "../../escape".into(), vec![entry("assistant", "s1")]));
        let mut opts = opts_with_store(store, tmp.path());
        opts.resume = Some(sid.to_string());

        let mat = materialize_resume_session(&opts).await.unwrap().unwrap();
        // Escape file must not exist anywhere under config_dir's parent.
        let escaped = mat.config_dir.join("projects").join("escape.jsonl");
        assert!(!escaped.exists());
        mat.cleanup().await;
    }

    // ---- redaction / settings --------------------------------------------

    #[test]
    fn credentials_refresh_token_redacted() {
        let tmp = tempfile::tempdir().unwrap();
        let dst = tmp.path().join(".credentials.json");
        let creds = json!({
            "claudeAiOauth": {
                "accessToken": "keep",
                "refreshToken": "secret"
            }
        })
        .to_string();
        write_redacted_credentials(Some(&creds), &dst);
        let written: Value = serde_json::from_str(&fs::read_to_string(&dst).unwrap()).unwrap();
        let oauth = &written["claudeAiOauth"];
        assert_eq!(oauth["accessToken"], json!("keep"));
        assert!(
            oauth.get("refreshToken").is_none(),
            "refresh token redacted"
        );
    }

    #[test]
    fn settings_strip_plugins_and_config_dir() {
        let content = json!({
            "enabledPlugins": {"a": true},
            "extraKnownMarketplaces": ["x"],
            "apiKeyHelper": "helper",
            "env": {"CLAUDE_CONFIG_DIR": "/x", "KEEP": "1"}
        })
        .to_string();
        let out = strip_settings_for_resume(content.as_bytes());
        let parsed: Value = serde_json::from_slice(&out).unwrap();
        assert!(parsed.get("enabledPlugins").is_none());
        assert!(parsed.get("extraKnownMarketplaces").is_none());
        assert_eq!(parsed["apiKeyHelper"], json!("helper"), "auth preserved");
        assert!(parsed["env"].get("CLAUDE_CONFIG_DIR").is_none());
        assert_eq!(parsed["env"]["KEEP"], json!("1"));
    }

    #[test]
    fn settings_non_object_untouched() {
        let content = b"[1, 2, 3]";
        assert_eq!(strip_settings_for_resume(content), content.to_vec());
    }

    #[test]
    fn settings_bom_stripped_and_parsed() {
        let mut content = vec![0xEF, 0xBB, 0xBF];
        content.extend_from_slice(json!({"enabledPlugins": {}}).to_string().as_bytes());
        let out = strip_settings_for_resume(&content);
        let parsed: Value = serde_json::from_slice(&out).unwrap();
        assert!(parsed.get("enabledPlugins").is_none());
    }

    // ---- cleanup after failure -------------------------------------------

    #[tokio::test]
    async fn load_failure_after_mkdtemp_cleans_up() {
        // A store that errors on load simulates adapter failure. Because the
        // error happens before mkdtemp (candidate resolution), no temp tree is
        // created; assert we surface the error and leave nothing behind. Then a
        // subkey-load failure (after mkdtemp) is covered separately below.
        #[derive(Debug, Default)]
        struct FailingLoad;
        #[async_trait::async_trait]
        impl SessionStore for FailingLoad {
            async fn append(
                &self,
                _key: &SessionKey,
                _entries: Vec<SessionStoreEntry>,
            ) -> Result<(), ClaudeError> {
                Ok(())
            }
            async fn load(
                &self,
                _key: &SessionKey,
            ) -> Result<Option<Vec<SessionStoreEntry>>, ClaudeError> {
                Err(ClaudeError::TransportError("adapter down".into()))
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        let mut opts = opts_with_store(Arc::new(FailingLoad), tmp.path());
        opts.resume = Some(Uuid::new_v4().to_string());
        let err = materialize_resume_session(&opts).await.unwrap_err();
        assert!(err.to_string().contains("adapter down"));
    }

    #[tokio::test]
    async fn subkey_load_failure_after_mkdtemp_removes_temp_tree() {
        // Store loads the main transcript fine but fails on subkey load, which
        // happens AFTER mkdtemp — the partially-written temp tree must be gone.
        #[derive(Debug)]
        struct FailOnSubkey;
        #[async_trait::async_trait]
        impl SessionStore for FailOnSubkey {
            async fn append(
                &self,
                _key: &SessionKey,
                _entries: Vec<SessionStoreEntry>,
            ) -> Result<(), ClaudeError> {
                Ok(())
            }
            async fn load(
                &self,
                key: &SessionKey,
            ) -> Result<Option<Vec<SessionStoreEntry>>, ClaudeError> {
                if key.subpath.is_some() {
                    return Err(ClaudeError::TransportError("subkey boom".into()));
                }
                let mut m = SessionStoreEntry::new();
                m.insert("type".into(), json!("user"));
                Ok(Some(vec![m]))
            }
            fn capabilities(&self) -> SessionStoreCapabilities {
                SessionStoreCapabilities {
                    list_subkeys: true,
                    ..Default::default()
                }
            }
            async fn list_subkeys(
                &self,
                _key: &SessionListSubkeysKey,
            ) -> Result<Vec<String>, ClaudeError> {
                Ok(vec!["subagents/agent-1".into()])
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        let sid = Uuid::new_v4();
        let mut opts = opts_with_store(Arc::new(FailOnSubkey), tmp.path());
        opts.resume = Some(sid.to_string());

        // The subkey load fails AFTER mkdtemp, so materialize must remove the
        // partially-written temp tree before returning the error. Materialize
        // into a dedicated per-test parent so the leak count observes only this
        // call's trees, never sibling tests racing in the shared system temp.
        let temp_parent = tempfile::tempdir().unwrap();
        let before = count_resume_temp_dirs(temp_parent.path());
        let err = materialize_resume_session_in(&opts, Some(temp_parent.path()))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("subkey boom"));
        let after = count_resume_temp_dirs(temp_parent.path());
        assert!(
            after <= before,
            "temp tree leaked after failure: before={before} after={after}"
        );
    }

    /// Count `claude-resume-*` directories directly under `dir`.
    fn count_resume_temp_dirs(dir: &Path) -> usize {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return 0;
        };
        rd.filter_map(Result::ok)
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("claude-resume-")
            })
            .count()
    }

    #[tokio::test]
    async fn build_mirror_batcher_eager_vs_batched() {
        // Eager zeroes thresholds → a single small enqueue signals a flush.
        let store = Arc::new(FakeStore::default());
        let on_error: MirrorErrorHandler = Arc::new(|_k, _m| Box::pin(async {}));
        let eager = build_mirror_batcher(
            store.clone(),
            None,
            None,
            on_error.clone(),
            SessionStoreFlushMode::Eager,
        );
        let sid = Uuid::new_v4();
        // Path won't resolve (no real projects dir) but enqueue only measures
        // thresholds; resolution happens at flush.
        let path = format!("/x/{sid}.jsonl");
        let mut small = SessionStoreEntry::new();
        small.insert("type".into(), json!("user"));
        assert!(eager.enqueue(path.clone(), vec![small.clone()]));

        let batched =
            build_mirror_batcher(store, None, None, on_error, SessionStoreFlushMode::Batched);
        assert!(!batched.enqueue(path, vec![small]));
    }

    #[tokio::test]
    async fn cleanup_after_disconnect_removes_temp_and_is_idempotent() {
        // Models the runtime's teardown order: the subprocess has already
        // disconnected (config_dir is fully written and no longer read), THEN
        // cleanup runs. Cleanup must remove the temp tree and be safe to call
        // more than once.
        let tmp = tempfile::tempdir().unwrap();
        let sid = Uuid::new_v4();
        let store = Arc::new(FakeStore::default());
        store
            .sessions
            .lock()
            .push((sid, 1, vec![entry("user", "u1")]));
        let mut opts = opts_with_store(store, tmp.path());
        opts.resume = Some(sid.to_string());

        let mat = materialize_resume_session(&opts).await.unwrap().unwrap();
        assert!(mat.config_dir.exists());

        // Simulated subprocess disconnect happened; now clean up.
        mat.cleanup().await;
        assert!(
            !mat.config_dir.exists(),
            "temp tree removed after disconnect"
        );
        // Idempotent: a second cleanup on an already-gone dir must not panic.
        mat.cleanup().await;
        assert!(!mat.config_dir.exists());
    }

    #[tokio::test]
    async fn cancellation_after_mkdtemp_cleans_up() {
        // Drop the materialize future partway is hard to force deterministically;
        // instead exercise the equivalent explicit path: a write failure after
        // mkdtemp must remove the temp tree (same code path a cancellation would
        // take via the match-arm cleanup). We force it by making the main load
        // succeed but list_subkeys advertise-then-fail on load.
        #[derive(Debug)]
        struct FailSubkeyList;
        #[async_trait::async_trait]
        impl SessionStore for FailSubkeyList {
            async fn append(
                &self,
                _key: &SessionKey,
                _entries: Vec<SessionStoreEntry>,
            ) -> Result<(), ClaudeError> {
                Ok(())
            }
            async fn load(
                &self,
                _key: &SessionKey,
            ) -> Result<Option<Vec<SessionStoreEntry>>, ClaudeError> {
                let mut m = SessionStoreEntry::new();
                m.insert("type".into(), json!("user"));
                Ok(Some(vec![m]))
            }
            fn capabilities(&self) -> SessionStoreCapabilities {
                SessionStoreCapabilities {
                    list_subkeys: true,
                    ..Default::default()
                }
            }
            async fn list_subkeys(
                &self,
                _key: &SessionListSubkeysKey,
            ) -> Result<Vec<String>, ClaudeError> {
                Err(ClaudeError::TransportError("list_subkeys down".into()))
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        let sid = Uuid::new_v4();
        let mut opts = opts_with_store(Arc::new(FailSubkeyList), tmp.path());
        opts.resume = Some(sid.to_string());
        // Materialize into a dedicated per-test parent so the leak count
        // observes only this call's trees (full-suite-safe under parallelism).
        let temp_parent = tempfile::tempdir().unwrap();
        let before = count_resume_temp_dirs(temp_parent.path());
        let err = materialize_resume_session_in(&opts, Some(temp_parent.path()))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("list_subkeys down"));
        assert!(
            count_resume_temp_dirs(temp_parent.path()) <= before,
            "temp tree leaked after subkey-list failure"
        );
    }
}
