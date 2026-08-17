//! Reusable [`SessionStore`] conformance suite.
//!
//! Ported from the official Python Agent SDK
//! (`testing/session_store_conformance.py`). Any adapter can be validated by
//! calling [`run_conformance`] with a fresh-store factory. Contracts that
//! require an optional capability are skipped when the store does not advertise
//! it (via [`SessionStore::capabilities`]).
//!
//! Keys are arbitrary strings (`"proj"`/`"sess"`, `"a"`, `"b"`, ...) exactly as
//! upstream uses them — session IDs are opaque, lossless strings at the store
//! layer, and UUID validation lives only at the APIs that officially require it
//! (resume, import, mutations), never in the store contract itself.

use std::future::Future;

use serde_json::{json, Map, Value};

use crate::sessions::key::{SessionKey, SessionListSubkeysKey};
use crate::sessions::store::{SessionStore, SessionStoreEntry};
use crate::sessions::summary::fold_session_summary;

fn entry(v: Value) -> SessionStoreEntry {
    match v {
        Value::Object(m) => m,
        _ => panic!("entry must be a JSON object"),
    }
}

/// Main-transcript key with arbitrary string ids (no UUID requirement).
fn key(project: &str, session: &str) -> SessionKey {
    SessionKey::new(project, session)
}

/// Subpath key with arbitrary string ids.
fn sub(project: &str, session: &str, subpath: &str) -> SessionKey {
    SessionKey::with_subpath(project, session, subpath).unwrap()
}

/// Runs every conformance contract against a store produced by `make_store`.
///
/// `make_store` is called once per contract so each contract sees a clean
/// store. Panics (via `assert!`) on the first violated contract.
///
/// # Panics
/// If any conformance contract is violated.
pub async fn run_conformance<S, F, Fut>(make_store: F)
where
    S: SessionStore,
    F: Fn() -> Fut,
    Fut: Future<Output = S>,
{
    // --- Required: append + load ------------------------------------------
    contract_append_load_roundtrip(&make_store().await).await;
    contract_append_ordering(&make_store().await).await;
    contract_load_missing_is_none(&make_store().await).await;
    contract_load_unknown_subpath_is_none(&make_store().await).await;
    contract_empty_append_is_noop(&make_store().await).await;
    contract_deep_equal_not_byte_equal(&make_store().await).await;
    contract_project_isolation(&make_store().await).await;
    contract_subpath_isolation(&make_store().await).await;
    contract_arbitrary_string_keys_roundtrip(&make_store().await).await;

    // --- Optional: list_sessions ------------------------------------------
    contract_list_sessions(&make_store().await).await;
    contract_list_sessions_mtime_epoch_ms(&make_store().await).await;
    contract_list_sessions_mtime_monotonic(&make_store().await).await;
    contract_list_sessions_excludes_subpaths(&make_store().await).await;

    // --- Optional: list_session_summaries ---------------------------------
    contract_list_session_summaries(&make_store().await).await;
    contract_summary_clock_refold_opaque_subpath_delete(&make_store().await).await;

    // --- Optional: delete -------------------------------------------------
    contract_delete(&make_store().await).await;
    contract_delete_cascades_subkeys(&make_store().await).await;
    contract_delete_targeted_subpath(&make_store().await).await;

    // --- Optional: list_subkeys -------------------------------------------
    contract_list_subkeys(&make_store().await).await;
    contract_list_subkeys_excludes_main(&make_store().await).await;
}

async fn contract_append_load_roundtrip<S: SessionStore>(store: &S) {
    let k = key("proj", "sess");
    let e1 = entry(json!({"type": "x", "uuid": "b", "n": 1}));
    let e2 = entry(json!({"type": "x", "uuid": "a", "n": 2}));
    store
        .append(&k, vec![e1.clone(), e2.clone()])
        .await
        .unwrap();
    let loaded = store.load(&k).await.unwrap().expect("session must exist");
    // Deep-equal is the contract; byte-equal serialization is intentionally not
    // checked (JSONB adapters may reorder keys).
    assert_eq!(
        loaded,
        vec![e1, e2],
        "append/load round-trip preserves order"
    );
}

async fn contract_append_ordering<S: SessionStore>(store: &S) {
    let k = key("proj", "sess");
    let e1 = entry(json!({"type": "x", "uuid": "z", "n": 1}));
    let e2 = entry(json!({"type": "x", "uuid": "a", "n": 2}));
    let e3 = entry(json!({"type": "x", "uuid": "m", "n": 3}));
    let e4 = entry(json!({"type": "x", "uuid": "b", "n": 4}));
    store.append(&k, vec![e1.clone()]).await.unwrap();
    store
        .append(&k, vec![e2.clone(), e3.clone()])
        .await
        .unwrap();
    store.append(&k, vec![e4.clone()]).await.unwrap();
    let loaded = store.load(&k).await.unwrap().unwrap();
    assert_eq!(
        loaded,
        vec![e1, e2, e3, e4],
        "multiple append calls preserve call order"
    );
}

async fn contract_load_missing_is_none<S: SessionStore>(store: &S) {
    assert!(
        store.load(&key("proj", "nope")).await.unwrap().is_none(),
        "load of an unknown session is None"
    );
}

async fn contract_load_unknown_subpath_is_none<S: SessionStore>(store: &S) {
    // A written main session does not imply an arbitrary subpath exists.
    store
        .append(
            &key("proj", "sess"),
            vec![entry(json!({"type": "x", "uuid": "x"}))],
        )
        .await
        .unwrap();
    assert!(
        store
            .load(&sub("proj", "sess", "subagents/nope"))
            .await
            .unwrap()
            .is_none(),
        "load of an unknown subpath under an existing session is None"
    );
}

async fn contract_empty_append_is_noop<S: SessionStore>(store: &S) {
    let k = key("proj", "sess");
    let e = entry(json!({"type": "x", "uuid": "a", "n": 1}));
    store.append(&k, vec![e.clone()]).await.unwrap();
    store.append(&k, Vec::new()).await.unwrap();
    assert_eq!(
        store.load(&k).await.unwrap().unwrap(),
        vec![e],
        "append([]) is a no-op (no phantom entries, no mutation)"
    );
}

async fn contract_deep_equal_not_byte_equal<S: SessionStore>(store: &S) {
    let k = key("proj", "sess");
    let mut obj = Map::new();
    obj.insert("type".into(), json!("x"));
    obj.insert("z_last".into(), json!(1));
    obj.insert("a_first".into(), json!({"nested": [1, 2, 3]}));
    store.append(&k, vec![obj.clone()]).await.unwrap();
    let loaded = store.load(&k).await.unwrap().unwrap();
    assert_eq!(
        loaded[0], obj,
        "deep-equal round-trip regardless of key order"
    );
}

async fn contract_project_isolation<S: SessionStore>(store: &S) {
    let a = key("A", "s1");
    let b = key("B", "s1");
    store
        .append(&a, vec![entry(json!({"type": "x", "from": "A"}))])
        .await
        .unwrap();
    store
        .append(&b, vec![entry(json!({"type": "x", "from": "B"}))])
        .await
        .unwrap();
    assert_eq!(
        store.load(&a).await.unwrap().unwrap()[0]
            .get("from")
            .unwrap(),
        "A",
        "same session id isolated across projects"
    );
    assert_eq!(
        store.load(&b).await.unwrap().unwrap()[0]
            .get("from")
            .unwrap(),
        "B",
        "same session id isolated across projects"
    );
    if store.capabilities().list_sessions {
        assert_eq!(store.list_sessions("A").await.unwrap().len(), 1);
        assert_eq!(store.list_sessions("B").await.unwrap().len(), 1);
    }
}

async fn contract_subpath_isolation<S: SessionStore>(store: &S) {
    let main = key("proj", "sess");
    let s = sub("proj", "sess", "subagents/agent-1");
    store
        .append(&main, vec![entry(json!({"type": "x", "who": "main"}))])
        .await
        .unwrap();
    store
        .append(&s, vec![entry(json!({"type": "x", "who": "sub"}))])
        .await
        .unwrap();
    assert_eq!(
        store.load(&main).await.unwrap().unwrap()[0]
            .get("who")
            .unwrap(),
        "main",
        "main/subpath stored independently"
    );
    assert_eq!(
        store.load(&s).await.unwrap().unwrap()[0]
            .get("who")
            .unwrap(),
        "sub",
        "main/subpath stored independently"
    );
}

async fn contract_arbitrary_string_keys_roundtrip<S: SessionStore>(store: &S) {
    // Session ids are opaque strings — a non-UUID stem such as `abc-123` must
    // round-trip losslessly through the store contract.
    let k = key("proj", "abc-123");
    let e = entry(json!({"type": "x", "uuid": "1"}));
    store.append(&k, vec![e.clone()]).await.unwrap();
    let loaded = store.load(&k).await.unwrap().expect("arbitrary key stored");
    assert_eq!(
        loaded,
        vec![e],
        "arbitrary (non-UUID) session key round-trips"
    );
    if store.capabilities().list_sessions {
        let ids: Vec<String> = store
            .list_sessions("proj")
            .await
            .unwrap()
            .into_iter()
            .map(|e| e.session_id)
            .collect();
        assert_eq!(
            ids,
            vec!["abc-123".to_string()],
            "arbitrary id is listed verbatim"
        );
    }
}

async fn contract_list_sessions<S: SessionStore>(store: &S) {
    if !store.capabilities().list_sessions {
        return;
    }
    store
        .append(&key("proj", "a"), vec![entry(json!({"type": "x", "n": 1}))])
        .await
        .unwrap();
    store
        .append(&key("proj", "b"), vec![entry(json!({"type": "x", "n": 1}))])
        .await
        .unwrap();
    store
        .append(
            &key("other", "c"),
            vec![entry(json!({"type": "x", "n": 1}))],
        )
        .await
        .unwrap();
    let mut ids: Vec<String> = store
        .list_sessions("proj")
        .await
        .unwrap()
        .into_iter()
        .map(|e| e.session_id)
        .collect();
    ids.sort();
    assert_eq!(
        ids,
        vec!["a".to_string(), "b".to_string()],
        "list_sessions scoped to project"
    );
    assert_eq!(
        store
            .list_sessions("never-appended-project")
            .await
            .unwrap()
            .len(),
        0,
        "list_sessions on an empty project is empty"
    );
}

async fn contract_list_sessions_mtime_epoch_ms<S: SessionStore>(store: &S) {
    if !store.capabilities().list_sessions {
        return;
    }
    store
        .append(&key("proj", "a"), vec![entry(json!({"type": "x"}))])
        .await
        .unwrap();
    store
        .append(&key("proj", "b"), vec![entry(json!({"type": "x"}))])
        .await
        .unwrap();
    // mtime must be epoch-ms; >1e12 rules out epoch-seconds (that threshold is
    // ~2001 in ms, but ~33000 CE in seconds, so it cleanly separates the two).
    for e in store.list_sessions("proj").await.unwrap() {
        assert!(
            e.mtime > 1_000_000_000_000,
            "list_sessions mtime must be epoch-ms, got {}",
            e.mtime
        );
    }
}

async fn contract_list_sessions_mtime_monotonic<S: SessionStore>(store: &S) {
    if !store.capabilities().list_sessions {
        return;
    }
    store
        .append(&key("proj", "a"), vec![entry(json!({"type": "x"}))])
        .await
        .unwrap();
    store
        .append(&key("proj", "b"), vec![entry(json!({"type": "x"}))])
        .await
        .unwrap();
    let entries = store.list_sessions("proj").await.unwrap();
    let ma = entries.iter().find(|e| e.session_id == "a").unwrap().mtime;
    let mb = entries.iter().find(|e| e.session_id == "b").unwrap().mtime;
    assert!(
        mb > ma,
        "later append has strictly greater mtime ({mb} > {ma})"
    );
}

async fn contract_list_sessions_excludes_subpaths<S: SessionStore>(store: &S) {
    if !store.capabilities().list_sessions {
        return;
    }
    store
        .append(&key("proj", "main"), vec![entry(json!({"type": "x"}))])
        .await
        .unwrap();
    store
        .append(
            &sub("proj", "main", "subagents/agent-1"),
            vec![entry(json!({"type": "x"}))],
        )
        .await
        .unwrap();
    let entries = store.list_sessions("proj").await.unwrap();
    let ids: Vec<&str> = entries.iter().map(|e| e.session_id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["main"],
        "list_sessions excludes subagent subpaths"
    );
}

async fn contract_list_session_summaries<S: SessionStore>(store: &S) {
    if !store.capabilities().list_session_summaries {
        return;
    }
    store
        .append(
            &key("proj", "summ"),
            vec![entry(json!({
                "type": "user",
                "timestamp": "2024-01-02T03:04:05Z",
                "message": {"content": "hi there"}
            }))],
        )
        .await
        .unwrap();
    let summaries = store.list_session_summaries("proj").await.unwrap();
    assert_eq!(summaries.len(), 1, "one summary per main session");
    let s = &summaries[0];
    assert_eq!(s.session_id, "summ");
    assert!(s.mtime > 1_000_000_000_000, "summary mtime is epoch-ms");
    assert_eq!(
        s.data.get("first_prompt").and_then(Value::as_str),
        Some("hi there"),
        "summary folds the first prompt"
    );
    assert_eq!(
        store
            .list_session_summaries("never-appended-project")
            .await
            .unwrap()
            .len(),
        0,
        "summaries on an empty project is empty"
    );
}

/// Contract 14 from upstream: the persisted summary shares `list_sessions`'
/// clock, round-trips back into `fold_session_summary`, is opaque (stored
/// verbatim), is unaffected by subagent appends, and is cleared by delete.
async fn contract_summary_clock_refold_opaque_subpath_delete<S: SessionStore>(store: &S) {
    if !store.capabilities().list_session_summaries {
        return;
    }
    let k = key("proj", "summ-sess");
    store
        .append(
            &k,
            vec![
                entry(json!({"type": "x", "timestamp": "2024-01-01T00:00:00.000Z", "customTitle": "first"})),
                entry(json!({"type": "x", "timestamp": "2024-01-01T00:00:01.000Z"})),
            ],
        )
        .await
        .unwrap();
    store
        .append(
            &k,
            vec![entry(json!({"type": "x", "timestamp": "2024-01-01T00:00:02.000Z", "customTitle": "second"}))],
        )
        .await
        .unwrap();
    store
        .append(
            &key("other", "elsewhere"),
            vec![entry(
                json!({"type": "x", "timestamp": "2024-01-01T00:00:00.000Z"}),
            )],
        )
        .await
        .unwrap();

    let summaries = store.list_session_summaries("proj").await.unwrap();
    let ids: Vec<&str> = summaries.iter().map(|s| s.session_id.as_str()).collect();
    assert_eq!(ids, vec!["summ-sess"], "summaries scoped to project");
    let summ = summaries.into_iter().next().unwrap();
    assert!(summ.mtime > 1_000_000_000_000, "summary mtime is epoch-ms");

    // Clock alignment: the sidecar mtime is storage write time and must share a
    // clock with list_sessions().mtime for the same session — a summary must
    // never look staler than the session's list-time mtime.
    if store.capabilities().list_sessions {
        let ls_mtime = store
            .list_sessions("proj")
            .await
            .unwrap()
            .into_iter()
            .find(|e| e.session_id == "summ-sess")
            .unwrap()
            .mtime;
        assert!(
            summ.mtime >= ls_mtime,
            "summary mtime ({}) must not predate list_sessions mtime ({ls_mtime})",
            summ.mtime
        );
    }

    // data is opaque: it round-trips back into the fold without interpretation,
    // and the fold preserves prev.mtime verbatim (mtime is adapter-stamped,
    // never set by the fold).
    let refolded = fold_session_summary(
        Some(&summ),
        &k,
        &[entry(
            json!({"type": "x", "timestamp": "2024-01-01T00:00:03.000Z"}),
        )],
    );
    assert_eq!(refolded.session_id, "summ-sess");
    assert_eq!(
        refolded.mtime, summ.mtime,
        "fold preserves prev.mtime verbatim"
    );

    // A subagent append must NOT change the main session's summary.
    store
        .append(
            &sub("proj", "summ-sess", "subagents/agent-1"),
            vec![entry(json!({"type": "x", "timestamp": "2024-01-01T00:00:09.000Z", "customTitle": "subagent"}))],
        )
        .await
        .unwrap();
    let after = store.list_session_summaries("proj").await.unwrap();
    let after_summ = after.iter().find(|s| s.session_id == "summ-sess").unwrap();
    assert_eq!(
        after_summ.data, summ.data,
        "subagent append does not alter the main session summary"
    );

    // delete clears the summary from the listing.
    if store.capabilities().delete {
        store.delete(&k).await.unwrap();
        let ids_after: Vec<String> = store
            .list_session_summaries("proj")
            .await
            .unwrap()
            .into_iter()
            .map(|s| s.session_id)
            .collect();
        assert!(
            !ids_after.contains(&"summ-sess".to_string()),
            "delete clears the session summary"
        );
    }
}

async fn contract_delete<S: SessionStore>(store: &S) {
    if !store.capabilities().delete {
        return;
    }
    // delete of a never-written session is a harmless no-op.
    store.delete(&key("proj", "never-written")).await.unwrap();
    let k = key("proj", "sess");
    store
        .append(&k, vec![entry(json!({"type": "x"}))])
        .await
        .unwrap();
    store.delete(&k).await.unwrap();
    assert!(
        store.load(&k).await.unwrap().is_none(),
        "delete removes the session"
    );
}

async fn contract_delete_cascades_subkeys<S: SessionStore>(store: &S) {
    if !store.capabilities().delete {
        return;
    }
    let main = key("proj", "sess");
    let sub1 = sub("proj", "sess", "subagents/agent-1");
    let sub2 = sub("proj", "sess", "subagents/agent-2");
    let other = key("proj", "sess2");
    let other_proj = key("other-proj", "sess");
    for k in [&main, &sub1, &sub2, &other, &other_proj] {
        store
            .append(k, vec![entry(json!({"type": "x"}))])
            .await
            .unwrap();
    }

    store.delete(&main).await.unwrap();

    assert!(store.load(&main).await.unwrap().is_none(), "main deleted");
    assert!(store.load(&sub1).await.unwrap().is_none(), "cascade sub1");
    assert!(store.load(&sub2).await.unwrap().is_none(), "cascade sub2");
    // Sibling session in the same project survives.
    assert_eq!(
        store.load(&other).await.unwrap().map(|e| e.len()),
        Some(1),
        "sibling session in the same project untouched"
    );
    // Same session id in a different project survives (project-scoped delete).
    assert_eq!(
        store.load(&other_proj).await.unwrap().map(|e| e.len()),
        Some(1),
        "same id in another project untouched"
    );
    if store.capabilities().list_subkeys {
        assert!(
            store
                .list_subkeys(&SessionListSubkeysKey::new("proj", "sess"))
                .await
                .unwrap()
                .is_empty(),
            "list_subkeys empty after cascade delete"
        );
    }
    if store.capabilities().list_sessions {
        let ids: Vec<String> = store
            .list_sessions("proj")
            .await
            .unwrap()
            .into_iter()
            .map(|e| e.session_id)
            .collect();
        assert!(
            !ids.contains(&"sess".to_string()),
            "deleted session excluded from list_sessions"
        );
        assert!(
            ids.contains(&"sess2".to_string()),
            "sibling session still listed"
        );
    }
}

async fn contract_delete_targeted_subpath<S: SessionStore>(store: &S) {
    if !store.capabilities().delete {
        return;
    }
    let main = key("proj", "sess");
    let sub1 = sub("proj", "sess", "subagents/agent-1");
    let sub2 = sub("proj", "sess", "subagents/agent-2");
    for k in [&main, &sub1, &sub2] {
        store
            .append(k, vec![entry(json!({"type": "x"}))])
            .await
            .unwrap();
    }
    store.delete(&sub1).await.unwrap();
    assert!(
        store.load(&sub1).await.unwrap().is_none(),
        "targeted subpath deleted"
    );
    assert_eq!(
        store.load(&sub2).await.unwrap().map(|e| e.len()),
        Some(1),
        "other subpath untouched"
    );
    assert_eq!(
        store.load(&main).await.unwrap().map(|e| e.len()),
        Some(1),
        "main untouched by targeted subpath delete"
    );
    if store.capabilities().list_subkeys {
        assert_eq!(
            store
                .list_subkeys(&SessionListSubkeysKey::new("proj", "sess"))
                .await
                .unwrap(),
            vec!["subagents/agent-2".to_string()],
            "only the remaining subkey is listed"
        );
    }
}

async fn contract_list_subkeys<S: SessionStore>(store: &S) {
    if !store.capabilities().list_subkeys {
        return;
    }
    store
        .append(&key("proj", "sess"), vec![entry(json!({"type": "x"}))])
        .await
        .unwrap();
    store
        .append(
            &sub("proj", "sess", "subagents/agent-1"),
            vec![entry(json!({"type": "x"}))],
        )
        .await
        .unwrap();
    store
        .append(
            &sub("proj", "sess", "subagents/agent-2"),
            vec![entry(json!({"type": "x"}))],
        )
        .await
        .unwrap();
    // A subpath under a DIFFERENT session must not leak into these results.
    store
        .append(
            &sub("proj", "other-sess", "subagents/agent-x"),
            vec![entry(json!({"type": "x"}))],
        )
        .await
        .unwrap();
    let mut subkeys = store
        .list_subkeys(&SessionListSubkeysKey::new("proj", "sess"))
        .await
        .unwrap();
    subkeys.sort();
    assert_eq!(
        subkeys,
        vec![
            "subagents/agent-1".to_string(),
            "subagents/agent-2".to_string()
        ],
        "list_subkeys returns only subpaths under the session"
    );
    assert!(
        !subkeys.contains(&"subagents/agent-x".to_string()),
        "subpaths of other sessions excluded"
    );
}

async fn contract_list_subkeys_excludes_main<S: SessionStore>(store: &S) {
    if !store.capabilities().list_subkeys {
        return;
    }
    store
        .append(&key("proj", "sess"), vec![entry(json!({"type": "x"}))])
        .await
        .unwrap();
    assert!(
        store
            .list_subkeys(&SessionListSubkeysKey::new("proj", "sess"))
            .await
            .unwrap()
            .is_empty(),
        "list_subkeys excludes the main transcript"
    );
    assert!(
        store
            .list_subkeys(&SessionListSubkeysKey::new("proj", "never-appended"))
            .await
            .unwrap()
            .is_empty(),
        "list_subkeys of a never-appended session is empty"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sessions::store::InMemorySessionStore;

    #[tokio::test]
    async fn in_memory_store_passes_conformance() {
        run_conformance(|| async { InMemorySessionStore::new() }).await;
    }
}
