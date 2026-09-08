//! Hook records: storage, API key generation/verification, admin auth.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use hmac::{Hmac, Mac};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use subtle::ConstantTimeEq;

pub const KEY_PREFIX: &str = "clipd_";
const MIN_ADMIN_PASSWORD: usize = 16;
const MIN_PEPPER: usize = 32;
const MIN_LABELS: usize = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hook {
    pub name: String,
    pub key_hash: String,
    pub key_preview: String,
    /// label id -> prompt sentence
    pub labels: HashMap<String, String>,
    /// prompt sentence -> embedding
    #[serde(default)]
    pub label_vectors: HashMap<String, Vec<f32>>,
    /// Hostnames this hook may fetch images from. Empty denies everything.
    #[serde(default)]
    pub allowlist: Vec<String>,
    pub created: String,
    pub rotated: Option<String>,
    pub last_used: Option<String>,
}

/// Public view — never carries key_hash or vectors.
#[derive(Debug, Serialize)]
pub struct HookView {
    pub id: String,
    pub name: String,
    pub labels: HashMap<String, String>,
    pub allowlist: Vec<String>,
    pub key_preview: String,
    pub created: String,
    pub rotated: Option<String>,
    pub last_used: Option<String>,
}

impl Hook {
    pub fn view(&self, id: &str) -> HookView {
        HookView {
            id: id.to_string(),
            name: self.name.clone(),
            labels: self.labels.clone(),
            allowlist: self.allowlist.clone(),
            key_preview: self.key_preview.clone(),
            created: self.created.clone(),
            rotated: self.rotated.clone(),
            last_used: self.last_used.clone(),
        }
    }
}

pub struct Store {
    path: PathBuf,
    hooks: HashMap<String, Hook>,
    pepper: Vec<u8>,
}

impl Store {
    pub fn open(path: &Path, pepper: &[u8]) -> Self {
        let hooks = fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Self {
            path: path.to_path_buf(),
            hooks,
            pepper: pepper.to_vec(),
        }
    }

    pub fn hash_key(&self, key: &str) -> String {
        let mut mac = <Hmac<Sha256>>::new_from_slice(&self.pepper).expect("hmac accepts any key");
        mac.update(key.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    /// Constant-time lookup of a presented key against one hook.
    pub fn verify(&self, id: &str, key: &str) -> bool {
        let Some(hook) = self.hooks.get(id) else {
            return false;
        };
        let presented = self.hash_key(key);
        presented.as_bytes().ct_eq(hook.key_hash.as_bytes()).into()
    }

    pub fn get(&self, id: &str) -> Option<&Hook> {
        self.hooks.get(id)
    }

    pub fn list(&self) -> Vec<HookView> {
        let mut v: Vec<_> = self.hooks.iter().map(|(id, h)| h.view(id)).collect();
        v.sort_by(|a, b| a.created.cmp(&b.created));
        v
    }

    /// Create a hook. Returns (id, plaintext key) — the key is never stored.
    pub fn create(
        &mut self,
        name: String,
        labels: HashMap<String, String>,
        vectors: HashMap<String, Vec<f32>>,
        allowlist: Vec<String>,
    ) -> Result<(String, String), String> {
        validate_labels(&labels)?;
        validate_allowlist(&allowlist)?;
        let id = random_hex(8);
        let key = generate_key();
        let hook = Hook {
            name,
            key_hash: self.hash_key(&key),
            key_preview: preview(&key),
            labels,
            label_vectors: vectors,
            allowlist,
            created: now(),
            rotated: None,
            last_used: None,
        };
        self.hooks.insert(id.clone(), hook);
        self.save().map_err(|e| e.to_string())?;
        Ok((id, key))
    }

    pub fn update(
        &mut self,
        id: &str,
        name: Option<String>,
        labels: Option<HashMap<String, String>>,
        vectors: Option<HashMap<String, Vec<f32>>>,
        allowlist: Option<Vec<String>>,
    ) -> Result<(), String> {
        if let Some(ref l) = labels {
            validate_labels(l)?;
        }
        if let Some(ref a) = allowlist {
            validate_allowlist(a)?;
        }
        let hook = self.hooks.get_mut(id).ok_or("no such hook")?;
        if let Some(n) = name {
            hook.name = n;
        }
        if let Some(l) = labels {
            hook.labels = l;
        }
        if let Some(v) = vectors {
            hook.label_vectors = v;
        }
        if let Some(a) = allowlist {
            hook.allowlist = a;
        }
        self.save().map_err(|e| e.to_string())
    }

    /// Replace the key in place, keeping id/name/created/labels.
    pub fn rotate(&mut self, id: &str) -> Result<String, String> {
        let key = generate_key();
        let hash = self.hash_key(&key);
        let prev = preview(&key);
        let hook = self.hooks.get_mut(id).ok_or("no such hook")?;
        hook.key_hash = hash;
        hook.key_preview = prev;
        hook.rotated = Some(now());
        self.save().map_err(|e| e.to_string())?;
        Ok(key)
    }

    pub fn delete(&mut self, id: &str) -> Result<(), String> {
        self.hooks.remove(id).ok_or("no such hook")?;
        self.save().map_err(|e| e.to_string())
    }

    pub fn touch(&mut self, id: &str) {
        if let Some(h) = self.hooks.get_mut(id) {
            h.last_used = Some(now());
        }
        let _ = self.save();
    }

    fn save(&self) -> std::io::Result<()> {
        if let Some(dir) = self.path.parent() {
            fs::create_dir_all(dir)?;
        }
        let tmp = self.path.with_extension("tmp");
        let mut f = fs::File::create(&tmp)?;
        f.write_all(serde_json::to_string_pretty(&self.hooks)?.as_bytes())?;
        f.sync_all()?;
        fs::rename(&tmp, &self.path)
    }
}

fn validate_allowlist(allowlist: &[String]) -> Result<(), String> {
    if allowlist.is_empty() {
        return Err("allowlist must list at least one hostname".into());
    }
    for entry in allowlist {
        let e = entry.trim();
        if e.is_empty() {
            return Err("allowlist entries must not be empty".into());
        }
        if e.contains("://") || e.contains('/') {
            return Err(format!("allowlist entry {e:?} must be a bare hostname"));
        }
        if e == "*" {
            return Err("a bare \"*\" allowlist is not permitted".into());
        }
    }
    Ok(())
}

fn validate_labels(labels: &HashMap<String, String>) -> Result<(), String> {
    if labels.len() < MIN_LABELS {
        return Err(format!(
            "at least {MIN_LABELS} labels required (CLIP always ranks something highest; \
             include a catch-all such as \"a photo of something else\")"
        ));
    }
    if labels.values().any(|v| v.trim().is_empty()) {
        return Err("label prompts must not be empty".into());
    }
    Ok(())
}

pub fn generate_key() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    format!("{KEY_PREFIX}{}", URL_SAFE_NO_PAD.encode(bytes))
}

fn random_hex(n: usize) -> String {
    let mut bytes = vec![0u8; n / 2];
    rand::thread_rng().fill_bytes(&mut bytes);
    hex::encode(bytes)
}

fn preview(key: &str) -> String {
    let body = &key[KEY_PREFIX.len()..];
    format!("{KEY_PREFIX}{}…{}", &body[..4], &body[body.len() - 4..])
}

fn now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

/// Constant-time admin password check.
pub fn admin_ok(expected: &str, presented: &str) -> bool {
    if expected.is_empty() {
        return false;
    }
    let a = <Hmac<Sha256>>::new_from_slice(b"clipd-admin")
        .map(|mut m| {
            m.update(expected.as_bytes());
            m.finalize().into_bytes()
        })
        .expect("hmac");
    let b = <Hmac<Sha256>>::new_from_slice(b"clipd-admin")
        .map(|mut m| {
            m.update(presented.as_bytes());
            m.finalize().into_bytes()
        })
        .expect("hmac");
    a.ct_eq(&b).into()
}

/// Read and validate required secrets. Returns an error rather than starting.
pub fn require_secrets() -> Result<(String, String), String> {
    let pw = std::env::var("CLIPD_ADMIN_PASSWORD").unwrap_or_default();
    let pepper = std::env::var("CLIPD_KEY_PEPPER").unwrap_or_default();
    validate_secrets(&pw, &pepper)?;
    Ok((pw, pepper))
}

pub fn validate_secrets(pw: &str, pepper: &str) -> Result<(), String> {
    if pw.len() < MIN_ADMIN_PASSWORD {
        return Err(format!(
            "CLIPD_ADMIN_PASSWORD must be set and at least {MIN_ADMIN_PASSWORD} characters"
        ));
    }
    if pepper.len() < MIN_PEPPER {
        return Err(format!(
            "CLIPD_KEY_PEPPER must be set and at least {MIN_PEPPER} bytes"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PEPPER: &[u8] = b"test-pepper-at-least-32-bytes-long!!";
    const OTHER_PEPPER: &[u8] = b"different-pepper-also-32-bytes-xx!!!";

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("clipd-hooks-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d.join("hooks.json")
    }

    fn al() -> Vec<String> {
        vec!["example.com".to_string()]
    }

    fn labels() -> HashMap<String, String> {
        HashMap::from([
            ("cat".to_string(), "a photo of a cat".to_string()),
            ("other".to_string(), "a photo of something else".to_string()),
        ])
    }

    #[test]
    fn correct_key_passes_wrong_key_fails() {
        let mut s = Store::open(&tmp("verify"), PEPPER);
        let (id, key) = s
            .create("t".into(), labels(), HashMap::new(), al())
            .unwrap();
        assert!(s.verify(&id, &key));
        assert!(!s.verify(&id, "clipd_wrong"));
        assert!(!s.verify("nosuchhook", &key));
    }

    #[test]
    fn deleted_hook_key_fails() {
        let mut s = Store::open(&tmp("delete"), PEPPER);
        let (id, key) = s
            .create("t".into(), labels(), HashMap::new(), al())
            .unwrap();
        assert!(s.verify(&id, &key));
        s.delete(&id).unwrap();
        assert!(!s.verify(&id, &key));
        assert!(s.delete(&id).is_err());
    }

    #[test]
    fn stored_json_never_contains_key_or_pepper() {
        let path = tmp("nosecrets");
        let mut s = Store::open(&path, PEPPER);
        let (_, key) = s
            .create("t".into(), labels(), HashMap::new(), al())
            .unwrap();

        let raw = fs::read_to_string(&path).unwrap();
        assert!(!raw.contains(&key), "plaintext key leaked to disk");
        assert!(
            !raw.contains(std::str::from_utf8(PEPPER).unwrap()),
            "pepper leaked to disk"
        );
        assert!(raw.contains("key_hash"));
    }

    #[test]
    fn hashing_is_deterministic() {
        let s = Store::open(&tmp("determ"), PEPPER);
        let key = generate_key();
        assert_eq!(s.hash_key(&key), s.hash_key(&key));
    }

    #[test]
    fn different_pepper_yields_different_digest() {
        let a = Store::open(&tmp("peppera"), PEPPER);
        let b = Store::open(&tmp("pepperb"), OTHER_PEPPER);
        let key = generate_key();
        assert_ne!(
            a.hash_key(&key),
            b.hash_key(&key),
            "pepper is not being mixed in"
        );
    }

    #[test]
    fn rotate_keeps_identity_and_invalidates_old_key() {
        let mut s = Store::open(&tmp("rotate"), PEPPER);
        let (id, old) = s
            .create("keepme".into(), labels(), HashMap::new(), al())
            .unwrap();
        let created = s.get(&id).unwrap().created.clone();

        let new = s.rotate(&id).unwrap();
        assert_ne!(old, new);
        assert!(!s.verify(&id, &old), "old key still works after rotate");
        assert!(s.verify(&id, &new));

        let h = s.get(&id).unwrap();
        assert_eq!(h.name, "keepme");
        assert_eq!(h.created, created);
        assert_eq!(h.labels.len(), 2);
        assert!(h.rotated.is_some());
    }

    #[test]
    fn persists_across_reopen() {
        let path = tmp("persist");
        let (id, key) = {
            let mut s = Store::open(&path, PEPPER);
            s.create("t".into(), labels(), HashMap::new(), al())
                .unwrap()
        };
        let s2 = Store::open(&path, PEPPER);
        assert!(s2.verify(&id, &key));
    }

    #[test]
    fn fewer_than_two_labels_rejected() {
        let mut s = Store::open(&tmp("minlabels"), PEPPER);
        let one = HashMap::from([("a".to_string(), "a photo of a cat".to_string())]);
        assert!(s.create("t".into(), one, HashMap::new(), al()).is_err());
        assert!(s
            .create("t".into(), HashMap::new(), HashMap::new(), al())
            .is_err());

        let (id, _) = s
            .create("t".into(), labels(), HashMap::new(), al())
            .unwrap();
        let one = HashMap::from([("a".to_string(), "a photo of a cat".to_string())]);
        assert!(s.update(&id, None, Some(one), None, None).is_err());
    }

    #[test]
    fn empty_label_prompt_rejected() {
        let mut s = Store::open(&tmp("emptyprompt"), PEPPER);
        let bad = HashMap::from([
            ("a".to_string(), "a photo of a cat".to_string()),
            ("b".to_string(), "   ".to_string()),
        ]);
        assert!(s.create("t".into(), bad, HashMap::new(), al()).is_err());
    }

    #[test]
    fn allowlist_is_required_and_validated() {
        let mut s = Store::open(&tmp("allowlist"), PEPPER);

        // empty allowlist is refused — a hook that can fetch nothing is a
        // configuration error, and defaulting to "everything" would be an SSRF.
        assert!(s
            .create("t".into(), labels(), HashMap::new(), vec![])
            .is_err());

        // a URL is not a hostname
        assert!(s
            .create(
                "t".into(),
                labels(),
                HashMap::new(),
                vec!["https://example.com/x".to_string()]
            )
            .is_err());

        // no blanket wildcard
        assert!(s
            .create("t".into(), labels(), HashMap::new(), vec!["*".to_string()])
            .is_err());

        // blank entry
        assert!(s
            .create("t".into(), labels(), HashMap::new(), vec!["  ".to_string()])
            .is_err());
    }

    #[test]
    fn allowlist_persists_and_updates() {
        let path = tmp("allowlistpersist");
        let id = {
            let mut s = Store::open(&path, PEPPER);
            let (id, _) = s
                .create(
                    "t".into(),
                    labels(),
                    HashMap::new(),
                    vec!["a.example.com".to_string()],
                )
                .unwrap();
            id
        };

        // survives a reload
        let mut s = Store::open(&path, PEPPER);
        assert_eq!(s.get(&id).unwrap().allowlist, vec!["a.example.com"]);

        // and can be replaced
        s.update(
            &id,
            None,
            None,
            None,
            Some(vec!["b.example.com".to_string()]),
        )
        .unwrap();
        assert_eq!(s.get(&id).unwrap().allowlist, vec!["b.example.com"]);

        // but not emptied
        assert!(s.update(&id, None, None, None, Some(vec![])).is_err());
        assert_eq!(s.get(&id).unwrap().allowlist, vec!["b.example.com"]);
    }

    #[test]
    fn view_omits_hash_and_vectors() {
        let mut s = Store::open(&tmp("view"), PEPPER);
        let vectors = HashMap::from([("a photo of a cat".to_string(), vec![0.5f32; 512])]);
        let (id, key) = s.create("t".into(), labels(), vectors, al()).unwrap();

        let json = serde_json::to_string(&s.get(&id).unwrap().view(&id)).unwrap();
        assert!(!json.contains("key_hash"));
        assert!(!json.contains("label_vectors"));
        assert!(!json.contains(&key));
        assert!(json.contains("key_preview"));
    }

    #[test]
    fn key_has_prefix_and_entropy() {
        let a = generate_key();
        let b = generate_key();
        assert!(a.starts_with(KEY_PREFIX));
        assert_ne!(a, b);
        assert!(a.len() > 40);
    }

    #[test]
    fn preview_hides_the_middle() {
        let key = generate_key();
        let p = preview(&key);
        assert!(p.starts_with(KEY_PREFIX));
        assert!(p.contains('…'));
        assert!(!key.contains(&p));
        assert!(p.len() < 20);
    }

    #[test]
    fn admin_password_compare() {
        assert!(admin_ok(
            "correct horse battery staple",
            "correct horse battery staple"
        ));
        assert!(!admin_ok("correct horse battery staple", "wrong"));
        assert!(!admin_ok("", ""), "empty password must never authenticate");
        assert!(!admin_ok("", "anything"));
    }

    #[test]
    fn secret_validation_rejects_weak_values() {
        assert!(validate_secrets("", "").is_err());
        assert!(validate_secrets("short", &"p".repeat(32)).is_err());
        assert!(validate_secrets(&"a".repeat(16), "").is_err());
        assert!(validate_secrets(&"a".repeat(16), &"p".repeat(31)).is_err());
        assert!(validate_secrets(&"a".repeat(16), &"p".repeat(32)).is_ok());
    }
}
