//! Image fetching with an SSRF hostname allowlist.

use image::DynamicImage;
use std::io::Read;
use std::time::Duration;

const MAX_BYTES: u64 = 20 * 1024 * 1024;
const TIMEOUT: Duration = Duration::from_secs(10);

pub struct Fetcher {
    allowlist: Vec<String>,
    agent: ureq::Agent,
}

impl Fetcher {
    pub fn new(allowlist: Vec<String>) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(TIMEOUT))
            .max_redirects(2)
            .user_agent(concat!(
                "clipd/",
                env!("CARGO_PKG_VERSION"),
                " (+https://github.com/Nachtalb/clipd)"
            ))
            .build();
        Self {
            allowlist,
            agent: config.into(),
        }
    }

    pub fn allows(&self, url: &str) -> bool {
        let Some(host) = host_of(url) else {
            return false;
        };
        self.allowlist.iter().any(|a| {
            let a = a.trim().to_ascii_lowercase();
            if let Some(suffix) = a.strip_prefix("*.") {
                host == suffix || host.ends_with(&format!(".{suffix}"))
            } else {
                host == a
            }
        })
    }

    pub fn fetch(&self, url: &str) -> Result<DynamicImage, String> {
        if !url.starts_with("https://") && !url.starts_with("http://") {
            return Err("only http(s) urls are supported".into());
        }
        if !self.allows(url) {
            return Err("host not in CLIPD_URL_ALLOWLIST".into());
        }

        let mut resp = self
            .agent
            .get(url)
            .call()
            .map_err(|e| format!("fetch failed: {e}"))?;

        let mut buf = Vec::new();
        resp.body_mut()
            .as_reader()
            .take(MAX_BYTES)
            .read_to_end(&mut buf)
            .map_err(|e| format!("read failed: {e}"))?;

        if buf.len() as u64 >= MAX_BYTES {
            return Err("image exceeds 20MB limit".into());
        }
        image::load_from_memory(&buf).map_err(|e| format!("decode failed: {e}"))
    }
}

fn host_of(url: &str) -> Option<String> {
    let rest = url.split_once("://")?.1;
    let authority = rest.split(['/', '?', '#']).next()?;
    let authority = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let host = match authority.rfind(':') {
        Some(i) if !authority[i + 1..].contains(']') => &authority[..i],
        _ => authority,
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    if host.is_empty() {
        None
    } else {
        Some(host.to_ascii_lowercase())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(list: &[&str]) -> Fetcher {
        Fetcher::new(list.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn exact_host_allowed() {
        let f = f(&["example.com"]);
        assert!(f.allows("https://example.com/a.jpg"));
        assert!(f.allows("https://EXAMPLE.com/a.jpg"));
        assert!(!f.allows("https://evil.com/a.jpg"));
    }

    #[test]
    fn wildcard_matches_subdomains_only() {
        let f = f(&["*.example.com"]);
        assert!(f.allows("https://cdn.example.com/a.jpg"));
        assert!(f.allows("https://example.com/a.jpg"));
        assert!(!f.allows("https://notexample.com/a.jpg"));
        assert!(!f.allows("https://example.com.evil.net/a.jpg"));
    }

    #[test]
    fn userinfo_cannot_spoof_host() {
        let f = f(&["example.com"]);
        assert!(!f.allows("https://example.com@evil.com/a.jpg"));
        assert!(!f.allows("https://user:pw@evil.com/a.jpg"));
    }

    #[test]
    fn port_is_ignored_for_matching() {
        let f = f(&["example.com"]);
        assert!(f.allows("https://example.com:8443/a.jpg"));
    }

    #[test]
    fn empty_allowlist_denies_everything() {
        let f = f(&[]);
        assert!(!f.allows("https://example.com/a.jpg"));
        assert!(!f.allows("https://127.0.0.1/a.jpg"));
    }

    #[test]
    fn non_http_schemes_rejected() {
        let f = f(&["example.com"]);
        assert!(f.fetch("file:///etc/passwd").is_err());
        assert!(f.fetch("gopher://example.com/").is_err());
    }

    #[test]
    fn metadata_endpoints_denied_unless_listed() {
        let f = f(&["example.com"]);
        assert!(!f.allows("http://169.254.169.254/latest/meta-data/"));
        assert!(!f.allows("http://localhost:8080/admin"));
        assert!(f.fetch("http://169.254.169.254/latest/meta-data/").is_err());
    }
}
