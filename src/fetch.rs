//! Image fetching with a per-hook SSRF hostname allowlist.

use image::DynamicImage;
use std::io::Read;
use std::time::Duration;

const MAX_BYTES: u64 = 20 * 1024 * 1024;
const TIMEOUT: Duration = Duration::from_secs(10);

pub struct Fetcher {
    agent: ureq::Agent,
}

impl Fetcher {
    pub fn new() -> Self {
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
            agent: config.into(),
        }
    }

    /// True when `url`'s host matches any entry in this hook's allowlist.
    /// An empty allowlist denies everything.
    pub fn allows(allowlist: &[String], url: &str) -> bool {
        let Some(host) = host_of(url) else {
            return false;
        };
        allowlist.iter().any(|a| {
            let a = a.trim().to_ascii_lowercase();
            if let Some(suffix) = a.strip_prefix("*.") {
                host == suffix || host.ends_with(&format!(".{suffix}"))
            } else {
                host == a
            }
        })
    }

    pub fn fetch(&self, allowlist: &[String], url: &str) -> Result<DynamicImage, String> {
        if !url.starts_with("https://") && !url.starts_with("http://") {
            return Err("only http(s) urls are supported".into());
        }
        if !Self::allows(allowlist, url) {
            return Err("host not in this hook's allowlist".into());
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

    fn list(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn exact_host_allowed() {
        let l = list(&["example.com"]);
        assert!(Fetcher::allows(&l, "https://example.com/a.jpg"));
        assert!(Fetcher::allows(&l, "https://EXAMPLE.com/a.jpg"));
        assert!(!Fetcher::allows(&l, "https://evil.com/a.jpg"));
    }

    #[test]
    fn wildcard_matches_subdomains_only() {
        let l = list(&["*.example.com"]);
        assert!(Fetcher::allows(&l, "https://cdn.example.com/a.jpg"));
        assert!(Fetcher::allows(&l, "https://example.com/a.jpg"));
        assert!(!Fetcher::allows(&l, "https://notexample.com/a.jpg"));
        assert!(!Fetcher::allows(&l, "https://example.com.evil.net/a.jpg"));
    }

    #[test]
    fn userinfo_cannot_spoof_host() {
        let l = list(&["example.com"]);
        assert!(!Fetcher::allows(&l, "https://example.com@evil.com/a.jpg"));
        assert!(!Fetcher::allows(&l, "https://user:pw@evil.com/a.jpg"));
    }

    #[test]
    fn port_is_ignored_for_matching() {
        let l = list(&["example.com"]);
        assert!(Fetcher::allows(&l, "https://example.com:8443/a.jpg"));
    }

    #[test]
    fn empty_allowlist_denies_everything() {
        let l: Vec<String> = Vec::new();
        assert!(!Fetcher::allows(&l, "https://example.com/a.jpg"));
        assert!(!Fetcher::allows(&l, "https://127.0.0.1/a.jpg"));
    }

    #[test]
    fn hooks_are_isolated_from_each_other() {
        let a = list(&["a.example.com"]);
        let b = list(&["b.example.com"]);
        assert!(Fetcher::allows(&a, "https://a.example.com/x.jpg"));
        assert!(!Fetcher::allows(&a, "https://b.example.com/x.jpg"));
        assert!(Fetcher::allows(&b, "https://b.example.com/x.jpg"));
        assert!(!Fetcher::allows(&b, "https://a.example.com/x.jpg"));
    }

    #[test]
    fn non_http_schemes_rejected() {
        let f = Fetcher::new();
        let l = list(&["example.com"]);
        assert!(f.fetch(&l, "file:///etc/passwd").is_err());
        assert!(f.fetch(&l, "gopher://example.com/").is_err());
    }

    #[test]
    fn metadata_endpoints_denied_unless_listed() {
        let f = Fetcher::new();
        let l = list(&["example.com"]);
        assert!(!Fetcher::allows(
            &l,
            "http://169.254.169.254/latest/meta-data/"
        ));
        assert!(!Fetcher::allows(&l, "http://localhost:8080/admin"));
        assert!(f
            .fetch(&l, "http://169.254.169.254/latest/meta-data/")
            .is_err());
    }
}
