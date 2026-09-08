//! Persistent label-vector cache, keyed by xxh3 of the prompt sentence.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use xxhash_rust::xxh3::xxh3_64;

use crate::vision::DIM;

pub struct Cache {
    path: PathBuf,
    map: HashMap<u64, Vec<f32>>,
}

impl Cache {
    pub fn load(path: &Path) -> Self {
        let map = fs::File::open(path)
            .ok()
            .and_then(|mut f| {
                let mut buf = Vec::new();
                f.read_to_end(&mut buf).ok()?;
                Some(decode(&buf))
            })
            .unwrap_or_default();
        Self {
            path: path.to_path_buf(),
            map,
        }
    }

    pub fn get(&self, prompt: &str) -> Option<&Vec<f32>> {
        self.map.get(&xxh3_64(prompt.as_bytes()))
    }

    pub fn insert(&mut self, prompt: &str, vec: Vec<f32>) {
        self.map.insert(xxh3_64(prompt.as_bytes()), vec);
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Atomic write: temp file then rename.
    pub fn save(&self) -> std::io::Result<()> {
        if let Some(dir) = self.path.parent() {
            fs::create_dir_all(dir)?;
        }
        let tmp = self.path.with_extension("tmp");
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&encode(&self.map))?;
        f.sync_all()?;
        fs::rename(&tmp, &self.path)
    }
}

fn encode(map: &HashMap<u64, Vec<f32>>) -> Vec<u8> {
    let mut out = Vec::with_capacity(map.len() * (8 + DIM * 4));
    for (k, v) in map {
        out.extend_from_slice(&k.to_le_bytes());
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

fn decode(buf: &[u8]) -> HashMap<u64, Vec<f32>> {
    let rec = 8 + DIM * 4;
    let mut map = HashMap::new();
    for chunk in buf.chunks_exact(rec) {
        let key = u64::from_le_bytes(chunk[..8].try_into().unwrap());
        let vec = chunk[8..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        map.insert(key, vec);
    }
    map
}

/// Split prompts into those already cached and those needing embedding.
/// `stored` is the hook's own label_vectors, checked before the disk cache.
pub fn resolve<'a>(
    prompts: &'a [String],
    stored: &HashMap<String, Vec<f32>>,
    cache: &Cache,
) -> (HashMap<&'a str, Vec<f32>>, Vec<String>) {
    let mut hits = HashMap::new();
    let mut misses = Vec::new();
    for p in prompts {
        if let Some(v) = stored.get(p) {
            hits.insert(p.as_str(), v.clone());
        } else if let Some(v) = cache.get(p) {
            hits.insert(p.as_str(), v.clone());
        } else if !misses.contains(p) {
            misses.push(p.clone());
        }
    }
    (hits, misses)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("clipd-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn vec_of(x: f32) -> Vec<f32> {
        vec![x; DIM]
    }

    #[test]
    fn miss_then_hit() {
        let dir = tmpdir("missthenhit");
        let mut c = Cache::load(&dir.join("cache.bin"));
        assert!(c.get("a photo of a cat").is_none());
        c.insert("a photo of a cat", vec_of(0.5));
        assert_eq!(c.get("a photo of a cat").unwrap()[0], 0.5);
    }

    #[test]
    fn survives_save_and_load() {
        let dir = tmpdir("roundtrip");
        let path = dir.join("cache.bin");
        let mut c = Cache::load(&path);
        c.insert("one", vec_of(0.25));
        c.insert("two", vec_of(0.75));
        c.save().unwrap();

        let c2 = Cache::load(&path);
        assert_eq!(c2.len(), 2);
        assert_eq!(c2.get("one").unwrap()[0], 0.25);
        assert_eq!(c2.get("two").unwrap()[0], 0.75);
        assert!(c2.get("three").is_none());
    }

    #[test]
    fn missing_file_loads_empty() {
        let dir = tmpdir("missingfile");
        assert!(Cache::load(&dir.join("nope.bin")).is_empty());
    }

    #[test]
    fn resolve_reports_hits_and_misses() {
        let dir = tmpdir("resolve");
        let mut cache = Cache::load(&dir.join("cache.bin"));
        cache.insert("from cache", vec_of(0.2));

        let mut stored = HashMap::new();
        stored.insert("from hook".to_string(), vec_of(0.9));

        let prompts = vec![
            "from hook".to_string(),
            "from cache".to_string(),
            "brand new".to_string(),
        ];
        let (hits, misses) = resolve(&prompts, &stored, &cache);

        assert_eq!(hits.len(), 2);
        assert_eq!(hits["from hook"][0], 0.9);
        assert_eq!(hits["from cache"][0], 0.2);
        assert_eq!(misses, vec!["brand new".to_string()]);
    }

    #[test]
    fn all_hits_yields_no_misses() {
        let dir = tmpdir("allhits");
        let mut cache = Cache::load(&dir.join("cache.bin"));
        cache.insert("a", vec_of(0.1));
        cache.insert("b", vec_of(0.2));

        let prompts = vec!["a".to_string(), "b".to_string()];
        let (hits, misses) = resolve(&prompts, &HashMap::new(), &cache);

        assert_eq!(hits.len(), 2);
        assert!(misses.is_empty(), "no text session should be needed");
    }

    #[test]
    fn duplicate_misses_are_deduped() {
        let dir = tmpdir("dedupe");
        let cache = Cache::load(&dir.join("cache.bin"));
        let prompts = vec!["same".to_string(), "same".to_string(), "other".to_string()];
        let (_, misses) = resolve(&prompts, &HashMap::new(), &cache);
        assert_eq!(misses, vec!["same".to_string(), "other".to_string()]);
    }

    #[test]
    fn hook_vectors_take_precedence_over_cache() {
        let dir = tmpdir("precedence");
        let mut cache = Cache::load(&dir.join("cache.bin"));
        cache.insert("shared", vec_of(0.1));

        let mut stored = HashMap::new();
        stored.insert("shared".to_string(), vec_of(0.9));

        let prompts = vec!["shared".to_string()];
        let (hits, _) = resolve(&prompts, &stored, &cache);
        assert_eq!(hits["shared"][0], 0.9);
    }
}
