//! clipd — CLIP image ranking over an API-key'd webhook.

mod cache;
mod fetch;
mod hooks;
mod preprocess;
mod score;
mod text;
mod vision;

use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use tiny_http::{Header, Request, Response, Server};

use cache::Cache;
use fetch::Fetcher;
use hooks::Store;
use text::Text;
use vision::Vision;

struct App {
    store: Mutex<Store>,
    cache: Mutex<Cache>,
    vision: Mutex<Vision>,
    fetcher: Fetcher,
    admin_password: String,
    model_dir: PathBuf,
}

#[derive(Deserialize)]
struct RankRequest {
    images: Vec<String>,
    #[serde(default)]
    labels: Option<HashMap<String, String>>,
}

#[derive(Deserialize)]
struct HookRequest {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    labels: Option<HashMap<String, String>>,
}

fn main() {
    let (admin_password, pepper) = match hooks::require_secrets() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("clipd: {e}");
            std::process::exit(1);
        }
    };

    let allowlist: Vec<String> = std::env::var("CLIPD_URL_ALLOWLIST")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if allowlist.is_empty() {
        eprintln!("clipd: CLIPD_URL_ALLOWLIST must list at least one hostname");
        std::process::exit(1);
    }

    let data_dir = PathBuf::from(env_or("DATA_DIR", "/data"));
    let model_dir = PathBuf::from(env_or("MODEL_DIR", "/models"));
    let port = env_or("PORT", "8080");

    let vision = match Vision::load(&model_dir.join("vision.onnx")) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("clipd: cannot load vision model: {e}");
            std::process::exit(1);
        }
    };

    let app = App {
        store: Mutex::new(Store::open(&data_dir.join("hooks.json"), pepper.as_bytes())),
        cache: Mutex::new(Cache::load(&data_dir.join("label_cache.bin"))),
        vision: Mutex::new(vision),
        fetcher: Fetcher::new(allowlist),
        admin_password,
        model_dir,
    };

    let addr = format!("0.0.0.0:{port}");
    let server = Server::http(&addr).unwrap_or_else(|e| {
        eprintln!("clipd: cannot bind {addr}: {e}");
        std::process::exit(1);
    });
    eprintln!("clipd listening on {addr}");

    for request in server.incoming_requests() {
        handle(&app, request);
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn handle(app: &App, mut request: Request) {
    let method = request.method().as_str().to_string();
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or("").to_string();
    let auth = bearer(&request);

    let body = match read_body(&mut request) {
        Ok(b) => b,
        Err(e) => return send(request, 400, json!({"error": e})),
    };

    let segments: Vec<&str> = path.trim_matches('/').split('/').collect();

    let (status, payload) = match (method.as_str(), segments.as_slice()) {
        ("GET", ["healthz"]) => {
            let _ = request.respond(Response::from_string("ok"));
            return;
        }
        ("POST", ["h", id]) => rank(app, id, &auth, &body),
        (_, ["admin", ..]) => {
            if !hooks::admin_ok(&app.admin_password, &auth) {
                (401, json!({"error": "unauthorized"}))
            } else {
                admin(app, &method, &segments[1..], &body)
            }
        }
        _ => (404, json!({"error": "not found"})),
    };

    send(request, status, payload);
}

fn rank(app: &App, id: &str, key: &str, body: &str) -> (u16, Value) {
    if !app.store.lock().unwrap().verify(id, key) {
        return (401, json!({"error": "unauthorized"}));
    }

    let req: RankRequest = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => return (400, json!({"error": format!("bad request: {e}")})),
    };
    if req.images.is_empty() {
        return (400, json!({"error": "images must not be empty"}));
    }

    let hook = match app.store.lock().unwrap().get(id).cloned() {
        Some(h) => h,
        None => return (404, json!({"error": "not found"})),
    };

    let labels = req.labels.unwrap_or_else(|| hook.labels.clone());
    if labels.len() < 2 {
        return (400, json!({"error": "at least 2 labels required"}));
    }

    let vectors = match resolve_vectors(app, &labels, &hook.label_vectors) {
        Ok(v) => v,
        Err(e) => return (500, json!({"error": e})),
    };

    let mut results = Vec::new();
    for url in &req.images {
        let image = match app.fetcher.fetch(url) {
            Ok(i) => i,
            Err(e) => {
                results.push(json!({"image": url, "error": e}));
                continue;
            }
        };
        let pixels = preprocess::preprocess(&image);
        let embedding = match app.vision.lock().unwrap().embed(&pixels) {
            Ok(v) => v,
            Err(e) => {
                results.push(json!({"image": url, "error": e.to_string()}));
                continue;
            }
        };
        let ranked = score::rank(&embedding, &vectors);
        results.push(json!({
            "image": url,
            "scores": ranked.scores,
            "top": ranked.top,
        }));
    }

    app.store.lock().unwrap().touch(id);
    (200, json!({"results": results}))
}

/// Label id -> vector, embedding any prompts not already known.
/// The text session is created and dropped inside this function.
fn resolve_vectors(
    app: &App,
    labels: &HashMap<String, String>,
    stored: &HashMap<String, Vec<f32>>,
) -> Result<Vec<(String, Vec<f32>)>, String> {
    let prompts: Vec<String> = labels.values().cloned().collect();
    let (mut hits, misses) = {
        let cache = app.cache.lock().unwrap();
        cache::resolve(&prompts, stored, &cache)
    };

    let mut owned: HashMap<String, Vec<f32>> =
        hits.drain().map(|(k, v)| (k.to_string(), v)).collect();

    if !misses.is_empty() {
        let mut text = Text::load(
            &app.model_dir.join("text.onnx"),
            &app.model_dir.join("tokenizer.json"),
        )
        .map_err(|e| e.to_string())?;
        let embedded = text.embed_batch(&misses).map_err(|e| e.to_string())?;
        drop(text);

        let mut cache = app.cache.lock().unwrap();
        for (prompt, vec) in misses.iter().zip(embedded) {
            cache.insert(prompt, vec.clone());
            owned.insert(prompt.clone(), vec);
        }
        let _ = cache.save();
    }

    let mut out = Vec::new();
    for (id, prompt) in labels {
        let vec = owned
            .get(prompt)
            .ok_or_else(|| format!("no embedding for label {id}"))?;
        out.push((id.clone(), vec.clone()));
    }
    Ok(out)
}

fn admin(app: &App, method: &str, segments: &[&str], body: &str) -> (u16, Value) {
    match (method, segments) {
        ("GET", ["hooks"]) => {
            let store = app.store.lock().unwrap();
            (200, serde_json::to_value(store.list()).unwrap_or(json!([])))
        }

        ("POST", ["hooks"]) => {
            let req: HookRequest = match serde_json::from_str(body) {
                Ok(r) => r,
                Err(e) => return (400, json!({"error": format!("bad request: {e}")})),
            };
            let (Some(name), Some(labels)) = (req.name, req.labels) else {
                return (400, json!({"error": "name and labels are required"}));
            };
            let vectors = match embed_labels(app, &labels) {
                Ok(v) => v,
                Err(e) => return (400, json!({"error": e})),
            };
            match app.store.lock().unwrap().create(name, labels, vectors) {
                Ok((id, key)) => (201, json!({"id": id, "key": key})),
                Err(e) => (400, json!({"error": e})),
            }
        }

        ("GET", ["hooks", id]) => {
            let store = app.store.lock().unwrap();
            match store.get(id) {
                Some(h) => (200, serde_json::to_value(h.view(id)).unwrap_or(json!({}))),
                None => (404, json!({"error": "not found"})),
            }
        }

        ("PATCH", ["hooks", id]) => {
            let req: HookRequest = match serde_json::from_str(body) {
                Ok(r) => r,
                Err(e) => return (400, json!({"error": format!("bad request: {e}")})),
            };
            let vectors = match &req.labels {
                Some(l) => match embed_labels(app, l) {
                    Ok(v) => Some(v),
                    Err(e) => return (400, json!({"error": e})),
                },
                None => None,
            };
            let mut store = app.store.lock().unwrap();
            match store.update(id, req.name, req.labels, vectors) {
                Ok(()) => match store.get(id) {
                    Some(h) => (200, serde_json::to_value(h.view(id)).unwrap_or(json!({}))),
                    None => (404, json!({"error": "not found"})),
                },
                Err(e) if e == "no such hook" => (404, json!({"error": "not found"})),
                Err(e) => (400, json!({"error": e})),
            }
        }

        ("POST", ["hooks", id, "rotate"]) => match app.store.lock().unwrap().rotate(id) {
            Ok(key) => (200, json!({"id": id, "key": key})),
            Err(_) => (404, json!({"error": "not found"})),
        },

        ("DELETE", ["hooks", id]) => match app.store.lock().unwrap().delete(id) {
            Ok(()) => (204, json!(null)),
            Err(_) => (404, json!({"error": "not found"})),
        },

        _ => (404, json!({"error": "not found"})),
    }
}

fn embed_labels(
    app: &App,
    labels: &HashMap<String, String>,
) -> Result<HashMap<String, Vec<f32>>, String> {
    let vectors = resolve_vectors(app, labels, &HashMap::new())?;
    let mut out = HashMap::new();
    for (id, vec) in vectors {
        if let Some(prompt) = labels.get(&id) {
            out.insert(prompt.clone(), vec);
        }
    }
    Ok(out)
}

fn bearer(request: &Request) -> String {
    request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Authorization"))
        .map(|h| h.value.as_str().to_string())
        .and_then(|v| {
            let lower = v.to_ascii_lowercase();
            lower
                .starts_with("bearer ")
                .then(|| v[7..].trim().to_string())
        })
        .unwrap_or_default()
}

fn read_body(request: &mut Request) -> Result<String, String> {
    use std::io::Read;
    let mut buf = String::new();
    request
        .as_reader()
        .take(1024 * 1024)
        .read_to_string(&mut buf)
        .map_err(|e| format!("unreadable body: {e}"))?;
    Ok(buf)
}

fn send(request: Request, status: u16, payload: Value) {
    if status == 204 {
        let _ = request.respond(Response::empty(204));
        return;
    }
    let body = payload.to_string();
    let header =
        Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).expect("static header");
    let _ = request.respond(
        Response::from_string(body)
            .with_status_code(status)
            .with_header(header),
    );
}
