# CLAUDE.md

Agent guidance for `clipd` — a CLIP image-ranking service behind API-key'd
webhooks. The README is the user-facing contract; this file is what an agent
needs on top of it.

## Read first

- [`README.md`](README.md) — what it does, endpoints, env vars, key format.

## What this is, and is not

A **generic** ranking tool. It takes image URLs and labelled prompts and returns
a cosine score per label. It knows nothing about any caller, and it must stay
that way: no caller-specific labels, thresholds, verdict logic, or naming
anywhere in this repo. Callers orchestrate; clipd ranks.

It **ranks**, it does not detect. There is no "none of these" — with N labels
one always wins, even when every score is poor. Any decision built on the output
belongs to the caller.

## Layout

```
src/
  main.rs        routes, auth, request handling
  hooks.rs       hook CRUD, key hashing, validation  (largest module)
  fetch.rs       image fetching + per-hook host allowlist (SSRF guard)
  preprocess.rs  resize/crop/normalise to the model's input tensor
  vision.rs      image encoder session
  text.rs        text encoder session — loaded on demand, then dropped
  score.rs       cosine + softmax
  cache.rs       label vector cache on disk
scripts/
  fetch-models.sh  ~155 MB of ONNX into ./models
  smoke.sh         end-to-end checks against a live binary
```

## Invariants — do not break these

- **The allowlist is per hook, required, and cannot be emptied.** A bare `"*"`
  is refused. There is no global allowlist env var and no opt-out. This is the
  SSRF boundary: one hook's key must never reach another hook's sources or a
  cloud metadata endpoint.
- **Keys are never stored in plaintext.** `HMAC-SHA256(pepper, key)` plus a
  masked preview. The plaintext is returned by create and rotate only. Nothing
  may add a way to read a key back.
- **Missing and wrong credentials return an identical 401.** No oracle.
- **The text encoder is not resident.** Prompts are embedded when a hook is
  created or updated and cached to disk; ordinary traffic never loads the text
  session. Do not make it a long-lived field to "save time".
- **No CLI.** Configuration is environment variables; management is the admin
  HTTP plane. Do not add flags or subcommands.
- **Model weights ship inside the image.** No PVC, no download-at-boot.

## Toolchain

Rust pinned by `rust-toolchain.toml`. CI runs clippy with `-D warnings`, so a
lint is a build failure.

```bash
./scripts/fetch-models.sh     # once; models are gitignored
cargo test                    # unit tests, no network, no model files needed
cargo clippy --all-targets -- -D warnings
cargo build --release
./scripts/smoke.sh            # spawns a real binary + a local image server
```

`smoke.sh` needs the models present and a free port. It is the only thing that
exercises fetch, preprocess and inference together — run it before claiming an
end-to-end change works.

## Deploy

Push to `master` builds and pushes `ghcr.io/nachtalb/clipd:latest` plus a
`sha-<full>` tag. A `v*` tag additionally publishes the semver tag via
`docker/metadata-action`. There is no GitHub Release job.

The k8s manifests live in the **infra** repo (`k8s/pods/clipd/`), not here.
Deployment is cluster-internal only — no HTTPRoute, because the admin plane
shares the port.

## Gotchas

- **Memory is the binding constraint, not CPU.** Measure before raising limits:
  a 400Mi cgroup survived 80 label embeddings across two text-session loads at a
  241.6 MiB peak.
- **glibc `__isoc23_*` symbols** mean the build image is newer than the runtime
  image. Keep both on the same Debian generation.
- **`/data` must be writable by uid 65534** — the container is non-root, so a
  fresh volume needs its ownership fixed before the process starts.
- **Wikimedia and similar hosts 403 a default user agent.** The fetcher sends a
  UA with a contact URL; do not strip it.
- **`grep -c` exits 1 on zero matches**, which kills `&&` chains in scripts.

## Working agreements

- One logical change per commit; messages explain why, in prose.
- `--no-gpg-sign`.
- Never commit models, keys, or a real `CLIPD_ADMIN_PASSWORD` / `CLIPD_KEY_PEPPER`.
- The repo is public. Nothing about who runs it, what they run it for, or any
  private hostname belongs in code, comments, commits, or docs.
