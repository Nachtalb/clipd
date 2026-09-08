# clipd

CLIP image ranking over an API-key'd webhook. Give it image URLs and a set of
labelled prompts; it returns a similarity score per label.

Runs on CPU. No GPU, no Python, no framework — a single Rust binary with
ONNX Runtime and CLIP ViT-B/32 (int8).

## What it does

`clipd` embeds each image once and each label prompt once, then scores every
pair by cosine similarity. It **ranks**; it does not detect. There are no
bounding boxes, and there is no "none of these" — the highest-scoring label
always wins, even when every score is poor. Include a catch-all label and look
at the raw cosine values, not just the softmax.

## Quick start

```bash
./scripts/fetch-models.sh          # ~155 MB into ./models
docker compose up --build
```

```bash
ADMIN='your-admin-password-here!!'

# create a hook — the key is shown exactly once
curl -sX POST localhost:8080/admin/hooks \
  -H "Authorization: Bearer $ADMIN" \
  -H 'Content-Type: application/json' \
  -d '{"name":"scenes",
       "allowlist":["f.example.com"],
       "labels":{
        "landscape":"a photograph of an outdoor landscape",
        "portrait":"a photograph of a person",
        "other":"a photograph of something else"}}'
# → {"id":"a1b2c3d4","key":"clipd_..."}

# rank an image
curl -sX POST localhost:8080/h/a1b2c3d4 \
  -H "Authorization: Bearer clipd_..." \
  -H 'Content-Type: application/json' \
  -d '{"images":["https://example.com/photo.jpg"]}'
```

```json
{
  "results": [
    {
      "image": "https://example.com/photo.jpg",
      "scores": {
        "landscape": {"cosine": 0.2338, "softmax": 0.4496},
        "other":     {"cosine": 0.2316, "softmax": 0.3622},
        "portrait":  {"cosine": 0.2251, "softmax": 0.1883}
      },
      "top": "landscape"
    }
  ]
}
```

## Configuration

All configuration is environment variables. There are no command-line flags.

| Variable | Required | Default | Notes |
|---|---|---|---|
| `CLIPD_ADMIN_PASSWORD` | yes | — | ≥16 characters; guards `/admin/*` |
| `CLIPD_KEY_PEPPER` | yes | — | ≥32 bytes; HMAC pepper for hook keys |
| `DATA_DIR` | no | `/data` | `hooks.json`, `label_cache.bin` |
| `MODEL_DIR` | no | `/models` | Baked into the image |
| `PORT` | no | `8080` | |

The service refuses to start if any required variable is missing or too short.

## Endpoints

### Data plane — `Authorization: Bearer <hook_key>`

```
POST /h/<hook_id>
{"images": ["https://…"], "labels": {"id": "prompt"}}   # labels optional
```

Omit `labels` to use the hook's own set. Supplying them overrides for that one
request; new prompts are embedded on the fly and cached permanently.

### Admin plane — `Authorization: Bearer $CLIPD_ADMIN_PASSWORD`

```
GET    /admin/hooks
POST   /admin/hooks              {name, labels, allowlist}
GET    /admin/hooks/<id>
PATCH  /admin/hooks/<id>         {name?, labels?, allowlist?}
POST   /admin/hooks/<id>/rotate
DELETE /admin/hooks/<id>
```

`GET /healthz` is unauthenticated.

## Allowlist

Each hook carries its own list of hostnames it may fetch images from:

```json
{"allowlist": ["f.example.com", "*.cdn.example.com"]}
```

`*.example.com` matches `example.com` and any subdomain. Matching is on the
parsed host only, so `https://f.example.com@evil.com/x.jpg` does not match.
Required at create time; a bare `"*"` and an empty list are both refused.

## Labels

A map of your own id to a natural-language prompt:

```json
{"cat": "a photograph of a cat", "other": "a photograph of something else"}
```

Scores come back keyed by id, so you can reword a prompt without changing what
your downstream code matches on.

Write full sentences. `"a photograph of a cat"` scores measurably better than
`"cat"` — CLIP was trained on captions, not tags. At least two labels are
required.

## Keys

```
key      = clipd_<43 url-safe base64 chars>       # 256 bits from a CSPRNG
key_hash = HMAC-SHA256(CLIPD_KEY_PEPPER, key)
```

Only the hash and a masked preview are stored. The plaintext key is returned by
exactly two endpoints — create and rotate — and cannot be recovered afterwards.
Lose it and rotate.

**Rotating `CLIPD_KEY_PEPPER` invalidates every hook key at once.** Every hook
then needs `/rotate`. Treat the pepper as permanent.

## Security notes

- Outbound fetches are restricted to **each hook's own `allowlist`**. A hook can
  only fetch hosts it was created with, so one hook's key cannot be used to reach
  another hook's sources or a cloud metadata endpoint. The field is required at
  create time, cannot be emptied, and a bare `"*"` is refused — there is no
  implicit default and no way to opt out.
- Images are capped at 20 MB, requests at 1 MB, fetches at 10 s.
- Missing and wrong credentials both return an identical 401.

## Performance

Measured on one vCPU (Intel Haswell), CLIP ViT-B/32 int8, batch of 1:

| | |
|---|---|
| Vision inference | 150 ms (p50) |
| Vision model load | 0.44 s |
| Text model load | 0.35 s |
| Resident memory | ~180–200 MiB |
| Image size | 211 MB |

The text encoder is not held in memory. Label prompts are embedded when a hook
is created or updated, so ordinary traffic never loads it. A request carrying
unseen prompts loads it, embeds them in one batch, caches the vectors, and drops
the session.

## Development

```bash
./scripts/fetch-models.sh
cargo test              # 44 unit tests
cargo build --release
./scripts/smoke.sh      # 24 end-to-end checks against a live binary
```

## License

MIT
