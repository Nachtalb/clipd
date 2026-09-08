#!/usr/bin/env bash
# End-to-end smoke test against a running clipd binary.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PORT=18080
BASE="http://127.0.0.1:$PORT"
DATA=$(mktemp -d)
ADMIN="smoke-admin-password-long-enough"
PEPPER="smoke-pepper-that-is-at-least-32b!!"
PASS=0; FAIL=0

check() { # name expected actual
  if [ "$2" = "$3" ]; then echo "  ok   $1 ($3)"; PASS=$((PASS+1));
  else echo "  FAIL $1: expected $2, got $3"; FAIL=$((FAIL+1)); fi
}

# Serve a local fixture image so the allowlist has something real to permit.
python3 - "$DATA" <<'EOF' &
import http.server, socketserver, sys, struct, zlib, os
d = sys.argv[1]
def png(path, rgb):
    w = h = 64
    raw = b''.join(b'\x00' + bytes(rgb) * w for _ in range(h))
    def chunk(t, data):
        c = t + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c))
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    open(path, 'wb').write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr)
        + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
png(os.path.join(d, 'red.png'), (220, 30, 30))
os.chdir(d)
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", 18081), http.server.SimpleHTTPRequestHandler) as s:
    s.serve_forever()
EOF
IMGPID=$!

CLIPD_ADMIN_PASSWORD="$ADMIN" CLIPD_KEY_PEPPER="$PEPPER" \
CLIPD_URL_ALLOWLIST="127.0.0.1" DATA_DIR="$DATA" MODEL_DIR="$PWD/models" PORT="$PORT" \
  ./target/release/clipd & CLIPDPID=$!

cleanup() { kill $CLIPDPID $IMGPID 2>/dev/null; rm -rf "$DATA"; }
trap cleanup EXIT

for _ in $(seq 1 40); do
  curl -sf "$BASE/healthz" >/dev/null 2>&1 && break
  sleep 0.25
done

code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

echo "== health + auth"
check "healthz"            200 "$(code $BASE/healthz)"
check "admin no auth"      401 "$(code $BASE/admin/hooks)"
check "admin wrong pw"     401 "$(code -H 'Authorization: Bearer nope' $BASE/admin/hooks)"
check "unknown route"      404 "$(code -H "Authorization: Bearer $ADMIN" $BASE/nope)"

echo "== hook lifecycle"
LABELS='{"name":"smoke","labels":{"red":"a photo of a red square","blue":"a photo of a blue square"}}'
CREATE=$(curl -s -X POST -H "Authorization: Bearer $ADMIN" -H 'Content-Type: application/json' \
  --data-binary "$LABELS" "$BASE/admin/hooks")
ID=$(echo "$CREATE" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("id",""))')
KEY=$(echo "$CREATE" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("key",""))')
[ -n "$ID" ] && { echo "  ok   created hook $ID"; PASS=$((PASS+1)); } || { echo "  FAIL create: $CREATE"; FAIL=$((FAIL+1)); }
case "$KEY" in clipd_*) echo "  ok   key has clipd_ prefix"; PASS=$((PASS+1));; *) echo "  FAIL key prefix: $KEY"; FAIL=$((FAIL+1));; esac

check "too few labels"     400 "$(code -X POST -H "Authorization: Bearer $ADMIN" \
  --data-binary '{"name":"x","labels":{"only":"a photo of one thing"}}' $BASE/admin/hooks)"

LIST=$(curl -s -H "Authorization: Bearer $ADMIN" "$BASE/admin/hooks")
echo "$LIST" | grep -q '"key"' && { echo "  FAIL list leaks key"; FAIL=$((FAIL+1)); } \
  || { echo "  ok   list has no key field"; PASS=$((PASS+1)); }
echo "$LIST" | grep -q 'key_hash' && { echo "  FAIL list leaks key_hash"; FAIL=$((FAIL+1)); } \
  || { echo "  ok   list has no key_hash"; PASS=$((PASS+1)); }
echo "$LIST" | grep -q 'key_preview' && { echo "  ok   list has key_preview"; PASS=$((PASS+1)); } \
  || { echo "  FAIL list missing key_preview"; FAIL=$((FAIL+1)); }

echo "== data plane"
check "no auth"            401 "$(code -X POST --data-binary '{"images":[]}' $BASE/h/$ID)"
check "wrong key"          401 "$(code -X POST -H 'Authorization: Bearer clipd_wrong' \
  --data-binary '{"images":[]}' $BASE/h/$ID)"
check "empty images"       400 "$(code -X POST -H "Authorization: Bearer $KEY" \
  --data-binary '{"images":[]}' $BASE/h/$ID)"

RANK=$(curl -s -X POST -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  --data-binary '{"images":["http://127.0.0.1:18081/red.png"]}' "$BASE/h/$ID")
TOP=$(echo "$RANK" | python3 -c 'import sys,json;r=json.load(sys.stdin)["results"][0];print(r.get("top") or r.get("error"))')
check "red image ranks red" "red" "$TOP"

CUSTOM=$(curl -s -X POST -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  --data-binary '{"images":["http://127.0.0.1:18081/red.png"],"labels":{"warm":"a warm crimson colour","cold":"a freezing blue glacier"}}' "$BASE/h/$ID")
echo "$CUSTOM" | grep -q '"warm"' && { echo "  ok   custom labels scored"; PASS=$((PASS+1)); } \
  || { echo "  FAIL custom labels: $CUSTOM"; FAIL=$((FAIL+1)); }

echo "== ssrf guard"
BLOCKED=$(curl -s -X POST -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  --data-binary '{"images":["http://169.254.169.254/latest/meta-data/"]}' "$BASE/h/$ID")
echo "$BLOCKED" | grep -qi 'ALLOWLIST' && { echo "  ok   metadata endpoint blocked"; PASS=$((PASS+1)); } \
  || { echo "  FAIL ssrf: $BLOCKED"; FAIL=$((FAIL+1)); }

echo "== rotate + delete"
NEWKEY=$(curl -s -X POST -H "Authorization: Bearer $ADMIN" "$BASE/admin/hooks/$ID/rotate" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin).get("key",""))')
check "old key rejected"   401 "$(code -X POST -H "Authorization: Bearer $KEY" \
  --data-binary '{"images":[]}' $BASE/h/$ID)"
check "new key accepted"   400 "$(code -X POST -H "Authorization: Bearer $NEWKEY" \
  --data-binary '{"images":[]}' $BASE/h/$ID)"
check "delete"             204 "$(code -X DELETE -H "Authorization: Bearer $ADMIN" $BASE/admin/hooks/$ID)"
check "key dead after del" 401 "$(code -X POST -H "Authorization: Bearer $NEWKEY" \
  --data-binary '{"images":[]}' $BASE/h/$ID)"
check "get deleted hook"   404 "$(code -H "Authorization: Bearer $ADMIN" $BASE/admin/hooks/$ID)"

echo "== startup refuses weak secrets"
OUT=$(CLIPD_ADMIN_PASSWORD="" CLIPD_KEY_PEPPER="$PEPPER" CLIPD_URL_ALLOWLIST=x \
  ./target/release/clipd 2>&1); [ $? -ne 0 ] && { echo "  ok   empty admin password refused"; PASS=$((PASS+1)); } \
  || { echo "  FAIL started without admin password"; FAIL=$((FAIL+1)); }
OUT=$(CLIPD_ADMIN_PASSWORD="$ADMIN" CLIPD_KEY_PEPPER="short" CLIPD_URL_ALLOWLIST=x \
  ./target/release/clipd 2>&1); [ $? -ne 0 ] && { echo "  ok   short pepper refused"; PASS=$((PASS+1)); } \
  || { echo "  FAIL started with short pepper"; FAIL=$((FAIL+1)); }
OUT=$(CLIPD_ADMIN_PASSWORD="$ADMIN" CLIPD_KEY_PEPPER="$PEPPER" CLIPD_URL_ALLOWLIST="" \
  ./target/release/clipd 2>&1); [ $? -ne 0 ] && { echo "  ok   empty allowlist refused"; PASS=$((PASS+1)); } \
  || { echo "  FAIL started with empty allowlist"; FAIL=$((FAIL+1)); }

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
