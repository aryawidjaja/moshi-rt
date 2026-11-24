#!/usr/bin/env bash
# Headless end-to-end check of the web app's model server on Modal: start it with `modal serve`,
# wait until the model is loaded, stream a test clip through the websocket (app/test_client.py),
# then stop the server. Usage: bash scripts/app_smoke.sh [--stock] [seconds]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STOCK=""; SECONDS_ARG=20
for a in "$@"; do case "$a" in --stock) STOCK=1 ;; *) SECONDS_ARG="$a" ;; esac; done
TOKEN=$("$ROOT/.venv/bin/python" -c 'import secrets; print(secrets.token_urlsafe(16))')
LOG="$ROOT/.cache/remote/app_smoke.log"; mkdir -p "$ROOT/.cache/remote"
(cd "$ROOT" && MOSHI_RT_TOKEN="$TOKEN" MOSHI_RT_STOCK="$STOCK" "$ROOT/.venv/bin/modal" serve app/modal_app.py >"$LOG" 2>&1) &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
REMOTE=""
for _ in $(seq 1 120); do
  REMOTE=$(grep -oE 'https://[A-Za-z0-9.-]+\.modal\.run' "$LOG" | head -1 || true)
  [ -n "$REMOTE" ] && break
  kill -0 $SERVER 2>/dev/null || { cat "$LOG"; exit 1; }
  sleep 2
done
[ -n "$REMOTE" ] || { cat "$LOG"; echo "no URL from modal serve"; exit 1; }
echo "server: $REMOTE"
for _ in $(seq 1 120); do
  curl -sf --max-time 30 "$REMOTE/api/health" && break
  kill -0 $SERVER 2>/dev/null || { cat "$LOG"; exit 1; }
  sleep 5
done
echo
"$ROOT/.venv/bin/python" "$ROOT/app/test_client.py" --server "$REMOTE" --token "$TOKEN" --seconds "$SECONDS_ARG"
