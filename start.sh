#!/usr/bin/env bash
# Start the moshi-rt web app and talk to Moshi in the browser.
#   ./start.sh            local NVIDIA GPU if there is one, otherwise a temporary L4 on your Modal account
#   ./start.sh --modal    always use Modal
#   ./start.sh --local    always use the local GPU
#   ./start.sh --stock    stock bf16 Moshi instead of moshi-rt (for comparison)
# Ctrl-C stops the app; on Modal the GPU is released when the command exits.
set -euo pipefail
cd "$(dirname "$0")"

MODE=auto; STOCK=""
for a in "$@"; do
  case "$a" in
    --modal) MODE=modal ;;
    --local) MODE=local ;;
    --stock) STOCK=1 ;;
    -h|--help) sed -n 2,7p "$0"; exit 0 ;;
    *) echo "unknown option: $a"; exit 2 ;;
  esac
done
if [ "$MODE" = auto ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then MODE=local; else MODE=modal; fi
fi

VENV=.venv
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
PY="$VENV/bin/python"
TOKEN=$("$PY" -c 'import secrets; print(secrets.token_urlsafe(16))')
PIDS=""
cleanup() { for p in $PIDS; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM HUP
open_url() {
  if command -v open >/dev/null 2>&1; then open "$1"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1" >/dev/null 2>&1
  else echo "open this in your browser: $1"; fi
}
wait_alive() { kill -0 "$1" 2>/dev/null || { echo "server exited; log above"; exit 1; }; }
wait_ready() {  # wait_ready <health url> <server pid> <label>: spinner and elapsed time until the server answers
  local url="$1" pid="$2" label="$3" t0=$SECONDS i=0 s chk
  local frames=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏)
  ( until curl -sf --max-time 20 "$url" >/dev/null 2>&1; do sleep 2; done ) &
  chk=$!; PIDS="$PIDS $chk"
  while kill -0 "$chk" 2>/dev/null; do
    s=$((SECONDS - t0))
    kill -0 "$pid" 2>/dev/null || { echo; [ -n "${LOG:-}" ] && tail -20 "$LOG"; echo "server exited"; exit 1; }
    if [ $s -gt 900 ]; then echo; echo "no answer after 15 minutes${LOG:+ (log: $LOG)}; stopping."; exit 1; fi
    if [ -t 1 ]; then
      printf "\r\033[K  %s %s  %d:%02d" "${frames[$i]}" "$label" $((s / 60)) $((s % 60))
      i=$(((i + 1) % 10)); sleep 0.15
    else sleep 5; fi
  done
  s=$((SECONDS - t0))
  if [ -t 1 ]; then printf "\r\033[K  ✓ %s  %d:%02d\n" "$label" $((s / 60)) $((s % 60)); fi
}
need() {  # need <import name> <pip spec...>: install only if the module is missing
  local mod="$1"; shift
  "$PY" -c "import $mod" >/dev/null 2>&1 && return 0
  "$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
  echo "installing $* (first run only) ..."
  "$PY" -m pip install -q "$@"
}

if [ "$MODE" = local ]; then
  need moshi "moshi==0.2.13"
  echo "loading the model on the local GPU (the first run downloads about 15 GB) ..."
  "$PY" -m app.server --port 8998 --token "$TOKEN" ${STOCK:+--stock} &
  SERVER=$!; PIDS="$SERVER"
  wait_ready http://127.0.0.1:8998/api/health "$SERVER" "loading Moshi on the local GPU"
  URL="http://127.0.0.1:8998/?token=$TOKEN"
else
  need modal modal
  if [ ! -f "$HOME/.modal.toml" ] && [ -z "${MODAL_TOKEN_ID:-}" ]; then
    echo "Modal is not set up yet: a browser window will open to log in (a free account works)."
    "$VENV/bin/modal" setup
  fi
  LOG=$(mktemp -t moshi-rt.XXXX)
  MOSHI_RT_TOKEN="$TOKEN" MOSHI_RT_STOCK="$STOCK" "$VENV/bin/modal" serve app/modal_app.py >"$LOG" 2>&1 &
  SERVER=$!; PIDS="$SERVER"
  REMOTE=""
  until [ -n "$REMOTE" ]; do
    sleep 2
    if ! kill -0 "$SERVER" 2>/dev/null; then cat "$LOG"; echo "modal serve exited"; exit 1; fi
    REMOTE=$(grep -oE 'https://[A-Za-z0-9.-]+\.modal\.run' "$LOG" | head -1 || true)
  done
  echo "starting an L4 on your Modal account: $REMOTE"
  echo "(usually 1-2 minutes; the very first start also downloads the weights, about 5 minutes)"
  wait_ready "$REMOTE/api/health" "$SERVER" "starting the model server"
  "$PY" -m http.server 8765 --bind 127.0.0.1 --directory app/web >/dev/null 2>&1 &
  PIDS="$PIDS $!"
  sleep 1
  URL="http://127.0.0.1:8765/?server=$REMOTE&token=$TOKEN"
fi

echo "ready: $URL"
echo "Ctrl-C to stop."
open_url "$URL"
wait "$SERVER"
