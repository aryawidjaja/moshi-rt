#!/usr/bin/env bash
# Run a repo command on a Google Colab GPU session and copy small outputs back.
# Usage: scripts/remote.sh <session> <command...>
#   e.g. scripts/remote.sh moshi python bench/stream_bench.py --tag bf16_b1
# Uses `colab exec` (long `colab ssh` commands can hang after the remote side exits).
# Copies back results/, profiles/ and audio/ files under 2 MB.
# The session is stopped after MAX_MIN minutes (default 30) in any case.
set -euo pipefail
SESSION="$1"; shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLAB="$ROOT/.venv/bin/colab"
TMP="$ROOT/.cache/remote"; mkdir -p "$TMP"
GIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo none)"
if [ -n "$(git -C "$ROOT" status --porcelain -- bench quant kernels scripts 2>/dev/null)" ]; then GIT="$GIT-dirty"; fi

# code only: never weights, docs or results
tar czf "$TMP/code.tgz" -C "$ROOT" --exclude=__pycache__ --exclude='*-settings.json' \
  bench quant kernels scripts $(cd "$ROOT" && ls audio/*.wav 2>/dev/null)
"$COLAB" upload -s "$SESSION" "$TMP/code.tgz" /content/code.tgz >/dev/null

CMD="$*"
STATUS=0

# stop the session after MAX_MIN minutes even if this script hangs
MAX_MIN="${MAX_MIN:-30}"
( sleep $((MAX_MIN * 60)) && "$COLAB" stop -s "$SESSION" >/dev/null 2>&1 ) &
WATCHDOG=$!
trap 'kill $WATCHDOG 2>/dev/null || true' EXIT
# --timeout: weight downloads and model loads are silent for minutes; the default 30 s kills the run.
# The watchdog above still bounds the whole job.
"$COLAB" exec -s "$SESSION" --timeout 900 <<EOF || STATUS=$?
import os, subprocess, sys
os.makedirs("/content/moshi-rt", exist_ok=True)
subprocess.run(["tar", "xzf", "/content/code.tgz", "-C", "/content/moshi-rt"], check=True)
for d in ("results", "profiles", "audio/generated"):
    os.makedirs("/content/moshi-rt/" + d, exist_ok=True)
env = dict(os.environ, MOSHI_RT_MACHINE="colab-L4", MOSHI_RT_GIT="$GIT", PYTHONUNBUFFERED="1")
p = subprocess.Popen(["bash", "-c", """$CMD"""], cwd="/content/moshi-rt", env=env,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    if "it/s]" not in line and "Fetching" not in line:
        print(line, end="", flush=True)
rc = p.wait()
subprocess.run("cd /content/moshi-rt && find results profiles audio audio/generated -maxdepth 1 -type f "
               "\\\\( -name '*.json' -o -name '*.csv' -o -name '*.txt' -o -name '*.md' -o -name '*.wav' -o -name '*.png' \\\\) "
               "-size -2M | tar czf /content/out.tgz -T -", shell=True, check=True)
print("REMOTE_EXIT", rc)
if rc:
    raise SystemExit(rc)
EOF

"$COLAB" download -s "$SESSION" /content/out.tgz "$TMP/out.tgz" >/dev/null
tar xzf "$TMP/out.tgz" -C "$ROOT"
rm -f "$TMP/out.tgz" "$TMP/code.tgz"
exit $STATUS
