#!/usr/bin/env bash
# One-time setup on a fresh Colab VM: pinned moshi (pulls torch 2.9.1+cu128).
set -e
pip install -q "moshi==0.2.13" 2>&1 | grep -v "requires\|incompatible" || true
python -c "import moshi, torch; print(\"moshi\", moshi.__version__, \"torch\", torch.__version__)"
