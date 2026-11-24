"""Run the moshi-rt web server on a Modal L4 under your own Modal account.

    MOSHI_RT_TOKEN=secret modal serve app/modal_app.py      # live while the command runs
    MOSHI_RT_STOCK=1 ... modal serve app/modal_app.py      # stock bf16 instead of moshi-rt

`start.sh` does this for you and opens the web app. Model weights are cached in the Modal Volume
'moshi-rt-hf', so only the first start downloads them. The GPU scales to zero a few minutes after
the last connection closes and stops when `modal serve` exits.
"""
import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/moshi-rt"
PORT = 8998

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("moshi==0.2.13")
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
)
for d in ("app", "bench", "kernels", "quant"):
    image = image.add_local_dir(ROOT / d, f"{REMOTE}/{d}", ignore=["**/__pycache__"])

app = modal.App("moshi-rt-app", image=image)
hf_cache = modal.Volume.from_name("moshi-rt-hf", create_if_missing=True)
settings = modal.Secret.from_dict({"MOSHI_RT_TOKEN": os.environ.get("MOSHI_RT_TOKEN", ""),
                                   "MOSHI_RT_STOCK": os.environ.get("MOSHI_RT_STOCK", "")})


@app.function(gpu="L4", volumes={"/cache": hf_cache}, secrets=[settings], timeout=3600,
              scaledown_window=180, max_containers=1, memory=24 * 1024)  # the int8 loader stages bf16 on the CPU
@modal.concurrent(max_inputs=4)
@modal.web_server(PORT, startup_timeout=900)
def serve():
    cmd = ["python", "-m", "app.server", "--host", "0.0.0.0", "--port", str(PORT)]
    if os.environ.get("MOSHI_RT_STOCK"):
        cmd.append("--stock")
    subprocess.Popen(cmd, cwd=REMOTE)
