"""Run a repo command on a Modal L4 and copy small outputs back into the repo.
Usage (from the repo root):
  modal run bench/modal_app.py --cmd "python scripts/make_clips.py && python bench/stream_bench.py"
Model weights are cached in the Modal Volume 'moshi-rt-hf'."""
import io
import subprocess
import tarfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/moshi-rt"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .pip_install("moshi==0.2.13", "torchao>=0.14,<0.15", "scipy", "soundfile")
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
)
for d in ("bench", "quant", "kernels", "scripts", "audio"):
    image = image.add_local_dir(ROOT / d, f"{REMOTE}/{d}",
                                ignore=["**/__pycache__", "generated", "*-settings.json"])

app = modal.App("moshi-rt", image=image)
hf_cache = modal.Volume.from_name("moshi-rt-hf", create_if_missing=True)


@app.function(gpu="L4", volumes={"/cache": hf_cache}, timeout=60 * 60, memory=24 * 1024)
def run_remote(cmd: str, git: str) -> bytes:
    import os
    env = dict(os.environ, MOSHI_RT_MACHINE="modal-L4", MOSHI_RT_GIT=git)
    for d in ("results", "profiles", "audio/generated"):
        os.makedirs(f"{REMOTE}/{d}", exist_ok=True)
    rc = subprocess.run(["bash", "-c", cmd], cwd=REMOTE, env=env).returncode
    hf_cache.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for d in ("results", "profiles", "audio", "audio/generated"):
            for p in Path(REMOTE, d).glob("*"):
                if p.is_file() and p.stat().st_size < 2_000_000 and p.suffix in {".json", ".csv", ".txt", ".md", ".wav", ".png"}:
                    t.add(p, arcname=str(p.relative_to(REMOTE)))
        t.addfile(tarfile.TarInfo(f"rc_{rc}"))
    return buf.getvalue()


@app.local_entrypoint()
def main(cmd: str):
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "none"
    if subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "bench", "quant", "kernels", "scripts"],
                      capture_output=True, text=True).stdout.strip():
        git += "-dirty"
    data = run_remote.remote(cmd, git)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        names = t.getnames()
        t.extractall(ROOT, members=[m for m in t.getmembers() if not m.name.startswith("rc_")], filter="data")
    rc = int(next(n for n in names if n.startswith("rc_"))[3:])
    print("pulled:", [n for n in names if not n.startswith("rc_")], "remote exit", rc)
    assert rc == 0, f"remote command failed with exit {rc}"
