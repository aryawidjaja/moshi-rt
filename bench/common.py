"""Shared helpers: run metadata, model loading, result writing. Runs on the remote GPU only."""
import datetime
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 42424242  # same seed moshi.server uses


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def meta(seed=SEED, extra=None):
    """Metadata block every result JSON carries: no result without its machine, versions and command."""
    import torch
    m = {
        "machine": os.environ.get("MOSHI_RT_MACHINE", "unknown"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": None,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "moshi": _pkg_version("moshi"),
        "torchao": _pkg_version("torchao"),
        "bitsandbytes": _pkg_version("bitsandbytes"),
        "triton": _pkg_version("triton"),
        "git": os.environ.get("MOSHI_RT_GIT", "unknown"),
        "seed": seed,
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "env": {k: os.environ[k] for k in ("NO_CUDA_GRAPH", "NO_TORCH_COMPILE") if k in os.environ},
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    try:
        m["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    except Exception:
        pass
    if extra:
        m.update(extra)
    return m


def write_result(tag, payload, seed=SEED):
    RESULTS.mkdir(exist_ok=True)
    out = {"meta": meta(seed), **payload}
    path = RESULTS / f"{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return path


def seed_all(seed=SEED):
    import random
    import numpy as np
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load(hf_repo=None, device="cuda", dtype=None):
    """Load Mimi, tokenizer and the Moshi LM exactly as moshi.server.main does."""
    import torch
    from moshi.models import loaders
    dtype = dtype or torch.bfloat16
    info = loaders.CheckpointInfo.from_hf_repo(hf_repo or loaders.DEFAULT_REPO)
    mimi = info.get_mimi(device=device)
    tok = info.get_text_tokenizer()
    lm = info.get_moshi(device=device, dtype=dtype)
    return info, mimi, tok, lm


def make_lm_gen(info, lm, batch_size=1, cfg_coef=1.0, **overrides):
    """Build LMGen the way moshi.server.ServerState does."""
    from moshi.models import LMGen
    from moshi.run_inference import get_condition_tensors
    cond = get_condition_tensors(info.model_type, lm, batch_size=batch_size, cfg_coef=cfg_coef)
    kw = dict(info.lm_gen_config)
    kw.update(overrides)
    return LMGen(lm, cfg_coef=cfg_coef, condition_tensors=cond, **kw)


if __name__ == "__main__":
    import torch
    m = meta()
    assert m["torch"] == torch.__version__ and m["seed"] == SEED
    assert m["date"].endswith("UTC")
    print("common.py self-check ok")
