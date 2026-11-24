"""Measured DRAM read and copy bandwidth, as a reference next to the datasheet peak.
Two probes on a 2 GiB bf16 buffer (far larger than the 48 MB L2):
  copy  y.copy_(x): reads + writes, counted as 2x bytes
  read  x.sum():    a streaming read, close to what a weight-streaming GEMV does
Writes results/bw_probe_<tag>.json."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import write_result

DATASHEET_GBS = {"NVIDIA L4": 300.0, "Tesla T4": 320.0}


def time_ms(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]  # median


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l4")
    args = ap.parse_args()
    n = 2 * 1024 ** 3 // 2  # 2 GiB of bf16
    x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    nbytes = x.numel() * x.element_size()
    copy_ms = time_ms(lambda: y.copy_(x))
    read_ms = time_ms(lambda: x.sum(dtype=torch.float32))
    gpu = torch.cuda.get_device_name(0)
    res = {
        "what": "achievable DRAM bandwidth, median of 50, 2 GiB bf16 buffer",
        "buffer_gb": round(nbytes / 1e9, 3),
        "copy_gbs": round(2 * nbytes / 1e9 / (copy_ms / 1e3), 1),
        "read_gbs": round(nbytes / 1e9 / (read_ms / 1e3), 1),
        "datasheet_gbs": DATASHEET_GBS.get(gpu),
    }
    write_result(f"bw_probe_{args.tag}", res)
    print(res)
    assert 10 < res["read_gbs"] < 5000 and 10 < res["copy_gbs"] < 5000


if __name__ == "__main__":
    main()
