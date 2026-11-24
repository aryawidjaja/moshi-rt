"""Real-time Moshi server for the moshi-rt web app.

Runs the same per-frame path as `moshi.server` (mimi.encode -> LMGen.step -> mimi.decode) with the
moshi-rt changes applied (W8A16 temporal weights, fused INT8-KV attention, graph-captured Mimi
encode), or stock bf16 with --stock. One conversation at a time.

Websocket protocol, /api/chat?token=...  (binary messages start with a kind byte)
  client -> server   0x01 + int16 PCM, 24 kHz mono (any length)
                     0x05 + 8 bytes   ping; echoed back unchanged (network round trip)
  server -> client   0x01 + int16 PCM, one 80 ms frame of Moshi's voice
                     0x02 + utf-8     a piece of Moshi's text
                     0x05 + 8 bytes   ping echo
                     text JSON        {"type": "ready", ...} once, then {"type": "frame", ...} per frame

Usage:  python -m app.server [--stock] [--port 8998] [--token TOKEN]
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WEB = Path(__file__).resolve().parent / "web"


class Engine:
    def __init__(self, stock: bool):
        from bench.common import make_lm_gen, seed_all
        from quant.int8 import load_quantized
        seed_all()
        self.stock = stock
        self.info, self.mimi, self.tok, lm = load_quantized("none" if stock else "w8triton")
        if not stock:
            from kernels.kv_attn import patch_temporal_attention
            patch_temporal_attention(lm, "int8")
        self.lm_gen = make_lm_gen(self.info, lm)
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        if not stock:
            from bench.mimi_graph import patch_mimi_encode
            patch_mimi_encode(self.mimi)
        self.frame = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.gpu = torch.cuda.get_device_name(0)
        self.config = "stock bf16" if stock else "W8A16 + INT8 KV + graph-captured Mimi"
        from importlib.metadata import version
        self.versions = {"torch": torch.__version__, "cuda": torch.version.cuda, "moshi": version("moshi"),
                         "triton": version("triton")}
        self.lock = asyncio.Lock()
        self._warmup()

    @torch.no_grad()
    def _warmup(self, n=12):
        """Capture CUDA graphs and autotune kernels before the first caller."""
        for _ in range(n):
            self.step(torch.zeros(1, 1, self.frame, device="cuda"), first=False)
        self.reset()
        torch.cuda.synchronize()

    def reset(self):
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    @torch.no_grad()
    def step(self, chunk, first):
        """One 80 ms frame. Returns (pcm int16 numpy or None, text piece or None, timings)."""
        t0 = time.perf_counter()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        ev[0].record()
        codes = self.mimi.encode(chunk)
        ev[1].record()
        if first:
            self.mimi.reset_streaming()  # as moshi.server: the first frame's encoder state is dropped
        tokens = self.lm_gen.step(codes[:, :, :1])
        ev[2].record()
        pcm, text = None, None
        if tokens is not None:
            out = self.mimi.decode(tokens[:, 1:])
            ev[3].record()
            pcm = (out[0, 0].float().clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
            tt = tokens[0, 0, 0].item()
            if tt not in (0, 3):
                text = self.tok.id_to_piece(tt).replace("▁", " ")
        else:
            ev[3].record()
            torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0)
        timing = {"ms": round(ms, 2), "encode": round(ev[0].elapsed_time(ev[1]), 2),
                  "lm": round(ev[1].elapsed_time(ev[2]), 2), "decode": round(ev[2].elapsed_time(ev[3]), 2)}
        return pcm, text, timing


async def chat(request):
    engine: Engine = request.app["engine"]
    token = request.app["token"]
    if token and request.query.get("token") != token:
        return web.Response(status=403, text="bad token")
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1 << 22)
    await ws.prepare(request)
    if engine.lock.locked():
        await ws.send_str(json.dumps({"type": "busy"}))
        await ws.close()
        return ws
    async with engine.lock:
        engine.reset()
        await ws.send_str(json.dumps({"type": "ready", "gpu": engine.gpu, "config": engine.config,
                                      "frame_ms": 80, "sample_rate": 24000, "versions": engine.versions}))
        buf = np.zeros(0, dtype=np.float32)
        first, n = True, 0
        async for msg in ws:
            if msg.type != web.WSMsgType.BINARY or not msg.data:
                continue
            kind, payload = msg.data[0], msg.data[1:]
            if kind == 0x05:
                await ws.send_bytes(b"\x05" + payload)
                continue
            if kind != 0x01:
                continue
            buf = np.concatenate([buf, np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768])
            while buf.shape[0] >= engine.frame:
                chunk = torch.from_numpy(buf[:engine.frame].copy()).cuda()[None, None]
                buf = buf[engine.frame:]
                pcm, text, timing = engine.step(chunk, first)
                first = False
                n += 1
                if pcm is not None:
                    await ws.send_bytes(b"\x01" + pcm.tobytes())
                if text:
                    await ws.send_bytes(b"\x02" + text.encode("utf-8"))
                await ws.send_str(json.dumps({"type": "frame", "n": n, **timing}))
    return ws


async def index(request):
    return web.FileResponse(WEB / "index.html")


def make_app(engine, token):
    app = web.Application()
    app["engine"], app["token"] = engine, token
    app.router.add_get("/api/chat", chat)
    app.router.add_get("/api/health", lambda r: web.json_response({"ok": True, "gpu": engine.gpu,
                                                                   "config": engine.config}))
    app.router.add_get("/", index)
    app.router.add_static("/", WEB)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8998)
    ap.add_argument("--stock", action="store_true", help="stock bf16 Moshi instead of moshi-rt")
    ap.add_argument("--token", default=os.environ.get("MOSHI_RT_TOKEN", ""))
    args = ap.parse_args()
    assert torch.cuda.is_available(), "a CUDA GPU is required (or use ./start.sh to run on Modal)"
    print("loading model ...", flush=True)
    engine = Engine(args.stock)
    print(f"ready: {engine.config} on {engine.gpu}", flush=True)
    web.run_app(make_app(engine, args.token), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
