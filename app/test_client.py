"""End-to-end check of a running moshi-rt server without a browser: stream a test clip in real time
over the websocket, collect Moshi's audio, text and per-frame timings, and check them.

    python app/test_client.py --server https://<...>.modal.run --token TOKEN [--seconds 20]
"""
import argparse
import asyncio
import json
import struct
import time
import wave
from pathlib import Path

import aiohttp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAME = 1920


async def run(server, token, seconds):
    with wave.open(str(sorted((ROOT / "audio").glob("clip*.wav"))[0])) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    n_frames = min(len(pcm) // FRAME, int(seconds / 0.08))
    url = server.replace("https", "wss").replace("http", "ws").rstrip("/") + f"/api/chat?token={token}"
    got_audio, text, timings, rtts, ready = 0, [], [], [], None
    async with aiohttp.ClientSession() as s, s.ws_connect(url, max_msg_size=1 << 22) as ws:
        async def reader():
            nonlocal got_audio, ready
            async for m in ws:
                if m.type == aiohttp.WSMsgType.TEXT:
                    d = json.loads(m.data)
                    if d["type"] == "ready":
                        ready = d
                    elif d["type"] == "frame":
                        timings.append(d)
                elif m.type == aiohttp.WSMsgType.BINARY:
                    k = m.data[0]
                    if k == 1:
                        got_audio += (len(m.data) - 1) // 2
                    elif k == 2:
                        text.append(m.data[1:].decode())
                    elif k == 5:
                        rtts.append(1000 * (time.perf_counter() - struct.unpack("d", m.data[1:9])[0]))
        task = asyncio.create_task(reader())
        while ready is None:
            await asyncio.sleep(0.05)
        t_next = time.perf_counter()
        for i in range(n_frames):
            await ws.send_bytes(b"\x01" + pcm[i * FRAME:(i + 1) * FRAME].tobytes())
            if i % 12 == 0:
                await ws.send_bytes(b"\x05" + struct.pack("d", time.perf_counter()))
            t_next += 0.08
            await asyncio.sleep(max(0.0, t_next - time.perf_counter()))
        await asyncio.sleep(1.0)
        await ws.close()
        task.cancel()
    ms = np.array([t["ms"] for t in timings])
    res = {"config": ready["config"], "gpu": ready["gpu"], "frames_sent": n_frames, "frames_timed": len(ms),
           "audio_seconds": round(got_audio / 24000, 2), "text": "".join(text).strip(),
           "frame_ms_p50": round(float(np.percentile(ms, 50)), 2), "frame_ms_p99": round(float(np.percentile(ms, 99)), 2),
           "late": int((ms > 80).sum()), "rtt_ms_p50": round(float(np.median(rtts)), 1) if rtts else None}
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--token", default="")
    ap.add_argument("--seconds", type=float, default=20)
    a = ap.parse_args()
    r = asyncio.run(run(a.server, a.token, a.seconds))
    assert r["frames_timed"] >= r["frames_sent"] - 2, "server did not process every frame"
    assert r["audio_seconds"] > 0.8 * a.seconds, "too little audio came back"
    assert r["frame_ms_p50"] < 80, "median frame over budget"
