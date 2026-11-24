"""Build the five fixed 30 s, 24 kHz test clips from LibriSpeech test-clean (CC BY 4.0).
Downloads test-clean (~350 MB) to /content/librispeech. Deterministic: the same
speakers and utterances every time. Writes audio/clip{1..5}.wav and audio/SOURCES.md."""
import tarfile
import urllib.request
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
WORK = Path("/content/librispeech")
SECONDS, SR_IN, SR_OUT, GAP = 30.0, 16000, 24000, 0.35
N_F, N_M = 3, 2  # three female, two male speakers


def fetch():
    tgz = WORK / "test-clean.tar.gz"
    if not (WORK / "LibriSpeech").exists():
        WORK.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(WORK)
        tgz.unlink()
    return WORK / "LibriSpeech"


def speakers(ls):
    sex = {}
    for line in (ls / "SPEAKERS.TXT").read_text().splitlines():
        if line.startswith(";"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[2] == "test-clean":
            sex[parts[0]] = parts[1]
    present = sorted(p.name for p in (ls / "test-clean").iterdir())
    f = [s for s in present if sex.get(s) == "F"][:N_F]
    m = [s for s in present if sex.get(s) == "M"][:N_M]
    return [(s, "F") for s in f] + [(s, "M") for s in m]


def build(ls, spk):
    gap = np.zeros(int(GAP * SR_IN), np.float32)
    parts, used, n = [], [], 0
    for flac in sorted((ls / "test-clean" / spk).rglob("*.flac")):
        x, sr = sf.read(flac, dtype="float32")
        assert sr == SR_IN
        parts += [x, gap]; used.append(flac.stem); n += len(x) + len(gap)
        if n >= SECONDS * SR_IN:
            break
    x = np.concatenate(parts)[: int(SECONDS * SR_IN)]
    y = resample_poly(x, 3, 2).astype(np.float32)  # 16 kHz -> 24 kHz
    y = y / max(1e-6, np.abs(y).max()) * 0.7  # same peak level for every clip
    return y, used


def save(path, y):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR_OUT)
        w.writeframes((y * 32767).astype(np.int16).tobytes())


def main():
    ls = fetch()
    out = ROOT / "audio"; out.mkdir(exist_ok=True)
    rows = []
    for i, (spk, sex) in enumerate(speakers(ls), 1):
        y, used = build(ls, spk)
        save(out / f"clip{i}.wav", y)
        rows.append(f"| clip{i}.wav | {spk} | {sex} | {used[0]} .. {used[-1]} ({len(used)} utterances) |")
        assert len(y) == int(SECONDS * SR_OUT), len(y)
    (out / "SOURCES.md").write_text(
        "# Test clips\n\nFive fixed 30 s mono clips at 24 kHz, built by `scripts/make_clips.py` from "
        "[LibriSpeech](https://www.openslr.org/12) test-clean (Panayotov et al., 2015), "
        "licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The underlying recordings "
        "are LibriVox public-domain audiobooks. Changes: utterances of one speaker joined with "
        f"{GAP} s gaps, cut to 30 s, resampled 16 kHz to 24 kHz, peak-normalised to 0.7.\n\n"
        "| File | Speaker | Sex | Utterances |\n| --- | --- | --- | --- |\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
    clips = sorted((ROOT / "audio").glob("clip*.wav"))
    assert len(clips) == 5
    for c in clips:
        with wave.open(str(c)) as w:
            assert w.getframerate() == SR_OUT and w.getnframes() == int(SECONDS * SR_OUT)
    print("make_clips self-check ok")
