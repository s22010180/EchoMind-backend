"""
Measure Whisper's REAL accuracy on your own viva audio (Word Error Rate, WER).

1. Record a short answer (<= 25 s, any common .wav) or export one from a mock viva.
2. Write down exactly what was said in a .txt file (the "reference").
3. Run:   python -m scripts.evaluate_asr --audio answer1.wav --reference answer1.txt
   or compare several:   python -m scripts.evaluate_asr --audio a.wav b.wav --reference a.txt b.txt

WER = (substitutions + deletions + insertions) / words in the reference. Lower is better; 0% is perfect and
real-world speech with accents and technical terms rarely reaches it. Report the figure you MEASURE.
"""
import argparse
import asyncio
import re
import wave
from typing import Dict, List

import numpy as np


def normalise(text: str) -> List[str]:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return text.split()


def wer(reference: str, hypothesis: str) -> Dict[str, float]:
    """Word error rate with a breakdown, via word-level edit distance."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    # backtrace for the S / D / I breakdown
    i, j, sub, dele, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            sub += 0 if ref[i - 1] == hyp[j - 1] else 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    errors = sub + dele + ins
    return {"wer": errors / n if n else (0.0 if m == 0 else 1.0), "substitutions": sub, "deletions": dele,
            "insertions": ins, "ref_words": n, "hyp_words": m}


def read_wav_16k(path: str) -> np.ndarray:
    """Read a PCM16 .wav as mono float32 at 16 kHz."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: only 16-bit PCM .wav files are supported")
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        t_out = np.arange(0, len(x) / sr, 1 / 16000)
        x = np.interp(t_out, np.arange(len(x)) / sr, x).astype(np.float32)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--reference", nargs="+", required=True)
    args = ap.parse_args()
    if len(args.audio) != len(args.reference):
        raise SystemExit("Give one reference .txt for every audio file.")

    from app.config import settings
    from app.services.asr_service import ASRService

    asr = ASRService(settings.ASR_MODEL, settings.ASR_ENGINE, settings.ASR_COMPUTE_TYPE, settings.ASR_BEAM_SIZE, settings.ASR_DEVICE)
    print(f"Engine: {asr.engine} | model: {settings.ASR_MODEL}\n")

    tot_err = tot_ref = 0.0
    for a_path, r_path in zip(args.audio, args.reference):
        audio = read_wav_16k(a_path)
        step = 25 * 16000
        text = " ".join(asyncio.run(asr.transcribe(audio[i:i + step])) for i in range(0, len(audio), step))
        ref = open(r_path, encoding="utf-8").read()
        r = wer(ref, text)
        tot_err += r["wer"] * r["ref_words"]
        tot_ref += r["ref_words"]
        print(f"{a_path}\n  WER {r['wer'] * 100:.1f}%  (S={r['substitutions']} D={r['deletions']} I={r['insertions']} of {r['ref_words']} words)")
        print(f"  heard: {text}\n")
    if tot_ref:
        print(f"OVERALL WER: {tot_err / tot_ref * 100:.1f}% over {int(tot_ref)} reference words")


if __name__ == "__main__":
    main()
