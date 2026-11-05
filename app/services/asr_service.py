"""
Speech-to-text for the viva transcript.

Accuracy measures built in (no ASR is ever 100% accurate - measure yours with scripts/evaluate_asr.py):
  * each person's own microphone is transcribed separately (no overlapping voices to untangle);
  * the VAD hands over one whole utterance at a time (words are never cut in half);
  * a software-engineering vocabulary hint helps with terms like "polymorphism" and "Kubernetes";
  * faster-whisper with beam search (preferred engine) - noticeably more accurate AND faster on CPU than the
    plain transformers pipeline at the same model size; the old engine is kept as an automatic fallback;
  * Whisper's known hallucinations on near-silence are filtered out.

Pick the model with ASR_MODEL: "small.en" (recommended), "base.en" (faster, less accurate), "medium.en" (best,
needs a strong CPU or GPU).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Optional

import numpy as np

from app.services.glossary import asr_prompt

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
_NOISE_TAGS = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")
_HALLUCINATIONS = {
    "you", "thank you", "thanks", "thanks for watching", "thank you for watching",
    "bye", "bye bye", "the end", "so", "oh",
}


def clean_transcript(text: str, duration_s: float) -> str:
    t = _NOISE_TAGS.sub(" ", text or "")
    t = " ".join(t.split()).strip()
    if not t:
        return ""
    norm = re.sub(r"[^a-z' ]", "", t.lower()).strip()
    if not norm:
        return ""
    if duration_s < 2.5 and norm in _HALLUCINATIONS:
        return ""
    # collapse runaway repetition ("the the the the ...")
    words = t.split()
    out, run = [], 0
    for i, w in enumerate(words):
        run = run + 1 if i and w.lower() == words[i - 1].lower() else 1
        if run <= 3:
            out.append(w)
    return " ".join(out)


def is_prompt_echo(text: str, prompt: Optional[str]) -> bool:
    """True when the output is just the vocabulary hint read back (what Whisper does on noise)."""
    if not prompt:
        return False
    tokens = re.findall(r"[a-z']+", text.lower())
    hint = set(re.findall(r"[a-z']+", prompt.lower()))
    return len(tokens) >= 4 and all(t in hint for t in tokens)


def faster_whisper_name(model_name: str) -> str:
    """'openai/whisper-base.en' (old setting) -> 'base.en'; local folders and Systran repos pass through."""
    if os.path.isdir(model_name) or "faster-whisper" in model_name:
        return model_name
    tail = model_name.split("/")[-1]
    return tail[len("whisper-"):] if tail.startswith("whisper-") else tail


class ASRService:
    def __init__(self, model_name: str, engine: str = "auto", compute_type: str = "int8",
                 beam_size: int = 5, device: str = "cpu", use_vocabulary_hint: bool = True):
        self.model_name = model_name
        self.engine: Optional[str] = None
        self.beam_size = beam_size
        self._prompt = asr_prompt() if use_vocabulary_hint else None
        self._sem = asyncio.Semaphore(1)
        self._fw = None
        self._pipe = None

        if engine in ("auto", "faster-whisper"):
            try:
                self._load_faster(model_name, device, compute_type)
            except Exception as exc:
                if engine == "faster-whisper":
                    raise
                logger.warning("faster-whisper unavailable (%s) - falling back to the transformers engine.", exc)
        if self.engine is None:
            self._load_transformers(model_name)

    # ----------------------------------------------------------------- loading
    def _load_faster(self, model_name: str, device: str, compute_type: str) -> None:
        from faster_whisper import WhisperModel

        name = faster_whisper_name(model_name)
        logger.info("Loading faster-whisper model %s (%s, %s) ...", name, device, compute_type)
        self._fw = WhisperModel(name, device=device, compute_type=compute_type)
        self.engine = "faster-whisper"
        logger.info("ASR ready (faster-whisper, beam=%d, vocabulary hint=%s)", self.beam_size, bool(self._prompt))

    def _load_transformers(self, model_name: str) -> None:
        import torch
        from transformers import pipeline

        name = model_name if "/" in model_name else f"openai/whisper-{model_name}"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info("Loading transformers Whisper %s on %s ...", name, device)
        self._pipe = pipeline("automatic-speech-recognition", model=name, device=device)
        self.engine = "transformers"
        logger.info("ASR ready (transformers)")

    # ----------------------------------------------------------------- running
    def _run(self, audio: np.ndarray) -> str:
        rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        if 20 * np.log10(rms) < -55:           # effectively silence: skip, avoids hallucination
            return ""
        audio = audio.astype(np.float32)
        if self.engine == "faster-whisper":
            segments, _ = self._fw.transcribe(
                audio, language="en", beam_size=self.beam_size, temperature=[0.0, 0.2, 0.4],
                condition_on_previous_text=False, initial_prompt=self._prompt, vad_filter=False,
                no_speech_threshold=0.5, log_prob_threshold=-1.0, compression_ratio_threshold=2.4,
            )
            parts = []
            for seg in segments:
                if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                    continue                    # confident non-speech
                parts.append(seg.text.strip())
            text = " ".join(parts)
        else:
            text = self._pipe({"raw": audio, "sampling_rate": SAMPLE_RATE}).get("text", "")
        text = clean_transcript(text, len(audio) / SAMPLE_RATE)
        return "" if is_prompt_echo(text, self._prompt) else text

    async def transcribe(self, audio: np.ndarray) -> str:
        if audio is None or audio.size < SAMPLE_RATE * 0.4:
            return ""
        async with self._sem:
            try:
                return await asyncio.to_thread(self._run, audio)
            except Exception:
                logger.exception("ASR failure")
                return ""
