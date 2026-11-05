"""
Streaming audio utilities.

Each participant streams their OWN microphone over a dedicated WebSocket, so the
speaker identity is known from the authenticated connection. This replaces
statistical diarization with a deterministic method for a two-party viva call,
and removes the need for a pyannote / HuggingFace token.

This module is pure NumPy (no torch / web-framework imports) so it is fast and
fully unit-testable.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

TARGET_SR = 16_000
FRAME_S = 0.02
FRAME_LEN = int(TARGET_SR * FRAME_S)  # 320 samples


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Little-endian signed 16-bit PCM -> float32 in [-1, 1]."""
    if len(data) < 2:
        return np.zeros(0, dtype=np.float32)
    if len(data) % 2:
        data = data[:-1]
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


class StreamingResampler:
    """
    Stateful linear resampler, used only when a client cannot capture at 16 kHz.
    The interpolation position is carried across chunks, so there are no chunk
    boundary discontinuities. (Capturing at 16 kHz on the client is preferred.)
    """

    def __init__(self, in_rate: int, out_rate: int = TARGET_SR):
        if in_rate < 8_000 or in_rate > 96_000:
            raise ValueError("Unsupported sample rate")
        self.step = in_rate / out_rate
        self._pos = 0.0
        self._tail = np.zeros(0, dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self._tail, x]) if self._tail.size else x
        n = buf.size
        if n < 2:
            self._tail = buf
            return np.zeros(0, dtype=np.float32)
        positions = np.arange(self._pos, n - 1, self.step)
        out = np.interp(positions, np.arange(n), buf).astype(np.float32)
        next_pos = (positions[-1] + self.step) if positions.size else self._pos
        keep = min(int(math.floor(next_pos)), n - 1)
        self._tail = buf[keep:]
        self._pos = next_pos - keep
        return out


class RunningStats:
    """Welford online mean / standard deviation."""

    __slots__ = ("n", "mean", "_m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self._m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self._m2 / self.n) if self.n > 1 else 0.0


@dataclass
class VadConfig:
    onset_frames: int = 4          # 80 ms above threshold opens a segment
    hangover_frames: int = 15      # 300 ms below threshold closes it
    warmup_frames: int = 10        # ignore start-up transients (200 ms)
    on_margin_db: float = 10.0     # speech must exceed the noise floor by this much
    off_margin_db: float = 6.0     # hysteresis: stay "speaking" above this margin
    min_abs_db: float = -50.0      # never call anything quieter than this speech
    preroll_frames: int = 10       # 200 ms of audio kept before onset for ASR
    max_chunk_s: float = 25.0      # ASR chunk cap (Whisper window is 30 s)
    min_chunk_s: float = 0.4       # do not send shorter chunks to ASR
    stuck_std_db: float = 1.5      # "speech" this flat for this long is really noise


@dataclass
class SpeechEvent:
    kind: str                      # "start" | "end"
    t_start: float
    t_end: Optional[float] = None
    discard: bool = False          # True when a long flat segment was judged to be noise


@dataclass
class AudioChunk:
    t_start: float
    t_end: float
    audio: np.ndarray              # float32, 16 kHz, mono


class SpeakerStream:
    """
    One participant's microphone stream:
    PCM bytes -> 20 ms frames -> adaptive-noise-floor VAD -> speech events + ASR chunks.

    Timestamps are on the server's monotonic clock. Position inside a stream is
    derived from the sample count (immune to network jitter); the offset between
    stream position and server clock is the windowed minimum of (arrival - position),
    the standard one-way-delay estimator, so both participants share a time base.
    """

    def __init__(self, sample_rate: int = TARGET_SR, cfg: Optional[VadConfig] = None):
        self.cfg = cfg or VadConfig()
        self.sample_rate = sample_rate
        self._resampler = None if sample_rate == TARGET_SR else StreamingResampler(sample_rate)
        self._carry = np.zeros(0, dtype=np.float32)
        self._frame_idx = 0
        self._offset = math.inf
        self._off_hist: Deque[Tuple[float, float]] = deque()   # (arrival, arrival - position)

        self._noise_db: Optional[float] = None
        self._speech = False
        self._above = 0
        self._below = 0
        self._seg_start_idx = 0
        self._chunk_start_idx = 0
        self._preroll: Deque[Tuple[np.ndarray, float]] = deque(maxlen=self.cfg.preroll_frames)
        self._buf: List[Tuple[np.ndarray, float]] = []

        self.speech_db = RunningStats()      # level of voiced frames (dBFS)
        self.samples_received = 0

    # ------------------------------------------------------------------ public
    @property
    def noise_db(self) -> Optional[float]:
        return self._noise_db

    @property
    def is_speaking(self) -> bool:
        return self._speech

    def feed(self, pcm: bytes, now: float) -> Tuple[List[SpeechEvent], List[AudioChunk]]:
        x = pcm16_to_float32(pcm)
        self.samples_received += x.size
        if self._resampler is not None:
            x = self._resampler.process(x)
        if x.size == 0:
            return [], []
        data = np.concatenate([self._carry, x]) if self._carry.size else x
        n_frames = data.size // FRAME_LEN
        self._carry = data[n_frames * FRAME_LEN:]
        if n_frames == 0:
            return [], []

        end_pos = (self._frame_idx + n_frames) * FRAME_S
        # One-way-delay estimate: minimum of (arrival - stream position) over a sliding
        # 30 s window. Tracks clock drift between the two mics without ever drifting far.
        self._off_hist.append((now, now - end_pos))
        while self._off_hist and now - self._off_hist[0][0] > 30.0:
            self._off_hist.popleft()
        self._offset = min(v for _, v in self._off_hist)

        frames = data[: n_frames * FRAME_LEN].reshape(n_frames, FRAME_LEN)
        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        db = 20.0 * np.log10(rms)

        events: List[SpeechEvent] = []
        chunks: List[AudioChunk] = []
        for i in range(n_frames):
            self._step(frames[i], float(db[i]), events, chunks)
            self._frame_idx += 1
        return events, chunks

    def flush(self) -> Tuple[List[SpeechEvent], List[AudioChunk]]:
        """Close any open segment (call when the session ends)."""
        events: List[SpeechEvent] = []
        chunks: List[AudioChunk] = []
        if self._speech and self._frame_idx > 0:
            self._finish(self._frame_idx - 1, events, chunks)
        return events, chunks

    # ----------------------------------------------------------------- private
    def _t(self, idx: int) -> float:
        return self._offset + idx * FRAME_S

    def _update_noise(self, db: float) -> None:
        if self._noise_db is None:
            self._noise_db = max(db, -80.0)
            return
        rate = 0.2 if db < self._noise_db else 0.01   # fall fast, rise slowly
        self._noise_db = max(self._noise_db + rate * (db - self._noise_db), -80.0)

    def _step(self, frame, db, events, chunks) -> None:
        cfg = self.cfg
        idx = self._frame_idx
        if self._noise_db is None:
            self._noise_db = max(db, -80.0)
        on_thr = max(self._noise_db + cfg.on_margin_db, cfg.min_abs_db)
        off_thr = max(self._noise_db + cfg.off_margin_db, cfg.min_abs_db - 3.0)

        if not self._speech:
            self._preroll.append((frame, db))
            if db < on_thr:
                self._update_noise(db)
                self._above = 0
                return
            if idx < cfg.warmup_frames:
                return
            self._above += 1
            if self._above >= cfg.onset_frames:
                self._speech = True
                self._below = 0
                self._seg_start_idx = idx - self._above + 1
                self._buf = list(self._preroll)
                self._chunk_start_idx = idx - len(self._buf) + 1
                events.append(SpeechEvent("start", self._t(self._seg_start_idx)))
            return

        # currently speaking
        self._buf.append((frame, db))
        if db >= off_thr:
            self._below = 0
            self.speech_db.update(db)
        else:
            self._below += 1
            if db < self._noise_db + 3.0:
                self._update_noise(db)

        if self._below >= cfg.hangover_frames:
            self._finish(idx, events, chunks)
        elif len(self._buf) * FRAME_S >= cfg.max_chunk_s:
            self._cut(idx, events, chunks)

    def _emit_chunk(self, frames, start_idx, chunks) -> None:
        if len(frames) * FRAME_S < self.cfg.min_chunk_s:
            return
        audio = np.concatenate([f for f, _ in frames]).astype(np.float32)
        chunks.append(AudioChunk(self._t(start_idx), self._t(start_idx + len(frames)), audio))

    def _finish(self, idx, events, chunks) -> None:
        end_idx = idx - self._below                      # last voiced frame
        events.append(SpeechEvent("end", self._t(self._seg_start_idx), self._t(end_idx + 1)))
        trim = max(0, self._below - 4)                   # keep ~80 ms of tail
        frames = self._buf[: len(self._buf) - trim] if trim else self._buf
        self._emit_chunk(frames, self._chunk_start_idx, chunks)
        self._speech = False
        self._above = 0
        self._below = 0
        self._buf = []
        self._preroll.clear()

    def _cut(self, idx, events, chunks) -> None:
        dbs = np.array([d for _, d in self._buf], dtype=np.float32)
        if float(dbs.std()) < self.cfg.stuck_std_db:
            # Very flat for a very long time: a fan / hum, not a person.
            self._noise_db = max(float(dbs.mean()), -80.0)
            events.append(SpeechEvent("end", self._t(self._seg_start_idx), self._t(idx + 1), discard=True))
            self._speech = False
            self._above = 0
            self._below = 0
            self._buf = []
            self._preroll.clear()
            return
        self._emit_chunk(self._buf, self._chunk_start_idx, chunks)
        self._buf = []
        self._chunk_start_idx = idx + 1
