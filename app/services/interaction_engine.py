"""
VivaSessionEngine - server-side, per-session interaction analysis.

Everything that used to be trusted from the browser (who is speaking, response
latency, interruptions) is now derived here from the two authenticated audio
streams, so it cannot be spoofed and is consistent across clients.

Roles: "student" and "examiner" (the lecturer).
No web-framework or ML imports: pure Python + NumPy (see tests/test_engine.py).
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

from .audio import AudioChunk, SpeakerStream, SpeechEvent, TARGET_SR, VadConfig
from .scoring import compute_metrics, sanitize

ROLES = ("student", "examiner")

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_FILLER_RE = re.compile(r"\b(?:um+|uh+|er+m?|erm|hmm+|you know|i mean|kind of|sort of)\b", re.IGNORECASE)


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def count_fillers(text: str) -> int:
    return len(_FILLER_RE.findall(text or ""))


def other(role: str) -> str:
    return "examiner" if role == "student" else "student"


@dataclass
class EngineConfig:
    min_segment_s: float = 0.30            # shorter bursts (clicks, coughs) are ignored
    turn_gap_s: float = 3.0                # silence this long ends a turn
    hesitation_min_s: float = 0.8          # in-turn pause counted as hesitation
    interruption_min_overlap_s: float = 0.4
    simultaneous_start_s: float = 0.3      # near-simultaneous starts are not interruptions
    question_min_s: float = 0.8            # examiner utterance long enough to be a "question"
    backchannel_max_s: float = 1.0         # short "yeah / mm-hm" inside the other's speech
    latency_cap_s: float = 15.0
    smooth_gap_range_s: tuple = (-0.15, 3.0)
    recent_latency_n: int = 8
    wpm_window_s: float = 90.0
    overlap_lookback_s: float = 60.0


@dataclass
class Segment:
    role: str
    s: float
    e: float
    consumed: bool = False
    backchannel: bool = False

    @property
    def dur(self) -> float:
        return self.e - self.s


class VivaSessionEngine:
    def __init__(self, config: Optional[EngineConfig] = None, vad: Optional[VadConfig] = None):
        self.cfg = config or EngineConfig()
        self._vad_cfg = vad
        self.t0: Optional[float] = None
        self._streams: Dict[str, SpeakerStream] = {}
        self._rates: Dict[str, int] = {r: TARGET_SR for r in ROLES}

        self._active: Dict[str, Optional[float]] = {r: None for r in ROLES}
        self._segs: Dict[str, List[Segment]] = {r: [] for r in ROLES}
        self._last_seg: Dict[str, Optional[Segment]] = {r: None for r in ROLES}
        self._last_end: Dict[str, float] = {r: float("-inf") for r in ROLES}
        self._global_last_end: Optional[float] = None
        self.speech_time: Dict[str, float] = {r: 0.0 for r in ROLES}

        self._cur_turn: Dict[str, Optional[List[float]]] = {r: None for r in ROLES}
        self._turn_spans: Dict[str, List[float]] = {r: [] for r in ROLES}

        self.interruptions: Dict[str, int] = {r: 0 for r in ROLES}
        self.backchannels: Dict[str, int] = {r: 0 for r in ROLES}
        self.overlap_s = 0.0
        self.latencies: List[float] = []
        self.transitions: List[float] = []
        self.hesitations = 0
        self.hesitation_time = 0.0
        self.dead_air_s = 0.0

        # ASR-derived (student)
        self.transcript: List[dict] = []
        self.words_total = 0
        self.fillers = 0
        self._asr_words = 0
        self._asr_dur = 0.0
        self._wpm_window: Deque[tuple] = deque()

    # ------------------------------------------------------------ audio input
    def configure_stream(self, role: str, sample_rate: int) -> bool:
        """Set the capture sample rate. Only allowed before audio has arrived."""
        if role not in ROLES:
            return False
        if role in self._streams and self._streams[role].samples_received > 0:
            return False
        self._rates[role] = int(sample_rate)
        self._streams.pop(role, None)
        return True

    def _stream(self, role: str) -> SpeakerStream:
        if role not in self._streams:
            self._streams[role] = SpeakerStream(self._rates[role], self._vad_cfg)
        return self._streams[role]

    def feed_audio(self, role: str, pcm: bytes, now: float) -> List[AudioChunk]:
        """Feed PCM16 mono audio. Returns finished chunks that should be transcribed."""
        if role not in ROLES:
            return []
        if self.t0 is None:
            self.t0 = now
        events, chunks = self._stream(role).feed(pcm, now)
        for ev in events:
            self._on_event(role, ev)
        return chunks

    def reset_stream(self, role: str) -> List[AudioChunk]:
        """Close and drop a participant's stream (call on reconnect so clocks re-sync)."""
        stream = self._streams.pop(role, None)
        if stream is None:
            return []
        events, chunks = stream.flush()
        for ev in events:
            self._on_event(role, ev)
        self._active[role] = None
        return chunks

    def flush(self) -> Dict[str, List[AudioChunk]]:
        out: Dict[str, List[AudioChunk]] = {}
        for role, stream in self._streams.items():
            events, chunks = stream.flush()
            for ev in events:
                self._on_event(role, ev)
            out[role] = chunks
        return out

    # ------------------------------------------------------- transcript input
    def add_transcript(self, role: str, t_start: float, t_end: float, text: str) -> None:
        words = count_words(text)
        item = {
            "role": role,
            "start": round(max(t_start - (self.t0 or t_start), 0.0), 2),
            "end": round(max(t_end - (self.t0 or t_end), 0.0), 2),
            "text": text,
            "words": words,
        }
        self.transcript.append(item)
        if role != "student":
            return
        self.words_total += words
        self.fillers += count_fillers(text)
        # chunk time includes ~0.25 s of pre-roll / tail padding
        dur = max((t_end - t_start) - 0.25, 0.5)
        if words >= 3 and dur >= 1.0:
            self._asr_words += words
            self._asr_dur += dur
            self._wpm_window.append((t_end, words, dur))

    # ------------------------------------------------------- event processing
    def _on_event(self, role: str, ev: SpeechEvent) -> None:
        if ev.kind == "start":
            self._active[role] = ev.t_start
            return
        self._active[role] = None
        if ev.discard or ev.t_end is None:
            return
        seg = Segment(role, ev.t_start, ev.t_end)
        if seg.dur < self.cfg.min_segment_s:
            return
        self._finish_segment(seg)

    def _finish_segment(self, seg: Segment) -> None:
        cfg = self.cfg
        r, o = seg.role, other(seg.role)

        # 0) Provisional backchannel: a short utterance made while the other person is
        #    still speaking ("mm-hm", "yes"). Cooperative, so it is not an interruption,
        #    does not open a turn, and does not count as overlap.
        o_since = self._active[o]
        if (o_since is not None and seg.dur < cfg.backchannel_max_s
                and seg.s >= o_since - cfg.simultaneous_start_s):
            seg.backchannel = True
            self.backchannels[r] += 1
            self.speech_time[r] += seg.dur
            self._segs[r].append(seg)
            return

        # 1) dead air: long gap since ANY speech ended
        if self._global_last_end is not None:
            gap = seg.s - self._global_last_end
            if gap >= cfg.turn_gap_s:
                self.dead_air_s += gap

        # 2) overlaps / interruptions (counted once per pair: by whoever finishes second)
        for y in reversed(self._segs[o]):
            if y.e < seg.s - cfg.overlap_lookback_s:
                break
            if y.backchannel:
                continue
            ov = min(seg.e, y.e) - max(seg.s, y.s)
            if ov <= 0:
                continue
            seg_in_y = seg.dur < cfg.backchannel_max_s and seg.s >= y.s - 0.3 and seg.e <= y.e + 0.3
            y_in_seg = y.dur < cfg.backchannel_max_s and y.s >= seg.s - 0.3 and y.e <= seg.e + 0.3
            if seg_in_y:
                self.backchannels[r] += 1
                seg.backchannel = True
                continue
            if y_in_seg:
                self.backchannels[o] += 1
                y.backchannel = True
                continue
            self.overlap_s += ov
            if ov >= cfg.interruption_min_overlap_s and abs(seg.s - y.s) >= cfg.simultaneous_start_s:
                interrupter = r if seg.s > y.s else o
                self.interruptions[interrupter] += 1

        # 3) speaker transitions + response latency
        q = self._last_seg[o]
        if q is not None and not q.consumed and not q.backchannel:
            gap = seg.s - q.e
            if gap >= 0:
                self.transitions.append(min(gap, 30.0))
                if r == "student" and q.dur >= cfg.question_min_s:
                    self.latencies.append(min(gap, cfg.latency_cap_s))
                q.consumed = True
            elif not seg.backchannel:
                if q.backchannel:
                    q.consumed = True
                else:
                    self.transitions.append(gap)     # negative gap = overlapping hand-over
                    q.consumed = True

        # 4) turns and in-turn hesitations (a backchannel never opens or extends a turn)
        if not seg.backchannel:
            self._register_turn(seg)

        # 5) bookkeeping
        self.speech_time[r] += seg.dur
        self._segs[r].append(seg)
        if len(self._segs[r]) > 400:
            del self._segs[r][:100]
        if not seg.backchannel:
            self._last_seg[r] = seg
            self._last_end[r] = max(self._last_end[r], seg.e)
            self._global_last_end = seg.e if self._global_last_end is None else max(self._global_last_end, seg.e)

    def _register_turn(self, seg: Segment) -> None:
        r, o = seg.role, other(seg.role)
        cur = self._cur_turn[r]
        if cur is not None:
            broken = self._last_end[o] > cur[1]
            gap = seg.s - cur[1]
            if not broken and gap < self.cfg.turn_gap_s:
                if r == "student" and gap >= self.cfg.hesitation_min_s:
                    self.hesitations += 1
                    self.hesitation_time += gap
                cur[1] = max(cur[1], seg.e)
                return
            self._turn_spans[r].append(cur[1] - cur[0])
        self._cur_turn[r] = [seg.s, seg.e]

    # --------------------------------------------------------------- features
    def _live(self, role: str, now: float) -> float:
        start = self._active[role]
        return max(now - start, 0.0) if start is not None else 0.0

    def _turns(self, role: str) -> tuple:
        spans = list(self._turn_spans[role])
        if self._cur_turn[role] is not None:
            c = self._cur_turn[role]
            spans.append(c[1] - c[0])
        return len(spans), (float(np.mean(spans)) if spans else None)

    def features(self, now: float) -> Dict[str, Optional[float]]:
        cfg = self.cfg
        elapsed = max(now - self.t0, 0.0) if self.t0 is not None else 0.0
        S = self.speech_time["student"] + self._live("student", now)
        E = self.speech_time["examiner"] + self._live("examiner", now)
        total_speech = S + E
        s_turns, s_avg = self._turns("student")
        e_turns, e_avg = self._turns("examiner")

        lat = self.latencies
        recent_lat = lat[-cfg.recent_latency_n:]
        lo, hi = cfg.smooth_gap_range_s
        smooth = [g for g in self.transitions if lo <= g <= hi]

        while self._wpm_window and self._wpm_window[0][0] < now - cfg.wpm_window_s:
            self._wpm_window.popleft()
        win_dur = sum(d for _, _, d in self._wpm_window)
        win_words = sum(w for _, w, _ in self._wpm_window)
        wpm_recent = 60.0 * win_words / win_dur if win_dur >= 5.0 else None
        wpm_overall = 60.0 * self._asr_words / self._asr_dur if self._asr_dur >= 5.0 else None

        stu = self._streams.get("student")
        noise = snr = None
        if stu is not None:
            noise = stu.noise_db
            if stu.speech_db.n >= 25 and noise is not None:
                snr = stu.speech_db.mean - noise

        minutes = elapsed / 60.0
        f = {
            "elapsed_s": elapsed,
            "student_speech_s": S,
            "examiner_speech_s": E,
            "talk_ratio": S / total_speech if total_speech > 1.0 else None,
            "student_turns": s_turns, "examiner_turns": e_turns,
            "avg_student_turn_s": s_avg, "avg_examiner_turn_s": e_avg,
            "interruptions_student": self.interruptions["student"],
            "interruptions_examiner": self.interruptions["examiner"],
            "interruptions_per_min": sum(self.interruptions.values()) / max(minutes, 0.5),
            "overlap_s": self.overlap_s,
            "overlap_ratio": self.overlap_s / total_speech if total_speech > 1.0 else 0.0,
            "backchannels_student": self.backchannels["student"],
            "backchannels_examiner": self.backchannels["examiner"],
            "n_latencies": len(lat),
            "latency_mean_s": float(np.mean(lat)) if lat else None,
            "latency_median_s": float(np.median(lat)) if lat else None,
            "latency_recent_s": float(np.median(recent_lat)) if recent_lat else None,
            "latency_p90_s": float(np.percentile(lat, 90)) if len(lat) >= 3 else None,
            "n_transitions": len(self.transitions),
            "smooth_transition_ratio": len(smooth) / len(self.transitions) if self.transitions else None,
            "hesitations": self.hesitations,
            "hesitations_per_min": self.hesitations / (S / 60.0) if S >= 20.0 else None,
            "dead_air_s": self.dead_air_s,
            "dead_air_ratio": self.dead_air_s / elapsed if elapsed > 10.0 else 0.0,
            "words": self.words_total,
            "wpm_recent": wpm_recent,
            "wpm_overall": wpm_overall,
            "fillers": self.fillers,
            "filler_per_100w": 100.0 * self.fillers / self.words_total if self.words_total >= 30 else None,
            "noise_db": noise, "snr_db": snr,
        }
        return sanitize(f)

    def snapshot(self, now: float) -> Dict[str, dict]:
        feats = self.features(now)
        return {"features": feats, "metrics": sanitize(compute_metrics(feats))}
