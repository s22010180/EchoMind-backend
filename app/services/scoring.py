"""
Transparent scoring of VIVA performance signals.

Every score is a *heuristic indicator of observable behaviour* (timing, delivery, answer content). None
measures knowledge, honesty or feelings, and none is a grade - the examiner decides. Breakpoints below are the
single place to calibrate once your own labelled viva data exists (see scripts/validate_metrics.py).

This module computes the LIVE signals (available while the viva runs, from audio timing alone):
  * promptness_score - how soon answers begin after a question
  * fluency_score    - how smoothly answers are delivered (rate, pauses, fillers)
The richer, question-by-question indicators (depth, technical content, consistency) are computed from the
transcript by app/services/performance.py.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

MIN_STUDENT_SPEECH_S = 45.0   # student speech needed for fully-trusted fluency
MIN_ANSWERS = 3               # answers needed for fully-trusted promptness

# Features produced by the live audio engine (interaction_engine.py)
ENGINE_FEATURE_NAMES: List[str] = [
    "elapsed_s", "student_speech_s", "examiner_speech_s", "talk_ratio",
    "student_turns", "examiner_turns", "avg_student_turn_s", "avg_examiner_turn_s",
    "interruptions_student", "interruptions_examiner", "interruptions_per_min",
    "overlap_s", "overlap_ratio", "backchannels_student", "backchannels_examiner",
    "n_latencies", "latency_mean_s", "latency_median_s", "latency_recent_s", "latency_p90_s",
    "n_transitions", "smooth_transition_ratio",
    "hesitations", "hesitations_per_min", "dead_air_s", "dead_air_ratio",
    "words", "wpm_recent", "wpm_overall", "fillers", "filler_per_100w",
    "noise_db", "snr_db",
]

# Added when a session is finalised, from the question/answer analysis
QA_FEATURE_NAMES: List[str] = [
    "n_questions", "answered_ratio", "mean_answer_words", "technical_terms_total", "mean_coverage", "hedge_per_100w",
]

FEATURE_NAMES: List[str] = ENGINE_FEATURE_NAMES + QA_FEATURE_NAMES


# ----------------------------------------------------------------- primitives
def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def curve(x: float, xs: Iterable[float], ys: Iterable[float]) -> float:
    """Piecewise-linear lookup (clamped at both ends)."""
    return float(np.interp(x, list(xs), list(ys)))


def wavg(pairs: List[Tuple[Optional[float], float]]) -> Optional[float]:
    """Weighted mean over the components that are available (None entries are skipped)."""
    num = den = 0.0
    for value, weight in pairs:
        if value is None:
            continue
        num += value * weight
        den += weight
    return num / den if den > 0 else None


def shrink(score: Optional[float], evidence: float, full_evidence: float) -> Optional[float]:
    """Pull a score toward the neutral 50 when there is little evidence. None stays None (no data)."""
    if score is None:
        return None
    k = clip(evidence / full_evidence, 0.0, 1.0)
    return 50.0 + (score - 50.0) * k


def first(*values):
    for v in values:
        if v is not None:
            return v
    return None


# ------------------------------------------------------------ component scores
def latency_score(seconds: float) -> float:
    # A viva answer needs thinking time: 0.3-2.5 s is ideal, >6 s is a long silence.
    return curve(seconds, [0, 0.3, 0.8, 2.5, 6.0, 10.0, 15.0], [70, 85, 100, 100, 45, 25, 15])


def wpm_score(wpm: float) -> float:
    # Conversational speech is roughly 110-160 words per minute.
    return curve(wpm, [0, 60, 110, 160, 220, 300], [10, 40, 100, 100, 45, 15])


def pause_score(hesitations_per_min: float) -> float:
    return curve(hesitations_per_min, [0, 2, 6, 12, 20], [100, 100, 65, 35, 20])


def filler_score(per_100_words: float) -> float:
    return curve(per_100_words, [0, 3, 6, 12, 20], [100, 100, 70, 35, 20])


def fluency_raw(f: Dict[str, Optional[float]]) -> Optional[float]:
    """Delivery fluency from speaking rate, in-answer hesitation pauses and filler words (0-100)."""
    wpm = first(f.get("wpm_recent"), f.get("wpm_overall"))
    hes = f.get("hesitations_per_min")
    fil = f.get("filler_per_100w")
    return wavg([
        (wpm_score(wpm) if wpm is not None else None, 0.40),
        (pause_score(hes) if hes is not None else None, 0.30),
        (filler_score(fil) if fil is not None else None, 0.30),
    ])


# --------------------------------------------------------------------- scoring
def _r(v: Optional[float], d: int = 1) -> Optional[float]:
    return None if v is None else round(float(v), d)


def compute_metrics(f: Dict[str, Optional[float]]) -> Dict[str, object]:
    """Live signals from audio timing + speech-rate. Returns None for a signal with no data yet."""
    S = float(f.get("student_speech_s") or 0.0)
    n_lat = float(f.get("n_latencies") or 0)
    talk_ratio = f.get("talk_ratio")

    lat = first(f.get("latency_recent_s"), f.get("latency_median_s"))
    promptness = shrink(latency_score(lat), n_lat, MIN_ANSWERS) if lat is not None else None
    fluency = shrink(fluency_raw(f), S, MIN_STUDENT_SPEECH_S)
    wpm = first(f.get("wpm_recent"), f.get("wpm_overall"))

    sufficiency = clip(S / MIN_STUDENT_SPEECH_S, 0, 1)
    return {
        "promptness_score": _r(promptness),
        "fluency_score": _r(fluency),
        "student_talk_ratio": _r(talk_ratio * 100.0, 1) if talk_ratio is not None else None,
        "avg_latency": _r(lat, 2),
        "avg_wpm": round(wpm) if wpm is not None else None,
        "data_sufficiency": round(sufficiency, 2),
        "status": "ok" if sufficiency >= 1.0 else "warming_up",
    }


def sanitize(obj):
    """Make nested dict/list JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
