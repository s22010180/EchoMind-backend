"""
How far should the examiner trust the indicators for THIS session?

Indicators are only as good as the audio and the amount of speech behind them. This turns the data-quality
signals into a plain High / Moderate / Low rating with the reasons, so a low-quality session is never
mistaken for a poor performance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_POINTS = {"good": 100, "warn": 60, "poor": 20}


def _factor(name: str, status: str, detail: str) -> Dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def assess_reliability(features: Dict[str, Optional[float]], qa_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    f = features or {}
    factors: List[Dict[str, str]] = []

    speech = float(f.get("student_speech_s") or 0.0)
    if speech >= 60:
        factors.append(_factor("Amount of speech", "good", f"{speech:.0f} s of student speech captured."))
    elif speech >= 30:
        factors.append(_factor("Amount of speech", "warn", f"Only {speech:.0f} s of student speech; indicators are less stable."))
    else:
        factors.append(_factor("Amount of speech", "poor", f"Only {speech:.0f} s of student speech; indicators are provisional."))

    snr = f.get("snr_db")
    if snr is None:
        factors.append(_factor("Audio clarity", "warn", "Audio clarity could not be measured."))
    elif snr >= 18:
        factors.append(_factor("Audio clarity", "good", f"Clear audio (~{snr:.0f} dB above background noise)."))
    elif snr >= 12:
        factors.append(_factor("Audio clarity", "warn", f"Moderate background noise (~{snr:.0f} dB)."))
    else:
        factors.append(_factor("Audio clarity", "poor", f"Noisy audio (~{snr:.0f} dB); timing measures may be unreliable."))

    words = float(f.get("words") or 0.0)
    if speech >= 10:
        expected = speech * 2.0                       # ~120 words per minute
        ratio = words / expected if expected > 0 else 0.0
        if ratio >= 0.6:
            factors.append(_factor("Transcript coverage", "good", "Speech recognition captured most of what was said."))
        elif ratio >= 0.3:
            factors.append(_factor("Transcript coverage", "warn", "Speech recognition missed a noticeable part of the speech."))
        else:
            factors.append(_factor("Transcript coverage", "poor", "Little of the speech was transcribed; word-based figures are unreliable."))

    overlap = float(f.get("overlap_ratio") or 0.0)
    if overlap <= 0.05:
        factors.append(_factor("Echo / overlap", "good", "Little overlapping speech."))
    elif overlap <= 0.12:
        factors.append(_factor("Echo / overlap", "warn", "Some overlapping speech; could be real interruptions or speaker echo."))
    else:
        factors.append(_factor("Echo / overlap", "poor", "A lot of overlapping speech - likely speaker echo or network lag, not behaviour."))

    n_lat = int(f.get("n_latencies") or 0)
    if n_lat >= 3:
        factors.append(_factor("Question-answer exchanges", "good", f"{n_lat} exchanges measured."))
    elif n_lat >= 1:
        factors.append(_factor("Question-answer exchanges", "warn", f"Only {n_lat} exchange(s) measured."))
    else:
        factors.append(_factor("Question-answer exchanges", "poor", "No complete question-answer exchange was measured."))

    score = sum(_POINTS[x["status"]] for x in factors) / len(factors)
    level = "high" if score >= 80 else "moderate" if score >= 55 else "low"
    advice = {
        "high": "Conditions were good. The indicators are a reasonable supplement to your own observations.",
        "moderate": "Use the indicators cautiously and lean on your own observations where a factor above is flagged.",
        "low": "Data quality was poor. Give the indicators little weight and assess mainly from your own observations.",
    }[level]
    return {"level": level, "score": round(score, 0), "factors": factors, "advice": advice}
