"""
Compare this session with the examiner's OTHER completed vivas (same examiner only) so marks stay consistent
over time. Percentiles describe position in the examiner's own cohort - they are not a target.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.services.performance import PERFORMANCE_KEYS, profile_mean

MIN_COHORT = 5
MIN_SIMILAR = 3
SIMILAR_BAND = 8.0


def percentile_rank(value: float, others: List[float]) -> Optional[float]:
    if not others:
        return None
    less = sum(1 for o in others if o < value)
    equal = sum(1 for o in others if o == value)
    return round(100.0 * (less + 0.5 * equal) / len(others), 0)


def build_cohort(current: Dict[str, Any], others: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    current: {"scores": {indicator: score|None}, "ok": bool}  (ok = enough evidence to compare)
    others : [{"scores": {...}, "label": float|None}]        (sufficiently-evidenced sessions only)
    """
    n = len(others)
    scores = (current or {}).get("scores") or {}
    usable = n >= MIN_COHORT and bool((current or {}).get("ok"))
    metrics = []
    for key in PERFORMANCE_KEYS:
        value = scores.get(key)
        pool = [float(o["scores"][key]) for o in others if (o.get("scores") or {}).get(key) is not None]
        metrics.append({"key": key, "value": value,
                        "percentile": percentile_rank(float(value), pool) if usable and value is not None and pool else None})

    similar = None
    mine = profile_mean(scores)
    if usable and mine is not None:
        marks = []
        for o in others:
            theirs = profile_mean(o.get("scores") or {})
            if o.get("label") is not None and theirs is not None and abs(theirs - mine) <= SIMILAR_BAND:
                marks.append(float(o["label"]))
        if len(marks) >= MIN_SIMILAR:
            similar = {"n": len(marks), "min": min(marks), "max": max(marks)}

    return {"n_sessions": n, "min_required": MIN_COHORT, "metrics": metrics, "similar": similar}
