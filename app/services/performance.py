"""
Student PERFORMANCE indicators for a viva (0-100 each) - the headline of the report.

Built from the question-by-question analysis (qa_analysis.py) plus the live delivery measures. There is
deliberately NO single overall score: five separate indicators let the examiner see WHAT was strong or weak,
and avoid a number that looks like a grade.

  promptness   how soon answers began after each question
  fluency      how smoothly answers were delivered (rate, pauses, fillers)
  depth        how many questions were answered, and how developed the answers were
  technical    syllabus terms used, and how many of each question's own terms the answer addressed
  consistency  share of questions that got an adequate, reasonably prompt answer

Technical / depth / consistency describe what was SAID. They cannot tell whether it was correct - only the
examiner can. Scores are pulled toward 50 until enough questions support them.
"""
from __future__ import annotations

from statistics import mean, median
from typing import Any, Dict, List, Optional

from app.services.scoring import (MIN_STUDENT_SPEECH_S, QA_FEATURE_NAMES, clip, curve, first, fluency_raw,
                                  latency_score, shrink, wavg)

PERFORMANCE_KEYS: List[str] = ["promptness", "fluency", "depth", "technical", "consistency"]

FULL_EVIDENCE_QUESTIONS = 5      # questions needed for fully-trusted question-based indicators
MIN_QUESTIONS = 3                # fewer than this -> the whole profile is flagged provisional
ADEQUATE_WORDS = 15              # an "adequate" answer has at least this many words
ADEQUATE_LATENCY_S = 8.0         # ...and began within this many seconds


def _ind(score: Optional[float], n: int, detail: str) -> Dict[str, Any]:
    return {"score": None if score is None else round(float(score), 1), "n": n, "detail": detail}


def compute_performance(features: Dict[str, Any], qa: Dict[str, Any]) -> Dict[str, Any]:
    f = features or {}
    exchanges: List[Dict[str, Any]] = (qa or {}).get("exchanges") or []
    summary: Dict[str, Any] = (qa or {}).get("summary") or {}
    answered = [e for e in exchanges if e.get("answered")]
    n_q, n_a = len(exchanges), len(answered)

    # ---- promptness: average of per-question latency scores (fallback: the live engine's latencies)
    lat_scores = [latency_score(e["latency_s"]) for e in answered if e.get("latency_s") is not None]
    if lat_scores:
        prompt = shrink(mean(lat_scores), len(lat_scores), FULL_EVIDENCE_QUESTIONS)
        prompt_ind = _ind(prompt, len(lat_scores), f"Based on {len(lat_scores)} answered question(s).")
    elif f.get("latency_median_s") is not None:
        n_lat = int(f.get("n_latencies") or 0)
        prompt_ind = _ind(shrink(latency_score(f["latency_median_s"]), n_lat, FULL_EVIDENCE_QUESTIONS), n_lat,
                          f"Based on {n_lat} measured question-answer exchange(s).")
    else:
        prompt_ind = _ind(None, 0, "No question-answer exchange was measured.")

    # ---- fluency
    speech = float(f.get("student_speech_s") or 0.0)
    fl = shrink(fluency_raw(f), speech, MIN_STUDENT_SPEECH_S)
    fluency_ind = _ind(fl, int(speech), f"Based on {speech:.0f} s of student speech.")

    # ---- depth: completion + elaboration
    if n_q:
        completion = 100.0 * n_a / n_q
        med_words = median([e["answer_words"] for e in answered]) if answered else 0.0
        elaboration = curve(med_words, [0, 8, 15, 30, 60], [0, 25, 55, 85, 100])
        depth = shrink(wavg([(completion, 0.4), (elaboration, 0.6)]), n_q, FULL_EVIDENCE_QUESTIONS)
        depth_ind = _ind(depth, n_q, f"{n_a} of {n_q} questions answered; median answer {med_words:.0f} words.")
    else:
        depth_ind = _ind(None, 0, "No questions were detected in the transcript.")

    # ---- technical content: specificity + coverage of the question's own terms
    if n_a:
        per_answer = sum(len(e.get("technical_terms") or []) for e in answered) / n_a
        specificity = curve(per_answer, [0, 0.5, 1.5, 3.0], [0, 40, 80, 100])
        cov = summary.get("mean_coverage")
        tech = shrink(wavg([(specificity, 0.6), (cov * 100.0 if cov is not None else None, 0.4)]), n_a, FULL_EVIDENCE_QUESTIONS)
        tech_ind = _ind(tech, n_a, f"{per_answer:.1f} syllabus terms per answer on average.")
    else:
        tech_ind = _ind(None, 0, "No answered questions to analyse.")

    # ---- consistency: share of questions with an adequate answer that began promptly
    if n_q >= MIN_QUESTIONS:
        adequate = sum(1 for e in answered if e["answer_words"] >= ADEQUATE_WORDS
                       and (e.get("latency_s") is None or e["latency_s"] <= ADEQUATE_LATENCY_S))
        cons_ind = _ind(100.0 * adequate / n_q, n_q, f"{adequate} of {n_q} questions received an adequate, prompt answer.")
    else:
        cons_ind = _ind(None, n_q, f"Needs at least {MIN_QUESTIONS} questions (found {n_q}).")

    indicators = {"promptness": prompt_ind, "fluency": fluency_ind, "depth": depth_ind,
                  "technical": tech_ind, "consistency": cons_ind}
    provisional = n_q < MIN_QUESTIONS or speech < MIN_STUDENT_SPEECH_S
    reason = None
    if n_q < MIN_QUESTIONS:
        reason = f"Only {n_q} question(s) were detected; at least {MIN_QUESTIONS} are needed for a stable profile."
    elif speech < MIN_STUDENT_SPEECH_S:
        reason = f"Only {speech:.0f} s of student speech was captured (about {MIN_STUDENT_SPEECH_S:.0f} s are needed)."
    return {
        "indicators": indicators,
        "scores": {k: v["score"] for k, v in indicators.items()},
        "n_questions": n_q,
        "n_answered": n_a,
        "provisional": provisional,
        "reason": reason,
    }


def qa_features(summary: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Question/answer measures stored with the session (for the research dataset)."""
    s = summary or {}
    mapping = {
        "n_questions": s.get("n_questions"),
        "answered_ratio": s.get("answered_ratio"),
        "mean_answer_words": s.get("mean_answer_words"),
        "technical_terms_total": s.get("technical_terms_total"),
        "mean_coverage": s.get("mean_coverage"),
        "hedge_per_100w": s.get("hedge_per_100w"),
    }
    return {k: mapping[k] for k in QA_FEATURE_NAMES}


def profile_mean(scores: Dict[str, Optional[float]]) -> Optional[float]:
    """INTERNAL helper (consistency check only): mean of the available indicators. Never shown as a score."""
    vals = [float(v) for v in scores.values() if v is not None]
    return mean(vals) if vals else None
