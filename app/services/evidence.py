"""
Rubric and rubric-aligned EVIDENCE.

The system never rates the student. For each rubric criterion it lists the observable measurements that are
relevant, and the examiner chooses the 1-5 rating themselves. The overall mark is plain arithmetic over the
examiner's own ratings.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

RUBRIC: List[Dict[str, Any]] = [
    {"id": "technical_knowledge", "title": "Technical knowledge & accuracy", "weight": 35,
     "description": "Correctness and relevance of the technical content. Only you can judge accuracy - the system shows only what was said."},
    {"id": "depth", "title": "Depth of explanation & reasoning", "weight": 20,
     "description": "Does the student justify, give examples and reason beyond a one-line answer?"},
    {"id": "communication", "title": "Clarity of communication", "weight": 20,
     "description": "How clearly and fluently ideas were conveyed."},
    {"id": "responsiveness", "title": "Responsiveness to questions", "weight": 15,
     "description": "Whether questions were answered, and how readily."},
    {"id": "interaction", "title": "Professional interaction", "weight": 10,
     "description": "Turn-taking and conduct during the discussion."},
]
RUBRIC_IDS = [c["id"] for c in RUBRIC]


def compute_overall(scores: Dict[str, int]) -> Optional[float]:
    """Weighted percentage of the examiner's own 1-5 ratings. None until every criterion is rated."""
    if any(not isinstance(scores.get(cid), int) or not 1 <= scores[cid] <= 5 for cid in RUBRIC_IDS):
        return None
    total = sum(c["weight"] for c in RUBRIC)
    return round(sum(c["weight"] * scores[c["id"]] / 5.0 for c in RUBRIC) / total * 100.0, 1)


def _item(key: str, label: str, display: str, status: str = "info", hint: Optional[str] = None) -> Dict[str, Any]:
    return {"key": key, "label": label, "display": display, "status": status, "hint": hint}   # status: typical | review | info


def _n(v: Optional[float], d: int = 0) -> str:
    return f"{v:.{d}f}"


def build_evidence(features: Dict[str, Any], scores: Dict[str, Any], qa: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    f, m, q = features or {}, scores or {}, qa or {}
    out: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in RUBRIC_IDS}

    # ---- technical knowledge
    if q.get("n_questions"):
        out["technical_knowledge"].append(_item("answered", "Questions answered", f"{q['n_answered']} of {q['n_questions']}",
                                                hint="Questions that received a spoken answer of at least 5 words."))
        terms = q.get("technical_terms") or []
        out["technical_knowledge"].append(_item("terms", "Distinct technical terms used", str(q.get("technical_terms_total", 0)),
                                                hint=("e.g. " + ", ".join(terms[:8])) if terms else "No glossary terms were detected."))
        if q.get("mean_coverage") is not None:
            out["technical_knowledge"].append(_item("coverage", "Question terms addressed", f"{q['mean_coverage'] * 100:.0f}%",
                                                    hint="Share of each question's technical terms (or key words) that the answer picked up - a rough on-topic cue. A correct answer can use other words."))
    # ---- depth
    if q.get("mean_answer_words") is not None:
        out["depth"].append(_item("mean_words", "Average answer length", f"{_n(q['mean_answer_words'])} words"))
        out["depth"].append(_item("longest", "Longest answer", f"{q['longest_answer_words']} words"))
        out["depth"].append(_item("short", "Very short answers (< 15 words)", str(q.get("short_answers", 0)),
                                  "review" if (q.get("short_answers") or 0) >= max(2, (q.get("n_answered") or 0) // 2) else "info",
                                  "Short answers may mean a closed question or limited elaboration."))
    # ---- communication
    wpm = f.get("wpm_overall")
    if wpm is not None:
        out["communication"].append(_item("wpm", "Speaking rate", f"{wpm:.0f} wpm", "typical" if 110 <= wpm <= 170 else "review",
                                          "Conversational speech is roughly 110-170 words per minute."))
    hes = f.get("hesitations_per_min")
    if hes is not None:
        out["communication"].append(_item("hesitations", "Hesitation pauses", f"{hes:.1f} / min", "typical" if hes <= 5 else "review",
                                          "Pauses of 0.8-3 s inside an answer."))
    fil = q.get("filler_per_100w") if q.get("filler_per_100w") is not None else f.get("filler_per_100w")
    if fil is not None:
        out["communication"].append(_item("fillers", "Filler words", f"{fil:.1f} per 100 words", "typical" if fil <= 5 else "review",
                                          "Speech recognition under-reports fillers, so the true rate may be higher."))
    if q.get("hedge_per_100w") is not None:
        out["communication"].append(_item("hedges", "Hedging phrases", f"{q['hedge_per_100w']:.1f} per 100 words",
                                          hint="Phrases such as 'I think' or 'maybe'. Common in careful speakers and in a second language - not a measure of knowledge."))
    if m.get("fluency") is not None:
        out["communication"].append(_item("fluency", "Delivery Fluency indicator", f"{m['fluency']:.0f} / 100",
                                          hint="Summary of speaking rate, hesitation pauses and filler words."))
    snr = f.get("snr_db")
    if snr is not None:
        out["communication"].append(_item("snr", "Audio clarity", f"{snr:.0f} dB", "typical" if snr >= 15 else "review",
                                          "Below ~12 dB, timing and word-based measures become less reliable."))
    # ---- responsiveness
    lat = q.get("median_latency_s") if q.get("median_latency_s") is not None else f.get("latency_median_s")
    if lat is not None:
        out["responsiveness"].append(_item("latency", "Median time to start answering", f"{lat:.1f} s",
                                           "typical" if lat <= 3.0 else "review" if lat > 6.0 else "info",
                                           "A short pause to think is normal for technical questions."))
    if q.get("within_3s_ratio") is not None:
        out["responsiveness"].append(_item("within3", "Answers started within 3 s", f"{q['within_3s_ratio'] * 100:.0f}%",
                                           "typical" if q["within_3s_ratio"] >= 0.7 else "info"))
    if q.get("answered_ratio") is not None:
        out["responsiveness"].append(_item("answered_ratio", "Questions answered", f"{q['answered_ratio'] * 100:.0f}%",
                                           "typical" if q["answered_ratio"] >= 0.9 else "review"))
    # ---- interaction
    if f.get("interruptions_student") is not None:
        out["interaction"].append(_item("int_student", "Interruptions by student", str(int(f["interruptions_student"] or 0)),
                                        "review" if (f["interruptions_student"] or 0) >= 3 else "typical"))
        out["interaction"].append(_item("int_examiner", "Interruptions by examiner", str(int(f.get("interruptions_examiner") or 0)),
                                        hint="Interruptions can disrupt answers."))
    if f.get("smooth_transition_ratio") is not None and (f.get("n_transitions") or 0) >= 4:
        out["interaction"].append(_item("smooth", "Well-timed hand-overs", f"{f['smooth_transition_ratio'] * 100:.0f}%",
                                        "typical" if f["smooth_transition_ratio"] >= 0.75 else "review"))
    if f.get("overlap_s") is not None and (f.get("student_speech_s") or 0) > 10:
        out["interaction"].append(_item("overlap", "Overlapping speech", f"{f['overlap_s']:.1f} s",
                                        "review" if (f.get("overlap_ratio") or 0) > 0.12 else "typical",
                                        "High overlap is often speaker echo or network lag rather than behaviour."))

    for cid in RUBRIC_IDS:
        if not out[cid]:
            out[cid].append(_item("none", "No measurements", "Not available", "info",
                                  "Analysis was off or too little speech was captured. Assess from your own observation."))
    return out


def build_conditions(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Unscored facts about the SESSION (not the student): who spoke, interruptions, overlap, silence. They help the
    examiner interpret the indicators - e.g. interruptions by the examiner are not the student's doing.
    """
    f = features or {}
    out: List[Dict[str, Any]] = []
    if f.get("talk_ratio") is not None:
        out.append(_item("talk", "Student's share of speaking time", f"{f['talk_ratio'] * 100:.0f}%",
                         hint="Share of all speech that came from the student."))
    if f.get("interruptions_examiner") is not None:
        out.append(_item("int_ex", "Interruptions by the examiner", str(int(f.get("interruptions_examiner") or 0)),
                         hint="Being cut off can shorten answers and delay the next one."))
        out.append(_item("int_st", "Interruptions by the student", str(int(f.get("interruptions_student") or 0))))
    if f.get("overlap_s") is not None and (f.get("student_speech_s") or 0) > 10:
        out.append(_item("overlap", "Overlapping speech", f"{f['overlap_s']:.1f} s",
                         "review" if (f.get("overlap_ratio") or 0) > 0.12 else "info",
                         "High overlap is often speaker echo or network lag rather than behaviour."))
    if f.get("dead_air_s") is not None and (f.get("elapsed_s") or 0) > 20:
        out.append(_item("silence", "Silence between speakers (3 s or more)", f"{f['dead_air_s']:.0f} s",
                         hint="Long silences can be thinking time, reading time or a technical problem."))
    return out
