"""
Question -> answer analysis from the transcript.

Splits the conversation into exchanges (an examiner question and the student's answer) and measures
OBSERVABLE properties of each answer: how soon it started, how long it was, how fast it was spoken,
which technical terms it contained, how many of the question's key words it picked up, and how many
hedging phrases / filler words it contained.

It does NOT judge whether an answer is correct - only the examiner can. Transcripts come from speech
recognition and can contain errors, so treat all figures as approximate evidence.
Pure Python (no web or ML imports): fully unit-tested in tests/test_grading_support.py.
"""
from __future__ import annotations

import re
from statistics import mean, median
from typing import Any, Dict, List, Optional

from app.services.glossary import find_terms
from app.services.interaction_engine import count_fillers, count_words

# Transcript chunks include ~0.12 s of pre-roll before speech starts and ~0.08 s of tail after it
# (see audio.py). Correct for that so latencies match the live engine's.
CHUNK_PAD_START_S = 0.12
CHUNK_PAD_END_S = 0.08

MIN_QUESTION_WORDS = 4        # shorter examiner utterances ("okay", "thank you") are acknowledgements
MERGE_GAP_S = 8.0             # consecutive examiner utterances this close form ONE (multi-part) question
MIN_ANSWER_WORDS = 5

_HEDGE_RE = re.compile(
    r"\b(?:i think|i guess|i suppose|i believe|maybe|perhaps|probably|not (?:really )?sure|"
    r"not (?:really )?certain|kind of|sort of|something like|or something|i don'?t (?:really )?(?:know|remember))\b",
    re.IGNORECASE,
)

_STOPWORDS = set("""
a an the and or but if then else of to in on at by for with from as is are was were be been being do does did
done have has had having can could would should will shall may might must you your yours me my we our us they
their them it its this that these those there here what why how when where which who whom whose explain describe
tell about difference differences between give example examples please mean means meaning define discuss
compare briefly short simply know understand say talk used use using also just like so very more most some any
into out up down over under again than too not no yes okay ok well one two three first second
""".split())

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _stem(word: str) -> str:
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
        return w[:-1]
    return w


def key_stems(text: str) -> set:
    return {_stem(w) for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS and len(w) > 2}


def _round(v: Optional[float], d: int = 2) -> Optional[float]:
    return None if v is None else round(float(v), d)


def analyze_transcript(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """`items`: dicts with keys speaker ('student'|'examiner'), start_s, end_s, text."""
    seq = sorted(
        (i for i in items if (i.get("text") or "").strip()),
        key=lambda i: (float(i["start_s"]), float(i["end_s"])),
    )

    raw: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for it in seq:
        text = it["text"].strip()
        if it["speaker"] == "examiner":
            if count_words(text) < MIN_QUESTION_WORDS and "?" not in text:
                continue                                    # acknowledgement, not a question
            if cur is not None and not cur["parts"] and float(it["start_s"]) - cur["q_end"] <= MERGE_GAP_S:
                cur["q_texts"].append(text)
                cur["q_end"] = max(cur["q_end"], float(it["end_s"]))
            else:
                cur = {"q_texts": [text], "q_start": float(it["start_s"]), "q_end": float(it["end_s"]), "parts": []}
                raw.append(cur)
        elif cur is not None:                               # student speech before the first question is ignored
            cur["parts"].append(it)

    exchanges: List[Dict[str, Any]] = []
    for idx, ex in enumerate(raw, start=1):
        question = " ".join(ex["q_texts"])
        parts = ex["parts"]
        answer = " ".join(p["text"].strip() for p in parts)
        a_words = count_words(answer)
        answered = a_words >= MIN_ANSWER_WORDS

        q_end = ex["q_end"] - CHUNK_PAD_END_S
        a_start = a_end = latency = None
        overlap = False
        speaking = 0.0
        if parts:
            a_start = float(parts[0]["start_s"]) + CHUNK_PAD_START_S
            a_end = float(parts[-1]["end_s"]) - CHUNK_PAD_END_S
            overlap = a_start < q_end
            latency = max(0.0, a_start - q_end)
            speaking = sum(max(0.0, (float(p["end_s"]) - CHUNK_PAD_END_S) - (float(p["start_s"]) + CHUNK_PAD_START_S)) for p in parts)

        # "Coverage": how much of the question's own vocabulary the answer picked up. Technical terms are the
        # fairest basis (a good answer need not repeat words like "explain" or "programming"); fall back to
        # general key words when the question has fewer than two glossary terms.
        q_terms, a_terms = find_terms(question), find_terms(answer)
        coverage, basis = None, None
        if answered and len(q_terms) >= 2:
            coverage, basis = len(set(q_terms) & set(a_terms)) / len(q_terms), "technical terms"
        elif answered:
            q_stems = key_stems(question)
            if len(q_stems) >= 2:
                coverage, basis = len(q_stems & key_stems(answer)) / len(q_stems), "key words"

        exchanges.append({
            "index": idx,
            "question": question,
            "q_start": _round(ex["q_start"]),
            "q_end": _round(q_end),
            "answer": answer,
            "a_start": _round(a_start),
            "a_end": _round(a_end),
            "latency_s": _round(latency),
            "overlap": overlap,
            "answered": answered,
            "answer_words": a_words,
            "answer_speaking_s": _round(speaking, 1),
            "wpm": _round(60.0 * a_words / speaking, 0) if speaking >= 4.0 and a_words >= 8 else None,
            "technical_terms": a_terms,
            "hedges": len(_HEDGE_RE.findall(answer)),
            "fillers": count_fillers(answer),
            "coverage": _round(coverage, 2),
            "coverage_basis": basis,
        })

    return {"exchanges": exchanges, "summary": _summarize(exchanges)}


def _summarize(ex: List[Dict[str, Any]]) -> Dict[str, Any]:
    answered = [e for e in ex if e["answered"]]
    lat = [e["latency_s"] for e in answered if e["latency_s"] is not None]
    words = [e["answer_words"] for e in answered]
    total_words = sum(e["answer_words"] for e in ex)
    terms = sorted({t for e in ex for t in e["technical_terms"]})
    cov = [e["coverage"] for e in answered if e["coverage"] is not None]
    hedges = sum(e["hedges"] for e in ex)
    fillers = sum(e["fillers"] for e in ex)
    return {
        "n_questions": len(ex),
        "n_answered": len(answered),
        "answered_ratio": _round(len(answered) / len(ex), 2) if ex else None,
        "median_latency_s": _round(median(lat), 2) if lat else None,
        "within_3s_ratio": _round(sum(1 for v in lat if v <= 3.0) / len(lat), 2) if lat else None,
        "mean_answer_words": _round(mean(words), 0) if words else None,
        "median_answer_words": _round(median(words), 0) if words else None,
        "longest_answer_words": max(words) if words else None,
        "short_answers": sum(1 for w in words if w < 15),
        "total_words": total_words,
        "technical_terms_total": len(terms),
        "technical_terms": terms[:40],
        "mean_coverage": _round(mean(cov), 2) if cov else None,
        "hedge_per_100w": _round(100.0 * hedges / total_words, 1) if total_words >= 30 else None,
        "filler_per_100w": _round(100.0 * fillers / total_words, 1) if total_words >= 30 else None,
    }
