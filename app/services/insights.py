"""Plain-language, rule-based observations for the lecturer's report."""
from __future__ import annotations

from typing import Dict, List, Optional

DISCLAIMER = (
    "These indicators describe observable viva behaviour (response timing, fluency, answer length and the "
    "technical vocabulary used). They are decision-support signals, not a grade and not a measure of "
    "knowledge, honesty or emotional state. Final assessment must remain with the examiner."
)


def _i(level: str, title: str, message: str) -> Dict[str, str]:
    return {"level": level, "title": title, "message": message}   # level: positive | info | attention


def generate_insights(f: Dict[str, Optional[float]], m: Dict[str, object]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    S = f.get("student_speech_s") or 0.0
    if S < 45:
        out.append(_i("info", "Limited data",
                      "The student spoke for less than 45 seconds, so indicators are provisional."))

    snr = f.get("snr_db")
    if snr is not None and snr < 12:
        out.append(_i("attention", "Audio quality",
                      f"Estimated signal-to-noise ratio was low ({snr:.0f} dB). Timing and speech-rate "
                      "indicators may be less reliable; a headset microphone is recommended."))

    lat = f.get("latency_median_s")
    n_lat = f.get("n_latencies") or 0
    if lat is not None and n_lat >= 2:
        if lat <= 3.0:
            out.append(_i("positive", "Response timing",
                          f"Median time to start answering was {lat:.1f} s, within a typical thinking window."))
        elif lat <= 6.0:
            out.append(_i("info", "Response timing",
                          f"Median time to start answering was {lat:.1f} s, a little long; this can reflect "
                          "careful thought on difficult questions."))
        else:
            out.append(_i("attention", "Response timing",
                          f"Median time to start answering was {lat:.1f} s. Long silences may indicate difficulty "
                          "with the questions or technical issues."))

    wpm = f.get("wpm_overall")
    if wpm is not None:
        if wpm < 100:
            out.append(_i("info", "Speech rate", f"Speaking rate was slow ({wpm:.0f} wpm)."))
        elif wpm > 190:
            out.append(_i("attention", "Speech rate", f"Speaking rate was fast ({wpm:.0f} wpm); clarity may suffer."))
        else:
            out.append(_i("positive", "Speech rate", f"Speaking rate was comfortable ({wpm:.0f} wpm)."))

    fil = f.get("filler_per_100w")
    if fil is not None and fil >= 6:
        out.append(_i("attention", "Filler words",
                      f"About {fil:.0f} filler expressions per 100 words were detected "
                      "(speech recognition tends to under-report fillers, so the true rate may be higher)."))

    hes = f.get("hesitations_per_min")
    if hes is not None and hes >= 6:
        out.append(_i("attention", "Hesitation pauses",
                      f"{hes:.1f} hesitation pauses per minute of speech were observed within answers."))

    tr = f.get("talk_ratio")
    if tr is not None and (f.get("student_speech_s") or 0) + (f.get("examiner_speech_s") or 0) > 60:
        if tr < 0.35:
            out.append(_i("attention", "Speaking share",
                          f"The student held {tr*100:.0f}% of speaking time, which is low for a viva."))
        elif tr > 0.85:
            out.append(_i("info", "Speaking share",
                          f"The student held {tr*100:.0f}% of speaking time; the examiner spoke very little."))
        else:
            out.append(_i("positive", "Speaking share",
                          f"The student held {tr*100:.0f}% of speaking time, a balanced exchange."))

    i_s = f.get("interruptions_student") or 0
    i_e = f.get("interruptions_examiner") or 0
    if i_s >= 3:
        out.append(_i("attention", "Interruptions",
                      f"The student interrupted the examiner {int(i_s)} times."))
    if i_e >= 3:
        out.append(_i("info", "Interruptions",
                      f"The examiner interrupted the student {int(i_e)} times, which may have disrupted answers."))

    sm = f.get("smooth_transition_ratio")
    if sm is not None and (f.get("n_transitions") or 0) >= 6:
        if sm >= 0.75:
            out.append(_i("positive", "Turn-taking", "Most speaker changes were smooth and well-timed."))
        elif sm < 0.5:
            out.append(_i("attention", "Turn-taking", "Many speaker changes were delayed or overlapping."))

    return out
