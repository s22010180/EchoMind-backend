"""Synchronous DB helpers. Call from async code via asyncio.to_thread()."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from app.database import session_scope
from app.models import (SessionStatus, SessionSummary, TelemetryRecord, TranscriptSegment,
                        VivaSession)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def mark_started(session_id: int) -> None:
    with session_scope() as db:
        s = db.get(VivaSession, session_id)
        if s and s.started_at is None:
            s.started_at = _now()


def save_telemetry(session_id: int, t_offset: float, m: Dict) -> None:
    with session_scope() as db:
        db.add(TelemetryRecord(
            session_id=session_id, t_offset=round(float(t_offset), 2),
            confidence_score=m.get("confidence_score"), engagement_score=m.get("engagement_score"),
            dominance_score=m.get("dominance_score"), tension_score=m.get("tension_score"),
            synchronization_score=m.get("synchronization_score"),
            interaction_quality=m.get("interaction_quality"),
            attention_index=m.get("attention_index"), student_talk_ratio=m.get("student_talk_ratio"),
            avg_latency=m.get("avg_latency"), avg_wpm=m.get("avg_wpm"),
        ))


def save_transcript(session_id: int, item: Dict) -> None:
    with session_scope() as db:
        db.add(TranscriptSegment(
            session_id=session_id, speaker=item["role"], start_s=item["start"], end_s=item["end"],
            text=item["text"], sentiment_label=item.get("sentiment_label"),
            sentiment_score=item.get("sentiment_score"),
        ))


def save_consent(session_id: int, consent: bool) -> None:
    with session_scope() as db:
        s = db.get(VivaSession, session_id)
        if s:
            s.consent_student = consent
            s.consent_at = _now()


def complete_session(session_id: int, duration_s: float, features: Dict, metrics: Dict,
                     insights: list, predicted: Optional[float]) -> None:
    with session_scope() as db:
        s = db.get(VivaSession, session_id)
        if not s:
            return
        s.status = SessionStatus.COMPLETED
        s.ended_at = _now()
        summary = db.query(SessionSummary).filter(SessionSummary.session_id == session_id).first()
        if summary is None:
            summary = SessionSummary(session_id=session_id, features={}, metrics={}, insights=[])
            db.add(summary)
        summary.duration_s = round(float(duration_s), 1)
        summary.features = features
        summary.metrics = metrics
        summary.insights = insights
        summary.predicted_rating = predicted
