from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, Column, DateTime, Enum, Float, ForeignKey, Index,
                        Integer, String, Text)
from sqlalchemy.orm import relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UserRole(str, enum.Enum):
    LECTURER = "lecturer"
    STUDENT = "student"


class SessionStatus:
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.STUDENT, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    sessions_as_lecturer = relationship("VivaSession", foreign_keys="VivaSession.lecturer_id", back_populates="lecturer")
    sessions_as_student = relationship("VivaSession", foreign_keys="VivaSession.student_id", back_populates="student")


class VivaSession(Base):
    __tablename__ = "viva_sessions"

    id = Column(Integer, primary_key=True, index=True)
    room_id = Column(String(50), unique=True, index=True, nullable=False)
    title = Column(String(150), nullable=True)
    lecturer_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    status = Column(String(20), default=SessionStatus.ACTIVE, nullable=False)

    consent_student = Column(Boolean, default=False, nullable=False)
    consent_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=utcnow)
    started_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    # Ground-truth label from the examiner (0-100 rubric score) for training your own scorer.
    lecturer_label = Column(Float, nullable=True)
    lecturer_notes = Column(Text, nullable=True)

    lecturer = relationship("User", foreign_keys=[lecturer_id], back_populates="sessions_as_lecturer")
    student = relationship("User", foreign_keys=[student_id], back_populates="sessions_as_student")
    telemetry_logs = relationship("TelemetryRecord", back_populates="session", cascade="all, delete-orphan")
    transcript_segments = relationship("TranscriptSegment", back_populates="session", cascade="all, delete-orphan")
    summary = relationship("SessionSummary", back_populates="session", uselist=False, cascade="all, delete-orphan")
    assessment = relationship("SessionAssessment", back_populates="session", uselist=False, cascade="all, delete-orphan")


class TelemetryRecord(Base):
    __tablename__ = "telemetry_records"
    __table_args__ = (Index("ix_telemetry_session_t", "session_id", "t_offset"),)

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("viva_sessions.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime(timezone=True), default=utcnow)
    t_offset = Column(Float, nullable=False, default=0.0)        # seconds since the session's first audio

    confidence_score = Column(Float)
    engagement_score = Column(Float)
    dominance_score = Column(Float)
    tension_score = Column(Float)
    synchronization_score = Column(Float)
    interaction_quality = Column(Float)
    attention_index = Column(Float, nullable=True)
    student_talk_ratio = Column(Float, nullable=True)
    avg_latency = Column(Float, nullable=True)
    avg_wpm = Column(Float, nullable=True)

    session = relationship("VivaSession", back_populates="telemetry_logs")


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"
    __table_args__ = (Index("ix_transcript_session_start", "session_id", "start_s"),)

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("viva_sessions.id", ondelete="CASCADE"), nullable=False)
    speaker = Column(String(20), nullable=False)                  # "student" | "examiner"
    start_s = Column(Float, nullable=False)
    end_s = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    sentiment_label = Column(String(20), nullable=True)
    sentiment_score = Column(Float, nullable=True)                # polarity in [-1, 1]

    session = relationship("VivaSession", back_populates="transcript_segments")


class SessionSummary(Base):
    __tablename__ = "session_summaries"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("viva_sessions.id", ondelete="CASCADE"), unique=True, nullable=False)
    duration_s = Column(Float, default=0.0)
    features = Column(JSON, nullable=False)                       # feature vector (training-ready)
    metrics = Column(JSON, nullable=False)                        # final indicator scores
    insights = Column(JSON, nullable=False)                       # rule-based observations
    predicted_rating = Column(Float, nullable=True)               # from the learned scorer, if available
    created_at = Column(DateTime(timezone=True), default=utcnow)

    session = relationship("VivaSession", back_populates="summary")


class SessionAssessment(Base):
    """The EXAMINER's own rubric ratings (1-5 per criterion). Stored separately from every AI output."""
    __tablename__ = "session_assessments"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("viva_sessions.id", ondelete="CASCADE"), unique=True, nullable=False)
    rubric_scores = Column(JSON, nullable=False, default=dict)   # {criterion_id: 1..5}
    overall_mark = Column(Float, nullable=True)                  # weighted % of the examiner's ratings (set when complete)
    notes = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    session = relationship("VivaSession", back_populates="assessment")
