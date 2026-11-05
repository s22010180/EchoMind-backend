from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.models import UserRole


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------- auth
class UserRegister(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: UserRole
    invite_code: Optional[str] = Field(default=None, max_length=100)

    @field_validator("full_name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 2:
            raise ValueError("Name is too short")
        return v

    @field_validator("password")
    @classmethod
    def _password(cls, v: str) -> str:
        if not any(c.isalpha() for c in v) or not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one letter and one number")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserBrief(ORM):
    id: int
    full_name: str


class UserResponse(ORM):
    id: int
    full_name: str
    email: EmailStr
    role: UserRole


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ---------------------------------------------------------------- sessions
class SessionCreate(BaseModel):
    student_id: int
    title: Optional[str] = Field(default=None, max_length=150)


class SessionResponse(ORM):
    id: int
    room_id: str
    title: Optional[str] = None
    status: str
    lecturer_id: int
    student_id: int
    consent_student: bool
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    lecturer_label: Optional[float] = None
    lecturer: Optional[UserBrief] = None
    student: Optional[UserBrief] = None


class ConsentUpdate(BaseModel):
    consent: bool


class LabelUpdate(BaseModel):
    label: float = Field(ge=0, le=100, description="Examiner's overall rubric score for the viva (0-100)")
    notes: Optional[str] = Field(default=None, max_length=4000)


class TelemetryPoint(ORM):
    t_offset: float
    confidence_score: Optional[float] = None
    engagement_score: Optional[float] = None
    dominance_score: Optional[float] = None
    tension_score: Optional[float] = None
    synchronization_score: Optional[float] = None
    interaction_quality: Optional[float] = None
    attention_index: Optional[float] = None
    student_talk_ratio: Optional[float] = None
    avg_latency: Optional[float] = None
    avg_wpm: Optional[float] = None


class TranscriptItem(ORM):
    speaker: str
    start_s: float
    end_s: float
    text: str
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None


class AssessmentIn(BaseModel):
    scores: Dict[str, int] = Field(description="Rubric criterion id -> examiner's rating 1-5")
    notes: Optional[str] = Field(default=None, max_length=4000)


class ReportResponse(BaseModel):
    session: SessionResponse
    duration_s: float
    metrics: Dict[str, Any]
    features: Dict[str, Any]
    insights: List[Dict[str, str]]
    predicted_rating: Optional[float] = None
    timeline: List[TelemetryPoint]
    transcript: List[TranscriptItem]
    disclaimer: str
    # grading support
    performance: Dict[str, Any]
    conditions: List[Dict[str, Any]]
    qa: Dict[str, Any]
    reliability: Dict[str, Any]
    evidence: Dict[str, List[Dict[str, Any]]]
    rubric: List[Dict[str, Any]]
    assessment: Optional[Dict[str, Any]] = None
    cohort: Optional[Dict[str, Any]] = None
