from __future__ import annotations

import csv
import hashlib
import io
import secrets
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session, joinedload

from app import schemas
from app.config import settings
from app.database import get_db
from app.models import (SessionAssessment, SessionStatus, SessionSummary, TranscriptSegment, User,
                        UserRole, VivaSession)
from app.security import get_current_user, require_lecturer, require_student
from app.services import persistence
from app.services.cohort import build_cohort
from app.services.evidence import RUBRIC, RUBRIC_IDS, build_conditions, build_evidence, compute_overall
from app.services.performance import PERFORMANCE_KEYS, compute_performance
from app.services.qa_analysis import analyze_transcript
from app.services.reliability import assess_reliability
from app.services.insights import DISCLAIMER
from app.services.runtime import ServiceContext, finalize_room, safe_send
from app.services.scoring import FEATURE_NAMES

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _ctx(request: Request) -> ServiceContext:
    return request.app.state.ctx


def _load(db: Session, session_id: int, user: User) -> VivaSession:
    s = (db.query(VivaSession)
         .options(joinedload(VivaSession.lecturer), joinedload(VivaSession.student))
         .filter(VivaSession.id == session_id).first())
    if s is None or user.id not in (s.lecturer_id, s.student_id):
        raise HTTPException(status_code=404, detail="Session not found")     # do not leak existence
    return s


# ------------------------------------------------------------------ create/list
@router.post("", response_model=schemas.SessionResponse, status_code=201)
@router.post("/create", response_model=schemas.SessionResponse, status_code=201, include_in_schema=False)
def create_session(data: schemas.SessionCreate, db: Session = Depends(get_db),
                   lecturer: User = Depends(require_lecturer)):
    student = db.get(User, data.student_id)
    if student is None or student.role != UserRole.STUDENT or not student.is_active:
        raise HTTPException(status_code=404, detail="Student not found")
    already = (db.query(VivaSession)
               .filter(VivaSession.student_id == student.id, VivaSession.status == SessionStatus.ACTIVE).first())
    if already is not None:
        raise HTTPException(status_code=409, detail="This student already has an active viva session")

    for _ in range(5):
        room_id = secrets.token_urlsafe(9)
        if not db.query(VivaSession.id).filter(VivaSession.room_id == room_id).first():
            break
    session = VivaSession(room_id=room_id, title=data.title, lecturer_id=lecturer.id,
                          student_id=student.id, status=SessionStatus.ACTIVE)
    db.add(session)
    db.commit()
    return _load(db, session.id, lecturer)


@router.get("", response_model=List[schemas.SessionResponse])
def list_sessions(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    col = VivaSession.lecturer_id if user.role == UserRole.LECTURER else VivaSession.student_id
    return (db.query(VivaSession)
            .options(joinedload(VivaSession.lecturer), joinedload(VivaSession.student))
            .filter(col == user.id)
            .order_by(VivaSession.created_at.desc())
            .limit(200).all())


# --------------------------------------------------- dataset export (before /{id})
@router.get("/export/dataset.csv")
def export_dataset(db: Session = Depends(get_db), lecturer: User = Depends(require_lecturer)):
    """One row per completed session: features + indicator scores + your rubric label.
    Student identity is replaced with a stable pseudonymous ID."""
    rows = (db.query(VivaSession, SessionSummary)
            .join(SessionSummary, SessionSummary.session_id == VivaSession.id)
            .filter(VivaSession.lecturer_id == lecturer.id, VivaSession.status == SessionStatus.COMPLETED)
            .order_by(VivaSession.id).all())
    buf = io.StringIO()
    w = csv.writer(buf)
    ratings = {a.session_id: (a.rubric_scores or {}) for a in db.query(SessionAssessment).all()}
    w.writerow(["session_id", "student_pseudo_id", "duration_s", "lecturer_label"] + FEATURE_NAMES
               + [f"perf_{k}" for k in PERFORMANCE_KEYS] + [f"rating_{c}" for c in RUBRIC_IDS])
    for sess, summ in rows:
        pseudo = hashlib.sha256(f"{sess.student_id}:{settings.SECRET_KEY}".encode()).hexdigest()[:10]
        feats, mets = summ.features or {}, summ.metrics or {}
        w.writerow([sess.id, pseudo, summ.duration_s, "" if sess.lecturer_label is None else sess.lecturer_label]
                   + ["" if feats.get(n) is None else feats[n] for n in FEATURE_NAMES]
                   + ["" if (mets.get("performance") or {}).get(k) is None else mets["performance"][k] for k in PERFORMANCE_KEYS]
                   + ["" if ratings.get(sess.id, {}).get(c) is None else ratings[sess.id][c] for c in RUBRIC_IDS])
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=viva_dataset.csv"})


# ------------------------------------------------------------------- single item
@router.get("/{session_id}", response_model=schemas.SessionResponse)
def get_session(session_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return _load(db, session_id, user)


@router.post("/{session_id}/consent", response_model=schemas.SessionResponse)
async def set_consent(session_id: int, body: schemas.ConsentUpdate, request: Request,
                      db: Session = Depends(get_db), student: User = Depends(require_student)):
    """The student must opt in before any audio/visual analysis starts; can withdraw any time."""
    s = _load(db, session_id, student)
    if s.status != SessionStatus.ACTIVE:
        raise HTTPException(status_code=409, detail="Session is not active")
    persistence.save_consent(s.id, body.consent)
    db.refresh(s)
    room = _ctx(request).get(s.room_id)
    if room is not None:
        room.consent = body.consent
        for r in ("student", "examiner"):          # audio pauses/resumes -> re-sync stream clocks
            room.engine.reset_stream(r)
        event = {"type": "consent_granted" if body.consent else "consent_withdrawn"}
        for ws in list(room.telemetry.values()) + list(room.signaling.values()):
            await safe_send(ws, event)
    return s


@router.post("/{session_id}/end")
async def end_session(session_id: int, request: Request, db: Session = Depends(get_db),
                      lecturer: User = Depends(require_lecturer)):
    s = _load(db, session_id, lecturer)
    if s.lecturer_id != lecturer.id:
        raise HTTPException(status_code=403, detail="Only the session's lecturer can end it")
    if s.status == SessionStatus.COMPLETED:
        return {"status": "COMPLETED", "already_completed": True}
    result = await finalize_room(_ctx(request), s.id, s.room_id)
    return {"status": "COMPLETED", **result}


@router.patch("/{session_id}/label", response_model=schemas.SessionResponse)
def set_label(session_id: int, body: schemas.LabelUpdate, db: Session = Depends(get_db),
              lecturer: User = Depends(require_lecturer)):
    """Record the examiner's own rubric score; this is the training target for your scorer."""
    s = _load(db, session_id, lecturer)
    if s.lecturer_id != lecturer.id:
        raise HTTPException(status_code=403, detail="Only the session's lecturer can label it")
    s.lecturer_label = body.label
    s.lecturer_notes = body.notes
    db.commit()
    db.refresh(s)
    return s


@router.get("/{session_id}/report", response_model=schemas.ReportResponse)
def get_report(session_id: int, db: Session = Depends(get_db), lecturer: User = Depends(require_lecturer)):
    s = _load(db, session_id, lecturer)
    if s.lecturer_id != lecturer.id:
        raise HTTPException(status_code=403, detail="Only the session's lecturer can view the report")
    summary = db.query(SessionSummary).filter(SessionSummary.session_id == s.id).first()
    if summary is None:
        raise HTTPException(status_code=409, detail="Report is available after the session has ended")

    transcript = (db.query(TranscriptSegment).filter(TranscriptSegment.session_id == s.id)
                  .order_by(TranscriptSegment.start_s).all())

    # ---- grading support (derived on the fly from stored data; nothing here is an AI grade)
    features, metrics = summary.features or {}, summary.metrics or {}
    qa = analyze_transcript([{"speaker": t.speaker, "start_s": t.start_s, "end_s": t.end_s, "text": t.text} for t in transcript])
    performance = compute_performance(features, qa)
    reliability = assess_reliability(features, qa["summary"])
    evidence = build_evidence(features, performance["scores"], qa["summary"])
    conditions = build_conditions(features)

    others = (db.query(VivaSession, SessionSummary)
              .join(SessionSummary, SessionSummary.session_id == VivaSession.id)
              .filter(VivaSession.lecturer_id == lecturer.id, VivaSession.status == SessionStatus.COMPLETED,
                      VivaSession.id != s.id).all())
    cohort = build_cohort(
        {"scores": performance["scores"], "ok": not performance["provisional"]},
        [{"scores": (sm.metrics or {}).get("performance") or {}, "label": ss.lecturer_label}
         for ss, sm in others if (sm.metrics or {}).get("performance_ok")])

    row = db.query(SessionAssessment).filter(SessionAssessment.session_id == s.id).first()
    assessment: Optional[Dict[str, Any]] = None
    if row is not None:
        assessment = {"scores": row.rubric_scores or {}, "overall_mark": row.overall_mark,
                      "notes": row.notes, "updated_at": row.updated_at}

    return {
        "session": s, "duration_s": summary.duration_s or 0.0, "metrics": metrics,
        "features": features, "insights": summary.insights or [],
        "predicted_rating": summary.predicted_rating, "timeline": [], "transcript": transcript,
        "disclaimer": DISCLAIMER,
        "qa": qa, "performance": performance, "conditions": conditions,
        "reliability": reliability, "evidence": evidence, "rubric": RUBRIC,
        "assessment": assessment, "cohort": cohort,
    }


@router.put("/{session_id}/assessment")
def save_assessment(session_id: int, body: schemas.AssessmentIn, db: Session = Depends(get_db),
                    lecturer: User = Depends(require_lecturer)):
    """Save the EXAMINER's own rubric ratings (1-5 per criterion). The overall mark is plain weighted arithmetic
    over those ratings and is also stored as the session's training label once every criterion is rated."""
    s = _load(db, session_id, lecturer)
    if s.lecturer_id != lecturer.id:
        raise HTTPException(status_code=403, detail="Only the session's lecturer can assess it")
    if s.status != SessionStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Assess the viva after it has ended")
    bad = [k for k in body.scores if k not in RUBRIC_IDS]
    if bad:
        raise HTTPException(status_code=422, detail=f"Unknown rubric criteria: {', '.join(bad)}")
    if any(not 1 <= v <= 5 for v in body.scores.values()):
        raise HTTPException(status_code=422, detail="Each rating must be between 1 and 5")

    overall = compute_overall(body.scores)
    row = db.query(SessionAssessment).filter(SessionAssessment.session_id == s.id).first()
    if row is None:
        row = SessionAssessment(session_id=s.id, rubric_scores={})
        db.add(row)
    row.rubric_scores = dict(body.scores)
    row.overall_mark = overall
    row.notes = (body.notes or "").strip() or None
    if overall is not None:
        s.lecturer_label = overall
        s.lecturer_notes = row.notes
    db.commit()
    db.refresh(row)
    return {"scores": row.rubric_scores, "overall_mark": row.overall_mark, "notes": row.notes,
            "updated_at": row.updated_at, "label_recorded": overall is not None}
