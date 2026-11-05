"""
Realtime runtime: in-memory rooms, the 1 Hz metrics ticker, transcription dispatch,
WebSocket authorization and session finalization.

NOTE: rooms live in process memory, so run ONE worker process
(`uvicorn app.main:app`, not `--workers N`). To scale horizontally, move room state
to Redis and pin each room to a worker.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

from fastapi import WebSocket

from app.config import settings
from app.database import session_scope
from app.models import SessionStatus, User, VivaSession
from app.security import decode_token
from app.services import persistence
from app.services.asr_service import ASRService
from app.services.audio import AudioChunk
from app.services.insights import DISCLAIMER, generate_insights
from app.services.interaction_engine import VivaSessionEngine
from app.services.performance import compute_performance, qa_features
from app.services.qa_analysis import analyze_transcript
from app.services.learned_scorer import LearnedScorer

logger = logging.getLogger(__name__)


@dataclass
class Room:
    room_id: str
    session_id: int
    consent: bool = False
    engine: VivaSessionEngine = field(default_factory=VivaSessionEngine)
    telemetry: Dict[str, WebSocket] = field(default_factory=dict)     # role -> socket
    signaling: Dict[str, WebSocket] = field(default_factory=dict)
    ticker: Optional[asyncio.Task] = None
    bg_tasks: Set[asyncio.Task] = field(default_factory=set)
    started_marked: bool = False
    closing: bool = False
    last_activity: float = field(default_factory=time.monotonic)


class ServiceContext:
    def __init__(self, asr: Optional[ASRService], scorer: LearnedScorer):
        self.asr = asr
        self.scorer = scorer
        self.rooms: Dict[str, Room] = {}

    def get(self, room_id: str) -> Optional[Room]:
        return self.rooms.get(room_id)

    def get_or_create(self, room_id: str, session_id: int, consent: bool) -> Room:
        self._purge_idle()
        room = self.rooms.get(room_id)
        if room is None:
            room = Room(room_id=room_id, session_id=session_id, consent=consent)
            self.rooms[room_id] = room
        room.last_activity = time.monotonic()
        return room

    def _purge_idle(self, max_idle_s: float = 6 * 3600) -> None:
        now = time.monotonic()
        for rid in [r for r, rm in self.rooms.items()
                    if not rm.telemetry and not rm.signaling and now - rm.last_activity > max_idle_s]:
            self.rooms.pop(rid, None)


# --------------------------------------------------------------------- helpers
async def safe_send(ws: Optional[WebSocket], payload: dict) -> bool:
    if ws is None:
        return False
    try:
        await ws.send_json(payload)
        return True
    except Exception:
        return False


async def safe_close(ws: Optional[WebSocket], code: int = 1000, reason: str = "") -> None:
    if ws is None:
        return
    with contextlib.suppress(Exception):
        await ws.close(code=code, reason=reason)


def spawn(room: Room, coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    room.bg_tasks.add(task)
    task.add_done_callback(room.bg_tasks.discard)
    return task


# ------------------------------------------------------------- WS authorization
def load_ws_context(room_id: str, token: str) -> Tuple[Optional[dict], Optional[Tuple[int, str]]]:
    """Blocking. Returns (context, None) or (None, (close_code, reason))."""
    payload = decode_token(token or "")
    if not payload or "sub" not in payload:
        return None, (4401, "Invalid or expired token")
    try:
        user_id = int(payload["sub"])
    except (TypeError, ValueError):
        return None, (4401, "Invalid token")
    with session_scope() as db:
        user = db.get(User, user_id)
        if user is None or not user.is_active:
            return None, (4401, "Unknown user")
        sess = db.query(VivaSession).filter(VivaSession.room_id == room_id).first()
        if sess is None:
            return None, (4404, "Room not found")
        if user.id not in (sess.lecturer_id, sess.student_id):
            return None, (4403, "You are not a participant of this session")
        if sess.status != SessionStatus.ACTIVE:
            return None, (4410, "Session is not active")
        return {
            "user_id": user.id,
            "full_name": user.full_name,
            "role": "examiner" if user.id == sess.lecturer_id else "student",
            "session_id": sess.id,
            "consent": bool(sess.consent_student),
        }, None


# --------------------------------------------------------------- audio ingestion
def handle_audio(ctx: ServiceContext, room: Room, role: str, data: bytes) -> None:
    if not room.consent or room.closing:
        return
    first = room.engine.t0 is None
    chunks = room.engine.feed_audio(role, data, time.monotonic())
    if first and not room.started_marked:
        room.started_marked = True
        spawn(room, asyncio.to_thread(persistence.mark_started, room.session_id))
    schedule_transcription(ctx, room, role, chunks)


def schedule_transcription(ctx: ServiceContext, room: Room, role: str, chunks) -> None:
    if ctx.asr is None or not chunks:
        return
    if role == "examiner" and not settings.ASR_TRANSCRIBE_EXAMINER:
        return
    for chunk in chunks:
        if len(room.bg_tasks) >= settings.MAX_PENDING_ASR_TASKS:
            logger.warning("ASR backlog full in room %s; dropping a chunk", room.room_id)
            return
        spawn(room, _transcribe(ctx, room, role, chunk))


async def _transcribe(ctx: ServiceContext, room: Room, role: str, chunk: AudioChunk) -> None:
    try:
        text = await ctx.asr.transcribe(chunk.audio)
        if not text:
            return
        if not room.consent:                       # consent withdrawn while we were working
            return
        room.engine.add_transcript(role, chunk.t_start, chunk.t_end, text)
        item = room.engine.transcript[-1]
        await asyncio.to_thread(persistence.save_transcript, room.session_id, item)
        await safe_send(room.telemetry.get("examiner"), {"type": "transcript", **item})
    except Exception:
        logger.exception("Transcription pipeline failed in room %s", room.room_id)


# ------------------------------------------------------------------- the ticker
def ensure_ticker(ctx: ServiceContext, room: Room) -> None:
    if room.ticker is None or room.ticker.done():
        room.ticker = asyncio.create_task(_ticker(room))


async def stop_ticker(room: Room) -> None:
    task, room.ticker = room.ticker, None
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def _ticker(room: Room) -> None:
    """Every second: compute metrics, push to the lecturer, persist every N seconds."""
    while True:
        await asyncio.sleep(settings.METRICS_INTERVAL_S)
        eng = room.engine
        if eng.t0 is None or not room.consent or room.closing:
            continue
        try:
            now = time.monotonic()
            snap = eng.snapshot(now)
            metrics, t_off = snap["metrics"], max(now - eng.t0, 0.0)
            payload = {"type": "metrics", "t": round(t_off, 1), **metrics, "details": snap["features"]}
            await safe_send(room.telemetry.get("examiner"), payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Ticker error in room %s", room.room_id)


# ---------------------------------------------------------------- finalization
async def finalize_room(ctx: ServiceContext, session_id: int, room_id: str) -> dict:
    """Stop analysis, drain pending transcription, store the summary, close sockets."""
    room = ctx.rooms.get(room_id)
    if room is not None:
        room.closing = True
        await stop_ticker(room)
        for role, chunks in room.engine.flush().items():
            schedule_transcription(ctx, room, role, chunks)
        if room.bg_tasks:
            await asyncio.wait(list(room.bg_tasks), timeout=45)
        engine = room.engine
    else:
        engine = VivaSessionEngine()

    now = time.monotonic()
    snap = engine.snapshot(now)
    features, metrics = snap["features"], snap["metrics"]

    # Question-by-question analysis -> the five student performance indicators, stored with the session.
    qa = analyze_transcript([{"speaker": t["role"], "start_s": t["start"], "end_s": t["end"], "text": t["text"]}
                             for t in engine.transcript])
    performance = compute_performance(features, qa)
    features.update(qa_features(qa["summary"]))
    metrics["performance"] = performance["scores"]
    metrics["performance_ok"] = not performance["provisional"]

    insights = generate_insights(features, metrics)
    predicted = ctx.scorer.predict(features) if ctx.scorer is not None else None
    duration = float(features.get("elapsed_s") or 0.0)

    await asyncio.to_thread(persistence.complete_session, session_id, duration, features, metrics, insights, predicted)

    if room is not None:
        for ws in list(room.telemetry.values()) + list(room.signaling.values()):
            await safe_send(ws, {"type": "session_ended"})
            await safe_close(ws, 1000, "Session ended")
        ctx.rooms.pop(room_id, None)

    return {"duration_s": duration, "metrics": metrics, "insights": insights,
            "predicted_rating": predicted, "disclaimer": DISCLAIMER}
