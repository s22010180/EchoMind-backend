"""
WebSocket endpoints.

/ws/signal/{room_id}?token=JWT     WebRTC signaling relay (2 authenticated peers max)
/ws/telemetry/{room_id}?token=JWT  Each participant streams THEIR OWN microphone

Telemetry wire protocol (client -> server)
  text   {"type":"hello","sample_rate":16000}                    optional, send first if not 16 kHz
  binary PCM16 little-endian mono audio frames (any size up to MAX_AUDIO_MESSAGE_BYTES; 100-250 ms typical)
  text   {"type":"ping"}

Telemetry wire protocol (server -> client)
  {"type":"ready", "role", "sample_rate", "audio_format", "consent"}
  {"type":"metrics", "t", <scores...>, "details":{...features}}   to the LECTURER only, ~1 Hz
  {"type":"transcript", "role", "start", "end", "text", ...}       to the LECTURER only
  {"type":"consent_granted"|"consent_withdrawn"|"session_ended"|"pong"|"hello_ack"}
Close codes: 4401 auth, 4403 not a participant, 4404 no such room, 4409 replaced by newer connection, 4410 session not active
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.config import settings
from app.services.interaction_engine import other
from app.services.runtime import (ServiceContext, ensure_ticker, handle_audio, load_ws_context,
                                  safe_close, safe_send, schedule_transcription, stop_ticker)

logger = logging.getLogger(__name__)
router = APIRouter()

SIGNAL_TYPES = {"offer", "answer", "candidate", "ice-candidate", "ready", "bye"}
MAX_SIGNAL_BYTES = 65_536


def _ctx(ws: WebSocket) -> ServiceContext:
    return ws.app.state.ctx


async def _reject(ws: WebSocket, code: int, reason: str) -> None:
    await ws.accept()                       # accept first so the browser can read the close code
    await ws.close(code=code, reason=reason)


# ------------------------------------------------------------------- signaling
@router.websocket("/ws/signal/{room_id}")
async def signaling(websocket: WebSocket, room_id: str, token: str = Query(default="")):
    ctx = _ctx(websocket)
    info, err = await asyncio.to_thread(load_ws_context, room_id, token)
    if err:
        return await _reject(websocket, *err)
    await websocket.accept()

    room = ctx.get_or_create(room_id, info["session_id"], info["consent"])
    role, peer_role = info["role"], other(info["role"])

    old = room.signaling.get(role)
    room.signaling[role] = websocket
    if old is not None and old is not websocket:
        await safe_close(old, 4409, "Replaced by a newer connection")

    await safe_send(websocket, {"type": "joined", "role": role, "peers": [r for r in room.signaling if r != role]})
    await safe_send(room.signaling.get(peer_role), {"type": "peer-joined", "role": role})

    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            text = msg.get("text")
            if text is None or len(text) > MAX_SIGNAL_BYTES:
                continue
            try:
                payload = json.loads(text)
            except ValueError:
                continue
            if not isinstance(payload, dict) or payload.get("type") not in SIGNAL_TYPES:
                continue
            payload["from"] = role
            await safe_send(room.signaling.get(peer_role), payload)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Signaling error in room %s", room_id)
    finally:
        if room.signaling.get(role) is websocket:
            room.signaling.pop(role, None)
            await safe_send(room.signaling.get(peer_role), {"type": "peer-left", "role": role})
        room.last_activity = time.monotonic()


# ------------------------------------------------------------------- telemetry
@router.websocket("/ws/telemetry/{room_id}")
async def telemetry(websocket: WebSocket, room_id: str, token: str = Query(default="")):
    ctx = _ctx(websocket)
    info, err = await asyncio.to_thread(load_ws_context, room_id, token)
    if err:
        return await _reject(websocket, *err)
    await websocket.accept()

    room = ctx.get_or_create(room_id, info["session_id"], info["consent"])
    role = info["role"]

    old = room.telemetry.get(role)
    room.telemetry[role] = websocket
    if old is not None and old is not websocket:
        await safe_close(old, 4409, "Replaced by a newer connection")
    # Fresh stream on every (re)connect so both participants' clocks stay aligned.
    schedule_transcription(ctx, room, role, room.engine.reset_stream(role))
    ensure_ticker(ctx, room)

    await safe_send(websocket, {"type": "ready", "role": role, "sample_rate": 16000,
                                "audio_format": "pcm_s16le_mono", "consent": room.consent})
    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break

            data = msg.get("bytes")
            if data is not None:
                if 0 < len(data) <= settings.MAX_AUDIO_MESSAGE_BYTES:
                    handle_audio(ctx, room, role, data)
                continue

            text = msg.get("text")
            if text is None:
                continue
            try:
                m = json.loads(text)
            except ValueError:
                continue
            if not isinstance(m, dict):
                continue

            kind = m.get("type")
            if kind == "hello":
                sr = m.get("sample_rate")
                ok = isinstance(sr, int) and 8000 <= sr <= 96000 and room.engine.configure_stream(role, sr)
                await safe_send(websocket, {"type": "hello_ack", "sample_rate": sr, "accepted": bool(ok)})
            elif kind == "ping":
                await safe_send(websocket, {"type": "pong", "t": time.time()})
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Telemetry error in room %s (%s)", room_id, role)
    finally:
        if room.telemetry.get(role) is websocket:
            room.telemetry.pop(role, None)
            schedule_transcription(ctx, room, role, room.engine.reset_stream(role))
        if not room.telemetry:
            await stop_ticker(room)
        room.last_activity = time.monotonic()
