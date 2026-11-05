from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import models  # noqa: F401  (registers tables on Base.metadata)
from app.config import settings
from app.database import Base, engine
from app.routers import auth, realtime, rtc, sessions, users
from app.services.asr_service import ASRService
from app.services.learned_scorer import LearnedScorer
from app.services.runtime import ServiceContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

    asr = None
    if settings.ENABLE_ASR:
        try:
            asr = await asyncio.to_thread(
                ASRService, settings.ASR_MODEL, settings.ASR_ENGINE, settings.ASR_COMPUTE_TYPE,
                settings.ASR_BEAM_SIZE, settings.ASR_DEVICE)
        except Exception:
            logger.exception("ASR unavailable - continuing without transcripts / speech-rate")

    app.state.ctx = ServiceContext(asr=asr, scorer=LearnedScorer(settings.LEARNED_SCORER_PATH))
    logger.info("Startup complete (asr=%s [%s], learned_scorer=%s)",
                asr is not None, getattr(asr, "engine", None), app.state.ctx.scorer.available)
    yield
    for room in list(app.state.ctx.rooms.values()):
        if room.ticker:
            room.ticker.cancel()


app = FastAPI(
    title=settings.APP_NAME,
    version="2.0.0",
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(rtc.router)
app.include_router(sessions.router)
app.include_router(realtime.router)


@app.get("/api/health", tags=["system"])
def health():
    ctx = getattr(app.state, "ctx", None)
    return {
        "status": "ok",
        "asr": bool(ctx and ctx.asr),
        "asr_engine": getattr(ctx.asr, "engine", None) if ctx else None,
        "learned_scorer": bool(ctx and ctx.scorer.available),
        "active_rooms": len(ctx.rooms) if ctx else 0,
    }


if __name__ == "__main__":
    import uvicorn

    # Single worker on purpose: room state lives in process memory.
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=not settings.is_production)
