from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_SECRET = "change-this-secret-before-deploying"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "SE Viva Chemistry Analyzer API"
    ENVIRONMENT: str = "development"          # development | production

    # --- database / auth
    DATABASE_URL: str = "sqlite:///./viva_app.db"
    SECRET_KEY: str = _DEFAULT_SECRET
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120
    LECTURER_INVITE_CODE: str = ""            # if set, required to register as a lecturer
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- WebRTC ICE servers handed to the frontend
    STUN_URLS: str = "stun:stun.l.google.com:19302"
    TURN_URL: str = ""
    TURN_USERNAME: str = ""
    TURN_CREDENTIAL: str = ""

    # --- AI models
    ENABLE_ASR: bool = True
    ASR_MODEL: str = "small.en"                # small.en = recommended; base.en = faster; medium.en = most accurate
    ASR_ENGINE: str = "auto"                   # auto | faster-whisper | transformers
    ASR_DEVICE: str = "cpu"                    # cpu | cuda
    ASR_COMPUTE_TYPE: str = "int8"             # int8 (CPU) | float16 (GPU)
    ASR_BEAM_SIZE: int = 5                     # higher = a little more accurate, slower
    ASR_TRANSCRIBE_EXAMINER: bool = True
    LEARNED_SCORER_PATH: str = "models/scorer.joblib"

    # --- realtime behaviour
    METRICS_INTERVAL_S: float = 1.0
    TELEMETRY_PERSIST_INTERVAL_S: float = 5.0
    MAX_AUDIO_MESSAGE_BYTES: int = 262_144
    MAX_PENDING_ASR_TASKS: int = 12

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def stun_url_list(self) -> List[str]:
        return [u.strip() for u in self.STUN_URLS.split(",") if u.strip()]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @model_validator(mode="after")
    def _check_secret(self):
        if self.is_production and (self.SECRET_KEY == _DEFAULT_SECRET or len(self.SECRET_KEY) < 32):
            raise ValueError("In production SECRET_KEY must be set to a random string of 32+ characters.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
