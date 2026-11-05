from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import settings
from app.models import User
from app.security import get_current_user

router = APIRouter(prefix="/api/rtc", tags=["webrtc"])


@router.get("/config")
def ice_config(_: User = Depends(get_current_user)):
    """ICE servers for RTCPeerConnection. Add a TURN server for students behind strict NATs."""
    servers = [{"urls": settings.stun_url_list}] if settings.stun_url_list else []
    if settings.TURN_URL:
        servers.append({"urls": [settings.TURN_URL],
                        "username": settings.TURN_USERNAME,
                        "credential": settings.TURN_CREDENTIAL})
    return {"iceServers": servers}
