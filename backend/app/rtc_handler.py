"""WebRTC signaling endpoint and session management."""

import asyncio
import logging
from dataclasses import dataclass

from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.rtcrtpsender import RTCRtpSender
from fastapi import APIRouter, Request

from .avatar import AvatarEngine
from .webrtc_tracks import HumanPlayer

log = logging.getLogger(__name__)

router = APIRouter()

# STUN servers — fix for Firefox mDNS obfuscation.
# Without these, only "host" ICE candidates exist. Firefox replaces host IPs
# with mDNS addresses the server can't resolve. STUN adds "srflx" candidates
# with real IPs, so connections work without touching about:config.
ICE_SERVERS = [
    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
]


@dataclass
class Session:
    engine: AvatarEngine
    pc: RTCPeerConnection
    player: HumanPlayer
    closed: asyncio.Event


sessions: dict[str, Session] = {}

_session_counter = 0


def _next_session_id() -> str:
    global _session_counter
    _session_counter += 1
    return str(_session_counter)


@router.post("/offer")
async def offer(request: Request):
    """WebRTC SDP offer/answer exchange. Creates an avatar session."""
    params = await request.json()
    remote_offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(RTCConfiguration(iceServers=ICE_SERVERS))

    # Get model + avatar data loaded at startup
    app = request.app
    model = app.state.wav2lip_model
    avatar_data = app.state.avatar_data
    batch_size = app.state.batch_size

    session_id = _next_session_id()

    engine = AvatarEngine(model, avatar_data, batch_size=batch_size)
    player = HumanPlayer(engine)

    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    # Prefer H264 for efficiency, then VP8
    caps = RTCRtpSender.getCapabilities("video")
    prefs = [c for c in caps.codecs if c.name == "H264"]
    prefs += [c for c in caps.codecs if c.name == "VP8"]
    prefs += [c for c in caps.codecs if c.name == "rtx"]
    transceiver = pc.getTransceivers()[1]  # video transceiver
    transceiver.setCodecPreferences(prefs)

    _disconnect_task: asyncio.Task | None = None

    @pc.on("connectionstatechange")
    async def on_state():
        nonlocal _disconnect_task
        log.info("[RTC] session=%s state=%s", session_id, pc.connectionState)

        if pc.connectionState == "disconnected":
            # Grace period — let ICE recover on slow networks (3G etc.)
            async def _grace():
                log.info("[RTC] session=%s disconnected — waiting 15s for recovery", session_id)
                await asyncio.sleep(15)
                if pc.connectionState in ("disconnected", "failed"):
                    log.info("[RTC] session=%s did not recover — closing", session_id)
                    session = sessions.pop(session_id, None)
                    if session:
                        session.closed.set()
                        await session.pc.close()

            if _disconnect_task is None or _disconnect_task.done():
                _disconnect_task = asyncio.create_task(_grace())
            return

        if pc.connectionState == "connected":
            # Recovered — cancel grace timer
            if _disconnect_task and not _disconnect_task.done():
                log.info("[RTC] session=%s reconnected — cancelling grace timer", session_id)
                _disconnect_task.cancel()
                _disconnect_task = None
            return

        if pc.connectionState in ("failed", "closed"):
            if _disconnect_task and not _disconnect_task.done():
                _disconnect_task.cancel()
            session = sessions.pop(session_id, None)
            if session:
                log.info("[RTC] Cleaning up session %s", session_id)
                session.closed.set()
                await session.pc.close()

    sessions[session_id] = Session(engine=engine, pc=pc, player=player, closed=asyncio.Event())

    await pc.setRemoteDescription(remote_offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    log.info("[RTC] Session %s created", session_id)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "session_id": session_id,
    }
