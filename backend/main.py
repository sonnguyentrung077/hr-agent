"""Voice assistant backend with Wav2Lip avatar.

Local Whisper STT -> GPT -> Cartesia TTS -> Wav2Lip -> WebRTC
"""

import logging
from contextlib import asynccontextmanager

log = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.avatar import load_avatar, load_model, warm_up
from app.stt import WhisperBackend
from app.dependencies import session_histories
from app.pipeline import generate_summary
from app.rtc_handler import router as rtc_router
from app.ws_handler import ws_endpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load Whisper STT model (streaming via LocalAgreement)
    stt = WhisperBackend(
        config.WHISPER_MODEL, config.WHISPER_DEVICE,
        config.WHISPER_COMPUTE_TYPE, config.WHISPER_LANGUAGE,
    )
    stt.warmup()
    app.state.stt = stt

    # Startup: load Wav2Lip model + avatar data + warm up GPU
    model = load_model(config.WAV2LIP_MODEL_PATH)
    warm_up(config.BATCH_SIZE, model, 256)
    avatar_data = load_avatar(config.AVATAR_PATH)

    app.state.wav2lip_model = model
    app.state.avatar_data = avatar_data
    app.state.batch_size = config.BATCH_SIZE

    yield
    # Shutdown: close all WebRTC sessions
    from app.rtc_handler import sessions

    for sid, session in list(sessions.items()):
        log.info("[SHUTDOWN] Closing session %s", sid)
        await session.pc.close()
    sessions.clear()
    session_histories.clear()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rtc_router)
app.websocket("/ws")(ws_endpoint)


@app.get("/summary/{session_id}")
async def get_summary(session_id: str):
    """Generate an interview summary from the session's conversation history."""
    log.info("[SUMMARY] Request for session %s (available: %s)", session_id, list(session_histories.keys()))
    history = session_histories.get(session_id)
    log.info("[HISTORY CONTENT] %s", history)
    if not history or len(history) <= 1:
        raise HTTPException(status_code=404, detail="No conversation history for this session")
    log.info("[SUMMARY] Generating summary from %d messages...", len(history))
    result = await generate_summary(history)
    session_histories.pop(session_id, None)
    log.info("[SUMMARY] Done")
    return result

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=False, host="0.0.0.0", port=config.PORT)
