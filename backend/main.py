"""Voice assistant backend with Wav2Lip avatar.

AssemblyAI STT -> GPT -> Cartesia TTS -> Wav2Lip -> WebRTC
"""

import logging
from contextlib import asynccontextmanager

log = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.avatar import load_avatar, load_model, warm_up
from app.pipeline import generate_summary
from app.rtc_handler import router as rtc_router
from app.ws_handler import ws_endpoint

# History survives session cleanup — keyed by session_id
_session_histories: dict[str, list[dict]] = {}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    history = _session_histories.get(session_id)
    if not history or len(history) <= 1:
        raise HTTPException(status_code=404, detail="No conversation history for this session")
    summary = await generate_summary(history)
    # Clean up after generating
    _session_histories.pop(session_id, None)
    return {"summary": summary}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=True, host="0.0.0.0", port=config.PORT)
