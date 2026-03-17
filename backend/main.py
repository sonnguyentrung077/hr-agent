"""Voice assistant backend with Wav2Lip avatar.

AssemblyAI STT -> GPT -> Cartesia TTS -> Wav2Lip -> WebRTC
"""

import logging
from contextlib import asynccontextmanager

log = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.avatar import load_avatar, load_model, warm_up
from app.rtc_handler import router as rtc_router
from app.ws_handler import ws_endpoint

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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rtc_router)
app.websocket("/ws")(ws_endpoint)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=True, host="0.0.0.0", port=8001)
