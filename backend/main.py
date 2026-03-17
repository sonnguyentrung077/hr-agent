"""Voice assistant backend: AssemblyAI STT -> GPT -> Cartesia TTS."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ws_handler import ws_endpoint

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.websocket("/ws")(ws_endpoint)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, reload=True, host="0.0.0.0", port=8000)
