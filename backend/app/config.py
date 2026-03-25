"""Configuration: environment variables and constants."""

import os

from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ────────────────────────────────────────────────────────────────

OPENAI_KEY = os.getenv("OPENAI_KEY")
CARTESIA_KEY = os.getenv("CARTESIA_KEY")

# ─── Models ──────────────────────────────────────────────────────────────────

MODEL = os.getenv("MODEL", "gpt-4.1-nano")
CARTESIA_MODEL = "sonic-3"
CARTESIA_VOICE_ID = "ca590fdc-df56-4d2e-94a4-ef5b423c7ddf"

# ─── Audio ───────────────────────────────────────────────────────────────────

SAMPLE_RATE_IN = 16000
SAMPLE_RATE_TTS = 44100
SAMPLE_RATE_AVATAR = 16000
TTS_LANGUAGE = "sk"

# ─── Avatar / Wav2Lip ────────────────────────────────────────────────────────

WAV2LIP_MODEL_PATH = os.getenv("WAV2LIP_MODEL_PATH", "models/wav2lip.pth")
AVATAR_PATH = os.getenv("AVATAR_PATH", "data/avatars/wav2lip256_avatar1")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))

# ─── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Si užitočný hlasový asistent. "
    "Odpovedaj stručne (1-3 vety). "
    "Hovor prirodzene — nepoužívaj markdown, odrážky ani špeciálne formátovanie."
)

SENTENCE_ENDS = frozenset(".!?。！？")

# ─── CORS ────────────────────────────────────────────────────────────────────

PORT = int(os.getenv("PORT", "8001"))

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "*").split(",")
    if o.strip()
]

# ─── Local Whisper STT ────────────────────────────────────────────────────────

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", None)  # None = auto-detect

# Turn detection
MIN_TURN_SILENCE_MS = int(os.getenv("MIN_TURN_SILENCE_MS", "800"))
PARTIAL_INTERVAL_MS = int(os.getenv("PARTIAL_INTERVAL_MS", "500"))
