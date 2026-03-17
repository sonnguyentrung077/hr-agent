"""Configuration: environment variables and constants."""

import os
from urllib.parse import urlencode

from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ────────────────────────────────────────────────────────────────

OPENAI_KEY = os.getenv("OPENAI_KEY")
ASSEMBLY_KEY = os.getenv("ASSEMBLY_KEY")
CARTESIA_KEY = os.getenv("CARTESIA_KEY")

# ─── Models ──────────────────────────────────────────────────────────────────

MODEL = os.getenv("MODEL", "gpt-4.1-nano")
CARTESIA_MODEL = "sonic-3"
CARTESIA_VOICE_ID = "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"

# ─── Audio ───────────────────────────────────────────────────────────────────

SAMPLE_RATE_IN = 16000
SAMPLE_RATE_TTS = 44100
TTS_LANGUAGE = "sk"

# ─── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Ste užitočný hlasový asistent. Odpovede udržiavajte stručné a konverzačné."
)

SENTENCE_ENDS = frozenset(".!?。！？")

# ─── AssemblyAI ──────────────────────────────────────────────────────────────

AAI_PARAMS = {
    "sample_rate": SAMPLE_RATE_IN,
    "speech_model": "whisper-rt",
    "language_detection": True,
}
AAI_URL = f"wss://streaming.assemblyai.com/v3/ws?{urlencode(AAI_PARAMS)}"
