"""Shared dependencies and service clients."""

import logging

from openai import AsyncOpenAI

from .config import OPENAI_KEY

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("voice")

openai_client = AsyncOpenAI(api_key=OPENAI_KEY)
