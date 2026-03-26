"""Local STT using faster-whisper with built-in VAD."""

import asyncio
import logging
import math
import time

import numpy as np
from faster_whisper import WhisperModel

from . import config

log = logging.getLogger(__name__)

# RMS energy threshold for silence detection (PCM16 range)
SILENCE_RMS_THRESHOLD = 300
# Minimum audio length (seconds) before attempting transcription
MIN_AUDIO_SECONDS = 0.2


class WhisperSTT:
    """Singleton wrapper around faster-whisper model, loaded once at startup."""

    def __init__(
        self,
        model_id: str = config.WHISPER_MODEL,
        device: str = config.WHISPER_DEVICE,
        compute_type: str = config.WHISPER_COMPUTE_TYPE,
    ):
        log.info("[STT] Loading model %s on %s (%s)...", model_id, device, compute_type)
        self.model = WhisperModel(model_id, device=device, compute_type=compute_type)
        log.info("[STT] Model loaded")

    def transcribe(self, audio_np: np.ndarray, language: str | None = None) -> str:
        """Synchronous transcription — call via asyncio.to_thread()."""
        segments, info = self.model.transcribe(
            audio_np,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=config.MIN_TURN_SILENCE_MS,
                threshold=0.5,
                speech_pad_ms=400,
            ),
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text

    def warmup(self):
        """Run a dummy transcription to pre-compile CUDA kernels."""
        log.info("[STT] Warming up...")
        dummy = np.zeros(config.SAMPLE_RATE_IN, dtype=np.float32)
        self.transcribe(dummy)
        log.info("[STT] Warm-up done")


class StreamingSession:
    """Per-connection streaming STT session with energy-based turn detection."""

    def __init__(self, stt: WhisperSTT, on_partial, on_final):
        self.stt = stt
        self._on_partial = on_partial
        self._on_final = on_final

        self._buf = bytearray()
        self._speech_active = False
        self._silence_samples = 0
        self._last_partial_time = 0.0
        self._queue: asyncio.Queue = asyncio.Queue()
        self._closed = False

        self._silence_threshold_samples = int(
            config.MIN_TURN_SILENCE_MS * config.SAMPLE_RATE_IN / 1000
        )
        self._partial_interval_s = config.PARTIAL_INTERVAL_MS / 1000

    async def feed_audio(self, chunk: bytes):
        """Accept a PCM16 chunk from the client."""
        self._buf.extend(chunk)

        # Energy-based silence detection
        samples = np.frombuffer(chunk, dtype=np.int16)
        rms = math.sqrt(np.mean(samples.astype(np.float64) ** 2)) if len(samples) else 0

        if rms > SILENCE_RMS_THRESHOLD:
            # Speech detected
            self._speech_active = True
            self._silence_samples = 0

            # Periodic partial transcription
            now = time.monotonic()
            if (now - self._last_partial_time) >= self._partial_interval_s:
                audio_len = len(self._buf) / 2  # PCM16 = 2 bytes per sample
                if audio_len / config.SAMPLE_RATE_IN >= MIN_AUDIO_SECONDS:
                    self._queue.put_nowait(("partial", bytes(self._buf)))
                    self._last_partial_time = now
        else:
            # Silence
            if self._speech_active:
                self._silence_samples += len(samples)
                if self._silence_samples >= self._silence_threshold_samples:
                    # End of turn — queue final transcription
                    audio_len = len(self._buf) / 2
                    if audio_len / config.SAMPLE_RATE_IN >= MIN_AUDIO_SECONDS:
                        self._queue.put_nowait(("final", bytes(self._buf)))
                    self._buf.clear()
                    self._speech_active = False
                    self._silence_samples = 0
                    self._last_partial_time = 0.0

    async def run(self):
        """Consumer loop — drains transcription queue, runs inference in thread."""
        language = config.WHISPER_LANGUAGE
        while not self._closed:
            try:
                job_type, audio_bytes = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            # Skip stale partials if a final is already queued
            if job_type == "partial" and not self._queue.empty():
                continue

            audio_np = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )

            try:
                text = await asyncio.to_thread(self.stt.transcribe, audio_np, language)
            except Exception as e:
                log.error("[STT] Transcription error: %s", e, exc_info=True)
                continue

            if not text:
                continue

            if job_type == "final":
                await self._on_final(text)
            else:
                await self._on_partial(text)

    def close(self):
        self._closed = True
