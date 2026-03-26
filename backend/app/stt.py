"""Local STT using whisper_streaming (LocalAgreement) for true streaming."""

import asyncio
import logging
import math
import re
import threading
import time

import numpy as np
from whisper_streaming.backend.faster_whisper_backend import (
    FasterWhisperASR,
    FasterWhisperFeatureExtractorConfig,
    FasterWhisperModelConfig,
    FasterWhisperTranscribeConfig,
)
from whisper_streaming.base import Word
from whisper_streaming.processor import TranscriptionBuffer

from . import config

log = logging.getLogger(__name__)

# RMS energy threshold for silence detection (PCM16 range)
SILENCE_RMS_THRESHOLD = 500
# Minimum audio length (seconds) before attempting transcription
MIN_AUDIO_SECONDS = 0.2

# Whisper hallucination patterns (common outputs on silence/noise)
_HALLUCINATION_RE = re.compile(
    r"^[\s\.\,\!\?]*$"  # whitespace / punctuation only
    r"|[\u2000-\u206f]"  # general punctuation block
    r"|[\u3000-\u9fff]"  # CJK characters (Chinese/Japanese/Korean)
    r"|[\uac00-\ud7af]"  # Korean Hangul
    r"|(?i)^(thanks?(\s+you)?(\s+for\s+watching)?|bye(\s+bye)*|good\s*bye"
    r"|subscribe|amen|subs|obrigad[oa]|gracias|yeah"
    r"|продолжение следует|ďakujem za pozornosť"
    r"|cảm ơn.*đã xem|xin chào|hẹn gặp lại|tạm biệt"
    r"|cám ơn|phụ đề|đăng ký)[\.\!\s]*$"
)

# Detect repeated phrases like "Good bye Good bye", "thank you thank you thank you"
_REPEATED_RE = re.compile(r"(?i)^(.{2,}?)(\s+\1){1,}\s*[.!?]*$")


def _is_hallucination(text: str) -> bool:
    """Return True if text looks like a known Whisper hallucination."""
    t = text.strip()
    if not t:
        return True
    if _HALLUCINATION_RE.search(t):
        log.debug("[STT] Filtered hallucination: %r", t)
        return True
    if _REPEATED_RE.match(t):
        log.debug("[STT] Filtered repeated hallucination: %r", t)
        return True
    return False


class WhisperBackend:
    """Singleton: loads the Whisper model once, shared across all sessions."""

    def __init__(
        self,
        model_id: str = config.WHISPER_MODEL,
        device: str = config.WHISPER_DEVICE,
        compute_type: str = config.WHISPER_COMPUTE_TYPE,
        language: str | None = config.WHISPER_LANGUAGE,
    ):
        log.info("[STT] Loading model %s on %s (%s)...", model_id, device, compute_type)
        model_config = FasterWhisperModelConfig(
            model_size_or_path=model_id,
            device=device,
            compute_type=compute_type,
        )
        transcribe_config = FasterWhisperTranscribeConfig(
            beam_size=5,
            vad_filter=True,
            vad_parameters={"threshold": 0.8}
        )
        feature_extractor_config = FasterWhisperFeatureExtractorConfig()

        self.asr = FasterWhisperASR(
            model_config=model_config,
            transcribe_config=transcribe_config,
            feature_extractor_config=feature_extractor_config,
            sample_rate=config.SAMPLE_RATE_IN,
            language=language,
        )
        self._lock = threading.Lock()
        log.info("[STT] Model loaded")

    def warmup(self):
        """Run a dummy transcription to pre-compile CUDA kernels."""
        log.info("[STT] Warming up...")
        dummy = np.zeros(config.SAMPLE_RATE_IN, dtype=np.float32)
        with self._lock:
            self.asr.transcribe(dummy, init_prompt="")
        log.info("[STT] Warm-up done")


class StreamingSession:
    """Per-connection streaming STT session with LocalAgreement + energy-based turn detection."""

    def __init__(self, backend: WhisperBackend, on_partial, on_final):
        self._backend = backend
        self._on_partial = on_partial
        self._on_final = on_final

        # Audio buffer (float32, 16kHz)
        self._audio_buffer = np.array([], dtype=np.float32)
        self._buffer_offset = 0.0  # seconds trimmed from start

        # LocalAgreement transcription state
        self._tbuf = TranscriptionBuffer(
            TranscriptionBuffer.Config(max_n_gramms_to_commit=5)
        )
        self._committed_text = ""

        # Silence / turn detection
        self._speech_active = False
        self._silence_samples = 0
        self._silence_threshold_samples = int(
            config.MIN_TURN_SILENCE_MS * config.SAMPLE_RATE_IN / 1000
        )

        # Timing
        self._last_process_time = 0.0
        self._process_interval_s = config.WHISPER_AUDIO_CHUNK_SEC
        self._last_audio_time = 0.0  # monotonic time of last feed_audio call
        self._stale_timeout_s = 3.0  # reset if no audio for this long during speech

        self._queue: asyncio.Queue = asyncio.Queue()
        self._closed = False

    async def feed_audio(self, chunk: bytes):
        """Accept a PCM16 chunk from the client."""
        samples_i16 = np.frombuffer(chunk, dtype=np.int16)
        samples_f32 = samples_i16.astype(np.float32) / 32768.0
        rms = math.sqrt(np.mean(samples_i16.astype(np.float64) ** 2)) if len(samples_i16) else 0

        self._last_audio_time = time.monotonic()

        # Always accumulate audio
        self._audio_buffer = np.append(self._audio_buffer, samples_f32)

        if rms > SILENCE_RMS_THRESHOLD:
            self._speech_active = True
            self._silence_samples = 0

            now = time.monotonic()
            if (now - self._last_process_time) >= self._process_interval_s:
                audio_len = len(self._audio_buffer) / config.SAMPLE_RATE_IN
                if audio_len >= MIN_AUDIO_SECONDS:
                    self._queue.put_nowait("process")
                    self._last_process_time = now
        else:
            if self._speech_active:
                self._silence_samples += len(samples_i16)
                if self._silence_samples >= self._silence_threshold_samples:
                    self._queue.put_nowait("finalize")
                    self._speech_active = False
                    self._silence_samples = 0

    async def run(self):
        """Consumer loop — drains queue, runs LocalAgreement inference in thread."""
        while not self._closed:
            try:
                job_type = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                # If speech was active but no audio arrived (muted/disconnected), reset
                if (
                    self._speech_active
                    and self._last_audio_time > 0
                    and (time.monotonic() - self._last_audio_time) > self._stale_timeout_s
                ):
                    log.info("[STT] No audio for %.1fs during speech — resetting (mute?)",
                             time.monotonic() - self._last_audio_time)
                    self._speech_active = False
                    self._silence_samples = 0
                    # Finalize whatever we had
                    final_text = self._committed_text.strip()
                    self._reset()
                    if final_text and not _is_hallucination(final_text):
                        await self._on_final(final_text)
                continue

            # Skip stale process jobs if a finalize is queued
            if job_type == "process" and not self._queue.empty():
                continue

            try:
                committed, limbo_text = await asyncio.to_thread(self._process_audio)
            except Exception as e:
                log.error("[STT] Transcription error: %s", e, exc_info=True)
                continue

            if job_type == "finalize":
                # End of turn: flush committed + in-limbo as final
                final_text = (self._committed_text + limbo_text).strip()
                self._reset()
                if final_text and not _is_hallucination(final_text):
                    await self._on_final(final_text)
            else:
                # Partial: show committed + in-limbo as preview
                partial_text = (self._committed_text + limbo_text).strip()
                if partial_text and not _is_hallucination(partial_text):
                    await self._on_partial(partial_text)

    def _process_audio(self) -> tuple[str, str]:
        """Synchronous LocalAgreement pass — called via asyncio.to_thread()."""
        # Build prompt from previously committed words
        prompt_candidates = self._tbuf.prompt_candidates
        prompt = "".join(w.word for w in prompt_candidates)[:config.WHISPER_PROMPT_SIZE]

        with self._backend._lock:
            segments, _lang = self._backend.asr.transcribe(
                self._audio_buffer, init_prompt=prompt
            )
            words = self._backend.asr.segments_to_words(segments)

        # Apply buffer offset to word timestamps
        offset_words = [w.with_offset(self._buffer_offset) for w in words]

        # LocalAgreement: insert and get newly committed words
        newly_committed = self._tbuf.insert(offset_words)

        if newly_committed:
            committed_text = Word.join(newly_committed).word
            self._committed_text += committed_text
        else:
            committed_text = ""

        # Trim audio buffer if too long
        audio_len_s = len(self._audio_buffer) / config.SAMPLE_RATE_IN
        max_s = config.WHISPER_BUFFER_TRIMMING_SEC
        if audio_len_s > max_s:
            trim_s = int(audio_len_s - max_s)
            trim_samples = trim_s * config.SAMPLE_RATE_IN
            self._audio_buffer = self._audio_buffer[trim_samples:]
            self._buffer_offset += trim_s
            self._tbuf.trim(self._buffer_offset)

        # Get in-limbo (not yet committed) text
        limbo = self._tbuf.transcriptions_in_limbo
        limbo_text = "".join(w.word for w in limbo) if limbo else ""

        return committed_text, limbo_text

    def _reset(self):
        """Reset state for next utterance."""
        self._audio_buffer = np.array([], dtype=np.float32)
        self._buffer_offset = 0.0
        self._tbuf = TranscriptionBuffer(
            TranscriptionBuffer.Config(max_n_gramms_to_commit=5)
        )
        self._committed_text = ""
        self._last_process_time = 0.0

    def close(self):
        self._closed = True
