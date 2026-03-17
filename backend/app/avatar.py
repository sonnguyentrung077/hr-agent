"""Wav2Lip avatar engine.

Receives 16kHz PCM float32 audio chunks, runs mel spectrogram extraction,
Wav2Lip GPU inference, and pushes lip-synced video+audio frames to WebRTC tracks.

Extracted and cleaned up from LiveTalking: lipreal.py, lipasr.py, baseasr.py, basereal.py.
"""

import asyncio
import copy
import glob
import logging
import os
import pickle
import queue
import time
from queue import Queue
from threading import Event, Thread

import cv2
import numpy as np
import torch
from av import AudioFrame, VideoFrame

from wav2lip import audio as wav2lip_audio
from wav2lip.models import Wav2Lip

log = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Audio constants
SAMPLE_RATE = 16000
FPS = 50  # 50 audio frames per second (20ms each)
CHUNK_SIZE = SAMPLE_RATE // FPS  # 320 samples per chunk
MEL_STEP_SIZE = 16
MEL_IDX_MULTIPLIER = 80.0 * 2 / FPS  # 3.2


# ── Model loading ─────────────────────────────────────────────────────────────


def load_model(path: str) -> Wav2Lip:
    """Load Wav2Lip model from checkpoint file."""
    log.info("Loading Wav2Lip model from %s", path)
    if device == "cuda":
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    state = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model = Wav2Lip()
    model.load_state_dict(state)
    return model.to(device).eval()


def load_avatar(avatar_path: str) -> tuple[list, list, list]:
    """Load pre-extracted avatar frames, face crops, and coordinates.

    Returns (frame_list, face_list, coord_list).
    """
    log.info("Loading avatar from %s", avatar_path)
    coords_path = os.path.join(avatar_path, "coords.pkl")
    full_imgs_path = os.path.join(avatar_path, "full_imgs")
    face_imgs_path = os.path.join(avatar_path, "face_imgs")

    with open(coords_path, "rb") as f:
        coord_list = pickle.load(f)

    full_imgs = sorted(
        glob.glob(os.path.join(full_imgs_path, "*.[jpJP][pnPN]*[gG]")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
    )
    face_imgs = sorted(
        glob.glob(os.path.join(face_imgs_path, "*.[jpJP][pnPN]*[gG]")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
    )

    frame_list = [cv2.imread(p) for p in full_imgs]
    face_list = [cv2.imread(p) for p in face_imgs]

    log.info("Loaded %d frames, %d faces", len(frame_list), len(face_list))
    return frame_list, face_list, coord_list


@torch.no_grad()
def warm_up(batch_size: int, model: Wav2Lip, resolution: int = 256):
    """Run a dummy forward pass to warm up CUDA kernels."""
    log.info("Warming up Wav2Lip (batch=%d, res=%d)...", batch_size, resolution)
    img = torch.ones(batch_size, 6, resolution, resolution).to(device)
    mel = torch.ones(batch_size, 1, 80, 16).to(device)
    model(mel, img)
    log.info("Warm-up done")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mirror_index(size: int, index: int) -> int:
    """Ping-pong index for cycling through avatar frames."""
    turn = index // size
    res = index % size
    return res if turn % 2 == 0 else size - res - 1


# ── AvatarEngine ──────────────────────────────────────────────────────────────


class AvatarEngine:
    """Manages a single avatar session.

    Usage:
        engine = AvatarEngine(model, avatar_data, batch_size=4)
        # HumanPlayer calls engine.render(quit_event, loop, audio_track, video_track)
        # Pipeline calls engine.put_audio_chunk(pcm_chunk, event) to feed TTS audio
    """

    def __init__(
        self,
        model: Wav2Lip,
        avatar_data: tuple[list, list, list],
        batch_size: int = 4,
        stride_left: int = 10,
        stride_right: int = 10,
    ):
        self.model = model
        self.frame_list, self.face_list, self.coord_list = avatar_data
        self.batch_size = batch_size
        self.stride_left = stride_left
        self.stride_right = stride_right

        # Queues
        self._audio_in: Queue = Queue()  # (pcm_chunk, eventpoint)
        self._feat_queue: Queue = Queue(maxsize=2)  # mel batches
        self._audio_out: Queue = Queue()  # (frame, type, eventpoint)
        self.res_frame_queue: Queue = Queue(maxsize=batch_size * 2)

        # State
        self._frames: list[np.ndarray] = []  # accumulated audio frames for mel
        self.speaking = False
        self._msg_submit_time: float | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def put_audio_chunk(self, pcm_f32_16khz: np.ndarray, event: dict | None = None):
        """Feed a 320-sample float32 16kHz PCM chunk into the pipeline."""
        if event and event.get("status") == "start":
            self._msg_submit_time = time.time()
        self._audio_in.put((pcm_f32_16khz, event))

    def notify(self, eventpoint):
        """Called by WebRTC track when an eventpoint is delivered."""
        if eventpoint and self._msg_submit_time:
            elapsed = time.time() - self._msg_submit_time
            log.info(
                "[TIMING] notify %s | %.3fs since feed | text: '%s'",
                eventpoint.get("status", "?"),
                elapsed,
                str(eventpoint.get("text", ""))[:50],
            )

    def flush(self):
        """Interrupt current speech — clear all queues."""
        self._audio_in.queue.clear()
        self._feat_queue.queue.clear()
        self._audio_out.queue.clear()
        self.res_frame_queue.queue.clear()

    # ── Render (called by HumanPlayer worker thread) ──────────────────────

    def render(self, quit_event: Event, loop, audio_track, video_track):
        """Main entry point called by HumanPlayer. Starts sub-threads."""
        self._warm_up_asr()

        infer_quit = Event()
        infer_thread = Thread(
            target=self._inference_loop,
            args=(infer_quit,),
            daemon=True,
            name="wav2lip-inference",
        )
        infer_thread.start()

        process_quit = Event()
        process_thread = Thread(
            target=self._process_frames,
            args=(process_quit, loop, audio_track, video_track),
            daemon=True,
            name="frame-processor",
        )
        process_thread.start()

        # ASR step loop runs in this thread (the worker thread)
        while not quit_event.is_set():
            self._asr_run_step()
            if video_track and video_track._queue.qsize() >= 5:
                time.sleep(0.04 * video_track._queue.qsize() * 0.8)

        log.info("AvatarEngine render loop stopping")
        infer_quit.set()
        infer_thread.join(timeout=5)
        process_quit.set()
        process_thread.join(timeout=5)

    # ── ASR: audio feature extraction (mel spectrograms) ──────────────────

    def _get_audio_frame(self) -> tuple[np.ndarray, int, dict | None]:
        """Get next audio frame. Returns (pcm, type, eventpoint).
        type: 0=speech, 1=silence
        """
        try:
            frame, eventpoint = self._audio_in.get(block=True, timeout=0.01)
            return frame, 0, eventpoint
        except queue.Empty:
            return np.zeros(CHUNK_SIZE, dtype=np.float32), 1, None

    def _warm_up_asr(self):
        """Fill stride buffers with silence before first real audio."""
        for _ in range(self.stride_left + self.stride_right):
            frame, frame_type, eventpoint = self._get_audio_frame()
            self._frames.append(frame)
            self._audio_out.put((frame, frame_type, eventpoint))
        for _ in range(self.stride_left):
            self._audio_out.get()

    def _asr_run_step(self):
        """Process one batch: read audio, extract mel features, enqueue results."""
        for _ in range(self.batch_size * 2):
            frame, frame_type, eventpoint = self._get_audio_frame()
            self._frames.append(frame)
            self._audio_out.put((frame, frame_type, eventpoint))

        if len(self._frames) <= self.stride_left + self.stride_right:
            return

        inputs = np.concatenate(self._frames)
        mel = wav2lip_audio.melspectrogram(inputs)

        left = max(0, self.stride_left * 80 / FPS)
        mel_chunks = []
        i = 0
        usable = (len(self._frames) - self.stride_left - self.stride_right) / 2
        while i < usable:
            start_idx = int(left + i * MEL_IDX_MULTIPLIER)
            if start_idx + MEL_STEP_SIZE > mel.shape[1]:
                mel_chunks.append(mel[:, mel.shape[1] - MEL_STEP_SIZE :])
            else:
                mel_chunks.append(mel[:, start_idx : start_idx + MEL_STEP_SIZE])
            i += 1

        self._feat_queue.put(mel_chunks)
        self._frames = self._frames[-(self.stride_left + self.stride_right) :]

    # ── Inference loop (Wav2Lip GPU) ──────────────────────────────────────

    def _inference_loop(self, quit_event: Event):
        """Runs Wav2Lip model on mel batches, outputs predicted face crops."""
        length = len(self.face_list)
        index = 0
        log.info("Inference loop started (batch=%d, faces=%d)", self.batch_size, length)

        while not quit_event.is_set():
            try:
                mel_batch = self._feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            # Collect audio frames (2 per video frame)
            is_all_silence = True
            audio_frames = []
            for _ in range(self.batch_size * 2):
                frame, frame_type, eventpoint = self._audio_out.get()
                audio_frames.append((frame, frame_type, eventpoint))
                if frame_type == 0:
                    is_all_silence = False

            if is_all_silence:
                for i in range(self.batch_size):
                    self.res_frame_queue.put(
                        (None, _mirror_index(length, index), audio_frames[i * 2 : i * 2 + 2])
                    )
                    index += 1
                continue

            # Build image batch
            img_batch = []
            for i in range(self.batch_size):
                idx = _mirror_index(length, index + i)
                img_batch.append(self.face_list[idx])
            img_batch = np.asarray(img_batch)
            mel_batch = np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_batch.shape[1] // 2 :] = 0

            img_input = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_input = mel_batch.reshape(len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1)

            img_tensor = torch.FloatTensor(np.transpose(img_input, (0, 3, 1, 2))).to(device)
            mel_tensor = torch.FloatTensor(np.transpose(mel_input, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                pred = self.model(mel_tensor, img_tensor)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            for i, res_frame in enumerate(pred):
                self.res_frame_queue.put(
                    (res_frame, _mirror_index(length, index), audio_frames[i * 2 : i * 2 + 2])
                )
                index += 1

        log.info("Inference loop stopped")

    # ── Frame processing (composite + push to WebRTC) ─────────────────────

    def _paste_back(self, pred_frame: np.ndarray, idx: int) -> np.ndarray:
        """Paste predicted face crop back onto the full frame."""
        y1, y2, x1, x2 = self.coord_list[idx]
        frame = copy.deepcopy(self.frame_list[idx])
        resized = cv2.resize(pred_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        frame[y1:y2, x1:x2] = resized
        return frame

    def _process_frames(self, quit_event: Event, loop, audio_track, video_track):
        """Read inference results, composite frames, push to WebRTC tracks.

        Each inference result yields 1 video frame + 2 audio chunks.
        Video at native 25fps (no interpolation), audio at 50fps.
        """
        while not quit_event.is_set():
            try:
                res_frame, idx, audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            # Determine if this is silence or speech
            is_silence = audio_frames[0][1] != 0 and audio_frames[1][1] != 0

            if is_silence:
                self.speaking = False
                combine_frame = self.frame_list[idx]
            else:
                if not self.speaking and self._msg_submit_time:
                    log.info(
                        "[TIMING] first lipsync frame: %.3fs since feed",
                        time.time() - self._msg_submit_time,
                    )
                self.speaking = True
                try:
                    combine_frame = self._paste_back(res_frame, idx)
                except Exception as e:
                    log.warning("paste_back error: %s", e)
                    continue

            # Push 1 video frame at 25fps
            new_vframe = VideoFrame.from_ndarray(combine_frame, format="bgr24")
            asyncio.run_coroutine_threadsafe(
                video_track._queue.put((new_vframe, None)), loop
            )

            # Push 2 audio frames at 50fps
            for pcm, frame_type, eventpoint in audio_frames:
                pcm_int16 = (pcm * 32767).astype(np.int16)
                af = AudioFrame(format="s16", layout="mono", samples=pcm_int16.shape[0])
                af.planes[0].update(pcm_int16.tobytes())
                af.sample_rate = SAMPLE_RATE
                asyncio.run_coroutine_threadsafe(
                    audio_track._queue.put((af, eventpoint)), loop
                )

        log.info("Frame processor stopped")
