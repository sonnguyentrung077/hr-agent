"""WebRTC media tracks for avatar streaming.

Extracted from LiveTalking/webrtc.py. Provides PlayerStreamTrack
(async queue-based frame delivery with PTS pacing) and HumanPlayer
(wraps audio+video tracks, manages the render worker thread).
"""

import asyncio
import fractions
import logging
import threading
import time
from typing import Optional, Set, Union

from aiortc import MediaStreamTrack
from av import AudioFrame
from av.frame import Frame
from av.packet import Packet

log = logging.getLogger(__name__)

# Timing constants
AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_PTIME = 0.040  # 40ms = 25fps
VIDEO_CLOCK_RATE = 90000
SAMPLE_RATE = 16000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)


class PlayerStreamTrack(MediaStreamTrack):
    """Media track that delivers frames from an async queue at correct pacing."""

    def __init__(self, player: "HumanPlayer", kind: str):
        super().__init__()
        self.kind = kind
        self._player = player
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.current_frame_count = 0

    _start: float
    _timestamp: int

    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception("Track not live")

        if self.kind == "video":
            ptime, clock, time_base = VIDEO_PTIME, VIDEO_CLOCK_RATE, VIDEO_TIME_BASE
        else:
            ptime, clock, time_base = AUDIO_PTIME, SAMPLE_RATE, AUDIO_TIME_BASE

        if hasattr(self, "_timestamp"):
            self._timestamp += int(ptime * clock)
            self.current_frame_count += 1
            wait = self._start + self.current_frame_count * ptime - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
            log.debug("%s track started at %.3f", self.kind, self._start)

        return self._timestamp, time_base

    async def recv(self) -> Union[Frame, Packet]:
        self._player._start(self)
        frame, eventpoint = await self._queue.get()

        if frame is None:
            self.stop()
            raise Exception("Track ended")

        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base

        if eventpoint and self._player is not None:
            self._player.notify(eventpoint)

        return frame

    def stop(self):
        super().stop()
        while not self._queue.empty():
            item = self._queue.get_nowait()
            del item
        if self._player is not None:
            self._player._stop(self)
            self._player = None


def _player_worker_thread(quit_event, loop, container, audio_track, video_track):
    container.render(quit_event, loop, audio_track, video_track)


class HumanPlayer:
    """Wraps audio+video PlayerStreamTracks and manages the render worker thread."""

    def __init__(self, avatar_engine):
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None
        self.__started: Set[PlayerStreamTrack] = set()
        self.__audio = PlayerStreamTrack(self, kind="audio")
        self.__video = PlayerStreamTrack(self, kind="video")
        self.__container = avatar_engine

    def notify(self, eventpoint):
        if self.__container is not None:
            self.__container.notify(eventpoint)

    @property
    def audio(self) -> MediaStreamTrack:
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        if self.__thread is None:
            log.debug("HumanPlayer: starting worker thread")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=_player_worker_thread,
                args=(
                    self.__thread_quit,
                    asyncio.get_event_loop(),
                    self.__container,
                    self.__audio,
                    self.__video,
                ),
                daemon=True,
            )
            self.__thread.start()

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)
        if not self.__started and self.__thread is not None:
            log.debug("HumanPlayer: stopping worker thread")
            self.__thread_quit.set()
            self.__thread.join(timeout=10)
            self.__thread = None
        if not self.__started and self.__container is not None:
            self.__container = None
