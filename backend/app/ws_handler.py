"""WebSocket endpoint: browser <-> local Whisper STT <-> pipeline."""

import asyncio
import json
import time

from fastapi import WebSocket, WebSocketDisconnect

from .config import SYSTEM_PROMPT
from .dependencies import log
from .pipeline import run_pipeline
from .rtc_handler import sessions
from .stt import StreamingSession

ECHO_COOLDOWN = 0.2  # seconds after bot stops before accepting mic audio

# Common Whisper hallucinations produced from silence / noise
HALLUCINATION_PHRASES = {
    "thank you", "thanks", "thank you.", "thanks.",
    "thank you for watching", "thanks for watching",
    "bye", "bye bye", "bye.",
    "obrigado", "obrigada", "obrigado.", "obrigada.",
    "продолжение следует", "阿 会", "gracias", "yeah"
}


async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    turn_queue: asyncio.Queue[str] = asyncio.Queue()
    responding = asyncio.Event()  # set while bot is speaking
    respond_end_time = [0.0]  # mutable container for cooldown timestamp

    # Look up avatar engine from WebRTC session
    session_id = ws.query_params.get("session_id")
    session = sessions.get(session_id) if session_id else None
    avatar_engine = session.engine if session else None
    session_closed = session.closed if session else None
    if avatar_engine:
        log.info("[WS] Bound to avatar session %s", session_id)
    else:
        log.info("[WS] No avatar session — audio-only mode")

    # Store history so /summary endpoint can access it after session cleanup
    if session_id:
        from .dependencies import session_histories
        session_histories[session_id] = history

    async def turn_worker():
        """Process queued turns one at a time."""
        while True:
            text = await turn_queue.get()
            if session_closed and session_closed.is_set():
                log.info("[WS] Ignoring turn — RTC session closed")
                turn_queue.task_done()
                continue
            responding.set()
            try:
                await ws.send_json({"type": "status", "status": "thinking"})
                await run_pipeline(ws, text, history, avatar_engine)
                await ws.send_json({"type": "status", "status": "listening"})
            except Exception as e:
                log.error(f"[PIPELINE] {e}", exc_info=True)
            finally:
                responding.clear()
                respond_end_time[0] = time.time()
                turn_queue.task_done()

    async def enqueue_turn(text: str):
        if not text:
            return
        log.info(f"[QUEUE] Enqueued turn (depth={turn_queue.qsize()}): {text[:80]!r}")
        await turn_queue.put(text)

    # --- Local STT callbacks ---

    async def on_partial(text: str):
        if not text:
            return
        await ws.send_json({"type": "partial_transcript", "text": text})

    async def on_final(text: str):
        if not text:
            return
        # Drop echo: while bot speaks OR during cooldown
        if responding.is_set() or (time.time() - respond_end_time[0]) < ECHO_COOLDOWN:
            log.info(f"[STT DROP] echo suppressed: {text[:80]!r}")
            return
        if text.lower().strip(" .!,") in HALLUCINATION_PHRASES:
            log.info(f"[STT DROP] hallucination filtered: {text!r}")
            return
        log.info(f"[STT FINAL] {text}")
        await ws.send_json({"type": "final_transcript", "text": text})
        await enqueue_turn(text)

    # --- Create local STT session ---

    stt_model = ws.app.state.stt
    stt_session = StreamingSession(stt_model, on_partial=on_partial, on_final=on_final)

    try:
        async def recv_client():
            try:
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if "bytes" in msg:
                        # Gate mic audio: don't feed echo into STT
                        if not responding.is_set() and (time.time() - respond_end_time[0]) > ECHO_COOLDOWN:
                            await stt_session.feed_audio(msg["bytes"])
                    elif "text" in msg:
                        ctrl = json.loads(msg["text"])
                        if ctrl.get("type") == "end_session":
                            stt_session.close()
                            break
            except WebSocketDisconnect:
                stt_session.close()

        async def watch_rtc():
            if not session_closed:
                return
            await session_closed.wait()
            log.info("[WS] RTC session %s closed — tearing down", session_id)

        worker_task = asyncio.create_task(turn_worker())
        tasks = [
            asyncio.create_task(stt_session.run()),
            asyncio.create_task(recv_client()),
            asyncio.create_task(watch_rtc()),
            worker_task,
        ]
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()

    except Exception as e:
        log.error(f"[WS] {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        stt_session.close()
        try:
            await ws.close()
        except Exception:
            pass
