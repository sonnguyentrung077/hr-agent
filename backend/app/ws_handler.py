"""WebSocket endpoint: browser <-> AssemblyAI STT <-> pipeline."""

import asyncio
import json
import time

import websockets
from fastapi import WebSocket, WebSocketDisconnect

from .config import AAI_URL, ASSEMBLY_KEY, SYSTEM_PROMPT
from .dependencies import log
from .pipeline import run_pipeline
from .rtc_handler import sessions

ECHO_COOLDOWN = 1.5  # seconds after bot stops before accepting mic audio


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
        from main import _session_histories
        _session_histories[session_id] = history

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

    log.info(f"[AAI] Connecting to {AAI_URL[:80]}...")
    try:
        async with websockets.connect(
            AAI_URL, additional_headers={"Authorization": ASSEMBLY_KEY}
        ) as aai:
            log.info("[AAI] Connected")

            async def recv_aai():
                async for raw in aai:
                    msg = json.loads(raw)
                    t = msg.get("type")
                    log.debug(f"[AAI RAW] {json.dumps(msg)[:300]}")
                    if t == "Begin":
                        log.info(f"[AAI] Session started: {msg.get('id')}")
                    elif t == "Turn":
                        text = msg.get("transcript", "").strip()
                        is_final = msg.get(
                            "turn_is_formatted", False
                        ) or msg.get("end_of_turn", False)
                        log.info(
                            f"[AAI Turn] final={is_final} "
                            f"turn_is_formatted={msg.get('turn_is_formatted')} "
                            f"end_of_turn={msg.get('end_of_turn')} "
                            f"text={text[:80]!r}"
                        )
                        if is_final and text:
                            # Drop echo: while bot speaks OR during cooldown
                            if responding.is_set() or (time.time() - respond_end_time[0]) < ECHO_COOLDOWN:
                                log.info(f"[STT DROP] echo suppressed: {text[:80]!r}")
                                continue
                            log.info(f"[STT FINAL] {text}")
                            await ws.send_json(
                                {"type": "final_transcript", "text": text}
                            )
                            await enqueue_turn(text)
                        elif text:
                            await ws.send_json(
                                {"type": "partial_transcript", "text": text}
                            )
                    elif t == "Termination":
                        log.info("[AAI] Session terminated")
                    else:
                        log.info(
                            f"[AAI] Unknown type: {t} — {json.dumps(msg)[:200]}"
                        )

            async def recv_client():
                try:
                    while True:
                        msg = await ws.receive()
                        if msg.get("type") == "websocket.disconnect":
                            break
                        if "bytes" in msg:
                            # Gate mic audio: don't feed echo into STT
                            if not responding.is_set() and (time.time() - respond_end_time[0]) > ECHO_COOLDOWN:
                                await aai.send(msg["bytes"])
                        elif "text" in msg:
                            ctrl = json.loads(msg["text"])
                            if ctrl.get("type") == "end_session":
                                await aai.send(
                                    json.dumps({"type": "Terminate"})
                                )
                                break
                except WebSocketDisconnect:
                    try:
                        await aai.send(json.dumps({"type": "Terminate"}))
                    except Exception:
                        pass

            async def watch_rtc():
                if not session_closed:
                    return
                await session_closed.wait()
                log.info("[WS] RTC session %s closed — tearing down", session_id)

            worker_task = asyncio.create_task(turn_worker())
            tasks = [
                asyncio.create_task(recv_aai()),
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
        try:
            await ws.close()
        except Exception:
            pass
