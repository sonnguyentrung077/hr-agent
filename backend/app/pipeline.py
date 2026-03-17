"""Pipeline: GPT streaming -> Cartesia TTS -> audio delivery."""

import asyncio
import threading
import time

from cartesia import Cartesia
from fastapi import WebSocket

from .config import (
    CARTESIA_KEY,
    CARTESIA_MODEL,
    CARTESIA_VOICE_ID,
    MODEL,
    SAMPLE_RATE_TTS,
    SENTENCE_ENDS,
    TTS_LANGUAGE,
)
from .dependencies import log, openai_client


async def run_pipeline(ws: WebSocket, user_text: str, history: list[dict]):
    t0 = time.time()
    history.append({"role": "user", "content": user_text})

    text_q: asyncio.Queue[str | None] = asyncio.Queue()
    audio_q: asyncio.Queue[bytes | None | Exception] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    tts_thread = threading.Thread(
        target=_tts_worker, args=(text_q, audio_q, loop), daemon=True
    )
    tts_thread.start()

    gpt_task = asyncio.create_task(_gpt_stream(ws, history, text_q))
    audio_task = asyncio.create_task(_audio_sender(ws, audio_q))

    full_response = await gpt_task
    await audio_task
    tts_thread.join(timeout=10)

    history.append({"role": "assistant", "content": full_response})
    elapsed = time.time() - t0
    log.info(f"[PIPELINE] Total: {elapsed:.2f}s")
    await ws.send_json(
        {
            "type": "response_complete",
            "text": full_response,
            "time": round(elapsed, 3),
        }
    )


# ─── GPT streaming ──────────────────────────────────────────────────────────


async def _gpt_stream(
    ws: WebSocket, history: list[dict], text_q: asyncio.Queue[str | None]
) -> str:
    t = time.time()
    log.info(f"[GPT] Streaming ({MODEL})...")
    stream = await openai_client.chat.completions.create(
        model=MODEL,
        messages=history,
        stream=True,
    )
    full = ""
    buf = ""
    first_pushed = False

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if not delta.content:
            continue
        if not full:
            log.info(f"[GPT] First token at {time.time()-t:.2f}s")
        full += delta.content
        await ws.send_json({"type": "response_text", "token": delta.content})

        if not first_pushed:
            buf += delta.content
            if any(c in SENTENCE_ENDS for c in buf) or len(buf) > 120:
                await text_q.put(buf)
                log.info(f"[GPT->TTS] First push: {buf[:60]!r}")
                first_pushed = True
        else:
            await text_q.put(delta.content)

    if not first_pushed and buf:
        await text_q.put(buf)
    await text_q.put(None)
    log.info(
        f"[GPT] Done: {len(full)} chars in {time.time()-t:.2f}s — {full[:120]!r}"
    )
    return full


# ─── TTS worker (runs in thread) ────────────────────────────────────────────


def _tts_worker(
    text_q: asyncio.Queue[str | None],
    audio_q: asyncio.Queue[bytes | None | Exception],
    loop: asyncio.AbstractEventLoop,
):
    t = time.time()
    n_chunks = 0
    try:
        client = Cartesia(api_key=CARTESIA_KEY)
        with client.tts.websocket_connect() as conn:
            log.info(f"[TTS] Connected ({time.time()-t:.2f}s)")
            ctx = conn.context(
                model_id=CARTESIA_MODEL,
                voice={"mode": "id", "id": CARTESIA_VOICE_ID},
                output_format={
                    "container": "raw",
                    "encoding": "pcm_f32le",
                    "sample_rate": SAMPLE_RATE_TTS,
                },
                language=TTS_LANGUAGE,
            )
            n_tokens = 0
            while True:
                tok = asyncio.run_coroutine_threadsafe(
                    text_q.get(), loop
                ).result()
                if tok is None:
                    break
                ctx.push(tok)
                n_tokens += 1
            log.info(
                f"[TTS] {n_tokens} tokens pushed ({time.time()-t:.2f}s)"
            )
            ctx.no_more_inputs()

            for resp in ctx.receive():
                if resp.type == "chunk" and resp.audio:
                    n_chunks += 1
                    if n_chunks == 1:
                        log.info(
                            f"[TTS] First audio at {time.time()-t:.2f}s"
                        )
                    loop.call_soon_threadsafe(audio_q.put_nowait, resp.audio)
            log.info(
                f"[TTS] Done: {n_chunks} chunks in {time.time()-t:.2f}s"
            )
    except Exception as e:
        log.error(f"[TTS] ERROR: {e}", exc_info=True)
        loop.call_soon_threadsafe(audio_q.put_nowait, e)
    finally:
        loop.call_soon_threadsafe(audio_q.put_nowait, None)


# ─── Audio sender ────────────────────────────────────────────────────────────


async def _audio_sender(
    ws: WebSocket, audio_q: asyncio.Queue[bytes | None | Exception]
):
    n = 0
    while True:
        item = await audio_q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            log.error(f"[AUDIO] TTS error: {item}")
            await ws.send_json({"type": "tts_error", "message": str(item)})
            break
        await ws.send_bytes(item)
        n += 1
    log.info(f"[AUDIO] {n} chunks sent to client")
    await ws.send_json({"type": "audio_chunk_end"})
