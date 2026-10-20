from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import struct

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from backend.contracts.schemas import DownstreamEvent
from backend.transport.live_queue import LiveRequestQueue
from backend.transport.runner import FullDuplexRunner
from backend.transport.session import InMemorySession

app = FastAPI(title="KHMER-SOVEREIGN-S2ST v10.7 Full-Duplex Server")
runner = FullDuplexRunner()
GPU_SEMAPHORE = asyncio.Semaphore(1)


@app.websocket("/ws/live-dub/{session_id}")
async def live_dub_endpoint(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = InMemorySession(session_id)
    queue = LiveRequestQueue()
    out_queue: asyncio.Queue[DownstreamEvent] = asyncio.Queue()

    async def upstream_reader() -> None:
        while True:
            msg = await websocket.receive()
            if msg.get("bytes"):
                raw: bytes = msg["bytes"]
                if len(raw) <= 8:
                    continue
                (ts_ms,) = struct.unpack("<d", raw[:8])
                pcm = np.frombuffer(raw[8:], dtype=np.float32).copy()
                await queue.send_realtime(pcm, timestamp_ms=ts_ms)
            elif msg.get("text"):
                data = json.loads(msg["text"])
                action = str(data.get("action", "config"))
                ts_ms = float(data.get("video_time_ms", 0.0))
                rate = float(data.get("playback_rate", queue.playback_rate))
                await queue.send_content(action, timestamp_ms=ts_ms, playback_rate=rate, payload=data)
                # Immediate <1ms Control-Plane Interrupt/Seek Execution (never blocked by GPU worker)
                if action in {"seek", "interrupt"}:
                    session.reset_on_seek(ts_ms)
                    await out_queue.put(
                        DownstreamEvent(
                            event_type="interrupted",
                            utterance_id=session.next_utterance_id,
                            utterance_start_ms=ts_ms,
                            playback_rate=queue.playback_rate,
                            interrupt_epoch=queue.interrupt_epoch,
                        )
                    )

    async def worker_loop() -> None:
        while True:
            pkt = await queue.get_next()
            if pkt.kind == "content":
                continue
            if pkt.kind == "realtime_audio" and pkt.pcm_data is not None:
                sliced = session.append_pcm_chunk(pkt.pcm_data, timestamp_ms=pkt.timestamp_ms)
                if sliced is not None:
                    uid, start_ms, slice_pcm, pause_ms = sliced
                    # v10.7 P0-B: snapshot the interrupt epoch BEFORE waiting
                    # for the GPU semaphore. v10.6 snapshotted it INSIDE the
                    # worker thread, i.e. after the semaphore wait, so a seek
                    # that landed while the worker was queued was invisible to
                    # the fence and the stale pre-seek audio was emitted after
                    # the 'interrupted' packet.
                    pkt_epoch = queue.interrupt_epoch
                    async with GPU_SEMAPHORE:
                        # Re-check after acquiring the semaphore: a seek may
                        # have arrived during the wait.
                        if queue.interrupt_epoch != pkt_epoch:
                            continue
                        evt = await asyncio.to_thread(
                            runner.process_utterance_slice,
                            session,
                            queue,
                            uid,
                            start_ms,
                            slice_pcm,
                            pause_ms,
                        )
                    if evt is not None:
                        await out_queue.put(evt)

    async def downstream_writer() -> None:
        while True:
            evt = await out_queue.get()
            await websocket.send_json(asdict(evt))

    tasks = [
        asyncio.create_task(upstream_reader()),
        asyncio.create_task(worker_loop()),
        asyncio.create_task(downstream_writer()),
    ]
    try:
        await asyncio.gather(*tasks)
    except WebSocketDisconnect:
        for t in tasks:
            t.cancel()
