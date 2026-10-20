from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import numpy as np

from backend.contracts.schemas import LivePacket


class LiveRequestQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[LivePacket] = asyncio.Queue()
        self.interrupt_epoch: int = 0
        self.playback_rate: float = 1.0

    async def send_realtime(self, pcm_chunk: np.ndarray, timestamp_ms: float) -> None:
        await self._queue.put(
            LivePacket(
                kind="realtime_audio",
                timestamp_ms=float(timestamp_ms),
                pcm_data=pcm_chunk,
            )
        )

    async def send_content(
        self,
        action: str,
        timestamp_ms: float,
        playback_rate: float = 1.0,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.playback_rate = float(playback_rate)
        if action in {"interrupt", "seek"}:
            self.flush_all()
        body = dict(payload or {})
        body["action"] = action
        body["playback_rate"] = self.playback_rate
        await self._queue.put(
            LivePacket(kind="content", timestamp_ms=float(timestamp_ms), payload=body)
        )

    def flush_all(self) -> int:
        dropped = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        self.interrupt_epoch += 1
        return dropped

    async def get_next(self) -> LivePacket:
        return await self._queue.get()
