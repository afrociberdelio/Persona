"""Captura de microfone via sounddevice.

O callback do PortAudio roda numa thread de audio de tempo real: ele so
empurra PCM cru numa fila thread-safe e retorna. Toda logica (VAD,
segmentacao) fica numa task asyncio que consome essa fila -- e a unica
forma de nao introduzir xruns/dropouts no stream de audio.
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 32,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.device = device
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning("Status do stream de captura: %s", status)
        assert self._loop is not None
        frame = indata[:, 0].copy()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, frame)

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "Captura de audio iniciada (device=%s, sr=%d, frame=%d amostras)",
            self.device,
            self.sample_rate,
            self.frame_samples,
        )

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def frames(self):
        """Async generator de frames PCM float32 mono, um por chunk de captura."""
        while True:
            frame = await self._queue.get()
            yield frame
