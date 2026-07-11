"""Wrapper fino sobre kokoro-onnx.

Escolhido sobre o Kokoro em PyTorch por ter um caminho de instalacao mais
simples no Windows (ONNX Runtime, sem custom CUDA ops) e ainda assim
acelerar por GPU via CUDAExecutionProvider. `.create()` e bloqueante e
roda em thread executor, igual ao WhisperEngine.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from kokoro_onnx import Kokoro

logger = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24000


class KokoroEngine:
    def __init__(
        self,
        model_path: str = "models/kokoro-v1.0.onnx",
        voices_path: str = "models/voices-v1.0.bin",
        device: str = "cuda",
    ) -> None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        logger.info("Carregando Kokoro ONNX (device=%s)", device)
        self._kokoro = Kokoro(model_path, voices_path, providers=providers)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kokoro")

    def _synthesize_sync(self, text: str, voice: str, speed: float, lang: str) -> tuple[np.ndarray, int]:
        samples, sample_rate = self._kokoro.create(text, voice=voice, speed=speed, lang=lang)
        return np.asarray(samples, dtype=np.float32), sample_rate

    async def synthesize(
        self, text: str, voice: str = "pf_dora", speed: float = 1.0, lang: str = "pt-br"
    ) -> tuple[np.ndarray, int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._synthesize_sync, text, voice, speed, lang)
