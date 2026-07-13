"""Gerencia jobs de sintese TTS por turno, com suporte a cancelamento.

Cada sentenca liberada pelo `SentenceChunker` vira um job aqui. Jobs sao
tageados com o `turn_id` do turno que os originou; num barge-in,
`cancel_turn()` cancela as tasks ainda nao concluidas e marca o turno
como obsoleto -- jobs que ja estavam no meio da sintese (rodando no
executor) terminam, mas seu resultado e descartado ao inves de enfileirado
pra tocar, porque o turno nao e mais o atual.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

import numpy as np

from persona.tts.kokoro_engine import KokoroEngine

logger = logging.getLogger(__name__)

OnChunkReady = Callable[[str, int, np.ndarray, int], None]


class SynthesisQueue:
    def __init__(self, engine: KokoroEngine, voice: str = "pf_dora", speed: float = 1.0) -> None:
        self._engine = engine
        self.voice = voice
        self.speed = speed
        self._current_turn_id: str | None = None
        self._tasks: dict[str, list[asyncio.Task]] = {}

    def start_turn(self, turn_id: str) -> None:
        self._current_turn_id = turn_id
        self._tasks[turn_id] = []

    def cancel_turn(self, turn_id: str, reason: str = "barge_in") -> None:
        for task in self._tasks.pop(turn_id, []):
            if not task.done():
                task.cancel()
        if self._current_turn_id == turn_id:
            self._current_turn_id = None
        logger.debug("Fila de sintese do turno %s cancelada (%s)", turn_id, reason)

    def submit(self, turn_id: str, sentence_id: int, text: str, on_ready: OnChunkReady) -> asyncio.Task:
        task = asyncio.ensure_future(self._run(turn_id, sentence_id, text, on_ready))
        self._tasks.setdefault(turn_id, []).append(task)
        return task

    async def _run(self, turn_id: str, sentence_id: int, text: str, on_ready: OnChunkReady) -> None:
        logger.info("Fila TTS: iniciando sintese da sentenca %d do turno %s", sentence_id, turn_id)
        try:
            pcm, sample_rate = await self._engine.synthesize(text, voice=self.voice, speed=self.speed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Falha ao sintetizar sentenca %d do turno %s", sentence_id, turn_id)
            return

        if turn_id != self._current_turn_id:
            logger.info(
                "Fila TTS: descartando audio da sentenca %d -- turno %s nao e mais o atual (atual=%s)",
                sentence_id,
                turn_id,
                self._current_turn_id,
            )
            return

        on_ready(turn_id, sentence_id, pcm, sample_rate)
