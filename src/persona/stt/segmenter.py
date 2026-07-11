"""Buferiza audio de uma fala guiado por VAD e decide quando transcrever.

Dois caminhos, como descrito no plano:

1. Primario: buferiza tudo desde o inicio da fala ate o VAD sinalizar fim;
   transcreve o buffer inteiro de uma vez. Cobre o caso comum (perguntas/
   comandos curtos).
2. Fala longa: se o buffer passar de `long_chunk_s` sem uma pausa, emite
   uma transcricao *parcial* (preview) do trecho acumulado. Import ante:
   essa parcial e so um preview (ex: para log/UI de "ouvindo...") -- a
   transcricao final e sempre feita transcrevendo o buffer inteiro no fim
   da fala, e nao por costura de texto entre chunks sobrepostos. Uma
   politica completa de reconciliacao tipo LocalAgreement adiciona
   complexidade real (alinhamento de texto, deteccao de fronteira de
   palavra) que nao se paga no v1 -- o whisper transcreve um buffer de
   60-90s em poucos segundos, entao "transcrever tudo de novo no final"
   e simples e correto.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import numpy as np

from persona.stt.whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)


class UtteranceSegmenter:
    def __init__(
        self,
        whisper: WhisperEngine,
        sample_rate: int = 16000,
        long_chunk_s: float = 20.0,
        overlap_s: float = 2.5,
        on_partial: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._whisper = whisper
        self.sample_rate = sample_rate
        self.long_chunk_samples = int(long_chunk_s * sample_rate)
        self.overlap_samples = int(overlap_s * sample_rate)
        self._on_partial = on_partial

        self._buffer: list[np.ndarray] = []
        self._buffered_samples = 0
        self._samples_since_last_chunk = 0
        self._background_tasks: set[asyncio.Task] = set()

    def start_utterance(self) -> None:
        self._buffer.clear()
        self._buffered_samples = 0
        self._samples_since_last_chunk = 0

    def add_frame(self, frame: np.ndarray) -> None:
        self._buffer.append(frame)
        self._buffered_samples += len(frame)
        self._samples_since_last_chunk += len(frame)

        if self._samples_since_last_chunk >= self.long_chunk_samples and self._on_partial is not None:
            self._samples_since_last_chunk = 0
            chunk = self._current_audio()[-(self.long_chunk_samples + self.overlap_samples) :]
            task = asyncio.ensure_future(self._emit_partial(chunk))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _emit_partial(self, chunk: np.ndarray) -> None:
        try:
            text, _ = await self._whisper.transcribe(chunk)
            if text and self._on_partial is not None:
                await self._on_partial(text)
        except Exception:
            logger.exception("Falha ao gerar transcricao parcial de fala longa")

    def _current_audio(self) -> np.ndarray:
        if not self._buffer:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._buffer)

    async def finalize(self) -> tuple[str, float]:
        """Transcreve o buffer inteiro da fala e retorna (texto, duracao_s)."""
        audio = self._current_audio()
        duration_s = len(audio) / self.sample_rate
        if len(audio) == 0:
            return "", 0.0
        text, _ = await self._whisper.transcribe(audio)
        return text, duration_s
