"""Playback de audio orientado a fila, com parada quase instantanea.

Este e o modulo que faz a meta de <100ms do barge-in ser alcancavel: o
callback do PortAudio consome de um buffer protegido por lock, nunca faz
escrita bloqueante, e "parar" e apenas limpar esse buffer -- o efeito e
audivel no bloco de callback seguinte (bloco tipico de alguns ms a
poucas dezenas de ms), independente de quanto trabalho de GPU o
LLM/TTS ainda estejam fazendo.

Distincao importante: silencio porque *ainda nao ha o proximo pedaco de
audio pronto* (TTS ainda sintetizando a proxima sentenca) nao pode disparar
o evento de "turno terminou naturalmente" -- por isso existe
`mark_no_more_chunks()`, que o orchestrator chama so depois que o LLM e o
TTS sinalizarem que nao ha mais sentencas vindo para aquele turno.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioPlayback:
    def __init__(
        self,
        sample_rate: int = 24000,
        device: int | str | None = None,
        on_naturally_stopped: Callable[[str], None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._on_naturally_stopped = on_naturally_stopped

        self._lock = threading.Lock()
        self._buffer: list[np.ndarray] = []
        self._current_turn_id: str | None = None
        self._no_more_chunks = False
        self._any_chunk_enqueued = False

        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sd.OutputStream | None = None

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("Playback de audio iniciado (device=%s, sr=%d)", self.device, self.sample_rate)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_on_naturally_stopped(self, callback: Callable[[str], None]) -> None:
        self._on_naturally_stopped = callback

    def start_turn(self, turn_id: str) -> None:
        """Chamado quando uma resposta nova comeca a ser falada."""
        with self._lock:
            self._buffer.clear()
            self._current_turn_id = turn_id
            self._no_more_chunks = False
            self._any_chunk_enqueued = False

    def enqueue(self, pcm: np.ndarray, turn_id: str) -> None:
        """Enfileira audio para tocar. Descartado silenciosamente se `turn_id`
        nao for mais o turno atual (ex: chegou depois de um barge-in)."""
        with self._lock:
            if turn_id != self._current_turn_id:
                logger.debug("Descartando chunk de audio de turno obsoleto %s", turn_id)
                return
            # .reshape(-1) e defensivo: um array com dimensao extra (ex:
            # shape (1, N) em vez de (N,), caso de alguns exports ONNX do
            # Kokoro) faz `len(chunk)` no callback de audio contar a
            # dimensao errada e corrompe o preenchimento do buffer -- ver
            # docs/TROUBLESHOOTING.md.
            self._buffer.append(np.asarray(pcm, dtype=np.float32).reshape(-1))
            self._any_chunk_enqueued = True

    def mark_no_more_chunks(self, turn_id: str) -> None:
        """LLM+TTS sinalizaram que nao ha mais sentencas para este turno."""
        with self._lock:
            if turn_id == self._current_turn_id:
                self._no_more_chunks = True

    def stop_immediately(self, reason: str) -> None:
        """O efeito colateral central de um barge-in: silencio no proximo bloco."""
        with self._lock:
            self._buffer.clear()
            self._current_turn_id = None
            self._no_more_chunks = False
            self._any_chunk_enqueued = False
        logger.info("Playback interrompido (%s)", reason)

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning("Status do stream de playback: %s", status)

        fire_natural_stop: str | None = None
        with self._lock:
            out = np.zeros(frames, dtype=np.float32)
            filled = 0
            while filled < frames and self._buffer:
                chunk = self._buffer[0]
                take = min(frames - filled, len(chunk))
                out[filled : filled + take] = chunk[:take]
                filled += take
                if take == len(chunk):
                    self._buffer.pop(0)
                else:
                    self._buffer[0] = chunk[take:]
            outdata[:, 0] = out

            drained = not self._buffer
            # `_any_chunk_enqueued` e essencial aqui: sem ele, o buffer
            # comeca vazio por padrao, e se `mark_no_more_chunks()` for
            # chamado antes do PRIMEIRO chunk de audio chegar (comum -- o
            # LLM termina de gerar texto rapido, mas a sintese de TTS de
            # cada sentenca ainda esta rodando em paralelo), essa condicao
            # disparava na hora, resetando `_current_turn_id` pra None
            # antes de qualquer audio real ser enfileirado. Todo
            # `enqueue()` seguinte pra esse turno via entao descartado
            # silenciosamente (turn_id nao bate mais) -- causa raiz real de
            # "nunca sai audio nenhum" mesmo com a sintese funcionando.
            if drained and self._no_more_chunks and self._any_chunk_enqueued and self._current_turn_id is not None:
                fire_natural_stop = self._current_turn_id
                self._current_turn_id = None
                self._no_more_chunks = False
                self._any_chunk_enqueued = False

        if fire_natural_stop is not None and self._on_naturally_stopped is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._on_naturally_stopped, fire_natural_stop)

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._current_turn_id is not None
