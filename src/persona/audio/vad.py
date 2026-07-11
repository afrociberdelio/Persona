"""Wrapper de streaming sobre o Silero VAD.

Roda deliberadamente na CPU (nunca na GPU): a deteccao de inicio de fala
para o barge-in nao pode ficar atras de uma fila de trabalho de GPU do
STT/LLM/TTS. O modelo e minusculo o suficiente para isso nao importar
(poucos ms de CPU por chunk de 32ms).

Usa `VADIterator` do pacote `silero-vad`, que ja implementa a logica de
onset/offset com padding -- so adicionamos o `hangover_ms` (tolerancia
extra antes de fechar o segmento) por cima, para nao cortar a fala em
pequenas pausas naturais.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from silero_vad import VADIterator, load_silero_vad

logger = logging.getLogger(__name__)


@dataclass
class VADEvent:
    kind: str  # "start" ou "end"
    duration_s: float | None = None  # preenchido so em "end"


class SileroVADStream:
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_ms: int = 200,
        end_silence_ms: int = 600,
        hangover_ms: int = 150,
    ) -> None:
        self.sample_rate = sample_rate
        self.min_speech_ms = min_speech_ms
        self.hangover_ms = hangover_ms

        model = load_silero_vad(onnx=True)
        self._iterator = VADIterator(
            model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=end_silence_ms,
        )

        self._speech_started_at: float | None = None
        self._pending_hangover_since: float | None = None

    def reset(self) -> None:
        self._iterator.reset_states()
        self._speech_started_at = None
        self._pending_hangover_since = None

    def process(self, frame: np.ndarray) -> VADEvent | None:
        """Alimenta um frame mono float32 e retorna um VADEvent se algo mudou."""
        result = self._iterator(frame, return_seconds=False)
        now = time.monotonic()

        if result is not None and "start" in result:
            self._pending_hangover_since = None
            if self._speech_started_at is None:
                self._speech_started_at = now
                logger.debug("VAD: inicio de fala detectado")
                return VADEvent(kind="start")

        if result is not None and "end" in result:
            # Silero ja aplicou min_silence_duration_ms internamente; ainda
            # assim damos um hangover extra configuravel antes de considerar
            # a fala definitivamente encerrada, para tolerar pausas curtas
            # de respiracao/raciocinio sem fragmentar a mesma frase em dois
            # segmentos.
            self._pending_hangover_since = now

        if self._pending_hangover_since is not None:
            elapsed_ms = (now - self._pending_hangover_since) * 1000
            if elapsed_ms >= self.hangover_ms and self._speech_started_at is not None:
                duration_s = now - self._speech_started_at
                self._pending_hangover_since = None
                self._speech_started_at = None
                if duration_s * 1000 < self.min_speech_ms:
                    logger.debug("VAD: fala descartada por ser curta demais (%.0fms)", duration_s * 1000)
                    return None
                logger.debug("VAD: fim de fala (%.2fs)", duration_s)
                return VADEvent(kind="end", duration_s=duration_s)

        return None
