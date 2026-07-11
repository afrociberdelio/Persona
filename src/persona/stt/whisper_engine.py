"""Wrapper fino sobre faster-whisper.

faster-whisper (CTranslate2) nao e streaming nativo -- ele transcreve um
buffer de audio de uma vez. O streaming "de verdade" e responsabilidade do
`UtteranceSegmenter` (segmenter.py), que decide *quando* chamar este
engine (fim de fala via VAD, ou chunk intermediario de fala longa).
`.transcribe()` e bloqueante e roda em thread executor para nao travar o
loop asyncio.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class WhisperEngine:
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str | None = "pt",
    ) -> None:
        self.language = language
        logger.info("Carregando faster-whisper '%s' (device=%s, compute_type=%s)", model_size, device, compute_type)
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")

    def _transcribe_sync(self, audio: np.ndarray) -> tuple[str, str | None]:
        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            vad_filter=False,  # VAD ja foi feito rio acima pelo Silero
            beam_size=1,  # greedy: prioriza latencia sobre a ultima fracao de precisao
        )
        text = "".join(segment.text for segment in segments).strip()
        return text, info.language if info else None

    async def transcribe(self, audio: np.ndarray) -> tuple[str, str | None]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._transcribe_sync, audio)
