"""Embeddings multilingues (PT-BR incluido) via FastEmbed.

FastEmbed roda em ONNX Runtime na CPU e nao depende de torch -- mantem a
VRAM inteira disponivel para STT/LLM/TTS, o que importa numa GPU pequena.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastembed import TextEmbedding


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self._model = TextEmbedding(model_name=model_name)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedder")
        self._dim: int | None = None

    def _embed_sync(self, text: str) -> np.ndarray:
        return next(iter(self._model.embed([text])))

    async def embed(self, text: str) -> np.ndarray:
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(self._executor, self._embed_sync, text)
        self._dim = len(vector)
        return vector

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("Chame embed() ao menos uma vez antes de ler `dim`, ou defina explicitamente.")
        return self._dim
