"""Wrapper fino sobre kokoro-onnx.

Escolhido sobre o Kokoro em PyTorch por ter um caminho de instalacao mais
simples no Windows (ONNX Runtime, sem custom CUDA ops) e ainda assim
acelerar por GPU via CUDAExecutionProvider. `.create()` e bloqueante e
roda em thread executor, igual ao WhisperEngine.

Duas particularidades da lib `kokoro-onnx` (0.5.0, a mais recente no PyPI)
que exigiram tratamento especial aqui:

1. `Kokoro.__init__` NAO aceita um parametro `providers` -- ela escolhe o
   provider sozinha (via deteccao do pacote `onnxruntime-gpu` instalado, ou
   via a variavel de ambiente `ONNX_PROVIDER`). Por isso o device e
   controlado aqui setando `ONNX_PROVIDER` antes de instanciar, nao passando
   um kwarg pro construtor.
2. Em `_create_audio`, quando o modelo ONNX usado e um export "novo" (input
   chamado `input_ids` em vez de `tokens` -- caso de varias conversoes da
   comunidade, incluindo exports do HuggingFace), a lib manda o input
   `speed` como `int32`, mas esses exports esperam `float32` -- um bug real
   da biblioteca (`onnxruntime.../onnxruntime_pybind11_state.InvalidArgument:
   Unexpected input data type. Actual: (tensor(int32)), expected: (tensor(float))`).
   Corrigimos isso com um monkeypatch minimo em `sess.run` que forca os
   dtypes corretos logo antes da sessao ONNX rodar de fato -- funciona
   independente de qual variante do modelo (input_ids ou tokens) foi
   baixada, e vira um no-op inofensivo se uma versao futura da lib corrigir
   o bug por conta propria.
3. O mesmo export tambem devolve o audio de saida com uma dimensao de lote
   extra (`shape (1, N)` em vez de `(N,)`). Isso quebra de duas formas:
   no nosso `playback.py` (corrigido achatando o array na fronteira do
   playback) e, mais insidioso, DENTRO da propria `Kokoro.create()` --
   quando uma sentenca longa e dividida em varios lotes de fonemas
   (`_split_phonemes`), a lib tenta `np.concatenate(audio_parts)` supondo
   arrays 1D; com arrays `(1, N_i)` de tamanhos `N_i` diferentes, isso
   quebra com `ValueError: all the input array dimensions except for the
   concatenation axis must match exactly`. Como isso acontece dentro do
   `create()` da lib, um `.reshape(-1)` no retorno (depois que ela ja
   quebrou) nao adianta -- por isso o mesmo monkeypatch de `sess.run` acima
   tambem achata a SAIDA logo apos cada chamada individual ao ONNX Runtime,
   antes da lib tentar concatenar os lotes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from kokoro_onnx import Kokoro

logger = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24000

# dtype esperado por input do grafo ONNX do Kokoro, independente da variante
# do export (`tokens`/`input_ids`) -- usado so para corrigir o input errado
# sem mexer nos demais.
_EXPECTED_INPUT_DTYPES = {
    "speed": np.float32,
    "style": np.float32,
}


def _patch_session_dtypes(session) -> None:
    """Envolve `session.run` para corrigir dtypes de input e a shape da saida.

    Ver docstring do modulo -- corrige dois bugs conhecidos da kokoro-onnx
    0.5.0 com exports "novos" (input `input_ids`): (1) `speed` enviado como
    int32 quando o grafo espera float32, e (2) a saida de audio vindo com
    uma dimensao de lote extra (`shape (1, N)`), que quebra a concatenacao
    interna da lib quando uma sentenca e dividida em varios lotes de
    fonemas. Achatar a saida AQUI, logo apos cada chamada individual ao
    ONNX Runtime, e o unico jeito de corrigir o segundo problema -- uma
    correcao so no valor de retorno de `Kokoro.create()` seria tarde demais,
    pois a lib ja teria quebrado tentando concatenar internamente.
    """
    original_run = session.run

    def patched_run(output_names, input_feed, run_options=None):
        for name, expected_dtype in _EXPECTED_INPUT_DTYPES.items():
            value = input_feed.get(name)
            if value is not None and getattr(value, "dtype", None) != expected_dtype:
                input_feed[name] = np.asarray(value, dtype=expected_dtype)

        outputs = original_run(output_names, input_feed, run_options)
        audio_out = outputs[0]
        if getattr(audio_out, "ndim", 1) > 1:
            outputs = [np.asarray(audio_out).reshape(-1), *outputs[1:]]
        return outputs

    session.run = patched_run


class KokoroEngine:
    def __init__(
        self,
        model_path: str = "models/kokoro-v1.0.onnx",
        voices_path: str = "models/voices-v1.0.bin",
        device: str = "cuda",
    ) -> None:
        # A lib nao aceita `providers` no construtor -- ela le essa variavel
        # de ambiente internamente (ver kokoro_onnx.Kokoro.__init__). Exige
        # o pacote `onnxruntime-gpu` instalado (nao so `onnxruntime`) para
        # CUDAExecutionProvider existir de fato -- ver docs/TROUBLESHOOTING.md.
        if device == "cuda":
            os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")
        else:
            os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"

        logger.info("Carregando Kokoro ONNX (device=%s)", device)
        self._kokoro = Kokoro(model_path, voices_path)
        _patch_session_dtypes(self._kokoro.sess)

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kokoro")

    def _synthesize_sync(self, text: str, voice: str, speed: float, lang: str) -> tuple[np.ndarray, int]:
        # Log de entrada/saida com timing: essencial pra diferenciar "esta
        # so lento" (CPU, sem onnxruntime-gpu) de "travou de vez" (ex: um
        # hang no backend do phonemizer/espeak-ng) -- sem isso, os dois
        # parecem identicos de fora (silencio, sem erro).
        started_at = time.monotonic()
        logger.info("Kokoro: sintetizando %r (voice=%s)", text, voice)
        samples, sample_rate = self._kokoro.create(text, voice=voice, speed=speed, lang=lang)
        elapsed_s = time.monotonic() - started_at
        # Alguns exports ONNX do Kokoro retornam o audio com uma dimensao de
        # batch (shape (1, N)) em vez de (N,) -- achatamos aqui para garantir
        # 1D, senao `len()` no array 2D conta o batch (=1) em vez das amostras
        # reais, e quebra o preenchimento do buffer de playback (ver
        # docs/TROUBLESHOOTING.md).
        flat = np.asarray(samples, dtype=np.float32).reshape(-1)
        audio_duration_s = len(flat) / sample_rate if sample_rate else 0.0
        logger.info(
            "Kokoro: sintese concluida em %.2fs (audio de %.2fs, RTF=%.2f)",
            elapsed_s,
            audio_duration_s,
            elapsed_s / audio_duration_s if audio_duration_s else float("inf"),
        )
        return flat, sample_rate

    async def synthesize(
        self, text: str, voice: str = "pf_dora", speed: float = 1.0, lang: str = "pt-br"
    ) -> tuple[np.ndarray, int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._synthesize_sync, text, voice, speed, lang)
