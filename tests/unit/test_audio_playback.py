"""Regressao para um bug real em AudioPlayback: `mark_no_more_chunks()` e
chamado assim que o LLM termina de gerar TEXTO, o que costuma acontecer
BEM antes da sintese de TTS de cada sentenca (assincrona, em paralelo)
realmente terminar. Sem rastrear se algum audio ja foi enfileirado, o
callback via o buffer vazio + no_more_chunks=True e concluia (errado) que o
turno "terminou naturalmente" -- resetando o turno atual pra None antes de
qualquer audio de verdade chegar. Todo `enqueue()` seguinte pra esse turno
era descartado silenciosamente (turn_id nao batia mais), sem erro nenhum.

Testado direto contra `AudioPlayback._callback`, sem precisar de stream de
audio real (nao chama `.start()`) -- o callback e so um metodo comum.
"""
from __future__ import annotations

import numpy as np

from persona.audio.playback import AudioPlayback


def _run_callback(pb: AudioPlayback, frames: int = 256) -> np.ndarray:
    outdata = np.zeros((frames, 1), dtype=np.float32)
    pb._callback(outdata, frames, None, None)
    return outdata[:, 0]


def test_natural_stop_does_not_fire_before_any_audio_was_ever_enqueued():
    pb = AudioPlayback(sample_rate=24000)
    pb.start_turn("t1")

    # LLM termina de gerar o texto rapido e sinaliza "nao vem mais nada" --
    # mas nenhuma sentenca terminou de sintetizar ainda.
    pb.mark_no_more_chunks("t1")
    _run_callback(pb)

    assert pb.is_speaking, "nao deveria ter 'terminado naturalmente' sem nunca ter tocado nada"


def test_audio_enqueued_after_mark_no_more_chunks_still_plays():
    pb = AudioPlayback(sample_rate=24000)
    pb.start_turn("t1")
    pb.mark_no_more_chunks("t1")
    _run_callback(pb)  # buffer vazio -- nao deve "fechar" o turno (ver teste acima)

    # Agora a sintese da sentenca termina e o audio real chega.
    pcm = np.full(100, 0.5, dtype=np.float32)
    pb.enqueue(pcm, "t1")
    assert pb.is_speaking

    played = _run_callback(pb)
    assert np.allclose(played[:100], 0.5)


def test_natural_stop_fires_after_real_audio_actually_drains():
    pb = AudioPlayback(sample_rate=24000)
    pb.start_turn("t1")
    pb.enqueue(np.full(50, 0.3, dtype=np.float32), "t1")
    pb.mark_no_more_chunks("t1")

    # As 50 amostras cabem inteiras num callback de 256 frames -- o buffer
    # esvazia e o turno fecha na mesma chamada (ja que no_more_chunks e
    # any_chunk_enqueued ja estavam True antes dela).
    _run_callback(pb)
    assert not pb.is_speaking


def test_natural_stop_waits_until_buffer_actually_empties_across_callbacks():
    pb = AudioPlayback(sample_rate=24000)
    pb.start_turn("t1")
    pb.enqueue(np.full(50, 0.3, dtype=np.float32), "t1")
    pb.mark_no_more_chunks("t1")

    # Um callback pequeno (menos frames que as amostras enfileiradas) so
    # consome parte do buffer -- o turno nao pode fechar antes de esvaziar
    # de fato.
    _run_callback(pb, frames=20)
    assert pb.is_speaking

    _run_callback(pb, frames=20)
    assert pb.is_speaking  # ainda sobram 10 amostras

    _run_callback(pb, frames=20)
    assert not pb.is_speaking  # esvaziou de vez -- so agora fecha o turno


def test_stale_turn_chunk_is_dropped():
    pb = AudioPlayback(sample_rate=24000)
    pb.start_turn("t1")
    pb.enqueue(np.full(50, 0.9, dtype=np.float32), "turno_antigo_obsoleto")

    played = _run_callback(pb)
    assert np.allclose(played, 0.0)  # descartado -- turn_id nao bate com o turno atual
