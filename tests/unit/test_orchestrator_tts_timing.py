"""Regressao para um bug real: a FSM voltava de THINKING pra IDLE assim que
o LLM terminava de gerar TEXTO, mesmo que a sintese de TTS das sentencas
ainda estivesse rodando em paralelo (o caso comum -- respostas curtas
terminam de gerar rapido, mas o Kokoro ainda leva algumas centenas de ms
por sentenca). Uma vez em IDLE, todo `TTSAudioChunkReady` que chegava
depois caia num estado sem transicao mapeada e era descartado
silenciosamente -- sem erro, sem log -- explicando "o texto aparece mas
nenhum audio sai".

Usa dublês simples em vez dos motores reais (Whisper/Kokoro/llama-server)
porque o bug é de orquestração/timing entre LLM e TTS, não de nenhum motor
específico -- não precisa de GPU nem de rede pra reproduzir.
"""
from __future__ import annotations

import numpy as np
import pytest

from persona.bus.events import STTFinal
from persona.coordinator.orchestrator import DialogueOrchestrator
from persona.coordinator.state_machine import TurnState
from persona.memory.short_term import ShortTermMemory


class FakeLlamaClient:
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens

    async def stream_chat(self, messages, max_tokens=512, temperature=0.7, tools=None, **kwargs):
        for token in self._tokens:
            yield token


class RaisingLlamaClient:
    """Simula uma falha de rede/HTTP (ex: 400 do llama-server) durante a
    geracao -- usado pra provar que isso nao deixa a FSM presa."""

    async def stream_chat(self, messages, max_tokens=512, temperature=0.7, tools=None, **kwargs):
        raise RuntimeError("simulated llama-server error")
        yield  # inalcancavel -- so pra manter a funcao um async generator


class FakeSynthesisQueue:
    """`submit` só enfileira -- quem decide quando a síntese "termina" é o
    teste, chamando `complete_all()` explicitamente. Isso simula o caso
    real onde a síntese ainda está rodando quando o streaming de texto já
    acabou."""

    def __init__(self) -> None:
        self.pending: list[tuple[str, int, callable]] = []

    def start_turn(self, turn_id: str) -> None:
        pass

    def cancel_turn(self, turn_id: str, reason: str = "barge_in") -> None:
        self.pending = [p for p in self.pending if p[0] != turn_id]

    def submit(self, turn_id: str, sentence_id: int, text: str, on_ready) -> None:
        self.pending.append((turn_id, sentence_id, on_ready))

    def complete_all(self) -> None:
        pending, self.pending = self.pending, []
        for turn_id, sentence_id, on_ready in pending:
            on_ready(turn_id, sentence_id, np.zeros(10, dtype=np.float32), 24000)


class FakePlayback:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    def set_on_naturally_stopped(self, callback) -> None:
        pass

    def start_turn(self, turn_id: str) -> None:
        pass

    def enqueue(self, pcm, turn_id: str) -> None:
        self.enqueued.append(turn_id)

    def mark_no_more_chunks(self, turn_id: str) -> None:
        pass

    def stop_immediately(self, reason: str) -> None:
        pass


class FakeProfileStore:
    def summary(self):
        return None


class FakeEpisodicStore:
    def search(self, vector, top_k=4):
        return []


class FakeEmbedder:
    async def embed(self, text):
        return np.zeros(4, dtype=np.float32)


def _build_orchestrator(llm_tokens: list[str]) -> tuple[DialogueOrchestrator, FakeSynthesisQueue, FakePlayback]:
    synth_queue = FakeSynthesisQueue()
    playback = FakePlayback()
    orchestrator = DialogueOrchestrator(
        capture=None,
        playback=playback,
        vad=None,
        segmenter=None,
        llm_client=FakeLlamaClient(llm_tokens),
        synth_queue=synth_queue,
        short_term=ShortTermMemory(),
        profile_store=FakeProfileStore(),
        episodic_store=FakeEpisodicStore(),
        embedder=FakeEmbedder(),
        consolidator=None,
        tool_registry=None,
    )
    return orchestrator, synth_queue, playback


@pytest.mark.asyncio
async def test_fsm_stays_in_thinking_until_tts_actually_completes():
    orchestrator, synth_queue, playback = _build_orchestrator(["Oi! ", "Tudo bem?"])

    # STTFinal so tem transicao mapeada a partir de TRANSCRIBING -- pula
    # direto pra la (o caminho USER_SPEAKING/VAD ja e coberto pelos testes
    # puros de state_machine.py; aqui o foco e o timing LLM/TTS).
    orchestrator._fsm.state = TurnState.TRANSCRIBING
    await orchestrator._apply_transition(STTFinal(utterance_id="u1", text="oi", duration_s=1.0))
    await orchestrator._llm_task  # espera _run_llm_turn terminar de gerar o texto

    # O texto ja terminou de gerar, mas a sintese TTS ainda nao foi
    # "concluida" (FakeSynthesisQueue.submit so enfileira). A FSM NAO pode
    # ter voltado pra IDLE aqui -- esse era exatamente o bug.
    assert orchestrator._fsm.state is TurnState.THINKING
    assert playback.enqueued == []

    synth_queue.complete_all()
    await _yield()

    assert orchestrator._fsm.state is TurnState.SPEAKING
    assert len(playback.enqueued) == 2


@pytest.mark.asyncio
async def test_empty_response_still_returns_to_idle():
    """Caso oposto: se o LLM nao gerar nenhuma sentenca de verdade, a FSM
    precisa voltar pra IDLE sozinha (nao ha TTSAudioChunkReady vindo por ai
    pra fazer isso por ela)."""
    orchestrator, _synth_queue, playback = _build_orchestrator([])

    orchestrator._fsm.state = TurnState.TRANSCRIBING
    await orchestrator._apply_transition(STTFinal(utterance_id="u1", text="oi", duration_s=1.0))
    await orchestrator._llm_task

    assert orchestrator._fsm.state is TurnState.IDLE
    assert playback.enqueued == []


@pytest.mark.asyncio
async def test_llm_error_returns_to_idle_instead_of_getting_stuck():
    """Regressao: um erro generico durante a geracao (ex: 400 do
    llama-server por causa de um schema de ferramenta MCP, timeout de
    rede) nao pode deixar a FSM presa em THINKING pra sempre -- antes desta
    correcao, so um barge-in manual do usuario destravava."""
    orchestrator, _synth_queue, playback = _build_orchestrator([])
    orchestrator._llm_client = RaisingLlamaClient()

    orchestrator._fsm.state = TurnState.TRANSCRIBING
    await orchestrator._apply_transition(STTFinal(utterance_id="u1", text="oi", duration_s=1.0))
    await orchestrator._llm_task

    assert orchestrator._fsm.state is TurnState.IDLE


async def _yield() -> None:
    import asyncio

    await asyncio.sleep(0)
