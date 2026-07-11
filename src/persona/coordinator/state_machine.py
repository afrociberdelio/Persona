"""Maquina de estados de turno: o mecanismo central do barge-in.

Design deliberado: uma tabela de transicao `dict[(TurnState, tipo_evento)] ->
handler`, e nao flags booleanas espalhadas pelos servicos. O mesmo evento
(ex: VADSpeechStart) tem significado diferente dependendo so do estado
atual -- inicia um turno novo se estiver IDLE, e um no-op se ja estiver
capturando, e um *barge-in* se a IA estiver pensando ou falando. Sem uma
tabela unica, cada servico acabaria com sua propria copia de "estou
falando?" e essas copias divergem sob concorrencia real.

Este modulo e puro (sem asyncio, sem I/O, sem bus) de proposito -- assim
da para testar toda transicao com testes unitarios triviais. Quem liga
isso a servicos de verdade e o `orchestrator.py`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto

from persona.bus.events import (
    Event,
    LLMDone,
    STTFinal,
    TTSAudioChunkReady,
    VADSpeechEnd,
    VADSpeechStart,
)


class TurnState(Enum):
    IDLE = auto()
    USER_SPEAKING = auto()
    TRANSCRIBING = auto()
    THINKING = auto()
    SPEAKING = auto()


class SideEffect(Enum):
    BEGIN_UTTERANCE = auto()
    FINALIZE_UTTERANCE = auto()
    DISPATCH_LLM = auto()
    PLAY_CHUNK = auto()
    START_IDLE_TIMER = auto()
    CANCEL_IDLE_TIMER = auto()
    # Os tres a seguir sao especificamente os efeitos de um barge-in.
    STOP_PLAYBACK = auto()
    CANCEL_LLM = auto()
    CANCEL_TTS = auto()


@dataclass(frozen=True)
class Transition:
    new_state: TurnState
    side_effects: tuple[SideEffect, ...]
    is_barge_in: bool = False


@dataclass(frozen=True)
class TransitionLogEntry:
    old_state: TurnState
    new_state: TurnState
    event_type: str
    side_effects: tuple[SideEffect, ...]
    is_barge_in: bool
    ts_monotonic: float = field(default_factory=time.monotonic)


BARGE_IN_EFFECTS: tuple[SideEffect, ...] = (
    SideEffect.STOP_PLAYBACK,
    SideEffect.CANCEL_LLM,
    SideEffect.CANCEL_TTS,
    SideEffect.BEGIN_UTTERANCE,
)

# playback_stopped e um evento sintetico publicado pelo servico de audio
# (nao um Event de dataclass do bus.events) quando a fila de playback
# esvazia naturalmente ao fim de uma resposta -- ver orchestrator.py.
PLAYBACK_NATURALLY_STOPPED = "playback_naturally_stopped"

# Disparado pelo orchestrator quando a transcricao final vem vazia (ruido/
# falso positivo do VAD) -- evita acionar o LLM para uma fala inexistente.
EMPTY_UTTERANCE = "empty_utterance"

_TABLE: dict[tuple[TurnState, type | str], Transition] = {
    (TurnState.IDLE, VADSpeechStart): Transition(
        TurnState.USER_SPEAKING, (SideEffect.BEGIN_UTTERANCE, SideEffect.CANCEL_IDLE_TIMER)
    ),
    (TurnState.USER_SPEAKING, VADSpeechStart): Transition(TurnState.USER_SPEAKING, ()),
    (TurnState.USER_SPEAKING, VADSpeechEnd): Transition(
        TurnState.TRANSCRIBING, (SideEffect.FINALIZE_UTTERANCE,)
    ),
    (TurnState.TRANSCRIBING, VADSpeechStart): Transition(TurnState.TRANSCRIBING, ()),
    (TurnState.TRANSCRIBING, STTFinal): Transition(TurnState.THINKING, (SideEffect.DISPATCH_LLM,)),
    (TurnState.TRANSCRIBING, EMPTY_UTTERANCE): Transition(TurnState.IDLE, (SideEffect.START_IDLE_TIMER,)),
    (TurnState.THINKING, VADSpeechStart): Transition(
        TurnState.USER_SPEAKING, BARGE_IN_EFFECTS, is_barge_in=True
    ),
    (TurnState.THINKING, TTSAudioChunkReady): Transition(
        TurnState.SPEAKING, (SideEffect.PLAY_CHUNK,)
    ),
    (TurnState.THINKING, LLMDone): Transition(TurnState.IDLE, (SideEffect.START_IDLE_TIMER,)),
    (TurnState.SPEAKING, VADSpeechStart): Transition(
        TurnState.USER_SPEAKING, BARGE_IN_EFFECTS, is_barge_in=True
    ),
    (TurnState.SPEAKING, TTSAudioChunkReady): Transition(TurnState.SPEAKING, (SideEffect.PLAY_CHUNK,)),
    (TurnState.SPEAKING, PLAYBACK_NATURALLY_STOPPED): Transition(
        TurnState.IDLE, (SideEffect.START_IDLE_TIMER,)
    ),
}


class TurnStateMachine:
    def __init__(self) -> None:
        self.state: TurnState = TurnState.IDLE
        self.history: list[TransitionLogEntry] = []

    def handle(self, event: Event | str) -> Transition:
        """Aplica o evento a tabela de transicao e atualiza o estado.

        Aceita tanto um Event tipado (bus.events) quanto uma string-marcador
        como PLAYBACK_NATURALLY_STOPPED, para eventos sinteticos que nao
        justificam uma dataclass propria.
        """
        key_type = type(event) if isinstance(event, Event) else event
        transition = _TABLE.get((self.state, key_type))

        if transition is None:
            # Evento nao mapeado para o estado atual: no-op deliberado.
            # Ex.: um VADSpeechEnd chegando enquanto ja estamos em
            # TRANSCRIBING (duplicado por causa de hangover) e ignorado
            # em vez de lançar excecao -- robustez > rigor aqui.
            transition = Transition(self.state, ())

        event_type_name = event if isinstance(event, str) else type(event).__name__
        entry = TransitionLogEntry(
            old_state=self.state,
            new_state=transition.new_state,
            event_type=event_type_name,
            side_effects=transition.side_effects,
            is_barge_in=transition.is_barge_in,
        )
        self.history.append(entry)
        self.state = transition.new_state
        return transition

    def last_barge_in_latency_s(self) -> float | None:
        """Tempo entre o VADSpeechStart que causou o ultimo barge-in e agora.

        Usado nos testes/logs para verificar a meta de <100ms do plano.
        """
        for entry in reversed(self.history):
            if entry.is_barge_in:
                return time.monotonic() - entry.ts_monotonic
        return None
