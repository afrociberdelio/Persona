"""Dataclasses tipadas de evento que trafegam pelo EventBus.

Cada evento carrega seu proprio `ts_monotonic` (time.monotonic()) para permitir
medir latencia entre estagios do pipeline sem depender de relogio de parede.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def _now() -> float:
    return time.monotonic()


@dataclass(frozen=True)
class Event:
    ts_monotonic: float = field(default_factory=_now, init=False)


@dataclass(frozen=True)
class VADSpeechStart(Event):
    pass


@dataclass(frozen=True)
class VADSpeechEnd(Event):
    duration_s: float


@dataclass(frozen=True)
class STTPartial(Event):
    utterance_id: str
    text: str


@dataclass(frozen=True)
class STTFinal(Event):
    utterance_id: str
    text: str
    duration_s: float
    language: str | None = None


@dataclass(frozen=True)
class LLMToken(Event):
    turn_id: str
    token: str


@dataclass(frozen=True)
class LLMSentenceReady(Event):
    turn_id: str
    sentence_id: int
    text: str


@dataclass(frozen=True)
class LLMDone(Event):
    turn_id: str
    full_text: str


@dataclass(frozen=True)
class LLMCancelled(Event):
    turn_id: str
    reason: str


@dataclass(frozen=True)
class TTSAudioChunkReady(Event):
    turn_id: str
    sentence_id: int
    pcm: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class TTSDone(Event):
    turn_id: str


@dataclass(frozen=True)
class TTSCancelled(Event):
    turn_id: str
    reason: str


@dataclass(frozen=True)
class PlaybackStarted(Event):
    turn_id: str


@dataclass(frozen=True)
class PlaybackStopped(Event):
    turn_id: str
    reason: str


@dataclass(frozen=True)
class TurnStateChanged(Event):
    old_state: str
    new_state: str
    reason: str


@dataclass(frozen=True)
class MemoryConsolidationDone(Event):
    facts_upserted: int
    episodic_entries_added: int
