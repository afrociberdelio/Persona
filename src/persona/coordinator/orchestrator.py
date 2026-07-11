"""Conecta o bus, a FSM e todos os servicos concretos.

`TurnStateMachine` (state_machine.py) e pura: decide qual estado vem a
seguir e *quais efeitos* devem acontecer, mas nao sabe como executa-los.
Este modulo e o unico lugar que traduz um `SideEffect` simbolico numa acao
real (cancelar uma task, tocar um chunk de audio, chamar o LLM). Manter essa
separacao e o que deixa a maquina de estados testável sem mockar audio/GPU.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncIterator

import numpy as np

from persona.audio.capture import AudioCapture
from persona.audio.playback import AudioPlayback
from persona.audio.vad import SileroVADStream
from persona.bus.event_bus import EventBus
from persona.bus.events import (
    LLMCancelled,
    LLMDone,
    STTFinal,
    TTSAudioChunkReady,
    TurnStateChanged,
    VADSpeechEnd,
    VADSpeechStart,
)
from persona.coordinator.state_machine import (
    EMPTY_UTTERANCE,
    PLAYBACK_NATURALLY_STOPPED,
    SideEffect,
    TurnState,
    TurnStateMachine,
)
from persona.llm.llama_client import LlamaClient
from persona.llm.prompt_builder import PromptContext, build_messages
from persona.llm.sentence_chunker import SentenceChunker
from persona.memory.consolidator import MemoryConsolidator
from persona.memory.embedder import Embedder
from persona.memory.episodic_store import EpisodicStore
from persona.memory.profile_store import ProfileStore
from persona.memory.short_term import ShortTermMemory
from persona.stt.segmenter import UtteranceSegmenter
from persona.tools.registry import ToolRegistry
from persona.tts.synth_queue import SynthesisQueue
from persona.utils.timing import TurnTimer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Voce e Persona, um assistente de voz local, prestativo e conciso. "
    "Responda em portugues do Brasil de forma natural, como numa conversa falada -- "
    "frases curtas, sem listas ou markdown, ja que sua resposta sera sintetizada em audio."
)


async def _single_shot_stream(text: str) -> AsyncIterator[str]:
    """Envolve uma string pronta (resposta pos tool-call) no mesmo formato de
    async generator do streaming normal, para reusar o mesmo caminho de
    sentence-chunking em ambos os casos."""
    yield text


class DialogueOrchestrator:
    def __init__(
        self,
        *,
        capture: AudioCapture,
        playback: AudioPlayback,
        vad: SileroVADStream,
        segmenter: UtteranceSegmenter,
        llm_client: LlamaClient,
        synth_queue: SynthesisQueue,
        short_term: ShortTermMemory,
        profile_store: ProfileStore,
        episodic_store: EpisodicStore,
        embedder: Embedder,
        consolidator: MemoryConsolidator,
        tool_registry: ToolRegistry | None = None,
        llm_context_size: int = 6144,
        llm_max_tokens: int = 512,
        llm_temperature: float = 0.7,
        episodic_top_k: int = 4,
        idle_consolidation_delay_s: float = 8.0,
    ) -> None:
        self._capture = capture
        self._playback = playback
        self._vad = vad
        self._segmenter = segmenter
        self._llm_client = llm_client
        self._synth_queue = synth_queue
        self._short_term = short_term
        self._profile_store = profile_store
        self._episodic_store = episodic_store
        self._embedder = embedder
        self._consolidator = consolidator
        self._tool_registry = tool_registry

        self._llm_context_size = llm_context_size
        self._llm_max_tokens = llm_max_tokens
        self._llm_temperature = llm_temperature
        self._episodic_top_k = episodic_top_k
        self._idle_consolidation_delay_s = idle_consolidation_delay_s

        self._bus = EventBus()
        self._fsm = TurnStateMachine()

        self._turn_id: str | None = None
        self._llm_task: asyncio.Task | None = None
        self._idle_timer_task: asyncio.Task | None = None

        self._playback.set_on_naturally_stopped(self._handle_playback_drained)

    async def run(self) -> None:
        self._capture.start()
        self._playback.start()
        logger.info("Orchestrator no ar. Estado inicial: %s", self._fsm.state.name)

        async for frame in self._capture.frames():
            vad_event = self._vad.process(frame)

            if vad_event is not None:
                event = VADSpeechStart() if vad_event.kind == "start" else VADSpeechEnd(
                    duration_s=vad_event.duration_s or 0.0
                )
                await self._apply_transition(event)

            if self._fsm.state == TurnState.USER_SPEAKING:
                self._segmenter.add_frame(frame)

    async def _apply_transition(self, event) -> None:
        old_state = self._fsm.state
        transition = self._fsm.handle(event)

        self._bus.publish(
            TurnStateChanged(
                old_state=old_state.name,
                new_state=transition.new_state.name,
                reason=event if isinstance(event, str) else type(event).__name__,
            )
        )
        if transition.is_barge_in:
            logger.info("BARGE-IN: %s -> %s", old_state.name, transition.new_state.name)

        for effect in transition.side_effects:
            await self._run_side_effect(effect, event)

    async def _run_side_effect(self, effect: SideEffect, event) -> None:
        if effect is SideEffect.BEGIN_UTTERANCE:
            self._segmenter.start_utterance()

        elif effect is SideEffect.CANCEL_IDLE_TIMER:
            self._cancel_idle_timer()

        elif effect is SideEffect.FINALIZE_UTTERANCE:
            asyncio.ensure_future(self._finalize_utterance())

        elif effect is SideEffect.DISPATCH_LLM:
            assert isinstance(event, STTFinal)
            self._turn_id = str(uuid.uuid4())
            self._playback.start_turn(self._turn_id)
            self._synth_queue.start_turn(self._turn_id)
            self._llm_task = asyncio.ensure_future(self._run_llm_turn(self._turn_id, event.text))

        elif effect is SideEffect.PLAY_CHUNK:
            assert isinstance(event, TTSAudioChunkReady)
            self._playback.enqueue(event.pcm, event.turn_id)

        elif effect is SideEffect.STOP_PLAYBACK:
            self._playback.stop_immediately(reason="barge_in")

        elif effect is SideEffect.CANCEL_LLM:
            if self._llm_task is not None and not self._llm_task.done():
                self._llm_task.cancel()

        elif effect is SideEffect.CANCEL_TTS:
            if self._turn_id is not None:
                self._synth_queue.cancel_turn(self._turn_id, reason="barge_in")

        elif effect is SideEffect.START_IDLE_TIMER:
            self._start_idle_timer()

    async def _finalize_utterance(self) -> None:
        text, duration_s = await self._segmenter.finalize()
        text = text.strip()
        if not text:
            await self._apply_transition(EMPTY_UTTERANCE)
            return
        await self._apply_transition(
            STTFinal(utterance_id=str(uuid.uuid4()), text=text, duration_s=duration_s)
        )

    async def _run_llm_turn(self, turn_id: str, user_text: str) -> None:
        timer = TurnTimer(turn_id)
        timer.mark("stt_final")

        profile_summary = self._profile_store.summary()
        query_vector = await self._embedder.embed(user_text)
        episodic_snippets = self._episodic_store.search(query_vector, top_k=self._episodic_top_k)

        ctx = PromptContext(
            system_prompt=_SYSTEM_PROMPT,
            user_utterance=user_text,
            profile_summary=profile_summary,
            episodic_snippets=episodic_snippets,
            short_term_turns=self._short_term.get_turns(),
            max_tokens=self._llm_context_size,
        )
        messages = build_messages(ctx)

        tools_schema = None
        if self._tool_registry is not None and not self._tool_registry.is_empty:
            tools_schema = self._tool_registry.openai_tool_schemas()

        try:
            token_source = await self._resolve_token_source(messages, tools_schema)

            chunker = SentenceChunker()
            sentence_id = 0
            full_text = ""

            timer.mark("llm_stream_start")
            async for token in token_source:
                if sentence_id == 0 and not full_text:
                    timer.mark("llm_first_token")
                full_text += token
                for sentence in chunker.feed(token):
                    sentence_id += 1
                    self._synth_queue.submit(turn_id, sentence_id, sentence, self._on_tts_chunk_ready)

            remainder = chunker.flush()
            if remainder:
                sentence_id += 1
                self._synth_queue.submit(turn_id, sentence_id, remainder, self._on_tts_chunk_ready)

        except asyncio.CancelledError:
            self._bus.publish(LLMCancelled(turn_id=turn_id, reason="barge_in"))
            raise

        timer.mark("llm_done")
        self._short_term.add("user", user_text)
        self._short_term.add("assistant", full_text)
        self._playback.mark_no_more_chunks(turn_id)
        self._bus.publish(LLMDone(turn_id=turn_id, full_text=full_text))
        await self._apply_transition(LLMDone(turn_id=turn_id, full_text=full_text))

    async def _resolve_token_source(
        self, messages: list[dict], tools_schema: list[dict] | None
    ) -> AsyncIterator[str]:
        """Decide entre o caminho de streaming direto e o caminho de tool-calling.

        Com ferramentas disponiveis, a primeira chamada e nao-streaming (para
        poder inspecionar `tool_calls` antes de decidir o que fazer). Se o
        modelo pediu uma ferramenta, ela e despachada via MCP e uma segunda
        chamada -- essa sim em streaming -- gera a resposta falada final.
        Sem ferramentas (config default da Fase 1), vai direto pro streaming.
        """
        if not tools_schema:
            return self._llm_client.stream_chat(
                messages, max_tokens=self._llm_max_tokens, temperature=self._llm_temperature
            )

        first = await self._llm_client.chat_once(
            messages, max_tokens=self._llm_max_tokens, temperature=self._llm_temperature, tools=tools_schema
        )
        tool_calls = first.get("tool_calls")
        if not tool_calls:
            return _single_shot_stream(first.get("content") or "")

        messages.append(first)
        for call in tool_calls:
            fn = call["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
                result_text = await self._tool_registry.dispatch(fn["name"], args)
            except Exception as exc:
                logger.exception("Falha ao executar ferramenta '%s'", fn.get("name"))
                result_text = f"(erro ao executar ferramenta: {exc})"
            messages.append(
                {"role": "tool", "tool_call_id": call.get("id", ""), "content": result_text}
            )

        return self._llm_client.stream_chat(
            messages, max_tokens=self._llm_max_tokens, temperature=self._llm_temperature
        )

    def _on_tts_chunk_ready(self, turn_id: str, sentence_id: int, pcm: np.ndarray, sample_rate: int) -> None:
        event = TTSAudioChunkReady(turn_id=turn_id, sentence_id=sentence_id, pcm=pcm, sample_rate=sample_rate)
        asyncio.ensure_future(self._apply_transition(event))

    def _start_idle_timer(self) -> None:
        self._cancel_idle_timer()
        self._idle_timer_task = asyncio.ensure_future(self._idle_timer_coro())

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer_task is not None and not self._idle_timer_task.done():
            self._idle_timer_task.cancel()

    async def _idle_timer_coro(self) -> None:
        try:
            await asyncio.sleep(self._idle_consolidation_delay_s)
        except asyncio.CancelledError:
            return
        turns = self._short_term.get_turns()
        if not turns:
            return
        try:
            await self._consolidator.consolidate(turns)
        except Exception:
            logger.exception("Falha na consolidacao de memoria em background")

    def _handle_playback_drained(self, turn_id: str) -> None:
        asyncio.ensure_future(self._apply_transition(PLAYBACK_NATURALLY_STOPPED))
