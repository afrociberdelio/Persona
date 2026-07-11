"""Ponto de entrada do Persona.

Monta todos os servicos a partir de config/default.yaml (+ config/local.yaml
se existir) e entrega ao `DialogueOrchestrator`, que roda o loop principal
ate ser interrompido (Ctrl+C) ou morto pelo supervisor (scripts/supervisor.py).
"""
from __future__ import annotations

import asyncio
import logging

from persona.audio.capture import AudioCapture
from persona.audio.playback import AudioPlayback
from persona.audio.vad import SileroVADStream
from persona.config.loader import load_config
from persona.coordinator.orchestrator import DialogueOrchestrator
from persona.llm.llama_client import LlamaClient
from persona.memory.consolidator import MemoryConsolidator
from persona.memory.embedder import Embedder
from persona.memory.episodic_store import EpisodicStore
from persona.memory.profile_store import ProfileStore
from persona.memory.short_term import ShortTermMemory
from persona.stt.segmenter import UtteranceSegmenter
from persona.stt.whisper_engine import WhisperEngine
from persona.tools.registry import ToolRegistry
from persona.tts.kokoro_engine import KokoroEngine
from persona.tts.synth_queue import SynthesisQueue
from persona.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


async def _log_partial(text: str) -> None:
    logger.info("[preview STT, fala longa] %s", text)


async def build_orchestrator(cfg) -> DialogueOrchestrator:
    whisper = WhisperEngine(
        model_size=cfg.stt.model_size,
        device=cfg.stt.device,
        compute_type=cfg.stt.compute_type,
        language=cfg.stt.language,
    )
    segmenter = UtteranceSegmenter(
        whisper,
        sample_rate=cfg.audio.sample_rate,
        long_chunk_s=cfg.stt.long_utterance_chunk_s,
        overlap_s=cfg.stt.long_utterance_overlap_s,
        on_partial=_log_partial,
    )

    vad = SileroVADStream(
        sample_rate=cfg.audio.sample_rate,
        threshold=cfg.vad.threshold,
        min_speech_ms=cfg.vad.min_speech_ms,
        end_silence_ms=cfg.vad.end_silence_ms,
        hangover_ms=cfg.vad.hangover_ms,
    )

    capture = AudioCapture(
        sample_rate=cfg.audio.sample_rate,
        frame_ms=cfg.audio.frame_ms,
        device=cfg.audio.input_device,
    )
    playback = AudioPlayback(sample_rate=24000, device=cfg.audio.output_device)

    llm_client = LlamaClient(
        base_url=cfg.llm.base_url,
        model_name=cfg.llm.model_name,
        request_timeout_s=cfg.llm.request_timeout_s,
    )

    kokoro = KokoroEngine(
        model_path=cfg.tts.model_path, voices_path=cfg.tts.voices_path, device=cfg.tts.device
    )
    synth_queue = SynthesisQueue(kokoro, voice=cfg.tts.voice, speed=cfg.tts.speed)

    short_term = ShortTermMemory(max_turns=cfg.memory.short_term_max_turns)
    profile_store = ProfileStore(cfg.memory.sqlite_path)
    embedder = Embedder(model_name=cfg.memory.embedder_model)
    probe_dim_vector = await embedder.embed("probe")
    episodic_store = EpisodicStore(cfg.memory.lancedb_path, embed_dim=len(probe_dim_vector))
    consolidator = MemoryConsolidator(llm_client, profile_store, episodic_store, embedder)

    tool_registry = ToolRegistry()
    for server_cfg in getattr(cfg.tools, "mcp_servers", []) or []:
        await tool_registry.add_server(
            name=server_cfg["name"], command=server_cfg["command"], args=server_cfg.get("args", [])
        )

    return DialogueOrchestrator(
        capture=capture,
        playback=playback,
        vad=vad,
        segmenter=segmenter,
        llm_client=llm_client,
        synth_queue=synth_queue,
        short_term=short_term,
        profile_store=profile_store,
        episodic_store=episodic_store,
        embedder=embedder,
        consolidator=consolidator,
        tool_registry=tool_registry,
        llm_context_size=cfg.llm.context_size,
        llm_max_tokens=cfg.llm.max_tokens_per_turn,
        llm_temperature=cfg.llm.temperature,
        episodic_top_k=cfg.memory.episodic_top_k,
        idle_consolidation_delay_s=cfg.memory.idle_consolidation_delay_s,
    ), tool_registry, profile_store


async def _amain() -> None:
    cfg = load_config()
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.dir)

    logger.info("Verificando conexao com llama-server em %s ...", cfg.llm.base_url)
    orchestrator, tool_registry, profile_store = await build_orchestrator(cfg)

    try:
        await orchestrator.run()
    except asyncio.CancelledError:
        pass
    finally:
        await tool_registry.close()
        profile_store.close()


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        logger.info("Encerrado pelo usuario (Ctrl+C)")


if __name__ == "__main__":
    main()
