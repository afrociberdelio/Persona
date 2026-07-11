"""Consolidacao em background: turnos recentes -> fatos estruturados + resumo episodico.

Disparado pelo orchestrator via um idle-timer (so depois de alguns segundos
sem fala nova, para nunca competir por um slot do llama-server com um turno
de conversa ao vivo -- por isso a recomendacao de `--parallel 2` no
llama-server, um slot para conversa e outro para isto).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from persona.llm.llama_client import LlamaClient
from persona.memory.embedder import Embedder
from persona.memory.episodic_store import EpisodicStore
from persona.memory.profile_store import ProfileStore

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """\
Voce analisa uma conversa recente entre um usuario e um assistente e extrai:
1. "facts": um dicionario de fatos estaveis sobre o usuario que valem a pena lembrar
   (nome, preferencias, projetos, rotina, habitos, objetivos). So inclua o que foi
   dito explicitamente ou inferido com alta confianca. Chaves curtas em snake_case.
2. "episodic_summary": um resumo de 1-2 frases do que foi conversado, util para
   recuperar esse momento depois.

Se nao houver nada relevante para lembrar, retorne {"facts": {}, "episodic_summary": null}.
Responda APENAS com JSON valido, sem explicacao, sem markdown."""


@dataclass
class ConsolidationResult:
    facts_upserted: int
    episodic_entries_added: int


def _format_turns(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{role}: {content}" for role, content in turns)


def _parse_json_response(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json") :]
    return json.loads(text.strip())


class MemoryConsolidator:
    def __init__(
        self,
        llm_client: LlamaClient,
        profile_store: ProfileStore,
        episodic_store: EpisodicStore,
        embedder: Embedder,
    ) -> None:
        self._llm = llm_client
        self._profile = profile_store
        self._episodic = episodic_store
        self._embedder = embedder

    async def consolidate(self, turns: list[tuple[str, str]]) -> ConsolidationResult:
        if not turns:
            return ConsolidationResult(0, 0)

        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": _format_turns(turns)},
        ]

        raw_text = ""
        try:
            async for token in self._llm.stream_chat(messages, max_tokens=400, temperature=0.1):
                raw_text += token
        except Exception:
            logger.exception("Falha ao chamar o LLM para consolidacao de memoria")
            return ConsolidationResult(0, 0)

        try:
            parsed = _parse_json_response(raw_text)
        except json.JSONDecodeError:
            logger.warning("Resposta de consolidacao nao era JSON valido: %r", raw_text)
            return ConsolidationResult(0, 0)

        facts: dict = parsed.get("facts") or {}
        for key, value in facts.items():
            self._profile.upsert(str(key), str(value))

        episodic_added = 0
        summary = parsed.get("episodic_summary")
        if summary:
            import datetime

            vector = await self._embedder.embed(summary)
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._episodic.add(summary, vector, ts)
            episodic_added = 1

        logger.info("Consolidacao: %d fatos, %d entradas episodicas", len(facts), episodic_added)
        return ConsolidationResult(facts_upserted=len(facts), episodic_entries_added=episodic_added)
