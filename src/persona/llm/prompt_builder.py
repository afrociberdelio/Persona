"""Monta a lista de mensagens enviada ao LLM a partir de memoria + turno atual.

Deliberadamente desacoplado dos stores de memoria reais (SQLite/LanceDB):
recebe strings/tuplas ja resolvidas, nao objetos de store. Isso mantem o
modulo puro e testavel sem precisar de banco de dados nos testes unitarios
(ver tests/unit/test_prompt_builder.py), e deixa quem monta o `PromptContext`
(o orchestrator/memory service) responsavel por ir buscar os dados.

Politica de corte quando estoura o orcamento de tokens (aproximado por
caracteres, ~4 chars/token em portugues): descarta primeiro os turnos mais
antigos de curto prazo; se ainda estourar, corta os trechos episodicos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass
class PromptContext:
    system_prompt: str
    user_utterance: str
    profile_summary: str | None = None
    episodic_snippets: list[str] = field(default_factory=list)
    short_term_turns: list[tuple[str, str]] = field(default_factory=list)  # (role, content), mais antigo primeiro
    max_tokens: int = 6144


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


def _memory_block(profile_summary: str | None, episodic_snippets: list[str]) -> str | None:
    parts = []
    if profile_summary:
        parts.append(f"O que voce sabe sobre o usuario:\n{profile_summary}")
    if episodic_snippets:
        joined = "\n".join(f"- {s}" for s in episodic_snippets)
        parts.append(f"Momentos relevantes de conversas anteriores:\n{joined}")
    if not parts:
        return None
    return "\n\n".join(parts)


def build_messages(ctx: PromptContext) -> list[dict[str, str]]:
    episodic = list(ctx.episodic_snippets)
    short_term = list(ctx.short_term_turns)

    def total_tokens() -> int:
        memory_block = _memory_block(ctx.profile_summary, episodic)
        text = ctx.system_prompt + (memory_block or "") + ctx.user_utterance
        text += "".join(content for _, content in short_term)
        return _estimate_tokens(text)

    # Corta turnos de curto prazo mais antigos primeiro.
    while total_tokens() > ctx.max_tokens and short_term:
        short_term.pop(0)

    # Se ainda estourar, corta trechos episodicos (menos relevantes primeiro,
    # assumindo que a lista ja vem ordenada por relevancia decrescente).
    while total_tokens() > ctx.max_tokens and episodic:
        episodic.pop()

    messages: list[dict[str, str]] = []

    system_content = ctx.system_prompt
    memory_block = _memory_block(ctx.profile_summary, episodic)
    if memory_block:
        system_content = f"{system_content}\n\n{memory_block}"
    messages.append({"role": "system", "content": system_content})

    for role, content in short_term:
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": ctx.user_utterance})
    return messages
