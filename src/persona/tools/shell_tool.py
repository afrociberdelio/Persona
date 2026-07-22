"""Servidor MCP de terminal com confirmacao obrigatoria antes de executar.

Confirmacao e feita com DUAS ferramentas MCP em vez de uma:
`propose_shell_command` so registra o comando pendente e devolve uma
mensagem pro LLM repassar por voz -- nao executa nada. `confirm_shell_command`
so executa se o token bater com uma proposta ainda valida (nao expirada).

A propriedade de seguranca real, de graca por causa da arquitetura: como
ferramentas MCP so sao despachadas dentro de `_run_llm_turn`
(persona/coordinator/orchestrator.py), e um turno novo so comeca depois de
uma fala nova de verdade chegar pelo microfone (STTFinal), o modelo NUNCA
consegue chamar `confirm_shell_command` no mesmo turno em que chamou
`propose_shell_command` -- precisa passar por uma rodada real de audio novo
do usuario entre as duas chamadas. Isso nao e uma trava de seguranca
absoluta (depende do LLM interpretar certo o que conta como confirmacao),
mas garante que nao tem como o modelo "se auto-confirmar" sem uma fala nova
de verdade vinda do microfone.
"""
from __future__ import annotations

import logging
import secrets
import subprocess
import time

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

PENDING_TTL_S = 120
COMMAND_TIMEOUT_S = 30
MAX_OUTPUT_CHARS = 2000


class PendingCommands:
    """Logica pura de propose/confirm/expiracao -- separada do FastMCP e do
    subprocess de verdade pra ser testavel sem executar nada real."""

    def __init__(self, ttl_s: float = PENDING_TTL_S) -> None:
        self._ttl_s = ttl_s
        self._pending: dict[str, tuple[str, float]] = {}

    def propose(self, command: str) -> str:
        token = secrets.token_hex(4)
        self._pending[token] = (command, time.monotonic())
        return token

    def confirm(self, token: str) -> str | None:
        """Retorna o comando pendente se o token for valido e nao expirado,
        removendo-o (uso unico -- nao da pra confirmar o mesmo token duas
        vezes). Retorna None se invalido/expirado."""
        entry = self._pending.pop(token, None)
        if entry is None:
            return None
        command, created_at = entry
        if time.monotonic() - created_at > self._ttl_s:
            return None
        return command

    def pending_count(self) -> int:
        return len(self._pending)


_pending_commands = PendingCommands()

mcp = FastMCP("persona-shell")


@mcp.tool()
def propose_shell_command(command: str) -> str:
    """Propoe um comando de shell/PowerShell para execucao no PC do usuario.
    NAO executa nada ainda. Use sempre que o usuario pedir para rodar algo
    no terminal. Depois de chamar isso, informe o usuario o que o comando
    faz e pergunte se ele confirma -- so chame confirm_shell_command depois
    que ele responder afirmativamente numa fala nova."""
    token = _pending_commands.propose(command)
    logger.info("Comando proposto (token=%s): %s", token, command)
    return (
        f"Comando pendente de confirmacao (token={token}): {command}\n"
        "Peca confirmacao explicita ao usuario antes de chamar confirm_shell_command."
    )


@mcp.tool()
def confirm_shell_command(token: str) -> str:
    """Executa um comando previamente proposto via propose_shell_command,
    depois de confirmacao explicita do usuario. So chame isso se o usuario
    realmente confirmou (disse sim/pode/confirmo etc) em resposta a
    proposta."""
    command = _pending_commands.confirm(token)
    if command is None:
        return (
            "Token invalido ou expirado (confirmacoes valem por "
            f"{PENDING_TTL_S}s). Proponha o comando de novo."
        )

    logger.info("Executando comando confirmado (token=%s): %s", token, command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"Comando excedeu o tempo limite de {COMMAND_TIMEOUT_S}s e foi interrompido."

    output = ((result.stdout or "") + (result.stderr or ""))[:MAX_OUTPUT_CHARS]
    return f"Comando executado (codigo de saida {result.returncode}):\n{output}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
