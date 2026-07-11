"""Servidor MCP de demonstracao: uma unica ferramenta trivial (get_current_time).

Proposito exclusivo: provar que o caminho ponta a ponta do scaffold MCP
funciona (LLM emite tool_call -> orchestrator despacha via MCP -> resultado
volta -> incorporado na resposta falada). Um scaffold vazio nao e
verificavel; por isso existe essa ferramenta minima.

Roda como processo separado, falado via stdio (padrao MCP), lancado pelo
`ToolRegistry` com o comando `python -m persona.tools.demo_time_tool`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("persona-demo-tools")


@mcp.tool()
def get_current_time(timezone_name: str = "America/Sao_Paulo") -> str:
    """Retorna a data e hora atuais num fuso horario informado (IANA tz name)."""
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        # No Windows, zoneinfo so encontra fusos IANA se o pacote `tzdata`
        # estiver instalado -- ja adicionado as dependencias do projeto.
        logger.warning("Fuso horario '%s' nao encontrado; usando UTC", timezone_name)
        tz = timezone.utc
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


if __name__ == "__main__":
    mcp.run(transport="stdio")
