"""Cliente MCP (Model Context Protocol) minimo, via stdio.

Cada servidor MCP roda como processo filho separado, falando o protocolo
MCP por stdio -- e assim que o MCP padroniza integracao de ferramentas
sem cada uma precisar de um adaptador proprio. Este cliente so abre a
conexao, lista as ferramentas disponiveis e despacha chamadas; a decisao
de *quando* chamar uma ferramenta e do LLM (via tool-calling nativo do
Qwen3), nao deste modulo.
"""
from __future__ import annotations

import logging
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPServerConnection:
    def __init__(self, name: str, command: str, args: list[str] | None = None) -> None:
        self.name = name
        self._params = StdioServerParameters(command=command, args=args or [])
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        read, write = await self._stack.enter_async_context(stdio_client(self._params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        logger.info("Conectado ao servidor MCP '%s'", self.name)

    async def close(self) -> None:
        await self._stack.aclose()

    async def list_tools(self) -> list[dict]:
        assert self._session is not None
        result = await self._session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in result.tools
        ]

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        assert self._session is not None
        result = await self._session.call_tool(tool_name, arguments)
        text_parts = [block.text for block in result.content if hasattr(block, "text")]
        return "\n".join(text_parts) if text_parts else str(result.content)
