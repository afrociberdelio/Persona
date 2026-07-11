"""Agrega multiplos servidores MCP num unico registro de ferramentas.

O `LlamaClient.stream_chat(tools=...)` espera a lista de ferramentas no
formato OpenAI (`{"type": "function", "function": {...}}`); este registro
so precisa saber, para cada nome de ferramenta, qual conexao MCP despachar
a chamada -- e o unico "roteador" entre "o LLM pediu a ferramenta X" e
"qual processo MCP sabe responder X".
"""
from __future__ import annotations

import logging

from persona.llm.mcp_client import MCPServerConnection

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._connections: list[MCPServerConnection] = []
        self._tool_owner: dict[str, MCPServerConnection] = {}
        self._tool_schemas: list[dict] = []

    async def add_server(self, name: str, command: str, args: list[str] | None = None) -> None:
        conn = MCPServerConnection(name=name, command=command, args=args)
        await conn.connect()
        self._connections.append(conn)

        for schema in await conn.list_tools():
            tool_name = schema["function"]["name"]
            self._tool_owner[tool_name] = conn
            self._tool_schemas.append(schema)
            logger.info("Ferramenta '%s' registrada (servidor '%s')", tool_name, name)

    async def close(self) -> None:
        for conn in self._connections:
            await conn.close()

    def openai_tool_schemas(self) -> list[dict]:
        return list(self._tool_schemas)

    async def dispatch(self, tool_name: str, arguments: dict) -> str:
        conn = self._tool_owner.get(tool_name)
        if conn is None:
            raise ValueError(f"Ferramenta desconhecida: {tool_name}")
        return await conn.call_tool(tool_name, arguments)

    @property
    def is_empty(self) -> bool:
        return not self._tool_schemas
