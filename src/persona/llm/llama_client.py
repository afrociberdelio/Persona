"""Cliente async/streaming para o endpoint OpenAI-compatible do llama-server.

O mecanismo de cancelamento do barge-in depende de uma propriedade especifica
deste cliente: `stream_chat` e um async generator que mantem a conexao HTTP
aberta dentro de um `async with`. Se a task que o consome for cancelada
(`task.cancel()`), o `CancelledError` e lancado no ponto do `yield`, propaga
para fora dos blocos `async with` e fecha a conexao HTTP. O llama-server
detecta esse disconnect e libera o slot de geracao imediatamente -- e assim
que "cancelar o LLM" funciona, sem precisar de um endpoint de abort dedicado.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

import httpx

logger = logging.getLogger(__name__)


class LlamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model_name: str = "qwen3-8b-instruct-q4_k_m",
        request_timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.request_timeout_s = request_timeout_s

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        payload: dict = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        timeout = httpx.Timeout(self.request_timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("Linha SSE nao-JSON ignorada: %r", data_str)
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    token = delta.get("content")
                    if token:
                        yield token

    async def chat_once(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        tools: list[dict] | None = None,
    ) -> dict:
        """Chamada nao-streaming; usada so na primeira volta do tool-calling,
        onde precisamos inspecionar `tool_calls` antes de decidir se a
        resposta final deve ser transmitida em streaming."""
        payload: dict = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.request_timeout_s)) as client:
            response = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{self.base_url}/health")
                return r.status_code == 200
        except httpx.HTTPError:
            return False
