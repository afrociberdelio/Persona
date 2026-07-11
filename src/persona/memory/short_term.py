"""Janela deslizante das ultimas trocas da conversa (memoria de curto prazo)."""
from __future__ import annotations


class ShortTermMemory:
    def __init__(self, max_turns: int = 12) -> None:
        self.max_turns = max_turns
        self._turns: list[tuple[str, str]] = []  # (role, content), mais antigo primeiro

    def add(self, role: str, content: str) -> None:
        self._turns.append((role, content))
        while len(self._turns) > self.max_turns:
            self._turns.pop(0)

    def get_turns(self) -> list[tuple[str, str]]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()
