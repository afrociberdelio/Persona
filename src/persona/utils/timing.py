"""Instrumentacao de latencia: registra timestamps monotonicos por estagio
do turno para permitir verificar as metas descritas no plano (ex: <100ms
de parada no barge-in, ~1-2s ponta a ponta ate o primeiro audio)."""
from __future__ import annotations

import logging
import time

logger = logging.getLogger("persona.timing")


class TurnTimer:
    def __init__(self, turn_id: str) -> None:
        self.turn_id = turn_id
        self._marks: dict[str, float] = {}

    def mark(self, label: str) -> None:
        self._marks[label] = time.monotonic()
        logger.debug("turn=%s stage=%s t=%.4f", self.turn_id, label, self._marks[label])

    def elapsed_between(self, start_label: str, end_label: str) -> float | None:
        start = self._marks.get(start_label)
        end = self._marks.get(end_label)
        if start is None or end is None:
            return None
        return end - start

    def summary(self) -> dict[str, float]:
        return dict(self._marks)
