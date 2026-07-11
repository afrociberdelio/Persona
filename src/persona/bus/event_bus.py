"""Event bus minimo em memoria, sobre asyncio.Queue.

Escolha deliberada de nao usar um broker externo (Kafka/RabbitMQ/ZeroMQ):
o Persona roda como processo unico numa unica maquina, entao o unico
requisito real e ordenacao dentro de um topico e baixa latencia de
publish->deliver. Um dict de filas resolve isso em ~100 linhas. Se um dia
for necessario distribuir os servicos entre processos/maquinas, os
contratos de evento (persona.bus.events) ja estao desacoplados o
suficiente para trocar o transporte sem tocar na logica de negocio.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Type, TypeVar

from persona.bus.events import Event

logger = logging.getLogger(__name__)

E = TypeVar("E", bound=Event)


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[type, list[asyncio.Queue]] = defaultdict(list)

    def subscribe(self, event_type: Type[E]) -> asyncio.Queue:
        """Retorna uma fila nova que recebera apenas eventos de `event_type`."""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[event_type].append(queue)
        return queue

    def unsubscribe(self, event_type: Type[E], queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(event_type)
        if subs and queue in subs:
            subs.remove(queue)

    def publish(self, event: Event) -> None:
        """Publica para todo subscriber do tipo exato do evento.

        Sincrono de proposito: enfileirar num asyncio.Queue nao bloqueia,
        entao publish() pode ser chamado de qualquer callback/handler sem
        precisar de await, o que importa no caminho critico do barge-in.
        """
        subs = self._subscribers.get(type(event), [])
        if not subs:
            logger.debug("Evento %s publicado sem nenhum subscriber", type(event).__name__)
        for queue in subs:
            queue.put_nowait(event)
