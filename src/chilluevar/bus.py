"""A small in-process event bus.

Modules do not call each other. They publish events on this bus and subscribe to
the ones they care about. That is what lets the audio pipeline and the reasoning
core be built, tested and replaced independently, and what makes satellite mode
a matter of putting a network hop in the middle rather than a rewrite.

The implementation is deliberately plain: one `asyncio.Queue` per subscriber.
A slow subscriber cannot block a publisher because each queue is bounded and
publishing drops the oldest event when a queue is full, which for audio is the
correct behaviour: stale audio is worthless, and a bus that blocks here would
stall the microphone.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TypeVar

from chilluevar.events import Event

__all__ = ["EventBus"]

logger = logging.getLogger(__name__)

E = TypeVar("E", bound=Event)


class EventBus:
    """Publish/subscribe over event types."""

    def __init__(self, *, queue_size: int = 64) -> None:
        self._queue_size = queue_size
        self._subscribers: dict[type[Event], list[asyncio.Queue[Event]]] = {}
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """How many events were discarded because a subscriber fell behind.

        Surfaced rather than hidden: if this number grows during a turn, some
        stage is too slow and that is worth seeing in the logs.
        """
        return self._dropped

    @asynccontextmanager
    async def subscribe(self, event_type: type[E]) -> AsyncIterator[AsyncIterator[E]]:
        """Subscribe to one event type for the duration of the context.

        Used as:

            async with bus.subscribe(Transcribed) as stream:
                async for event in stream:
                    ...
        """
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.setdefault(event_type, []).append(queue)
        try:
            yield self._drain(queue)  # type: ignore[arg-type]
        finally:
            self._subscribers[event_type].remove(queue)

    async def _drain(self, queue: asyncio.Queue[Event]) -> AsyncIterator[Event]:
        while True:
            yield await queue.get()

    async def publish(self, event: Event) -> None:
        """Deliver an event to every subscriber of its exact type.

        Subtypes are not matched: a subscriber asks for the event it wants. This
        keeps delivery predictable, which matters more here than cleverness.
        """
        for queue in self._subscribers.get(type(event), []):
            if queue.full():
                queue.get_nowait()
                self._dropped += 1
                logger.warning(
                    "subscriber behind, dropped an event",
                    extra={"event_type": type(event).__name__, "turn_id": event.turn_id},
                )
            queue.put_nowait(event)
