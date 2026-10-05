"""The bus is the only thing every module touches, so it gets real tests."""

from __future__ import annotations

import asyncio

import pytest

from chilluevar.bus import EventBus
from chilluevar.events import SpeakRequest, Transcribed


async def test_subscriber_receives_published_event() -> None:
    bus = EventBus()
    async with bus.subscribe(Transcribed) as stream:
        await bus.publish(Transcribed(turn_id="t1", text="hola"))
        event = await asyncio.wait_for(anext(stream), timeout=1)
    assert event.text == "hola"


async def test_other_event_types_are_not_delivered() -> None:
    bus = EventBus()
    async with bus.subscribe(Transcribed) as stream:
        await bus.publish(SpeakRequest(turn_id="t1", text="hola"))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(anext(stream), timeout=0.05)


async def test_two_subscribers_both_receive() -> None:
    bus = EventBus()
    async with bus.subscribe(Transcribed) as first, bus.subscribe(Transcribed) as second:
        await bus.publish(Transcribed(turn_id="t1", text="hola"))
        assert (await asyncio.wait_for(anext(first), timeout=1)).text == "hola"
        assert (await asyncio.wait_for(anext(second), timeout=1)).text == "hola"


async def test_slow_subscriber_drops_oldest_instead_of_blocking() -> None:
    # Publishing must never block the microphone. Stale audio is worthless, so
    # the oldest event is the right one to lose.
    bus = EventBus(queue_size=2)
    async with bus.subscribe(Transcribed) as stream:
        for i in range(4):
            await bus.publish(Transcribed(turn_id=f"t{i}", text=str(i)))
        assert bus.dropped == 2
        first = await asyncio.wait_for(anext(stream), timeout=1)
    assert first.text == "2"


async def test_unsubscribing_stops_delivery() -> None:
    bus = EventBus()
    async with bus.subscribe(Transcribed):
        pass
    # No subscribers left: publishing must not raise.
    await bus.publish(Transcribed(turn_id="t1", text="hola"))
