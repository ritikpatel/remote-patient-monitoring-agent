"""How raw watch batches reach this process. One protocol (protocol.WatchBatch),
two transports:

  - BleTransport: a real BLE GATT central using `bleak`, connecting to the Wear OS
    app's characteristic (protocol.SERVICE_UUID / BATCH_CHARACTERISTIC_UUID).
    Complete, real code -- but it needs a physical Wear OS watch running
    edge/wear_os/ and a BLE adapter, neither of which this development environment
    has, so it has never been exercised end-to-end here. Everything downstream of
    "a WatchBatch arrived" (features.py, outbox.py, mqtt_publisher.py) has been.
  - ReplayTransport: replays a JSONL file of WatchBatch records at real or
    accelerated timing. This is what every test and CLI demo in this module actually
    runs against.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Protocol

from edge.edge_agent.protocol import WatchBatch


class Transport(Protocol):
    # NOT `async def`: calling an async-generator method (one with `yield` in its
    # body, as both implementations below have) returns the AsyncIterator directly,
    # with no `await` involved -- an `async def -> AsyncIterator[...]` signature with
    # no `yield` describes a coroutine that RETURNS an iterator, a different (and
    # here, wrong) shape that mypy would otherwise hold both implementations to.
    def batches(self) -> AsyncIterator[WatchBatch]: ...


class ReplayTransport:
    def __init__(self, path: Path, compress: float = 1.0, sleep: bool = True) -> None:
        self.path = path
        self.compress = compress
        self.sleep = sleep

    async def batches(self) -> AsyncIterator[WatchBatch]:
        prev_start = None
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                batch = WatchBatch.model_validate_json(line)
                if self.sleep and prev_start is not None:
                    dt = (batch.batch_start - prev_start).total_seconds()
                    await asyncio.sleep(max(dt / self.compress, 0.0))
                prev_start = batch.batch_start
                yield batch


class BleTransport:
    """Connects to a Wear OS device's GATT characteristic and decodes each
    notification as one JSON-encoded WatchBatch. Requires `bleak` (a real
    dependency, not optional) and a physical adapter + paired watch.
    """

    def __init__(self, device_address: str, characteristic_uuid: str) -> None:
        self.device_address = device_address
        self.characteristic_uuid = characteristic_uuid

    async def batches(self) -> AsyncIterator[WatchBatch]:
        from bleak import (
            BleakClient,  # imported lazily: no BLE hardware needed to import this module
        )

        queue: asyncio.Queue[WatchBatch] = asyncio.Queue()

        def _on_notify(_handle: int, data: bytearray) -> None:
            batch = WatchBatch.model_validate(json.loads(bytes(data).decode("utf-8")))
            queue.put_nowait(batch)

        async with BleakClient(self.device_address) as client:
            await client.start_notify(self.characteristic_uuid, _on_notify)
            while True:
                yield await queue.get()


def write_replay_file(path: Path, batches: list[WatchBatch]) -> None:
    with open(path, "w") as f:
        for b in batches:
            f.write(b.model_dump_json() + "\n")


async def drain(
    transport: Transport, on_batch: Callable[[WatchBatch], None], max_batches: int | None = None
) -> int:
    n = 0
    async for batch in transport.batches():
        on_batch(batch)
        n += 1
        if max_batches is not None and n >= max_batches:
            break
    return n
