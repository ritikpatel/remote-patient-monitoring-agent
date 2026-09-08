"""Where a replay simulator's Observations go. Shared by icu_replay.py and
wearable_replay.py so both producers write the exact same wire shapes.

ConsoleSink and JSONLSink are deliberately dependency-free, so both simulators stay
runnable and demoable with no infrastructure at all.

HTTPSink is the third implementation the module docstring originally anticipated, and
its absence was review finding F3: with only console and jsonl, neither replay could
reach the running system, so deliverable 1's acceptance test ("live watch + both
replays raise alerts through one engine") was not demonstrable even though every
piece of the engine worked. It POSTs to `ingest-gateway`, which publishes to Kafka,
which `stream-processor` consumes -- the same path the Wear OS edge agent takes via
MQTT, so all three producers now converge on one ingress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Protocol

import httpx
from services.contracts.observation import Observation

# Matches services/ingest-gateway/app.py's DEFAULT_API_KEY. A real deployment reads
# this from a secret store on both sides; kept aligned here so a demo works with no
# configuration, and overridable for anything that is not a demo.
DEFAULT_INGEST_API_KEY = "capstone-rpm-dev-ingest-key"


class Sink(Protocol):
    def emit(self, obs: Observation) -> None: ...
    def close(self) -> None: ...


@dataclass
class ConsoleSink:
    def emit(self, obs: Observation) -> None:
        print(
            f"{obs.effective_time.isoformat()}  {obs.patient_ref:<16} {obs.display:<32} "
            f"{obs.value:>9.3f} {obs.unit:<8} {','.join(f.value for f in obs.quality_flags)}"
        )

    def close(self) -> None:
        pass


@dataclass
class JSONLSink:
    path: Path
    _fh: IO[str] | None = None

    def __post_init__(self) -> None:
        self._fh = open(self.path, "w")

    def emit(self, obs: Observation) -> None:
        assert self._fh is not None
        self._fh.write(obs.model_dump_json() + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


@dataclass
class HTTPSink:
    """Streams Observations into a running ingest-gateway (finding F3).

    Batched, because a replay under time compression emits far faster than one HTTP
    round trip per reading: a 20-day stay at --compress 3600 produces thousands of
    observations in a few seconds. ``/observations/batch`` takes a list, so the
    default batch of 50 keeps the wire shape identical while cutting round trips.

    Failures are counted and reported rather than raised: a replay is a demo tool, and
    aborting a 3,000-observation stream because the gateway blipped once is worse than
    finishing and saying "17 failed". ``close()`` flushes whatever is left.
    """

    base_url: str = "http://localhost:8000"
    api_key: str = DEFAULT_INGEST_API_KEY
    batch_size: int = 50
    timeout: float = 10.0
    sent: int = 0
    failed: int = 0
    _buf: list[Observation] = field(default_factory=list)
    _client: httpx.Client | None = None

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            headers={"X-API-Key": self.api_key},
            timeout=self.timeout,
        )

    def emit(self, obs: Observation) -> None:
        self._buf.append(obs)
        if len(self._buf) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf or self._client is None:
            return
        payload = [o.model_dump(mode="json") for o in self._buf]
        try:
            resp = self._client.post("/observations/batch", json=payload)
            if resp.status_code in (200, 201, 202):
                self.sent += len(payload)
            else:
                self.failed += len(payload)
                print(f"  [HTTPSink] gateway returned {resp.status_code}: {resp.text[:160]}")
        except httpx.HTTPError as exc:
            self.failed += len(payload)
            print(f"  [HTTPSink] {type(exc).__name__}: {exc}")
        finally:
            self._buf.clear()

    def close(self) -> None:
        self._flush()
        if self._client is not None:
            self._client.close()
        print(f"  [HTTPSink] {self.sent} observations accepted, {self.failed} failed")


def make_sink(
    kind: str,
    out: Path,
    *,
    gateway_url: str = "http://localhost:8000",
    api_key: str = DEFAULT_INGEST_API_KEY,
) -> Sink:
    if kind == "console":
        return ConsoleSink()
    if kind == "jsonl":
        return JSONLSink(out)
    if kind == "http":
        return HTTPSink(base_url=gateway_url, api_key=api_key)
    raise ValueError(f"unknown sink kind: {kind}")
