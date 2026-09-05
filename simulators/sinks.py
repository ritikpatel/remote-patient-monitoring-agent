"""Where a replay simulator's Observations go. Shared by icu_replay.py and
wearable_replay.py so both producers write the exact same wire shapes.

Kafka/EMQX don't exist yet (Phase 8 infra) -- these are deliberately dependency-free
so both simulators are runnable and demoable standalone. Wiring a KafkaSink in here
is Phase 4's `ingest-gateway`'s job, at which point it becomes a third Sink
implementation behind the same protocol; nothing above this module should need to
change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol

from services.contracts.observation import Observation


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


def make_sink(kind: str, out: Path) -> Sink:
    if kind == "console":
        return ConsoleSink()
    if kind == "jsonl":
        return JSONLSink(out)
    raise ValueError(f"unknown sink kind: {kind}")
