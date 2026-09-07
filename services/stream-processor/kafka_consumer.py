"""The other half of ingest-gateway's `KafkaPublisher` (services/common/publisher.py):
a real consumer of the `raw.*` topics it writes to, standing idle -- like every other
not-yet-deployed dependency in this project -- until Phase 8 (docker-compose Kafka)
gives it a broker to actually connect to.

Design: subscribe to every `raw.*` topic (one per `ObservationSource`, per
`topic_for()`), and for each (patient_ref, code) channel maintain a bounded rolling
window of recent Observations, recomputing the same `windowing.rolling_stats` /
`trend_slope` PROJECT_PLAN.md section 10 asks `/window/process` to compute over
HTTP -- this consumer computes it continuously as messages arrive instead, which is
what "windowing" actually means once there is a real stream rather than a
one-off batch POST. `WindowStore` is deliberately the only piece of state here and
is plain, lock-protected Python -- no framework dependency -- so it is unit-testable
without a broker, and the broker-dependent part (`KafkaConsumerThread`) is a thin
shell around it that is exercised for real only when Kafka is actually reachable
(see eval/tests/test_latency.py's `k6_available()` for the same "self-skip if the
real infra isn't up" pattern this module's own test follows).
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.contracts.observation import Observation  # noqa: E402
from windowing import rolling_stats, trend_slope  # noqa: E402

logger = logging.getLogger(__name__)

RAW_TOPIC_PATTERN = re.compile(r"^raw\..*")
# Bounds per-channel memory: enough points to cover a multi-hour rolling window at
# any of this project's real sampling rates (ICU monitor observations are roughly
# hourly per capstone.hourly_grid; wearable channels sample far faster) without
# unbounded growth for a long-running consumer process.
WINDOW_MAXLEN = 120


@dataclass(frozen=True)
class LatestWindow:
    patient_ref: str
    code: str
    mean: float
    std: float
    n_samples: int
    trend_slope_per_hour: float
    last_ingest_time: str


class WindowStore:
    """Thread-safe: one lock guards both the per-channel buffers and the latest-
    computed-window cache, since every write touches both together. A single
    consumer thread is this project's only writer (see the single-writer note in
    services/common/audit.py and services/alert-service/store.py -- the same
    constraint, for the same reason), so a plain `threading.Lock` is the right
    amount of concurrency control, not a false economy.
    """

    def __init__(self, maxlen: int = WINDOW_MAXLEN) -> None:
        self._lock = threading.Lock()
        self._buffers: dict[tuple[str, str], deque[Observation]] = defaultdict(
            lambda: deque(maxlen=maxlen)
        )
        self._latest: dict[tuple[str, str], LatestWindow] = {}

    def ingest(self, obs: Observation) -> LatestWindow:
        key = (obs.patient_ref, obs.code)
        with self._lock:
            buf = self._buffers[key]
            buf.append(obs)
            ordered = sorted(buf, key=lambda o: o.effective_time)
            first_time = ordered[0].effective_time
            values = [o.value for o in ordered]
            hours = [(o.effective_time - first_time).total_seconds() / 3600 for o in ordered]
            stats = rolling_stats(values)
            slope = trend_slope(hours, values)
            latest = LatestWindow(
                patient_ref=obs.patient_ref,
                code=obs.code,
                mean=stats.mean,
                std=stats.std,
                n_samples=stats.n,
                trend_slope_per_hour=slope,
                last_ingest_time=obs.ingest_time.isoformat(),
            )
            self._latest[key] = latest
            return latest

    def get(self, patient_ref: str, code: str) -> LatestWindow | None:
        with self._lock:
            return self._latest.get((patient_ref, code))

    def keys(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._latest.keys())


DEFAULT_GROUP_ID = "stream-processor"


def _consume_forever(
    store: WindowStore, bootstrap_servers: str, stop_event: threading.Event, group_id: str
) -> None:
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        bootstrap_servers=bootstrap_servers,
        # "earliest", not "latest": auto_offset_reset only applies the very first
        # time this group sees a partition (a committed offset always wins after
        # that), but "latest" resolves against whatever the log end offset is
        # *at partition-assignment time* -- for a `raw.<source>` topic that gets
        # auto-created by ingest-gateway's *first* publish, a message produced
        # right as this consumer's periodic metadata refresh (below) discovers
        # that brand-new topic can already be sitting at offset 0 before "latest"
        # resolves, so the seek lands past it and the message is silently never
        # delivered. Found for real: a docker-compose end-to-end test published
        # one message through a fresh stack and it never reached
        # GET /window/latest, despite the consumer group's committed offset
        # showing the message as "consumed" (it was skipped by the seek, not
        # processed). "earliest" makes a freshly-assigned partition replay
        # everything currently on it instead of racing a brand-new topic's very
        # first message -- steady-state behaviour across restarts is unaffected,
        # since a committed offset always takes over after the first assignment.
        auto_offset_reset="earliest",
        group_id=group_id,
        # Bounds how long a `for message in consumer` iteration blocks with no
        # messages, so the loop actually notices `stop_event` instead of hanging
        # forever on a quiet topic.
        consumer_timeout_ms=1000,
        # A pattern subscription only picks up newly-created matching topics on
        # its next metadata refresh -- kafka-python-ng's default
        # (metadata_max_age_ms=300000, 5 minutes) means a `raw.<source>` topic
        # that ingest-gateway creates *after* this consumer subscribes can sit
        # invisible to it for up to 5 minutes. Found for real: the first end-to-
        # end test against a live broker (test_kafka_consumer.py) published one
        # message and never saw it consumed within a 15s deadline. This project's
        # topic set is small and effectively static (one per ObservationSource),
        # so refreshing every 3s to notice a new one costs nothing real.
        metadata_max_age_ms=3000,
    )
    consumer.subscribe(pattern=RAW_TOPIC_PATTERN)
    try:
        while not stop_event.is_set():
            for message in consumer:
                if stop_event.is_set():
                    break
                try:
                    obs = Observation.model_validate_json(message.value)
                except Exception:
                    # A malformed message on the wire must not take the whole
                    # consumer thread down -- log and keep windowing everything
                    # else, the same tolerance handle_mqtt_message's caller would
                    # need against a real, imperfect device population.
                    logger.warning("stream-processor: skipping unparseable message", exc_info=True)
                    continue
                store.ingest(obs)
    finally:
        consumer.close()


class KafkaConsumerThread:
    """Owns the background thread's lifecycle. `start`/`stop` are idempotent-ish
    (calling `stop` before `start` is a no-op) so app.py's startup/shutdown hooks
    don't need extra state to call them safely.
    """

    def __init__(
        self, store: WindowStore, bootstrap_servers: str, group_id: str = DEFAULT_GROUP_ID
    ) -> None:
        self.store = store
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=_consume_forever,
            args=(self.store, self.bootstrap_servers, self._stop_event, self.group_id),
            name="stream-processor-kafka-consumer",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
