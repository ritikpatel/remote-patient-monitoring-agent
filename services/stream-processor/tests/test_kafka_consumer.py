import socket
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "services" / "stream-processor"))

from kafka_consumer import KafkaConsumerThread, WindowStore  # noqa: E402
from services.common.publisher import KafkaPublisher, topic_for  # noqa: E402
from services.contracts.observation import Observation, ObservationSource  # noqa: E402

KAFKA_BOOTSTRAP = "localhost:9092"


def _obs(value: float, hours_offset: float, patient_ref: str = "ICUStay/1") -> Observation:
    return Observation.for_channel(
        channel="hr",
        patient_ref=patient_ref,
        device_id="monitor-1",
        source=ObservationSource.icu_monitor,
        value=value,
        effective_time=datetime(2110, 1, 1, tzinfo=UTC) + timedelta(hours=hours_offset),
    )


# --------------------------------------------------------------------------
# WindowStore: pure logic, no broker needed -- always runs.
# --------------------------------------------------------------------------


def test_window_store_single_observation_has_zero_std_and_slope():
    store = WindowStore()
    latest = store.ingest(_obs(90.0, 0))
    assert latest.mean == 90.0
    assert latest.std == 0.0
    assert latest.n_samples == 1
    assert latest.trend_slope_per_hour == 0.0


def test_window_store_computes_rolling_mean_and_rising_trend():
    store = WindowStore()
    store.ingest(_obs(80.0, 0))
    store.ingest(_obs(90.0, 1))
    latest = store.ingest(_obs(100.0, 2))
    assert latest.mean == 90.0
    assert latest.n_samples == 3
    # 80 -> 90 -> 100 over 2h is a clean +10 bpm/hour OLS slope.
    assert latest.trend_slope_per_hour == pytest.approx(10.0)


def test_window_store_keeps_channels_separate():
    store = WindowStore()
    store.ingest(_obs(80.0, 0, patient_ref="ICUStay/1"))
    store.ingest(_obs(150.0, 0, patient_ref="ICUStay/2"))
    assert store.get("ICUStay/1", "8867-4").mean == 80.0
    assert store.get("ICUStay/2", "8867-4").mean == 150.0
    assert set(store.keys()) == {("ICUStay/1", "8867-4"), ("ICUStay/2", "8867-4")}


def test_window_store_orders_out_of_order_arrivals_by_effective_time():
    # A late-arriving but earlier-timestamped message must not be treated as the
    # window's most recent point -- ingest() sorts by effective_time, not arrival
    # order, before computing the slope.
    store = WindowStore()
    store.ingest(_obs(100.0, 2))
    latest = store.ingest(_obs(80.0, 0))
    assert latest.trend_slope_per_hour == pytest.approx(10.0)


def test_window_store_get_returns_none_before_any_message():
    store = WindowStore()
    assert store.get("ICUStay/nobody", "8867-4") is None


def test_window_store_respects_maxlen():
    store = WindowStore(maxlen=3)
    for i in range(5):
        store.ingest(_obs(float(i), float(i)))
    assert store.get("ICUStay/1", "8867-4").n_samples == 3


# --------------------------------------------------------------------------
# Real Kafka: self-skips if a broker isn't actually reachable, the same pattern
# eval/tests/test_latency.py uses for k6/the pipeline services.
# --------------------------------------------------------------------------


def _kafka_reachable() -> bool:
    host, _, port = KAFKA_BOOTSTRAP.partition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _kafka_reachable(),
    reason="no Kafka broker reachable at localhost:9092 -- see infra/compose/README.md",
)
def test_kafka_publisher_to_consumer_thread_end_to_end():
    """The real, broker-dependent path: KafkaPublisher (already used by
    ingest-gateway when PUBLISHER_BACKEND=kafka) publishes a real Observation,
    KafkaConsumerThread (what stream-processor starts when KAFKA_BOOTSTRAP_SERVERS
    is set) consumes it for real, and WindowStore reflects it -- proving the two
    previously-never-connected halves of this project's Kafka code actually work
    together against a real broker, not just against each other's mocks.
    """
    # A unique group_id, not the real service's "stream-processor" default --
    # sharing that id with an actual running instance (e.g. a docker-compose
    # deployment on this same broker) makes Kafka's group protocol split this
    # topic's one partition between the two, and whichever wins the assignment
    # is arbitrary. Found for real: this test failed against a broker that also
    # had a live docker-compose stream-processor consuming under the same
    # group id -- not because the code was wrong, but because the test and the
    # real service were fighting over one partition.
    store = WindowStore()
    consumer = KafkaConsumerThread(store, KAFKA_BOOTSTRAP, group_id=f"test-{uuid.uuid4().hex[:8]}")
    consumer.start()
    try:
        # Give the background thread's initial subscribe/poll a moment before
        # publishing, so its first metadata refresh has a chance to see the
        # topic before or shortly after this test creates it.
        time.sleep(2)

        publisher = KafkaPublisher(bootstrap_servers=KAFKA_BOOTSTRAP)
        patient_ref = f"ICUStay/kafka-e2e-{int(time.time())}"
        obs = _obs(77.0, 0, patient_ref=patient_ref)
        publisher.publish(topic_for(obs.source.value), obs)
        publisher.close()

        deadline = time.time() + 15
        latest = None
        while time.time() < deadline:
            latest = store.get(patient_ref, obs.code)
            if latest is not None:
                break
            time.sleep(0.5)

        assert latest is not None, "consumer never observed the published message"
        assert latest.mean == 77.0
        assert latest.n_samples == 1
    finally:
        consumer.stop()
