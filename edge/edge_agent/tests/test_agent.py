import asyncio
import socket

import pytest
from services.contracts.observation import ObservationSource

from edge.edge_agent.agent import EdgeAgent, batch_to_observations, make_demo_batches, run
from edge.edge_agent.features import summarize_batch
from edge.edge_agent.mqtt_publisher import MqttConfig, MqttPublisher
from edge.edge_agent.outbox import Outbox
from edge.edge_agent.protocol import SAMPLE_WINDOW_S
from edge.edge_agent.transport import ReplayTransport, write_replay_file


def _broker_reachable(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def test_summarize_batch_computes_hr_mean_and_activity_index():
    batches = make_demo_batches(minutes=SAMPLE_WINDOW_S / 60, seed=1)
    features = summarize_batch(batches[0])
    assert "hr" in features
    assert "activity_index" in features
    assert features["activity_index"] >= 0


def test_batch_to_observations_uses_watch_batch_identity():
    batch = make_demo_batches(minutes=SAMPLE_WINDOW_S / 60, patient_ref="Patient/999")[0]
    obs = batch_to_observations(batch)
    assert obs
    assert all(o.patient_ref == "Patient/999" for o in obs)
    assert all(o.source == ObservationSource.wearable for o in obs)
    assert all(o.device_id == batch.device_id for o in obs)


def test_outbox_round_trip(tmp_path):
    outbox = Outbox(tmp_path / "outbox.db")
    batch = make_demo_batches(minutes=SAMPLE_WINDOW_S / 60)[0]
    obs = batch_to_observations(batch)
    for o in obs:
        outbox.add(o)
    assert outbox.pending_count() == len(obs)
    pending = outbox.pending()
    outbox.mark_published([rid for rid, _ in pending])
    assert outbox.pending_count() == 0
    outbox.close()


def test_agent_buffers_everything_with_no_publisher(tmp_path):
    outbox = Outbox(tmp_path / "outbox.db")
    agent = EdgeAgent(outbox, publisher=None, sink=None)
    batches = make_demo_batches(minutes=1.0, seed=2)
    replay_path = tmp_path / "batches.jsonl"
    write_replay_file(replay_path, batches)
    transport = ReplayTransport(replay_path, sleep=False)

    n = asyncio.run(run(transport, agent))

    assert n == len(batches)
    assert agent.observations_published == 0
    assert agent.observations_buffered == outbox.pending_count() > 0
    outbox.close()


def test_agent_falls_back_to_buffering_when_broker_unreachable(tmp_path):
    outbox = Outbox(tmp_path / "outbox.db")
    publisher = MqttPublisher(MqttConfig(host="127.0.0.1", port=18830, connect_timeout_s=0.5))
    assert publisher.connect() is False  # nothing is listening on this port

    agent = EdgeAgent(outbox, publisher=publisher, sink=None)
    batches = make_demo_batches(minutes=0.5, seed=3)
    replay_path = tmp_path / "batches.jsonl"
    write_replay_file(replay_path, batches)
    transport = ReplayTransport(replay_path, sleep=False)

    asyncio.run(run(transport, agent))

    assert agent.observations_published == 0
    assert outbox.pending_count() > 0
    outbox.close()


@pytest.mark.skipif(
    not _broker_reachable("localhost", 1883), reason="no local MQTT broker on :1883"
)
def test_agent_publishes_and_drains_outbox_against_a_real_broker(tmp_path):
    outbox = Outbox(tmp_path / "outbox.db")
    publisher = MqttPublisher(MqttConfig(host="localhost", port=1883))
    assert publisher.connect()

    agent = EdgeAgent(outbox, publisher=publisher, sink=None)
    batches = make_demo_batches(minutes=0.5, seed=4)
    replay_path = tmp_path / "batches.jsonl"
    write_replay_file(replay_path, batches)
    transport = ReplayTransport(replay_path, sleep=False)

    asyncio.run(run(transport, agent))

    assert agent.observations_published > 0
    assert outbox.pending_count() == 0
    publisher.close()
    outbox.close()
