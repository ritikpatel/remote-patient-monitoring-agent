"""The phone/gateway-side half of PROJECT_PLAN.md section 8, item 6.

Pipeline: Transport (BLE from a real watch, or ReplayTransport for demo/test) ->
features.summarize_batch (windowed features, computed locally) -> Observation ->
MqttPublisher (publish now) or Outbox (buffer offline, if the broker is unreachable).
Every batch is durable the moment it's received: it only ever leaves the outbox once
`mark_published` confirms a successful publish.

Usage:
    # generate a demo batch stream, then run the agent against it (no MQTT needed)
    python -m edge.edge_agent.agent make-demo --out demo_batches.jsonl --minutes 5
    python -m edge.edge_agent.agent run --transport file --in demo_batches.jsonl \
        --patient-ref Patient/10005866 --sink jsonl --out edge_observations.jsonl

    # against a real broker
    python -m edge.edge_agent.agent run --transport file --in demo_batches.jsonl \
        --mqtt-host localhost --mqtt-port 8883 \
        --ca-cert ca.pem --client-cert c.pem --client-key c.key
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.contracts.observation import Observation, ObservationSource  # noqa: E402
from simulators.sinks import Sink, make_sink  # noqa: E402

from edge.edge_agent.features import summarize_batch  # noqa: E402
from edge.edge_agent.mqtt_publisher import MqttConfig, MqttPublisher  # noqa: E402
from edge.edge_agent.outbox import Outbox  # noqa: E402
from edge.edge_agent.protocol import SAMPLE_WINDOW_S, WatchBatch, WatchSample  # noqa: E402
from edge.edge_agent.transport import (  # noqa: E402
    ReplayTransport,
    Transport,
    drain,
    write_replay_file,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTBOX_PATH = REPO_ROOT / "edge" / "edge_agent" / "outbox.db"


def batch_to_observations(batch: WatchBatch) -> list[Observation]:
    features = summarize_batch(batch)
    window_end = batch.batch_start + timedelta(seconds=SAMPLE_WINDOW_S)
    return [
        Observation.for_channel(
            channel=channel,
            patient_ref=batch.patient_ref,
            device_id=batch.device_id,
            source=ObservationSource.wearable,
            value=value,
            effective_time=window_end,
        )
        for channel, value in features.items()
    ]


class EdgeAgent:
    def __init__(self, outbox: Outbox, publisher: MqttPublisher | None, sink: Sink | None) -> None:
        self.outbox = outbox
        self.publisher = publisher
        self.sink = sink
        self.observations_published = 0
        self.observations_buffered = 0

    def handle_batch(self, batch: WatchBatch) -> None:
        for obs in batch_to_observations(batch):
            self._deliver(obs)
        self.flush_outbox()

    def _deliver(self, obs: Observation) -> None:
        if self.sink is not None:
            self.sink.emit(obs)
        if self.publisher is not None and self.publisher.connected and self.publisher.publish(obs):
            self.observations_published += 1
        else:
            self.outbox.add(obs)
            self.observations_buffered += 1

    def flush_outbox(self) -> None:
        if self.publisher is None or not self.publisher.connected:
            return
        published_ids = []
        for row_id, obs in self.outbox.pending():
            if self.publisher.publish(obs):
                published_ids.append(row_id)
                self.observations_published += 1
        if published_ids:
            self.outbox.mark_published(published_ids)
            self.observations_buffered -= len(published_ids)


def make_demo_batches(
    minutes: float,
    device_id: str = "wear-os-demo",
    patient_ref: str = "Patient/10005866",
    start: datetime | None = None,
    seed: int = 0,
) -> list[WatchBatch]:
    """A watch-shaped stream for demo/testing: 1 Hz HR around a slow sinusoid + noise,
    32 Hz-equivalent-but-batch-summarised accelerometer with periodic movement bursts.
    """
    rng = np.random.default_rng(seed)
    start = start or datetime.now(UTC)
    n_batches = max(int(minutes * 60 / SAMPLE_WINDOW_S), 1)
    batches = []
    for i in range(n_batches):
        batch_start = start + timedelta(seconds=i * SAMPLE_WINDOW_S)
        samples = []
        for s in range(int(SAMPLE_WINDOW_S)):  # 1 Hz HR
            hr = 72 + 5 * np.sin(i / 6) + rng.normal(0, 2)
            samples.append(WatchSample(offset_ms=s * 1000, channel="hr", value=float(hr)))
        moving = (i // 3) % 2 == 0  # alternate resting / moving every 3 windows
        n_acc = 50
        for s in range(n_acc):
            offset_ms = int(s * SAMPLE_WINDOW_S * 1000 / n_acc)
            base = rng.normal(0, 0.35 if moving else 0.02, size=3)
            x, y, z = base[0], base[1], 1.0 + base[2]
            samples.append(WatchSample(offset_ms=offset_ms, channel="acc_x", value=float(x)))
            samples.append(WatchSample(offset_ms=offset_ms, channel="acc_y", value=float(y)))
            samples.append(WatchSample(offset_ms=offset_ms, channel="acc_z", value=float(z)))
        batches.append(
            WatchBatch(
                device_id=device_id,
                patient_ref=patient_ref,
                battery_pct=max(100 - i * 0.05, 0),
                batch_start=batch_start,
                samples=samples,
            )
        )
    return batches


async def run(transport: Transport, agent: EdgeAgent, max_batches: int | None = None) -> int:
    return await drain(transport, agent.handle_batch, max_batches=max_batches)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("make-demo", help="generate a synthetic watch-batch JSONL file")
    demo.add_argument("--out", type=Path, required=True)
    demo.add_argument("--minutes", type=float, default=5.0)
    demo.add_argument("--patient-ref", default="Patient/10005866")

    run_p = sub.add_parser("run", help="run the agent against a transport")
    run_p.add_argument("--transport", choices=["file", "ble"], default="file")
    run_p.add_argument("--in", dest="in_path", type=Path, help="JSONL file for --transport file")
    run_p.add_argument("--ble-address", help="device address for --transport ble")
    run_p.add_argument("--compress", type=float, default=1.0)
    run_p.add_argument("--no-sleep", action="store_true")
    run_p.add_argument("--max-batches", type=int, default=None)
    run_p.add_argument("--outbox", type=Path, default=DEFAULT_OUTBOX_PATH)
    run_p.add_argument("--sink", choices=["none", "console", "jsonl"], default="console")
    run_p.add_argument("--out", type=Path, default=Path("edge_observations.jsonl"))
    run_p.add_argument("--mqtt-host")
    run_p.add_argument("--mqtt-port", type=int, default=8883)
    run_p.add_argument("--ca-cert", type=Path)
    run_p.add_argument("--client-cert", type=Path)
    run_p.add_argument("--client-key", type=Path)

    args = ap.parse_args()

    if args.cmd == "make-demo":
        batches = make_demo_batches(args.minutes, patient_ref=args.patient_ref)
        write_replay_file(args.out, batches)
        print(f"Wrote {len(batches)} batches ({args.minutes:.1f} min) to {args.out}")
        return 0

    if args.transport == "file":
        if not args.in_path:
            print("ERROR: --transport file requires --in", file=sys.stderr)
            return 1
        transport: Transport = ReplayTransport(
            args.in_path, compress=args.compress, sleep=not args.no_sleep
        )
    else:
        from edge.edge_agent.protocol import BATCH_CHARACTERISTIC_UUID
        from edge.edge_agent.transport import BleTransport

        if not args.ble_address:
            print("ERROR: --transport ble requires --ble-address", file=sys.stderr)
            return 1
        transport = BleTransport(args.ble_address, BATCH_CHARACTERISTIC_UUID)

    publisher = None
    if args.mqtt_host:
        config = MqttConfig(
            host=args.mqtt_host,
            port=args.mqtt_port,
            ca_cert=args.ca_cert,
            client_cert=args.client_cert,
            client_key=args.client_key,
        )
        publisher = MqttPublisher(config)
        ok = publisher.connect()
        status = "ok" if ok else "FAILED -- buffering to outbox"
        print(f"MQTT connect to {config.host}:{config.port}: {status}", file=sys.stderr)

    outbox = Outbox(args.outbox)
    sink = None if args.sink == "none" else make_sink(args.sink, args.out)
    agent = EdgeAgent(outbox, publisher, sink)

    try:
        n = asyncio.run(run(transport, agent, max_batches=args.max_batches))
    finally:
        if sink is not None:
            sink.close()
        if publisher is not None:
            publisher.close()
        outbox.close()

    print(
        f"Processed {n} batches -> {agent.observations_published} published, "
        f"{agent.observations_buffered} buffered in {args.outbox}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
