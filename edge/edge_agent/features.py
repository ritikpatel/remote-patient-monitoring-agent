"""Windowed feature computation over one watch batch.

PROJECT_PLAN.md section 8, item 6: "computes windowed features locally." The window
is the batch itself (protocol.SAMPLE_WINDOW_S) -- see protocol.py's docstring for why
there is no separate rolling-window step. Two features come out of every batch:

  - hr: mean heart rate over the batch (smooths sensor noise before it ever leaves
    the device).
  - activity_index: RMS deviation of 3-axis accelerometer magnitude from 1g. A
    stationary wrist reads ~1g on the magnitude; deviation reflects movement
    intensity. This is a genuinely edge-computed feature -- raw accelerometer never
    needs to reach the server for this purpose, which is the point of computing it
    on-device rather than shipping raw samples.
"""

from __future__ import annotations

import numpy as np

from edge.edge_agent.protocol import WatchBatch


def summarize_batch(batch: WatchBatch) -> dict[str, float]:
    by_channel: dict[str, list[float]] = {}
    for s in batch.samples:
        by_channel.setdefault(s.channel, []).append(s.value)

    features: dict[str, float] = {}
    if by_channel.get("hr"):
        features["hr"] = float(np.mean(by_channel["hr"]))

    axes = [by_channel.get(a, []) for a in ("acc_x", "acc_y", "acc_z")]
    lengths = {len(a) for a in axes}
    if all(axes) and len(lengths) == 1:
        x, y, z = (np.array(a) for a in axes)
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        features["activity_index"] = float(np.sqrt(np.mean((magnitude - 1.0) ** 2)))

    return features
