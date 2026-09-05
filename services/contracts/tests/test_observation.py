from datetime import UTC, datetime
from typing import Any

from services.contracts.observation import (
    CHANNELS,
    Observation,
    ObservationSource,
    QualityFlag,
    from_avro_bytes,
    to_avro_bytes,
)


def make_obs(**overrides: Any) -> Observation:
    defaults: dict[str, Any] = dict(
        channel="hr",
        patient_ref="ICUStay/34547401",
        device_id="monitor-01",
        source=ObservationSource.icu_monitor,
        value=88.0,
        effective_time=datetime(2110, 3, 4, 10, 0, 0, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Observation.for_channel(**defaults)


def test_for_channel_fills_in_registry_fields():
    obs = make_obs()
    assert obs.code == CHANNELS["hr"].code == "8867-4"
    assert obs.unit == "/min"
    assert obs.code_system == "http://loinc.org"


def test_naive_datetime_is_coerced_to_utc():
    obs = make_obs(effective_time=datetime(2110, 3, 4, 10, 0, 0))
    assert obs.effective_time.tzinfo is not None


def test_avro_round_trip_preserves_every_field():
    obs = make_obs(quality_flags=[QualityFlag.imputed, QualityFlag.synthetic])
    back = from_avro_bytes(to_avro_bytes(obs))
    assert back == obs


def test_every_channel_round_trips():
    for name in CHANNELS:
        obs = make_obs(channel=name, value=1.0)
        assert from_avro_bytes(to_avro_bytes(obs)) == obs


def test_local_code_system_used_for_device_native_signals():
    obs = make_obs(channel="bvp", source=ObservationSource.wearable)
    assert obs.code_system != "http://loinc.org"
