"""Tests for the MIMIC-grounded home-kit simulator.

The claim this module makes is "the physiology is real, only the sensor is
simulated". These tests exist to keep that claim true, because it is the sort of
claim that decays silently: an off-by-one in the imputation filter or a bridged
measurement gap would still produce plausible-looking output.
"""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import numpy as np
import pytest

from simulators.home_kit_stream import (
    DEFAULT_DB_PATH,
    HOME_KITS,
    MAX_BRIDGE_H,
    NO_HOME_SENSOR,
    SENSORS,
    ChannelSensor,
    build_stream,
    deterioration_candidates,
    load_stay_vitals,
    synthesize_channel,
)

pytestmark = pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="warehouse not built")
INTIME = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def conn():
    c = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    yield c
    c.close()


@pytest.fixture
def a_deteriorating_stay(conn):
    candidates = deterioration_candidates(conn, limit=1)
    if not candidates:
        pytest.skip("no escalating stays in this warehouse")
    return candidates[0][0]


class TestKitDefinitions:
    def test_no_kit_claims_a_channel_that_has_no_home_sensor(self):
        """The one non-negotiable in this module: core temperature, GCS and FiO2
        have no home instrument at any price. A kit offering one would make every
        downstream post-discharge number a fiction."""
        for kit in HOME_KITS.values():
            assert not set(kit.channels) & set(NO_HOME_SENSOR), kit.name

    def test_every_kit_channel_has_a_sensor_model(self):
        for kit in HOME_KITS.values():
            for channel in kit.channels:
                assert channel in SENSORS, f"{kit.name}: {channel}"

    def test_kits_are_nested_from_sparsest_to_richest(self):
        """watch_only < watch_plus_cuff < full_home. Nesting is what makes the
        channel_dropout comparison across kits a single monotonic story rather than
        three unrelated experiments."""
        watch = set(HOME_KITS["watch_only"].channels)
        cuff = set(HOME_KITS["watch_plus_cuff"].channels)
        full = set(HOME_KITS["full_home"].channels)
        assert watch < cuff < full


class TestRealPhysiologyOnly:
    def test_carried_forward_hours_are_never_loaded(self, conn, a_deteriorating_stay):
        """`capstone.hourly_grid` forward-fills and flags it with `*_was_imputed`.
        An imputed hour means no measurement was taken, so streaming it would
        manufacture an observation that existed in neither the hospital nor the home.
        """
        kit = HOME_KITS["full_home"]
        vitals = load_stay_vitals(conn, a_deteriorating_stay, kit)

        for channel, observed in vitals.observed.items():
            if not observed:
                continue
            hours = list(observed)
            placeholders = ", ".join("?" * len(hours))
            imputed = conn.execute(
                f"SELECT COUNT(*) FROM capstone.hourly_grid WHERE stay_id = ? "
                f"AND hour IN ({placeholders}) AND {channel}_was_imputed",
                [a_deteriorating_stay, *hours],
            ).fetchone()[0]
            assert imputed == 0, f"{channel}: {imputed} carried-forward hours leaked in"

    def test_loaded_values_match_the_warehouse_exactly(self, conn, a_deteriorating_stay):
        """No scaling, no unit conversion, no smoothing at load time -- the anchors
        must be the recorded numbers."""
        vitals = load_stay_vitals(conn, a_deteriorating_stay, HOME_KITS["watch_only"])
        for hour, value in list(vitals.observed["hr"].items())[:10]:
            recorded = conn.execute(
                "SELECT hr FROM capstone.hourly_grid WHERE stay_id = ? AND hour = ?",
                [a_deteriorating_stay, hour],
            ).fetchone()[0]
            assert value == pytest.approx(recorded)


class TestSensorModel:
    def test_cadence_is_global_not_per_anchor(self):
        """The bug this pins: applying the device cadence *within* each anchor pair
        emits at least one sample per anchor regardless of cadence, so a 12-hourly BP
        cuff produced 218 readings over 315 hours instead of ~26 -- eight times
        richer than any real cuff, which destroys the point of modelling cadence.
        """
        observed = dict.fromkeys(range(0, 48), 120.0)  # hourly anchors, 48 hours
        twelve_hourly = ChannelSensor(interval_min=720.0, noise_sd=0.0, device="cuff")

        samples = synthesize_channel(
            "sbp", observed, INTIME, twelve_hourly, set(), np.random.default_rng(0)
        )

        assert 3 <= len(samples) <= 5, f"48h at 12h cadence should be ~4, got {len(samples)}"

    def test_a_fast_cadence_upsamples_between_hourly_anchors(self):
        observed = {0: 80.0, 1: 90.0}
        per_minute = ChannelSensor(interval_min=1.0, noise_sd=0.0, device="wrist PPG")

        samples = synthesize_channel(
            "hr", observed, INTIME, per_minute, set(), np.random.default_rng(0)
        )

        assert len(samples) > 30, "a per-minute device should sample within the hour"
        # Interpolation follows the real anchors: first sample at the first value,
        # last approaching the second.
        assert samples[0][1] == pytest.approx(80.0, abs=0.5)
        assert samples[-1][1] > samples[0][1]

    def test_a_long_measurement_gap_is_not_bridged(self):
        """The patient genuinely went unmeasured between the anchors. Drawing a
        smooth ramp across a 20-hour hole is the one fabrication capable of flipping
        the sign of a trend feature."""
        observed = {0: 80.0, 24: 140.0}  # a 24h gap, well over MAX_BRIDGE_H
        per_minute = ChannelSensor(interval_min=1.0, noise_sd=0.0, device="wrist PPG")

        samples = synthesize_channel(
            "hr", observed, INTIME, per_minute, set(), np.random.default_rng(0)
        )

        assert samples == [], f"bridged a {24}h gap (MAX_BRIDGE_H={MAX_BRIDGE_H})"

    def test_non_wear_hours_emit_nothing(self):
        observed = dict.fromkeys(range(0, 6), 100.0)
        per_minute = ChannelSensor(interval_min=1.0, noise_sd=0.0, device="wrist PPG")

        with_wear = synthesize_channel(
            "hr", observed, INTIME, per_minute, set(), np.random.default_rng(0)
        )
        with_nonwear = synthesize_channel(
            "hr", observed, INTIME, per_minute, {2, 3}, np.random.default_rng(0)
        )

        assert len(with_nonwear) < len(with_wear)
        emitted_hours = {int((t - INTIME).total_seconds() // 3600) for t, _ in with_nonwear}
        assert not emitted_hours & {2, 3}

    def test_noise_is_applied_but_does_not_move_the_trajectory(self):
        """Measurement error should scatter around the real value, not shift it --
        a biased sensor model would make the whole stream a different patient."""
        observed = dict.fromkeys(range(0, 24), 100.0)
        noisy = ChannelSensor(interval_min=1.0, noise_sd=5.0, device="wrist PPG")

        samples = synthesize_channel("hr", observed, INTIME, noisy, set(), np.random.default_rng(0))
        values = np.array([v for _, v in samples])

        assert values.std() > 1.0, "noise was not applied"
        assert values.mean() == pytest.approx(100.0, abs=1.0), "sensor model is biased"


class TestStream:
    def test_observations_are_flagged_synthetic_and_reference_a_subject(
        self, conn, a_deteriorating_stay
    ):
        """R7: never present this as measured. `Subject/<id>` rather than
        `ICUStay/<id>` because a patient at home is not an ICU stay -- that is what
        the reference form exists for."""
        observations, provenance = build_stream(conn, a_deteriorating_stay, HOME_KITS["full_home"])

        assert observations
        assert all("synthetic" in [f.value for f in o.quality_flags] for o in observations)
        assert all(o.device_id == "home-kit-sim" for o in observations)
        assert all(o.patient_ref.startswith("Subject/HOME-") for o in observations)
        assert "SYNTHETIC SENSOR LAYER" in provenance["watermark"]

    def test_observations_are_time_ordered(self, conn, a_deteriorating_stay):
        observations, _ = build_stream(conn, a_deteriorating_stay, HOME_KITS["watch_only"])
        times = [o.effective_time for o in observations]
        assert times == sorted(times)

    def test_a_richer_kit_emits_strictly_more_channels(self, conn, a_deteriorating_stay):
        watch, _ = build_stream(conn, a_deteriorating_stay, HOME_KITS["watch_only"])
        full, _ = build_stream(conn, a_deteriorating_stay, HOME_KITS["full_home"])

        assert {o.code for o in watch} < {o.code for o in full}

    def test_the_stream_is_reproducible_for_a_fixed_seed(self, conn, a_deteriorating_stay):
        a, _ = build_stream(conn, a_deteriorating_stay, HOME_KITS["watch_only"], seed=7)
        b, _ = build_stream(conn, a_deteriorating_stay, HOME_KITS["watch_only"], seed=7)
        assert [o.value for o in a] == [o.value for o in b]

    def test_provenance_reports_real_hours_against_simulated_samples(
        self, conn, a_deteriorating_stay
    ):
        """The two counts side by side are the honesty mechanism: a reader can see
        immediately how much of the stream is real anchors and how much is
        interpolated device texture."""
        _, provenance = build_stream(conn, a_deteriorating_stay, HOME_KITS["full_home"])

        assert provenance["real_observed_hours_per_channel"]["hr"] > 0
        assert (
            provenance["simulated_samples_per_channel"]["hr"]
            > provenance["real_observed_hours_per_channel"]["hr"]
        )
        assert provenance["channels_with_no_home_sensor"] == list(NO_HOME_SENSOR)


class TestCandidates:
    def test_candidates_really_do_escalate(self, conn):
        """Picked by the shared escalation predicate, not by hand -- so a demo cannot
        be quietly staged on a patient who merely looked good."""
        candidates = deterioration_candidates(conn, limit=5)
        assert candidates
        for stay_id, escalating_hours, _dx, _last in candidates:
            assert escalating_hours > 0
            confirmed = conn.execute(
                "SELECT COUNT(*) FROM capstone.news2 WHERE stay_id = ? "
                "AND (tier_icu = 'high' OR max_component_nongcs >= 3)",
                [stay_id],
            ).fetchone()[0]
            assert confirmed == escalating_hours
