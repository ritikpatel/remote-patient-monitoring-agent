"""Tests for the post-discharge digest, rewritten onto the MIMIC-grounded stream.

The previous suite asserted that HR trended up and SpO2 down across the week --
which was guaranteed, because the deterioration was *synthesised* to that shape by
`morphing.py`'s config. It tested the generator's arithmetic, not the pipeline.

With the physiology now real, that assertion is no longer available and should not
be replaced by a weaker version of itself: a real patient's real week does not move
monotonically. So these tests pin the things that must hold for the digest to be
honest -- real days, no invented channels, no HRV, and a narrative that says what it
is -- rather than a trajectory shape nobody controls.
"""

from __future__ import annotations

import duckdb
import pytest
from simulators.home_kit_stream import (
    DEFAULT_DB_PATH,
    HOME_KITS,
    NO_HOME_SENSOR,
    deterioration_candidates,
)

from reports.post_discharge_digest import (
    N_DIGEST_DAYS,
    build_post_discharge_digest,
)

pytestmark = pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="warehouse not built")


@pytest.fixture(scope="module")
def a_deteriorating_stay():
    conn = duckdb.connect(str(DEFAULT_DB_PATH), read_only=True)
    try:
        candidates = deterioration_candidates(conn, limit=1)
    finally:
        conn.close()
    if not candidates:
        pytest.skip("no escalating stays in this warehouse")
    return candidates[0][0]


def test_produces_one_bucket_per_real_day(a_deteriorating_stay):
    result = build_post_discharge_digest(a_deteriorating_stay, llm=None)

    assert len(result.daily_buckets) == N_DIGEST_DAYS
    assert [b.day_index for b in result.daily_buckets] == list(range(1, N_DIGEST_DAYS + 1))


def test_the_days_are_real_elapsed_days_not_a_compressed_session(a_deteriorating_stay):
    """The old digest divided a tens-of-minutes recording into seven fictional
    "days". This one windows a genuinely multi-day record, so there must be enough
    real recorded hours behind it for seven days to exist at all."""
    result = build_post_discharge_digest(a_deteriorating_stay, llm=None)

    assert result.provenance["icu_hours_in_record"] >= 24 * N_DIGEST_DAYS
    populated = [b for b in result.daily_buckets if b.means]
    assert len(populated) >= 5, "a multi-day record should populate most days"


def test_no_channel_without_a_home_sensor_ever_appears(a_deteriorating_stay):
    """Core temperature, GCS and FiO2 have no home instrument. A digest reporting
    one would be presenting a hospital measurement as a home reading."""
    result = build_post_discharge_digest(a_deteriorating_stay, llm=None)

    for bucket in result.daily_buckets:
        assert not set(bucket.means) & set(NO_HOME_SENSOR)


def test_hrv_is_absent_rather_than_approximated(a_deteriorating_stay):
    """MIMIC charts HR hourly, not beat-to-beat, so RMSSD is not computable. The
    honest output is no HRV at all -- deriving one from hourly HR would be a
    fabricated number wearing a real metric's name."""
    result = build_post_discharge_digest(a_deteriorating_stay, llm=None)

    for bucket in result.daily_buckets:
        assert "hrv" not in bucket.means
        assert "hrv_rmssd_ms" not in bucket.means
    assert not hasattr(result.daily_buckets[0], "hrv_rmssd_ms")


def test_an_unmeasured_channel_is_absent_not_zero(a_deteriorating_stay):
    """ "Not measured" and "measured as zero" are different clinical statements and a
    nurse must be able to tell them apart. The bucket holds only channels that
    actually produced a sample."""
    result = build_post_discharge_digest(a_deteriorating_stay, llm=None)

    kit_channels = set(HOME_KITS[result.kit].channels)
    for bucket in result.daily_buckets:
        assert set(bucket.means) <= kit_channels
        assert all(v == v for v in bucket.means.values()), "NaN leaked into a mean"


def test_narrative_and_provenance_declare_the_simulated_sensor_layer(a_deteriorating_stay):
    """R7. The digest must never read as if a home kit recorded this."""
    result = build_post_discharge_digest(a_deteriorating_stay, llm=None)

    assert result.narrative.startswith("[no LLM configured]")
    assert "simulated" in result.narrative
    assert "SYNTHETIC SENSOR LAYER" in result.provenance["watermark"]
    assert result.subject_ref.startswith("Subject/HOME-")


def test_a_sparser_kit_reports_fewer_channels(a_deteriorating_stay):
    full = build_post_discharge_digest(a_deteriorating_stay, llm=None, kit_name="full_home")
    watch = build_post_discharge_digest(a_deteriorating_stay, llm=None, kit_name="watch_only")

    full_channels = {c for b in full.daily_buckets for c in b.means}
    watch_channels = {c for b in watch.daily_buckets for c in b.means}
    assert watch_channels < full_channels
