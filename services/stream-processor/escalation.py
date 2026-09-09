"""Closes the loop from a streamed Observation to a raised alert (finding F3).

Before this module the pipeline had every piece but no wiring: ingest-gateway
published to Kafka, stream-processor windowed what it consumed, risk-engine scored,
alert-service deduped and escalated -- but nothing joined them. The only thing that
ever drove the full chain was `eval/load/ramp.js`, which called each service in turn
itself. That made deliverable 1's acceptance test ("live watch + both replays raise
alerts through one engine") undemonstrable: the engine worked, and nothing reached it.

``EscalationLoop`` is that wiring, and it lives here rather than in a tenth service
because stream-processor is already the component that sees every observation.

What it deliberately does NOT do: call agent-orchestrator. The agent graph makes LLM
calls and is the right thing to run *once a patient is alerting*, from the dashboard
or on demand -- not once per observation. The policy applied here is the same
``should_escalate`` predicate the agent's EscalationDecider uses, served by
risk-engine's ``/score/live``, so the cheap deterministic path and the expensive
narrative path can never disagree about whether to escalate.

``run_now()`` is the second entry point: a one-shot, synchronous version of
``on_observation()`` for a caller that already holds a complete vitals dict
instead of a single streamed channel -- ``event-studio``'s composed events,
not a Kafka-consumed ``Observation``. Same real services, same score -> alert
sequence, same order, just called directly instead of triggered by a consumed
message. See ``services/event-studio/app.py``.

**This module does NOT call notification-gateway itself, and used to --
that was a real bug, found while wiring the SMS sender in.** alert-service's
own `POST /alerts` handler already calls notification-gateway internally on
every genuinely new alert (`_notify_dashboard`, in `services/alert-service/
app.py`) and, since that fix, returns the result. A second, independent call
from here as well meant every real alert notified *twice* -- harmless for a
WebSocket toast, but this project also wires a guarded SMS sender to fire on
every high-severity notification (`services/common/sms.py`), and that turns
"harmless duplicate" into "a clinician's phone gets the drill text twice."
`_raise_alert` below now just reads `alert-service`'s own embedded
`"notification"` field rather than requesting a second one.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.contracts.observation import CHANNELS, Observation  # noqa: E402

# code -> channel name, so a consumer that only sees LOINC codes on the wire can map
# back to the vitals field names risk-engine's /score/live expects.
CODE_TO_CHANNEL: dict[str, str] = {ch.code: name for name, ch in CHANNELS.items()}

# Only these channels change a NEWS2 score. A wearable streaming BVP at 64 Hz must not
# trigger a scoring round trip per sample.
SCORING_CHANNELS = frozenset({"hr", "rr", "spo2", "sbp", "temp_c", "gcs_total", "fio2"})

GCS_TRAJECTORY_HOURS = 4.0
# Throttle in STREAM time (effective_time), not wall-clock time. A replay under
# --compress 3600 (or --no-sleep) delivers 40 hours of ICU time in under a second, so
# a wall-clock throttle silently drops the entire stay after its first observation --
# found exactly that way: 336 observations went in, one was scored, no alert came out.
# Fifteen simulated minutes means the same thing whether the stream is live or
# compressed, which is the property a replay-and-live pipeline needs.
DEFAULT_MIN_SCORE_INTERVAL_MIN = 15.0
ALERT_TYPE = "news2_escalation"


@dataclass
class EscalationLoop:
    """Scores a patient when a NEWS2-relevant channel arrives, and raises an alert
    through the real alert-service when the shared policy says to escalate.

    ``min_score_interval_min`` throttles per patient in *stream* time: scoring on all
    nine channels of every simulated hour is nine round trips per patient-hour for no
    added signal, and alert-service's 4-hourly dedup collapses the repeats into one
    alert anyway (R6). Measuring the interval in effective_time rather than wall clock
    is what makes the throttle behave identically for a live watch and a 3600x replay.
    """

    risk_engine_url: str
    alert_service_url: str
    min_score_interval_min: float = DEFAULT_MIN_SCORE_INTERVAL_MIN
    timeout: float = 5.0
    # Injectable for tests (an httpx.Client wired to an httpx.MockTransport, or
    # to another service's real ASGI app) -- real callers leave this None and
    # get a real network-calling client, exactly as before this field existed.
    client: httpx.Client | None = None

    scored: int = 0
    escalated: int = 0
    alerts_raised: int = 0
    errors: int = 0

    _last_scored_at: dict[str, datetime] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _client: httpx.Client | None = None

    def __post_init__(self) -> None:
        self._client = (
            self.client if self.client is not None else httpx.Client(timeout=self.timeout)
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def _due(self, patient_ref: str, effective_time: datetime) -> bool:
        """Stream-time throttle. Out-of-order arrivals within a simulated hour must not
        reset the clock backwards, so a timestamp older than the last one scored is
        simply not due.
        """
        with self._lock:
            last = self._last_scored_at.get(patient_ref)
            if last is not None:
                elapsed_min = (effective_time - last).total_seconds() / 60
                if elapsed_min < self.min_score_interval_min:
                    return False
            self._last_scored_at[patient_ref] = effective_time
            return True

    def on_observation(self, store: object, obs: Observation) -> dict | None:
        """Called for every consumed Observation. Returns the alert payload when one
        was raised, None otherwise. Never raises: a scoring or alerting failure must
        not take the consumer thread down (same rule as the consumer's own loop).
        """
        channel = CODE_TO_CHANNEL.get(obs.code)
        if channel not in SCORING_CHANNELS or not self._due(obs.patient_ref, obs.effective_time):
            return None
        try:
            vitals = self._collect_vitals(store, obs.patient_ref)
            if not vitals:
                return None
            score = self._score(obs.patient_ref, vitals)
            self.scored += 1
            if not score.get("escalation_recommended"):
                return None
            self.escalated += 1
            return self._raise_alert(obs.patient_ref, score)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            self.errors += 1
            print(f"  [EscalationLoop] {type(exc).__name__}: {exc}")
            return None

    def _collect_vitals(self, store: object, patient_ref: str) -> dict:
        """Latest value per NEWS2 channel for this patient, plus the recent best GCS
        so risk-engine can evaluate the GCS-drop limb. Sedation is not sent: a stream
        carries no medication feed, and /score/live is explicit about treating an
        unknown sedation status as "not sedated" (the safer error) and saying so.
        """
        latest = store.latest_values(patient_ref)  # type: ignore[attr-defined]
        vitals = {
            CODE_TO_CHANNEL[code]: value
            for code, value in latest.items()
            if CODE_TO_CHANNEL.get(code) in SCORING_CHANNELS
        }
        if not vitals:
            return {}
        gcs_code = CHANNELS["gcs_total"].code
        prev_max = store.recent_max(patient_ref, gcs_code, GCS_TRAJECTORY_HOURS)  # type: ignore[attr-defined]
        if prev_max is not None:
            vitals["gcs_prev_max"] = prev_max
        return vitals

    def _score(self, patient_ref: str, vitals: dict) -> dict:
        assert self._client is not None
        resp = self._client.post(
            f"{self.risk_engine_url.rstrip('/')}/score/live",
            json={"patient_ref": patient_ref, "vitals": vitals},
        )
        resp.raise_for_status()
        return resp.json()

    def _raise_alert(self, patient_ref: str, score: dict) -> dict | None:
        """POSTs the alert; alert-service itself calls notification-gateway on a
        genuinely new one and embeds the result as `"notification"` in its
        response (see the module docstring for why that call does not also
        happen here). This method only reads that field back, never requests
        a second one.
        """
        assert self._client is not None
        resp = self._client.post(
            f"{self.alert_service_url.rstrip('/')}/alerts",
            json={
                "patient_ref": patient_ref,
                "alert_type": ALERT_TYPE,
                "severity": "high",
                "message": f"NEWS2 {score['news2']}: {score['escalation_reason']}",
            },
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("was_new"):
            self.alerts_raised += 1
        return body

    def run_now(self, patient_ref: str, vitals: dict) -> dict:
        """One-shot, synchronous version of ``on_observation()`` for a caller that
        already has a complete vitals dict rather than a single streamed channel.

        Skips the per-channel filter and the stream-time throttle: both exist to
        stop a 64Hz wearable stream from triggering a scoring round trip on every
        sample, which does not apply to a caller making one deliberate call with
        every channel already in hand. Otherwise this is exactly
        ``on_observation``'s score -> alert sequence, against the same real
        services in the same order -- what "the event reaches risk-engine and
        alert-service immediately" means concretely. alert-service notifies
        (dashboard, push, and a high-severity SMS) internally on a new alert
        and returns that result embedded on the alert body; it is not a
        separate step this method takes.
        """
        try:
            score = self._score(patient_ref, vitals)
            self.scored += 1
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            self.errors += 1
            return {"scored": False, "escalated": False, "error": f"{type(exc).__name__}: {exc}"}

        if not score.get("escalation_recommended"):
            return {"scored": True, "escalated": False, "score": score}

        self.escalated += 1
        try:
            alert = self._raise_alert(patient_ref, score)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            self.errors += 1
            return {
                "scored": True,
                "escalated": True,
                "score": score,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {"scored": True, "escalated": True, "score": score, "alert": alert}
