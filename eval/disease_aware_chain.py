"""Runs the whole disease-aware escalation chain against one real patient, and
prints what a clinician would actually receive.

    composite deterioration event
      -> alert-service          raise + 4-hourly dedup (R6)
      -> agent-orchestrator     /assess: 8-node graph
           DiseaseContext         what is this patient being treated for
           RiskScorer             deterministic NEWS2/SOFA  +  learned model
           ContextRetriever       this admission's notes  +  guideline corpus
           EscalationDecider      the policy gate (NEWS2 only)
           CarePlanner            what should be done, grounded in the above
      -> notification-gateway   dashboard, push, and the escalation email

Every hop is the real service handler, wired over Starlette's TestClient (a genuine
`httpx.Client` subclass), so nothing here is a mock and nothing needs a listening
socket. The one thing that is *not* real is the email transport: `EMAIL_MODE`
defaults to `dry_run`, so the body is composed exactly as it would be sent and then
not sent (services/common/email.py's four guards).

This exists because the chain is the deliverable and it spans five services -- no
single unit test shows the whole thing, and "it works" is not a claim worth making
from a green test suite alone.

    python eval/disease_aware_chain.py [--stay-id N --hour H]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import duckdb  # noqa: E402
from services.common.testing import load_service_app  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

# Chosen because it exercises every branch of the chain at once:
#   * it escalates on NEWS2's SINGLE-PARAMETER limb (finding F1), not the aggregate
#     -- its ICU tier is only 'medium';
#   * its diagnosis chapter (Circulatory, 41 stays) is the one chapter that earned
#     its own recalibrated cut-point, so the disease-specific threshold is in play;
#   * the promoted model grades it high (p ~ 0.999) at hour 2, inside the model's
#     validated 6-hour scope -- so severity survives the fail-safe and the care plan
#     and the page both fire.
DEFAULT_STAY_ID, DEFAULT_HOUR = 36558922, 2
DB = REPO_ROOT / "warehouse" / "mimic4_demo.db"


def build_chain(db: Path):
    """Wire the five real service apps to each other and return the alert client."""
    risk_engine = load_service_app("risk-engine", REPO_ROOT)
    rag_service = load_service_app("rag-service", REPO_ROOT)
    agent = load_service_app("agent-orchestrator", REPO_ROOT)
    alert_service = load_service_app("alert-service", REPO_ROOT)
    notification_gateway = load_service_app("notification-gateway", REPO_ROOT)

    sys.path.insert(0, str(REPO_ROOT / "services" / "agent-orchestrator"))
    sys.path.insert(0, str(REPO_ROOT / "services" / "alert-service"))
    from nodes import Dependencies
    from services.common.audit import AuditLog
    from store import AlertStore

    agent.set_deps(
        Dependencies(
            db_path=db,
            risk_engine_client=TestClient(risk_engine.app, base_url="http://risk-engine"),
            rag_client=TestClient(rag_service.app, base_url="http://rag"),
            audit_log=AuditLog(Path(tempfile.mkdtemp()) / "audit.db"),
            llm=_llm(),
        )
    )
    # A fresh store every run. alert-service's 4-hourly dedup (R6, E16) is real and
    # persistent, so a second run against the committed alerts.db is correctly
    # treated as a repeat -- no new alert, and therefore no agent call at all. That
    # is right in production and useless in a demo.
    alert_service.set_store(AlertStore(Path(tempfile.mkdtemp()) / "alerts.db"))
    alert_service.set_agent_client(TestClient(agent.app, base_url="http://agent"))
    alert_service.set_notify_client(TestClient(notification_gateway.app, base_url="http://notify"))
    return TestClient(alert_service.app), TestClient(risk_engine.app)


def _llm():
    """The real Groq backend when a key is configured, else None -- which exercises
    CarePlanner's deterministic fallback rather than failing. Importing
    notes_synth.backends also loads .env, which is where the key normally lives."""
    from notes_synth.backends import GroqBackend

    if os.environ.get("GROQ_API_KEY"):
        print("LLM: GroqBackend (real generation)\n")
        return GroqBackend()
    print("LLM: not configured -- CarePlanner will use its deterministic fallback\n")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stay-id", type=int, default=DEFAULT_STAY_ID)
    ap.add_argument("--hour", type=int, default=DEFAULT_HOUR)
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    if not args.db.exists():
        print(f"No warehouse at {args.db} -- run warehouse/build_duckdb.py first")
        return 1

    conn = duckdb.connect(str(args.db), read_only=True)
    row = conn.execute(
        "SELECT n.news2, n.tier_icu, n.dx_group, n.threshold_is_disease_specific, d.dx_title "
        "FROM capstone.news2 n JOIN capstone.disease_context d USING (stay_id) "
        "WHERE n.stay_id = ? AND n.hour = ?",
        [args.stay_id, args.hour],
    ).fetchone()
    conn.close()
    if row is None:
        print(f"No scored hour for stay_id={args.stay_id} hour={args.hour}")
        return 1
    news2, tier_icu, dx_group, disease_specific, dx_title = row

    alert_client, risk_client = build_chain(args.db)

    print(f"Patient ICUStay/{args.stay_id}, ICU hour {args.hour}")
    print(f"  Diagnosis : {dx_title}")
    print(f"  Chapter   : {dx_group} (own cut-point: {bool(disease_specific)})")

    det = risk_client.get(f"/score/{args.stay_id}/{args.hour}").json()
    ml = risk_client.post(f"/score/ml/{args.stay_id}/{args.hour}").json()
    print("\n[1] risk-engine")
    print(
        f"  deterministic : NEWS2 {det['news2']}, tier {det['news2_tier_icu']}, "
        f"escalate={det['escalation_recommended']}"
    )
    print(
        f"  learned model : p={ml.get('probability', float('nan')):.4f}, "
        f"severity={ml.get('severity')}, in_scope={ml.get('in_validated_scope')}"
    )

    print("\n[2] alert-service  POST /alerts")
    body = alert_client.post(
        "/alerts",
        json={
            "patient_ref": f"ICUStay/{args.stay_id}",
            "alert_type": "news2_escalation",
            "severity": "high",
            "message": f"NEWS2 {det['news2']}: {det['news2_tier_icu']} tier",
            "hour": args.hour,
        },
    ).json()
    print(
        f"  was_new={body['was_new']}  regraded_from={body['regraded_from']}  "
        f"final severity={body['alert']['severity']}"
    )

    a = body["assessment"]
    print("\n[3] agent-orchestrator  POST /assess")
    if not a or not a.get("assessable"):
        print(f"  not assessable: {(a or {}).get('reason')}")
        return 0
    print(f"  hour {a['hour']} ({a['hour_basis']}), {len(a['audit_rows'])} audit rows")
    print(f"  comorbidities : {', '.join(a['comorbidities']) or 'none recorded'}")
    print(f"  severity      : {a['severity']} -- {a['severity_source']}")
    print(f"  escalate      : {a['escalate']} -- {a['escalation_reason']}")

    plan = a.get("care_plan")
    print("\n[4] CarePlanner")
    if plan:
        print(f"  generated={plan['generated']}")
        print(f"  grounded in {len(plan['citations'])} passages")
    else:
        print(f"  skipped -- {a.get('care_plan_skipped_reason')}")

    print("\n[5] notification-gateway")
    notif = body["notification"]
    print(f"  channels={notif['channels']}  " f"assessment_attached={notif['assessment_attached']}")
    mail = notif.get("email")
    if not mail:
        print("  no email -- severity below page-worthy")
        return 0
    print(f"  email mode={mail['mode']} sent={mail['sent']}")
    print(f"\n{'=' * 74}\n{mail['body']}\n{'=' * 74}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
