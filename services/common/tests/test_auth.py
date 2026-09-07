import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from services.common.auth import (
    SmartScope,
    decode_token,
    issue_local_test_token,
    require_scope,
)


def test_smart_scope_parse():
    s = SmartScope.parse("patient/Observation.read")
    assert s.compartment == "patient"
    assert s.resource == "Observation"
    assert s.access == "read"


def test_smart_scope_wildcard_covers_anything():
    s = SmartScope.parse("user/*.read")
    assert s.covers("Observation", "read")
    assert s.covers("Condition", "read")
    assert not s.covers("Observation", "write")


def test_malformed_scope_raises():
    with pytest.raises(ValueError):
        SmartScope.parse("not-a-scope")


def test_decode_token_round_trip():
    token = issue_local_test_token("clinician:jdoe", ["patient/Observation.read"])
    ctx = decode_token(token)
    assert ctx.subject == "clinician:jdoe"
    assert ctx.has("Observation", "read")
    assert not ctx.has("Observation", "write")


def test_expired_token_rejected():
    token = issue_local_test_token("clinician:jdoe", ["patient/*.read"], expires_in_s=-10)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        decode_token(token)
    assert exc_info.value.status_code == 401


def _make_test_app() -> FastAPI:
    app = FastAPI()

    @app.get("/observations")
    def read_observations(ctx=Depends(require_scope("Observation", "read"))):
        return {"subject": ctx.subject}

    return app


def test_require_scope_dependency_allows_authorized_request():
    client = TestClient(_make_test_app())
    token = issue_local_test_token("clinician:jdoe", ["patient/Observation.read"])
    resp = client.get("/observations", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json() == {"subject": "clinician:jdoe"}


def test_require_scope_dependency_rejects_missing_scope():
    client = TestClient(_make_test_app())
    token = issue_local_test_token("clinician:jdoe", ["patient/Condition.read"])
    resp = client.get("/observations", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_require_scope_dependency_rejects_no_token():
    client = TestClient(_make_test_app())
    resp = client.get("/observations")
    assert resp.status_code in (401, 403)  # HTTPBearer itself returns 403 with no header


# --------------------------------------------------------------------------
# Real Keycloak: self-skips if no realm is actually reachable, the same
# pattern used throughout Phase 8 for infra this project does not assume is
# always running (see eval/tests/test_latency.py, test_kafka_consumer.py).
# --------------------------------------------------------------------------

import socket  # noqa: E402

import httpx  # noqa: E402

import services.common.auth as auth_module  # noqa: E402

KEYCLOAK_BASE = "http://localhost:8180"
REALM = "capstone-rpm"


def _keycloak_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 8180), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _keycloak_reachable(),
    reason="no Keycloak reachable at localhost:8180 -- see infra/compose/README.md",
)
def test_decode_token_verifies_a_real_keycloak_issued_token(monkeypatch):
    """The real, Phase-8-infra-dependent path: fetch an actual access token from
    a live Keycloak realm (infra/compose/keycloak/realm-export.json's
    `clinician-api` client, client_credentials grant), and prove
    `decode_token` verifies it against Keycloak's real JWKS (RS256, not the
    local HS256 test secret) and recovers exactly the SMART scopes that realm
    config grants -- the actual OIDC issuer PROJECT_PLAN.md section 14 calls
    for, not a second local stand-in for it.
    """
    monkeypatch.setattr(
        auth_module,
        "KEYCLOAK_JWKS_URL",
        f"{KEYCLOAK_BASE}/realms/{REALM}/protocol/openid-connect/certs",
    )
    monkeypatch.setattr(auth_module, "KEYCLOAK_ISSUER", f"{KEYCLOAK_BASE}/realms/{REALM}")
    monkeypatch.setattr(auth_module, "_jwks_client", None)  # don't reuse another test's cache

    resp = httpx.post(
        f"{KEYCLOAK_BASE}/realms/{REALM}/protocol/openid-connect/token",
        data={
            "grant_type": "client_credentials",
            "client_id": "clinician-api",
            "client_secret": "capstone-rpm-dev-not-a-real-secret",
            "scope": "patient/Observation.read patient/RiskAssessment.read",
        },
        timeout=10,
    )
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]

    ctx = auth_module.decode_token(token)
    assert ctx.has("Observation", "read")
    assert ctx.has("RiskAssessment", "read")
    assert not ctx.has("Communication", "write")
