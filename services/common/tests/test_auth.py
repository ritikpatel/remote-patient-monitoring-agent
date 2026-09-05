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
