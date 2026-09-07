"""SMART-on-FHIR scope enforcement, without a live Keycloak.

PROJECT_PLAN.md section 10: "clinician-api ... enforces SMART-on-FHIR scopes."
Section 14 (Phase 8) is where a real OIDC provider (Keycloak) actually issues these
tokens; this module verifies and enforces scopes on whatever bearer token arrives,
which is exactly what clinician-api needs today against locally-issued test tokens
and will need unchanged against Keycloak-issued ones once Phase 8 stands it up --
this is JWT verification + SMART scope matching, not an OIDC client, so nothing here
is Keycloak-specific.

SMART scope shape: "patient/Observation.read", "user/RiskAssessment.write", etc. --
`<compartment>/<resource>.<access>`. `require_scope` checks the token carries a scope
that covers the requested (resource, access) pair, `*` matching any resource/access
in that position.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Local test signing key -- the default (and every test's) path. HS256 + a shared
# secret is the right amount of real for testing scope logic without a live IdP.
LOCAL_TEST_SECRET = "capstone-rpm-local-test-secret-not-for-production"
ALGORITHM = "HS256"

# Phase 8: a real Keycloak (infra/compose/docker-compose.yml) issuing real,
# asymmetric, rotating RS256 tokens -- set both env vars and decode_token verifies
# against Keycloak's own JWKS instead of the local HS256 secret, with everything
# downstream (SmartScope parsing, require_scope) completely unchanged, because
# Keycloak's client scopes (infra/compose/keycloak/realm-export.json) are named
# exactly like this module's SMART scope strings ("patient/Observation.read", ...)
# and land in the same "scope" claim either way.
KEYCLOAK_JWKS_URL = os.environ.get("KEYCLOAK_JWKS_URL")
KEYCLOAK_ISSUER = os.environ.get("KEYCLOAK_ISSUER")

security = HTTPBearer()
_jwks_client: jwt.PyJWKClient | None = None


def _get_jwks_client() -> jwt.PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        assert KEYCLOAK_JWKS_URL is not None
        # PyJWKClient caches keys itself, so a token signed with a still-cached
        # key doesn't refetch the JWKS on every single request.
        _jwks_client = jwt.PyJWKClient(KEYCLOAK_JWKS_URL, cache_keys=True)
    return _jwks_client


@dataclass
class SmartScope:
    compartment: str  # "patient" | "user" | "system"
    resource: str  # a FHIR resource type, or "*"
    access: str  # "read" | "write" | "*"

    @classmethod
    def parse(cls, raw: str) -> SmartScope:
        compartment, _, rest = raw.partition("/")
        resource, _, access = rest.partition(".")
        if not compartment or not resource or not access:
            raise ValueError(f"malformed SMART scope: {raw!r}")
        return cls(compartment, resource, access)

    def covers(self, resource: str, access: str) -> bool:
        resource_ok = self.resource == "*" or self.resource == resource
        access_ok = self.access == "*" or self.access == access
        return resource_ok and access_ok


def issue_local_test_token(subject: str, scopes: list[str], expires_in_s: int = 3600) -> str:
    """For tests and local development only -- mints a token this module can also
    verify, standing in for what Keycloak would issue in Phase 8.
    """
    now = int(time.time())
    payload = {"sub": subject, "scope": " ".join(scopes), "iat": now, "exp": now + expires_in_s}
    return jwt.encode(payload, LOCAL_TEST_SECRET, algorithm=ALGORITHM)


@dataclass
class AuthContext:
    subject: str
    scopes: list[SmartScope]

    def has(self, resource: str, access: str) -> bool:
        return any(s.covers(resource, access) for s in self.scopes)


def decode_token(token: str) -> AuthContext:
    try:
        if KEYCLOAK_JWKS_URL:
            signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=KEYCLOAK_ISSUER,
                # Keycloak-issued tokens carry an "aud" this module has no fixed
                # value for (it depends on which client requested the token) --
                # the scope check below is this module's actual authorization
                # boundary, not the audience claim.
                options={"verify_aud": False},
            )
        else:
            payload = jwt.decode(token, LOCAL_TEST_SECRET, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token expired") from exc
    except jwt.PyJWTError as exc:
        # Covers jwt.InvalidTokenError (malformed/bad-signature tokens) and
        # jwt.PyJWKClientError (Keycloak's JWKS endpoint unreachable, or the
        # token names a key id Keycloak doesn't have) -- both are the caller's
        # credential being unusable, not this service's fault, so both are a
        # real 401, never an unhandled 500.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token") from exc
    raw_scopes = payload.get("scope", "").split()
    scopes = []
    for raw in raw_scopes:
        try:
            scopes.append(SmartScope.parse(raw))
        except ValueError:
            continue  # a malformed scope in the token just doesn't grant anything
    return AuthContext(subject=payload["sub"], scopes=scopes)


def current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> AuthContext:
    return decode_token(credentials.credentials)


def require_scope(resource: str, access: str):
    """FastAPI dependency factory: `Depends(require_scope("Observation", "read"))`."""

    def _check(ctx: AuthContext = Depends(current_user)) -> AuthContext:
        if not ctx.has(resource, access):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"missing required scope covering {resource}.{access}",
            )
        return ctx

    return _check
