"""Real HAPI FHIR client -- app.py's own module docstring names this exact gap
as "Phase 8 point of contact": POST a resource this service already built (via
mappers.py) to a live HAPI FHIR server's `/fhir/<ResourceType>` endpoint, and
return its response. This is a second, independent validation of every mapped
resource beyond mappers.py's own fhir.resources/Pydantic construction --
proof the resource actually satisfies a real FHIR server's conformance
checking, which is what deliverable 5's acceptance test ("HAPI FHIR validates
every emitted resource", PROJECT_PLAN.md section 3) actually asks for.
"""

from __future__ import annotations

import httpx


class HapiValidationError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"HAPI FHIR rejected the resource (HTTP {status_code}): {body}")


def post_resource(resource: dict, base_url: str, *, timeout: float = 10.0) -> dict:
    """POSTs to base_url/<resourceType> (a create -- HAPI assigns the id) and
    returns the stored resource HAPI echoes back, which carries the id/meta
    HAPI itself assigned -- proof it was actually persisted and validated, not
    just accepted and discarded.
    """
    resource_type = resource["resourceType"]
    resp = httpx.post(
        f"{base_url.rstrip('/')}/{resource_type}",
        json=resource,
        headers={"Content-Type": "application/fhir+json"},
        timeout=timeout,
    )
    if resp.status_code not in (200, 201):
        raise HapiValidationError(resp.status_code, resp.text)
    return resp.json()
