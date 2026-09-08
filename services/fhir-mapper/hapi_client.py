"""Real HAPI FHIR client -- app.py's own module docstring names this exact gap
as "Phase 8 point of contact": publish a resource this service already built (via
mappers.py) to a live HAPI FHIR server and return its response. This is a second,
independent validation of every mapped resource beyond mappers.py's own
fhir.resources/Pydantic construction -- proof the resource actually satisfies a real
FHIR server's conformance checking, which is what deliverable 5's acceptance test
("HAPI FHIR validates every emitted resource", PROJECT_PLAN.md section 3) asks for.

Why this is not a plain POST (review finding F2)
------------------------------------------------
A plain ``POST /fhir/Patient`` is a *create*: the server assigns the id. Our
``Patient/10006053`` came back as ``Patient/2``, so every resource referencing
``Patient/10006053`` -- Encounter, Condition, MedicationAdministration, Procedure,
RiskAssessment, DocumentReference, DiagnosticReport, all of which mappers.py builds
with ``subject=Reference("Patient/{subject_id}")`` -- was then rejected:

    HAPI-1094: Resource Patient/10006053 not found, specified in path: Encounter.subject

So "validates every emitted resource" actually held only for the two resource types
with no outbound references (Patient, Device).

``PUT /fhir/Patient/10006053`` does not fix it either -- HAPI refuses purely numeric
client-assigned ids, which are reserved for server assignment:

    HAPI-0960: ... clients may only assign IDs which contain at least one
    non-numeric character

The FHIR-idiomatic answer, and what this module now does, is to key resources on their
**business identifier** rather than on a server id:

  * every publish goes through a ``transaction`` Bundle;
  * a resource carrying a known business identifier (see IDENTIFIER_SYSTEMS) is sent as
    a **conditional update** -- ``PUT Patient?identifier=urn:mimic-iv:subject_id|10006053``
    -- which matches an existing resource if there is one and creates it otherwise, so
    identity is stable across runs and publishing twice is idempotent;
  * outbound references are rewritten to **conditional references**
    (``Patient?identifier=urn:mimic-iv:subject_id|10006053``), which FHIR resolves
    server-side at transaction time regardless of what local id the referent ended up
    with.

Publishing several resources in one call puts them in one Bundle, so a referent
created in that same transaction resolves for its dependants.
"""

from __future__ import annotations

import copy
from typing import Any

import httpx

# resourceType -> the identifier system mappers.py stamps on it. A type absent from
# this map has no business identifier, so it is created rather than conditionally
# updated (Observation, Condition, MedicationAdministration, Procedure,
# RiskAssessment, DocumentReference -- all leaf resources nothing else references).
IDENTIFIER_SYSTEMS: dict[str, str] = {
    "Patient": "urn:mimic-iv:subject_id",
    "Encounter": "urn:mimic-iv:hadm_id",
    "Device": "urn:capstone-rpm:device_id",
    "DiagnosticReport": "urn:mimic-iv-ecg:study_id",
}


class HapiValidationError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"HAPI FHIR rejected the resource (HTTP {status_code}): {body}")


def _business_identifier(resource: dict) -> tuple[str, str] | None:
    """(system, value) of the identifier this resource type is keyed on, if present."""
    system = IDENTIFIER_SYSTEMS.get(resource.get("resourceType", ""))
    if system is None:
        return None
    for ident in resource.get("identifier") or []:
        if ident.get("system") == system and ident.get("value"):
            return system, str(ident["value"])
    return None


def conditional_url(resource: dict) -> str | None:
    """``Type?identifier=system|value``, or None when the type has no business key."""
    found = _business_identifier(resource)
    if found is None:
        return None
    system, value = found
    return f"{resource['resourceType']}?identifier={system}|{value}"


def rewrite_reference(ref: str) -> str:
    """``Patient/10006053`` -> ``Patient?identifier=urn:mimic-iv:subject_id|10006053``.

    Left untouched when the type has no business identifier, when the reference is
    already conditional, or when it is one of the non-FHIR reference strings the
    Observation contract legitimately carries (``ICUStay/34547401``, ``Subject/S05`` --
    see observation_to_fhir's docstring). Rewriting those would invent a mapping that
    does not exist; they are a separate modelling question from F2.
    """
    if "?" in ref or "/" not in ref:
        return ref
    resource_type, _, local_id = ref.partition("/")
    system = IDENTIFIER_SYSTEMS.get(resource_type)
    if system is None or not local_id:
        return ref
    return f"{resource_type}?identifier={system}|{local_id}"


def rewrite_references(node: Any) -> Any:
    """Recursively rewrite every ``{"reference": ...}`` in a resource. Returns a copy;
    the caller's resource is never mutated.
    """
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "reference" and isinstance(value, str):
                out[key] = rewrite_reference(value)
            else:
                out[key] = rewrite_references(value)
        return out
    if isinstance(node, list):
        return [rewrite_references(item) for item in node]
    return node


def build_transaction_bundle(resources: list[dict]) -> dict:
    """One transaction Bundle: conditional update where the type has a business
    identifier, plain create otherwise, with every reference made conditional.
    """
    entries = []
    for original in resources:
        resource = rewrite_references(copy.deepcopy(original))
        cond = conditional_url(resource)
        # A conditional update must not carry a client-assigned id -- the match is on
        # the identifier, and a numeric id here is exactly what HAPI-0960 rejects.
        resource.pop("id", None)
        request = (
            {"method": "PUT", "url": cond}
            if cond
            else {"method": "POST", "url": resource["resourceType"]}
        )
        entries.append({"resource": resource, "request": request})
    return {"resourceType": "Bundle", "type": "transaction", "entry": entries}


def publish_resources(resources: list[dict], base_url: str, *, timeout: float = 30.0) -> list[dict]:
    """Publishes resources as one transaction and returns the stored resource for each,
    in the order given. ``Prefer: return=representation`` asks HAPI to echo each stored
    resource back, so the caller still sees the real server-assigned id and meta --
    the same proof-of-persistence the previous POST-based implementation returned.
    """
    if not resources:
        return []
    bundle = build_transaction_bundle(resources)
    resp = httpx.post(
        base_url.rstrip("/"),
        json=bundle,
        headers={
            "Content-Type": "application/fhir+json",
            "Prefer": "return=representation",
        },
        timeout=timeout,
    )
    if resp.status_code not in (200, 201):
        raise HapiValidationError(resp.status_code, resp.text)

    body = resp.json()
    stored = []
    for entry in body.get("entry", []):
        response = entry.get("response", {})
        status = str(response.get("status", ""))
        if not status.startswith(("200", "201")):
            raise HapiValidationError(
                int(status[:3]) if status[:3].isdigit() else 502,
                entry.get("response", {}).get("outcome") or resp.text,
            )
        # With Prefer: return=representation HAPI echoes the resource; some
        # configurations return only a location, so fall back to fetching it.
        resource = entry.get("resource")
        if resource is None and response.get("location"):
            got = httpx.get(f"{base_url.rstrip('/')}/{response['location']}", timeout=timeout)
            resource = got.json() if got.status_code == 200 else {}
        stored.append(resource or {})
    return stored


def post_resource(resource: dict, base_url: str, *, timeout: float = 30.0) -> dict:
    """Single-resource publish, kept for the existing call site. Now idempotent and
    reference-safe, since it goes through the same transaction path.
    """
    stored = publish_resources([resource], base_url, timeout=timeout)
    return stored[0] if stored else {}
