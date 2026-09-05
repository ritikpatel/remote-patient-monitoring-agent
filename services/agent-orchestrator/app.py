"""agent-orchestrator: the LangGraph agent graph over HTTP.

See nodes.py's module docstring for the three constraints this service enforces
(the LLM never scores; the policy engine reads the LLM, not the reverse; every
step is audited) and graph.py for the six-node graph itself.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from graph import build_graph  # noqa: E402
from nodes import Dependencies, LLMBackend  # noqa: E402
from notes_synth.backends import GroqBackend  # noqa: E402
from services.common.audit import AuditLog  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "warehouse" / "mimic4_demo.db"
DEFAULT_AUDIT_DB_PATH = Path(__file__).resolve().parent / "agent_audit.db"

RISK_ENGINE_URL = os.environ.get("RISK_ENGINE_URL", "http://localhost:8001")
RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://localhost:8004")

app = FastAPI(title="agent-orchestrator", version="0.1.0")
_deps: Dependencies | None = None


def _default_llm() -> LLMBackend | None:
    if os.environ.get("GROQ_API_KEY"):
        return GroqBackend()
    return None  # falls back to the deterministic no-LLM path in nodes.py


def get_deps() -> Dependencies:
    global _deps
    if _deps is None:
        _deps = Dependencies(
            db_path=DEFAULT_DB_PATH,
            risk_engine_client=httpx.Client(base_url=RISK_ENGINE_URL),
            rag_client=httpx.Client(base_url=RAG_SERVICE_URL),
            audit_log=AuditLog(DEFAULT_AUDIT_DB_PATH),
            llm=_default_llm(),
        )
    return _deps


def set_deps(deps: Dependencies) -> None:
    global _deps
    _deps = deps


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "agent-orchestrator",
        "llm_configured": get_deps().llm is not None,
    }


class RunRequest(BaseModel):
    stay_id: int
    hour: int
    patient_ref: str


@app.post("/run")
def run(req: RunRequest) -> dict:
    deps = get_deps()
    graph = build_graph(deps)
    result = graph.invoke(
        {"stay_id": req.stay_id, "hour": req.hour, "patient_ref": req.patient_ref, "audit_rows": []}
    )
    return dict(result)


@app.get("/audit/verify")
def verify_audit() -> dict:
    ok, bad_seq = get_deps().audit_log.verify_chain()
    return {"intact": ok, "first_broken_seq": bad_seq}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8008)
