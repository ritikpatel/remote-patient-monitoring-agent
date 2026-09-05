"""rag-service: retrieval over HTTP. See corpus.py and retrieval.py for the real
logic (pgvector substitute: TF-IDF; see retrieval.py's module docstring)."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import load_corpus  # noqa: E402
from retrieval import TfidfIndex  # noqa: E402

app = FastAPI(title="rag-service", version="0.1.0")

# Built once at startup from whatever notes_synth/output currently holds. A real
# deployment rebuilds this whenever notes_synth/generate.py produces new output
# (or, post-Phase-8, the pgvector table is simply queried live -- no rebuild step).
_index: TfidfIndex | None = None


def get_index() -> TfidfIndex:
    global _index
    if _index is None:
        _index = TfidfIndex(load_corpus())
    return _index


def reset_index() -> None:
    """For tests: forces a rebuild against whatever corpus exists right now."""
    global _index
    _index = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "rag-service", "corpus_size": len(get_index().passages)}


class SearchResultOut(BaseModel):
    passage_id: str
    text: str
    source: str
    fact_ids: list[str]
    hadm_id: int | None
    score: float


@app.get("/search", response_model=list[SearchResultOut])
def search(q: str, k: int = 5, source: str | None = None) -> list[SearchResultOut]:
    results = get_index().search(q, k=k, source=source)
    return [
        SearchResultOut(
            passage_id=r.passage.passage_id,
            text=r.passage.text,
            source=r.passage.source,
            fact_ids=r.passage.fact_ids,
            hadm_id=r.passage.hadm_id,
            score=r.score,
        )
        for r in results
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)
