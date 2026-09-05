"""Local TF-IDF retrieval, standing in for pgvector until Phase 8 stands up
Postgres.

PROJECT_PLAN.md section 10 says pgvector; Postgres does not exist in this
environment (Phase 8 infra). TF-IDF + cosine similarity is a real, classical
lexical-retrieval method (not a mock) -- swapping it for pgvector-backed neural
embeddings later changes the vectorisation and storage, not the `search(query, k)`
interface agent-orchestrator's ContextRetriever calls.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import Passage  # noqa: E402


@dataclass
class SearchResult:
    passage: Passage
    score: float


class TfidfIndex:
    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages
        self._vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
        self._matrix = (
            self._vectorizer.fit_transform([p.text for p in passages]) if passages else None
        )

    def search(self, query: str, k: int = 5, source: str | None = None) -> list[SearchResult]:
        if not self.passages:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(zip(self.passages, scores, strict=True), key=lambda t: t[1], reverse=True)
        if source is not None:
            ranked = [(p, s) for p, s in ranked if p.source == source]
        return [SearchResult(p, float(s)) for p, s in ranked[:k] if s > 0]
