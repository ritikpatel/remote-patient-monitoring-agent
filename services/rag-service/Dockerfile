FROM python:3.11-slim AS base

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY notes_synth/output/notes/ notes_synth/output/notes/
COPY notes_synth/output/fact_ledger.parquet notes_synth/output/fact_ledger.parquet
COPY services/rag-service/app.py services/rag-service/corpus.py services/rag-service/retrieval.py services/rag-service/

ENV PYTHONPATH=/app
EXPOSE 8004
CMD ["/app/.venv/bin/uvicorn", "app:app", "--app-dir", "services/rag-service", "--host", "0.0.0.0", "--port", "8004"]
