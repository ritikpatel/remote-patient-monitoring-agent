FROM python:3.11-slim AS base

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY services/common/ services/common/
COPY services/contracts/ services/contracts/
COPY services/ingest-gateway/app.py services/ingest-gateway/

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "app:app", "--app-dir", "services/ingest-gateway", "--host", "0.0.0.0", "--port", "8000"]
