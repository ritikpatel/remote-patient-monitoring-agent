# Built from the repo root: docker build -f services/risk-engine/Dockerfile .
# Monorepo pattern: every service Dockerfile installs the single shared
# pyproject.toml/uv.lock (one dependency set, one lockfile) then copies in just its
# own service code plus the shared services/common and services/contracts packages
# and the warehouse module it reads from directly.
FROM python:3.11-slim AS base

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY services/common/ services/common/
COPY services/contracts/ services/contracts/
COPY warehouse/news2.py warehouse/build_duckdb.py warehouse/__init__.py warehouse/
COPY services/risk-engine/app.py services/risk-engine/app.py

ENV PYTHONPATH=/app
EXPOSE 8001
# --app-dir, not a dotted module path: "risk-engine" is not a valid Python package
# name (the hyphen), so uvicorn loads app.py directly from that directory instead.
CMD ["/app/.venv/bin/uvicorn", "app:app", "--app-dir", "services/risk-engine", "--host", "0.0.0.0", "--port", "8001"]
