# Built from the repo root: docker build -f services/risk-engine/Dockerfile .
# Monorepo pattern: every service Dockerfile installs the single shared
# pyproject.toml/uv.lock (one dependency set, one lockfile) then copies in just its
# own service code plus the shared services/common and services/contracts packages
# and the warehouse module it reads from directly.
FROM python:3.11-slim AS base

# LightGBM's compiled core links against GNU OpenMP at runtime, which
# python:3.11-slim does not ship (found for real: /score/ml 500'd with
# "OSError: libgomp.so.1: cannot open shared object file" the moment the
# promoted model's joblib.load() tried to import lightgbm).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY services/common/ services/common/
COPY services/contracts/ services/contracts/
COPY warehouse/news2.py warehouse/build_duckdb.py warehouse/__init__.py warehouse/
COPY services/risk-engine/app.py services/risk-engine/app.py

# Phase 5: /score/ml serves ml/models/serving.py's promoted-model lookup,
# which pulls in the same feature engineering the training pipeline used
# (ml/features/) -- including its reuse of stream-processor's windowing.py.
# The actual model file under ml/models/promoted/ is excluded by
# .dockerignore (it's a build/registry artefact -- "not a pickle in a
# folder", PROJECT_PLAN.md section 11 -- produced by
# `python ml/evaluation/run_all.py` and pulled from the registry or mounted
# at deploy time, not baked into the image), so this COPY brings in an
# empty promoted/ directory. Its absence is handled for real -- /score/ml
# returns a genuine 503, not a fabricated prediction.
COPY ml/ ml/
COPY services/stream-processor/windowing.py services/stream-processor/windowing.py

ENV PYTHONPATH=/app
EXPOSE 8001
# --app-dir, not a dotted module path: "risk-engine" is not a valid Python package
# name (the hyphen), so uvicorn loads app.py directly from that directory instead.
CMD ["/app/.venv/bin/uvicorn", "app:app", "--app-dir", "services/risk-engine", "--host", "0.0.0.0", "--port", "8001"]
