FROM python:3.11-slim AS base

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY services/common/ services/common/
COPY services/contracts/ services/contracts/
COPY warehouse/build_duckdb.py warehouse/__init__.py warehouse/
COPY services/fhir-mapper/app.py services/fhir-mapper/mappers.py services/fhir-mapper/

ENV PYTHONPATH=/app
EXPOSE 8002
CMD ["/app/.venv/bin/uvicorn", "app:app", "--app-dir", "services/fhir-mapper", "--host", "0.0.0.0", "--port", "8002"]
