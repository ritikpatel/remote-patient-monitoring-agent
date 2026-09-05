FROM python:3.11-slim AS base

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY services/common/ services/common/
COPY services/clinician-api/app.py services/clinician-api/

ENV PYTHONPATH=/app
EXPOSE 8007
CMD ["/app/.venv/bin/uvicorn", "app:app", "--app-dir", "services/clinician-api", "--host", "0.0.0.0", "--port", "8007"]
