FROM python:3.11-slim AS base

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY simulators/arrival_models.json simulators/arrival_models.json
COPY services/stream-processor/app.py services/stream-processor/windowing.py services/stream-processor/

ENV PYTHONPATH=/app
EXPOSE 8003
CMD ["/app/.venv/bin/uvicorn", "app:app", "--app-dir", "services/stream-processor", "--host", "0.0.0.0", "--port", "8003"]
