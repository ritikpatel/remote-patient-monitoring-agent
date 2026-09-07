FROM python:3.11-slim AS base

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY simulators/arrival_models.json simulators/arrival_models.json
COPY services/common/ services/common/
COPY services/contracts/ services/contracts/
# kafka_consumer.py: the real Kafka consumer half of the raw.* topic contract
# (Phase 8) -- needs services/common (observability) and services/contracts
# (Observation) above it, which is why this COPY is listed after them.
COPY services/stream-processor/app.py services/stream-processor/windowing.py services/stream-processor/kafka_consumer.py services/stream-processor/

ENV PYTHONPATH=/app
EXPOSE 8003
CMD ["/app/.venv/bin/uvicorn", "app:app", "--app-dir", "services/stream-processor", "--host", "0.0.0.0", "--port", "8003"]
