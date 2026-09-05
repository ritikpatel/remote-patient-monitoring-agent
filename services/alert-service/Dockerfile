FROM python:3.11-slim AS base

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY services/alert-service/app.py services/alert-service/store.py services/alert-service/

ENV PYTHONPATH=/app
EXPOSE 8005
CMD ["/app/.venv/bin/uvicorn", "app:app", "--app-dir", "services/alert-service", "--host", "0.0.0.0", "--port", "8005"]
