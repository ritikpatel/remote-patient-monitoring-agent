FROM python:3.11-slim AS base

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY services/common/ services/common/
COPY services/notification-gateway/app.py services/notification-gateway/push.py services/notification-gateway/routing.py services/notification-gateway/

ENV PYTHONPATH=/app
EXPOSE 8006
CMD ["/app/.venv/bin/uvicorn", "app:app", "--app-dir", "services/notification-gateway", "--host", "0.0.0.0", "--port", "8006"]
