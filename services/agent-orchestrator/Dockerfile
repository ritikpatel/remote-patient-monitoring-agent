FROM python:3.11-slim AS base

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY services/common/ services/common/
COPY notes_synth/backends.py notes_synth/facts.py notes_synth/__init__.py notes_synth/
COPY services/agent-orchestrator/app.py services/agent-orchestrator/graph.py services/agent-orchestrator/nodes.py services/agent-orchestrator/state.py services/agent-orchestrator/

ENV PYTHONPATH=/app
EXPOSE 8008
CMD ["/app/.venv/bin/uvicorn", "app:app", "--app-dir", "services/agent-orchestrator", "--host", "0.0.0.0", "--port", "8008"]
