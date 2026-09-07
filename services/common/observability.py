"""Prometheus metrics and OpenTelemetry tracing for every service, added once
here rather than nine times -- Phase 8 infra (infra/compose/docker-compose.yml:
Prometheus, Grafana, Jaeger).

Both follow this project's standing pattern for infra that doesn't exist in a
plain `pytest` run: metrics are always safe to enable (`prometheus_client` has no
network dependency, so `instrument_metrics` is called unconditionally and every
service simply gains a real, harmless `/metrics` endpoint); tracing needs an OTLP
collector to actually send spans to, so `instrument_tracing` is a genuine no-op
-- FastAPI/httpx are never monkeypatched -- unless `OTEL_EXPORTER_OTLP_ENDPOINT`
is set, exactly like ingest-gateway's `PUBLISHER_BACKEND` switch.
"""

from __future__ import annotations

import os

from fastapi import FastAPI


def instrument_metrics(app: FastAPI, service_name: str) -> None:
    """Exposes GET /metrics with real request-count/latency histograms for every
    route this app serves, labelled by service so one Prometheus can scrape all
    nine and Grafana can break a dashboard down by `service`.
    """
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(
        app, include_in_schema=False
    )
    app.state.observability_service_name = service_name


def instrument_tracing(app: FastAPI, service_name: str) -> None:
    """Real span export via OTLP/HTTP to Jaeger -- only once
    OTEL_EXPORTER_OTLP_ENDPOINT names a real collector (docker-compose sets this;
    a bare `pytest` run or `uvicorn` without it leaves tracing off, matching
    every other Phase-8-infra-gated switch in this project).
    """
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    # Traces this service's own outgoing httpx calls to its neighbours (e.g.
    # clinician-api -> risk-engine) as child spans of the inbound request, so one
    # Jaeger trace shows the real cross-service call chain, not nine isolated ones.
    HTTPXClientInstrumentor().instrument()
