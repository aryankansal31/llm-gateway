"""OpenTelemetry tracing to a self-hosted Arize Phoenix (spec 5.9)."""

from fastapi import FastAPI
from opentelemetry import trace

tracer = trace.get_tracer("llm-gateway")


def setup_tracing(app: FastAPI, endpoint: str | None) -> None:
    """Export spans to Phoenix over OTLP/HTTP, e.g. http://phoenix:6006/v1/traces."""
    if not endpoint:
        return
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {"service.name": "llm-gateway", "openinference.project.name": "llm-gateway"}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    OpenAIInstrumentor().instrument(tracer_provider=provider)
    app.state.tracer_provider = provider
