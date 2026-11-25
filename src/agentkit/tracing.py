from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

_configured = False


class JsonlSpanExporter(SpanExporter):
    """Writes one JSON object per span, keeping OTel GenAI attribute names."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self.path.open("a") as fh:
            for span in spans:
                ctx = span.get_span_context()
                parent = span.parent
                record = {
                    "trace_id": f"{ctx.trace_id:032x}",
                    "span_id": f"{ctx.span_id:016x}",
                    "parent_id": f"{parent.span_id:016x}" if parent else None,
                    "name": span.name,
                    "start_ns": span.start_time,
                    "end_ns": span.end_time,
                    "duration_ms": ((span.end_time or 0) - (span.start_time or 0)) / 1e6,
                    "status": span.status.status_code.name,
                    "attributes": dict(span.attributes or {}),
                }
                fh.write(json.dumps(record, default=str) + "\n")
        return SpanExportResult.SUCCESS


def configure(service: str = "deployable-agent-kit", jsonl_path: str | Path | None = None) -> None:
    global _configured
    if _configured:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    path = jsonl_path or os.environ.get("AGENTKIT_TRACE_FILE")
    if path:
        provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(Path(path))))
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True


tracer = trace.get_tracer("agentkit")


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(
                    key, value if isinstance(value, str | int | float | bool) else str(value)
                )
        yield current
