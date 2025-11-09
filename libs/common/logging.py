"""Structured logging setup."""

import logging
import sys
from typing import Optional

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

from libs.common.config import config


def setup_logging(service_name: str):
    """Setup structured logging and OpenTelemetry tracing."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Setup OpenTelemetry if endpoint provided and package is available
    if HAS_OPENTELEMETRY and config.otlp_endpoint:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

    return logging.getLogger(service_name)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)

