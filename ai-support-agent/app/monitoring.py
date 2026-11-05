"""Production-grade observability for the AI Support Agent.

Bundles the four observability pillars in one module:

1. **Structured JSON logging** — ``configure_json_logging`` augments the
   existing ``app.logging_config`` formatter with request-scoped context
   (``request_id``, ``user_id``, ``ticket_id``) pulled from contextvars.
2. **OpenTelemetry tracing** — ``setup_tracing`` instruments the FastAPI app
   and provides ``traced_span``/``trace_agent_node``/``trace_tool_call``
   helpers for the agent graph and its tools.
3. **Prometheus metrics** — ``metrics`` registry exposed at ``/metrics`` with
   conversation counters, tool usage, escalation rate, response time and
   ticket-creation rate.
4. **LangSmith + Sentry hooks** — opt-in via environment settings.

All integrations degrade gracefully: missing optional packages or unconfigured
DSN/keys never break the application.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager, asynccontextmanager
from contextvars import ContextVar
from typing import Any, AsyncIterator, Dict, Iterator, Optional

# ---------------------------------------------------------------------------
# Optional dependencies — every import is guarded so the app runs even when
# observability packages are absent (e.g. minimal test environments).
# ---------------------------------------------------------------------------

try:  # Prometheus client
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        REGISTRY,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROMETHEUS_AVAILABLE = False

try:  # OpenTelemetry API/SDK
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    OTEL_AVAILABLE = False

try:  # OTLP exporter (optional — tracing still works without an exporter)
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )

    OTLP_AVAILABLE = True
except ImportError:  # pragma: no cover
    OTLP_AVAILABLE = False

try:  # Sentry error tracking
    import sentry_sdk

    SENTRY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SENTRY_AVAILABLE = False

logger = logging.getLogger("support_agent.monitoring")

# ---------------------------------------------------------------------------
# Request-scoped context (consumed by logs, traces and metrics)
# ---------------------------------------------------------------------------

request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
ticket_id_ctx: ContextVar[Optional[str]] = ContextVar("ticket_id", default=None)
session_id_ctx: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


def set_log_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    ticket_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """Set request-scoped context variables surfaced in every structured log line."""
    if request_id is not None:
        request_id_ctx.set(request_id)
    if user_id is not None:
        user_id_ctx.set(str(user_id))
    if ticket_id is not None:
        ticket_id_ctx.set(str(ticket_id))
    if session_id is not None:
        session_id_ctx.set(str(session_id))


def clear_log_context() -> None:
    """Reset all request-scoped context variables (call at request teardown)."""
    for var in (request_id_ctx, user_id_ctx, ticket_id_ctx, session_id_ctx):
        var.set(None)


def get_log_context() -> Dict[str, Optional[str]]:
    """Return the current request-scoped logging context as a dict."""
    return {
        "request_id": request_id_ctx.get(),
        "user_id": user_id_ctx.get(),
        "ticket_id": ticket_id_ctx.get(),
        "session_id": session_id_ctx.get(),
    }


class ContextualLogFilter(logging.Filter):
    """Injects request_id / user_id / ticket_id into every LogRecord.

    The JSON formatter in ``app.logging_config`` picks these up as ``extra``
    fields so log aggregators can correlate lines across a request, a user and
    a ticket.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.request_id = request_id_ctx.get()
        record.user_id = user_id_ctx.get()
        record.ticket_id = ticket_id_ctx.get()
        record.session_id = session_id_ctx.get()
        return True


def configure_json_logging() -> None:
    """Attach the context filter to the root logger (idempotent)."""
    root = logging.getLogger()
    if not any(isinstance(f, ContextualLogFilter) for f in root.filters):
        root.addFilter(ContextualLogFilter())


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:

    METRICS = Dict[str, Any]  # typing alias

    _registry = REGISTRY

    # --- Counters -----------------------------------------------------------
    conversations_total = Counter(
        "support_agent_conversations_total",
        "Total number of chat turns / conversations processed.",
        ["session_id_kind"],  # 'new' or 'continued'
        registry=_registry,
    )
    tool_usage_total = Counter(
        "support_agent_tool_usage_total",
        "Number of times each agent tool was invoked.",
        ["tool_name", "status"],  # status: success | error
        registry=_registry,
    )
    escalations_total = Counter(
        "support_agent_escalations_total",
        "Number of conversations escalated to a human agent.",
        ["reason"],
        registry=_registry,
    )
    tickets_created_total = Counter(
        "support_agent_tickets_created_total",
        "Number of support tickets created by the agent.",
        ["priority"],
        registry=_registry,
    )
    http_requests_total = Counter(
        "support_agent_http_requests_total",
        "Total HTTP requests handled.",
        ["method", "endpoint", "status_code"],
        registry=_registry,
    )

    # --- Histograms ----------------------------------------------------------
    response_time_seconds = Histogram(
        "support_agent_response_time_seconds",
        "Agent end-to-end response latency (seconds).",
        ["endpoint"],
        buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
        registry=_registry,
    )
    http_request_duration_seconds = Histogram(
        "support_agent_http_request_duration_seconds",
        "HTTP request latency (seconds).",
        ["method", "endpoint"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
        registry=_registry,
    )
    tool_duration_seconds = Histogram(
        "support_agent_tool_duration_seconds",
        "Tool execution latency (seconds).",
        ["tool_name"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 5, 15),
        registry=_registry,
    )
    graph_node_duration_seconds = Histogram(
        "support_agent_graph_node_duration_seconds",
        "LangGraph node execution latency (seconds).",
        ["node"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 5, 15),
        registry=_registry,
    )

    # --- Gauges ---------------------------------------------------------------
    active_sessions = Gauge(
        "support_agent_active_sessions",
        "Number of chat sessions currently active in this process.",
        registry=_registry,
    )

    METRICS_AVAILABLE = True
else:  # pragma: no cover — graceful no-op shims
    METRICS_AVAILABLE = False

    class _NoopMetric:
        def labels(self, *a, **k):
            return self

        def inc(self, *a, **k): ...

        def observe(self, *a, **k): ...

        def set(self, *a, **k): ...

        def time(self):  # context manager fallback
            import contextlib

            return contextlib.nullcontext()

    conversations_total = _NoopMetric()
    tool_usage_total = _NoopMetric()
    escalations_total = _NoopMetric()
    tickets_created_total = _NoopMetric()
    http_requests_total = _NoopMetric()
    response_time_seconds = _NoopMetric()
    http_request_duration_seconds = _NoopMetric()
    tool_duration_seconds = _NoopMetric()
    graph_node_duration_seconds = _NoopMetric()
    active_sessions = _NoopMetric()


def render_metrics() -> bytes:
    """Render the Prometheus exposition format (body for the /metrics route)."""
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus client not installed\n"
    from prometheus_client import generate_latest  # noqa: PLC0415

    return generate_latest(_registry)


def record_conversation(new_session: bool = True) -> None:
    """Increment the conversation counter."""
    conversations_total.labels(session_id_kind="new" if new_session else "continued").inc()


def record_tool_usage(tool_name: str, success: bool = True, duration: Optional[float] = None) -> None:
    """Record one tool invocation (count + latency)."""
    tool_usage_total.labels(tool_name=tool_name, status="success" if success else "error").inc()
    if duration is not None:
        tool_duration_seconds.labels(tool_name=tool_name).observe(duration)


def record_escalation(reason: str = "unknown") -> None:
    """Record an escalation-to-human event."""
    escalations_total.labels(reason=reason).inc()


def record_ticket_creation(priority: str = "unknown") -> None:
    """Record a ticket created by the agent."""
    tickets_created_total.labels(priority=priority).inc()


def record_response_time(endpoint: str, duration_seconds: float) -> None:
    """Record agent end-to-end response latency."""
    response_time_seconds.labels(endpoint=endpoint).observe(duration_seconds)


def observe_conversation_outcome(final_state: Dict[str, Any], duration_seconds: float, endpoint: str = "chat") -> None:
    """Record all conversation-level metrics derived from a finished agent run."""
    record_response_time(endpoint, duration_seconds)

    tools_used = final_state.get("tools_used") or []
    for tool_name in tools_used:
        tool_usage_total.labels(tool_name=str(tool_name), status="success").inc()

    if final_state.get("escalation_needed"):
        record_escalation(str(final_state.get("escalation_reason") or "agent_decision"))

    ticket = final_state.get("ticket_data") or {}
    if ticket and ticket.get("status") == "open":
        record_ticket_creation(str(ticket.get("priority") or "unknown"))
        ticket_value = ticket.get("id") or ticket.get("ticket_number")
        if ticket_value:
            ticket_id_ctx.set(str(ticket_value))


# ---------------------------------------------------------------------------
# OpenTelemetry tracing
# ---------------------------------------------------------------------------

_tracing_configured = False


def setup_tracing(app: Any, settings: Any) -> None:
    """Configure the global TracerProvider and instrument FastAPI.

    - Service resource: ``ai-support-agent`` with environment attribute.
    - Optional OTLP exporter when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set.
    - FastAPI ASGI instrumentation creates a span per HTTP request.
    """
    global _tracing_configured
    if _tracing_configured or not OTEL_AVAILABLE:
        if not OTEL_AVAILABLE:
            logger.warning("opentelemetry not installed — tracing disabled.")
        return

    env = getattr(settings.APP_ENV, "value", str(settings.APP_ENV))
    resource = Resource.create(
        {
            "service.name": "ai-support-agent",
            "service.environment": env,
            "service.version": "0.1.0",
        }
    )
    provider = TracerProvider(resource=resource)

    otlp_endpoint = getattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT", None) or None
    if otlp_endpoint and OTLP_AVAILABLE:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
        logger.info("OTLP trace exporter configured -> %s", otlp_endpoint)
    else:
        logger.info("No OTLP endpoint configured — traces kept in-process only.")

    trace.set_tracer_provider(provider)

    if app is not None:
        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)

    _tracing_configured = True
    logger.info("OpenTelemetry tracing initialized.")


def _tracer():
    if OTEL_AVAILABLE:
        return trace.get_tracer("support_agent")
    return None


@contextmanager
def traced_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    """Generic span context manager; no-ops cleanly when OTEL is unavailable."""
    tracer = _tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        if attributes and span is not None:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, str(value))
        yield span


@asynccontextmanager
async def trace_agent_node(node_name: str, session_id: Optional[str] = None) -> AsyncIterator[Any]:
    """Async span for a LangGraph node execution (also records Prometheus duration)."""
    t0 = time.perf_counter()
    tracer = _tracer()
    ctx_attrs: Dict[str, Any] = {"agent.node": node_name}
    if session_id:
        ctx_attrs["session.id"] = session_id
    if tracer is None:
        yield None
    else:
        with tracer.start_as_current_span(f"agent.node.{node_name}") as span:
            for key, value in ctx_attrs.items():
                span.set_attribute(key, str(value))
            yield span
    graph_node_duration_seconds.labels(node=node_name).observe(time.perf_counter() - t0)


@asynccontextmanager
async def trace_tool_call(tool_name: str, session_id: Optional[str] = None) -> AsyncIterator[Any]:
    """Async span for one tool invocation (also records Prometheus metrics)."""
    t0 = time.perf_counter()
    tracer = _tracer()
    if tracer is None:
        yield None
    else:
        with tracer.start_as_current_span(f"tool.{tool_name}") as span:
            span.set_attribute("tool.name", tool_name)
            if session_id:
                span.set_attribute("session.id", str(session_id))
            yield span
    duration = time.perf_counter() - t0
    tool_duration_seconds.labels(tool_name=tool_name).observe(duration)


def trace_tool_sync(tool_name: str):
    """Decorator for sync tool callables: adds tracing + usage metrics."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            async with trace_tool_call(tool_name or func.__name__):
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    tool_usage_total.labels(tool_name=tool_name or func.__name__, status="error").inc()
                    raise

        import functools
        import inspect

        if inspect.iscoroutinefunction(func):
            return functools.wraps(func)(async_wrapper)
        return func

    return decorator


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Attach an event to the currently active span (no-op without OTEL)."""
    if not OTEL_AVAILABLE:
        return
    span = trace.get_current_span()
    if span is not None and span.is_recording():
        span.add_event(name, attributes or {})


def set_span_attributes(attributes: Dict[str, Any]) -> None:
    """Set attributes on the currently active span (no-op without OTEL)."""
    if not OTEL_AVAILABLE:
        return
    span = trace.get_current_span()
    if span is not None and span.is_recording():
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, str(value))


# ---------------------------------------------------------------------------
# Sentry (placeholder — activates only when SENTRY_DSN is configured)
# ---------------------------------------------------------------------------


def init_sentry(settings: Any) -> bool:
    """Initialize Sentry error tracking when ``SENTRY_DSN`` is set.

    Returns True when Sentry was actually initialized. Environment, release
    and trace sample rates are read from settings with sensible defaults.
    """
    dsn = getattr(settings, "SENTRY_DSN", None)
    if not dsn or not SENTRY_AVAILABLE:
        if dsn and not SENTRY_AVAILABLE:
            logger.warning("SENTRY_DSN set but sentry-sdk not installed — error tracking disabled.")
        return False

    traces_sample_rate = float(getattr(settings, "SENTRY_TRACES_SAMPLE_RATE", 0.1) or 0.1)
    env = getattr(settings.APP_ENV, "value", str(settings.APP_ENV))

    sentry_sdk.init(  # type: ignore[union-attr]
        dsn=dsn,
        environment=env,
        release=getattr(settings, "SENTRY_RELEASE", None) or "ai-support-agent@0.1.0",
        traces_sample_rate=traces_sample_rate,
        attach_stacktrace=True,
        send_default_pii=False,
    )
    logger.info("Sentry error tracking initialized (env=%s, sample_rate=%s).", env, traces_sample_rate)
    return True


def capture_exception(exc: Optional[BaseException] = None) -> None:
    """Forward an exception to Sentry when configured (safe no-op otherwise)."""
    if SENTRY_AVAILABLE:
        sentry_sdk.capture_exception(exc)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# LangSmith (opt-in via env vars / settings)
# ---------------------------------------------------------------------------


def configure_langsmith(settings: Any) -> bool:
    """Enable LangSmith tracing when ``LANGCHAIN_API_KEY`` + flag are set.

    LangChain/LangGraph read these env vars natively; we only mirror the
    settings into ``os.environ`` and report the resulting state.
    """
    import os

    enabled = bool(getattr(settings, "LANGCHAIN_TRACING_V2", False)) and bool(
        getattr(settings, "LANGCHAIN_API_KEY", None)
    )
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enabled else "false"
    if getattr(settings, "LANGCHAIN_API_KEY", None):
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_ENDPOINT"] = getattr(settings, "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ["LANGCHAIN_PROJECT"] = getattr(settings, "LANGCHAIN_PROJECT", "ai-support-agent")

    if enabled:
        logger.info("LangSmith tracing enabled (project=%s).", os.environ["LANGCHAIN_PROJECT"])
    else:
        logger.info("LangSmith tracing disabled (set LANGCHAIN_TRACING_V2 + LANGCHAIN_API_KEY to enable).")
    return enabled


# ---------------------------------------------------------------------------
# One-call bootstrap
# ---------------------------------------------------------------------------


def init_monitoring(settings: Any, app: Any = None) -> Dict[str, bool]:
    """Initialize the full observability stack. Called once at app startup.

    Returns a status dict of which integrations activated.
    """
    configure_json_logging()
    status = {
        "logging": True,
        "metrics": PROMETHEUS_AVAILABLE,
        "tracing": False,
        "langsmith": configure_langsmith(settings),
        "sentry": init_sentry(settings),
    }
    try:
        setup_tracing(app, settings)
        status["tracing"] = OTEL_AVAILABLE
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tracing setup failed (non-fatal): %s", exc)
    logger.info("Observability stack initialized: %s", status)
    return status


__all__ = [
    # bootstrap
    "init_monitoring",
    # logging context
    "set_log_context",
    "clear_log_context",
    "get_log_context",
    "ContextualLogFilter",
    "configure_json_logging",
    # metrics
    "render_metrics",
    "record_conversation",
    "record_tool_usage",
    "record_escalation",
    "record_ticket_creation",
    "record_response_time",
    "observe_conversation_outcome",
    "active_sessions",
    # tracing
    "setup_tracing",
    "traced_span",
    "trace_agent_node",
    "trace_tool_call",
    "trace_tool_sync",
    "add_span_event",
    "set_span_attributes",
    # error tracking
    "init_sentry",
    "capture_exception",
    # langsmith
    "configure_langsmith",
]
