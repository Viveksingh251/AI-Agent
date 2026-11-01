"""Structured JSON Logging System for AI Support Agent.

Provides JSON-formatted output for centralized log aggregators (ELK, Datadog, CloudWatch)
with contextual metadata (timestamp, environment, severity, module, line, exception traces).
"""

import datetime
import json
import logging
import sys
import traceback
from typing import Any, Dict, Optional

from app.config import Settings, get_settings


class JSONLogFormatter(logging.Formatter):
    """Formats log records into standard JSON objects for automated ingestion."""

    def __init__(
        self,
        environment: str = "development",
        service_name: str = "ai-support-agent",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.environment = environment
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        """Transforms a standard LogRecord into a JSON string."""
        # 1. Base log structure
        log_payload: Dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "environment": self.environment,
            "location": {
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "path": record.pathname,
            },
            "process": {
                "pid": record.process,
                "thread_name": record.threadName,
            },
        }

        # 2. Extract exception information if available
        if record.exc_info:
            log_payload["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "UnknownException",
                "message": str(record.exc_info[1]) if record.exc_info[1] else "",
                "stack_trace": traceback.format_exception(*record.exc_info),
            }

        # 3. Capture any custom extra fields passed in logger calls
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message",
        }
        extra_data = {
            key: value
            for key, value in record.__dict__.items()
            if key not in standard_attrs and not key.startswith("_")
        }
        if extra_data:
            log_payload["extra"] = extra_data

        return json.dumps(log_payload, default=str)


class StandardConsoleFormatter(logging.Formatter):
    """Human-friendly readable console log formatter for local development."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            *args,
            **kwargs,
        )


def setup_logging(settings: Optional[Settings] = None) -> None:
    """Configures root logger and standard library logging handlers based on application settings.
    
    Args:
        settings: Application Settings instance. Defaults to get_settings().
    """
    _settings = settings or get_settings()
    log_level = getattr(logging, _settings.LOG_LEVEL.upper(), logging.INFO)

    # Choose formatter based on configuration
    if _settings.LOG_JSON_FORMAT:
        formatter = JSONLogFormatter(
            environment=str(_settings.APP_ENV.value if hasattr(_settings.APP_ENV, "value") else _settings.APP_ENV),
            service_name=_settings.APP_NAME,
        )
    else:
        formatter = StandardConsoleFormatter()

    # Configure stdout stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)

    # Reset root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)

    # Intercept third-party loggers for consistent formatting
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "sqlalchemy.engine"):
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.handlers.clear()
        third_party_logger.addHandler(stream_handler)
        third_party_logger.propagate = False
        # Suppress verbose debug SQL unless explicitly requested
        if logger_name == "sqlalchemy.engine" and not _settings.DB_ECHO:
            third_party_logger.setLevel(logging.WARNING)

    logging.getLogger("support_agent.logging").info(
        f"Logging system initialized (level={_settings.LOG_LEVEL}, json_format={_settings.LOG_JSON_FORMAT}, env={_settings.APP_ENV})"
    )

