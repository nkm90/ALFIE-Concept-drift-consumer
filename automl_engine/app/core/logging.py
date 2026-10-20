import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any

import structlog
from structlog.stdlib import ProcessorFormatter

from app.core.config import get_settings
from app.core.process_log import ProcessLogHandler

_TEN_MB = 10 * 1024 * 1024


def _add_service_name(logger, method_name, event_dict):
    """Structlog processor that injects the configured service name."""
    service = event_dict.get("_service_name", "")
    if service:
        event_dict["service"] = service
    event_dict.pop("_service_name", None)
    return event_dict


def configure_structlog() -> None:
    """Configure structlog processors for both structlog and stdlib loggers.

    Must be called once at startup (idempotent).
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _add_service_name,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib logging into structlog
    formatter = ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            (
                structlog.dev.ConsoleRenderer()
                if _is_dev_mode()
                else structlog.processors.JSONRenderer()
            ),
        ],
        foreign_pre_chain=shared_processors,
    )

    # Replace stdlib root handler(s) with structlog-formatted one
    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Capture app.* log records into the per-request process log payload.
    root.addHandler(ProcessLogHandler())


def configure_service_logging(service_name: str) -> None:
    """Set up a RotatingFileHandler for the given service name.

    Reads:
      - ALFIE_LOG_DIR   — directory for log files (default: ./logs)
      - ALFIE_LOG_LEVEL — logging level name (default: INFO)

    Creates logs/<service_name>.log with maxBytes=10 MB, backupCount=5.
    Attaches the handler to the root logger so all module loggers inherit it.
    Safe to call multiple times; duplicate handlers are not added.
    """
    # Ensure structlog is configured
    configure_structlog()

    log_dir = get_settings().alfie_log_dir
    log_level_name = get_settings().alfie_log_level.upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{service_name}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid adding duplicate handlers if called more than once
    existing_files = {
        h.baseFilename
        for h in root_logger.handlers
        if isinstance(h, RotatingFileHandler)
    }
    if log_file in existing_files:
        return

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _add_service_name,
    ]

    file_formatter = ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=_TEN_MB,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Inject service name into all log entries via contextvars
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(_service_name=service_name)


def _is_dev_mode() -> bool:
    """Check if running in development mode (non-JSON console output)."""
    return get_settings().alfie_log_format.lower() == "console"


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog-bound logger.

    Existing code using ``logging.getLogger(__name__)`` will also work
    through the stdlib bridge, but new code can call this directly for
    structured key-value logging.
    """
    return structlog.get_logger(name)
