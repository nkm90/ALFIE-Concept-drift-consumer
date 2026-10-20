from app.core.logging import configure_structlog

# Fallback logging configuration — console only, WARNING level.
# Each service calls app.core.logging.configure_service_logging() inside its
# lifespan to add a rotating file handler for its own log file.
configure_structlog()

import logging  # noqa: E402
import os  # noqa: E402

_log_level_name = os.getenv("ALFIE_LOG_LEVEL", "WARNING").upper()
_log_level = getattr(logging, _log_level_name, logging.WARNING)
logging.getLogger().setLevel(_log_level)
