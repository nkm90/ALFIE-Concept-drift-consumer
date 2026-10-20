"""Per-request transformation log ("process log").

Collects a structured, chronological record of what worked and what failed
during a request and exposes it as the ``process_log`` field of the JSON
response payload.

Two kinds of entries are collected:

* ``log`` entries — INFO+ records emitted by ``app.*`` loggers (the same
  messages that go to the console/rotating file logs). They are captured by
  :class:`ProcessLogHandler`, which is attached once to the root logger in
  ``app.core.logging.configure_structlog``.
* ``step`` entries — explicit markers appended via :func:`log_step` or the
  :func:`step` context manager by orchestrators and routers.

The entry list for the current request lives in a ``ContextVar``. The list is
only ever mutated (never rebound), so it stays shared with the threadpool
workers spawned by ``offload()`` (child contexts are copied, but the list
reference is the same object) while remaining isolated between requests.

No-op semantics: when no process log was started (e.g. services or
orchestrators used directly in tests), every helper here is a no-op and
``get_process_log`` returns an empty list, so call sites never need to guard.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Iterator

_entries: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "process_log_entries", default=None
)

#: Safety cap on captured ``log``-type entries so a chatty training run
#: cannot blow up the response payload. Step markers are never dropped.
_MAX_LOG_ENTRIES = 500


def start_process_log(task_id: str | None = None) -> None:
    """Start collecting a fresh process log for the current request."""
    first: dict[str, Any] = {
        "timestamp": _now_iso(),
        "type": "step",
        "step": "request_received",
        "status": "ok",
    }
    if task_id:
        first["task_id"] = task_id
    _entries.set([first])


def get_process_log() -> list[dict[str, Any]]:
    """Return a copy of the collected process-log entries (empty if none)."""
    entries = _entries.get()
    return list(entries) if entries else []


def log_step(step: str, status: str = "ok", **details: Any) -> None:
    """Append an explicit step marker to the current process log.

    ``status`` is typically ``"ok"`` or ``"failed"``; any extra keyword
    arguments (e.g. ``error=str(e)``) are merged into the entry, dropping
    ``None`` values.
    """
    entries = _entries.get()
    if entries is None:
        return
    entry: dict[str, Any] = {
        "timestamp": _now_iso(),
        "type": "step",
        "step": step,
        "status": status,
    }
    entry.update({k: v for k, v in details.items() if v is not None})
    entries.append(entry)


@contextmanager
def step(name: str, **details: Any) -> Iterator[None]:
    """Mark a pipeline step: ``status="ok"`` on success, ``"failed"`` on error.

    Usage::

        with step("download"):
            zip_path = await offload(download_dataset, ...)
    """
    try:
        yield
    except Exception as e:
        log_step(name, status="failed", error=str(e), **details)
        raise
    log_step(name, status="ok", **details)


class ProcessLogHandler(logging.Handler):
    """Capture INFO+ records from ``app.*`` loggers into the process log."""

    def emit(self, record: logging.LogRecord) -> None:
        entries = _entries.get()
        if entries is None:
            return
        if not record.name.startswith("app.") or record.levelno < logging.INFO:
            return

        log_seen = sum(1 for e in entries if e.get("type") == "log")
        if log_seen >= _MAX_LOG_ENTRIES:
            if log_seen == _MAX_LOG_ENTRIES:
                entries.append(
                    {
                        "timestamp": _now_iso(),
                        "type": "log",
                        "level": "WARNING",
                        "logger": __name__,
                        "message": (
                            f"Process log truncated at {_MAX_LOG_ENTRIES} "
                            "captured log entries"
                        ),
                    }
                )
            return

        entries.append(
            {
                "timestamp": _now_iso(),
                "type": "log",
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="milliseconds")
