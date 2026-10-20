"""Shared async concurrency helpers.

The AutoML request handlers are ``async def`` FastAPI endpoints, but the
service layer they call (``app.core.service_helpers``, the AutoML+ VLM tools,
the AutoML trainers) is synchronous and performs blocking network and CPU
work. Invoking that work directly inside an ``async def`` handler blocks the
event loop and serializes concurrent requests.

``offload`` runs a synchronous callable in the Starlette threadpool so the
event loop stays free to serve other requests. It is a thin, project-standard
seam over ``fastapi.concurrency.run_in_threadpool``: every router goes through
``offload`` instead of calling ``run_in_threadpool`` (or blocking code)
directly, so the offloading strategy can be changed or instrumented in one
place. Exceptions raised by the offloaded callable propagate to the caller
unchanged.
"""

from typing import Any, Callable, TypeVar

from fastapi.concurrency import run_in_threadpool

T = TypeVar("T")


async def offload(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking/synchronous callable off the event loop.

    Equivalent to ``await run_in_threadpool(func, *args, **kwargs)``; exists so
    callers express intent ("push this blocking call off the loop") at the call
    site rather than naming the threading primitive.
    """
    return await run_in_threadpool(func, *args, **kwargs)
