"""Orchestration pipeline for web accessibility analysis.

This module coordinates the full accessibility analysis workflow: it splits an
HTML document into chunks, groups them into batches (several chunks per LLM
request, controlled by ``chunks_per_request``), fans out concurrent
LLM-over-text analysis via ``_process_chunk_batch``, and aggregates results.
It is intentionally thin — all tool logic lives in ``app.automlplus.tools``.

- ``run_accessibility_pipeline`` — main entry point; returns a list of
  ``ChunkResult`` objects, one per chunk.
- ``resolve_coroutines`` — utility to recursively await coroutine-valued
  attributes when serialising results.
- ``stream_accessibility_results`` — streams the resolved results as a single
  JSON array (used for streaming response endpoints).
"""

import asyncio
import json
import logging
from typing import Any, List

from app.automlplus.tools.static import split_chunks
from app.automlplus.tools.text import ChunkResult, _process_chunk_batch
from app.core.exceptions import AutoMLRuntimeError, AutoMLValidationError

logger = logging.getLogger(__name__)


async def run_accessibility_pipeline(
    content: str,
    filename: str,
    jinja_environment,
    chunk_size: int,
    concurrency: int = 4,
    context: str = "",
    page: str | None = None,
    chunk_offset: int = 0,
    chunks_per_request: int = 1,
) -> List[ChunkResult]:
    """Split HTML into chunks and process them concurrently with a semaphore.

    ``page`` tags every chunk result with the page (or file) it came from and
    ``chunk_offset`` shifts chunk indices so results from multiple pages can be
    merged into one globally-numbered list. ``chunks_per_request`` groups that
    many consecutive chunks into a single LLM request (1 = one request per
    chunk); ``concurrency`` caps how many requests are in flight at once.
    """
    if not content or not content.strip():
        logger.warning("Empty content provided to run_accessibility_pipeline")
        return []

    if chunk_size <= 0:
        raise AutoMLValidationError(f"chunk_size must be > 0, got {chunk_size}")

    if concurrency <= 0:
        raise AutoMLValidationError(f"concurrency must be > 0, got {concurrency}")

    if chunks_per_request <= 0:
        raise AutoMLValidationError(
            f"chunks_per_request must be > 0, got {chunks_per_request}"
        )

    try:
        chunks, ranges = split_chunks(content, chunk_size)
    except Exception as e:
        logger.exception("Failed to split content into chunks")
        raise AutoMLRuntimeError(f"Failed to split content into chunks: {e}") from e

    if not chunks:
        logger.warning("No chunks generated from content")
        return []

    items = [
        (i, chunk, start, end)
        for i, (chunk, (start, end)) in enumerate(zip(chunks, ranges))
    ]
    batches = [
        items[j : j + chunks_per_request]
        for j in range(0, len(items), chunks_per_request)
    ]
    logger.info(
        "Processing the website in %d chunks via %d LLM request(s) "
        "(up to %d chunk(s) per request, concurrency %d)",
        len(chunks),
        len(batches),
        chunks_per_request,
        concurrency,
    )

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _process_chunk_batch(
            batch, len(chunks), filename, jinja_environment, sem, context
        )
        for batch in batches
    ]

    try:
        batched_results: List[List[ChunkResult]] = await asyncio.gather(
            *tasks, return_exceptions=False
        )
    except Exception as e:
        logger.exception("Failed to process chunks")
        raise AutoMLRuntimeError(f"Failed to process chunks: {e}") from e

    results: List[ChunkResult] = [
        result for batch_results in batched_results for result in batch_results
    ]

    for result in results:
        result.page = page
        result.chunk += chunk_offset

    return results


async def resolve_coroutines(obj: Any) -> Any:
    """Recursively await any coroutine attributes in an object."""
    if asyncio.iscoroutine(obj):
        return await obj
    elif isinstance(obj, dict):
        return {k: await resolve_coroutines(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [await resolve_coroutines(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        new_obj = {}
        for k, v in vars(obj).items():
            new_obj[k] = await resolve_coroutines(v)
        return new_obj
    else:
        return obj


async def stream_accessibility_results(results):
    """Stream results as a single JSON array instead of JSONL."""
    resolved = []
    for item in results:
        if asyncio.iscoroutine(item):
            try:
                item = await item
            except Exception as e:
                resolved.append({"error": str(e)})
                continue

        try:
            data = await resolve_coroutines(item)
        except Exception as e:
            data = {"error": f"Failed to resolve item: {e}"}

        resolved.append(data)

    yield json.dumps(resolved, indent=2).encode("utf-8")
