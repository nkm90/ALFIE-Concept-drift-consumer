"""LLM-over-text tools for AutoML+.

Text tools send plain-text content (HTML chunks, documents, etc.) to a language
model and parse the structured response. Unlike VLM tools, no image input is
required; unlike static tools, they rely on an external LLM API.

Current tools:

- ``ChunkResult`` — dataclass holding the outcome (score, image feedback, LLM
  response, or error) for a single processed text chunk.
- ``_process_single_chunk`` — sends one HTML chunk to the LLM for WCAG analysis,
  extracts a numeric score from the response, and runs ``AltTextChecker`` on any
  ``<img>`` tags found in the chunk.
- ``_process_chunk_batch`` — analyzes a group of chunks in a single LLM request
  (batching several chunks per call to reduce request count). Singleton batches
  reuse the single-chunk prompt; larger batches use
  ``build_batched_chunk_prompt.txt`` and parse one review section per chunk.
- ``summarize_accessibility_results`` — asks the LLM to condense aggregated
  pipeline results (scores, recurring issues, readability) into a markdown
  summary report.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from app.automlplus.tools.vlm import AltTextChecker
from app.core.chat_handler import ChatHandler
from app.core.config import get_settings
from app.core.utils import render_template

logger = logging.getLogger(__name__)

# One chunk to analyse: (chunk index, content, start_line, end_line).
ChunkBatchItem = Tuple[int, str, int, int]


@dataclass
class ChunkResult:
    """Result for processing a single chunk of an HTML file."""

    chunk: int
    start_line: int
    end_line: int
    score: float | None
    image_feedback: List[Dict[str, Any]]
    llm_response: str | None
    error: str | None = None
    page: str | None = None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


async def _wcag_chat(prompt: str, context: str = "") -> str:
    """Send a WCAG-analysis prompt to the configured chat model."""
    settings = get_settings()
    backend = settings.model_backend.lower()
    model = (settings.web_accessibility_chat_model or "").strip() or "gpt-4o-mini"
    response = await ChatHandler.chat(
        prompt, context=context, backend=backend, model=model, stream=False
    )
    return response if isinstance(response, str) else ""


def _extract_score(response_text: str, i: int) -> float | None:
    """Extract a 0-100 score from an LLM response, if present."""
    score_match = re.search(
        r"\bScore[:\s]*([0-9]+(?:\.[0-9]+)?)", response_text, re.IGNORECASE
    )
    if not score_match:
        return None
    try:
        score_val = float(score_match.group(1))
        if 0 <= score_val <= 100:
            return score_val
        logger.warning(
            "Score %f out of valid range [0, 100] for chunk %d",
            score_val,
            i,
        )
    except ValueError as e:
        logger.warning(
            "Failed to parse score '%s' for chunk %d: %s",
            score_match.group(1),
            i,
            e,
        )
    return None


def _check_images_in_chunk(
    chunk: str, i: int, jinja_environment
) -> List[Dict[str, Any]]:
    """Run ``AltTextChecker`` on every ``<img>`` tag found in the chunk."""
    images = re.findall(r'<img[^>]+src="([^"]+)"[^>]+alt="([^"]+)"', chunk)
    image_feedback: List[Dict[str, Any]] = []
    for src, alt in images:
        try:
            result = AltTextChecker.check(jinja_environment, src, alt)
            if isinstance(result, str):
                result = _normalize_text(result)
            image_feedback.append({"src": src, "alt_text": alt, "result": result})
        except Exception as e:
            logger.warning(
                "Failed to check alt text for image '%s' in chunk %d: %s",
                src,
                i,
                e,
            )
            image_feedback.append({"src": src, "alt_text": alt, "error": str(e)})
    return image_feedback


def _build_chunk_result(
    i: int,
    chunk: str,
    start: int,
    end: int,
    response_text: str,
    jinja_environment,
) -> ChunkResult:
    """Assemble a ChunkResult from an LLM review of the chunk."""
    return ChunkResult(
        chunk=i,
        start_line=start,
        end_line=end,
        score=_extract_score(response_text, i),
        image_feedback=_check_images_in_chunk(chunk, i, jinja_environment),
        llm_response=response_text,
        error=None,
    )


def _empty_chunk_result(i: int, start: int, end: int) -> ChunkResult:
    return ChunkResult(
        chunk=i,
        start_line=start,
        end_line=end,
        score=None,
        image_feedback=[],
        llm_response=None,
        error="Empty chunk provided",
    )


async def _analyze_single_chunk(
    i: int,
    chunk: str,
    start: int,
    end: int,
    total: int,
    filename: str,
    jinja_environment,
    context: str,
) -> ChunkResult:
    """Analyze one chunk via its own dedicated LLM request (no semaphore)."""
    try:
        if not chunk or not chunk.strip():
            logger.warning("Empty chunk provided for processing at index %d", i)
            return _empty_chunk_result(i, start, end)

        prompt = render_template(
            jinja_environment=jinja_environment,
            template_name="build_chunk_prompt.txt",
            filename=filename,
            chunk=chunk,
            idx=i,
            total=total,
            start_line=start,
            end_line=end,
        )
        response_text = _normalize_text(await _wcag_chat(prompt, context))
        return _build_chunk_result(
            i, chunk, start, end, response_text, jinja_environment
        )
    except Exception as e:
        logger.exception("Failed to process chunk %d", i)
        return ChunkResult(
            chunk=i,
            start_line=start,
            end_line=end,
            score=None,
            image_feedback=[],
            llm_response=None,
            error=str(e),
        )


async def _process_single_chunk(
    i: int,
    chunk: str,
    start: int,
    end: int,
    total: int,
    filename: str,
    jinja_environment,
    sem: asyncio.Semaphore,
    context: str,
) -> ChunkResult:
    """Process a single chunk: prompt LLM and validate image alt texts."""
    async with sem:
        return await _analyze_single_chunk(
            i, chunk, start, end, total, filename, jinja_environment, context
        )


# Per-chunk review sections emitted by build_batched_chunk_prompt.txt, e.g.
# "=== CHUNK 2 REVIEW START === Score: 8 ... === CHUNK 2 REVIEW END ===".
_BATCH_SECTION_RE = re.compile(
    r"===\s*CHUNK\s+(\d+)\s+REVIEW\s+START\s*===\s*(.*?)===\s*CHUNK\s+\d+\s+REVIEW\s+END\s*===",
    re.DOTALL,
)


def _parse_batch_sections(response_text: str) -> Dict[int, str]:
    """Map 1-based chunk numbers from the LLM response to their review text."""
    return {
        int(match.group(1)): _normalize_text(match.group(2))
        for match in _BATCH_SECTION_RE.finditer(response_text)
    }


async def _process_chunk_batch(
    batch: Sequence[ChunkBatchItem],
    total: int,
    filename: str,
    jinja_environment,
    sem: asyncio.Semaphore,
    context: str,
) -> List[ChunkResult]:
    """Analyze a group of chunks in a single LLM request.

    Singleton batches reuse the single-chunk prompt; larger batches render
    ``build_batched_chunk_prompt.txt`` and send all chunks in one call. The
    response must contain one review section per chunk; chunks whose section
    is missing get an error result rather than failing the whole batch.
    """
    async with sem:
        if len(batch) == 1:
            i, chunk, start, end = batch[0]
            return [
                await _analyze_single_chunk(
                    i, chunk, start, end, total, filename, jinja_environment, context
                )
            ]
        try:
            results_by_index: Dict[int, ChunkResult] = {}
            pending: List[ChunkBatchItem] = []
            for i, chunk, start, end in batch:
                if not chunk or not chunk.strip():
                    logger.warning("Empty chunk provided in batch at index %d", i)
                    results_by_index[i] = _empty_chunk_result(i, start, end)
                else:
                    pending.append((i, chunk, start, end))

            if pending:
                prompt = render_template(
                    jinja_environment=jinja_environment,
                    template_name="build_batched_chunk_prompt.txt",
                    filename=filename,
                    chunks=[
                        {
                            "idx": i,
                            "chunk": chunk,
                            "start_line": start,
                            "end_line": end,
                        }
                        for i, chunk, start, end in pending
                    ],
                    num_chunks=len(pending),
                    total=total,
                )
                response_text = _normalize_text(await _wcag_chat(prompt, context))
                sections = _parse_batch_sections(response_text)
                for i, chunk, start, end in pending:
                    review = sections.get(i + 1)
                    if review is None:
                        logger.warning(
                            "No review section for chunk %d in batched LLM response",
                            i,
                        )
                        results_by_index[i] = ChunkResult(
                            chunk=i,
                            start_line=start,
                            end_line=end,
                            score=None,
                            image_feedback=[],
                            llm_response=response_text or None,
                            error="No review section for chunk in batched response",
                        )
                    else:
                        results_by_index[i] = _build_chunk_result(
                            i, chunk, start, end, review, jinja_environment
                        )

            return [results_by_index[i] for i, _, _, _ in batch]
        except Exception as e:
            logger.exception(
                "Failed to process chunk batch %s", [item[0] for item in batch]
            )
            return [
                ChunkResult(
                    chunk=i,
                    start_line=start,
                    end_line=end,
                    score=None,
                    image_feedback=[],
                    llm_response=None,
                    error=str(e),
                )
                for i, _, start, end in batch
            ]


_MAX_FINDING_CHARS = 600


async def summarize_accessibility_results(
    jinja_environment,
    source: str,
    pages: List[str],
    average_score: float | None,
    chunk_scores: List[Dict[str, Any]],
    readability: Dict[str, Any] | None,
    llm_responses: List[str],
) -> str:
    """Condense aggregated pipeline results into a markdown summary via the LLM."""
    findings = "\n\n".join(
        f"[{idx + 1}] {response[:_MAX_FINDING_CHARS]}"
        for idx, response in enumerate(llm_responses)
        if response
    )
    if not findings:
        findings = "No chunk findings available."

    prompt = render_template(
        jinja_environment=jinja_environment,
        template_name="website_accessibility_summary.txt",
        source=source,
        pages="\n".join(f"- {p}" for p in pages) or "- (none)",
        chunk_scores="\n".join(
            f"- page={cs.get('page')} chunk={cs.get('chunk')} score={cs.get('score')}"
            for cs in chunk_scores
        )
        or "- (none)",
        average_score=average_score if average_score is not None else "N/A",
        readability=readability if readability is not None else {},
        findings=findings,
    )

    response = await _wcag_chat(prompt)
    return response
