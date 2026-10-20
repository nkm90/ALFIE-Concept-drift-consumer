"""Static analysis tools for AutoML+.

Static tools derive insights from content using deterministic, rule-based libraries
— no LLM calls are made. They are fast, reproducible, and require no API credentials.

Current tools:

- ``ReadabilityAnalyzer`` — computes textstat readability metrics (Flesch Reading
  Ease, word counts, sentence length, etc.) over a plain-text string.
- ``split_chunks`` — splits an HTML/text string into fixed-size character chunks
  while tracking the original 1-based line ranges for each chunk.
"""

import bisect
import logging
from math import isfinite
from typing import Any, Dict, List, Tuple

import textstat

from app.core.exceptions import AutoMLValidationError

logger = logging.getLogger(__name__)


class ReadabilityAnalyzer:
    """Compute readability metrics for a piece of text."""

    METRICS = {
        "Flesch Reading Ease": textstat.flesch_reading_ease,
        "Difficult Words": textstat.difficult_words,
        "Lexicon Count": textstat.lexicon_count,
        "Avg Sentence Length": textstat.words_per_sentence,
    }

    @staticmethod
    def apply_metric(metric, text: str) -> Any:
        try:
            value = metric(text)
            if isinstance(value, float) and not isfinite(value):
                return None
            if isinstance(value, (int, float, str)):
                return value
            return str(value)
        except Exception:
            logger.warning("Metric failed: %s", metric.__name__)
            return "N/A"

    @classmethod
    def analyze(cls, text: str) -> Dict[str, Any]:
        logger.info("Running readability metrics")
        return {
            name: cls.apply_metric(metric, text) for name, metric in cls.METRICS.items()
        }


def split_chunks(
    content: str, chunk_size: int
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Split content into fixed-size character chunks and return
    1-based (start_line, end_line) ranges for each chunk.

    Line ranges are accurate even when chunks start/end mid-line.
    """
    if chunk_size <= 0:
        raise AutoMLValidationError("chunk_size must be > 0")

    if not content or not content.strip():
        logger.warning("Empty content provided to split_chunks")
        return [], []

    lines = content.splitlines(keepends=True)

    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    chunks: List[str] = []
    line_ranges: List[Tuple[int, int]] = []

    content_len = len(content)
    i = 0

    while i < content_len:
        end = min(i + chunk_size, content_len)
        chunks.append(content[i:end])

        start_line = bisect.bisect_right(line_offsets, i) - 1
        end_line = bisect.bisect_left(line_offsets, end) - 1

        line_ranges.append((start_line + 1, max(start_line + 1, end_line + 1)))

        i = end

    return chunks, line_ranges
