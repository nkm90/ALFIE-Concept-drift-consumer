"""Recursive website crawler for the accessibility pipeline.

Fetches the seed URL and follows same-origin links breadth-first up to a
configured depth, returning the HTML of every page fetched. The crawler is
deliberately conservative:

- only ``http(s)`` links whose host matches the seed are followed;
- links to obvious non-HTML assets (images, stylesheets, archives, ...) are
  skipped;
- URLs are normalised (fragments stripped, trailing slash collapsed) so each
  page is fetched at most once;
- a hard ``max_pages`` cap bounds the crawl.

Errors on *sub-pages* are collected and returned instead of aborting the
crawl; only a failure on the seed page itself raises.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from app.core.exceptions import AutoMLValidationError

logger = logging.getLogger(__name__)

_SKIP_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".bmp",
    ".css",
    ".js",
    ".mjs",
    ".json",
    ".xml",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
)

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ALFIE-Accessibility/1.0)"}


@dataclass
class CrawledPage:
    """A single successfully fetched page."""

    url: str
    content: str


@dataclass
class CrawlResult:
    """Outcome of a crawl: fetched pages plus per-URL sub-page failures."""

    pages: list[CrawledPage] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


def _normalize(url: str) -> str:
    """Return the dedup key for a URL (strip fragment and trailing slash)."""
    no_frag, _ = urldefrag(url)
    return no_frag.rstrip("/")


def _is_html_candidate(url: str) -> bool:
    """Heuristically exclude asset/asset-like URLs."""
    path = urlparse(url).path.lower()
    return not path.endswith(_SKIP_EXTENSIONS)


def _same_origin(seed: str, url: str) -> bool:
    seed_parts = urlparse(seed)
    url_parts = urlparse(url)
    return (
        url_parts.scheme in ("http", "https") and url_parts.netloc == seed_parts.netloc
    )


def _extract_links(page_url: str, html: str) -> list[str]:
    """Extract normalised, same-origin page URLs from an HTML document."""
    soup = BeautifulSoup(html, features="html.parser")
    links: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href", "")).strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absolute = urljoin(page_url, href)
        if not _same_origin(page_url, absolute) or not _is_html_candidate(absolute):
            continue
        key = _normalize(absolute)
        if key and key not in seen:
            seen.add(key)
            links.append(key)
    return links


def crawl_website(
    start_url: str,
    depth: int = 2,
    timeout: int = 10,
    max_pages: int = 25,
) -> CrawlResult:
    """Crawl ``start_url`` and same-origin links up to ``depth`` levels.

    ``depth=1`` fetches only the seed page; ``depth=2`` additionally fetches
    pages linked from the seed, and so on. At most ``max_pages`` pages are
    fetched in total. Raises ``AutoMLValidationError`` for invalid depth and
    propagates seed-fetch failures to the caller.
    """
    if depth < 1:
        raise AutoMLValidationError(f"depth must be >= 1, got {depth}")
    if max_pages < 1:
        raise AutoMLValidationError(f"max_pages must be >= 1, got {max_pages}")

    result = CrawlResult()
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(_normalize(start_url), 1)])

    while queue and len(result.pages) < max_pages:
        url, level = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        logger.debug("Crawling %s (level %d)", url, level)
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=timeout)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            if level == 1:
                raise
            logger.warning("Failed to fetch %s during crawl: %s", url, e)
            result.errors.append({"url": url, "error": str(e)})
            continue

        result.pages.append(CrawledPage(url=url, content=html))

        if level < depth:
            for link in _extract_links(url, html):
                if link not in visited:
                    queue.append((link, level + 1))

    if len(result.pages) >= max_pages and queue:
        logger.warning(
            "Crawl of %s stopped early: max_pages=%d reached", start_url, max_pages
        )

    logger.info(
        "Crawled %d page(s) from %s at depth %d (%d sub-page failure(s))",
        len(result.pages),
        start_url,
        depth,
        len(result.errors),
    )
    return result
