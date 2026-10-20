"""Render a website-accessibility JSON report to a human-readable HTML file.

Standalone tool (no service required):

    uv run python app/automlplus/render_accessibility_report.py report.json report.html

Reads the JSON produced by ``POST /automl/automl_plus/web_access/analyze/``
and writes a self-contained HTML page (inline CSS, no external assets) with
the summary, readability metrics, crawled pages, per-chunk findings grouped
by page, and the process log.
"""

import argparse
import html
import json
import re
import sys
from collections.abc import Iterable

_MIN_MD_HEADERS = 1
_UNESCAPE_PATTERN = re.compile(r"\\(\\|n|r|t|\")")


def _unescape_json_safe(text: str) -> str:
    """Restore newlines/quotes flattened by ``json_safe`` in the API payload."""

    def repl(match: re.Match[str]) -> str:
        token = match.group(1)
        return {"n": "\n", "r": "\r", "t": "\t", '"': '"'}.get(token, token)

    return _UNESCAPE_PATTERN.sub(repl, text)


def _reinflate_flattened_markdown(text: str) -> str:
    """Re-insert line breaks before block markers in whitespace-flattened text.

    The pipeline normalises LLM responses with ``\\s+ -> " "`` for score
    parsing, which flattens ``###`` headers, numbered lists, bullets and code
    fences onto one line. This heuristic restores enough structure for the
    markdown renderer; it only runs when the text has no real newlines.
    """
    if "\n" in text:
        return text
    inflated = text
    inflated = re.sub(r"\s+(```\w*)\s+", r"\n\1\n", inflated)
    inflated = re.sub(r"\s+```\s+", "\n```\n", inflated)
    inflated = re.sub(r"\s+(#{1,6})\s", r"\n\1 ", inflated)
    inflated = re.sub(r'(?<=[.!"\'>:.)\]])\s+(\d+\.)\s+', r"\n\1 ", inflated)
    inflated = re.sub(r"(?<=[.!:>])\s+[-*]\s+", "\n- ", inflated)
    return inflated


def _inline_markdown(text: str) -> str:
    """Render bold, italic, inline-code and links inside a line of markdown.

    The text is HTML-escaped first so that code snippets quoted by the LLM
    (e.g. ``<img>``, ``<title>``) display as text instead of being parsed as
    live HTML by the browser.
    """
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def markdown_to_html(markdown_text: str) -> str:
    """Render the markdown subset used by the LLM responses to HTML.

    Supports fenced code blocks, ``#``-``######`` headers, bullet and ordered
    lists, and paragraphs. Anything unrecognised falls through as paragraph
    text, which is fine for a report view.
    """
    lines = markdown_text.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    list_items: list[tuple[str, str]] = []
    code_lines: list[str] | None = None

    def flush_paragraph() -> None:
        if paragraph:
            out.append(
                "<p>"
                + "<br>".join(_inline_markdown(line) for line in paragraph)
                + "</p>"
            )
            paragraph.clear()

    def flush_list() -> None:
        if not list_items:
            return
        ordered = list_items[0][0] == "ol"
        tag = "ol" if ordered else "ul"
        out.append(
            f"<{tag}>"
            + "".join(f"<li>{_inline_markdown(item)}</li>" for _, item in list_items)
            + f"</{tag}>"
        )
        list_items.clear()

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            if code_lines is None:
                flush_paragraph()
                flush_list()
                code_lines = []
            else:
                out.append(
                    "<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>"
                )
                code_lines = None
            continue
        if code_lines is not None:
            code_lines.append(line)
            continue

        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        header = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if header:
            flush_paragraph()
            flush_list()
            level = min(len(header.group(1)) + _MIN_MD_HEADERS, 6)
            out.append(f"<h{level}>{_inline_markdown(header.group(2))}</h{level}>")
            continue

        bullet = re.match(r"^[-*+]\s+(.*)$", stripped)
        ordered = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if bullet or ordered:
            flush_paragraph()
            kind, content = (
                ("ul", bullet.group(1)) if bullet else ("ol", ordered.group(1))  # type: ignore[union-attr]
            )
            if list_items and list_items[0][0] != kind:
                flush_list()
            list_items.append((kind, content))
            continue

        flush_list()
        paragraph.append(stripped)

    if code_lines is not None:
        out.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
    flush_paragraph()
    flush_list()
    return "\n".join(out)


def _score_class(score: float | None) -> str:
    if score is None:
        return "score-none"
    if score >= 8:
        return "score-good"
    if score >= 5:
        return "score-mid"
    return "score-bad"


def _score_label(score: float | None) -> str:
    return f"{score:g}" if score is not None else "n/a"


def _readability_rows(readability: dict | None) -> str:
    if not readability:
        return "<p class='muted'>No readability metrics available.</p>"
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in readability.items()
    )
    return f"<table>{rows}</table>"


def _chunk_details(chunk: dict, index_within_page: int) -> str:
    score = chunk.get("score")
    score = float(score) if isinstance(score, (int, float)) else None
    badge = f"<span class='badge {_score_class(score)}'>{_score_label(score)}/10</span>"
    error = chunk.get("error")
    error_html = (
        f"<div class='error'>Error: {html.escape(str(error))}</div>" if error else ""
    )

    images = chunk.get("image_feedback") or []
    images_html = ""
    if images:
        items = []
        for img in images:
            if "error" in img:
                items.append(
                    f"<li><code>{html.escape(str(img.get('src', '')))}</code> "
                    f"— <span class='error'>{html.escape(str(img['error']))}</span></li>"
                )
            else:
                items.append(
                    f"<li><code>{html.escape(str(img.get('src', '')))}</code> "
                    f"alt=&quot;{html.escape(str(img.get('alt_text', '')))}&quot; — "
                    f"{_inline_markdown(_unescape_json_safe(str(img.get('result', ''))))}</li>"
                )
        images_html = (
            "<div class='image-feedback'><h4>Image alt-text checks</h4>"
            f"<ul>{''.join(items)}</ul></div>"
        )

    llm = _reinflate_flattened_markdown(
        _unescape_json_safe(str(chunk.get("llm_response") or ""))
    )
    return f"""
<details{" open" if index_within_page == 0 else ""}>
  <summary>{badge} Chunk {html.escape(str(chunk.get("chunk", "")))}
    <span class='muted'>lines {html.escape(str(chunk.get("start_line", "")))}–{html.escape(str(chunk.get("end_line", "")))}</span>
  </summary>
  <div class='chunk-body'>
    {error_html}
    <div class='llm'>{markdown_to_html(llm)}</div>
    {images_html}
  </div>
</details>"""


def _grouped_results(results: Iterable[dict]) -> str:
    pages: dict[str, list[dict]] = {}
    for chunk in results:
        pages.setdefault(str(chunk.get("page") or "unknown"), []).append(chunk)

    sections = []
    for page, chunks in pages.items():
        scores = [
            float(c["score"])
            for c in chunks
            if isinstance(c.get("score"), (int, float))
        ]
        page_avg = sum(scores) / len(scores) if scores else None
        page_badge = (
            f"<span class='badge {_score_class(page_avg)}'>"
            f"avg {_score_label(page_avg)}/10</span>"
        )
        details = "".join(_chunk_details(c, i) for i, c in enumerate(chunks))
        sections.append(
            f"<section class='page'><h3>{html.escape(page)} {page_badge}</h3>"
            f"{details}</section>"
        )
    return "\n".join(sections) or "<p class='muted'>No chunk results.</p>"


def _crawl_errors(errors: list[dict] | None) -> str:
    if not errors:
        return ""
    items = "".join(
        f"<li><code>{html.escape(str(e.get('url', '')))}</code> — "
        f"{html.escape(str(e.get('error', '')))}</li>"
        for e in errors
    )
    return f"<section class='card'><h2>Crawl errors</h2><ul class='errors'>{items}</ul></section>"


def _process_log(log: list[dict] | None) -> str:
    if not log:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(str(e.get('timestamp', '')))}</td>"
        f"<td>{html.escape(str(e.get('step', e.get('type', ''))))}</td>"
        f"<td>{html.escape(str(e.get('status', '')))}</td></tr>"
        for e in log
    )
    return (
        "<details class='card'><summary>Process log</summary>"
        f"<table><tr><th>Time</th><th>Step</th><th>Status</th></tr>{rows}</table>"
        "</details>"
    )


_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
       margin: 0; background: #f4f6f9; color: #1f2933; line-height: 1.55; }
header { background: #1f3b5c; color: #fff; padding: 28px 36px; }
header h1 { margin: 0 0 6px; font-size: 1.5rem; word-break: break-all; }
header .meta { display: flex; flex-wrap: wrap; gap: 18px; opacity: .9; font-size: .95rem; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 48px; }
.card { background: #fff; border-radius: 10px; padding: 20px 24px; margin: 18px 0;
        box-shadow: 0 1px 3px rgba(16,24,40,.08); }
h2 { font-size: 1.15rem; margin: 0 0 12px; color: #1f3b5c; }
h3 { font-size: 1rem; margin: 0 0 10px; word-break: break-all; }
h4 { margin: 12px 0 6px; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #e4e9f0; }
th { color: #52606d; font-weight: 600; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px;
         font-weight: 700; font-size: .8rem; color: #fff; margin-right: 8px; }
.score-good { background: #178a5b; } .score-mid { background: #b7791f; }
.score-bad { background: #c0392b; } .score-none { background: #7b8794; }
.page { background: #fff; border-radius: 10px; padding: 18px 22px; margin: 18px 0;
        box-shadow: 0 1px 3px rgba(16,24,40,.08); }
details { margin: 8px 0; }
details > summary { cursor: pointer; padding: 8px 10px; border-radius: 6px;
                    background: #f0f4f8; font-weight: 600; }
details[open] > summary { border-bottom-left-radius: 0; border-bottom-right-radius: 0; }
.chunk-body { border: 1px solid #e4e9f0; border-top: none; border-radius: 0 0 6px 6px;
              padding: 12px 16px; }
.llm h3 { word-break: normal; }
.llm pre { background: #0f1720; color: #e6edf3; padding: 12px; border-radius: 6px;
           overflow-x: auto; }
.llm code { background: #eef2f7; padding: 1px 5px; border-radius: 4px; font-size: .9em; }
.llm pre code { background: none; padding: 0; }
.image-feedback { margin-top: 12px; }
.image-feedback ul { padding-left: 20px; }
.errors li { margin: 4px 0; }
.error { color: #c0392b; }
.muted { color: #7b8794; font-weight: 400; font-size: .85rem; }
.pages { display: flex; flex-wrap: wrap; gap: 8px; }
.pages span { background: #e8eef7; color: #1f3b5c; padding: 3px 10px; border-radius: 6px;
              font-size: .85rem; word-break: break-all; }
"""


def render_report_html(report: dict) -> str:
    """Build the full HTML document for a parsed report payload."""
    source = html.escape(str(report.get("source", "unknown")))
    average = report.get("average_score")
    average = float(average) if isinstance(average, (int, float)) else None
    pages = report.get("pages_crawled") or []
    results = report.get("results") or []
    readability = report.get("readability")
    summary = _unescape_json_safe(str(report.get("summary") or ""))

    chips = "".join(f"<span>{html.escape(str(p))}</span>" for p in pages)
    summary_html = (
        f"<section class='card'><h2>Summary</h2>{markdown_to_html(summary)}</section>"
        if summary.strip()
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Accessibility report — {source}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>Accessibility report</h1>
  <div class='meta'>
    <span><strong>Source:</strong> {source}</span>
    <span><strong>Average score:</strong>
      <span class='badge {_score_class(average)}'>{_score_label(average)}/10</span></span>
    <span><strong>Pages analysed:</strong> {len(pages)}</span>
    <span><strong>Chunks analysed:</strong> {len(results)}</span>
  </div>
</header>
<main>
{summary_html}
<section class='card'><h2>Readability</h2>{_readability_rows(readability)}</section>
<section class='card'><h2>Pages analysed</h2><div class='pages'>{chips or "<span class='muted'>none</span>"}</div></section>
{_crawl_errors(report.get("crawl_errors"))}
<section class='card'><h2>Findings per page</h2>
  <p class='muted'>Each chunk below was scored 0–10 by the LLM against WCAG guidelines. Click a chunk to expand its findings.</p>
</section>
{_grouped_results(results)}
{_process_log(report.get("process_log"))}
</main>
</body>
</html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a website-accessibility JSON report to HTML."
    )
    parser.add_argument("input_file", help="JSON report from /web_access/analyze/")
    parser.add_argument("output_file", help="destination .html path")
    args = parser.parse_args(argv)

    try:
        with open(args.input_file, encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Failed to read report '{args.input_file}': {e}", file=sys.stderr)
        return 1

    if not isinstance(report, dict):
        print("Report JSON must be an object", file=sys.stderr)
        return 1

    html_doc = render_report_html(report)
    try:
        with open(args.output_file, "w", encoding="utf-8") as fh:
            fh.write(html_doc)
    except OSError as e:
        print(f"Failed to write '{args.output_file}': {e}", file=sys.stderr)
        return 1

    print(f"Wrote {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
