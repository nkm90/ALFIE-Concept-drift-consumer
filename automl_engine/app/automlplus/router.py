"""Route definitions for the AutoML+ service."""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from jinja2 import Environment, FileSystemLoader

from app.automlplus.render_accessibility_report import render_report_html
from app.automlplus.tools.static import ReadabilityAnalyzer
from app.automlplus.tools.text import ChunkResult, summarize_accessibility_results
from app.automlplus.tools.vlm import AltTextChecker, ImagePromptRunner
from app.automlplus.utils import (
    automl_plus_data_instructions,
    extract_text_from_html_bytes,
    json_safe,
)
from app.automlplus.website_accessibility.crawler import crawl_website
from app.automlplus.website_accessibility.pipeline import (
    resolve_coroutines,
    run_accessibility_pipeline,
)
from app.core.concurrency import offload
from app.core.config import get_settings
from app.core.process_log import get_process_log, start_process_log, step
from app.core.schemas.responses import (
    AltTextCheckResponse,
    ErrorResponse,
    ImagePromptResponse,
    InstructionsResponse,
    WebAccessibilityResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["automl_plus"])

_jinja_path = get_settings().jinja_path

jinja_environment = Environment(loader=FileSystemLoader(_jinja_path))

_COMMON_RESPONSES: dict[int | str, dict[str, Any]] = {
    500: {"description": "Internal server error", "model": ErrorResponse},
}


@router.post(
    "/accepted_format/",
    response_model=InstructionsResponse,
    responses=_COMMON_RESPONSES,
)
async def show_accepted_format_instructions() -> JSONResponse:
    """Show accepted format instructions from a template"""
    try:
        return JSONResponse(
            content={"instructions": automl_plus_data_instructions()}, status_code=200
        )
    except Exception as e:
        logger.exception(
            "Unexpected error in finding data format instructions in automlplus"
        )
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post(
    "/image_tools/image_to_website/",
    response_model=ErrorResponse,
    responses={501: {"description": "Not implemented"}},
)
async def image_to_website(
    image_file: UploadFile | None = File(default=None),
) -> JSONResponse:
    """Convert an uploaded image into a basic HTML website structure."""
    return JSONResponse(content={"error": "Not implemented"}, status_code=501)


@router.post(
    "/web_access/check-alt-text/",
    response_model=AltTextCheckResponse,
    responses=_COMMON_RESPONSES,
)
async def check_alt_text(
    image_url: str = Form(...),
    alt_text: str = Form(...),
) -> JSONResponse:
    """Evaluate provided alt text against the referenced image using an LLM."""
    logger.info("Checking alt text for image URL: %s", image_url)
    start_process_log()
    try:
        with step("evaluate_alt_text"):
            result: str = await offload(
                AltTextChecker.check, jinja_environment, image_url, alt_text
            )
        logger.info("Alt-text evaluation completed successfully")

        safe_result = json_safe(
            {
                "src": image_url,
                "alt_text": alt_text,
                "evaluation": result,
                "process_log": get_process_log(),
            }
        )
        return JSONResponse(content=safe_result)
    except Exception as e:
        logger.exception("Error during alt-text check: %s", e)
        return JSONResponse(
            content={"error": str(e), "process_log": get_process_log()},
            status_code=500,
        )


@router.post(
    "/image_tools/run_on_image/",
    response_model=ImagePromptResponse,
    responses={
        400: {"description": "Missing image input", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def run_on_image(
    prompt: str = Form(...),
    model: str | None = Form(default=None),
    image_file: UploadFile | None = File(default=None),
    image_url: str | None = Form(default=None),
) -> JSONResponse:
    """Run a vision-language model on an image and return the text output."""
    logger.info("Running model on image with prompt: %s", prompt)
    start_process_log()

    if image_file is None and not image_url:
        logger.error("Missing both image_file and image_url")
        return JSONResponse(
            {
                "error": "Provide image_file or image_url",
                "process_log": get_process_log(),
            },
            status_code=400,
        )

    try:
        image_bytes: bytes | None = await image_file.read() if image_file else None
        if image_file:
            await image_file.close()
            logger.debug("Image file successfully read and closed")

        with step("run_model_on_image"):
            result = await offload(
                ImagePromptRunner.run,
                image_bytes=image_bytes,
                image_path_or_url=image_url,
                prompt=prompt,
                model=model,
                jinja_environment=jinja_environment,
            )

        safe_result = json_safe({"response": result, "process_log": get_process_log()})
        logger.info("Image prompt run completed successfully")
        return JSONResponse(content=safe_result)
    except Exception as e:
        logger.exception("Error during image prompt run: %s", e)
        return JSONResponse(
            {"error": str(e), "process_log": get_process_log()}, status_code=500
        )


@router.post(
    "/image_tools/run_on_image_stream/",
    response_model=None,
    responses={
        400: {"description": "Missing image input", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def run_on_image_stream(
    prompt: Annotated[str, Form(..., description="Prompt to apply on the image")] = "",
    model: Annotated[
        str | None, Form(..., description="Model to apply on the image")
    ] = None,
    image_file: Annotated[
        UploadFile | None, File(..., description="Image file if not a URL")
    ] = None,
    image_url: Annotated[
        str | None, Form(..., description="Image URL if not a file but an URL")
    ] = None,
) -> Response:
    """Stream a vision-language model's output on an image and prompt."""
    logger.info("Streaming model output for image prompt: %s", prompt)

    if image_file is None and not image_url:
        logger.error("No image or URL provided for streaming run")
        return JSONResponse(
            content={"error": "Provide image_file or image_url"}, status_code=400
        )

    try:
        image_bytes: bytes | None = None
        if image_file is not None:
            try:
                image_bytes = await image_file.read()
                logger.debug("Image file successfully read for streaming")
            finally:
                try:
                    await image_file.close()
                except Exception:
                    logger.warning("Failed to properly close image file", exc_info=True)

        def generator():
            logger.debug("Starting stream generator for image model run")
            for chunk in ImagePromptRunner.run_stream(
                image_bytes=image_bytes,
                image_path_or_url=image_url,
                prompt=prompt,
                model=model,
                jinja_environment=jinja_environment,
            ):
                yield chunk

        logger.info("Image stream initiated successfully")
        return StreamingResponse(generator(), media_type="text/plain")

    except Exception as e:
        logger.exception("Error during image prompt streaming run: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post(
    "/web_access/analyze/",
    response_model=WebAccessibilityResponse,
    responses={
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def analyze_web_accessibility_and_readability(
    file: Annotated[
        UploadFile | None, File(description="HTML file (optional when url is given)")
    ] = None,
    url: Annotated[str | None, Form(description="URL of website")] = None,
    depth: Annotated[
        int, Form(description="Crawl depth when a url is given (1 = seed page only)")
    ] = get_settings().web_accessibility_crawl_depth,
    include_html: Annotated[
        bool,
        Form(
            description=(
                "Also include a rendered, human-readable HTML report of these "
                "results as the html_report field"
            )
        ),
    ] = True,
    extra_file_input: Annotated[
        UploadFile | None, File(description="Extra file for LLM context")
    ] = None,
) -> JSONResponse:
    """Run WCAG-inspired accessibility checks and optional readability analysis on HTML."""
    logger.info("Starting web accessibility and readability analysis")
    start_process_log()

    settings = get_settings()
    timeout: int = settings.web_accessibility_url_retry_timeout

    if not file and not url:
        logger.error("No HTML file or URL provided for accessibility analysis")
        return JSONResponse(
            content={
                "error": "Provide an HTML file or a url",
                "process_log": get_process_log(),
            },
            status_code=400,
        )

    if depth < 1:
        logger.error("Invalid crawl depth: %s", depth)
        return JSONResponse(
            content={
                "error": f"depth must be >= 1, got {depth}",
                "process_log": get_process_log(),
            },
            status_code=400,
        )

    # --- Resolve pages to analyse (uploaded file or crawled website) ---
    pages_to_analyse: list[tuple[str, str]] = []
    crawl_errors: list[dict[str, str]] = []
    source_name: str = "uploaded.html"

    if file:
        try:
            with step("load_html"):
                content = (await file.read()).decode("utf-8", errors="replace")
                source_name = file.filename or source_name
                logger.debug("HTML file '%s' successfully loaded", source_name)
        finally:
            try:
                await file.close()
            except Exception:
                logger.warning("Failed to close uploaded HTML file", exc_info=True)
        pages_to_analyse.append((source_name, content))

    if url:
        try:
            with step("fetch_url"):
                logger.debug("Crawling website from URL: %s (depth %d)", url, depth)
                crawl_result = await offload(
                    crawl_website,
                    start_url=url,
                    depth=depth,
                    timeout=timeout,
                    max_pages=settings.web_accessibility_max_pages,
                )
                pages_to_analyse = [
                    (page.url, page.content) for page in crawl_result.pages
                ]
                crawl_errors = crawl_result.errors
                source_name = url
                logger.debug("Crawled %d page(s) from URL", len(pages_to_analyse))
        except Exception as e:
            logger.error("Failed to fetch HTML from URL: %s", e)
            return JSONResponse(
                content={
                    "error": f"Failed to fetch URL: {e}",
                    "process_log": get_process_log(),
                },
                status_code=400,
            )

    pages_to_analyse = [
        (name, content)
        for name, content in pages_to_analyse
        if content and str(content).strip()
    ]

    if not pages_to_analyse:
        logger.error("Resolved HTML content is empty")
        return JSONResponse(
            content={
                "error": "Resolved content is empty",
                "process_log": get_process_log(),
            },
            status_code=400,
        )

    # --- Load guidelines file if provided ---
    context_str: str = ""
    if extra_file_input is not None:
        try:
            with step("load_context"):
                logger.debug("Reading extra context file for accessibility analysis")
                guidelines_bytes = await extra_file_input.read()
                guidelines_text = guidelines_bytes.decode("utf-8", errors="replace")
                context_str = f"Accessibility guidelines to follow (user-provided):\n\n{guidelines_text}"
                logger.debug("Extra context file successfully loaded")
        finally:
            try:
                await extra_file_input.close()
            except Exception:
                logger.warning("Failed to close extra context file", exc_info=True)

    # --- Run accessibility pipeline per page ---
    chunk_size: int = settings.chunk_size_for_accessibility
    concurrency_num: int = settings.concurrency_num_for_accessibility
    chunks_per_request: int = settings.chunks_per_llm_request
    logger.debug(
        "Running accessibility pipeline with chunk size %s, concurrency %s, "
        "%s chunk(s) per LLM request",
        chunk_size,
        concurrency_num,
        chunks_per_request,
    )

    all_results: list[ChunkResult] = []
    with step("accessibility_analysis"):
        chunk_offset = 0
        for page_name, page_content in pages_to_analyse:
            page_results = await run_accessibility_pipeline(
                content=page_content,
                filename=page_name,
                jinja_environment=jinja_environment,
                chunk_size=chunk_size,
                concurrency=concurrency_num,
                context=context_str,
                page=page_name,
                chunk_offset=chunk_offset,
                chunks_per_request=chunks_per_request,
            )
            chunk_offset += len(page_results)
            all_results.extend(page_results)
        logger.info("Accessibility pipeline completed successfully")

        # --- Aggregate results ---
        resolved_results = [await resolve_coroutines(r) for r in all_results]

        scores = [
            r.get("score")
            for r in resolved_results
            if isinstance(r.get("score"), (int, float))
        ]
        average_score: float | None = (sum(scores) / len(scores)) if scores else None
        logger.debug("Computed average accessibility score: %s", average_score)

    # --- Readability analysis ---
    readability_scores: dict[str, Any] | None = None
    try:
        with step("readability_analysis"):
            texts = [
                extract_text_from_html_bytes(content.encode("utf-8"))
                for _, content in pages_to_analyse
            ]
            text = "\n".join(t for t in texts if t.strip())
            if text.strip():
                readability_scores = ReadabilityAnalyzer.analyze(text)
                logger.debug("Readability analysis completed successfully")
    except Exception as e:
        logger.warning("Error during readability analysis: %s", e)
        readability_scores = {"error": str(e)}

    # --- LLM summary of aggregated results ---
    summary: str | None = None
    try:
        with step("summary"):
            summary = await summarize_accessibility_results(
                jinja_environment=jinja_environment,
                source=source_name,
                pages=[name for name, _ in pages_to_analyse],
                average_score=average_score,
                chunk_scores=[
                    {
                        "page": r.get("page"),
                        "chunk": r.get("chunk"),
                        "score": r.get("score"),
                    }
                    for r in resolved_results
                ],
                readability=readability_scores,
                llm_responses=[r.get("llm_response") or "" for r in resolved_results],
            )
            logger.debug("Accessibility summary generated successfully")
    except Exception as e:
        logger.warning("Error during accessibility summary generation: %s", e)
        summary = None

    payload = {
        "source": source_name,
        "pages_crawled": [name for name, _ in pages_to_analyse],
        "average_score": average_score,
        "results": resolved_results,
        "readability": readability_scores,
        "summary": summary,
        "crawl_errors": crawl_errors,
        "process_log": get_process_log(),
    }

    # --- Render the human-readable HTML report (kept out of json_safe so the
    # --- HTML arrives unescaped and can be saved straight to a .html file) ---
    html_report: str | None = None
    if include_html:
        try:
            with step("render_html_report"):
                html_report = await offload(render_report_html, payload)
                logger.debug("HTML accessibility report rendered successfully")
        except Exception as e:
            logger.warning("Error during HTML report rendering: %s", e)
            html_report = None

    safe_payload = json_safe(payload)
    # Overwrite (bypassing json_safe) so the HTML arrives unescaped and can be
    # saved straight to a .html file
    safe_payload["html_report"] = html_report
    logger.info("Web accessibility and readability analysis finished successfully")
    return JSONResponse(content=safe_payload)
