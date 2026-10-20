"""Pydantic response models for all API endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: str
    process_log: list[dict[str, Any]] = []


class InstructionsResponse(BaseModel):
    instructions: str


# ---------------------------------------------------------------------------
# AutoML+  (/automl/automl_plus)
# ---------------------------------------------------------------------------


class AltTextCheckResponse(BaseModel):
    src: str
    alt_text: str
    evaluation: str
    process_log: list[dict[str, Any]] = []


class ImagePromptResponse(BaseModel):
    response: str
    process_log: list[dict[str, Any]] = []


class WebAccessibilityResponse(BaseModel):
    source: str
    pages_crawled: list[str] = []
    average_score: float | None = None
    results: list[dict[str, Any]]
    readability: dict[str, Any] | None = None
    summary: str | None = None
    crawl_errors: list[dict[str, str]] = []
    html_report: str | None = None
    process_log: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Tabular AutoML  (/automl/tabular)
# ---------------------------------------------------------------------------


class TrainingSuccessResponse(BaseModel):
    message: str
    leaderboard: str
    deployment_instructions: str | None = None
    process_log: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Vision AutoML  (/automl/vision)
# ---------------------------------------------------------------------------


class VisionTrainingSuccessResponse(BaseModel):
    message: str
    leaderboard: str
    deployment_instructions: str | None = None
    process_log: list[dict[str, Any]] = []


class MultimodalTrainingSuccessResponse(BaseModel):
    message: str
    leaderboard: str
    auxiliary_columns: list[str]
    deployment_instructions: str | None = None
    process_log: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Audio AutoML  (/automl/audio)
# ---------------------------------------------------------------------------


class AudioTrainingSuccessResponse(BaseModel):
    message: str
    leaderboard: str
    deployment_instructions: str | None = None
    process_log: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Text AutoML  (/automl/text)
# ---------------------------------------------------------------------------


class TextTrainingSuccessResponse(BaseModel):
    message: str
    leaderboard: str
    deployment_instructions: str | None = None
    process_log: list[dict[str, Any]] = []
