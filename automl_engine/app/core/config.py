"""Centralized application configuration.

All runtime configuration is declared here as a single source of truth: the
environment-variable name, its type, and its default live in one place.

`get_settings()` returns a *fresh* ``Settings`` on every call. This intentionally
mirrors the previous ``os.getenv(...)`` semantics: a number of modules call
``load_dotenv()`` at different points during startup, which mutates
``os.environ`` in-place, so a cached singleton could return values that are stale
relative to a direct ``os.getenv`` read. Reading the environment on each access
keeps behaviour identical to the scattered ``os.getenv`` calls this replaces.
"""

from __future__ import annotations

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view over the process environment.

    Unknown environment variables are ignored so that unrelated keys (e.g.
    service ports, MongoDB URLs used by other services) do not cause validation
    failures.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Document paths -----------------------------------------------------
    jinja_path: str = Field(
        default="app/core/prompt_templates", validation_alias="JINJAPATH"
    )

    # --- Logging -----------------------------------------------------------
    alfie_log_dir: str = os.path.join(os.getcwd(), "logs")
    alfie_log_level: str = "INFO"
    alfie_log_format: str = "json"

    # --- Model backend / Azure --------------------------------------------
    model_backend: str = "azure"
    azure_openai_endpoint_large_model: str | None = None
    azure_openai_key: str | None = None

    # --- AutoMLPlus / web accessibility -----------------------------------
    web_accessibility_chat_model: str | None = None
    alt_text_checker_model: str | None = None
    image_prompt_model: str = "gpt-4o-mini"
    chunk_size_for_accessibility: int = 3000
    concurrency_num_for_accessibility: int = 4
    # Number of HTML chunks combined into a single LLM request. 1 restores the
    # one-request-per-chunk behaviour; higher values reduce request count and
    # are safe on large-context models (e.g. GPT-5, 400K input tokens).
    chunks_per_llm_request: int = 4
    web_accessibility_url_retry_timeout: int = 10
    web_accessibility_crawl_depth: int = 2
    web_accessibility_max_pages: int = 25

    # --- Tabular AutoML ----------------------------------------------------
    autodw_url: str = "http://localhost:8000"
    default_tabular_train_test_split_size: float = 0.8
    default_time_limit: int = 100

    # --- Vision AutoML -----------------------------------------------------
    default_batch_size: int = 32
    default_num_workers: int = 0
    default_val_split: float = 0.2
    default_test_split: float = 0.1
    default_image_classifier_hf_id: str = "google/vit-base-patch16-224"
    model_small_max_param_size: int = 50_000_000
    model_medium_max_param_size: int = 200_000_000

    # --- Training efficiency (generic ML engine) ---------------------------
    # "auto" picks bf16-mixed / 16-mixed on CUDA (depending on hardware
    # support) and falls back to full precision elsewhere.
    training_precision: str = "auto"
    # Use PIL draft-mode JPEG downscaling during decode (large speedup when
    # images are much bigger than the processor's target resolution).
    fast_image_decode: bool = True
    # Upper bound for auto-selected DataLoader worker processes.
    auto_num_workers_max: int = 8


def get_settings() -> Settings:
    """Return a ``Settings`` instance populated from the current environment.

    Not cached: see module docstring.
    """
    return Settings()
