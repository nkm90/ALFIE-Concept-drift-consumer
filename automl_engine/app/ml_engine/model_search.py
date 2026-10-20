"""Hugging Face model discovery helpers for the ML engine.

Searches the Hugging Face hub for candidate models annotated with parameter
counts and filters them into small/medium/large tiers so pipelines can pick
models matching a requested ``model_size``.
"""

import logging
from typing import Any

from huggingface_hub import HfApi

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def get_num_params_if_available(
    repo_id: str, revision: str | None = None
) -> int | None:
    """Try to retrieve number of parameters for a HF model, if available."""
    logger.debug("Fetching parameter count for model %s", repo_id)
    api = HfApi()
    try:
        info = api.model_info(repo_id, revision=revision, files_metadata=True)
        num_params = getattr(info, "safetensors", None)
        if num_params is not None:
            return num_params.total
    except Exception as e:
        logger.warning("Failed to retrieve num_params for %s: %s", repo_id, e)
    return None


def search_hf_for_pytorch_models_with_estimated_parameters(
    filter: str = "image-classification", limit: int = 3, sort: str = "downloads"
) -> list[dict[str, Any]]:
    """Search HF for PyTorch image-classification models annotated with param counts."""
    logger.info("Searching Hugging Face models for filter='%s'", filter)
    api = HfApi()
    models = api.list_models(
        filter=filter,
        library="pytorch",
        sort=sort,
        direction=-1,
        limit=limit,
    )

    results: list[dict[str, Any]] = []
    for m in models:
        num_params = get_num_params_if_available(m.id)
        if num_params:
            results.append(
                {
                    "model_id": m.id,
                    "downloads": getattr(m, "downloads", None),
                    "likes": getattr(m, "likes", None),
                    "last_modified": getattr(m, "lastModified", None),
                    "private": getattr(m, "private", None),
                    "num_params": num_params,
                }
            )

    logger.info("Found %d models with parameter info", len(results))
    return results


def sort_models_by_size(
    models: list[dict[str, Any]], size_tier: str
) -> list[dict[str, Any]]:
    """Filter and sort models by size tier based on estimated parameter counts."""
    logger.info("Sorting models by size tier: %s", size_tier)
    tier = str(size_tier).strip().lower()

    settings = get_settings()
    SMALL_MAX: int = settings.model_small_max_param_size
    MEDIUM_MIN: int = SMALL_MAX + 1
    MEDIUM_MAX: int = settings.model_medium_max_param_size
    LARGE_MIN: int = MEDIUM_MAX + 1

    def in_tier(m: dict[str, Any]) -> bool:
        n = m.get("num_params")
        if n is None:
            return False
        if tier == "small":
            return 0 <= n <= SMALL_MAX
        if tier == "medium":
            return MEDIUM_MIN <= n <= MEDIUM_MAX
        if tier == "large":
            return n >= LARGE_MIN
        return True

    filtered = [m for m in models if in_tier(m)]
    if not filtered:
        logger.warning("No models matched tier '%s'; falling back to all models", tier)
        filtered = models

    return sorted(
        filtered, key=lambda m: (m.get("num_params") is None, m.get("num_params", 0))
    )
