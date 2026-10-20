"""Orchestration of the text AutoML training pipeline.

The router layer is intentionally thin: it binds HTTP form parameters, builds a
``TextTrainingRequest``, delegates to ``run_text_pipeline``, and translates
the domain exceptions raised here into HTTP status codes via
``app.core.api_errors.automl_exception_to_response``.

The pipeline shares the same shape as vision — fetch → resolve → download →
extract → validate → train → serialize → upload — and leans on the service
layer to raise the right typed exceptions. Training itself is delegated to the
consolidated generic ML engine (``app.ml_engine``).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from app.core.concurrency import offload
from app.core.exceptions import AutoDWUploadError, AutoMLValidationError
from app.core.process_log import step
from app.ml_engine.tasks import SUPPORTED_TEXT_TASK_TYPES
from app.text_automl.services import (
    build_upload_payload,
    convert_leaderboard_safely,
    download_dataset,
    extract_and_locate_dataset,
    fetch_dataset_metadata,
    metadata_indicates_zip_download,
    resolve_download_url,
    serialize_and_zip_model,
    train_automl,
    upload_model,
    validate_text_inputs,
)

logger = logging.getLogger(__name__)

_TEXT_SUCCESS_MESSAGE = (
    "Text AutoML training completed successfully and model uploaded to AutoDW"
)


@contextmanager
def dataset_workspace(prefix: str) -> Iterator[Path]:
    """Create a temporary working directory and clean it up afterwards."""
    path = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@dataclass(frozen=True)
class TextTrainingRequest:
    """Inputs collected from the HTTP form by the router (best_model)."""

    user_id: str
    dataset_id: str
    text_column: str
    label_column: str
    task_type: str
    time_budget: int
    model_size: str
    num_cpus: int | str
    num_gpus: int | str
    dataset_version: str | None = "v1"
    dataset_split: str | None = None


@dataclass(frozen=True)
class TextTrainingResult:
    """Successful outcome of the text pipeline."""

    message: str
    leaderboard: str


def _require_zip(metadata: dict) -> None:
    """Reject datasets AutoDW will not serve as a ZIP download."""
    if not metadata_indicates_zip_download(metadata):
        raise AutoMLValidationError("Text AutoML requires a ZIP dataset.")


def _validate_resource_counts(num_cpus: int | str, num_gpus: int | str) -> None:
    """Reject bad CPU/GPU counts (mirrors the vision inline guards)."""
    if isinstance(num_cpus, str):
        if num_cpus != "auto":
            raise AutoMLValidationError("num_cpus must be either a number or auto")
    elif num_cpus < 0:
        raise AutoMLValidationError("num_cpus must be greater than 1")

    if isinstance(num_gpus, str):
        if num_gpus != "auto":
            raise AutoMLValidationError("num_gpus must be either a number or auto")
    elif num_gpus < 0:
        raise AutoMLValidationError("num_gpus must be greater than 1")


def _autodw_base() -> str:
    """Return the AutoDW base URL from the environment (localhost fallback)."""
    return os.getenv("AUTODW_URL", "http://localhost:8000")


async def run_text_pipeline(
    req: TextTrainingRequest, *, task_id: str | None
) -> TextTrainingResult:
    """Run the text AutoML pipeline and upload the best model.

    Raises a typed ``AutoMLError`` subclass on any failure; the router maps
    these to HTTP status codes.
    """
    autodw_base = _autodw_base()
    upload_url = f"{autodw_base}/ai-models/upload/single/{req.user_id}"

    # 1. Fetch dataset metadata from AutoDW.
    with step("fetch_metadata"):
        metadata = await offload(
            fetch_dataset_metadata,
            autodw_base,
            req.user_id,
            req.dataset_id,
            req.dataset_version,
        )

    # 2. Validate dataset kind and resource counts.
    with step("validate_dataset"):
        _require_zip(metadata)
        _validate_resource_counts(req.num_cpus, req.num_gpus)

    # 3. Resolve the correct download URL (respecting splits if present).
    with step("resolve_download_url"):
        download_url = resolve_download_url(
            autodw_base,
            req.user_id,
            req.dataset_id,
            req.dataset_version,
            metadata,
            req.dataset_split,
        )

    with dataset_workspace(f"text_automl_{req.dataset_id}") as workdir:
        # 4. Download & extract.
        with step("download_and_extract"):
            zip_path = await offload(
                download_dataset,
                download_url,
                workdir,
                metadata.get("original_filename", "dataset.zip"),
            )
            csv_path, _media_dir = await offload(
                extract_and_locate_dataset, zip_path, workdir
            )

        # 5. Validate task type and CSV structure.
        with step("validate_inputs"):
            if req.task_type not in SUPPORTED_TEXT_TASK_TYPES:
                raise AutoMLValidationError(
                    f"Unsupported task_type '{req.task_type}'. "
                    f"Supported: {sorted(SUPPORTED_TEXT_TASK_TYPES)}"
                )

            validation_error = await offload(
                validate_text_inputs,
                csv_path,
                req.task_type,
                req.text_column,
                req.label_column,
            )
            if validation_error:
                raise AutoMLValidationError(validation_error)

        # 6. Train.
        with step("train"):
            optuna_result = await train_automl(
                csv_path=csv_path,
                text_column=req.text_column,
                label_column=req.label_column,
                time_budget=req.time_budget,
                model_size=req.model_size,
                workdir=workdir,
                task_type=req.task_type,
                num_cpus=req.num_cpus,
                num_gpus=req.num_gpus,
            )

        # 7. Serialize.
        with step("serialize_model"):
            zip_path = await offload(serialize_and_zip_model, workdir)
            leaderboard_json, leaderboard_str = convert_leaderboard_safely(
                optuna_result
            )

        # 8. Upload.
        with step("build_upload_payload"):
            _, payload = build_upload_payload(
                req.dataset_id,
                req.dataset_version,
                metadata,
                req.task_type,
                leaderboard_json,
            )
        with step("upload_model"):
            upload_resp = await offload(
                upload_model, upload_url, zip_path, payload, task_id
            )
            if upload_resp.status_code >= 400:
                logger.error("Model upload failed: %s", upload_resp.text)
                raise AutoDWUploadError(
                    f"Failed to upload model: {upload_resp.text}",
                    status_code=upload_resp.status_code,
                )

    logger.info("Text AutoML training completed and model uploaded successfully.")
    return TextTrainingResult(
        message=_TEXT_SUCCESS_MESSAGE, leaderboard=leaderboard_str
    )
