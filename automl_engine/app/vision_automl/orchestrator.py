"""Orchestration of the vision AutoML training pipelines.

The router layer is intentionally thin: it binds HTTP form parameters, builds a
request dataclass, delegates to ``run_vision_pipeline`` or
``run_multimodal_pipeline``, and translates the domain exceptions raised
here into HTTP status codes via ``app.core.api_errors.automl_exception_to_response``.

Both pipelines share the same shape — fetch → resolve → download → extract →
validate → train → serialize → upload — and lean on the service layer to raise
the right typed exceptions (``AutoDWDownloadError`` for AutoDW communication
failures, ``AutoMLValidationError``/``AutoMLRuntimeError`` for the rest). The
orchestrator only converts the few inline checks (ZIP requirement, resource
counts, supported task types, returned validation strings, and the upstream
upload status) into those typed exceptions.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from app.core.concurrency import offload
from app.core.exceptions import AutoDWUploadError, AutoMLValidationError
from app.core.process_log import step
from app.ml_engine.tasks import SUPPORTED_VISION_TASK_TYPES
from app.vision_automl.services import (
    build_upload_payload,
    convert_leaderboard_safely,
    download_dataset,
    extract_and_locate_dataset,
    fetch_dataset_metadata,
    metadata_indicates_zip_download,
    resolve_download_url,
    serialize_and_zip_model,
    train_automl,
    train_automl_multimodal,
    upload_model,
    validate_multimodal_inputs,
    validate_vision_inputs,
)

logger = logging.getLogger(__name__)

_VISION_SUCCESS_MESSAGE = (
    "Vision AutoML training completed successfully and model uploaded to AutoDW"
)
_MULTIMODAL_SUCCESS_MESSAGE = (
    "Multimodal vision AutoML training completed successfully "
    "and model uploaded to AutoDW"
)
_MULTIMODAL_TASK_TYPE = "image_classification_multimodal"


@contextmanager
def dataset_workspace(prefix: str) -> Iterator[Path]:
    """Create a temporary working directory and clean it up afterwards."""
    path = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@dataclass(frozen=True)
class VisionTrainingRequest:
    """Inputs collected from the HTTP form by the router (best_model)."""

    user_id: str
    dataset_id: str
    filename_column: str
    label_column: str
    task_type: str
    time_budget: int
    model_size: str
    num_cpus: int | str
    num_gpus: int | str
    dataset_version: str | None = "v1"
    dataset_split: str | None = None


@dataclass(frozen=True)
class MultimodalTrainingRequest:
    """Inputs collected from the HTTP form by the router (multimodal_best_model)."""

    user_id: str
    dataset_id: str
    filename_column: str
    label_column: str
    time_budget: int
    model_size: str
    dataset_version: str | None = None
    exclude_columns: str | None = None
    dataset_split: str | None = None


@dataclass(frozen=True)
class VisionTrainingResult:
    """Successful outcome of the vision pipeline."""

    message: str
    leaderboard: str
    model_id: str # Added for retraining
    version: str # Added for retraining


@dataclass(frozen=True)
class MultimodalTrainingResult:
    """Successful outcome of the multimodal vision pipeline."""

    message: str
    leaderboard: str
    auxiliary_columns: list[str] = field(default_factory=list)


def _require_zip(metadata: dict) -> None:
    """Reject datasets AutoDW will not serve as a ZIP download."""
    if not metadata_indicates_zip_download(metadata):
        raise AutoMLValidationError("Vision AutoML requires a ZIP dataset.")


def _validate_resource_counts(num_cpus: int | str, num_gpus: int | str) -> None:
    """Reject bad CPU/GPU counts (mirrors the previous inline guards)."""
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


async def run_vision_pipeline(
    req: VisionTrainingRequest, *, task_id: str | None
) -> VisionTrainingResult:
    """Run the vision AutoML pipeline and upload the best model.

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

    with dataset_workspace(f"automl_{req.dataset_id}") as workdir:
        # 4. Download & extract.
        with step("download_and_extract"):
            zip_path = await offload(
                download_dataset,
                download_url,
                workdir,
                metadata.get("original_filename", "dataset.zip"),
            )
            csv_path, images_dir = await offload(
                extract_and_locate_dataset, zip_path, workdir
            )

        # 5. Validate task type and dataset structure.
        with step("validate_inputs"):
            if req.task_type not in SUPPORTED_VISION_TASK_TYPES:
                raise AutoMLValidationError(
                    f"Unsupported task_type '{req.task_type}'. "
                    f"Supported: {sorted(SUPPORTED_VISION_TASK_TYPES)}"
                )

            validation_error = await offload(
                validate_vision_inputs,
                csv_path,
                images_dir,
                req.filename_column,
                req.label_column,
                req.task_type,
            )
            if validation_error:
                raise AutoMLValidationError(validation_error)

        # 6. Train.
        with step("train"):
            optuna_result = await train_automl(
                csv_path=csv_path,
                images_dir=images_dir,
                filename_column=req.filename_column,
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
            model_id, payload = build_upload_payload(
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
            upload_data = upload_resp.json()

            new_version = upload_data.get(
                "version",
                "v1",
            )
            if upload_resp.status_code >= 400:
                logger.error("Model upload failed: %s", upload_resp.text)
                raise AutoDWUploadError(
                    f"Failed to upload model: {upload_resp.text}",
                    status_code=upload_resp.status_code,
                )

    logger.info("Vision AutoML training completed and model uploaded successfully.")
    return VisionTrainingResult(
        message=_VISION_SUCCESS_MESSAGE, leaderboard=leaderboard_str, model_id=model_id, version=new_version,
    )


async def run_multimodal_pipeline(
    req: MultimodalTrainingRequest, *, task_id: str | None
) -> MultimodalTrainingResult:
    """Run the multimodal vision AutoML pipeline and upload the best model.

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

    # 2. Validate dataset kind.
    with step("validate_dataset"):
        _require_zip(metadata)

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

    with dataset_workspace(f"multimodal_{req.dataset_id}") as workdir:
        # 4. Download & extract.
        with step("download_and_extract"):
            zip_path = await offload(
                download_dataset,
                download_url,
                workdir,
                metadata.get("original_filename", "dataset.zip"),
            )
            csv_path, images_dir = await offload(
                extract_and_locate_dataset, zip_path, workdir
            )

        # 5. Auto-discover auxiliary columns and validate dataset structure.
        exclude_cols = (
            [c.strip() for c in req.exclude_columns.split(",") if c.strip()]
            if req.exclude_columns
            else None
        )

        with step("validate_inputs"):
            validation_error, auxiliary_columns = await offload(
                validate_multimodal_inputs,
                csv_path,
                images_dir,
                req.filename_column,
                req.label_column,
                exclude_cols,
            )
            if validation_error:
                raise AutoMLValidationError(validation_error)

        # 6. Train.
        with step("train"):
            optuna_result = await train_automl_multimodal(
                csv_path,
                images_dir,
                req.filename_column,
                req.label_column,
                auxiliary_columns,
                req.time_budget,
                req.model_size,
                workdir=workdir,
            )

        # 7. Serialize.
        with step("serialize_model"):
            zip_path = await offload(serialize_and_zip_model, workdir)
            leaderboard_json, leaderboard_str = convert_leaderboard_safely(
                optuna_result
            )

        # 8. Upload.
        with step("build_upload_payload"):
            model_id, payload = build_upload_payload(
                req.dataset_id,
                req.dataset_version,
                metadata,
                _MULTIMODAL_TASK_TYPE,
                leaderboard_json,
            )
        with step("upload_model"):
            upload_resp = await offload(
                upload_model, upload_url, zip_path, payload, task_id
            )
            upload_data = upload_resp.json()

            new_version = upload_data.get(
                "version",
                "v1",
            )

            if upload_resp.status_code >= 400:
                logger.error("Model upload failed: %s", upload_resp.text)
                raise AutoDWUploadError(
                    f"Failed to upload model: {upload_resp.text}",
                    status_code=upload_resp.status_code,
                )
            return VisionTrainingResult(
                message=_VISION_SUCCESS_MESSAGE,
                leaderboard=leaderboard_str,
                model_id=model_id,
                version=new_version,
            )

    logger.info(
        "Multimodal vision AutoML training completed and model uploaded successfully."
    )
    return MultimodalTrainingResult(
        message=_MULTIMODAL_SUCCESS_MESSAGE,
        leaderboard=leaderboard_str,
        auxiliary_columns=auxiliary_columns,
    )
