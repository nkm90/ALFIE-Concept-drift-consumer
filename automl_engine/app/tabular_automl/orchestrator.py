"""Orchestration of the tabular AutoML training pipeline.

The router layer is intentionally thin: it binds HTTP form parameters, builds a
``TabularTrainingRequest``, delegates to ``run_training_pipeline``, and
translates the domain exceptions raised here into HTTP status codes via
``app.core.api_errors.automl_exception_to_response``.

All of the multi-step business logic — fetch → resolve → download → validate →
train → serialize → upload — lives in this module so it can be tested and reused
independently of FastAPI.

Every failure path raises a typed ``AutoMLError`` subclass whose message carries
the same context the previous inline handlers produced, so callers and tests can
rely on stable error text:

* AutoMLValidationError (→ 400) for bad caller inputs and unsupported
  dataset/task types.
* AutoDWDownloadError (→ 502) for AutoDW communication failures during metadata
  fetch or dataset download.
* AutoDWUploadError (→ upstream status) for upload failures.
* AutoMLRuntimeError (→ 500) for everything else.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests

from app.core.concurrency import offload
from app.core.config import get_settings
from app.core.exceptions import (
    AutoDWDownloadError,
    AutoDWUploadError,
    AutoMLRuntimeError,
    AutoMLValidationError,
)
from app.core.process_log import step
from app.ml_engine.model import SUPPORTED_TABULAR_TASK_TYPES
from app.tabular_automl.services import (
    SUPPORTED_FILE_TYPES,
    build_upload_payload,
    convert_leaderboard_safely,
    download_dataset,
    fetch_dataset_metadata,
    resolve_download_url,
    serialize_and_zip_predictor,
    train_automl,
    upload_model,
    validate_tabular_inputs,
)

logger = logging.getLogger(__name__)

_SUCCESS_MESSAGE = "AutoML training completed successfully and model uploaded to AutoDW"


@dataclass(frozen=True)
class TabularTrainingRequest:
    """Inputs collected from the HTTP form by the router."""

    user_id: str
    dataset_id: str
    target_column_name: str
    task_type: str
    time_budget: int
    num_cpus: int | str
    num_gpus: int | str
    dataset_version: str | None = "v1"
    time_stamp_column_name: str | None = None
    dataset_split: str | None = None


@dataclass(frozen=True)
class TrainingResult:
    """Successful outcome of the training pipeline."""

    message: str
    leaderboard: str


def _validate_request(req: TabularTrainingRequest) -> None:
    """Reject obviously bad caller inputs.

    Raises AutoMLValidationError (→ HTTP 400) with the same messages the
    previous inline guards produced.
    """
    if not req.user_id or not isinstance(req.user_id, str) or not req.user_id.strip():
        raise AutoMLValidationError("user_id must be a non-empty string")

    if (
        not req.dataset_id
        or not isinstance(req.dataset_id, str)
        or not req.dataset_id.strip()
    ):
        raise AutoMLValidationError("dataset_id must be a non-empty string")

    if isinstance(req.num_cpus, str):
        if req.num_cpus != "auto":
            raise AutoMLValidationError("num_cpus must be either a number or auto")
    elif req.num_cpus < 1:
        raise AutoMLValidationError("num_cpus must be a positive integer (at least 1)")

    if isinstance(req.num_gpus, str):
        if req.num_gpus != "auto":
            raise AutoMLValidationError("num_gpus must be either a number or auto")
    elif req.num_gpus < 1:
        raise AutoMLValidationError("num_gpus must be a positive integer (at least 1)")

    if (
        not req.target_column_name
        or not isinstance(req.target_column_name, str)
        or not req.target_column_name.strip()
    ):
        raise AutoMLValidationError("target_column_name must be a non-empty string")

    if req.task_type not in SUPPORTED_TABULAR_TASK_TYPES:
        raise AutoMLValidationError(
            f"Invalid task_type '{req.task_type}'. "
            f"Must be one of: {SUPPORTED_TABULAR_TASK_TYPES}"
        )

    if not isinstance(req.time_budget, int) or req.time_budget <= 0:
        raise AutoMLValidationError("time_budget must be a positive integer")

    if req.dataset_split is not None and req.dataset_split not in (
        "train",
        "test",
        "drift",
    ):
        raise AutoMLValidationError(
            "dataset_split must be one of: 'train', 'test', 'drift'"
        )


def _validate_metadata(metadata: object) -> None:
    """Check the metadata blob returned by AutoDW.

    A missing/empty blob is an AutoDW problem (→ 502); a missing or unsupported
    ``file_type`` is a caller/dataset problem (→ 400).
    """
    if not isinstance(metadata, dict) or not metadata:
        raise AutoDWDownloadError("Invalid or empty metadata received from AutoDW")

    file_type = metadata.get("file_type")
    if not file_type:
        raise AutoMLValidationError("file_type not found in dataset metadata")
    if file_type not in SUPPORTED_FILE_TYPES:
        raise AutoMLValidationError(
            f"Unsupported file type '{file_type}'. "
            f"Supported types: {SUPPORTED_FILE_TYPES}"
        )


async def run_training_pipeline(
    req: TabularTrainingRequest, *, task_id: str | None
) -> TrainingResult:
    """Run the full tabular AutoML pipeline and upload the best model.

    Raises a typed ``AutoMLError`` subclass on any failure; the router maps
    these to HTTP status codes.
    """
    with step("validate_request"):
        _validate_request(req)

    autodw_base = get_settings().autodw_url
    if not autodw_base:
        raise AutoMLRuntimeError("AUTODW_URL environment variable is not set")
    upload_url = f"{autodw_base}/ai-models/upload/single/{req.user_id}"

    # 1. Fetch dataset metadata from AutoDW.
    try:
        with step("fetch_metadata"):
            metadata = await offload(
                fetch_dataset_metadata,
                autodw_base,
                req.user_id,
                req.dataset_id,
                req.dataset_version,
            )
    except requests.RequestException as e:
        logger.error("Failed to fetch dataset metadata: %s", e)
        raise AutoDWDownloadError(
            f"Failed to fetch dataset metadata from AutoDW: {e}"
        ) from e
    except Exception as e:
        logger.error("Unexpected error fetching metadata: %s", e)
        raise AutoMLRuntimeError(f"Unexpected error fetching metadata: {e}") from e

    with step("validate_metadata"):
        _validate_metadata(metadata)

    # 2. Resolve the correct download URL (respecting splits if present).
    try:
        with step("resolve_download_url"):
            download_url = resolve_download_url(
                autodw_base,
                req.user_id,
                req.dataset_id,
                req.dataset_version,
                metadata,
                req.dataset_split,
            )
    except Exception as e:
        logger.error("Failed to resolve download URL: %s", e)
        raise AutoMLRuntimeError(f"Failed to resolve download URL: {e}") from e

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        original_filename = metadata.get("original_filename", "train.csv")
        if not isinstance(original_filename, str) or not original_filename:
            original_filename = "train.csv"
        dataset_path = tmp_path / original_filename

        # 3. Download the dataset file.
        try:
            with step("download_dataset"):
                await offload(download_dataset, download_url, dataset_path)
        except requests.RequestException as e:
            logger.error("Failed to download dataset: %s", e)
            raise AutoDWDownloadError(
                f"Failed to download dataset from AutoDW: {e}"
            ) from e
        except Exception as e:
            logger.error("Unexpected error downloading dataset: %s", e)
            raise AutoMLRuntimeError(
                f"Unexpected error downloading dataset: {e}"
            ) from e

        if not dataset_path.exists():
            raise AutoMLRuntimeError("Dataset file was not created after download")

        # 4. Validate user-supplied parameters against the dataset.
        try:
            with step("validate_inputs"):
                validation_error = await offload(
                    validate_tabular_inputs,
                    train_path=dataset_path,
                    target_column_name=req.target_column_name,
                    time_stamp_column_name=req.time_stamp_column_name,
                    task_type=req.task_type,
                )
                if validation_error:
                    raise AutoMLValidationError(validation_error)
        except AutoMLValidationError:
            raise
        except Exception as e:
            logger.error("Unexpected error during validation: %s", e)
            raise AutoMLRuntimeError(f"Unexpected error during validation: {e}") from e

        # 5. Train an AutoML model within the given time budget.
        save_model_path = tmp_path / "automl_model"
        try:
            with step("train"):
                leaderboard, predictor = await offload(
                    train_automl,
                    dataset_path=dataset_path,
                    save_model_path=save_model_path,
                    target_column_name=req.target_column_name,
                    task_type=req.task_type,
                    time_budget=req.time_budget,
                    num_cpus=req.num_cpus,
                    num_gpus=req.num_gpus,
                )
                if leaderboard is None:
                    raise AutoMLRuntimeError(
                        "Training completed but leaderboard is empty"
                    )
        except AutoMLValidationError as e:
            logger.error("Validation error during training: %s", e)
            raise AutoMLValidationError(f"Training validation failed: {e}") from e
        except AutoMLRuntimeError as e:
            logger.error("Training runtime error: %s", e)
            raise AutoMLRuntimeError(f"Model training failed: {e}") from e
        except Exception as e:
            logger.error("Unexpected error during training: %s", e)
            raise AutoMLRuntimeError(f"Unexpected error during training: {e}") from e

        # 6. Serialize and zip the best predictor.
        try:
            with step("serialize_model"):
                zip_path = await offload(
                    serialize_and_zip_predictor, predictor, save_model_path, tmp_path
                )
                leaderboard_json, leaderboard_str = convert_leaderboard_safely(
                    leaderboard
                )
        except Exception as e:
            logger.error("Failed to serialize and zip model: %s", e)
            raise AutoMLRuntimeError(f"Failed to serialize model: {e}") from e

        if not zip_path.exists():
            raise AutoMLRuntimeError("Model zip file was not created")

        # 7. Upload the model and leaderboard back to AutoDW.
        try:
            with step("build_upload_payload"):
                _, payload = build_upload_payload(
                    req.dataset_id,
                    req.dataset_version,
                    metadata,
                    req.task_type,
                    leaderboard_json,
                )
        except Exception as e:
            logger.error("Failed to build upload payload: %s", e)
            raise AutoMLRuntimeError(f"Failed to build upload payload: {e}") from e

        try:
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
        except AutoDWUploadError:
            raise
        except requests.RequestException as e:
            logger.error("Network error uploading model: %s", e)
            raise AutoDWUploadError(f"Failed to upload model to AutoDW: {e}") from e
        except Exception as e:
            logger.error("Unexpected error uploading model: %s", e)
            raise AutoMLRuntimeError(f"Unexpected error uploading model: {e}") from e

    logger.info("AutoML training completed and model uploaded successfully.")
    return TrainingResult(message=_SUCCESS_MESSAGE, leaderboard=leaderboard_str)
