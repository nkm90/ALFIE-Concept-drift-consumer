"""Service layer for vision AutoML workflows.

Mirrors the structure of tabular_automl/services.py so both pipelines
share a consistent public API consumed by their orchestrators. Training is
delegated to the consolidated generic ML engine (``app.ml_engine``).
"""

import json
import logging
import os
import shutil
from pathlib import Path

import pandas as pd

from app.core.concurrency import offload
from app.core.dataset_extraction import (  # noqa: F401 – re-exported for router
    collect_missing_files,
    extract_and_locate_dataset,
    normalize_dataframe_filenames,
)
from app.core.exceptions import AutoMLSerializationError
from app.core.service_helpers import (  # noqa: F401 – re-exported for router
    build_upload_payload as _core_build_upload_payload,
    download_dataset as _core_download_dataset,
    fetch_dataset_metadata,
    metadata_indicates_zip_download,
    resolve_download_url,
    upload_model,
)
from app.core.utils import jinja_environment, render_template
from app.ml_engine.trainer import run_optuna_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset fetch & extraction
# ---------------------------------------------------------------------------


def download_dataset(download_url: str, workdir: Path, original_filename: str) -> Path:
    """Stream-download the ZIP dataset and return its local path."""
    dest_path = workdir / original_filename
    return _core_download_dataset(download_url, dest_path)


# ---------------------------------------------------------------------------
# Validation  (mirrors tabular: validate_tabular_inputs)
# ---------------------------------------------------------------------------


_IMAGE_TASK_TYPES: frozenset[str] = frozenset(
    {
        "image_classification",
        "image_segmentation",
        "object_detection",
        "video_classification",
        "keypoint_detection",
    }
)

_DETECTION_EXTRA_COLUMNS: dict[str, list[str]] = {
    "object_detection": ["boxes", "class_labels"],
    "keypoint_detection": ["keypoints"],
}

# File types PIL/torchvision can decode for the image-based tasks.
IMAGE_FILE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
)


def collect_non_image_files(df: pd.DataFrame, filename_column: str) -> list[str]:
    """Return referenced files whose type is not a supported image format.

    Modality-specific counterpart to the shared ``collect_missing_files``:
    the shared helpers check packaging/existence, this checks that the
    filenames in the CSV actually point at images.
    """
    invalid = []
    for name in df[filename_column].astype(str):
        ext = os.path.splitext(str(name).strip())[1].lower()
        if ext not in IMAGE_FILE_EXTENSIONS:
            invalid.append(name)
    return invalid


def validate_vision_inputs(
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    task_type: str = "image_classification",
) -> str | None:
    """Validate dataset structure for the given image task type.

    Returns an error string on failure, or None if everything is valid.
    Mirrors the signature/contract of tabular's ``validate_tabular_inputs``.

    Args:
        csv_path: Path to the labels CSV.
        images_dir: Root directory containing image/video files.
        filename_column: Column name containing file paths.
        label_column: Column name containing labels (classification tasks).
        task_type: One of the supported image task type slugs.
    """
    if task_type not in _IMAGE_TASK_TYPES:
        return f"Unsupported vision task_type '{task_type}'"

    # Image tasks — CSV + image presence checks
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Could not read labels CSV: {e}"

    for col, role in [(filename_column, "Filename"), (label_column, "Label")]:
        if col not in df.columns:
            return f"{role} column '{col}' not found in labels CSV"

    # Detection/segmentation tasks — validate annotation columns
    if task_type in _DETECTION_EXTRA_COLUMNS:
        extra = _DETECTION_EXTRA_COLUMNS[task_type]
        missing_cols = [c for c in extra if c not in df.columns]
        if missing_cols:
            return (
                f"Required annotation column(s) missing for {task_type}: {missing_cols}"
            )

    df = normalize_dataframe_filenames(df, filename_column, csv_path)

    invalid = collect_non_image_files(df, filename_column)
    if invalid:
        preview = invalid[:5]
        suffix = "..." if len(invalid) > 5 else ""
        return f"Found {len(invalid)} non-image file(s): {preview}{suffix}"

    missing = collect_missing_files(df, images_dir, filename_column, label_column)
    if missing:
        preview = missing[:5]
        suffix = "..." if len(missing) > 5 else ""
        return f"Missing {len(missing)} image file(s): {preview}{suffix}"

    return None


def _discover_auxiliary_columns(
    df: pd.DataFrame,
    filename_column: str,
    label_column: str,
    exclude_columns: list[str] | None = None,
) -> list[str]:
    """Auto-discover auxiliary columns in *df*.

    All columns except ``filename_column``, ``label_column``, and any
    columns listed in ``exclude_columns`` are treated as auxiliary features.
    """
    exclude = {filename_column, label_column}
    if exclude_columns:
        exclude.update(exclude_columns)
    return [col for col in df.columns if col not in exclude]


def validate_multimodal_inputs(
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    exclude_columns: list[str] | None = None,
) -> tuple[str | None, list[str]]:
    """Validate dataset structure for multimodal image classification.

    Auto-discovers auxiliary columns (all columns except ``filename_column``,
    ``label_column``, and ``exclude_columns``), validates their presence and
    contents, and checks image file existence.

    Returns:
        (error_string_or_None, auxiliary_columns_list).
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Could not read labels CSV: {e}", []

    for col, role in [(filename_column, "Filename"), (label_column, "Label")]:
        if col not in df.columns:
            return f"{role} column '{col}' not found in labels CSV", []

    auxiliary_columns = _discover_auxiliary_columns(
        df, filename_column, label_column, exclude_columns
    )

    if not auxiliary_columns:
        return (
            "No auxiliary columns found in the CSV. "
            "Use the standard /best_model/ endpoint for image-only classification.",
            [],
        )

    for col in auxiliary_columns:
        if df[col].isnull().all():
            return f"Auxiliary column '{col}' is entirely null", []

    df = normalize_dataframe_filenames(df, filename_column, csv_path)

    invalid = collect_non_image_files(df, filename_column)
    if invalid:
        preview = invalid[:5]
        suffix = "..." if len(invalid) > 5 else ""
        return f"Found {len(invalid)} non-image file(s): {preview}{suffix}", []

    missing = collect_missing_files(df, images_dir, filename_column, label_column)
    if missing:
        preview = missing[:5]
        suffix = "..." if len(missing) > 5 else ""
        return f"Missing {len(missing)} image file(s): {preview}{suffix}", []

    return None, auxiliary_columns


# ---------------------------------------------------------------------------
# Training  (mirrors tabular: train_automl)
# ---------------------------------------------------------------------------


async def train_automl(
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    time_budget: int,
    model_size: str,
    workdir: Path,
    task_type: str = "image_classification",
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
) -> dict:
    """Run Optuna-based vision AutoML and return the result dict."""
    return await offload(
        run_optuna_search,
        task_type=task_type,
        csv_path=csv_path,
        images_dir=images_dir,
        filename_column=filename_column,
        label_column=label_column,
        n_trials=max(1, min(25, time_budget // 60)),
        timeout=time_budget,
        model_size=model_size,
        workdir=workdir,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )


async def train_automl_multimodal(
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    auxiliary_columns: list[str],
    time_budget: int,
    model_size: str,
    workdir: Path,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
) -> dict:
    """Run Optuna-based multimodal vision AutoML and return the result dict."""
    return await offload(
        run_optuna_search,
        task_type="image_classification_multimodal",
        csv_path=csv_path,
        images_dir=images_dir,
        filename_column=filename_column,
        label_column=label_column,
        auxiliary_columns=auxiliary_columns,
        n_trials=max(1, min(25, time_budget // 60)),
        timeout=time_budget,
        model_size=model_size,
        workdir=workdir,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )


# ---------------------------------------------------------------------------
# Artifact packaging  (mirrors tabular: serialize_and_zip_predictor)
# ---------------------------------------------------------------------------


def deployment_instructions() -> str:
    """Return the vision deployment instructions rendered from a template."""
    if jinja_environment is not None:
        return render_template(jinja_environment, "vision_deployment_instructions.md")
    else:
        return "No instructions found"


def vision_data_instructions() -> str:
    """Return the instructions from what kind of data is accepted by the vision AutoML engine"""
    if jinja_environment is not None:
        try:
            return render_template(jinja_environment, "vision_accepted_format.md")
        except Exception as e:
            logger.error(f"Failed to render accepted format instructions: {e}")
            return "No accepted format instructions available"
    else:
        logger.warning("jinja_environment is None, returning default formats")
        return "Ask the agent for help"


def serialize_and_zip_model(workdir: Path) -> Path:
    """
    Package the trained model directory into a ZIP archive.

    Returns the path to the ZIP file.
    Mirrors tabular's ``serialize_and_zip_predictor``.
    """
    model_dir = workdir / "model"
    model_dir.mkdir(exist_ok=True)

    try:
        with open(model_dir / "vision_deployment_instructions.md", "w") as f:
            f.write(deployment_instructions())
    except Exception as e:
        logger.debug(f"No deployment_instructions found, {e}")

    zip_base = workdir / "vision_model"
    try:
        shutil.make_archive(str(zip_base), "zip", model_dir)
    except Exception as e:
        logger.error(f"Failed to create zip archive: {e}")
        raise AutoMLSerializationError(f"Failed to zip model: {e}") from e
    zip_path = zip_base.with_suffix(".zip")
    logger.debug("Model artifacts zipped to %s", zip_path)
    return zip_path


# ---------------------------------------------------------------------------
# Leaderboard  (mirrors tabular: convert_leaderboard_safely)
# ---------------------------------------------------------------------------


def convert_leaderboard_safely(optuna_result: dict) -> tuple[dict, str]:
    """
    Extract leaderboard information from an Optuna result dict.

    Returns (leaderboard_json, leaderboard_str) — mirrors the tabular
    ``convert_leaderboard_safely`` signature so orchestrators can treat both
    pipelines identically.
    """
    leaderboard_json = {
        "best_loss": optuna_result.get("best_value"),
        "best_params": optuna_result.get("best_params"),
        "trials": optuna_result.get("n_trials"),
    }
    leaderboard_str = json.dumps(leaderboard_json, indent=2)
    return leaderboard_json, leaderboard_str


# ---------------------------------------------------------------------------
# Upload payload  (mirrors tabular: build_upload_payload + upload_model)
# ---------------------------------------------------------------------------


def build_upload_payload(
    dataset_id: str,
    dataset_version: str | None,
    metadata: dict,
    task_type: str,
    leaderboard_json: dict,
) -> tuple[str, dict]:
    """Return (model_id, form_data_dict) for the AutoDW upload request."""
    return _core_build_upload_payload(
        dataset_id,
        dataset_version,
        metadata,
        task_type,
        leaderboard_json,
        model_id_prefix="vision_automl",
        name=f"Vision AutoML Model - {dataset_id}",
        description="AutoML trained vision model",
        framework="pytorch",
    )
