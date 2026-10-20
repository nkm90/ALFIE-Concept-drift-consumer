"""Service layer for audio AutoML workflows.

Mirrors the structure of vision_automl/services.py: validation, the training
wrapper around the consolidated generic ML engine (``app.ml_engine``),
artifact packaging, and the instruction templates.
"""

import json
import logging
import os
import shutil
from pathlib import Path

import pandas as pd

from app.core.concurrency import offload
from app.core.dataset_extraction import (
    extract_and_locate_dataset,  # noqa: F401 – re-exported for orchestrator
)
from app.core.exceptions import AutoMLSerializationError
from app.core.service_helpers import (  # noqa: F401 – re-exported for orchestrator
    build_upload_payload as _core_build_upload_payload,
    download_dataset as _core_download_dataset,
    fetch_dataset_metadata,
    metadata_indicates_zip_download,
    resolve_download_url,
    upload_model,
)
from app.core.utils import jinja_environment, render_template
from app.ml_engine.tasks import SUPPORTED_AUDIO_TASK_TYPES
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
# Validation
# ---------------------------------------------------------------------------

# File types torchaudio can decode.
AUDIO_FILE_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
)


def collect_non_audio_files(df: pd.DataFrame, filename_column: str) -> list[str]:
    """Return referenced files whose type is not a supported audio format.

    Modality-specific counterpart to the shared ``collect_missing_files``:
    the shared helpers check packaging/existence, this checks that the
    filenames in the CSV actually point at audio clips.
    """
    invalid = []
    for name in df[filename_column].astype(str):
        ext = os.path.splitext(str(name).strip())[1].lower()
        if ext not in AUDIO_FILE_EXTENSIONS:
            invalid.append(name)
    return invalid


def validate_audio_inputs(
    csv_path: Path,
    audio_dir: Path,
    filename_column: str,
    label_column: str,
    task_type: str = "audio_classification",
) -> str | None:
    """Validate dataset structure for the given audio task type.

    Returns an error string on failure, or None if everything is valid.
    Mirrors the signature/contract of vision's ``validate_vision_inputs``.
    """
    if task_type not in SUPPORTED_AUDIO_TASK_TYPES:
        return f"Unsupported audio task_type '{task_type}'"

    if not csv_path.exists():
        return f"Labels CSV not found: {csv_path}"
    if not audio_dir.exists():
        return f"Audio directory not found: {audio_dir}"
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Could not read labels CSV: {e}"
    for col, role in [(filename_column, "Filename"), (label_column, "Label")]:
        if col not in df.columns:
            return f"{role} column '{col}' not found in labels CSV"

    invalid = collect_non_audio_files(df, filename_column)
    if invalid:
        preview = invalid[:5]
        suffix = "..." if len(invalid) > 5 else ""
        return f"Found {len(invalid)} non-audio file(s): {preview}{suffix}"
    return None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


async def train_automl(
    csv_path: Path,
    audio_dir: Path,
    filename_column: str,
    label_column: str,
    time_budget: int,
    model_size: str,
    workdir: Path,
    task_type: str = "audio_classification",
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
) -> dict:
    """Run Optuna-based audio AutoML and return the result dict."""
    return await offload(
        run_optuna_search,
        task_type=task_type,
        csv_path=csv_path,
        images_dir=audio_dir,
        audio_dir=audio_dir,
        filename_column=filename_column,
        label_column=label_column,
        n_trials=max(1, min(25, time_budget // 60)),
        timeout=time_budget,
        model_size=model_size,
        workdir=workdir,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )


# ---------------------------------------------------------------------------
# Artifact packaging
# ---------------------------------------------------------------------------


def deployment_instructions() -> str:
    """Return the audio deployment instructions rendered from a template."""
    if jinja_environment is not None:
        return render_template(jinja_environment, "audio_deployment_instructions.md")
    else:
        return "No instructions found"


def audio_data_instructions() -> str:
    """Return the instructions from what kind of data is accepted by the audio AutoML engine"""
    if jinja_environment is not None:
        try:
            return render_template(jinja_environment, "audio_accepted_format.md")
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
    Mirrors vision's ``serialize_and_zip_model``.
    """
    model_dir = workdir / "model"
    model_dir.mkdir(exist_ok=True)

    try:
        with open(model_dir / "audio_deployment_instructions.md", "w") as f:
            f.write(deployment_instructions())
    except Exception as e:
        logger.debug(f"No deployment_instructions found, {e}")

    zip_base = workdir / "audio_model"
    try:
        shutil.make_archive(str(zip_base), "zip", model_dir)
    except Exception as e:
        logger.error(f"Failed to create zip archive: {e}")
        raise AutoMLSerializationError(f"Failed to zip model: {e}") from e
    zip_path = zip_base.with_suffix(".zip")
    logger.debug("Model artifacts zipped to %s", zip_path)
    return zip_path


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def convert_leaderboard_safely(optuna_result: dict) -> tuple[dict, str]:
    """
    Extract leaderboard information from an Optuna result dict.

    Returns (leaderboard_json, leaderboard_str) — mirrors the vision
    ``convert_leaderboard_safely`` signature.
    """
    leaderboard_json = {
        "best_loss": optuna_result.get("best_value"),
        "best_params": optuna_result.get("best_params"),
        "trials": optuna_result.get("n_trials"),
    }
    leaderboard_str = json.dumps(leaderboard_json, indent=2)
    return leaderboard_json, leaderboard_str


# ---------------------------------------------------------------------------
# Upload payload
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
        model_id_prefix="audio_automl",
        name=f"Audio AutoML Model - {dataset_id}",
        description="AutoML trained audio model",
        framework="pytorch",
    )
