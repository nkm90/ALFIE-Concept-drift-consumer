"""Service layer for text AutoML workflows.

Mirrors the structure of vision_automl/services.py: validation, the training
wrapper around the consolidated generic ML engine (``app.ml_engine``),
artifact packaging, and the instruction templates.
"""

import json
import logging
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
from app.ml_engine.tasks import SUPPORTED_TEXT_TASK_TYPES
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

_TEXT_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "text_classification": ["text", "label"],
    "question_answering": ["question", "context", "answer_start", "answer_text"],
    "causal_lm": ["text"],
    "seq2seq_lm": ["input_text", "target_text"],
    "masked_lm": ["text"],
}


def validate_text_inputs(
    csv_path: Path,
    task_type: str,
    text_column: str = "text",
    label_column: str = "label",
) -> str | None:
    """Validate CSV structure for the given text task type.

    Returns an error string on failure, or None if everything is valid.
    Mirrors the signature/contract of vision's ``validate_vision_inputs``.
    """
    if task_type not in SUPPORTED_TEXT_TASK_TYPES:
        return f"Unsupported text task_type '{task_type}'"

    # Text tasks — validate CSV + required columns
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Could not read labels CSV: {e}"

    if task_type == "text_classification":
        required = [text_column, label_column]
    else:
        required = _TEXT_REQUIRED_COLUMNS[task_type]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        return f"Required column(s) missing for {task_type}: {missing_cols}"
    return None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


async def train_automl(
    csv_path: Path,
    text_column: str,
    label_column: str,
    time_budget: int,
    model_size: str,
    workdir: Path,
    task_type: str = "text_classification",
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
) -> dict:
    """Run Optuna-based text AutoML and return the result dict."""
    return await offload(
        run_optuna_search,
        task_type=task_type,
        csv_path=csv_path,
        text_column=text_column,
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
    """Return the text deployment instructions rendered from a template."""
    if jinja_environment is not None:
        return render_template(jinja_environment, "text_deployment_instructions.md")
    else:
        return "No instructions found"


def text_data_instructions() -> str:
    """Return the instructions from what kind of data is accepted by the text AutoML engine"""
    if jinja_environment is not None:
        try:
            return render_template(jinja_environment, "text_accepted_format.md")
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
        with open(model_dir / "text_deployment_instructions.md", "w") as f:
            f.write(deployment_instructions())
    except Exception as e:
        logger.debug(f"No deployment_instructions found, {e}")

    zip_base = workdir / "text_model"
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
        model_id_prefix="text_automl",
        name=f"Text AutoML Model - {dataset_id}",
        description="AutoML trained text model",
        framework="pytorch",
    )
