#!/usr/bin/env python3
"""
Kafka Consumer for Concept Drift Trigger Events

Listens to concept-drift-trigger-events. On trigger:
- Downloads the drift split of the dataset (15%) from the Data Warehouse
- Downloads the trained model (from AutoML) from the Data Warehouse
- Optionally induces synthetic drift on the drift split (for testing)
- Runs concept drift detection (ADWIN) and adaptive retraining (AutoGluon)
  - Supports both classification and regression (problem type from predictor or heuristic)
  - Drift signal: 0/1 error for classification, absolute error for regression
- If drift is detected: retrains, uploads the new model to the Data Warehouse, POSTs the concept drift report to the DW, sends concept-drift-complete with new model version
- If no drift is detected: skips upload, POSTs the concept drift report to the DW, sends concept-drift-complete with success and message "No concept drift detected"

Requires: river, autogluon.tabular, pandas, requests, aiokafka.
Optional (needed when AutoGluon loads a FastAI model): ipython (fastai/fastprogress imports IPython.display).
Dataset must have train/test/drift split (use split=drift when downloading).
"""

import os
import io
import shutil
import sys
import time
import asyncio
import json
import logging
import pathlib
import tempfile
import uuid
import zipfile
import threading
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
import importlib.util
import types

# Allow loading models trained on Linux (PosixPath) when running on Windows, and vice versa.
# AutoGluon/torch pickle can contain pathlib paths from the training OS.
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Required by AutoGluon during TabularPredictor.load; provided by setuptools (in requirements.txt / Dockerfile).
try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    raise RuntimeError(
        "pkg_resources not found (setuptools). Install it and rerun:\n"
        "  pip install 'setuptools>=65.0.0'\n"
        "When running in Docker, rebuild the image: docker compose build --no-cache concept-drift-consumer"
    ) from None

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import numpy as np
import requests
import pandas as pd
from dotenv import load_dotenv, find_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("kafka_concept_drift_consumer")

load_dotenv(find_dotenv())

# Kafka
# Default is deployment Kafka; override with env for local testing (e.g. localhost:9092).
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_DRIFT_TRIGGER_TOPIC = os.getenv("KAFKA_CONCEPT_DRIFT_TRIGGER_TOPIC", "concept-drift-trigger-events")
KAFKA_DRIFT_COMPLETE_TOPIC = os.getenv("KAFKA_CONCEPT_DRIFT_COMPLETE_TOPIC", "concept-drift-complete-events")
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONCEPT_DRIFT_CONSUMER_GROUP", "concept-drift-consumer")
WORKER_ID = os.getenv("CONCEPT_DRIFT_WORKER_ID", "1")
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("CONCEPT_DRIFT_HEARTBEAT_INTERVAL_SECONDS", "60"))

# Data Warehouse API
DW_HOST = os.getenv("DW_HOST", "localhost")
DW_PORT = os.getenv("DW_PORT", "8000")
API_BASE = os.getenv("API_BASE") or f"http://{DW_HOST}:{DW_PORT}"
AUTOML_ENGINE_HOST = os.getenv(
    "AUTOML_ENGINE_HOST",
    "localhost",
)

AUTOML_ENGINE_PORT = os.getenv(
    "AUTOML_ENGINE_PORT",
    "8001",
)

AUTOML_ENGINE_URL = (
    f"http://{AUTOML_ENGINE_HOST}:"
    f"{AUTOML_ENGINE_PORT}"
)

AUTOML_VISION_BEST_MODEL_URL = (
    f"{AUTOML_ENGINE_URL}"
    "/automl/vision/best_model/"
)

VISION_RETRAIN_TIME_BUDGET_SECONDS = int(
    os.getenv(
        "VISION_RETRAIN_TIME_BUDGET_SECONDS",
        "60",
    )
)

VISION_RETRAIN_READ_TIMEOUT_SECONDS = int(
    os.getenv(
        "VISION_RETRAIN_READ_TIMEOUT_SECONDS",
        "21600",   # 6 hours
    )
)

# Minimum improvement required before a retrained Vision
# model is allowed to replace the source model.
VISION_MIN_ACCURACY_IMPROVEMENT = float(
    os.getenv(
        "VISION_MIN_ACCURACY_IMPROVEMENT",
        "0.01",
    )
)

VISION_MIN_F1_IMPROVEMENT = float(
    os.getenv(
        "VISION_MIN_F1_IMPROVEMENT",
        "0.01",
    )
)

# Minimum number of images that must remain completely
# outside AutoML for source-vs-replacement validation.
VISION_MIN_HOLDOUT_IMAGES = int(
    os.getenv(
        "VISION_MIN_HOLDOUT_IMAGES",
        "20",
    )
)

# River ADWIN sensitivity.
# Lower values make drift detection more conservative.
VISION_ADWIN_DELTA = float(
    os.getenv(
        "VISION_ADWIN_DELTA",
        "0.002",
    )
)

SEGMENTATION_IGNORE_BACKGROUND = (
    os.getenv(
        "SEGMENTATION_IGNORE_BACKGROUND",
        "false",
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)

SEGMENTATION_MIN_MIOU_IMPROVEMENT = float(
    os.getenv(
        "SEGMENTATION_MIN_MIOU_IMPROVEMENT",
        "0.01",
    )
)

SEGMENTATION_MIN_DICE_IMPROVEMENT = float(
    os.getenv(
        "SEGMENTATION_MIN_DICE_IMPROVEMENT",
        "0.01",
    )
)

logger.info(
    "Configuration: API_BASE=%s, trigger_topic=%s, complete_topic=%s, group=%s, worker_id=%s",
    API_BASE,
    KAFKA_DRIFT_TRIGGER_TOPIC,
    KAFKA_DRIFT_COMPLETE_TOPIC,
    KAFKA_CONSUMER_GROUP,
    WORKER_ID,
)

CONSUMER_IMPLEMENTATION = "v4"

logger.info(
    "Concept Drift Consumer implementation=%s",
    CONSUMER_IMPLEMENTATION,
)

logger.info(
    "AUTOML_ENGINE_REPO=%s",
    os.getenv("AUTOML_ENGINE_REPO"),
)


def normalize_vision_task_type(task_type: str | None) -> str | None:
    if task_type is None:
        return None

    t = str(task_type).strip().lower()

    legacy_map = {
        # Legacy -> canonical
        "classification": "image_classification",
        "segmentation": "image_segmentation",
        "semantic_segmentation": "image_segmentation",

        # Canonical task names
        "image_classification": "image_classification",
        "image_segmentation": "image_segmentation",
        "object_detection": "object_detection",
        "video_classification": "video_classification",
        "keypoint_detection": "keypoint_detection",
    }

    return legacy_map.get(t, t)


def build_job_key(
    *,
    user_id: str,
    dataset_id: str,
    dataset_version: str | None,
    model_id: str,
    model_version: str | None,
) -> str:
    return "|".join([
        user_id,
        dataset_id,
        dataset_version or "v1",
        model_id,
        model_version or "v1",
    ])


def claim_task_id(
    task_id: str,
    *,
    user_id: str,
    dataset_id: str,
    dataset_version: str | None,
    model_id: str,
    model_version: str | None,
) -> bool:
    """
    Idempotency guard across consumer replicas.
    Returns True when this worker should process the task; False to skip.
    """
    if not task_id:
        return True
    url = f"{API_BASE.rstrip('/')}/jobs/concept-drift/claim"
    payload = {
        "task_id": task_id,
        "job_key": build_job_key(
            user_id=user_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            model_id=model_id,
            model_version=model_version,
        ),
        "user_id": user_id,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "model_id": model_id,
        "model_version": model_version,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 409:
            detail = ""
            try:
                detail = r.json().get("detail", "")
            except Exception:
                detail = r.text
            logger.info(
                "Worker %s skipping task_id=%s (%s)",
                WORKER_ID,
                task_id,
                detail or "already claimed",
            )
            return False
        r.raise_for_status()
        body = r.json()
        if body.get("reclaimed"):
            logger.warning(
                "Worker %s reclaimed stale lock for task_id=%s job_key=%s",
                WORKER_ID,
                task_id,
                body.get("job_key"),
            )
        return True
    except Exception as e:
        logger.warning(
            "Worker %s could not claim task_id=%s (proceeding anyway): %s",
            WORKER_ID,
            task_id,
            e,
        )
        return True


def heartbeat_task_id(task_id: str) -> None:
    if not task_id:
        return
    url = f"{API_BASE.rstrip('/')}/jobs/concept-drift/heartbeat"
    try:
        r = requests.post(url, json={"task_id": task_id}, timeout=15)
        if r.status_code == 404:
            logger.warning("Heartbeat: lock not found for task_id=%s", task_id)
            return
        r.raise_for_status()
        logger.debug("Heartbeat ok for task_id=%s", task_id)
    except Exception as e:
        logger.warning("Heartbeat failed for task_id=%s: %s", task_id, e)


def release_task_id(task_id: str) -> None:
    if not task_id:
        return
    url = f"{API_BASE.rstrip('/')}/jobs/concept-drift/release"
    try:
        r = requests.post(url, json={"task_id": task_id}, timeout=15)
        r.raise_for_status()
        logger.info("Released lock for task_id=%s", task_id)
    except Exception as e:
        logger.warning("Failed to release lock for task_id=%s: %s", task_id, e)


class _DriftJobHeartbeat:
    """Background heartbeat while sync AutoGluon work blocks the asyncio loop."""

    def __init__(self, task_id: str, interval_seconds: int = HEARTBEAT_INTERVAL_SECONDS):
        self.task_id = task_id
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            heartbeat_task_id(self.task_id)

    def start(self) -> None:
        if not self.task_id:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"drift-hb-{self.task_id[:12]}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)


def download_tabular_drift_dataset(user_id: str, dataset_id: str, version: str) -> pd.DataFrame:
    """Download the drift split of the dataset (ZIP) and load first CSV as DataFrame."""
    url = f"{API_BASE}/datasets/{user_id}/{dataset_id}/version/{version}/download?split=drift"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    zip_bytes = r.content
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        names = [n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__")]
        if not names:
            raise ValueError("No CSV file in drift split ZIP")
        with z.open(names[0]) as f:
            return pd.read_csv(io.BytesIO(f.read()))




def download_dataset_bytes(
    user_id: str,
    dataset_id: str,
    version: str | None = None,
    split: str | None = None,
) -> bytes:

    if version:
        url = (
            f"{API_BASE}/datasets/"
            f"{user_id}/{dataset_id}/"
            f"version/{version}/download"
        )
    else:
        url = (
            f"{API_BASE}/datasets/"
            f"{user_id}/{dataset_id}/download"
        )

    if split:
        if split not in (
            "train",
            "test",
            "drift",
        ):
            raise ValueError(
                f"Unsupported split: {split}"
            )

        url += f"?split={split}"

    logger.info(
        "Downloading dataset bytes: %s",
        url,
    )

    response = requests.get(
        url,
        timeout=300,
    )

    response.raise_for_status()

    return response.content


def extract_vision_dataset(
    dataset_bytes: bytes,
    extract_to: str,
) -> list[str]:

    os.makedirs(
        extract_to,
        exist_ok=True,
    )

    if not zipfile.is_zipfile(
        io.BytesIO(dataset_bytes)
    ):
        raise ValueError(
            "Vision dataset download is expected "
            "to be a ZIP archive."
        )

    with zipfile.ZipFile(
        io.BytesIO(dataset_bytes),
        "r",
    ) as zf:

        zf.extractall(extract_to)

    extracted = []

    for root, _, files in os.walk(extract_to):
        for filename in files:
            extracted.append(
                os.path.join(
                    root,
                    filename,
                )
            )

    return extracted


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
}


def find_image_manifest(
    dataset_root: str,
    filename_column: str,
    label_column: str,
) -> str:
    """
    Find the CSV manifest containing the configured
    filename and label columns.
    """

    root = Path(dataset_root)

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}"
        )

    # Prefer conventional names first.
    preferred = []

    for name in (
        "labels.csv",
        "metadata.csv",
    ):
        preferred.extend(
            root.rglob(name)
        )

    # Then annotations*.csv
    preferred.extend(
        root.rglob("annotations*.csv")
    )

    # Finally any other CSV.
    all_csvs = list(
        root.rglob("*.csv")
    )

    candidates = []

    for path in preferred + all_csvs:
        if path not in candidates:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No CSV manifest found under {root}"
        )

    for csv_path in candidates:
        try:
            header = pd.read_csv(
                csv_path,
                nrows=0,
            )
        except Exception as exc:
            logger.warning(
                "[Vision Drift] Could not inspect CSV %s: %s",
                csv_path,
                exc,
            )
            continue

        columns = set(
            header.columns
        )

        if (
            filename_column in columns
            and label_column in columns
        ):
            resolved = str(
                csv_path.resolve()
            )

            logger.info(
                "[Vision Drift] Found image manifest: %s",
                resolved,
            )

            return resolved

    raise ValueError(
        "No CSV manifest contains both required columns "
        f"{filename_column!r} and {label_column!r}. "
        f"CSV files found: {[str(p) for p in candidates]}"
    )


def find_segmentation_manifest(
    dataset_root: str,
    filename_column: str,
    mask_column: str,
) -> str:
    """
    Find a CSV manifest containing the configured
    image filename and segmentation-mask columns.
    """

    root = Path(dataset_root)

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}"
        )

    preferred = []

    for name in (
        "labels.csv",
        "metadata.csv",
    ):
        preferred.extend(
            root.rglob(name)
        )

    preferred.extend(
        root.rglob("annotations*.csv")
    )

    all_csvs = list(
        root.rglob("*.csv")
    )

    candidates = []

    for path in preferred + all_csvs:
        if path not in candidates:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No CSV manifest found under {root}"
        )

    for csv_path in candidates:

        try:
            header = pd.read_csv(
                csv_path,
                nrows=0,
            )

        except Exception as exc:
            logger.warning(
                "[Segmentation Drift] "
                "Could not inspect CSV %s: %s",
                csv_path,
                exc,
            )
            continue

        columns = set(
            header.columns
        )

        if (
            filename_column in columns
            and mask_column in columns
        ):
            resolved = str(
                csv_path.resolve()
            )

            logger.info(
                "[Segmentation Drift] "
                "Found manifest: %s",
                resolved,
            )

            return resolved

    raise ValueError(
        "No CSV manifest contains both required "
        "segmentation columns "
        f"{filename_column!r} and {mask_column!r}. "
        f"CSV files found: "
        f"{[str(p) for p in candidates]}"
    )


def resolve_dataset_file(
    *,
    dataset_root: str,
    raw_filename: str,
    preferred_subdirs: list[str],
) -> str:

    dataset_root = os.path.abspath(
        dataset_root
    )

    filename = os.path.basename(
        str(raw_filename).replace(
            "\\",
            "/",
        )
    )

    candidates = []

    for subdir in preferred_subdirs:
        candidates.append(
            os.path.join(
                dataset_root,
                subdir,
                filename,
            )
        )

    candidates.append(
        os.path.join(
            dataset_root,
            filename,
        )
    )

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(
                candidate
            )

    matches = []

    for root, _dirs, files in os.walk(
        dataset_root
    ):
        if filename in files:
            matches.append(
                os.path.join(
                    root,
                    filename,
                )
            )

    if len(matches) == 1:
        return os.path.abspath(
            matches[0]
        )

    if not matches:
        raise FileNotFoundError(
            f"Dataset file not found: "
            f"{filename!r}"
        )

    raise RuntimeError(
        f"Ambiguous dataset filename "
        f"{filename!r}. Matches: {matches}"
    )


def resolve_segmentation_paths(
    manifest_df: pd.DataFrame,
    dataset_root: str,
    filename_column: str,
    mask_column: str,
) -> pd.DataFrame:

    if filename_column not in manifest_df.columns:
        raise ValueError(
            f"Missing filename column: "
            f"{filename_column}"
        )

    if mask_column not in manifest_df.columns:
        raise ValueError(
            f"Missing mask column: "
            f"{mask_column}"
        )

    image_paths = []
    mask_paths = []

    for _, row in manifest_df.iterrows():

        image_path = resolve_dataset_file(
            dataset_root=dataset_root,
            raw_filename=row[
                filename_column
            ],
            preferred_subdirs=[
                "images",
            ],
        )

        mask_path = resolve_dataset_file(
            dataset_root=dataset_root,
            raw_filename=row[
                mask_column
            ],
            preferred_subdirs=[
                "masks",
                "mask",
                "annotations",
            ],
        )

        image_paths.append(
            image_path
        )

        mask_paths.append(
            mask_path
        )

    result = manifest_df.copy()

    result["_image_path"] = (
        image_paths
    )

    result["_mask_path"] = (
        mask_paths
    )

    logger.info(
        "[Segmentation Drift] "
        "Resolved %d image/mask pairs",
        len(result),
    )

    return result


def validate_segmentation_manifest(
    manifest_df: pd.DataFrame,
) -> None:

    from PIL import Image

    if manifest_df.empty:
        raise ValueError(
            "Segmentation manifest is empty"
        )

    for position, (_, row) in enumerate(
        manifest_df.iterrows(),
        start=1,
    ):

        image_path = str(
            row["_image_path"]
        )

        mask_path = str(
            row["_mask_path"]
        )

        with Image.open(image_path) as image:
            image_size = image.size

        with Image.open(mask_path) as mask:
            mask_size = mask.size
            mask_array = np.asarray(
                mask
            )

        if image_size != mask_size:
            raise ValueError(
                "Image/mask size mismatch: "
                f"image={image_path} {image_size}, "
                f"mask={mask_path} {mask_size}"
            )

        if mask_array.ndim != 2:
            raise ValueError(
                "Segmentation mask must be a "
                "single-channel class-ID image: "
                f"{mask_path}, "
                f"shape={mask_array.shape}"
            )

        if (
            position == 1
            or position % 100 == 0
            or position == len(
                manifest_df
            )
        ):
            logger.info(
                "[Segmentation Drift] "
                "Validated pair %d/%d",
                position,
                len(manifest_df),
            )


def resolve_image_paths(
    manifest_df,
    dataset_root: str,
    filename_column: str,
    label_column: str,
):
    """
    Resolve each manifest row to the corresponding image on disk.

    Supports layouts such as:

        dataset/
            labels.csv
            images/
                cat/
                    cat_0000.png
                dog/
                    dog_0000.png

    and flatter layouts such as:

        dataset/
            labels.csv
            images/
                image_0001.png
    """

    dataset_root = os.path.abspath(dataset_root)

    if filename_column not in manifest_df.columns:
        raise ValueError(
            f"Missing filename column: {filename_column}"
        )

    if label_column not in manifest_df.columns:
        raise ValueError(
            f"Missing label column: {label_column}"
        )

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".webp",
        ".tif",
        ".tiff",
    }

    resolved_paths = []

    for _, row in manifest_df.iterrows():

        raw_filename = str(
            row[filename_column]
        ).strip()

        label = str(
            row[label_column]
        ).strip()

        # The AutoML pipeline works with basenames.
        filename = os.path.basename(
            raw_filename.replace("\\", "/")
        )

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension not in supported_extensions:
            raise ValueError(
                f"Unsupported image extension "
                f"for {filename}: {extension}"
            )

        # Try the most likely layouts first.
        candidates = [
            # Current AutoML Vision layout
            os.path.join(
                dataset_root,
                "images",
                label,
                filename,
            ),

            # Flat images directory
            os.path.join(
                dataset_root,
                "images",
                filename,
            ),

            # Label directory directly below root
            os.path.join(
                dataset_root,
                label,
                filename,
            ),

            # Completely flat dataset
            os.path.join(
                dataset_root,
                filename,
            ),
        ]

        resolved = None

        for candidate in candidates:
            if os.path.isfile(candidate):
                resolved = candidate
                break

        # Fallback: recursively search for this basename.
        if resolved is None:
            matches = []

            for root, _dirs, files in os.walk(
                dataset_root
            ):
                if filename in files:
                    matches.append(
                        os.path.join(
                            root,
                            filename,
                        )
                    )

            if len(matches) == 1:
                resolved = matches[0]

            elif len(matches) > 1:
                # Prefer a match whose parent directory
                # corresponds to the class label.
                label_matches = [
                    path
                    for path in matches
                    if os.path.basename(
                        os.path.dirname(path)
                    ) == label
                ]

                if len(label_matches) == 1:
                    resolved = label_matches[0]

                else:
                    raise RuntimeError(
                        f"Ambiguous image filename "
                        f"{filename!r}. Matches: {matches}"
                    )

        if resolved is None:
            raise FileNotFoundError(
                f"Image does not exist for "
                f"filename={filename!r}, "
                f"label={label!r}. "
                f"Searched under: {dataset_root}"
            )

        resolved_paths.append(
            os.path.abspath(resolved)
        )

    result = manifest_df.copy()

    result["_image_path"] = resolved_paths

    logger.info(
        "[Vision Drift] Resolved %d image paths",
        len(result),
    )

    if len(result) > 0:
        logger.info(
            "[Vision Drift] First resolved image: %s",
            result.iloc[0]["_image_path"],
        )

    return result


def fetch_dataset_metadata(
    user_id: str,
    dataset_id: str,
    version: str | None = None,
) -> dict:

    if version:
        url = (
            f"{API_BASE}/datasets/"
            f"{user_id}/{dataset_id}/"
            f"version/{version}"
        )
    else:
        url = (
            f"{API_BASE}/datasets/"
            f"{user_id}/{dataset_id}"
        )

    response = requests.get(
        url,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def download_model_folder(user_id: str, model_id: str, model_version: str, extract_to: str) -> str:
    """Download model ZIP from DW and extract to extract_to. Returns path to extracted folder (root of model)."""
    url = f"{API_BASE}/ai-models/{user_id}/{model_id}/download?version={model_version}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content), "r") as z:
        z.extractall(extract_to)
    return extract_to


def find_vision_model_dir(extract_to: str) -> str:
    """
    Find the directory containing:
      - model.pt
      - feature_mapping.json

    Handles both direct extraction and a nested vision_model.zip.
    """

    def search(root_dir: str):
        for root, _dirs, files in os.walk(root_dir):
            if (
                "model.pt" in files
                and "feature_mapping.json" in files
            ):
                return root
        return None

    # First try the files already extracted.
    found = search(extract_to)

    if found:
        return found

    # AutoDW may have returned an outer ZIP containing vision_model.zip.
    for root, _dirs, files in os.walk(extract_to):
        for filename in files:

            if not filename.lower().endswith(".zip"):
                continue

            zip_path = os.path.join(
                root,
                filename,
            )

            inner_dir = os.path.join(
                root,
                f"extracted_{pathlib.Path(filename).stem}",
            )

            os.makedirs(
                inner_dir,
                exist_ok=True,
            )

            try:
                with zipfile.ZipFile(
                    zip_path,
                    "r",
                ) as zf:
                    zf.extractall(inner_dir)

            except zipfile.BadZipFile:
                continue

            found = search(inner_dir)

            if found:
                return found

    raise FileNotFoundError(
        f"No Vision model artifact found under {extract_to}. "
        "Expected model.pt and feature_mapping.json."
    )


def load_automl_image_model_class():
    """
    Load app.ml_engine.model directly without executing the heavy
    app.ml_engine.__init__.py.

    This is needed because model.pt was pickled with the class path:
        app.ml_engine.model.ImageClassificationModel
    """

    engine_repo = os.getenv("AUTOML_ENGINE_REPO")

    if not engine_repo:
        raise RuntimeError(
            "AUTOML_ENGINE_REPO is not configured."
        )

    engine_repo = os.path.abspath(engine_repo)

    app_dir = os.path.join(
        engine_repo,
        "app",
    )

    ml_engine_dir = os.path.join(
        app_dir,
        "ml_engine",
    )

    model_py = os.path.join(
        ml_engine_dir,
        "model.py",
    )

    if not os.path.isfile(model_py):
        raise FileNotFoundError(
            f"AutoML model.py not found: {model_py}"
        )

    logger.info(
        "[Vision] Loading AutoML model module directly from %s",
        model_py,
    )

    # --------------------------------------------------
    # Create/extend the 'app' package without executing
    # alfie_automl_engine/app/__init__.py.
    # --------------------------------------------------

    app_package = sys.modules.get("app")

    if app_package is None:
        app_package = types.ModuleType("app")
        app_package.__package__ = "app"
        app_package.__path__ = [app_dir]
        sys.modules["app"] = app_package

    else:
        package_paths = list(
            getattr(
                app_package,
                "__path__",
                [],
            )
        )

        if app_dir not in package_paths:
            package_paths.insert(
                0,
                app_dir,
            )

            app_package.__path__ = package_paths

    # --------------------------------------------------
    # Create app.ml_engine as a lightweight package
    # WITHOUT executing ml_engine/__init__.py.
    # --------------------------------------------------

    ml_package = sys.modules.get(
        "app.ml_engine"
    )

    if ml_package is None:
        ml_package = types.ModuleType(
            "app.ml_engine"
        )

        ml_package.__package__ = (
            "app.ml_engine"
        )

        ml_package.__path__ = [
            ml_engine_dir
        ]

        sys.modules[
            "app.ml_engine"
        ] = ml_package

        setattr(
            app_package,
            "ml_engine",
            ml_package,
        )

    # --------------------------------------------------
    # Load model.py under its REAL pickle module name.
    # --------------------------------------------------

    module_name = "app.ml_engine.model"

    existing = sys.modules.get(
        module_name
    )

    if existing is not None:
        logger.info(
            "[Vision] AutoML model module already loaded"
        )

        return (
            existing.ImageClassificationModel
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        model_py,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not create import spec for {model_py}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    logger.info(
        "[Vision] ImageClassificationModel loaded directly"
    )

    return module.ImageClassificationModel


def load_automl_segmentation_model_class():
    """
    Make ImageSegmentationModel available under its real
    pickle module path:

        app.ml_engine.model.ImageSegmentationModel

    The saved artifact may have a Lightning _FabricModule
    at the top level, but that wrapper contains the AutoML
    segmentation model, so the class must still be resolvable
    during torch.load().
    """

    # Reuse the existing direct model.py loader.
    load_automl_image_model_class()

    module = sys.modules.get(
        "app.ml_engine.model"
    )

    if module is None:
        raise RuntimeError(
            "AutoML model module was not loaded"
        )

    model_class = getattr(
        module,
        "ImageSegmentationModel",
        None,
    )

    if model_class is None:
        raise RuntimeError(
            "app.ml_engine.model does not expose "
            "ImageSegmentationModel"
        )

    logger.info(
        "[Segmentation] ImageSegmentationModel "
        "class ready: %s",
        model_class,
    )

    return model_class


def ensure_automl_engine_importable() -> None:
    """
    Add the local alfie_automl_engine repository to sys.path.

    This is required because the Vision AutoML artifact is saved as a
    full PyTorch object whose class lives at:

        app.ml_engine.model.ImageClassificationModel

    Therefore that class must be importable before torch.load().
    """

    engine_repo = os.getenv("AUTOML_ENGINE_REPO")

    if not engine_repo:
        raise RuntimeError(
            "AUTOML_ENGINE_REPO is not configured. "
            "Add it to the Data Warehouse .env file."
        )

    engine_repo = os.path.abspath(engine_repo)

    if not os.path.isdir(engine_repo):
        raise RuntimeError(
            f"AUTOML_ENGINE_REPO does not exist: {engine_repo}"
        )

    # Put it first because both repositories contain a top-level
    # Python package named 'app'.
    if engine_repo in sys.path:
        sys.path.remove(engine_repo)

    sys.path.insert(0, engine_repo)

    logger.info(
        "[Vision] AutoML Engine repository added to Python path: %s",
        engine_repo,
    )


class VisionModelAdapter:
    def __init__(
        self,
        *,
        model,
        processor,
        id2label: dict[int, str],
        label2id: dict[str, int],
        device,
    ):
        self.model = model
        self.processor = processor
        self.id2label = id2label
        self.label2id = label2id
        self.device = device

    @classmethod
    def load(
        cls,
        model_dir: str,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoImageProcessor

        # Make the AutoML Engine package visible.
        ensure_automl_engine_importable()

        # Required for torch.load() to resolve the class stored
        # inside the full PyTorch model pickle.
        logger.info(
            "[Vision] Loading ImageClassificationModel class"
        )

        ImageClassificationModel = (
            load_automl_image_model_class()
        )

        logger.info(
            "[Vision] ImageClassificationModel class ready: %s",
            ImageClassificationModel,
        )

        artifact_dir = find_vision_model_dir(
            model_dir
        )

        model_path = os.path.join(
            artifact_dir,
            "model.pt",
        )

        mapping_path = os.path.join(
            artifact_dir,
            "feature_mapping.json",
        )

        logger.info(
            "[Vision] Artifact directory: %s",
            artifact_dir,
        )

        logger.info(
            "[Vision] Model path: %s",
            model_path,
        )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Vision model file not found: {model_path}"
            )

        if not os.path.isfile(mapping_path):
            raise FileNotFoundError(
                f"Feature mapping not found: {mapping_path}"
            )

        with open(
            mapping_path,
            "r",
            encoding="utf-8",
        ) as f:
            feature_mapping = json.load(f)

        label_map = feature_mapping.get(
            "label_map",
            {},
        )

        raw_id2label = label_map.get(
            "id2label",
            {},
        )

        raw_label2id = label_map.get(
            "label2id",
            {},
        )

        id2label = {
            int(k): str(v)
            for k, v in raw_id2label.items()
        }

        label2id = {
            str(k): int(v)
            for k, v in raw_label2id.items()
        }

        if not id2label:
            raise ValueError(
                "feature_mapping.json does not contain "
                "label_map.id2label"
            )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        logger.info(
            "[Vision] Loading model on device=%s",
            device,
        )

        model = torch.load(
            model_path,
            weights_only=False,
            map_location=device,
        )
        logger.info("[Vision] torch.load completed")

        model.to(device)
        model.eval()

        if not hasattr(model, "model"):
            raise ValueError(
                "Loaded Vision model does not expose "
                "the expected underlying Hugging Face model."
            )

        hf_model_id = (
            model.model.config._name_or_path
        )

        logger.info(
            "[Vision] Loading image processor for %s",
            hf_model_id,
        )

        processor = AutoImageProcessor.from_pretrained(
            hf_model_id,
            use_fast=False,
        )

        logger.info(
            "[Vision] Image processor loaded successfully"
        )

        return cls(
            model=model,
            processor=processor,
            id2label=id2label,
            label2id=label2id,
            device=device,
        )


    def predict_proba(
        self,
        image_paths: list[str],
    ) -> list[dict[str, float]]:

        import torch
        from PIL import Image

        images = []

        try:
            for image_path in image_paths:

                image = Image.open(
                    image_path
                ).convert("RGB")

                images.append(image)

            inputs = self.processor(
                images=images,
                return_tensors="pt",
            )

            pixel_values = inputs[
                "pixel_values"
            ].to(self.device)

            with torch.no_grad():

                logits = self.model(
                    pixel_values=pixel_values
                )

                probabilities = torch.softmax(
                    logits,
                    dim=-1,
                )

            probabilities = (
                probabilities
                .detach()
                .cpu()
                .numpy()
            )

            results = []

            for row in probabilities:

                result = {
                    self.id2label[index]: float(
                        probability
                    )
                    for index, probability
                    in enumerate(row)
                }

                results.append(result)

            return results

        finally:
            for image in images:
                try:
                    image.close()
                except Exception:
                    pass


    def predict(
        self,
        image_paths: list[str],
    ) -> list[str]:

        probabilities = self.predict_proba(
            image_paths
        )

        return [
            max(
                row,
                key=row.get,
            )
            for row in probabilities
        ]


class SegmentationModelAdapter:
    def __init__(
        self,
        *,
        model,
        processor,
        id2label: dict[int, str],
        label2id: dict[str, int],
        device,
    ):
        self.model = model
        self.processor = processor
        self.id2label = id2label
        self.label2id = label2id
        self.device = device

    @classmethod
    def load(
        cls,
        model_dir: str,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoImageProcessor

        ensure_automl_engine_importable()

        # Required so torch.load() can resolve the
        # ImageSegmentationModel contained by the
        # saved Lightning/Fabric wrapper.
        load_automl_segmentation_model_class()

        artifact_dir = find_vision_model_dir(
            model_dir
        )

        model_path = os.path.join(
            artifact_dir,
            "model.pt",
        )

        mapping_path = os.path.join(
            artifact_dir,
            "feature_mapping.json",
        )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Segmentation model file not found: "
                f"{model_path}"
            )

        if not os.path.isfile(mapping_path):
            raise FileNotFoundError(
                f"Segmentation feature mapping not found: "
                f"{mapping_path}"
            )

        with open(
            mapping_path,
            "r",
            encoding="utf-8",
        ) as f:
            feature_mapping = json.load(f)

        task_type = feature_mapping.get(
            "task_type"
        )

        if task_type != "image_segmentation":
            raise ValueError(
                "Expected image_segmentation artifact, "
                f"found task_type={task_type!r}"
            )

        label_map = feature_mapping.get(
            "label_map",
            {},
        )

        raw_id2label = label_map.get(
            "id2label",
            {},
        )

        raw_label2id = label_map.get(
            "label2id",
            {},
        )

        id2label = {
            int(k): str(v)
            for k, v in raw_id2label.items()
        }

        label2id = {
            str(k): int(v)
            for k, v in raw_label2id.items()
        }

        if not id2label:
            raise ValueError(
                "Segmentation feature_mapping.json "
                "does not contain label_map.id2label"
            )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        logger.info(
            "[Segmentation] Loading model on device=%s",
            device,
        )

        model = torch.load(
            model_path,
            weights_only=False,
            map_location=device,
        )

        model.to(device)
        model.eval()

        if not hasattr(model, "model"):
            raise ValueError(
                "Loaded segmentation model does not "
                "expose an underlying Hugging Face model"
            )

        config = getattr(
            model.model,
            "config",
            None,
        )

        if config is None:
            raise ValueError(
                "Loaded segmentation model does not "
                "expose model.model.config"
            )

        hf_model_id = getattr(
            config,
            "_name_or_path",
            None,
        )

        if not hf_model_id:
            raise ValueError(
                "Could not determine Hugging Face "
                "segmentation backbone"
            )

        logger.info(
            "[Segmentation] Loading image processor for %s",
            hf_model_id,
        )

        processor = AutoImageProcessor.from_pretrained(
            hf_model_id,
            use_fast=False,
        )

        return cls(
            model=model,
            processor=processor,
            id2label=id2label,
            label2id=label2id,
            device=device,
        )

    def predict_mask(
        self,
        image_path: str,
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        original_width, original_height = (
            image.size
        )

        inputs = self.processor(
            images=[image],
            return_tensors="pt",
        )

        pixel_values = inputs[
            "pixel_values"
        ].to(
            self.device
        )

        with torch.no_grad():
            logits = self.model(
                pixel_values=pixel_values,
            )

        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "Segmentation model did not return "
                "a logits tensor"
            )

        if logits.ndim != 4:
            raise ValueError(
                "Expected segmentation logits with "
                f"4 dimensions, got shape={tuple(logits.shape)}"
            )

        resized_logits = F.interpolate(
            logits,
            size=(
                original_height,
                original_width,
            ),
            mode="bilinear",
            align_corners=False,
        )

        predicted_mask = (
            resized_logits
            .argmax(dim=1)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        return predicted_mask


def calculate_image_classification_error(
    adapter: VisionModelAdapter,
    image_path: str,
    true_label: str,
) -> tuple[float, str, dict[str, float]]:

    probabilities = adapter.predict_proba(
        [image_path]
    )[0]

    predicted_label = max(
        probabilities,
        key=probabilities.get,
    )

    true_probability = probabilities.get(
        str(true_label)
    )

    if true_probability is None:
        logger.warning(
            "[Vision] True label %r is not present "
            "in model classes %s",
            true_label,
            list(probabilities.keys()),
        )

        # Maximum error if the true class cannot
        # be mapped to the model output.
        error = 1.0

    else:
        error = 1.0 - float(
            true_probability
        )

    return (
        error,
        predicted_label,
        probabilities,
    )


def calculate_vision_metrics(
    y_true,
    y_pred,
):
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    if not y_true or not y_pred:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def load_segmentation_mask(
    mask_path: str,
) -> np.ndarray:
    from PIL import Image

    with Image.open(mask_path) as image:
        mask = np.asarray(
            image,
            dtype=np.int64,
        )

    if mask.ndim == 3:
        if mask.shape[2] != 1:
            raise ValueError(
                "Segmentation mask must be "
                "single-channel: "
                f"path={mask_path} "
                f"shape={mask.shape}"
            )

        mask = mask[:, :, 0]

    if mask.ndim != 2:
        raise ValueError(
            "Segmentation mask must be 2D: "
            f"path={mask_path} "
            f"shape={mask.shape}"
        )

    return mask


def calculate_segmentation_metrics(
    true_mask: np.ndarray,
    predicted_mask: np.ndarray,
    *,
    ignore_background: bool = (
        SEGMENTATION_IGNORE_BACKGROUND
    ),
    ignore_index: int = 255,
) -> dict:
    true_mask = np.asarray(
        true_mask,
        dtype=np.int64,
    )

    predicted_mask = np.asarray(
        predicted_mask,
        dtype=np.int64,
    )

    if true_mask.ndim != 2:
        raise ValueError(
            "true_mask must be 2D"
        )

    if predicted_mask.ndim != 2:
        raise ValueError(
            "predicted_mask must be 2D"
        )

    if true_mask.shape != predicted_mask.shape:
        raise ValueError(
            "Segmentation mask shape mismatch: "
            f"true={true_mask.shape} "
            f"predicted={predicted_mask.shape}"
        )

    # Void pixels are removed before every metric,
    # including pixel accuracy.
    valid = (
        true_mask != ignore_index
    )

    true_valid = true_mask[valid]
    predicted_valid = predicted_mask[valid]

    if true_valid.size == 0:
        raise ValueError(
            "No valid segmentation pixels remain "
            "after ignore_index filtering"
        )

    class_ids = sorted(
        set(true_valid.tolist())
        | set(predicted_valid.tolist())
    )

    if ignore_background:
        class_ids = [
            class_id
            for class_id in class_ids
            if class_id != 0
        ]

    if not class_ids:
        raise ValueError(
            "No classes remain for segmentation "
            "metric calculation"
        )

    per_class_iou = {}
    per_class_dice = {}

    for class_id in class_ids:
        true_class = (
            true_valid == class_id
        )

        predicted_class = (
            predicted_valid == class_id
        )

        intersection = int(
            np.logical_and(
                true_class,
                predicted_class,
            ).sum()
        )

        union = int(
            np.logical_or(
                true_class,
                predicted_class,
            ).sum()
        )

        true_count = int(
            true_class.sum()
        )

        predicted_count = int(
            predicted_class.sum()
        )

        if union == 0:
            continue

        iou = (
            intersection / union
        )

        denominator = (
            true_count
            + predicted_count
        )

        dice = (
            (2.0 * intersection)
            / denominator
            if denominator > 0
            else 1.0
        )

        per_class_iou[
            int(class_id)
        ] = float(iou)

        per_class_dice[
            int(class_id)
        ] = float(dice)

    if not per_class_iou:
        raise ValueError(
            "No evaluable segmentation classes"
        )

    mean_iou = float(
        np.mean(
            list(
                per_class_iou.values()
            )
        )
    )

    mean_dice = float(
        np.mean(
            list(
                per_class_dice.values()
            )
        )
    )

    pixel_accuracy = float(
        np.mean(
            true_valid
            == predicted_valid
        )
    )

    return {
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "pixel_accuracy": pixel_accuracy,
        "per_class_iou": per_class_iou,
        "per_class_dice": per_class_dice,
    }


def calculate_segmentation_error(
    true_mask: np.ndarray,
    predicted_mask: np.ndarray,
    *,
    ignore_background: bool = (
        SEGMENTATION_IGNORE_BACKGROUND
    ),
    ignore_index: int = 255,
) -> float:
    metrics = calculate_segmentation_metrics(
        true_mask,
        predicted_mask,
        ignore_background=ignore_background,
        ignore_index=ignore_index,
    )

    return (
        1.0
        - metrics["mean_iou"]
    )


def monitor_image_drift(
    manifest_df,
    adapter: VisionModelAdapter,
    filename_column: str,
    label_column: str,
    window_size: int,
    adwin_delta: float = VISION_ADWIN_DELTA,
):
    from river.drift import ADWIN

    detector = ADWIN(delta=adwin_delta)
    logger.info(
        "[Vision Drift] ADWIN configured: "
        "delta=%s requested_window_size=%d",
        adwin_delta,
        window_size,
    )

    errors = []
    y_true = []
    y_pred = []

    drift_detected = False
    drift_index = None

    total = len(manifest_df)

    for position, (_, row) in enumerate(
            manifest_df.iterrows(),
            start=1,
    ):

        image_path = str(
            row["_image_path"]
        )

        true_label = str(
            row[label_column]
        )

        (
            error,
            predicted_label,
            probabilities,
        ) = calculate_image_classification_error(
            adapter,
            image_path,
            true_label,
        )

        errors.append(error)
        y_true.append(true_label)
        y_pred.append(predicted_label)

        detector.update(error)

        if (
                position == 1
                or position % 25 == 0
                or position == total
        ):
            logger.info(
                "[Vision Drift] Processed %d/%d images; "
                "true=%s predicted=%s error=%.4f",
                position,
                total,
                true_label,
                predicted_label,
                error,
            )

        if detector.drift_detected:
            drift_detected = True
            drift_index = position - 1

            logger.info(
                "[Vision Drift] Drift detected at image %d",
                drift_index,
            )

            break

    return {
        "drift_detected": drift_detected,
        "drift_index": drift_index,
        "errors": errors,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def evaluate_segmentation_model_on_manifest(
    *,
    manifest_df: pd.DataFrame,
    adapter: SegmentationModelAdapter,
    ignore_background: bool = (
        SEGMENTATION_IGNORE_BACKGROUND
    ),
    ignore_index: int = 255,
) -> dict:

    if manifest_df.empty:
        raise ValueError(
            "Segmentation evaluation manifest is empty"
        )

    mean_ious = []
    mean_dices = []
    pixel_accuracies = []

    total = len(
        manifest_df
    )

    for position, (_, row) in enumerate(
        manifest_df.iterrows(),
        start=1,
    ):
        image_path = str(
            row["_image_path"]
        )

        mask_path = str(
            row["_mask_path"]
        )

        true_mask = load_segmentation_mask(
            mask_path
        )

        predicted_mask = (
            adapter.predict_mask(
                image_path
            )
        )

        sample_metrics = (
            calculate_segmentation_metrics(
                true_mask,
                predicted_mask,
                ignore_background=ignore_background,
                ignore_index=ignore_index,
            )
        )

        mean_ious.append(
            float(
                sample_metrics["mean_iou"]
            )
        )

        mean_dices.append(
            float(
                sample_metrics["mean_dice"]
            )
        )

        pixel_accuracies.append(
            float(
                sample_metrics[
                    "pixel_accuracy"
                ]
            )
        )

        if (
            position == 1
            or position % 25 == 0
            or position == total
        ):
            logger.info(
                "[Segmentation Validation] "
                "Processed %d/%d; "
                "mIoU=%.4f Dice=%.4f "
                "pixel_accuracy=%.4f",
                position,
                total,
                sample_metrics["mean_iou"],
                sample_metrics["mean_dice"],
                sample_metrics[
                    "pixel_accuracy"
                ],
            )

    return {
        "mean_iou": float(
            np.mean(mean_ious)
        ),
        "mean_dice": float(
            np.mean(mean_dices)
        ),
        "pixel_accuracy": float(
            np.mean(pixel_accuracies)
        ),
        "samples": total,
    }


def monitor_segmentation_drift(
    manifest_df,
    adapter: SegmentationModelAdapter,
    filename_column: str,
    mask_column: str,
    window_size: int,
    adwin_delta: float = VISION_ADWIN_DELTA,
    ignore_background: bool = (
        SEGMENTATION_IGNORE_BACKGROUND
    ),
    ignore_index: int = 255,
):
    from river.drift import ADWIN

    detector = ADWIN(
        delta=adwin_delta
    )

    logger.info(
        "[Segmentation Drift] ADWIN configured: "
        "delta=%s requested_window_size=%d",
        adwin_delta,
        window_size,
    )

    if "_image_path" not in manifest_df.columns:
        raise ValueError(
            "Segmentation manifest does not contain "
            "_image_path"
        )

    if "_mask_path" not in manifest_df.columns:
        raise ValueError(
            "Segmentation manifest does not contain "
            "_mask_path"
        )

    if filename_column not in manifest_df.columns:
        raise ValueError(
            f"Missing filename column: "
            f"{filename_column}"
        )

    if mask_column not in manifest_df.columns:
        raise ValueError(
            f"Missing mask column: "
            f"{mask_column}"
        )

    errors = []

    mean_ious = []
    mean_dices = []
    pixel_accuracies = []

    drift_detected = False
    drift_index = None

    total = len(
        manifest_df
    )

    for position, (_, row) in enumerate(
        manifest_df.iterrows(),
        start=1,
    ):
        image_path = str(
            row["_image_path"]
        )

        mask_path = str(
            row["_mask_path"]
        )

        true_mask = load_segmentation_mask(
            mask_path
        )

        predicted_mask = (
            adapter.predict_mask(
                image_path
            )
        )

        metrics = (
            calculate_segmentation_metrics(
                true_mask,
                predicted_mask,
                ignore_background=ignore_background,
                ignore_index=ignore_index,
            )
        )

        # One scalar per image for ADWIN.
        #
        # Higher mIoU = better segmentation,
        # therefore error is:
        #
        #     1 - mIoU
        #
        error = (
            1.0
            - metrics["mean_iou"]
        )

        errors.append(
            float(error)
        )

        mean_ious.append(
            float(
                metrics["mean_iou"]
            )
        )

        mean_dices.append(
            float(
                metrics["mean_dice"]
            )
        )

        pixel_accuracies.append(
            float(
                metrics["pixel_accuracy"]
            )
        )

        detector.update(
            error
        )

        if (
            position == 1
            or position % 25 == 0
            or position == total
        ):
            logger.info(
                "[Segmentation Drift] "
                "Processed %d/%d images; "
                "mIoU=%.4f Dice=%.4f "
                "pixel_accuracy=%.4f "
                "error=%.4f",
                position,
                total,
                metrics["mean_iou"],
                metrics["mean_dice"],
                metrics[
                    "pixel_accuracy"
                ],
                error,
            )

        if detector.drift_detected:
            drift_detected = True
            drift_index = (
                position - 1
            )

            logger.info(
                "[Segmentation Drift] "
                "Drift detected at image %d",
                drift_index,
            )

            break

    return {
        "drift_detected": drift_detected,
        "drift_index": drift_index,
        "errors": errors,
        "mean_ious": mean_ious,
        "mean_dices": mean_dices,
        "pixel_accuracies": (
            pixel_accuracies
        ),
        "samples_processed": len(
            errors
        ),
    }


def apply_synthetic_image_drift(
    manifest_df,
    start_fraction: float = 0.5,
) -> None:
    """
    Modify the latter portion of the temporary Vision dataset
    in-place to simulate image distribution drift.

    Only temporary extracted images are changed.
    """

    from PIL import (
        Image,
        ImageEnhance,
        ImageFilter,
    )

    if manifest_df.empty:
        return

    start_index = int(
        len(manifest_df) * start_fraction
    )

    logger.info(
        "[Vision Drift] Applying synthetic image drift "
        "from index %d/%d",
        start_index,
        len(manifest_df),
    )

    drift_rows = manifest_df.iloc[
        start_index:
    ]

    for _, row in drift_rows.iterrows():

        image_path = str(
            row["_image_path"]
        )

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            # Darken image.
            image = ImageEnhance.Brightness(
                image
            ).enhance(0.30)

            # Reduce contrast.
            image = ImageEnhance.Contrast(
                image
            ).enhance(0.40)

            # Add blur.
            image = image.filter(
                ImageFilter.GaussianBlur(
                    radius=2.0
                )
            )

            image.save(
                image_path
            )

    logger.info(
        "[Vision Drift] Synthetic drift applied to %d images",
        len(drift_rows),
    )


def find_autogluon_model_dir(extract_to: str) -> str:
    """
    Find the directory that contains predictor.pkl (AutoGluon save format).
    DW zip may have: predictor.pkl at root, or a subfolder (e.g. AgAutogluon/predictor.pkl),
    or a single inner model.zip that must be extracted first.
    Returns the path to the directory containing predictor.pkl for TabularPredictor.load().
    """
    # First pass: look for predictor.pkl anywhere under extract_to
    for root, _dirs, files in os.walk(extract_to):
        if "predictor.pkl" in files:
            return root
    # Second pass: if there is a single .zip (nested model), extract and search again
    for root, _dirs, files in os.walk(extract_to):
        zips = [f for f in files if f.lower().endswith(".zip")]
        for z in zips:
            zip_path = os.path.join(root, z)
            inner_dir = os.path.join(root, "inner_model")
            os.makedirs(inner_dir, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(inner_dir)
                for r2, _d2, f2 in os.walk(inner_dir):
                    if "predictor.pkl" in f2:
                        return r2
            except (zipfile.BadZipFile, OSError):
                continue
    raise FileNotFoundError(
        f"No predictor.pkl found under {extract_to}. "
        "Expected AutoGluon TabularPredictor save (predictor.pkl) in the model zip from the Data Warehouse."
    )


def predict_autogluon(predictor, row, feature_cols):
    """Single-row prediction with AutoGluon TabularPredictor."""
    X = pd.DataFrame([row[feature_cols]])
    return predictor.predict(X).iloc[0]


def monitor_drift_and_retrain(
    df,
    predictor,
    feature_cols,
    target_col,
    retrain_path: str,
    window_size: int = 150,
    delta: float = 0.1,
    clock: int = 1,
    batch_predict_size: int = 512,
    retrain_presets: str = "medium_quality_faster_train",
    min_samples_before_detection: int = 50,
    cooldown_samples: int = 50,
    verbose: bool = True,
):
    """
    Monitor drift with ADWIN and retrain AutoGluon on a sliding window.

    Classification:
        ADWIN monitors 1 - P(true_class)

    Regression:
        ADWIN monitors absolute error
    """
    from river.drift import ADWIN
    from autogluon.tabular import TabularPredictor
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    def infer_problem_type(df_: pd.DataFrame, target: str, pred) -> str:
        pt = getattr(pred, "problem_type", None)
        if isinstance(pt, str) and pt:
            return pt.lower()

        y = df_[target]
        if pd.api.types.is_numeric_dtype(y):
            nunique = y.nunique(dropna=True)
            n = len(y)
            if nunique <= 20 or (n > 0 and (nunique / n) < 0.05):
                return "multiclass"
            return "regression"
        return "multiclass"

    def normalize_label(v):
        if pd.isna(v):
            return "NA"
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            v = int(v)
        return str(v).strip()

    def coerce_classification_labels(y_true_ser, y_pred_ser):
        y_true_labels = y_true_ser.map(normalize_label)
        y_pred_labels = y_pred_ser.map(normalize_label)
        return y_true_labels, y_pred_labels

    if df is None or df.empty:
        raise ValueError("Input df is empty.")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in df: {missing_features}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in df.")

    problem_type = infer_problem_type(df, target_col, predictor)
    is_regression = (problem_type == "regression") or (
        "regress" in str(problem_type).lower()
    )

    n_rows = len(df)
    logger.info(
        "[Drift] Starting monitoring: rows=%d, target=%s, window_size=%d, retrain_path=%s",
        n_rows,
        target_col,
        window_size,
        retrain_path,
    )

    adwin = ADWIN(delta=delta, clock=clock)
    window_rows = deque(maxlen=window_size)
    events = []
    error_stream = []
    y_true_all, y_pred_all = [], []
    window_y_true = deque(maxlen=window_size)
    window_y_pred = deque(maxlen=window_size)
    total_retrain_time = 0.0
    start_time = __import__("time").time()
    log_interval = max(1, n_rows // 10)
    samples_seen = 0
    cooldown_until = -1

    for batch_start in range(0, n_rows, batch_predict_size):
        batch_end = min(batch_start + batch_predict_size, n_rows)
        batch_df = df.iloc[batch_start:batch_end].copy()
        X_batch = batch_df[feature_cols]

        batch_preds = predictor.predict(X_batch)

        if not is_regression:
            batch_proba = predictor.predict_proba(X_batch).copy()
            batch_proba.columns = [normalize_label(c) for c in batch_proba.columns]
        else:
            batch_proba = None

        for offset, (_, row) in enumerate(batch_df.iterrows()):
            i = batch_start + offset
            y_true = row[target_col]
            y_pred = batch_preds.iloc[offset]

            if is_regression:
                y_true_num = pd.to_numeric(
                    pd.Series([y_true]), errors="coerce"
                ).iloc[0]
                y_pred_num = pd.to_numeric(
                    pd.Series([y_pred]), errors="coerce"
                ).iloc[0]

                if pd.isna(y_true_num) or pd.isna(y_pred_num):
                    error = 0.0
                else:
                    error = float(abs(y_pred_num - y_true_num))
            else:
                y_true_key = normalize_label(y_true)
                y_pred_key = normalize_label(y_pred)

                if y_true_key in batch_proba.columns:
                    true_class_prob = float(batch_proba.iloc[offset][y_true_key])
                    error = 1.0 - true_class_prob
                else:
                    error = 0.0 if y_pred_key == y_true_key else 1.0

            error_stream.append(error)

            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
            window_y_true.append(y_true)
            window_y_pred.append(y_pred)
            window_rows.append(row)

            samples_seen += 1
            elapsed = __import__("time").time() - start_time

            if (i + 1) % log_interval == 0 or i == 0:
                correct_so_far = sum(
                    1 for a, b in zip(y_true_all, y_pred_all) if a == b
                )
                logger.info(
                    "[Drift] Progress row %d/%d, cumulative_accuracy=%.4f, window_fill=%d",
                    i + 1,
                    n_rows,
                    correct_so_far / len(y_true_all) if y_true_all else 0,
                    len(window_rows),
                )

            adwin.update(error)

            detection_allowed = (
                samples_seen >= min_samples_before_detection
                and i > cooldown_until
                and len(window_rows) >= min_samples_before_detection
            )

            if detection_allowed and adwin.drift_detected:
                if verbose:
                    logger.info(
                        "[Drift] Concept drift DETECTED at index %d. Retraining on last %d samples at path=%s",
                        i,
                        len(window_rows),
                        retrain_path,
                    )

                window_df = pd.DataFrame(list(window_rows))
                retrain_start = __import__("time").time()

                unique_path = os.path.join(
                    retrain_path, f"retrain_{i}_{uuid.uuid4().hex[:8]}"
                )

                fit_kwargs = {}
                if is_regression:
                    fit_kwargs["problem_type"] = "regression"
                elif problem_type == "binary":
                    fit_kwargs["problem_type"] = "binary"
                else:
                    fit_kwargs["problem_type"] = "multiclass"

                predictor = TabularPredictor(
                    label=target_col,
                    path=unique_path,
                    **fit_kwargs,
                ).fit(train_data=window_df, presets=retrain_presets)

                retrain_time = __import__("time").time() - retrain_start
                total_retrain_time += retrain_time

                events.append(
                    {
                        "index": i,
                        "elapsed_seconds": elapsed,
                        "window_size": len(window_rows),
                        "retrain_time_seconds": retrain_time,
                        "mean_error_so_far": float(np.mean(error_stream))
                        if error_stream
                        else np.nan,
                    }
                )

                if verbose:
                    logger.info("[Drift] Retraining took %.4f seconds.", retrain_time)

                adwin = ADWIN(delta=delta, clock=clock)
                cooldown_until = i + cooldown_samples

                try:
                    contents = []
                    for _root, _dirs, files in os.walk(retrain_path):
                        contents.extend(
                            os.path.join(os.path.relpath(_root, retrain_path), f)
                            for f in files
                        )
                    logger.info(
                        "[Drift] After retrain: path=%s has %d files: %s",
                        retrain_path,
                        len(contents),
                        contents[:20],
                    )
                except Exception as e:
                    logger.warning(
                        "[Drift] Could not list retrain_path after fit: %s", e
                    )

    drift_log = pd.DataFrame(events)
    total_runtime = __import__("time").time() - start_time
    total_drifts = len(events)
    n_drifts = len(events)
    total_count = len(y_true_all)
    correct_count = sum(1 for a, b in zip(y_true_all, y_pred_all) if a == b)

    try:
        retrain_files = []
        for root, _d, files in os.walk(retrain_path):
            retrain_files.extend(os.path.join(root, f) for f in files)
        logger.info(
            "[Drift] Monitoring complete. Drifts detected=%d, total retrain_time=%.2fs, retrain_path has %d files",
            n_drifts,
            total_retrain_time,
            len(retrain_files),
        )
        if n_drifts == 0:
            logger.info(
                "[Drift] No drift was detected during this run; no retrain occurred. "
                "Consumer will send success without uploading a new model."
            )
    except Exception as e:
        logger.warning("[Drift] Could not list retrain_path at end: %s", e)

    if not y_true_all:
        if is_regression:
            metrics = {
                "problem_type": "regression",
                "mae": float("nan"),
                "mse": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "final_window_mae": float("nan"),
                "total_drifts": total_drifts,
                "total_retrain_time_seconds": total_retrain_time,
                "total_runtime_seconds": total_runtime,
            }
        else:
            metrics = {
                "problem_type": "classification",
                "cumulative_accuracy": float("nan"),
                "final_accuracy": float("nan"),
                "final_window_accuracy": float("nan"),
                "f1_score": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "total_drifts": total_drifts,
                "total_retrain_time_seconds": total_retrain_time,
                "total_runtime_seconds": total_runtime,
            }
        return predictor, drift_log, metrics

    if is_regression:
        y_true_arr = np.asarray(
            pd.to_numeric(pd.Series(y_true_all), errors="coerce")
        )
        y_pred_arr = np.asarray(
            pd.to_numeric(pd.Series(y_pred_all), errors="coerce")
        )

        valid_mask = ~np.isnan(y_true_arr) & ~np.isnan(y_pred_arr)
        y_true_arr = y_true_arr[valid_mask]
        y_pred_arr = y_pred_arr[valid_mask]

        if len(y_true_arr) == 0:
            mae = mse = rmse = r2 = float("nan")
        else:
            mae = mean_absolute_error(y_true_arr, y_pred_arr)
            mse = mean_squared_error(y_true_arr, y_pred_arr)
            rmse = float(np.sqrt(mse))
            r2 = r2_score(y_true_arr, y_pred_arr)

        if window_y_true:
            wy_true = np.asarray(
                pd.to_numeric(pd.Series(list(window_y_true)), errors="coerce")
            )
            wy_pred = np.asarray(
                pd.to_numeric(pd.Series(list(window_y_pred)), errors="coerce")
            )
            valid_mask = ~np.isnan(wy_true) & ~np.isnan(wy_pred)
            wy_true = wy_true[valid_mask]
            wy_pred = wy_pred[valid_mask]
            final_window_mae = (
                mean_absolute_error(wy_true, wy_pred) if len(wy_true) else float("nan")
            )
        else:
            final_window_mae = float("nan")

        metrics = {
            "problem_type": "regression",
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "final_window_mae": final_window_mae,
            "total_drifts": total_drifts,
            "total_retrain_time_seconds": total_retrain_time,
            "total_runtime_seconds": total_runtime,
        }

    else:
        y_true_ser = pd.Series(y_true_all)
        y_pred_ser = pd.Series(y_pred_all)

        y_true_labels, y_pred_labels = coerce_classification_labels(
            y_true_ser, y_pred_ser
        )

        final_accuracy = accuracy_score(y_true_labels, y_pred_labels)
        f1 = f1_score(y_true_labels, y_pred_labels, average="weighted")
        precision = precision_score(
            y_true_labels, y_pred_labels, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_true_labels, y_pred_labels, average="weighted", zero_division=0
        )
        cumulative_accuracy = float((y_true_labels == y_pred_labels).mean())

        if window_y_true:
            wy_true = pd.Series(list(window_y_true))
            wy_pred = pd.Series(list(window_y_pred))
            wy_true_labels, wy_pred_labels = coerce_classification_labels(
                wy_true, wy_pred
            )
            final_window_accuracy = accuracy_score(wy_true_labels, wy_pred_labels)
        else:
            final_window_accuracy = float("nan")

        metrics = {
            "problem_type": "classification",
            "cumulative_accuracy": cumulative_accuracy,
            "final_accuracy": final_accuracy,
            "final_window_accuracy": final_window_accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "total_drifts": total_drifts,
            "total_retrain_time_seconds": total_retrain_time,
            "total_runtime_seconds": total_runtime,
        }

    return predictor, drift_log, metrics


def infer_task_type(df, target_col):
    y = df[target_col]
    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique(dropna=True)
        if nunique <= 20:
            return "classification"
        return "regression"
    return "classification"


def drift_induction(
    data,
    drift_strength=0.3,
    random_state=None,
    target_col=None,
    task="auto",
    n_drifts=10,
):
    """
    Inject exactly `n_drifts` abrupt drifts by splitting the data into `n_drifts + 1`
    consecutive segments and applying a different transformation to each segment.
    """
    if random_state is not None:
        np.random.seed(random_state)

    is_dataframe = isinstance(data, pd.DataFrame)

    if not is_dataframe:
        arr = np.asarray(data).copy()
        if arr.ndim != 2:
            raise ValueError("NumPy input must be 2D.")
        if len(arr) < (n_drifts + 1):
            raise ValueError(
                f"Need at least {n_drifts + 1} rows to induce {n_drifts} drifts, got {len(arr)}."
            )

        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError("NumPy input must be numeric.")

        segment_edges = np.linspace(0, len(arr), n_drifts + 2, dtype=int)
        drifted = arr.copy()

        for seg_id in range(n_drifts + 1):
            start = segment_edges[seg_id]
            end = segment_edges[seg_id + 1]
            if start >= end:
                continue

            if seg_id == 0:
                continue

            segment = drifted[start:end].copy()

            seg_strength = drift_strength * seg_id

            shift = seg_strength * (1 if seg_id % 2 == 1 else -1)
            scale = 1.0 + (0.8 if seg_id % 3 == 0 else -0.5)
            noise_std = max(0.05, drift_strength * 0.25)

            segment = segment * scale
            segment = segment + shift
            segment = segment + np.random.normal(0, noise_std, size=segment.shape)

            drifted[start:end] = segment

        return drifted

    df = data.copy()

    if len(df) < (n_drifts + 1):
        raise ValueError(
            f"Need at least {n_drifts + 1} rows to induce {n_drifts} drifts, got {len(df)}."
        )

    if task == "auto":
        if target_col is None or target_col not in df.columns:
            task = "regression"
        else:
            y = df[target_col]
            if pd.api.types.is_numeric_dtype(y):
                nunique = y.nunique(dropna=True)
                if nunique <= 20:
                    task = "classification"
                else:
                    task = "regression"
            else:
                task = "classification"

    if task == "classification" and (
        target_col is None or target_col not in df.columns
    ):
        raise ValueError(
            "For classification, target_col must be provided and exist in the DataFrame."
        )

    drifted = df.copy()

    numeric_cols = drifted.select_dtypes(include=[np.number]).columns.tolist()
    # select_dtypes returns a DataFrame subset; extract column names as a list
    categorical_cols = drifted.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if target_col is not None and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col is not None and target_col in categorical_cols:
        categorical_cols.remove(target_col)

    segment_edges = np.linspace(0, len(drifted), n_drifts + 2, dtype=int)

    for seg_id in range(n_drifts + 1):
        start = segment_edges[seg_id]
        end = segment_edges[seg_id + 1]
        if start >= end:
            continue

        idx = drifted.index[start:end]

        if seg_id == 0:
            continue

        seg_strength = drift_strength * seg_id

        if numeric_cols:
            block = drifted.loc[idx, numeric_cols].to_numpy(dtype=float)

            shift = seg_strength * (1 if seg_id % 2 == 1 else -1)
            scale = 1.0 + (0.8 if seg_id % 3 == 0 else -0.5)
            noise_std = max(0.05, drift_strength * 0.25)

            block = block * scale
            block = block + shift
            block = block + np.random.normal(0, noise_std, size=block.shape)

            drifted.loc[idx, numeric_cols] = block

        if categorical_cols:
            for col in categorical_cols:
                values = drifted.loc[idx, col].astype("object")
                unique_vals = values.dropna().unique()

                if len(unique_vals) > 1:
                    replace_frac = min(0.6, 0.15 + 0.05 * seg_id)
                    n_replace = int(len(idx) * replace_frac)

                    if n_replace > 0:
                        replace_idx = np.random.choice(
                            idx, size=n_replace, replace=False
                        )

                        def sample_other(v):
                            choices = [x for x in unique_vals if x != v]
                            return (
                                np.random.choice(choices)
                                if len(choices) > 0
                                else v
                            )

                        drifted.loc[replace_idx, col] = drifted.loc[
                            replace_idx, col
                        ].apply(sample_other)

        if target_col is not None:
            if task == "classification":
                classes = drifted[target_col].dropna().unique()
                if len(classes) > 1:
                    flip_frac = min(0.7, 0.20 + 0.05 * seg_id)
                    n_flip = int(len(idx) * flip_frac)

                    if n_flip > 0:
                        flip_idx = np.random.choice(
                            idx, size=n_flip, replace=False
                        )

                        def sample_other_class(v):
                            choices = [c for c in classes if c != v]
                            return (
                                np.random.choice(choices)
                                if len(choices) > 0
                                else v
                            )

                        drifted.loc[flip_idx, target_col] = drifted.loc[
                            flip_idx, target_col
                        ].apply(sample_other_class)

            elif task == "regression" and pd.api.types.is_numeric_dtype(
                drifted[target_col]
            ):
                target_shift = seg_strength * (1 if seg_id % 2 == 1 else -1) * 0.75
                target_noise = max(0.02, drift_strength * 0.08)
                drifted.loc[idx, target_col] = (
                    pd.to_numeric(drifted.loc[idx, target_col], errors="coerce")
                    + target_shift
                    + np.random.normal(0, target_noise, size=len(idx))
                )

    return drifted


def zip_dir(path: str) -> bytes:
    """Zip a directory and return bytes."""
    buf = io.BytesIO()
    count = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(path):
            for f in files:
                abspath = os.path.join(root, f)
                arcname = os.path.relpath(abspath, path)
                # Use forward slashes in archive so API (possibly Linux) extracts correctly
                arcname = arcname.replace("\\", "/")
                z.write(abspath, arcname)
                count += 1
    if count == 0:
        logger.warning("zip_dir: no files found under %s", path)
    else:
        logger.info("zip_dir: zipped %d files from %s", count, path)
    buf.seek(0)
    return buf.getvalue()


def build_vision_retraining_subset(
    *,
    manifest_df: pd.DataFrame,
    output_dir: str,
    filename_column: str,
    label_column: str,
) -> None:
    """
    Build a standalone Vision dataset containing only the
    rows supplied in manifest_df.

    Output layout:

        output_dir/
            labels.csv
            images/
                cat/
                dog/

    The internal _image_path column is not written to labels.csv.
    """

    if manifest_df.empty:
        raise ValueError(
            "Vision retraining subset is empty"
        )

    if "_image_path" not in manifest_df.columns:
        raise ValueError(
            "Vision retraining subset is missing _image_path"
        )

    if filename_column not in manifest_df.columns:
        raise ValueError(
            f"Missing filename column: {filename_column}"
        )

    if label_column not in manifest_df.columns:
        raise ValueError(
            f"Missing label column: {label_column}"
        )

    # Detect the layout of the source dataset.
    #
    # Flat:
    #   images/image.jpg
    #
    # Label subdirectories:
    #   images/0/image.jpg
    #   images/1/image.jpg

    layout_flags = []

    for _, row in manifest_df.iterrows():
        source_path = str(
            row["_image_path"]
        )

        label = str(
            row[label_column]
        ).strip()

        source_parent = os.path.basename(
            os.path.dirname(source_path)
        )

        layout_flags.append(
            source_parent == label
        )

    if any(layout_flags) and not all(layout_flags):
        raise ValueError(
            "Vision retraining dataset contains mixed "
            "flat and label-subdirectory image layouts"
        )

    use_label_subdirs = bool(
        layout_flags
        and all(layout_flags)
    )

    logger.info(
        "[Vision Retrain] Source image layout: %s",
        (
            "label_subdirectories"
            if use_label_subdirs
            else "flat"
        ))

    # Start with a clean output directory.
    if os.path.isdir(output_dir):
        shutil.rmtree(
            output_dir
        )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    images_root = os.path.join(
        output_dir,
        "images",
    )

    os.makedirs(
        images_root,
        exist_ok=True,
    )

    # Copy the manifest, excluding our internal helper column.
    output_manifest = manifest_df.drop(
        columns=["_image_path"],
        errors="ignore",
    ).copy()

    normalized_filenames = []

    for _, row in manifest_df.iterrows():

        source_path = str(
            row["_image_path"]
        )

        if not os.path.isfile(source_path):
            raise FileNotFoundError(
                f"Retraining source image does not exist: "
                f"{source_path}"
            )

        label = str(
            row[label_column]
        ).strip()

        filename = os.path.basename(
            str(
                row[filename_column]
            ).replace("\\", "/")
        )

        if use_label_subdirs:

            destination_dir = os.path.join(
                images_root,
                label,
            )

        else:

            destination_dir = images_root

        os.makedirs(
            destination_dir,
            exist_ok=True,
        )

        destination_path = os.path.join(
            destination_dir,
            filename,
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        normalized_filenames.append(
            filename
        )

    # Match the layout expected by the Vision AutoML engine.
    output_manifest[
        filename_column
    ] = normalized_filenames

    output_manifest_path = os.path.join(
        output_dir,
        "labels.csv",
    )

    output_manifest.to_csv(
        output_manifest_path,
        index=False,
    )

    logger.info(
        "[Vision Retrain] Prepared retraining subset: "
        "%d images -> %s",
        len(output_manifest),
        output_dir,
    )

    logger.info(
        "[Vision Retrain] Retraining class counts: %s",
        output_manifest[
            label_column
        ].astype(str).value_counts().to_dict(),
    )


def build_segmentation_retraining_subset(
    *,
    manifest_df: pd.DataFrame,
    output_dir: str,
    filename_column: str,
    mask_column: str,
) -> None:
    """
    Build a standalone semantic-segmentation dataset.

    Output layout:

        output_dir/
            labels.csv
            images/
            masks/

    Internal resolved-path columns are not written to labels.csv.
    """

    if manifest_df.empty:
        raise ValueError(
            "Segmentation retraining subset is empty"
        )

    required_columns = {
        filename_column,
        mask_column,
        "_image_path",
        "_mask_path",
    }

    missing = (
        required_columns
        - set(manifest_df.columns)
    )

    if missing:
        raise ValueError(
            "Segmentation retraining subset is "
            f"missing columns: {sorted(missing)}"
        )

    if os.path.isdir(output_dir):
        shutil.rmtree(
            output_dir
        )

    images_root = os.path.join(
        output_dir,
        "images",
    )

    masks_root = os.path.join(
        output_dir,
        "masks",
    )

    os.makedirs(
        images_root,
        exist_ok=True,
    )

    os.makedirs(
        masks_root,
        exist_ok=True,
    )

    output_manifest = manifest_df.drop(
        columns=[
            "_image_path",
            "_mask_path",
        ],
        errors="ignore",
    ).copy()

    normalized_filenames = []
    normalized_masks = []

    for _, row in manifest_df.iterrows():

        source_image = str(
            row["_image_path"]
        )

        source_mask = str(
            row["_mask_path"]
        )

        if not os.path.isfile(
            source_image
        ):
            raise FileNotFoundError(
                "Retraining image does not exist: "
                f"{source_image}"
            )

        if not os.path.isfile(
            source_mask
        ):
            raise FileNotFoundError(
                "Retraining mask does not exist: "
                f"{source_mask}"
            )

        filename = os.path.basename(
            str(
                row[filename_column]
            ).replace("\\", "/")
        )

        mask_filename = os.path.basename(
            str(
                row[mask_column]
            ).replace("\\", "/")
        )

        destination_image = os.path.join(
            images_root,
            filename,
        )

        destination_mask = os.path.join(
            masks_root,
            mask_filename,
        )

        shutil.copy2(
            source_image,
            destination_image,
        )

        shutil.copy2(
            source_mask,
            destination_mask,
        )

        normalized_filenames.append(
            filename
        )

        normalized_masks.append(
            mask_filename
        )

    output_manifest[
        filename_column
    ] = normalized_filenames

    output_manifest[
        mask_column
    ] = normalized_masks

    output_manifest_path = os.path.join(
        output_dir,
        "labels.csv",
    )

    output_manifest.to_csv(
        output_manifest_path,
        index=False,
    )

    logger.info(
        "[Segmentation Retrain] "
        "Prepared retraining subset: "
        "%d image/mask pairs -> %s",
        len(output_manifest),
        output_dir,
    )


def split_vision_retraining_and_holdout(
    *,
    manifest_df: pd.DataFrame,
    drift_start_index: int,
    holdout_fraction: float = 1.0 / 3.0,
    min_holdout_size:  int = VISION_MIN_HOLDOUT_IMAGES,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Split a Vision stream into:

      retraining rows:
          everything before the reserved holdout

      holdout rows:
          the final portion of the post-drift region

    The holdout must never be uploaded to AutoML.

    Returns:
        retraining_manifest,
        validation_manifest,
        validation_start
    """

    total = len(
        manifest_df
    )

    if total == 0:
        raise ValueError(
            "Vision manifest is empty"
        )

    if drift_start_index < 0:
        raise ValueError(
            "drift_start_index must be >= 0"
        )

    if drift_start_index >= total:
        raise ValueError(
            "drift_start_index must be smaller "
            "than the manifest size"
        )

    post_drift_count = (
        total - drift_start_index
    )

    if min_holdout_size < 1:
        raise ValueError(
            "min_holdout_size must be >= 1"
        )

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(
            "holdout_fraction must be between 0 and 1"
        )

    # We need at least the configured holdout plus
    # at least one post-drift image left for retraining.
    if post_drift_count <= min_holdout_size:
        raise RuntimeError(
            "Not enough post-drift samples to create "
            "a reliable retraining/validation split: "
            f"post_drift_count={post_drift_count}, "
            f"minimum_holdout={min_holdout_size}"
        )

    holdout_size = max(
        min_holdout_size,
        int(
            post_drift_count
            * holdout_fraction
        ),
    )

    # Always leave at least one post-drift sample
    # available for retraining.
    holdout_size = min(
        holdout_size,
        post_drift_count - 1,
    )

    validation_start = (
        total - holdout_size
    )

    retraining_manifest = manifest_df.iloc[
        :validation_start
    ].copy()

    validation_manifest = manifest_df.iloc[
        validation_start:
    ].copy()

    if retraining_manifest.empty:
        raise RuntimeError(
            "Vision retraining manifest is empty"
        )

    if validation_manifest.empty:
        raise RuntimeError(
            "Vision validation manifest is empty"
        )

    return (
        retraining_manifest,
        validation_manifest,
        validation_start,
    )


def upload_vision_retraining_dataset(
    *,
    user_id: str,
    source_dataset_id: str,
    task_id: str,
    dataset_dir: str,
) -> dict:
    """
    Package the locally modified Vision dataset and upload it
    to AutoDW as a new dataset for retraining.
    """

    logger.info(
        "[Vision Retrain] Packaging modified dataset: %s",
        dataset_dir,
    )

    zip_bytes = zip_dir(
        dataset_dir
    )

    if not zip_bytes or len(zip_bytes) < 22:
        raise RuntimeError(
            "Derived Vision dataset ZIP is empty"
        )

    safe_task_id = (
        str(task_id or "unknown")
        .replace("/", "_")
        .replace("\\", "_")
    )

    retrain_dataset_id = (
        f"{source_dataset_id}"
        f"-drift-retrain-"
        f"{safe_task_id}"
    )

    # IMPORTANT:
    # AutoDW dataset upload endpoint.
    # user_id is supplied as a path parameter.
    upload_url = (
        f"{API_BASE.rstrip('/')}"
        f"/datasets/upload/{user_id}"
    )

    zip_buffer = io.BytesIO(
        zip_bytes
    )

    zip_buffer.seek(0)

    files = {
        "file": (
            "vision_drift_retrain.zip",
            zip_buffer,
            "application/zip",
        )
    }

    data = {
        "dataset_id": retrain_dataset_id,
        "name": retrain_dataset_id,
        "description": (
            "Vision concept-drift retraining dataset "
            f"created by task {task_id}"
        ),
        "version": "v1",
        "tags": (
            "vision,concept-drift,retraining"
        ),
    }

    logger.info(
        "[Vision Retrain] Uploading derived dataset "
        "user=%s dataset=%s version=v1 url=%s",
        user_id,
        retrain_dataset_id,
        upload_url,
    )

    response = requests.post(
        upload_url,
        files=files,
        data=data,
        timeout=300,
    )

    if not response.ok:
        logger.error(
            "[Vision Retrain] Derived dataset upload failed "
            "HTTP %s: %s",
            response.status_code,
            response.text[:2000],
        )

    response.raise_for_status()

    upload_result = response.json()

    logger.info(
        "[Vision Retrain] Derived dataset upload response: %s",
        json.dumps(
            upload_result,
            default=str,
        ),
    )

    # Do not blindly trust response shape.
    uploaded_dataset_id = (
        upload_result.get("dataset_id")
        or retrain_dataset_id
    )

    uploaded_version = (
        upload_result.get("version")
        or "v1"
    )

    return {
        "dataset_id": uploaded_dataset_id,
        "version": uploaded_version,
        "raw_response": upload_result,
    }


def post_concept_drift_report(
    user_id: str,
    dataset_id: str,
    dataset_version: str,
    model_id: str,
    model_version: str,
    report: dict,
) -> dict:
    """POST concept drift report (JSON) to Data Warehouse. Does not send Kafka event (consumer sends that)."""
    url = f"{API_BASE}/concept-drift-reports/"
    payload = {
        "user_id": user_id,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "model_id": model_id,
        "model_version": model_version,
        "report": report,
    }
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def upload_model_to_dw(
    user_id: str,
    model_id: str,
    model_zip_bytes: bytes,
    dataset_id: str,
    dataset_version: str,
    model_type: str = "classification",
    framework: str = "other",
    description_prefix: str = "Concept drift retrained model",
) -> dict:
    """Upload retrained model (ZIP folder) to Data Warehouse. Returns created model metadata."""
    url = f"{API_BASE}/ai-models/upload/folder/{user_id}"
    # API expects File(...) under key "zip_file" and Form fields; ensure zip is non-empty and stream at 0
    if not model_zip_bytes or len(model_zip_bytes) < 22:  # minimal zip is ~22 bytes
        raise ValueError(f"Model zip is empty or too small ({len(model_zip_bytes) if model_zip_bytes else 0} bytes)")
    zip_buffer = io.BytesIO(model_zip_bytes)
    zip_buffer.seek(0)
    files = {"zip_file": ("model.zip", zip_buffer, "application/zip")}
    data = {
        "model_id": model_id,
        "name": f"{description_prefix} - {model_id}",
        "description": f"{description_prefix} (drift retrain on dataset {dataset_id} {dataset_version})",
        "framework": framework,
        "model_type": model_type,
        "training_dataset": dataset_id,
        "preserve_structure": "true",
    }
    r = requests.post(url, files=files, data=data, timeout=300)
    if not r.ok:
        try:
            err_body = r.json()
            logger.error(f"Upload model failed {r.status_code}: {err_body}")
        except Exception:
            logger.error(f"Upload model failed {r.status_code}: {r.text[:500]}")
        r.raise_for_status()
    return r.json()


async def process_drift_trigger(event: dict, producer: AIOKafkaProducer) -> None:
    """Handle one concept-drift-trigger event."""
    task_id = event.get("task_id") or event.get("event_id")
    input_obj = event.get("input", event)
    user_id = input_obj.get("user_id")
    dataset_id = input_obj.get("dataset_id")
    dataset_version = input_obj.get("dataset_version", "v1")
    model_id = input_obj.get("model_id")
    model_version = input_obj.get("model_version", "v1")
    task_category = (input_obj.get("task_category") or "tabular").strip().lower()
    task_type = (input_obj.get("task_type") or "classification").strip().lower()

    filename_column = input_obj.get("filename_column", "filename",)

    label_column = input_obj.get("label_column","label",)

    mask_column = input_obj.get(
        "mask_column",
        "mask_filename",
    )

    dataset_split = (input_obj.get("dataset_split") or "drift").strip().lower()

    model_size = (input_obj.get("model_size") or "small").strip().lower()
    target_column = input_obj.get("target_column_name") or input_obj.get("target_column")
    window_size = int(input_obj.get("window_size", 100))
    raw_induce_drift = input_obj.get(
        "induce_drift",
        False,
    )

    if isinstance(raw_induce_drift, str):
        induce_drift = (
                raw_induce_drift.strip().lower()
                in {"1", "true", "yes", "on"}
        )
    else:
        induce_drift = bool(raw_induce_drift)

    if not all([user_id, dataset_id, model_id]):
        logger.warning("Missing user_id, dataset_id or model_id; skipping")
        await _send_complete(producer, task_id, user_id, model_id, None, dataset_id, dataset_version, success=False, error_message="Missing required fields")
        return

    if task_category == "tabular":
        if not target_column:
            logger.warning(
                "Tabular drift task requires target_column_name"
            )

            await _send_complete(
                producer,
                task_id,
                user_id,
                model_id,
                None,
                dataset_id,
                dataset_version,
                success=False,
                error_message=(
                    "Tabular task requires target_column_name"
                ),
            )
            return

    elif task_category == "vision":

        normalized_vision_task = normalize_vision_task_type(
            task_type
        )

        supported_vision_tasks = {
            "image_classification",
            "image_segmentation",
        }

        if normalized_vision_task not in supported_vision_tasks:
            await _send_complete(
                producer,
                task_id,
                user_id,
                model_id,
                model_version,
                dataset_id,
                dataset_version,
                success=False,
                error_message=(
                    "Unsupported Vision concept-drift task_type="
                    f"{normalized_vision_task}. "
                    f"Supported tasks: "
                    f"{sorted(supported_vision_tasks)}"
                ),
            )
            return

    else:
        await _send_complete(
            producer,
            task_id,
            user_id,
            model_id,
            None,
            dataset_id,
            dataset_version,
            success=False,
            error_message=(
                f"Unsupported task_category={task_category}"
            ),
        )
        return

    if not claim_task_id(
        task_id,
        user_id=user_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        model_id=model_id,
        model_version=model_version,
    ):
        return

    heartbeat = _DriftJobHeartbeat(task_id)
    heartbeat.start()
    try:
        logger.info(
            "Worker %s processing concept drift task_id=%s user=%s dataset=%s model=%s",
            WORKER_ID,
            task_id,
            user_id,
            dataset_id,
            model_id,
        )
        await _process_drift_trigger_body(
            producer=producer,
            task_id=task_id,
            user_id=user_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            model_id=model_id,
            model_version=model_version,

            task_category=task_category,
            task_type=task_type,

            target_column=target_column,

            filename_column=filename_column,
            label_column=label_column,
            mask_column=mask_column,
            dataset_split=dataset_split,
            model_size=model_size,

            window_size=window_size,
            induce_drift=induce_drift,
        )
    finally:
        heartbeat.stop()
        release_task_id(task_id)


async def _process_drift_trigger_body(
    *,
    producer: AIOKafkaProducer,
    task_id: str,
    user_id: str,
    dataset_id: str,
    dataset_version: str,
    model_id: str,
    model_version: str,

    task_category: str,
    task_type: str,

    target_column: str | None,

    filename_column: str,
    label_column: str,
    mask_column: str,
    dataset_split: str,
    model_size: str,

    window_size: int,
    induce_drift: bool,
) -> None:
    if task_category == "tabular":
        await _process_tabular_drift(
            producer=producer,
            task_id=task_id,
            user_id=user_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            model_id=model_id,
            model_version=model_version,
            target_column=target_column,
            window_size=window_size,
        )

        return

    if task_category == "vision":
        await _process_vision_drift(
            producer=producer,
            task_id=task_id,
            user_id=user_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            model_id=model_id,
            model_version=model_version,
            task_type=normalize_vision_task_type(
                task_type
            ),
            filename_column=filename_column,
            label_column=label_column,
            mask_column=mask_column,
            dataset_split=dataset_split,
            model_size=model_size,
            window_size=window_size,
            induce_drift=induce_drift,
        )

        return

    raise ValueError(
        f"Unsupported task_category={task_category}"
    )


async def _process_image_segmentation_drift(
    *,
    producer: AIOKafkaProducer,
    task_id: str,
    user_id: str,
    dataset_id: str,
    dataset_version: str,
    model_id: str,
    model_version: str,
    filename_column: str,
    mask_column: str,
    dataset_split: str,
    model_size: str,
    window_size: int,
    induce_drift: bool,
) -> None:

    try:
        logger.info(
            "[Segmentation Drift] Starting "
            "image_segmentation workflow"
        )

        metadata = fetch_dataset_metadata(
            user_id,
            dataset_id,
            dataset_version,
        )

        logger.info(
            "[Segmentation Drift] "
            "Dataset metadata: %s",
            json.dumps(
                metadata,
                default=str,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:

            dataset_dir = os.path.join(
                tmp,
                "dataset",
            )
            model_dir = os.path.join(
                tmp,
                "model",
            )

            os.makedirs(
                dataset_dir,
                exist_ok=True,
            )

            os.makedirs(
                model_dir,
                exist_ok=True,
            )

            dataset_bytes = (
                download_dataset_bytes(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    version=dataset_version,
                    split=dataset_split,
                )
            )

            extracted = (
                extract_vision_dataset(
                    dataset_bytes,
                    dataset_dir,
                )
            )

            logger.info(
                "[Segmentation Drift] "
                "Extracted %d files",
                len(extracted),
            )

            manifest_path = (
                find_segmentation_manifest(
                    dataset_root=dataset_dir,
                    filename_column=(
                        filename_column
                    ),
                    mask_column=mask_column,
                )
            )

            manifest = pd.read_csv(
                manifest_path
            )

            manifest = (
                resolve_segmentation_paths(
                    manifest_df=manifest,
                    dataset_root=dataset_dir,
                    filename_column=(
                        filename_column
                    ),
                    mask_column=mask_column,
                )
            )

            validate_segmentation_manifest(
                manifest
            )

            logger.info(
                "[Segmentation Drift] "
                "Dataset preparation PASSED: "
                "%d image/mask pairs",
                len(manifest),
            )

            # --------------------------------------------------
            # Optional synthetic RGB drift
            # --------------------------------------------------

            if induce_drift:
                manifest = manifest.sample(
                    frac=1.0,
                    random_state=42,
                ).reset_index(
                    drop=True
                )

                logger.info(
                    "[Vision Drift] Manifest shuffled "
                    "before synthetic drift"
                )

                logger.info(
                    "[Vision Drift] Synthetic image drift enabled"
                )
                logger.info(
                    "[Segmentation Drift] "
                    "Applying synthetic RGB drift"
                )

                apply_synthetic_image_drift(
                    manifest,
                    start_fraction=0.5,
                )

                logger.info(
                    "[Segmentation Drift] "
                    "Synthetic RGB drift applied"
                )

            # --------------------------------------------------
            # Download source segmentation model
            # --------------------------------------------------

            logger.info(
                "[Segmentation Drift] "
                "Downloading source model "
                "user=%s model=%s version=%s",
                user_id,
                model_id,
                model_version,
            )

            download_model_folder(
                user_id,
                model_id,
                model_version,
                model_dir,
            )

            # --------------------------------------------------
            # Load segmentation adapter
            # --------------------------------------------------

            logger.info(
                "[Segmentation Drift] "
                "Loading segmentation model adapter"
            )

            adapter = SegmentationModelAdapter.load(
                model_dir
            )

            logger.info(
                "[Segmentation Drift] "
                "Segmentation model loaded successfully"
            )

            # --------------------------------------------------
            # Run ADWIN monitoring
            # --------------------------------------------------

            logger.info(
                "[Segmentation Drift] "
                "Starting ADWIN monitoring "
                "(requested window_size=%d; "
                "ADWIN uses adaptive windows)",
                window_size,
            )

            result = await asyncio.to_thread(
                monitor_segmentation_drift,
                manifest_df=manifest,
                adapter=adapter,
                filename_column=filename_column,
                mask_column=mask_column,
                window_size=window_size,
                adwin_delta=VISION_ADWIN_DELTA,
                ignore_background=(
                    SEGMENTATION_IGNORE_BACKGROUND
                ),
                ignore_index=255,
            )

            # --------------------------------------------------
            # Aggregate monitoring metrics
            # --------------------------------------------------

            if result["mean_ious"]:
                mean_iou = float(
                    np.mean(
                        result["mean_ious"]
                    )
                )
            else:
                mean_iou = 0.0

            if result["mean_dices"]:
                mean_dice = float(
                    np.mean(
                        result["mean_dices"]
                    )
                )
            else:
                mean_dice = 0.0

            if result["pixel_accuracies"]:
                pixel_accuracy = float(
                    np.mean(
                        result[
                            "pixel_accuracies"
                        ]
                    )
                )
            else:
                pixel_accuracy = 0.0

            if result["errors"]:
                mean_error = float(
                    np.mean(
                        result["errors"]
                    )
                )
            else:
                mean_error = 0.0

            metrics = {
                "problem_type": (
                    "image_segmentation"
                ),
                "drift_index": (
                    result["drift_index"]
                ),
                "samples_processed": (
                    result["samples_processed"]
                ),
                "total_images": len(
                    manifest
                ),
                "mean_iou": mean_iou,
                "mean_dice": mean_dice,
                "pixel_accuracy": (
                    pixel_accuracy
                ),
                "mean_error": mean_error,
                "ignore_background": (
                    SEGMENTATION_IGNORE_BACKGROUND
                ),
                "ignore_index": 255,
                "synthetic_drift": (
                    induce_drift
                ),
                "synthetic_drift_start_index": (
                    int(
                        len(manifest) * 0.5
                    )
                    if induce_drift
                    else None
                ),
            }

            logger.info(
                "[Segmentation Drift] "
                "Drift=%s drift_index=%s "
                "samples=%d/%d "
                "mIoU=%.4f Dice=%.4f "
                "pixel_accuracy=%.4f "
                "mean_error=%.4f",
                result["drift_detected"],
                result["drift_index"],
                result["samples_processed"],
                len(manifest),
                mean_iou,
                mean_dice,
                pixel_accuracy,
                mean_error,
            )

            # ==================================================
            # NO DRIFT
            # ==================================================

            if not result["drift_detected"]:

                logger.info(
                    "[Segmentation Drift] "
                    "No concept drift detected; "
                    "segmentation retraining is not required"
                )

                drift_report = {
                    "task_id": task_id,
                    "timestamp": datetime.now(
                        tz=timezone.utc
                    ).isoformat(),
                    "task_category": "vision",
                    "task_type": (
                        "image_segmentation"
                    ),
                    "drift_detected": False,
                    "drift_metrics": metrics,
                    "source_model": {
                        "model_id": model_id,
                        "model_version": (
                            model_version
                        ),
                    },
                    "replacement_model": None,
                    "message": (
                        "No concept drift detected"
                    ),
                }

                try:
                    saved = post_concept_drift_report(
                        user_id=user_id,
                        dataset_id=dataset_id,
                        dataset_version=(
                            dataset_version
                        ),
                        model_id=model_id,
                        model_version=model_version,
                        report=drift_report,
                    )

                    logger.info(
                        "[Segmentation Drift] "
                        "No-drift report saved: id=%s",
                        saved.get(
                            "id",
                            "",
                        ),
                    )

                except Exception as report_error:
                    logger.warning(
                        "[Segmentation Drift] "
                        "Failed to save no-drift report: %s",
                        report_error,
                    )

                await _send_complete(
                    producer,
                    task_id,
                    user_id,
                    model_id,
                    model_version,
                    dataset_id,
                    dataset_version,
                    success=True,
                    drift_metrics=metrics,
                    drift_detected=False,
                )

                return

            # ==================================================
            # DRIFT DETECTED
            # ==================================================

            logger.info(
                "[Segmentation Retrain] "
                "Drift detected at index %s; "
                "preparing leakage-free retraining "
                "and holdout datasets",
                result["drift_index"],
            )

            # --------------------------------------------------
            # 1. Split retraining vs reserved holdout
            # --------------------------------------------------

            if induce_drift:
                # Synthetic drift begins halfway through
                # the stream, not at the later ADWIN
                # detection point.
                drift_start_index = int(
                    len(manifest) * 0.5
                )

            else:
                drift_start_index = (
                    result["drift_index"]
                    if result["drift_index"] is not None
                    else 0
                )

            (
                retraining_manifest,
                validation_manifest,
                validation_start,
            ) = split_vision_retraining_and_holdout(
                manifest_df=manifest,
                drift_start_index=drift_start_index,
                holdout_fraction=1.0 / 3.0,
            )

            if retraining_manifest.empty:
                raise RuntimeError(
                    "Segmentation retraining "
                    "manifest is empty"
                )

            if validation_manifest.empty:
                raise RuntimeError(
                    "Segmentation validation "
                    "manifest is empty"
                )

            logger.info(
                "[Segmentation Retrain] "
                "Holdout split prepared: "
                "total=%d drift_start=%d "
                "retraining=%d validation=%d "
                "validation_start=%d",
                len(manifest),
                drift_start_index,
                len(retraining_manifest),
                len(validation_manifest),
                validation_start,
            )

            # --------------------------------------------------
            # 2. Build standalone retraining dataset
            #
            # IMPORTANT:
            # validation_manifest is never copied into
            # this directory and therefore cannot be
            # uploaded to AutoML.
            # --------------------------------------------------

            retrain_dataset_dir = os.path.join(
                tmp,
                "segmentation_retrain_dataset",
            )

            await asyncio.to_thread(
                build_segmentation_retraining_subset,
                manifest_df=retraining_manifest,
                output_dir=retrain_dataset_dir,
                filename_column=filename_column,
                mask_column=mask_column,
            )

            # --------------------------------------------------
            # 3. Upload retraining subset to AutoDW
            # --------------------------------------------------

            logger.info(
                "[Segmentation Retrain] "
                "Uploading retraining subset to AutoDW"
            )

            retrain_dataset_metadata = (
                await asyncio.to_thread(
                    upload_vision_retraining_dataset,
                    user_id=user_id,
                    source_dataset_id=dataset_id,
                    task_id=task_id,
                    dataset_dir=retrain_dataset_dir,
                )
            )

            retrain_dataset_id = (
                retrain_dataset_metadata[
                    "dataset_id"
                ]
            )

            retrain_dataset_version = (
                retrain_dataset_metadata[
                    "version"
                ]
            )

            logger.info(
                "[Segmentation Retrain] "
                "Uploaded retraining dataset: "
                "dataset=%s version=%s",
                retrain_dataset_id,
                retrain_dataset_version,
            )

            # --------------------------------------------------
            # 4. Call AutoML for semantic segmentation
            # --------------------------------------------------

            logger.info(
                "[Segmentation Retrain] "
                "Starting AutoML segmentation retraining"
            )

            automl_response = (
                await retrain_vision_model_via_automl(
                    task_id=task_id,
                    user_id=user_id,
                    dataset_id=retrain_dataset_id,
                    dataset_version=(
                        retrain_dataset_version
                    ),
                    filename_column=filename_column,

                    # AutoML's generic Vision endpoint
                    # uses label_column for the target.
                    # For semantic segmentation, the
                    # target is the mask filename column.
                    label_column=mask_column,

                    model_size=model_size,
                    task_type="image_segmentation",

                    # This is already a standalone
                    # uploaded retraining dataset.
                    dataset_split=None,

                    time_budget=(
                        VISION_RETRAIN_TIME_BUDGET_SECONDS
                    ),
                )
            )

            # --------------------------------------------------
            # 5. Extract candidate model identity
            # --------------------------------------------------

            candidate_model_id = (
                automl_response.get(
                    "model_id"
                )
            )

            candidate_model_version = (
                    automl_response.get(
                        "version"
                    )
                    or automl_response.get(
                "model_version"
            )
            )

            if not candidate_model_id:
                raise RuntimeError(
                    "AutoML segmentation retraining "
                    "did not return model_id: "
                    f"{json.dumps(automl_response, default=str)}"
                )

            if not candidate_model_version:
                raise RuntimeError(
                    "AutoML segmentation retraining "
                    "did not return model version: "
                    f"{json.dumps(automl_response, default=str)}"
                )

            logger.info(
                "[Segmentation Retrain] "
                "Candidate model created: "
                "model=%s version=%s",
                candidate_model_id,
                candidate_model_version,
            )

            # ==================================================
            # 6. DOWNLOAD AND LOAD CANDIDATE
            # ==================================================

            candidate_model_dir = os.path.join(
                tmp,
                "candidate_segmentation_model",
            )

            os.makedirs(
                candidate_model_dir,
                exist_ok=True,
            )

            logger.info(
                "[Segmentation Retrain] "
                "Downloading candidate model "
                "model=%s version=%s",
                candidate_model_id,
                candidate_model_version,
            )

            await asyncio.to_thread(
                download_model_folder,
                user_id,
                candidate_model_id,
                candidate_model_version,
                candidate_model_dir,
            )

            candidate_adapter = (
                SegmentationModelAdapter.load(
                    candidate_model_dir
                )
            )

            logger.info(
                "[Segmentation Retrain] "
                "Candidate model loaded successfully"
            )

            # ==================================================
            # 7. EVALUATE SOURCE AND CANDIDATE
            #    ON EXACT SAME RESERVED HOLDOUT
            # ==================================================

            if validation_manifest.empty:
                raise RuntimeError(
                    "Segmentation validation holdout is empty"
                )

            logger.info(
                "[Segmentation Retrain] "
                "Evaluating source model on "
                "%d reserved holdout pairs",
                len(validation_manifest),
            )

            source_validation_metrics = (
                await asyncio.to_thread(
                    evaluate_segmentation_model_on_manifest,
                    manifest_df=validation_manifest,
                    adapter=adapter,
                    ignore_background=(
                        SEGMENTATION_IGNORE_BACKGROUND
                    ),
                    ignore_index=255,
                )
            )

            logger.info(
                "[Segmentation Retrain] "
                "Source holdout metrics: "
                "mIoU=%.4f Dice=%.4f "
                "pixel_accuracy=%.4f",
                source_validation_metrics[
                    "mean_iou"
                ],
                source_validation_metrics[
                    "mean_dice"
                ],
                source_validation_metrics[
                    "pixel_accuracy"
                ],
            )

            logger.info(
                "[Segmentation Retrain] "
                "Evaluating candidate model on "
                "%d reserved holdout pairs",
                len(validation_manifest),
            )

            candidate_validation_metrics = (
                await asyncio.to_thread(
                    evaluate_segmentation_model_on_manifest,
                    manifest_df=validation_manifest,
                    adapter=candidate_adapter,
                    ignore_background=(
                        SEGMENTATION_IGNORE_BACKGROUND
                    ),
                    ignore_index=255,
                )
            )

            logger.info(
                "[Segmentation Retrain] "
                "Candidate holdout metrics: "
                "mIoU=%.4f Dice=%.4f "
                "pixel_accuracy=%.4f",
                candidate_validation_metrics[
                    "mean_iou"
                ],
                candidate_validation_metrics[
                    "mean_dice"
                ],
                candidate_validation_metrics[
                    "pixel_accuracy"
                ],
            )

            # ==================================================
            # 8. ACCEPTANCE GATE
            # ==================================================

            miou_improvement = (
                    candidate_validation_metrics[
                        "mean_iou"
                    ]
                    - source_validation_metrics[
                        "mean_iou"
                    ]
            )

            dice_improvement = (
                    candidate_validation_metrics[
                        "mean_dice"
                    ]
                    - source_validation_metrics[
                        "mean_dice"
                    ]
            )

            replacement_accepted = (
                    miou_improvement
                    >= SEGMENTATION_MIN_MIOU_IMPROVEMENT
                    and
                    dice_improvement
                    >= SEGMENTATION_MIN_DICE_IMPROVEMENT
            )

            logger.info(
                "[Segmentation Retrain] "
                "Replacement accepted=%s "
                "mIoU improvement=%+.4f "
                "required=%+.4f "
                "Dice improvement=%+.4f "
                "required=%+.4f",
                replacement_accepted,
                miou_improvement,
                SEGMENTATION_MIN_MIOU_IMPROVEMENT,
                dice_improvement,
                SEGMENTATION_MIN_DICE_IMPROVEMENT,
            )

            if replacement_accepted:
                completion_model_id = (
                    candidate_model_id
                )

                completion_model_version = (
                    candidate_model_version
                )

            else:
                completion_model_id = (
                    model_id
                )

                completion_model_version = (
                    model_version
                )

            logger.info(
                "[Segmentation Retrain] "
                "Selected model for completion: "
                "model=%s version=%s "
                "replacement_accepted=%s",
                completion_model_id,
                completion_model_version,
                replacement_accepted,
            )

            # ==================================================
            # 9. FINAL METRICS
            # ==================================================

            metrics[
                "retraining_implemented"
            ] = True

            metrics[
                "retraining_dataset_id"
            ] = retrain_dataset_id

            metrics[
                "retraining_dataset_version"
            ] = retrain_dataset_version

            metrics[
                "retraining_pairs"
            ] = len(
                retraining_manifest
            )

            metrics[
                "validation_pairs"
            ] = len(
                validation_manifest
            )

            metrics[
                "validation_start_index"
            ] = validation_start

            metrics[
                "validation_overlaps_retraining_dataset"
            ] = False

            metrics[
                "source_validation_metrics"
            ] = source_validation_metrics

            metrics[
                "candidate_validation_metrics"
            ] = candidate_validation_metrics

            metrics[
                "candidate_model_id"
            ] = candidate_model_id

            metrics[
                "candidate_model_version"
            ] = candidate_model_version

            metrics[
                "candidate_evaluated"
            ] = True

            metrics[
                "miou_improvement"
            ] = float(
                miou_improvement
            )

            metrics[
                "dice_improvement"
            ] = float(
                dice_improvement
            )

            metrics[
                "min_miou_improvement_required"
            ] = (
                SEGMENTATION_MIN_MIOU_IMPROVEMENT
            )

            metrics[
                "min_dice_improvement_required"
            ] = (
                SEGMENTATION_MIN_DICE_IMPROVEMENT
            )

            metrics[
                "replacement_accepted"
            ] = replacement_accepted

            metrics[
                "selected_model_id"
            ] = completion_model_id

            metrics[
                "selected_model_version"
            ] = completion_model_version

            # ==================================================
            # 10. BUILD FINAL SEGMENTATION DRIFT REPORT
            # ==================================================

            if replacement_accepted:
                report_message = (
                    "Concept drift detected; segmentation model "
                    "retrained and replacement model accepted"
                )
            else:
                report_message = (
                    "Concept drift detected; segmentation model "
                    "retrained but replacement model was rejected. "
                    "Source model retained"
                )

            drift_report = {
                "task_id": task_id,
                "timestamp": datetime.now(
                    tz=timezone.utc
                ).isoformat(),

                "task_category": "vision",
                "task_type": "image_segmentation",

                "drift_detected": True,

                "source_model": {
                    "model_id": model_id,
                    "model_version": model_version,
                },

                "replacement_model": {
                    "model_id": candidate_model_id,
                    "model_version": candidate_model_version,
                },

                "selected_model": {
                    "model_id": completion_model_id,
                    "model_version": completion_model_version,
                },

                "source_dataset": {
                    "dataset_id": dataset_id,
                    "dataset_version": dataset_version,
                },

                "retraining_dataset": {
                    "dataset_id": retrain_dataset_id,
                    "dataset_version": retrain_dataset_version,
                },

                "validation": {
                    "strategy": (
                        "reserved_synthetic_post_drift_holdout"
                        if induce_drift
                        else "reserved_real_post_drift_holdout"
                    ),
                    "validation_pairs": len(
                        validation_manifest
                    ),
                    "retraining_pairs": len(
                        retraining_manifest
                    ),
                    "validation_start_index": (
                        validation_start
                    ),
                    "overlaps_retraining_dataset": False,
                },

                "drift_metrics": metrics,

                "source_validation_metrics": (
                    source_validation_metrics
                ),

                "candidate_validation_metrics": (
                    candidate_validation_metrics
                ),

                "replacement_accepted": (
                    replacement_accepted
                ),

                "model_comparison": {
                    "miou_improvement": float(
                        miou_improvement
                    ),
                    "dice_improvement": float(
                        dice_improvement
                    ),
                    "min_miou_improvement_required": (
                        SEGMENTATION_MIN_MIOU_IMPROVEMENT
                    ),
                    "min_dice_improvement_required": (
                        SEGMENTATION_MIN_DICE_IMPROVEMENT
                    ),
                },

                "message": report_message,
            }

            # ==================================================
            # 11. SAVE FINAL SEGMENTATION DRIFT REPORT
            # ==================================================

            try:
                saved = await asyncio.to_thread(
                    post_concept_drift_report,
                    user_id=user_id,
                    dataset_id=dataset_id,
                    dataset_version=dataset_version,

                    # Associate the report with the source
                    # model on which drift was detected.
                    model_id=model_id,
                    model_version=model_version,

                    report=drift_report,
                )

                logger.info(
                    "[Segmentation Retrain] "
                    "Final drift report saved: id=%s",
                    saved.get(
                        "id",
                        "",
                    ),
                )

            except Exception as report_error:
                logger.warning(
                    "[Segmentation Retrain] "
                    "Failed to save final drift report: %s",
                    report_error,
                )

            # ==================================================
            # 12. FINAL COMPLETION
            # ==================================================

            await _send_complete(
                producer,
                task_id,
                user_id,
                completion_model_id,
                completion_model_version,
                dataset_id,
                dataset_version,
                success=True,
                drift_metrics=metrics,
                drift_detected=True,
            )

            return

    except Exception as exc:

        logger.exception(
            "[Segmentation Drift] "
            "Processing failed: %s",
            exc,
        )

        await _send_complete(
            producer,
            task_id,
            user_id,
            model_id,
            model_version,
            dataset_id,
            dataset_version,
            success=False,
            error_message=str(exc),
        )


async def _process_tabular_drift(
    *,
    producer: AIOKafkaProducer,
    task_id: str,
    user_id: str,
    dataset_id: str,
    dataset_version: str,
    model_id: str,
    model_version: str,
    target_column: str,
    window_size: int,
) -> None:
    """Core drift detection logic (called after lock claim)."""
    try:
        logger.info(f"Downloading drift split: dataset {dataset_id} version {dataset_version}")
        df_drift = download_tabular_drift_dataset(user_id, dataset_id, dataset_version)
        if df_drift.empty or len(df_drift) < 2:
            await _send_complete(producer, task_id, user_id, model_id, None, dataset_id, dataset_version, success=False, error_message="Drift split empty or too small")
            return

        feature_cols = [c for c in df_drift.columns if c != target_column]
        if not feature_cols or target_column not in df_drift.columns:
            await _send_complete(producer, task_id, user_id, model_id, None, dataset_id, dataset_version, success=False, error_message="Target or feature columns invalid")
            return

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "model")
            os.makedirs(model_dir, exist_ok=True)
            download_model_folder(user_id, model_id, model_version, model_dir)
            load_path = find_autogluon_model_dir(model_dir)
            from autogluon.tabular import TabularPredictor
            predictor = None
            try:
                predictor = TabularPredictor.load(load_path)
            except AssertionError as e:
                msg = str(e)
                # AutoGluon can require exact version and Python match. For pipeline testing, bypass both.
                if "require_version_match" in msg or "require_py_version_match" in msg or "version" in msg.lower():
                    logger.warning(
                        "AutoGluon predictor version/Python mismatch. "
                        "Retrying with require_version_match=False, require_py_version_match=False (NOT for production). "
                        f"Error was: {e}"
                    )
                    try:
                        predictor = TabularPredictor.load(
                            load_path,
                            require_version_match=False,
                            require_py_version_match=False,
                        )
                    except AssertionError as e2:
                        logger.error(f"Still failed after bypassing version checks: {e2}")
                        raise
                else:
                    raise
            if predictor is None:
                raise RuntimeError("Failed to load TabularPredictor")
            retrain_path = os.path.join(tmp, "retrained")
            os.makedirs(retrain_path, exist_ok=True)

            # Configure drift parameters based on task type and sample size (v4 pipeline)
            task_type = infer_task_type(df_drift, target_column)
            logger.info("[Drift] Starting drift configuration!")
            if task_type == "classification":
                if len(df_drift) < 500:
                    window_size = 150
                    drift_strength = 1.5
                else:
                    window_size = 200
                    drift_strength = 0.80
            else:
                if len(df_drift) < 100:
                    window_size = max(20, len(df_drift) // 4)
                    drift_strength = 0.08
                else:
                    window_size = min(200, max(50, len(df_drift) // 10))
                    drift_strength = 0.05
            logger.info(
                "[Drift] Drift configuration finished. Using window_size=%s, drift_strength=%s",
                window_size,
                drift_strength,
            )

            drifted_df = drift_induction(
                df_drift,
                drift_strength=drift_strength,
                random_state=45,
                target_col=target_column,
                task=task_type,
                n_drifts=10,
            )
            predictor, drift_log, metrics = monitor_drift_and_retrain(
                df=drifted_df,
                predictor=predictor,
                feature_cols=feature_cols,
                target_col=target_column,
                retrain_path=retrain_path,
                window_size=window_size,
                delta=0.2,
                clock=1,
                batch_predict_size=512,
                min_samples_before_detection=30,
                cooldown_samples=30,
                retrain_presets="medium_quality_faster_train",
                verbose=True,
            )
            n_drifts = metrics.get("total_drifts", 0)

            if n_drifts == 0:
                # No drift detected: do not upload (retrain_path is empty). POST report to DW, then send success.
                logger.info("[Drift] No concept drift detected; skipping model upload and sending success")
                drift_report = {
                    "drift_detected": False,
                    "drift_metrics": metrics,
                    "message": "No concept drift detected",
                }
                try:
                    saved = post_concept_drift_report(
                        user_id, dataset_id, dataset_version, model_id, model_version, drift_report
                    )
                    logger.info(f"Concept drift report saved (no drift): id={saved.get('id', '')}")
                except Exception as e:
                    logger.warning("Failed to POST concept drift report to DW: %s", e)
                await _send_complete(
                    producer, task_id, user_id, model_id, model_version, dataset_id, dataset_version,
                    success=True, drift_metrics=metrics, drift_detected=False
                )
                return
            # Drift detected: zip retrained model and upload to DW
            debug_dir = os.path.join(os.getcwd(), "concept_drift_upload_debug")
            os.makedirs(debug_dir, exist_ok=True)
            safe_id = (task_id or "unknown").replace("/", "_").replace("\\", "_")[:50]
            ts = int(time.time())
            dest_retrained = os.path.join(debug_dir, f"{safe_id}_{ts}_retrained")
            dest_full_tmp = os.path.join(debug_dir, f"{safe_id}_{ts}_full_tmp")
            try:
                shutil.copytree(retrain_path, dest_retrained, dirs_exist_ok=True)
                shutil.copytree(tmp, dest_full_tmp, dirs_exist_ok=True)
                logger.info(
                    "Saved upload folder for debugging: retrained=%s, full_tmp=%s",
                    dest_retrained,
                    dest_full_tmp,
                )
            except Exception as copy_err:
                logger.warning("Could not save debug copies: %s", copy_err)
            model_zip = zip_dir(retrain_path)
        new_metadata = upload_model_to_dw(
            user_id=user_id,
            model_id=model_id,
            model_zip_bytes=model_zip,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
        )
        new_version = new_metadata.get("version", "v2")
        logger.info(f"Uploaded retrained model version {new_version}")
        drift_report = {
            "drift_detected": True,
            "drift_metrics": metrics,
            "new_model_version": new_version,
        }
        try:
            saved = post_concept_drift_report(
                user_id, dataset_id, dataset_version, model_id, new_version, drift_report
            )
            logger.info(f"Concept drift report saved (drift detected): id={saved.get('id', '')}")
        except Exception as e:
            logger.warning("Failed to POST concept drift report to DW: %s", e)
        await _send_complete(
            producer, task_id, user_id, model_id, new_version, dataset_id, dataset_version,
            success=True, drift_metrics=metrics, drift_detected=True
        )
    except Exception as e:
        logger.exception(f"Concept drift processing failed: {e}")
        await _send_complete(
            producer, task_id, user_id, model_id, None, dataset_id, dataset_version,
            success=False, error_message=str(e)
        )


async def retrain_vision_model_via_automl(
    *,
    task_id: str,
    user_id: str,
    dataset_id: str,
    dataset_version: str,
    filename_column: str,
    label_column: str,
    model_size: str,
    task_type: str = "image_classification",
    dataset_split: str | None = None,
    time_budget: int = VISION_RETRAIN_TIME_BUDGET_SECONDS,
) -> dict:
    normalized_task_type = (
        normalize_vision_task_type(
            task_type
        )
    )

    if normalized_task_type not in {
        "image_classification",
        "image_segmentation",
    }:
        raise ValueError(
            "Unsupported Vision AutoML retraining "
            f"task_type={task_type!r}"
        )

    data = {
        "user_id": user_id,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "filename_column": filename_column,
        "label_column": label_column,
        "task_type": normalized_task_type,
        "time_budget": time_budget,
        "model_size": model_size or "small",
        "num_cpus": "auto",
        "num_gpus": "auto",
    }

    if dataset_split:
        data["dataset_split"] = dataset_split

    headers = {
        "X-Task-ID": f"{task_id}-vision-retrain"
    }

    logger.info(
        "[Vision Retrain] Calling AutoML Vision: %s",
        AUTOML_VISION_BEST_MODEL_URL,
    )

    logger.info(
        "[Vision Retrain] "
        "task_type=%s dataset=%s version=%s "
        "filename_column=%s label_column=%s split=%s",
        normalized_task_type,
        dataset_id,
        dataset_version,
        filename_column,
        label_column,
        dataset_split,
    )

    def do_request():
        response = requests.post(
            AUTOML_VISION_BEST_MODEL_URL,
            data=data,
            headers=headers,
            timeout=(
                10,
                VISION_RETRAIN_READ_TIMEOUT_SECONDS,
            ),
        )

        if not response.ok:
            logger.error(
                "[Vision Retrain] HTTP %s: %s",
                response.status_code,
                response.text[:2000],
            )

        response.raise_for_status()

        return (
            response.json()
            if response.content
            else {}
        )

    # Don't block the asyncio event loop during training.
    response_data = await asyncio.to_thread(
        do_request
    )

    logger.info(
        "[Vision Retrain] AutoML completed: %s",
        json.dumps(
            response_data,
            default=str,
        ),
    )

    return response_data


async def _process_vision_drift(
    *,
    producer: AIOKafkaProducer,
    task_id: str,
    user_id: str,
    dataset_id: str,
    dataset_version: str,
    model_id: str,
    model_version: str,
    task_type: str,
    filename_column: str,
    label_column: str,
    mask_column: str,
    dataset_split: str,
    model_size: str,
    window_size: int,
    induce_drift: bool,
) -> None:

    logger.info(
        "[Vision Drift] Dispatching task_type=%s "
        "filename_column=%s label_column=%s mask_column=%s",
        task_type,
        filename_column,
        label_column,
        mask_column,
    )
    if task_type == "image_classification":

        await _process_image_classification_drift(
            producer=producer,
            task_id=task_id,
            user_id=user_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            model_id=model_id,
            model_version=model_version,
            task_type=task_type,
            filename_column=filename_column,
            label_column=label_column,
            dataset_split=dataset_split,
            model_size=model_size,
            window_size=window_size,
            induce_drift=induce_drift,
        )

        return

    if task_type == "image_segmentation":

        await _process_image_segmentation_drift(
            producer=producer,
            task_id=task_id,
            user_id=user_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            model_id=model_id,
            model_version=model_version,
            filename_column=filename_column,
            mask_column=mask_column,
            dataset_split=dataset_split,
            model_size=model_size,
            window_size=window_size,
            induce_drift=induce_drift,
        )

        return

    raise ValueError(
        f"Unsupported Vision task_type={task_type}"
    )


async def _process_image_classification_drift(
    *,
    producer: AIOKafkaProducer,
    task_id: str,
    user_id: str,
    dataset_id: str,
    dataset_version: str,
    model_id: str,
    model_version: str,
    task_type: str,
    filename_column: str,
    label_column: str,
    dataset_split: str,
    model_size: str,
    window_size: int,
    induce_drift: bool,
) -> None:

    try:
        metadata = fetch_dataset_metadata(
            user_id,
            dataset_id,
            dataset_version,
        )

        logger.info(
            "[Vision Drift] Dataset metadata: %s",
            json.dumps(
                metadata,
                default=str,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:

            dataset_dir = os.path.join(
                tmp,
                "dataset",
            )

            model_dir = os.path.join(
                tmp,
                "model",
            )

            os.makedirs(
                dataset_dir,
                exist_ok=True,
            )

            os.makedirs(
                model_dir,
                exist_ok=True,
            )

            # --------------------------------------------------
            # 1. Download and prepare image dataset
            # --------------------------------------------------

            dataset_bytes = download_dataset_bytes(
                user_id=user_id,
                dataset_id=dataset_id,
                version=dataset_version,
                split=dataset_split,
            )

            extracted = extract_vision_dataset(
                dataset_bytes,
                dataset_dir,
            )

            logger.info(
                "[Vision Drift] Extracted %d files",
                len(extracted),
            )

            manifest_path = find_image_manifest(
                dataset_root=dataset_dir,
                filename_column=filename_column,
                label_column=label_column,
            )

            manifest = pd.read_csv(
                manifest_path
            )

            manifest = resolve_image_paths(
                manifest_df=manifest,
                dataset_root=dataset_dir,
                filename_column=filename_column,
                label_column=label_column,
            )

            logger.info(
                "[Vision Drift] Loaded %d labeled images",
                len(manifest),
            )

            if induce_drift:
                # Synthetic tests should not depend on the original
                # class ordering of the uploaded manifest.
                manifest = manifest.sample(
                    frac=1.0,
                    random_state=42,
                ).reset_index(
                    drop=True
                )

                logger.info(
                    "[Vision Drift] Manifest shuffled "
                    "before synthetic drift"
                )

                logger.info(
                    "[Vision Drift] Synthetic image drift enabled"
                )

                apply_synthetic_image_drift(
                    manifest,
                    start_fraction=0.5,
                )

                logger.info(
                    "[Vision Drift] Synthetic image drift applied"
                )

            # --------------------------------------------------
            # 2. Download trained model
            # --------------------------------------------------

            logger.info(
                "[Vision Drift] Downloading model "
                "user=%s model=%s version=%s",
                user_id,
                model_id,
                model_version,
            )

            download_model_folder(
                user_id,
                model_id,
                model_version,
                model_dir,
            )

            # --------------------------------------------------
            # 3. Load Vision model adapter
            # --------------------------------------------------

            logger.info(
                "[Vision Drift] Loading Vision model adapter"
            )

            adapter = VisionModelAdapter.load(
                model_dir
            )

            logger.info(
                "[Vision Drift] Vision model loaded successfully"
            )

            # --------------------------------------------------
            # 4. Run ADWIN monitoring
            # --------------------------------------------------

            logger.info(
                "[Vision Drift] Starting ADWIN monitoring "
                "(requested window_size=%d; ADWIN uses adaptive windows)",
                window_size,
            )

            result = monitor_image_drift(
                manifest_df=manifest,
                adapter=adapter,
                filename_column=filename_column,
                label_column=label_column,
                window_size=window_size,
                adwin_delta=VISION_ADWIN_DELTA,
            )

            # --------------------------------------------------
            # 5. Calculate classification metrics
            # --------------------------------------------------

            metrics = calculate_vision_metrics(
                result["y_true"],
                result["y_pred"],
            )

            logger.info(
                "[Vision Drift] Drift=%s drift_index=%s "
                "accuracy=%.4f precision=%.4f "
                "recall=%.4f f1=%.4f",
                result["drift_detected"],
                result["drift_index"],
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
            )

            # Add useful monitoring information.
            metrics["problem_type"] = "image_classification"
            metrics["drift_index"] = result["drift_index"]
            metrics["samples_processed"] = len(
                result["y_true"]
            )
            metrics["total_images"] = len(
                manifest
            )

            metrics["synthetic_drift"] = induce_drift
            metrics["synthetic_drift_start_index"] = (
                int(len(manifest) * 0.5)
                if induce_drift
                else None
            )


            if result["errors"]:
                metrics["mean_error"] = float(
                    np.mean(result["errors"])
                )
            else:
                metrics["mean_error"] = 0.0

            # ==================================================
            # 6. NO DRIFT -> no retraining
            # ==================================================

            if not result["drift_detected"]:

                logger.info(
                    "[Vision Drift] No concept drift detected; "
                    "Vision retraining is not required"
                )

                drift_report = {
                    "task_id": task_id,
                    "timestamp": datetime.now(
                        tz=timezone.utc
                    ).isoformat(),
                    "task_category": "vision",
                    "task_type": "image_classification",
                    "drift_detected": False,
                    "drift_metrics": metrics,
                    "source_model": {
                        "model_id": model_id,
                        "model_version": model_version,
                    },
                    "replacement_model": None,
                    "message": "No concept drift detected",
                }

                try:
                    saved = post_concept_drift_report(
                        user_id=user_id,
                        dataset_id=dataset_id,
                        dataset_version=dataset_version,
                        model_id=model_id,
                        model_version=model_version,
                        report=drift_report,
                    )

                    logger.info(
                        "[Vision Drift] No-drift report saved: id=%s",
                        saved.get("id", ""),
                    )

                except Exception as report_error:
                    logger.warning(
                        "[Vision Drift] Failed to save no-drift report: %s",
                        report_error,
                    )

                await _send_complete(
                    producer,
                    task_id,
                    user_id,
                    model_id,
                    model_version,
                    dataset_id,
                    dataset_version,
                    success=True,
                    drift_metrics=metrics,
                    drift_detected=False,
                )

                return

            # ==================================================
            # 7. DRIFT DETECTED
            # ==================================================

            logger.info(
                "[Vision Retrain] Drift detected at index %s; "
                "starting retraining workflow",
                result["drift_index"],
            )
            if induce_drift:

                logger.info(
                    "[Vision Retrain] Synthetic drift was used; "
                    "creating separate retraining and holdout sets"
                )

                # Synthetic drift starts halfway through the stream.
                synthetic_drift_start = int(
                    len(manifest) * 0.5
                )

                (
                    retraining_manifest,
                    validation_manifest,
                    validation_start,
                ) = split_vision_retraining_and_holdout(
                    manifest_df=manifest,
                    drift_start_index=synthetic_drift_start,
                    holdout_fraction=1.0 / 3.0,
                )

                if retraining_manifest.empty:
                    raise RuntimeError(
                        "Vision retraining manifest is empty"
                    )

                if validation_manifest.empty:
                    raise RuntimeError(
                        "Vision holdout manifest is empty"
                    )

                logger.info(
                    "[Vision Retrain] Holdout split prepared: "
                    "total=%d drift_start=%d "
                    "retraining=%d validation=%d "
                    "validation_start=%d",
                    len(manifest),
                    synthetic_drift_start,
                    len(retraining_manifest),
                    len(validation_manifest),
                    validation_start,
                )

                logger.info(
                    "[Vision Retrain] Holdout class counts: %s",
                    validation_manifest[
                        label_column
                    ].astype(str).value_counts().to_dict(),
                )

                retrain_dataset_dir = os.path.join(
                    tmp,
                    "retrain_dataset",
                )

                build_vision_retraining_subset(
                    manifest_df=retraining_manifest,
                    output_dir=retrain_dataset_dir,
                    filename_column=filename_column,
                    label_column=label_column,
                )

                logger.info(
                    "[Vision Retrain] Uploading retraining subset "
                    "to AutoDW"
                )

                retrain_dataset_metadata = (
                    await asyncio.to_thread(
                        upload_vision_retraining_dataset,
                        user_id=user_id,
                        source_dataset_id=dataset_id,
                        task_id=task_id,
                        dataset_dir=retrain_dataset_dir,
                    )
                )

                retrain_dataset_id = (
                    retrain_dataset_metadata.get(
                        "dataset_id"
                    )
                )

                retrain_dataset_version = (
                    retrain_dataset_metadata.get(
                        "version"
                    )
                )

                # Validate identifiers BEFORE calling AutoDW again.
                if not retrain_dataset_id:
                    raise RuntimeError(
                        "Derived dataset upload did not "
                        "return dataset_id"
                    )

                if not retrain_dataset_version:
                    raise RuntimeError(
                        "Derived dataset upload did not "
                        "return version"
                    )

                logger.info(
                    "[Vision Retrain] Verifying derived dataset "
                    "exists in AutoDW before AutoML call"
                )

                verified_metadata = await asyncio.to_thread(
                    fetch_dataset_metadata,
                    user_id,
                    retrain_dataset_id,
                    retrain_dataset_version,
                )

                logger.info(
                    "[Vision Retrain] Derived dataset verified: %s",
                    json.dumps(
                        verified_metadata,
                        default=str,
                    ),
                )

                # The new ZIP itself contains the complete prepared
                # dataset, so do not request the original drift split.
                retrain_dataset_split = None

            else:

                logger.info(
                    "[Vision Retrain] Real drift detected; "
                    "creating separate retraining and holdout sets"
                )

                drift_start_index = (
                    result["drift_index"]
                    if result["drift_index"] is not None
                    else 0
                )

                (
                    retraining_manifest,
                    validation_manifest,
                    validation_start,
                ) = split_vision_retraining_and_holdout(
                    manifest_df=manifest,
                    drift_start_index=drift_start_index,
                    holdout_fraction=1.0 / 3.0,
                )

                logger.info(
                    "[Vision Retrain] Real-drift holdout prepared: "
                    "total=%d drift_start=%d "
                    "retraining=%d validation=%d "
                    "validation_start=%d",
                    len(manifest),
                    drift_start_index,
                    len(retraining_manifest),
                    len(validation_manifest),
                    validation_start,
                )

                logger.info(
                    "[Vision Retrain] Real-drift holdout class counts: %s",
                    validation_manifest[
                        label_column
                    ].astype(str).value_counts().to_dict(),
                )

                retrain_dataset_dir = os.path.join(
                    tmp,
                    "retrain_dataset",
                )

                build_vision_retraining_subset(
                    manifest_df=retraining_manifest,
                    output_dir=retrain_dataset_dir,
                    filename_column=filename_column,
                    label_column=label_column,
                )

                logger.info(
                    "[Vision Retrain] Uploading real-drift "
                    "retraining subset to AutoDW"
                )

                retrain_dataset_metadata = (
                    await asyncio.to_thread(
                        upload_vision_retraining_dataset,
                        user_id=user_id,
                        source_dataset_id=dataset_id,
                        task_id=task_id,
                        dataset_dir=retrain_dataset_dir,
                    )
                )

                retrain_dataset_id = (
                    retrain_dataset_metadata.get(
                        "dataset_id"
                    )
                )

                retrain_dataset_version = (
                    retrain_dataset_metadata.get(
                        "version"
                    )
                )

                if not retrain_dataset_id:
                    raise RuntimeError(
                        "Real-drift retraining dataset upload "
                        "did not return dataset_id"
                    )

                if not retrain_dataset_version:
                    raise RuntimeError(
                        "Real-drift retraining dataset upload "
                        "did not return version"
                    )

                verified_metadata = await asyncio.to_thread(
                    fetch_dataset_metadata,
                    user_id,
                    retrain_dataset_id,
                    retrain_dataset_version,
                )

                logger.info(
                    "[Vision Retrain] Real-drift derived dataset verified: %s",
                    json.dumps(
                        verified_metadata,
                        default=str,
                    ),
                )

                # This is now a standalone uploaded dataset,
                # not the source dataset's drift split.
                retrain_dataset_split = None

            logger.info(
                "[Vision Retrain] Retraining dataset selected: "
                "dataset=%s version=%s split=%s",
                retrain_dataset_id,
                retrain_dataset_version,
                retrain_dataset_split,
            )
            # ==================================================
            # 8. CALL AUTOML VISION RETRAINING
            # ==================================================

            logger.info(
                "[Vision Retrain] About to call AutoML Vision"
            )

            logger.info(
                "[Vision Retrain] PRE-AUTOML CHECK PASSED: "
                "user=%s dataset=%s version=%s",
                user_id,
                retrain_dataset_id,
                retrain_dataset_version,
            )

            retrain_response = (
                await retrain_vision_model_via_automl(
                    task_id=task_id,
                    user_id=user_id,
                    dataset_id=retrain_dataset_id,
                    dataset_version=retrain_dataset_version,
                    filename_column=filename_column,
                    label_column=label_column,
                    model_size=model_size,
                    dataset_split=retrain_dataset_split,
                )
            )

            logger.info(
                "[Vision Retrain] AutoML response received: %s",
                json.dumps(
                    retrain_response,
                    default=str,
                ),
            )

            new_model_id = retrain_response.get(
                "model_id"
            )

            new_model_version = retrain_response.get(
                "version"
            )

            if not new_model_id:
                raise RuntimeError(
                    "Vision AutoML completed but did not "
                    "return model_id"
                )

            if not new_model_version:
                raise RuntimeError(
                    "Vision AutoML completed but did not "
                    "return version"
                )

            logger.info(
                "[Vision Retrain] Replacement model created: "
                "model=%s version=%s",
                new_model_id,
                new_model_version,
            )

            # ==================================================
            # 9. DOWNLOAD AND LOAD REPLACEMENT MODEL
            # ==================================================

            retrained_model_dir = os.path.join(
                tmp,
                "retrained_model",
            )

            os.makedirs(
                retrained_model_dir,
                exist_ok=True,
            )

            logger.info(
                "[Vision Retrain] Downloading replacement model "
                "model=%s version=%s",
                new_model_id,
                new_model_version,
            )

            download_model_folder(
                user_id,
                new_model_id,
                new_model_version,
                retrained_model_dir,
            )

            logger.info(
                "[Vision Retrain] Loading replacement model"
            )

            new_adapter = VisionModelAdapter.load(
                retrained_model_dir
            )

            logger.info(
                "[Vision Retrain] Replacement model loaded successfully"
            )

            # ==================================================
            # 10. VALIDATE REPLACEMENT MODEL
            # ==================================================

            logger.info(
                "[Vision Retrain] Using reserved %s holdout "
                "for model comparison",
                (
                    "synthetic"
                    if induce_drift
                    else "real-drift"
                ),
            )

            if validation_manifest.empty:
                raise RuntimeError(
                    "Vision replacement validation set is empty"
                )

            logger.info(
                "[Vision Retrain] Validation range: "
                "start=%d total_manifest=%d validation_images=%d",
                validation_start,
                len(manifest),
                len(validation_manifest),
            )

            logger.info(
                "[Vision Retrain] Validating replacement model "
                "on %d images starting from index %d",
                len(validation_manifest),
                validation_start,
            )

            new_y_true = []
            new_y_pred = []

            total_validation = len(
                validation_manifest
            )

            for position, (_, row) in enumerate(
                    validation_manifest.iterrows(),
                    start=1,
            ):

                image_path = str(
                    row["_image_path"]
                )

                true_label = str(
                    row[label_column]
                )

                predicted_label = new_adapter.predict(
                    [image_path]
                )[0]

                new_y_true.append(
                    true_label
                )

                new_y_pred.append(
                    predicted_label
                )

                if (
                        position == 1
                        or position % 25 == 0
                        or position == total_validation
                ):
                    logger.info(
                        "[Vision Retrain] Validation %d/%d "
                        "true=%s predicted=%s",
                        position,
                        total_validation,
                        true_label,
                        predicted_label,
                    )

            retrained_metrics = (
                calculate_vision_metrics(
                    new_y_true,
                    new_y_pred,
                )
            )

            logger.info(
                "[Vision Retrain] Replacement metrics: "
                "accuracy=%.4f precision=%.4f "
                "recall=%.4f f1=%.4f",
                retrained_metrics["accuracy"],
                retrained_metrics["precision"],
                retrained_metrics["recall"],
                retrained_metrics["f1"],
            )

            # ==================================================
            # 10A. EVALUATE SOURCE MODEL ON SAME VALIDATION SET
            # ==================================================

            old_validation_y_true = []
            old_validation_y_pred = []

            logger.info(
                "[Vision Retrain] Evaluating source model "
                "on same %d validation images",
                len(validation_manifest),
            )

            for position, (_, row) in enumerate(
                    validation_manifest.iterrows(),
                    start=1,
            ):
                image_path = str(
                    row["_image_path"]
                )

                true_label = str(
                    row[label_column]
                )

                predicted_label = adapter.predict(
                    [image_path]
                )[0]

                old_validation_y_true.append(
                    true_label
                )

                old_validation_y_pred.append(
                    predicted_label
                )

                if (
                        position == 1
                        or position % 25 == 0
                        or position == total_validation
                ):
                    logger.info(
                        "[Vision Retrain] Source validation %d/%d "
                        "true=%s predicted=%s",
                        position,
                        total_validation,
                        true_label,
                        predicted_label,
                    )

            source_validation_metrics = (
                calculate_vision_metrics(
                    old_validation_y_true,
                    old_validation_y_pred,
                )
            )

            logger.info(
                "[Vision Retrain] Source model validation metrics: "
                "accuracy=%.4f precision=%.4f "
                "recall=%.4f f1=%.4f",
                source_validation_metrics["accuracy"],
                source_validation_metrics["precision"],
                source_validation_metrics["recall"],
                source_validation_metrics["f1"],
            )

            logger.info(
                "[Vision Retrain] Model comparison: "
                "source_accuracy=%.4f replacement_accuracy=%.4f "
                "source_f1=%.4f replacement_f1=%.4f",
                source_validation_metrics["accuracy"],
                retrained_metrics["accuracy"],
                source_validation_metrics["f1"],
                retrained_metrics["f1"],
            )
            accuracy_improvement = (
                    retrained_metrics["accuracy"]
                    - source_validation_metrics["accuracy"]
            )

            f1_improvement = (
                    retrained_metrics["f1"]
                    - source_validation_metrics["f1"]
            )

            replacement_accepted = (
                    accuracy_improvement
                    >= VISION_MIN_ACCURACY_IMPROVEMENT
                    and
                    f1_improvement
                    >= VISION_MIN_F1_IMPROVEMENT
            )

            logger.info(
                "[Vision Retrain] Replacement accepted=%s "
                "accuracy_improvement=%+.4f required=%+.4f "
                "f1_improvement=%+.4f required=%+.4f",
                replacement_accepted,
                accuracy_improvement,
                VISION_MIN_ACCURACY_IMPROVEMENT,
                f1_improvement,
                VISION_MIN_F1_IMPROVEMENT,
            )
            if replacement_accepted:
                completion_model_id = new_model_id
                completion_model_version = new_model_version
            else:
                completion_model_id = model_id
                completion_model_version = model_version

            logger.info(
                "[Vision Retrain] Selected model for completion: "
                "model=%s version=%s replacement_accepted=%s",
                completion_model_id,
                completion_model_version,
                replacement_accepted,
            )

            # ==================================================
            # 11. BUILD FINAL DRIFT/RETRAINING REPORT
            # ==================================================

            completion_metrics = dict(
                metrics
            )

            # Performance of the original/source model on the
            # exact same validation images used for the candidate.
            completion_metrics[
                "source_validation_metrics"
            ] = source_validation_metrics

            # Performance of the newly retrained candidate.
            completion_metrics[
                "retrained_model_metrics"
            ] = retrained_metrics

            # Record whether the candidate passed the acceptance gate.
            completion_metrics[
                "replacement_accepted"
            ] = replacement_accepted

            completion_metrics[
                "accuracy_improvement"
            ] = accuracy_improvement

            completion_metrics[
                "f1_improvement"
            ] = f1_improvement

            completion_metrics[
                "min_accuracy_improvement_required"
            ] = VISION_MIN_ACCURACY_IMPROVEMENT

            completion_metrics[
                "min_f1_improvement_required"
            ] = VISION_MIN_F1_IMPROVEMENT

            # Candidate replacement model produced by AutoML.
            completion_metrics[
                "replacement_model_id"
            ] = new_model_id

            completion_metrics[
                "replacement_model_version"
            ] = new_model_version

            # Model actually selected for downstream processing.
            completion_metrics[
                "selected_model_id"
            ] = completion_model_id

            completion_metrics[
                "selected_model_version"
            ] = completion_model_version

            completion_metrics[
                "validation_images"
            ] = len(
                validation_manifest
            )

            completion_metrics[
                "validation_overlaps_retraining_dataset"
            ] = False

            completion_metrics[
                "retraining_images"
            ] = len(
                retraining_manifest
            )

            # Build a message that accurately describes the result.
            if replacement_accepted:
                report_message = (
                    "Concept drift detected; Vision model retrained "
                    "and replacement model accepted"
                )
            else:
                report_message = (
                    "Concept drift detected; Vision model retrained "
                    "but replacement model was rejected. "
                    "Source model retained pending further retraining"
                )

            drift_report = {
                "task_id": task_id,
                "timestamp": datetime.now(
                    tz=timezone.utc
                ).isoformat(),
                "task_category": "vision",
                "task_type": "image_classification",

                "drift_detected": True,

                "source_model": {
                    "model_id": model_id,
                    "model_version": model_version,
                },

                # This is the candidate produced by AutoML,
                # whether accepted or rejected.
                "replacement_model": {
                    "model_id": new_model_id,
                    "model_version": new_model_version,
                },

                # This is what downstream processing should use.
                "selected_model": {
                    "model_id": completion_model_id,
                    "model_version": completion_model_version,
                },

                "source_dataset": {
                    "dataset_id": dataset_id,
                    "dataset_version": dataset_version,
                },

                "retraining_dataset": {
                    "dataset_id": retrain_dataset_id,
                    "dataset_version": retrain_dataset_version,
                },
                "validation": {
                    "strategy": (
                        "reserved_synthetic_post_drift_holdout"
                        if induce_drift
                        else "reserved_real_post_drift_holdout"
                    ),

                    "validation_images": len(
                        validation_manifest
                    ),

                    "validation_start_index": (
                        validation_start
                    ),

                    "overlaps_retraining_dataset": False,
                },

                # Metrics observed while detecting drift.
                "drift_metrics": metrics,

                # Old and new model evaluated on the same images.
                "source_validation_metrics": (
                    source_validation_metrics
                ),

                "retrained_model_metrics": (
                    retrained_metrics
                ),

                "replacement_accepted": (
                    replacement_accepted
                ),

                "model_comparison": {
                    "accuracy_improvement": (
                        accuracy_improvement
                    ),
                    "f1_improvement": (
                        f1_improvement
                    ),
                    "min_accuracy_improvement_required": (
                        VISION_MIN_ACCURACY_IMPROVEMENT
                    ),
                    "min_f1_improvement_required": (
                        VISION_MIN_F1_IMPROVEMENT
                    ),
                },

                "message": report_message,
            }
            drift_report[
                "validation"
            ][
                "retraining_images"
            ] = len(
                retraining_manifest
            )

            # ==================================================
            # 12. SAVE FINAL CONCEPT-DRIFT REPORT
            # ==================================================

            try:
                saved = post_concept_drift_report(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    dataset_version=dataset_version,

                    # Associate the drift report with the model
                    # on which drift was originally detected.
                    model_id=model_id,
                    model_version=model_version,

                    report=drift_report,
                )

                logger.info(
                    "[Vision Retrain] Final drift report saved: id=%s",
                    saved.get("id", ""),
                )

            except Exception as report_error:
                logger.warning(
                    "[Vision Retrain] Failed to save "
                    "final drift report: %s",
                    report_error,
                )

            # ==================================================
            # 13. SEND COMPLETION EVENT
            # ==================================================

            logger.info(
                "[Vision Retrain] Sending completion with "
                "selected model=%s version=%s",
                completion_model_id,
                completion_model_version,
            )

            await _send_complete(
                producer,
                task_id,
                user_id,

                # IMPORTANT:
                # Send the model selected by the acceptance gate.
                completion_model_id,
                completion_model_version,

                # Preserve original dataset context.
                dataset_id,
                dataset_version,

                success=True,
                drift_metrics=completion_metrics,
                drift_detected=True,
            )

            logger.info(
                "[Vision Retrain] Retraining workflow "
                "completed successfully; "
                "replacement_accepted=%s selected_model=%s version=%s",
                replacement_accepted,
                completion_model_id,
                completion_model_version,
            )

            return


    except Exception as exc:
        logger.exception(
            "[Vision Drift] Processing failed: %s",
            exc,
        )

        drift_was_detected = False

        if(
            "result" in locals()and isinstance(result, dict)
        ):
            drift_was_detected = bool(
                result.get(
                    "drift_detected",
                    False,
                )
            )

        failure_report = {
            "task_id": task_id,
            "timestamp": datetime.now(
                tz=timezone.utc
            ).isoformat(),
            "task_category": "vision",
            "task_type": "image_classification",
            "drift_detected": (
                drift_was_detected
            ),
            "retraining_success": False,
            "source_model": {
                "model_id": model_id,
                "model_version": model_version,
            },
            "source_dataset": {
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
            },

            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },

            "message": (
                "Vision concept-drift processing failed"
            ),
        }

        try:
            saved = post_concept_drift_report(
                user_id=user_id,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                model_id=model_id,
                model_version=model_version,
                report=failure_report,
            )

            logger.info(
                "[Vision Drift] Failure report saved: id=%s",
                saved.get("id", ""),
            )

        except Exception as report_error:
            logger.warning(
                "[Vision Drift] Could not persist "
                "failure report: %s",
                report_error,
            )

        await _send_complete(
            producer,
            task_id,
            user_id,
            # Preserve source-model identity on failure.
            model_id,
            model_version,
            dataset_id,
            dataset_version,
            success=False,
            error_message=str(exc),
        )


async def _send_complete(
    producer: AIOKafkaProducer,
    task_id: str,
    user_id: str,
    model_id: str,
    model_version: str,
    dataset_id: str,
    dataset_version: str,
    success: bool,
    error_message: str = None,
    drift_metrics: dict = None,
    drift_detected: bool = None,
) -> None:
    payload = {
        "task_id": task_id,
        "event_type": "concept-drift-complete",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    if success:
        payload["output"] = {
            "model_id": model_id,
            "model_version": model_version,
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "user_id": user_id,
        }
        if drift_detected is not None:
            payload["output"]["drift_detected"] = drift_detected
            if not drift_detected:
                payload["output"]["message"] = "No concept drift detected"
        if drift_metrics:
            payload["output"]["drift_metrics"] = drift_metrics
        payload["failure"] = None
    else:
        # Include context so orchestrators can still proceed to XAI / next step
        payload["output"] = {
            "user_id": user_id,
            "dataset_id": dataset_id,
            "model_id": model_id,
            "model_version": model_version,
            "dataset_version": dataset_version,
        }
        payload["failure"] = {"error_type": "ConceptDriftError", "error_message": error_message or "Concept drift failed"}
    await producer.send_and_wait(KAFKA_DRIFT_COMPLETE_TOPIC, value=payload, key=task_id)
    logger.info(f"Sent concept-drift-complete task_id={task_id} success={success}")


async def run_consumer() -> None:
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        key_serializer=lambda k: (k or "").encode("utf-8"),
    )
    await producer.start()
    consumer = AIOKafkaConsumer(
        KAFKA_DRIFT_TRIGGER_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_CONSUMER_GROUP,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        max_poll_interval_ms=int(os.getenv("KAFKA_MAX_POLL_INTERVAL_MS", str(4 * 60 * 60 * 1000))),
        session_timeout_ms=int(os.getenv("KAFKA_SESSION_TIMEOUT_MS", "30000")),
        heartbeat_interval_ms=int(os.getenv("KAFKA_HEARTBEAT_INTERVAL_MS", "10000")),
        max_poll_records=int(os.getenv("KAFKA_MAX_POLL_RECORDS", "1")),
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        key_deserializer=lambda m: m.decode("utf-8") if m else None,
    )
    await consumer.start()
    logger.info(
        "Listening to %s (group=%s, worker_id=%s); completions -> %s",
        KAFKA_DRIFT_TRIGGER_TOPIC,
        KAFKA_CONSUMER_GROUP,
        WORKER_ID,
        KAFKA_DRIFT_COMPLETE_TOPIC,
    )
    try:
        async for msg in consumer:

            logger.info(
                "Concept drift trigger received: %s",
                json.dumps(
                    msg.value,
                    indent=2,
                ),
            )

            try:
                await process_drift_trigger(
                    msg.value,
                    producer,
                )

            except Exception as exc:
                logger.exception(
                    "Unhandled concept-drift task failure; "
                    "consumer will continue listening: %s",
                    exc,
                )
    finally:
        await consumer.stop()
        await producer.stop()


if __name__ == "__main__":
    try:
        asyncio.run(run_consumer())
    except KeyboardInterrupt:
        logger.info("Interrupted")
