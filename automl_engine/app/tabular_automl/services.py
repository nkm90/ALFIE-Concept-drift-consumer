"""Service layer for tabular AutoML workflows.

Wraps the AutoGluon trainer with validation, leaderboard conversion, artifact
packaging, and the instruction templates, and re-exports the shared AutoDW
helpers so the orchestrator has one import surface.
"""

import json
import logging
import os
import pickle
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from app.core.exceptions import (
    AutoMLConfigError,
    AutoMLDataError,
    AutoMLRuntimeError,
    AutoMLSerializationError,
    AutoMLValidationError,
)
from app.core.service_helpers import (  # noqa: F401 – re-exported for router
    build_upload_payload as _core_build_upload_payload,
    download_dataset as _core_download_dataset,
    fetch_dataset_metadata,
    resolve_download_url,
    upload_model,
)
from app.core.utils import jinja_environment, render_template
from app.ml_engine.model import SUPPORTED_TABULAR_TASK_TYPES
from app.ml_engine.trainer import AutoGluonTrainer

logger = logging.getLogger(__name__)


UPLOAD_ROOT = Path("uploaded_data")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

SUPPORTED_FILE_TYPES = {"csv", "tsv", "parquet"}

_TABLE_MEMBER_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".pq", ".json"}


def _first_table_member(member_names: list[str]) -> str | None:
    """Return the first ZIP member that looks like a tabular file, if any."""
    for name in member_names:
        if name.endswith("/"):
            continue
        if "__MACOSX" in name:
            continue
        base = name.rsplit("/", 1)[-1]
        if base.startswith("."):
            continue
        if Path(name).suffix.lower() in _TABLE_MEMBER_SUFFIXES:
            return name
    return None


def _load_table_by_suffix(suffix: str, source: Path | BytesIO) -> pd.DataFrame:
    """Load a table from a path or file-like object by its suffix."""
    if suffix in [".csv"]:
        logging.debug("csv file loaded")
        return pd.read_csv(source)
    if suffix in [".tsv"]:
        logging.debug("tsv file loaded")
        return pd.read_csv(source, sep="\t")
    if suffix in [".xlsx"]:
        logging.debug("excel file loaded")
        return pd.read_excel(source)
    if suffix in [".parquet", ".pq"]:
        logging.debug("Parquet file loaded")
        return pd.read_parquet(source)
    if suffix in [".json"]:
        logging.debug("Json file loaded")
        return pd.read_json(source)
    # Fallback: try csv to keep previous behavior
    return pd.read_csv(source)


def _load_table_from_zip(file_path: Path) -> pd.DataFrame:
    """Load the first tabular file inside a ZIP archive.

    AutoDW returns split downloads (and some folder datasets) as ZIP bytes even
    when metadata reports ``file_type=csv`` / a ``.csv`` ``original_filename``.
    Detect that case by content, not by extension.
    """
    with zipfile.ZipFile(file_path) as archive:
        member = _first_table_member(archive.namelist())
        if member is None:
            raise AutoMLDataError(
                f"ZIP archive contains no tabular files (.csv/.tsv/.parquet/...): {file_path}"
            )
        logger.info(
            "Loading tabular member '%s' from ZIP archive %s", member, file_path
        )
        suffix = Path(member).suffix.lower()
        with archive.open(member) as handle:
            payload = BytesIO(handle.read())
        return _load_table_by_suffix(suffix, payload)


def load_table(file_path: Path) -> pd.DataFrame:
    """Load a table file into a DataFrame based on file extension.

    Also accepts ZIP archives that wrap a single tabular file (AutoDW split
    downloads), regardless of whether the on-disk name ends in ``.zip``.
    """
    if not file_path.exists():
        raise AutoMLDataError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise AutoMLDataError(f"Path is not a file: {file_path}")

    try:
        # OOXML formats (.xlsx/…) are ZIP containers themselves — only sniff
        # for a wrapped table when the name does not already say spreadsheet.
        suffix = file_path.suffix.lower()
        if suffix not in (".xlsx", ".xlsm", ".xlsb") and zipfile.is_zipfile(file_path):
            return _load_table_from_zip(file_path)
        return _load_table_by_suffix(suffix, file_path)
    except pd.errors.EmptyDataError:
        logging.error(f"File is empty: {file_path}")
        raise AutoMLDataError(f"File is empty: {file_path}") from None
    except pd.errors.ParserError as e:
        logging.error(f"Failed to parse file {file_path}: {e}")
        raise AutoMLDataError(f"Failed to parse file: {e}") from e
    except AutoMLDataError:
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading table from {file_path}: {e}")
        raise AutoMLRuntimeError(f"Failed to load table: {e}") from e


def validate_tabular_inputs(
    train_path: Path,
    target_column_name: str,
    time_stamp_column_name: str | None = None,
    task_type: str = "tabular_classification",
) -> str | None:
    """Validate required columns and task type for tabular training."""

    if not train_path.exists():
        logger.error(f"Training file not found: {train_path}")
        return f"Training file not found: {train_path}"

    if not target_column_name or not isinstance(target_column_name, str):
        logger.error("target_column_name must be a non-empty string")
        return "target_column_name must be a non-empty string"

    if task_type not in SUPPORTED_TABULAR_TASK_TYPES:
        logger.error(f"Invalid task type {task_type}")
        return f"Invalid task_type '{task_type}'. Must be one of: {SUPPORTED_TABULAR_TASK_TYPES}"

    try:
        train_df = load_table(train_path)
    except AutoMLDataError as e:
        logging.error(f"Training file not found: {e}")
        return f"Training file not found: {e}"
    except Exception as e:
        logging.error(f"Unexpected error reading training data: {e}")
        return f"Unexpected error reading training data: {e}"

    if train_df.empty:
        logger.error("Training dataframe is empty")
        return "Training dataframe is empty"

    if target_column_name not in train_df.columns:
        available_columns = ", ".join(train_df.columns.tolist())
        logger.error(
            f"Target column '{target_column_name}' not found. Available columns: {available_columns}"
        )
        return f"Target column '{target_column_name}' not found. Available columns: {available_columns}"

    if time_stamp_column_name and time_stamp_column_name not in train_df.columns:
        available_columns = ", ".join(train_df.columns.tolist())
        logger.error(
            f"Timestamp column '{time_stamp_column_name}' not found. Available columns: {available_columns}"
        )
        return f"Timestamp column '{time_stamp_column_name}' not found. Available columns: {available_columns}"

    return None


def convert_leaderboard_safely(leaderboard) -> tuple[list | dict, str]:
    """Turn a leaderboard into a JSON-ready structure plus a markdown string.

    DataFrames become a list of row dicts and a markdown table; anything else
    falls back to its plain string form.
    """
    if isinstance(leaderboard, pd.DataFrame):
        leaderboard_json = leaderboard.to_dict(orient="records")
        leaderboard_str = str(leaderboard.to_markdown())
    else:
        leaderboard_json = {"result": str(leaderboard)}
        leaderboard_str = str(leaderboard)
    return leaderboard_json, leaderboard_str


def download_dataset(download_url: str, dest_path: Path) -> None:
    """Stream-download a dataset file to dest_path."""
    _core_download_dataset(download_url, dest_path)


def train_automl(
    dataset_path: Path,
    save_model_path: Path,
    target_column_name: str,
    task_type: str,
    time_budget: int,
    num_cpus: int | str = "auto",
    num_gpus: int | str = "auto",
):
    """Train an AutoML model and return (leaderboard, predictor)."""

    if not target_column_name or not isinstance(target_column_name, str):
        raise AutoMLDataError("target_column_name must be a non-empty string")

    if not isinstance(time_budget, int) or time_budget <= 0:
        raise AutoMLConfigError("time_budget must be a positive integer")

    try:
        os.makedirs(save_model_path, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create model directory {save_model_path}: {e}")
        raise AutoMLRuntimeError(f"Failed to create model directory: {e}") from e

    try:
        trainer = AutoGluonTrainer(save_model_path=save_model_path)
    except AutoMLConfigError as e:
        logger.error(f"Failed to initialize AutoML trainer: {e}")
        raise AutoMLRuntimeError(f"Failed to initialize trainer: {e}") from e

    try:
        train_df = load_table(dataset_path)
    except AutoMLDataError as e:
        logger.error(f"Failed to load training data: {e}")
        raise AutoMLRuntimeError(f"Failed to load training data: {e}") from e

    try:
        return trainer.train(
            train_df=train_df,
            test_df=None,
            target_column=target_column_name,
            time_limit=int(time_budget),
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )
    except AutoMLValidationError as e:
        logger.error(f"Training validation error: {e}")
        raise
    except AutoMLRuntimeError as e:
        logger.error(f"Training runtime error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        raise AutoMLRuntimeError(f"Unexpected error during training: {e}") from e


def deployment_instructions() -> str:
    """Return the deployment instructions from a markdown file"""
    if jinja_environment is not None:
        try:
            return render_template(
                jinja_environment, "tabular_deployment_instructions.md"
            )
        except Exception as e:
            logger.error(f"Failed to render deployment instructions: {e}")
            return "No instructions available"
    else:
        logger.warning("jinja_environment is None, returning default instructions")
        return "No instructions found"


def tabular_data_instructions() -> str:
    """Return the instructions from what kind of data is accepted by the tabular AutoML engine"""
    if jinja_environment is not None:
        try:
            return render_template(jinja_environment, "tabular_accepted_format.md")
        except Exception as e:
            logger.error(f"Failed to render accepted format instructions: {e}")
            return "No accepted format instructions available"
    else:
        logger.warning("jinja_environment is None, returning default formats")
        return "Ask the agent for help"


_MULTIMODAL_TOKENIZER_MARKERS = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "vocab.json",
        "spiece.model",
        "special_tokens_map.json",
    }
)


def extract_text_feature_mapping(predictor, save_model_path: Path) -> dict:
    """Extract text-feature mappings from a fitted AutoGluon ``TabularPredictor``.

    Captures both text-handling paths AutoGluon may take:

      * ``ngram_features``: word/n-gram -> column-index vocabulary pulled from any
        ``TextNgramFeatureGenerator`` in the default tabular feature pipeline.
      * ``multimodal_text_model_dirs``: relative paths under ``save_model_path``
        that contain transformer tokenizer artifacts (``tokenizer.json`` etc.),
        indicating deep-embedding text models. The tokenizer vocab files for
        those models are already part of the saved artifacts and ship inside
        the zip; this entry just points consumers at them.

    Returns a dict with both keys always present (empty when the corresponding
    path was not used) so callers can rely on JSON serialization succeeding.
    """
    mapping: dict = {
        "ngram_features": {},
        "multimodal_text_model_dirs": [],
    }
    try:
        mapping["ngram_features"] = _extract_ngram_vocab(predictor)
    except Exception as e:
        logger.warning(f"Failed to extract n-gram text vocabulary: {e}")
    try:
        mapping["multimodal_text_model_dirs"] = _scan_multimodal_tokenizer_dirs(
            save_model_path
        )
    except Exception as e:
        logger.warning(f"Failed to scan multimodal text-model artifacts: {e}")
    return mapping


def _get_feature_generator(predictor):
    """Return the fitted feature generator, or ``None`` if it cannot be reached.

    AutoGluon 1.x exposes the generator on the ``Learner`` (``predictor._learner``)
    rather than on the public ``TabularPredictor`` surface. ``_learner`` is
    private but stable across the 1.x line; we guard defensively so a future
    restructure degrades to a warning instead of breaking training.
    """
    learner = getattr(predictor, "_learner", None)
    if learner is None:
        return None
    return getattr(learner, "feature_generator", None)


def _find_ngram_generators(root) -> list:
    """Depth-first walk of a feature-generator pipeline, collecting n-gram generators."""
    found: list = []
    visited: set[int] = set()

    def _walk(node) -> None:
        if node is None:
            return
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if type(node).__name__ == "TextNgramFeatureGenerator":
            found.append(node)

        for attr in (
            "generators",
            "feature_generators",
            "pre_generators",
            "post_generators",
        ):
            sub = getattr(node, attr, None)
            if not sub:
                continue
            if isinstance(sub, (list, tuple)):
                for group in sub:
                    items = group if isinstance(group, (list, tuple)) else [group]
                    for item in items:
                        _walk(item)

    _walk(root)
    return found


def _json_safe_vocab(vocab) -> dict[str, int]:
    """Convert a sklearn ``vocabulary_`` mapping to plain JSON-serializable types.

    Vectorizers often store ``numpy.int64`` indices; those break ``json.dump``
    mid-write and can leave a truncated ``text_feature_mapping.json`` that still
    gets packaged into ``automl_predictor.zip``.
    """
    if not vocab:
        return {}
    safe: dict[str, int] = {}
    for key, value in dict(vocab).items():
        safe[str(key)] = int(value)
    return safe


def _extract_ngram_vocab(predictor) -> dict:
    """Pull word/n-gram -> column-index vocabularies out of any fitted n-gram generators."""
    feature_generator = _get_feature_generator(predictor)
    if feature_generator is None:
        logger.info(
            "No feature_generator accessible on predictor; skipping n-gram vocab extraction"
        )
        return {}

    ngram_gens = _find_ngram_generators(feature_generator)
    if not ngram_gens:
        logger.info(
            "No TextNgramFeatureGenerator found — text was either dropped, "
            "deep-embedded via multimodal, or no text columns were present"
        )
        return {}

    out: dict = {}
    for ngram_gen in ngram_gens:
        feature_names = list(getattr(ngram_gen, "vectorizer_features", []) or [])
        vectorizers = list(getattr(ngram_gen, "vectorizers", []) or [])
        for feat_name, vectorizer in zip(feature_names, vectorizers):
            vocab = getattr(vectorizer, "vocabulary_", None)
            try:
                vocab_features = [
                    str(name) for name in vectorizer.get_feature_names_out()
                ]
            except Exception:
                vocab_features = []
            out[str(feat_name)] = {
                "vocabulary": _json_safe_vocab(vocab),
                "feature_names": vocab_features,
            }
    return out


def _write_text_feature_mapping_atomic(mapping: dict, mapping_path: Path) -> None:
    """Serialize ``mapping`` to JSON, validate it, then atomically replace ``mapping_path``.

    Builds the full payload in memory first so a serialization failure never
    leaves a half-written sidecar next to the predictor (which
    ``shutil.make_archive`` would otherwise zip into ``automl_predictor.zip``).
    """
    payload = json.dumps(mapping, indent=2, sort_keys=True, ensure_ascii=False)
    # Round-trip before touching disk so consumers never see invalid JSON.
    json.loads(payload)

    tmp_path = mapping_path.with_suffix(mapping_path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, mapping_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    # Confirm the on-disk sidecar is complete before the zip step.
    with open(mapping_path, encoding="utf-8") as handle:
        json.load(handle)


def _scan_multimodal_tokenizer_dirs(save_model_path: Path) -> list:
    """List relative paths under ``save_model_path`` containing tokenizer artifacts."""
    if not save_model_path.exists() or not save_model_path.is_dir():
        return []
    hits: set[str] = set()
    for candidate in save_model_path.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.name in _MULTIMODAL_TOKENIZER_MARKERS:
            rel = candidate.parent.relative_to(save_model_path)
            hits.add(str(rel))
    return sorted(hits)


def serialize_and_zip_predictor(
    predictor, save_model_path: Path, tmp_path: Path
) -> Path:
    """Pickle the predictor and zip the model directory. Returns the zip path."""

    if predictor is None:
        raise AutoMLValidationError("predictor cannot be None")

    if not save_model_path.exists():
        raise AutoMLValidationError(
            f"save_model_path does not exist: {save_model_path}"
        )

    predictor_path = save_model_path / "predictor.pkl"

    try:
        with open(predictor_path, "wb") as f:
            pickle.dump(predictor, f)
        logger.debug(f"Predictor serialized to {predictor_path}")
    except IOError as e:
        logger.error(f"Failed to write predictor pickle: {e}")
        raise AutoMLSerializationError(f"Failed to serialize predictor: {e}") from e
    except pickle.PicklingError as e:
        logger.error(f"Failed to pickle predictor: {e}")
        raise AutoMLSerializationError(f"Failed to pickle predictor: {e}") from e

    try:
        instructions_path = save_model_path / "tabular_deployment_instructions.md"
        with open(instructions_path, "w") as f:
            f.write(deployment_instructions())
        logger.debug(f"Deployment instructions written to {instructions_path}")
    except Exception as e:
        logger.debug(f"No deployment_instructions found, {e}")

    mapping_path = save_model_path / "text_feature_mapping.json"
    try:
        mapping = extract_text_feature_mapping(predictor, save_model_path)
        _write_text_feature_mapping_atomic(mapping, mapping_path)
        logger.debug(
            "Text feature mapping written to %s (%s bytes)",
            mapping_path,
            mapping_path.stat().st_size,
        )
    except Exception as e:
        # Never ship a truncated/invalid sidecar inside automl_predictor.zip.
        if mapping_path.exists():
            try:
                mapping_path.unlink()
            except OSError:
                pass
        logger.warning("Failed to write text feature mapping (omitted from zip): %s", e)

    zip_path = tmp_path / "automl_predictor.zip"

    try:
        base_name = str(zip_path).replace(".zip", "")
        shutil.make_archive(
            base_name=base_name,
            format="zip",
            root_dir=save_model_path,
        )
        logger.debug(f"Model zipped to {zip_path}")
    except Exception as e:
        logger.error(f"Failed to create zip archive: {e}")
        raise AutoMLSerializationError(f"Failed to zip model: {e}") from e

    if not zip_path.exists():
        raise AutoMLSerializationError(f"Zip file was not created at {zip_path}")

    return zip_path


def build_upload_payload(
    dataset_id: str,
    dataset_version: str | None,
    metadata: dict,
    task_type: str,
    leaderboard_json: list | dict,
) -> tuple[str, dict]:
    """Return (model_id, form_data_dict) for the AutoDW upload request."""
    instructions = deployment_instructions()
    extra = {}
    if isinstance(instructions, str):
        extra["deployment_instructions"] = instructions

    return _core_build_upload_payload(
        dataset_id,
        dataset_version,
        metadata,
        task_type,
        leaderboard_json,
        name=f"AutoML Model - {dataset_id}",
        description="AutoML trained model for tabular data",
        framework="sklearn",
        extra_fields=extra,
    )
