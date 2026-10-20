"""Shared helpers for extracting and validating CSV + media-file datasets.

The vision, audio, and text AutoML services all consume the same dataset
packaging: a ZIP archive containing a labels CSV (``labels.csv`` or
``metadata.csv``) plus a folder of media files (images, audio clips, or any
other per-row assets). These helpers locate and unpack that structure.

Structure only: they inspect ZIP/folder packaging, never file *types*. Which
extensions are valid for a modality (image vs audio) is decided by the
per-modality services (e.g. ``collect_non_image_files`` in
``app.vision_automl.services``).
"""

import logging
import os
import shutil
from pathlib import Path

import pandas as pd

from app.core.exceptions import AutoMLDataError

logger = logging.getLogger(__name__)


def normalize_dataframe_filenames(
    df: pd.DataFrame, filename_column: str, csv_path: Path
) -> pd.DataFrame:
    """Normalize filenames to basenames and persist CSV back to disk."""
    logger.info("Normalizing filenames in column '%s'", filename_column)
    if filename_column in df.columns:
        df[filename_column] = (
            df[filename_column]
            .astype(str)
            .map(lambda s: os.path.basename(str(s).replace("\\", "/")))
        )
        df.to_csv(csv_path, index=False)
        logger.debug("Normalized filenames saved to %s", csv_path)
    else:
        logger.warning(
            "Filename column '%s' not found during normalization", filename_column
        )
    return df


def resolve_media_root(media_dir: Path) -> Path:
    """Resolve common nested packaging patterns inside uploaded media zips.

    Modality-agnostic: unwraps a nested ``images`` folder (the packaging
    convention shared by all media datasets) and single top-level folders.
    """
    logger.info("Resolving media directory structure at %s", media_dir)
    nested_images_dir = media_dir / "images"
    if nested_images_dir.exists() and nested_images_dir.is_dir():
        logger.debug("Detected nested 'images' folder, using it as root")
        media_dir = nested_images_dir

    try:
        top_level_entries = list(media_dir.iterdir())
        only_dirs = [p for p in top_level_entries if p.is_dir()]
        only_files = [p for p in top_level_entries if p.is_file()]
        if len(only_files) == 0 and len(only_dirs) == 1:
            logger.debug("Detected single top-level directory: %s", only_dirs[0])
            media_dir = only_dirs[0]
    except Exception as e:
        logger.warning("Error resolving media root: %s", e)

    return media_dir


def collect_missing_files(
    df: pd.DataFrame, images_dir: Path, filename_col: str, label_col: str
) -> list[str]:
    """Return a list of filenames referenced in the CSV but absent on disk."""
    missing = []
    for _, row in df.iterrows():
        filename = row[filename_col]

        img_path = images_dir / filename
        if img_path.exists():
            continue

        matches = list(images_dir.rglob(str(filename)))
        if len(matches) == 1:
            continue
        elif len(matches) > 1:
            logger.warning("Multiple matches for %s: %s", filename, matches)

        missing.append(filename)
    return missing


def extract_and_locate_dataset(zip_path: Path, workdir: Path) -> tuple[Path, Path]:
    """
    Extract a media dataset ZIP and return (csv_path, media_dir).

    Raises AutoMLDataError for structural problems.
    """
    extract_dir = workdir / "dataset"
    extract_dir.mkdir(exist_ok=True)
    shutil.unpack_archive(zip_path, extract_dir)

    dataset_root = _find_valid_dataset_root(extract_dir)
    csv_path = _find_csv_file(dataset_root)
    media_dir = _find_or_resolve_media_dir(dataset_root, csv_path)
    return csv_path, media_dir


def _payload_children(extract_dir: Path) -> list[Path]:
    """Direct children of *extract_dir*, ignoring macOS/hidden junk."""
    return [
        child
        for child in sorted(extract_dir.iterdir())
        if "__MACOSX" not in child.parts and not child.name.startswith(".")
    ]


def _find_valid_dataset_root(extract_dir: Path) -> Path:
    """Pick the folder that actually holds the CSV + media payload.

    Handles common AutoDW / export layouts:

    - Flat: ``annotations_new.csv`` beside ``images/`` at the ZIP root
    - Wrapped: ``MyDataset/{labels.csv, images/}``
    - Multi-top-level: search from the extract dir as a whole
    """
    entries = _payload_children(extract_dir)
    if not entries:
        raise AutoMLDataError("No valid dataset folder found in ZIP")

    files = [p for p in entries if p.is_file()]
    dirs = [p for p in entries if p.is_dir()]

    # Flat layout — CSV and/or images/ live directly under the extract dir.
    if any(p.suffix.lower() == ".csv" for p in files) or any(
        d.name.lower() == "images" for d in dirs
    ):
        return extract_dir

    # Single wrapper directory (e.g. dataset.zip → MyDataset/...).
    if len(dirs) == 1 and not files:
        return dirs[0]

    # Several top-level folders (train/, extras/, …) — search the whole tree.
    if dirs:
        return extract_dir

    raise AutoMLDataError("No valid dataset folder found in ZIP")


def _find_csv_file(dataset_root: Path) -> Path:
    """Locate the dataset labels CSV anywhere under the dataset root.

    Preference order:
    1. ``labels.csv`` / ``metadata.csv`` (canonical names)
    2. any ``annotations*.csv`` (e.g. ``annotations_new.csv``)
    3. the only ``.csv`` present in the tree

    Matching is case-insensitive. macOS junk (``__MACOSX``, ``._*``) is
    ignored. If several non-canonical CSVs exist with no annotations match,
    raise so the caller can rename.
    """
    csv_candidates = [
        p
        for p in dataset_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".csv"
        and "__MACOSX" not in p.parts
        and not p.name.startswith("._")
    ]
    if not csv_candidates:
        preview = sorted(
            str(p.relative_to(dataset_root))
            for p in dataset_root.rglob("*")
            if p.is_file() and "__MACOSX" not in p.parts
        )[:20]
        raise AutoMLDataError(
            "No CSV label/metadata file found in dataset "
            "(expected labels.csv, metadata.csv, annotations*.csv, or a single .csv). "
            f"Files seen under {dataset_root}: {preview or '(none)'}"
        )

    by_lower_name = {p.name.lower(): p for p in csv_candidates}
    for preferred in ("labels.csv", "metadata.csv"):
        if preferred in by_lower_name:
            return by_lower_name[preferred]

    annotations = sorted(
        (p for p in csv_candidates if p.name.lower().startswith("annotations")),
        key=lambda p: p.as_posix(),
    )
    if annotations:
        return annotations[0]

    if len(csv_candidates) == 1:
        return csv_candidates[0]

    names = ", ".join(sorted(p.name for p in csv_candidates))
    raise AutoMLDataError(
        f"Multiple CSV files found ({names}); expected labels.csv, "
        "metadata.csv, annotations*.csv, or exactly one .csv"
    )


def _find_or_resolve_media_dir(dataset_root: Path, csv_path: Path) -> Path:
    """Find the media directory in the ZIP, or default to one beside the CSV.

    Falls back to ``<csv parent>/images`` when no ``images`` folder exists in
    the archive (the folder is named ``images`` for all modalities by
    packaging convention), then unwraps nested packaging before checking it
    exists.
    """
    media_candidates = [
        p
        for p in dataset_root.rglob("*")
        if p.is_dir() and p.name.lower() == "images" and "__MACOSX" not in p.parts
    ]
    media_dir = (
        media_candidates[0] if media_candidates else (csv_path.parent / "images")
    )
    resolved_dir = resolve_media_root(media_dir)
    if not resolved_dir.exists():
        raise AutoMLDataError("images/ directory not found in dataset ZIP")
    return resolved_dir
