"""Datamodules for every task type supported by the generic ML engine.

Each datamodule reads a CSV (plus a media root directory where relevant),
splits it into train/val/test, builds task-specific datasets and DataLoaders,
and exposes the fitted preprocessors (image processor, tokenizer, scaler,
encoder) so feature mappings can be extracted after training.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from app.core.config import get_settings
from app.core.exceptions import AutoMLDataError, AutoMLRuntimeError

from .dataset import (
    CausalLMFromCSVDataset,
    ImageClassificationFromCSVDataset,
    MultimodalClassificationDataset,
    QuestionAnsweringFromCSVDataset,
    Seq2SeqFromCSVDataset,
    TextClassificationFromCSVDataset,
)

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

_settings = get_settings()
DEFAULT_BATCH_SIZE: int = _settings.default_batch_size
DEFAULT_NUM_WORKERS: int = _settings.default_num_workers
DEFAULT_VAL_SPLIT: float = _settings.default_val_split
DEFAULT_TEST_SPLIT: float = _settings.default_test_split
DEFAULT_IMAGE_CLASSIFIER_HF_ID: str = _settings.default_image_classifier_hf_id


def auto_num_workers(max_workers: int | None = None) -> int:
    """Pick a sensible DataLoader worker count for "auto" resource requests.

    Uses all but one CPU, capped by ``auto_num_workers_max``; never negative.
    """
    if max_workers is None:
        max_workers = get_settings().auto_num_workers_max
    return max(0, min(max_workers, (os.cpu_count() or 2) - 1))


def _processor_draft_size(processor: Any) -> tuple[int, int] | None:
    """Derive a PIL draft-mode decode target from an HF image processor.

    Draft mode only downscales during JPEG decode, so we keep a 2x margin
    above the processor's target edge to preserve resize quality. Returns
    None when the target size cannot be determined or fast decode is off.
    """
    if not get_settings().fast_image_decode:
        return None

    size = getattr(processor, "size", None)
    edge: int | None = None
    if isinstance(size, int):
        edge = size
    elif isinstance(size, dict):
        shortest = size.get("shortest_edge")
        if isinstance(shortest, int):
            edge = shortest
        elif isinstance(size.get("height"), int) and isinstance(size.get("width"), int):
            edge = min(size["height"], size["width"])

    if edge is None or edge <= 0:
        return None
    target = int(edge) * 2
    return (target, target)


class BaseDataModule(ABC):
    """Common scaffolding for CSV-backed datamodules.

    Reads the CSV, splits it into train/val/test (optionally stratified), and
    exposes the split-specific DataLoaders plus label<->id mappings. Subclasses
    provide the actual dataset construction and collate logic.
    """

    def __init__(
        self,
        csv_file: Path,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        shuffle: bool = True,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        seed: int = 42,
        hf_model_id: str = DEFAULT_IMAGE_CLASSIFIER_HF_ID,
    ) -> None:
        self.csv_file = Path(csv_file)
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.shuffle: bool = shuffle
        self.val_split: float = val_split
        self.test_split: float = test_split
        self.seed: int = seed
        self.hf_model_id = hf_model_id

        self.num_classes: int = 0
        self.id2label: dict[int, str] = {}
        self.label2id: dict[str, int] = {}

        logger.info("Initializing %s with CSV: %s", self.__class__.__name__, csv_file)
        self.setup()

    @abstractmethod
    def setup(self) -> None:
        """Build the train/val/test datasets. Called automatically on init."""

    @abstractmethod
    def _collate_fn(self, batch) -> dict[str, torch.Tensor]:
        """Turn a list of samples into a batch of tensors."""

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader over the training split."""

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Return the DataLoader over the validation split."""

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Return the DataLoader over the test split."""

    @staticmethod
    def _read_csv(csv_file: Path) -> pd.DataFrame:
        """Load a CSV into a DataFrame, mapping common parse failures to typed errors."""
        try:
            logger.info("Reading dataset from %s", csv_file)
            df: pd.DataFrame = pd.read_csv(csv_file)
        except FileNotFoundError as e:
            logger.error("Dataset file not found: %s", e)
            raise
        except pd.errors.EmptyDataError:
            logger.error("Dataset file is empty: %s", csv_file)
            raise AutoMLDataError(f"Dataset file is empty: {csv_file}")
        except pd.errors.ParserError as e:
            logger.error("Failed to parse dataset CSV: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error reading dataset: %s", e)
            raise
        return df

    @staticmethod
    def _split_df(
        df: pd.DataFrame,
        val_split: float,
        test_split: float,
        seed: int,
        stratify_col: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split a DataFrame into train/val/test parts.

        Cuts off the combined val+test fraction first, then splits that chunk
        again so each split gets its requested share. Stratifies on the given
        column when one is supplied.
        """
        stratify_train = df[stratify_col] if stratify_col is not None else None
        try:
            train_df, temp_df = train_test_split(
                df,
                test_size=val_split + test_split,
                stratify=stratify_train,
                random_state=seed,
            )
        except ValueError as e:
            logger.error(
                "Failed to split dataset (insufficient samples or invalid stratification): %s",
                e,
            )
            raise

        stratify_val = temp_df[stratify_col] if stratify_col is not None else None
        try:
            relative_val = val_split / (val_split + test_split)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=1 - relative_val,
                stratify=stratify_val,
                random_state=seed,
            )
        except ValueError as e:
            logger.error("Failed to split validation/test data: %s", e)
            raise

        logger.info(
            "Split completed: train=%d, val=%d, test=%d",
            len(train_df),
            len(val_df),
            len(test_df),
        )
        return train_df, val_df, test_df

    def _loader_kwargs(self) -> dict[str, Any]:
        """Extra DataLoader kwargs for throughput (pinning, prefetching)."""
        kwargs: dict[str, Any] = {}
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        if torch.cuda.is_available():
            kwargs["pin_memory"] = True
        return kwargs

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        """Build a DataLoader wired to this module's batch size and collate fn."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            **self._loader_kwargs(),
        )

    def _build_label_maps(self, classes: list[str]) -> None:
        """Fill num_classes and the id<->label lookup dicts from a class list."""
        self.num_classes = len(classes)
        self.id2label = {i: c for i, c in enumerate(classes)}
        self.label2id = {c: i for i, c in enumerate(classes)}


class ImageClassificationDataModule(BaseDataModule):
    """Handles dataset preparation and dataloaders for image classification tasks."""

    def __init__(
        self,
        csv_file: Path,
        root_dir: Path,
        img_col: str = "filename",
        label_col: str = "label",
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        transform: Callable | None = None,
        shuffle: bool = True,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        seed: int = 42,
        hf_model_id: str = DEFAULT_IMAGE_CLASSIFIER_HF_ID,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.img_col: str = img_col
        self.label_col: str = label_col
        self.transform: Callable | None = transform
        self.train_dataset: ImageClassificationFromCSVDataset | None = None
        self.val_dataset: ImageClassificationFromCSVDataset | None = None
        self.test_dataset: ImageClassificationFromCSVDataset | None = None
        self.processor: AutoImageProcessor | None = None
        super().__init__(
            csv_file=csv_file,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            hf_model_id=hf_model_id,
        )

    def _draft_decode_enabled(self) -> bool:
        """Whether draft-mode JPEG decoding may be used for this task.

        Subclasses whose targets are pixel-aligned (segmentation masks,
        keypoints) disable it to avoid any resampling of the source image.
        """
        return True

    def setup(self) -> None:
        """Split the CSV, build the image datasets, and load the image processor."""
        df = self._read_csv(self.csv_file)
        train_df, val_df, test_df = self._split_df(
            df, self.val_split, self.test_split, self.seed, stratify_col=self.label_col
        )

        try:
            self.processor = AutoImageProcessor.from_pretrained(self.hf_model_id)
        except Exception as e:
            logger.error("Failed to load processor from %s: %s", self.hf_model_id, e)
            raise
        logger.info("Loaded processor from: %s", self.hf_model_id)

        decode_size = (
            _processor_draft_size(self.processor)
            if self._draft_decode_enabled()
            else None
        )

        try:
            self.train_dataset = ImageClassificationFromCSVDataset(
                csv_file=train_df,
                root_dir=self.root_dir,
                img_col=self.img_col,
                label_col=self.label_col,
                transform=self.transform,
                decode_size=decode_size,
            )
            self.val_dataset = ImageClassificationFromCSVDataset(
                csv_file=val_df,
                root_dir=self.root_dir,
                img_col=self.img_col,
                label_col=self.label_col,
                transform=self.transform,
                decode_size=decode_size,
            )
            self.test_dataset = ImageClassificationFromCSVDataset(
                csv_file=test_df,
                root_dir=self.root_dir,
                img_col=self.img_col,
                label_col=self.label_col,
                transform=self.transform,
                decode_size=decode_size,
            )
        except Exception as e:
            logger.error("Failed to create datasets: %s", e)
            raise

        self._build_label_maps(self.train_dataset.classes)

    def _collate_fn(self, batch: list[tuple[Any, Any]]) -> dict[str, torch.Tensor]:
        """Pixel-ize a batch of PIL images and stack the labels into a tensor."""
        images, labels = zip(*batch)
        if self.processor is None:
            raise AutoMLRuntimeError("Processor not initialized. Call setup() first.")
        pixel_values = self.processor(
            images=list(images), return_tensors="pt"
        ).pixel_values
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise AutoMLRuntimeError(
                "Train dataset not initialized. Call setup() first."
            )
        return self._make_loader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise AutoMLRuntimeError(
                "Validation dataset not initialized. Call setup() first."
            )
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise AutoMLRuntimeError(
                "Test dataset not initialized. Call setup() first."
            )
        return self._make_loader(self.test_dataset, shuffle=False)


# ---------------------------------------------------------------------------
# Multimodal image classification (image + tabular auxiliary features)
# ---------------------------------------------------------------------------


def _infer_column_types(
    df: pd.DataFrame, columns: list[str]
) -> tuple[list[str], list[str]]:
    """Split *columns* into (numeric_cols, categorical_cols) based on dtype."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in columns:
        if col not in df.columns:
            continue
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


class MultimodalClassificationDataModule(BaseDataModule):
    """Handles dataset preparation and dataloaders for multimodal image
    classification tasks where the CSV contains auxiliary metadata columns
    alongside the filename and label.

    Numeric auxiliary columns are standard-scaled; categorical/string columns
    are ordinal-encoded.  Scalers/encoders are fit on the **training split
    only** and then applied to validation and test splits.
    """

    def __init__(
        self,
        csv_file: Path,
        root_dir: Path,
        img_col: str = "filename",
        label_col: str = "label",
        auxiliary_columns: list[str] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        transform: Callable | None = None,
        shuffle: bool = True,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        seed: int = 42,
        hf_model_id: str = DEFAULT_IMAGE_CLASSIFIER_HF_ID,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.img_col: str = img_col
        self.label_col: str = label_col
        self.auxiliary_columns: list[str] = auxiliary_columns or []
        self.transform: Callable | None = transform
        self.train_dataset: MultimodalClassificationDataset | None = None
        self.val_dataset: MultimodalClassificationDataset | None = None
        self.test_dataset: MultimodalClassificationDataset | None = None
        self.processor: AutoImageProcessor | None = None
        self.aux_feature_dim: int = 0
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.scaler: StandardScaler | None = None
        self.encoder: OrdinalEncoder | None = None
        super().__init__(
            csv_file=csv_file,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            hf_model_id=hf_model_id,
        )

    def setup(self) -> None:
        df = self._read_csv(self.csv_file)

        self.numeric_cols, self.categorical_cols = _infer_column_types(
            df, self.auxiliary_columns
        )
        self.aux_feature_dim = len(self.auxiliary_columns)
        logger.info(
            "Auxiliary columns \u2014 numeric: %s, categorical: %s (total dim=%d)",
            self.numeric_cols,
            self.categorical_cols,
            self.aux_feature_dim,
        )

        train_df, val_df, test_df = self._split_df(
            df, self.val_split, self.test_split, self.seed, stratify_col=self.label_col
        )

        train_df = self._encode_auxiliary(train_df, fit=True)
        val_df = self._encode_auxiliary(val_df, fit=False)
        test_df = self._encode_auxiliary(test_df, fit=False)

        try:
            self.processor = AutoImageProcessor.from_pretrained(self.hf_model_id)
        except Exception as e:
            logger.error("Failed to load processor from %s: %s", self.hf_model_id, e)
            raise
        logger.info("Loaded processor from: %s", self.hf_model_id)

        decode_size = _processor_draft_size(self.processor)

        try:
            self.train_dataset = MultimodalClassificationDataset(
                csv_file=train_df,
                root_dir=self.root_dir,
                img_col=self.img_col,
                label_col=self.label_col,
                auxiliary_columns=self.auxiliary_columns,
                transform=self.transform,
                decode_size=decode_size,
            )
            self.val_dataset = MultimodalClassificationDataset(
                csv_file=val_df,
                root_dir=self.root_dir,
                img_col=self.img_col,
                label_col=self.label_col,
                auxiliary_columns=self.auxiliary_columns,
                transform=self.transform,
                decode_size=decode_size,
            )
            self.test_dataset = MultimodalClassificationDataset(
                csv_file=test_df,
                root_dir=self.root_dir,
                img_col=self.img_col,
                label_col=self.label_col,
                auxiliary_columns=self.auxiliary_columns,
                transform=self.transform,
                decode_size=decode_size,
            )
        except Exception as e:
            logger.error("Failed to create datasets: %s", e)
            raise

        self._build_label_maps(self.train_dataset.classes)

    def _encode_auxiliary(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encode auxiliary columns in-place.  If *fit* is True, fit the
        scaler/encoder on *df* (must be the training split)."""
        df = df.copy()

        if self.numeric_cols:
            subset = df[self.numeric_cols].fillna(0.0)
            if fit:
                self.scaler = StandardScaler()
                df[self.numeric_cols] = self.scaler.fit_transform(subset)
            else:
                if self.scaler is None:
                    raise AutoMLRuntimeError(
                        "Scaler not fitted. Call setup() with training data first."
                    )
                df[self.numeric_cols] = self.scaler.transform(subset)

        if self.categorical_cols:
            subset = df[self.categorical_cols].astype(str).fillna("missing")
            if fit:
                self.encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                encoded = self.encoder.fit_transform(subset)
            else:
                if self.encoder is None:
                    raise AutoMLRuntimeError(
                        "OrdinalEncoder not fitted. Call setup() with training data first."
                    )
                encoded = self.encoder.transform(subset)
            for i, col in enumerate(self.categorical_cols):
                df[col] = encoded[:, i].astype(float)

        return df

    def _collate_fn(self, batch: list[tuple[Any, Any, Any]]) -> dict[str, torch.Tensor]:
        images, aux_values, labels = zip(*batch)
        if self.processor is None:
            raise AutoMLRuntimeError("Processor not initialized. Call setup() first.")
        pixel_values = self.processor(
            images=list(images), return_tensors="pt"
        ).pixel_values
        return {
            "pixel_values": pixel_values,
            "aux_features": torch.tensor(np.stack(aux_values), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise AutoMLRuntimeError(
                "Train dataset not initialized. Call setup() first."
            )
        return self._make_loader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise AutoMLRuntimeError(
                "Validation dataset not initialized. Call setup() first."
            )
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise AutoMLRuntimeError(
                "Test dataset not initialized. Call setup() first."
            )
        return self._make_loader(self.test_dataset, shuffle=False)


ClassificationData = ImageClassificationDataModule


# ---------------------------------------------------------------------------
# Image segmentation
# ---------------------------------------------------------------------------


class ImageSegmentationFromCSVDataset(Dataset):
    """
    Dataset for semantic image segmentation.

    Each CSV row points to:
        - an RGB source image
        - a single-channel integer class-ID mask

    Supported layouts include:

        root/
            images/
                image_0001.png
            masks/
                mask_0001.png

    and paths already containing their subdirectory:

        filename = images/image_0001.png
        mask_filename = masks/mask_0001.png
    """

    def __init__(
        self,
        csv_file,
        root_dir: Path,
        img_col: str = "filename",
        mask_col: str = "mask_filename",
    ) -> None:

        self.root_dir = Path(
            root_dir
        )

        self.img_col = img_col
        self.mask_col = mask_col

        if isinstance(
            csv_file,
            pd.DataFrame,
        ):
            self.df = csv_file.reset_index(
                drop=True
            ).copy()
        else:
            self.df = pd.read_csv(
                csv_file
            )

        required_columns = {
            self.img_col,
            self.mask_col,
        }

        missing_columns = (
            required_columns
            - set(self.df.columns)
        )

        if missing_columns:
            raise ValueError(
                "Segmentation dataset is missing "
                f"required columns: "
                f"{sorted(missing_columns)}"
            )

    def __len__(self) -> int:
        return len(
            self.df
        )

    def _resolve_file(
            self,
            raw_value,
            preferred_dir: str,
    ) -> Path:
        raw_path = Path(str(raw_value))

        # Absolute path supplied directly.
        if raw_path.is_absolute():
            if raw_path.exists() and raw_path.is_file():
                return raw_path

        candidates = [
            # root/<value>
            self.root_dir / raw_path,

            # root/<preferred_dir>/<basename>
            self.root_dir
            / preferred_dir
            / raw_path.name,

            # Production AutoML may give us:
            #
            #   root_dir = dataset/images
            #
            # while segmentation masks live at:
            #
            #   dataset/masks
            #
            # so also search relative to root_dir.parent.
            self.root_dir.parent
            / raw_path,

            self.root_dir.parent
            / preferred_dir
            / raw_path.name,
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        # Search below the supplied root first.
        matches = list(
            self.root_dir.rglob(
                raw_path.name
            )
        )

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise FileNotFoundError(
                f"Multiple matches for "
                f"{raw_value!r} under "
                f"{self.root_dir}: "
                f"{matches}"
            )

        # Then search from the dataset parent so sibling
        # images/ and masks/ directories are visible.
        parent_matches = list(
            self.root_dir.parent.rglob(
                raw_path.name
            )
        )

        if len(parent_matches) == 1:
            return parent_matches[0]

        if len(parent_matches) > 1:
            raise FileNotFoundError(
                f"Multiple matches for "
                f"{raw_value!r} under "
                f"{self.root_dir.parent}: "
                f"{parent_matches}"
            )

        raise FileNotFoundError(
            f"Could not resolve "
            f"{preferred_dir} file "
            f"{raw_value!r} under "
            f"{self.root_dir}"
        )

    def __getitem__(
        self,
        index: int,
    ):

        from PIL import Image as _Image
        import numpy as _np

        row = self.df.iloc[
            index
        ]

        image_path = self._resolve_file(
            row[self.img_col],
            "images",
        )

        mask_path = self._resolve_file(
            row[self.mask_col],
            "masks",
        )

        with _Image.open(
            image_path
        ) as image_file:

            image = image_file.convert(
                "RGB"
            ).copy()

        with _Image.open(
            mask_path
        ) as mask_file:

            mask_array = _np.asarray(
                mask_file
            ).copy()

        if mask_array.ndim == 3:

            if (
                mask_array.shape[2]
                == 1
            ):
                mask_array = (
                    mask_array[..., 0]
                )

            else:
                raise ValueError(
                    "Segmentation mask must "
                    "contain integer class IDs "
                    "in a single channel: "
                    f"{mask_path}, "
                    f"shape={mask_array.shape}"
                )

        if mask_array.ndim != 2:
            raise ValueError(
                "Segmentation mask must "
                f"be 2-D: {mask_path}, "
                f"shape={mask_array.shape}"
            )

        expected_mask_size = (
            image.height,
            image.width,
        )

        if (
            mask_array.shape
            != expected_mask_size
        ):
            raise ValueError(
                "Image and segmentation mask "
                "dimensions do not match: "
                f"image={image_path} "
                f"size={image.size}, "
                f"mask={mask_path} "
                f"shape={mask_array.shape}"
            )

        mask_tensor = torch.from_numpy(
            mask_array.astype(
                _np.int64,
                copy=False,
            )
        ).long()

        return (
            image,
            mask_tensor,
        )


class ImageSegmentationDataModule(ImageClassificationDataModule):
    """Datamodule for image segmentation tasks.

    Uses the same CSV + class-subdir image layout as image classification.
    The collate function passes ``labels`` (pixel-level segmentation maps)
    to the processor.  The labels CSV must contain a ``mask_filename``
    column pointing to the segmentation mask image (same class-subdir layout).
    """

    def __init__(
        self,
        csv_file: Path,
        root_dir: Path,
        img_col: str = "filename",
        label_col: str = "label",
        mask_col: str = "mask_filename",
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        seed: int = 42,
        hf_model_id: str = DEFAULT_IMAGE_CLASSIFIER_HF_ID,
    ) -> None:
        self.mask_col = mask_col
        super().__init__(
            csv_file=csv_file,
            root_dir=root_dir,
            img_col=img_col,
            label_col=label_col,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            hf_model_id=hf_model_id,
        )

    def _draft_decode_enabled(self) -> bool:
        """Disabled: masks must stay pixel-aligned with the source image."""
        return False

    def setup(self) -> None:
        """
        Prepare train/validation/test datasets for
        semantic image segmentation.

        Unlike image classification:
          - do not stratify by a row-level label
          - load mask images as pixel-level targets
          - derive class IDs from mask pixel values
        """

        df = self._read_csv(
            self.csv_file
        )

        required_columns = {
            self.img_col,
            self.mask_col,
        }

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:
            raise AutoMLRuntimeError(
                "Image segmentation dataset "
                "is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        # Segmentation has no single row-level class
        # suitable for stratification.
        train_df, val_df, test_df = (
            self._split_df(
                df,
                self.val_split,
                self.test_split,
                self.seed,
            )
        )

        try:
            self.processor = (
                AutoImageProcessor
                .from_pretrained(
                    self.hf_model_id
                )
            )

        except Exception as e:

            logger.error(
                "Failed to load segmentation "
                "processor from %s: %s",
                self.hf_model_id,
                e,
            )

            raise

        logger.info(
            "Loaded segmentation processor "
            "from: %s",
            self.hf_model_id,
        )

        try:

            self.train_dataset = (
                ImageSegmentationFromCSVDataset(
                    csv_file=train_df,
                    root_dir=self.root_dir,
                    img_col=self.img_col,
                    mask_col=self.mask_col,
                )
            )

            self.val_dataset = (
                ImageSegmentationFromCSVDataset(
                    csv_file=val_df,
                    root_dir=self.root_dir,
                    img_col=self.img_col,
                    mask_col=self.mask_col,
                )
            )

            self.test_dataset = (
                ImageSegmentationFromCSVDataset(
                    csv_file=test_df,
                    root_dir=self.root_dir,
                    img_col=self.img_col,
                    mask_col=self.mask_col,
                )
            )

        except Exception as e:

            logger.error(
                "Failed to create image "
                "segmentation datasets: %s",
                e,
            )

            raise

        # --------------------------------------------------
        # Determine the segmentation class IDs.
        # --------------------------------------------------

        class_ids = set()

        full_dataset = (
            ImageSegmentationFromCSVDataset(
                csv_file=df,
                root_dir=self.root_dir,
                img_col=self.img_col,
                mask_col=self.mask_col,
            )
        )

        for index in range(
            len(full_dataset)
        ):

            _, mask = (
                full_dataset[index]
            )

            class_ids.update(
                int(value)
                for value
                in torch.unique(
                    mask
                ).tolist()
                if int(value) != 255
            )

        if not class_ids:
            raise AutoMLRuntimeError(
                "No segmentation class IDs "
                "were found in the masks."
            )

        sorted_class_ids = sorted(
            class_ids
        )

        # For the current ALFIE segmentation model,
        # masks should use contiguous IDs:
        # 0, 1, 2, ...
        expected_class_ids = list(
            range(
                max(sorted_class_ids)
                + 1
            )
        )

        if (
            sorted_class_ids
            != expected_class_ids
        ):
            raise AutoMLRuntimeError(
                "Segmentation mask class IDs "
                "must be contiguous starting "
                "from 0. "
                f"Found={sorted_class_ids}, "
                f"expected={expected_class_ids}"
            )

        self.num_classes = len(
            sorted_class_ids
        )

        self.id2label = {
            class_id: str(
                class_id
            )
            for class_id
            in sorted_class_ids
        }

        self.label2id = {
            label: class_id
            for class_id, label
            in self.id2label.items()
        }

        logger.info(
            "Image segmentation dataset ready: "
            "train=%d val=%d test=%d "
            "num_classes=%d class_ids=%s",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
            self.num_classes,
            sorted_class_ids,
        )

    def _collate_fn(self, batch: list[tuple[Any, Any]]) -> dict[str, torch.Tensor]:
        images, labels = zip(*batch)
        if self.processor is None:
            raise AutoMLRuntimeError("Processor not initialized.")
        encoding = self.processor(images=list(images), return_tensors="pt")
        return {
            "pixel_values": encoding.pixel_values,
            "labels": torch.stack(
                [
                    label if isinstance(label, torch.Tensor) else torch.tensor(label)
                    for label in labels
                ]
            ),
        }


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------


class ObjectDetectionDataModule(BaseDataModule):
    """Datamodule for object detection tasks.

    CSV columns: ``filename`` (image file), ``boxes`` (JSON list of
    ``[x_min, y_min, x_max, y_max]``), ``class_labels`` (JSON list of
    int class IDs).  Images live in class-neutral flat layout under
    ``root_dir/images/``.
    """

    def __init__(
        self,
        csv_file: Path,
        root_dir: Path,
        img_col: str = "filename",
        boxes_col: str = "boxes",
        class_labels_col: str = "class_labels",
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        seed: int = 42,
        hf_model_id: str = "facebook/detr-resnet-50",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.img_col = img_col
        self.boxes_col = boxes_col
        self.class_labels_col = class_labels_col
        self.processor: AutoImageProcessor | None = None
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        super().__init__(
            csv_file=csv_file,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            hf_model_id=hf_model_id,
        )

    def setup(self) -> None:
        import json as _json

        df = self._read_csv(self.csv_file)

        try:
            all_labels: set[int] = set()
            for row in df[self.class_labels_col]:
                all_labels.update(_json.loads(row))
        except _json.JSONDecodeError as e:
            logger.error("Failed to parse JSON in class_labels column: %s", e)
            raise
        except KeyError as e:
            logger.error("Column not found in dataset: %s", e)
            raise

        self.num_classes = len(all_labels)
        self.id2label = {i: str(i) for i in sorted(all_labels)}
        self.label2id = {v: k for k, v in self.id2label.items()}

        train_df, val_df, test_df = self._split_df(
            df, self.val_split, self.test_split, self.seed
        )

        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        try:
            self.processor = AutoImageProcessor.from_pretrained(self.hf_model_id)
        except Exception as e:
            logger.error("Failed to load processor from %s: %s", self.hf_model_id, e)
            raise

    def _make_dataset(self, df: pd.DataFrame) -> Dataset:
        import json as _json

        from PIL import Image as _Image

        root = self.root_dir

        class _DetectionDataset(Dataset):
            def __init__(self, df, root, img_col, boxes_col, class_labels_col):
                self.df = df
                self.root = root
                self.img_col = img_col
                self.boxes_col = boxes_col
                self.class_labels_col = class_labels_col

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                img = _Image.open(self.root / str(row[self.img_col])).convert("RGB")
                boxes = _json.loads(row[self.boxes_col])
                class_labels = _json.loads(row[self.class_labels_col])
                return img, {
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "class_labels": torch.tensor(class_labels, dtype=torch.long),
                }

        return _DetectionDataset(
            df, root, self.img_col, self.boxes_col, self.class_labels_col
        )

    def _collate_fn(self, batch):
        images, targets = zip(*batch)
        encoding = self.processor(images=list(images), return_tensors="pt")
        return {"pixel_values": encoding.pixel_values, "labels": list(targets)}

    def _make_df_loader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        return DataLoader(
            self._make_dataset(df),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            **self._loader_kwargs(),
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_df_loader(self.train_df, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_df_loader(self.val_df, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_df_loader(self.test_df, shuffle=False)


# ---------------------------------------------------------------------------
# Video classification
# ---------------------------------------------------------------------------


def _evenly_spaced_timestamps(duration: float, n: int) -> list[float]:
    """Return ``n`` timestamps evenly spread across [0, duration)."""
    if n <= 0:
        return []
    if n == 1:
        return [0.0]
    step = duration / n
    return [i * step for i in range(n)]


def _decode_clip_streaming(video_path: str, num_frames: int) -> torch.Tensor:
    """Decode only the sampled frames via ``torchvision.io.VideoReader``.

    Raises on any failure (missing PyAV, no duration metadata, decode
    errors) so callers can fall back to a full ``read_video`` decode.
    """
    from torchvision.io import VideoReader

    reader = VideoReader(video_path, stream="video")
    metadata = reader.get_metadata()
    duration = metadata.get("video", {}).get("duration", [None])[0]
    if not duration or duration <= 0:
        raise ValueError(f"No duration metadata for {video_path}")

    frames: list[torch.Tensor] = []
    last: torch.Tensor | None = None
    for ts in _evenly_spaced_timestamps(float(duration), num_frames):
        reader.seek(ts)
        frame = next(reader, None)
        if frame is not None:
            last = frame["data"]  # (C, H, W) uint8
        if last is not None:
            frames.append(last)

    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return torch.stack(frames)  # (T, C, H, W)


class VideoClassificationDataModule(BaseDataModule):
    """Datamodule for video classification tasks.

    CSV columns: ``video_path`` (relative to ``root_dir``) and ``label``.
    Frames are decoded using ``torchvision.io.read_video``.
    """

    def __init__(
        self,
        csv_file: Path,
        root_dir: Path,
        video_col: str = "video_path",
        label_col: str = "label",
        class_labels_col: str = "class_labels",
        num_frames: int = 8,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        seed: int = 42,
        hf_model_id: str = "MCG-NJU/videomae-base",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.video_col = video_col
        self.label_col = label_col
        self.class_labels_col = class_labels_col
        self.num_frames = num_frames
        self.processor: AutoImageProcessor | None = None
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        super().__init__(
            csv_file=csv_file,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            hf_model_id=hf_model_id,
        )

    def setup(self) -> None:
        import json

        df = self._read_csv(self.csv_file)

        try:
            all_labels: set[int] = set()
            for row in df[self.class_labels_col]:
                all_labels.update(json.loads(row))
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON in class_labels column: %s", e)
            raise
        except KeyError as e:
            logger.error("Column not found in dataset: %s", e)
            raise

        self.num_classes = len(all_labels)
        self.id2label = {i: str(i) for i in sorted(all_labels)}
        self.label2id = {v: k for k, v in self.id2label.items()}

        train_df, val_df, test_df = self._split_df(
            df, self.val_split, self.test_split, self.seed
        )

        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        try:
            self.processor = AutoImageProcessor.from_pretrained(self.hf_model_id)
        except pd.errors.EmptyDataError:
            logger.error("Dataset file is empty: %s", self.csv_file)
            raise AutoMLDataError(f"Dataset file is empty: {self.csv_file}")
        except pd.errors.ParserError as e:
            logger.error("Failed to parse dataset CSV: %s", e)
            raise

        try:
            classes = sorted(df[self.label_col].unique().tolist())
        except KeyError as e:
            logger.error(
                "Label column '%s' not found in dataset: %s", self.label_col, e
            )
            raise

        self._build_label_maps(classes)
        df = df.copy()
        df[self.label_col] = df[self.label_col].map(self.label2id)

        train_df, val_df, test_df = self._split_df(
            df, self.val_split, self.test_split, self.seed, stratify_col=self.label_col
        )

        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        try:
            self.processor = AutoImageProcessor.from_pretrained(self.hf_model_id)
        except Exception as e:
            logger.error("Failed to load processor from %s: %s", self.hf_model_id, e)
            raise

    def _make_dataset(self, df: pd.DataFrame) -> Dataset:
        from torchvision.io import read_video

        root = self.root_dir
        num_frames = self.num_frames
        video_col = self.video_col
        label_col = self.label_col

        def _decode_clip(video_path: str) -> torch.Tensor:
            """Decode ``num_frames`` evenly-spaced frames as (T, C, H, W)."""
            try:
                return _decode_clip_streaming(video_path, num_frames)
            except Exception as e:
                # Streaming decode needs PyAV + valid duration metadata; fall
                # back to a full decode when either is unavailable.
                logger.debug("Streaming video decode failed for %s: %s", video_path, e)
                frames, _, _ = read_video(
                    video_path, output_format="TCHW", pts_unit="sec"
                )
                total = frames.shape[0]
                indices = torch.linspace(0, total - 1, num_frames).long()
                return frames[indices]  # (T, C, H, W)

        class _VideoDataset(Dataset):
            def __init__(self, df):
                self.df = df

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                video_path = str(root / str(row[video_col]))
                frames = _decode_clip(video_path)
                return frames.float() / 255.0, torch.tensor(
                    int(row[label_col]), dtype=torch.long
                )

        return _VideoDataset(df)

    def _collate_fn(self, batch):
        clips, labels = zip(*batch)
        return {
            "pixel_values": torch.stack(clips),  # (B, T, C, H, W)
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _make_df_loader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        return DataLoader(
            self._make_dataset(df),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            **self._loader_kwargs(),
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_df_loader(self.train_df, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_df_loader(self.val_df, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_df_loader(self.test_df, shuffle=False)


# ---------------------------------------------------------------------------
# Keypoint detection
# ---------------------------------------------------------------------------


class KeypointDetectionDataModule(ImageClassificationDataModule):
    """Datamodule for keypoint detection tasks.

    Uses the same CSV + image layout as image classification.
    The ``keypoints_col`` should contain a JSON list of
    ``[x, y, visibility]`` entries (one per keypoint).
    """

    def __init__(
        self,
        csv_file: Path,
        root_dir: Path,
        img_col: str = "filename",
        label_col: str = "label",
        keypoints_col: str = "keypoints",
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        seed: int = 42,
        hf_model_id: str = DEFAULT_IMAGE_CLASSIFIER_HF_ID,
    ) -> None:
        self.keypoints_col = keypoints_col
        super().__init__(
            csv_file=csv_file,
            root_dir=root_dir,
            img_col=img_col,
            label_col=label_col,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            hf_model_id=hf_model_id,
        )

    def _draft_decode_enabled(self) -> bool:
        """Disabled: keypoint coordinates are absolute pixels."""
        return False

    def _collate_fn(self, batch):
        images, labels = zip(*batch)
        if self.processor is None:
            raise AutoMLRuntimeError("Processor not initialized.")
        encoding = self.processor(images=list(images), return_tensors="pt")
        return {
            "pixel_values": encoding.pixel_values,
            "labels": torch.stack(
                [
                    label if isinstance(label, torch.Tensor) else torch.tensor(label)
                    for label in labels
                ]
            ),
        }


# ---------------------------------------------------------------------------
# Audio classification
# ---------------------------------------------------------------------------


class AudioClassificationDataModule(BaseDataModule):
    """Datamodule for audio classification tasks.

    CSV columns: ``audio_path`` (relative to ``root_dir``) and ``label``.
    Audio is loaded with ``torchaudio`` (must be installed separately).
    """

    def __init__(
        self,
        csv_file: Path,
        root_dir: Path,
        audio_col: str = "audio_path",
        label_col: str = "label",
        sampling_rate: int = 16000,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        seed: int = 42,
        hf_model_id: str = "facebook/wav2vec2-base",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.audio_col = audio_col
        self.label_col = label_col
        self.sampling_rate = sampling_rate
        self.feature_extractor: AutoFeatureExtractor | None = None
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        super().__init__(
            csv_file=csv_file,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            hf_model_id=hf_model_id,
        )

    def setup(self) -> None:
        df = self._read_csv(self.csv_file)

        try:
            classes = sorted(df[self.label_col].unique().tolist())
        except KeyError as e:
            logger.error(
                "Label column '%s' not found in dataset: %s", self.label_col, e
            )
            raise

        self._build_label_maps(classes)
        df = df.copy()
        df[self.label_col] = df[self.label_col].map(self.label2id)

        train_df, val_df, test_df = self._split_df(
            df, self.val_split, self.test_split, self.seed, stratify_col=self.label_col
        )

        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.hf_model_id
            )
        except Exception as e:
            logger.error(
                "Failed to load feature extractor from %s: %s", self.hf_model_id, e
            )
            raise

    def _make_dataset(self, df: pd.DataFrame) -> Dataset:
        try:
            import torchaudio
        except ImportError as e:
            raise ImportError(
                "torchaudio is required for audio tasks. "
                "Install it with: pip install torchaudio"
            ) from e

        root = self.root_dir
        audio_col = self.audio_col
        label_col = self.label_col
        target_sr = self.sampling_rate

        class _AudioDataset(Dataset):
            def __init__(self, df):
                self.df = df

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                waveform, sr = torchaudio.load(str(root / str(row[audio_col])))
                if sr != target_sr:
                    waveform = torchaudio.functional.resample(waveform, sr, target_sr)
                waveform = waveform.mean(0)  # mono
                return waveform, torch.tensor(int(row[label_col]), dtype=torch.long)

        return _AudioDataset(df)

    def _collate_fn(self, batch):
        waveforms, labels = zip(*batch)
        if self.feature_extractor is None:
            raise AutoMLRuntimeError("Feature extractor not initialized.")
        inputs = self.feature_extractor(
            [w.numpy() for w in waveforms],
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        return {
            "input_values": inputs.input_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _make_df_loader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        return DataLoader(
            self._make_dataset(df),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            **self._loader_kwargs(),
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_df_loader(self.train_df, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_df_loader(self.val_df, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_df_loader(self.test_df, shuffle=False)


# ---------------------------------------------------------------------------
# Text / sequence classification
# ---------------------------------------------------------------------------


class SequenceClassificationDataModule(BaseDataModule):
    """Datamodule for text sequence classification tasks.

    CSV columns: ``text`` and ``label``.
    """

    def __init__(
        self,
        csv_file: Path,
        text_col: str = "text",
        label_col: str = "label",
        max_length: int = 128,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        seed: int = 42,
        hf_model_id: str = "distilbert-base-uncased",
    ) -> None:
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length
        self.tokenizer: AutoTokenizer | None = None
        self.train_dataset: TextClassificationFromCSVDataset | None = None
        self.val_dataset: TextClassificationFromCSVDataset | None = None
        self.test_dataset: TextClassificationFromCSVDataset | None = None
        super().__init__(
            csv_file=csv_file,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            hf_model_id=hf_model_id,
        )

    def setup(self) -> None:
        df = self._read_csv(self.csv_file)
        train_df, val_df, test_df = self._split_df(
            df, self.val_split, self.test_split, self.seed, stratify_col=self.label_col
        )

        try:
            self.train_dataset = TextClassificationFromCSVDataset(
                train_df, self.text_col, self.label_col
            )
            self.val_dataset = TextClassificationFromCSVDataset(
                val_df, self.text_col, self.label_col
            )
            self.test_dataset = TextClassificationFromCSVDataset(
                test_df, self.text_col, self.label_col
            )
        except Exception as e:
            logger.error("Failed to create datasets: %s", e)
            raise

        classes = self.train_dataset.classes
        self.num_classes = len(classes)
        self.id2label = {i: str(c) for i, c in enumerate(classes)}
        self.label2id = {str(c): i for i, c in enumerate(classes)}

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        except Exception as e:
            logger.error("Failed to load tokenizer from %s: %s", self.hf_model_id, e)
            raise

    def _collate_fn(self, batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts, labels = zip(*batch)
        if self.tokenizer is None:
            raise AutoMLRuntimeError("Tokenizer not initialized.")
        encoding = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------


class QuestionAnsweringDataModule(BaseDataModule):
    """Datamodule for extractive question answering tasks.

    CSV columns: ``question``, ``context``, ``answer_start`` (char offset),
    ``answer_text``.
    """

    def __init__(
        self,
        csv_file: Path,
        question_col: str = "question",
        context_col: str = "context",
        answer_start_col: str = "answer_start",
        answer_text_col: str = "answer_text",
        max_length: int = 384,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        seed: int = 42,
        hf_model_id: str = "distilbert-base-uncased-distilled-squad",
    ) -> None:
        self.question_col = question_col
        self.context_col = context_col
        self.answer_start_col = answer_start_col
        self.answer_text_col = answer_text_col
        self.max_length = max_length
        self.tokenizer: AutoTokenizer | None = None
        self.train_dataset: QuestionAnsweringFromCSVDataset | None = None
        self.val_dataset: QuestionAnsweringFromCSVDataset | None = None
        self.test_dataset: QuestionAnsweringFromCSVDataset | None = None
        super().__init__(
            csv_file=csv_file,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            hf_model_id=hf_model_id,
        )

    def setup(self) -> None:
        df = self._read_csv(self.csv_file)
        train_df, val_df, test_df = self._split_df(
            df, self.val_split, self.test_split, self.seed
        )

        try:
            self.train_dataset = QuestionAnsweringFromCSVDataset(
                train_df,
                self.question_col,
                self.context_col,
                self.answer_start_col,
                self.answer_text_col,
            )
            self.val_dataset = QuestionAnsweringFromCSVDataset(
                val_df,
                self.question_col,
                self.context_col,
                self.answer_start_col,
                self.answer_text_col,
            )
            self.test_dataset = QuestionAnsweringFromCSVDataset(
                test_df,
                self.question_col,
                self.context_col,
                self.answer_start_col,
                self.answer_text_col,
            )
        except Exception as e:
            logger.error("Failed to create datasets: %s", e)
            raise

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        except Exception as e:
            logger.error("Failed to load tokenizer from %s: %s", self.hf_model_id, e)
            raise

    def _collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise AutoMLRuntimeError("Tokenizer not initialized.")
        questions = [b["question"] for b in batch]
        contexts = [b["context"] for b in batch]
        answer_starts = [b["answer_start"] for b in batch]
        answer_texts = [b["answer_text"] for b in batch]

        encoding = self.tokenizer(
            questions,
            contexts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = encoding.pop("offset_mapping")

        # Convert character-level answer positions to token positions
        start_positions = []
        end_positions = []
        for i, (start_char, answer) in enumerate(zip(answer_starts, answer_texts)):
            end_char = start_char + len(answer)
            offsets = offset_mapping[i].tolist()
            token_start = token_end = 0
            for j, (s, e) in enumerate(offsets):
                if s <= start_char < e:
                    token_start = j
                if s < end_char <= e:
                    token_end = j
                    break
            start_positions.append(token_start)
            end_positions.append(token_end)

        return {
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask,
            "start_positions": torch.tensor(start_positions, dtype=torch.long),
            "end_positions": torch.tensor(end_positions, dtype=torch.long),
        }

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)


# ---------------------------------------------------------------------------
# Causal language modelling
# ---------------------------------------------------------------------------


class CausalLMDataModule(BaseDataModule):
    """Datamodule for causal language modelling tasks.

    CSV column: ``text``.  Labels are produced by shifting ``input_ids``
    right by one position (handled by the model internally when ``labels``
    equals ``input_ids``).
    """

    def __init__(
        self,
        csv_file: Path,
        text_col: str = "text",
        max_length: int = 256,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        seed: int = 42,
        hf_model_id: str = "distilgpt2",
    ) -> None:
        self.text_col = text_col
        self.max_length = max_length
        self.tokenizer: AutoTokenizer | None = None
        self.train_dataset: CausalLMFromCSVDataset | None = None
        self.val_dataset: CausalLMFromCSVDataset | None = None
        self.test_dataset: CausalLMFromCSVDataset | None = None
        super().__init__(
            csv_file=csv_file,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            hf_model_id=hf_model_id,
        )

    def setup(self) -> None:
        df = self._read_csv(self.csv_file)
        train_df, val_df, test_df = self._split_df(
            df, self.val_split, self.test_split, self.seed
        )

        try:
            self.train_dataset = CausalLMFromCSVDataset(train_df, self.text_col)
            self.val_dataset = CausalLMFromCSVDataset(val_df, self.text_col)
            self.test_dataset = CausalLMFromCSVDataset(test_df, self.text_col)
        except Exception as e:
            logger.error("Failed to create datasets: %s", e)
            raise

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        except Exception as e:
            logger.error("Failed to load tokenizer from %s: %s", self.hf_model_id, e)
            raise

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _collate_fn(self, batch: list[str]) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise AutoMLRuntimeError("Tokenizer not initialized.")
        encoding = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # For causal LM, labels = input_ids (model shifts internally)
        return {
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask,
            "labels": encoding.input_ids.clone(),
        }

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)


# ---------------------------------------------------------------------------
# Seq2Seq language modelling
# ---------------------------------------------------------------------------


class Seq2SeqLMDataModule(BaseDataModule):
    """Datamodule for sequence-to-sequence tasks.

    CSV columns: ``input_text`` and ``target_text``.
    """

    def __init__(
        self,
        csv_file: Path,
        input_col: str = "input_text",
        target_col: str = "target_text",
        max_source_length: int = 256,
        max_target_length: int = 128,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        seed: int = 42,
        hf_model_id: str = "t5-small",
    ) -> None:
        self.input_col = input_col
        self.target_col = target_col
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer: AutoTokenizer | None = None
        self.train_dataset: Seq2SeqFromCSVDataset | None = None
        self.val_dataset: Seq2SeqFromCSVDataset | None = None
        self.test_dataset: Seq2SeqFromCSVDataset | None = None
        super().__init__(
            csv_file=csv_file,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            hf_model_id=hf_model_id,
        )

    def setup(self) -> None:
        df = self._read_csv(self.csv_file)
        train_df, val_df, test_df = self._split_df(
            df, self.val_split, self.test_split, self.seed
        )

        try:
            self.train_dataset = Seq2SeqFromCSVDataset(
                train_df, self.input_col, self.target_col
            )
            self.val_dataset = Seq2SeqFromCSVDataset(
                val_df, self.input_col, self.target_col
            )
            self.test_dataset = Seq2SeqFromCSVDataset(
                test_df, self.input_col, self.target_col
            )
        except Exception as e:
            logger.error("Failed to create datasets: %s", e)
            raise

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        except Exception as e:
            logger.error("Failed to load tokenizer from %s: %s", self.hf_model_id, e)
            raise

    def _collate_fn(self, batch: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise AutoMLRuntimeError("Tokenizer not initialized.")
        inputs, targets = zip(*batch)
        src = self.tokenizer(
            list(inputs),
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        tgt = self.tokenizer(
            list(targets),
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        labels = tgt.input_ids.clone()
        # Replace pad token id with -100 so it's ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": src.input_ids,
            "attention_mask": src.attention_mask,
            "labels": labels,
        }

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)


# ---------------------------------------------------------------------------
# Masked language modelling
# ---------------------------------------------------------------------------


class MaskedLMDataModule(BaseDataModule):
    """Datamodule for masked language modelling tasks.

    CSV column: ``text``.  Uses ``DataCollatorForLanguageModeling`` to
    randomly mask tokens at runtime.
    """

    def __init__(
        self,
        csv_file: Path,
        text_col: str = "text",
        mlm_probability: float = 0.15,
        max_length: int = 256,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        val_split: float = DEFAULT_VAL_SPLIT,
        test_split: float = DEFAULT_TEST_SPLIT,
        seed: int = 42,
        hf_model_id: str = "bert-base-uncased",
    ) -> None:
        self.text_col = text_col
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.tokenizer: AutoTokenizer | None = None
        self.data_collator: DataCollatorForLanguageModeling | None = None
        self.train_dataset: CausalLMFromCSVDataset | None = None
        self.val_dataset: CausalLMFromCSVDataset | None = None
        self.test_dataset: CausalLMFromCSVDataset | None = None
        super().__init__(
            csv_file=csv_file,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            hf_model_id=hf_model_id,
        )

    def setup(self) -> None:
        df = self._read_csv(self.csv_file)
        train_df, val_df, test_df = self._split_df(
            df, self.val_split, self.test_split, self.seed
        )

        try:
            # Reuse CausalLMFromCSVDataset as it just returns text strings
            self.train_dataset = CausalLMFromCSVDataset(train_df, self.text_col)
            self.val_dataset = CausalLMFromCSVDataset(val_df, self.text_col)
            self.test_dataset = CausalLMFromCSVDataset(test_df, self.text_col)
        except Exception as e:
            logger.error("Failed to create datasets: %s", e)
            raise

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        except Exception as e:
            logger.error("Failed to load tokenizer from %s: %s", self.hf_model_id, e)
            raise

        try:
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=self.mlm_probability,
            )
        except Exception as e:
            logger.error("Failed to create data collator: %s", e)
            raise

    def _tokenize(self, batch: list[str]) -> Any:
        if self.tokenizer is None:
            raise AutoMLRuntimeError("Tokenizer not initialized.")
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _collate_fn(self, batch: list[str]) -> dict[str, torch.Tensor]:
        encoding = self._tokenize(batch)
        if self.data_collator is None:
            raise AutoMLRuntimeError("Data collator not initialized.")
        # data_collator applies random masking and returns input_ids + labels
        collated = self.data_collator(
            [{"input_ids": ids} for ids in encoding.input_ids]
        )
        return {
            "input_ids": collated["input_ids"],
            "attention_mask": encoding.attention_mask,
            "labels": collated["labels"],
        }

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps task type slug -> datamodule class for lookups by task type.
DATAMODULE_REGISTRY: dict[str, type] = {
    "image_classification": ImageClassificationDataModule,
    "image_segmentation": ImageSegmentationDataModule,
    "object_detection": ObjectDetectionDataModule,
    "video_classification": VideoClassificationDataModule,
    "keypoint_detection": KeypointDetectionDataModule,
    "audio_classification": AudioClassificationDataModule,
    "text_classification": SequenceClassificationDataModule,
    "question_answering": QuestionAnsweringDataModule,
    "causal_lm": CausalLMDataModule,
    "seq2seq_lm": Seq2SeqLMDataModule,
    "masked_lm": MaskedLMDataModule,
}
