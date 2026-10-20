"""Torch datasets that read samples from CSV files for the generic ML engine."""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from app.core.exceptions import AutoMLDataError, AutoMLValidationError

logger = logging.getLogger(__name__)


class BaseCSVDataset(Dataset, ABC):
    """Base class for all CSV-backed datasets.

    Handles loading from a CSV file path or an existing DataFrame with
    uniform error handling, and provides ``__len__`` / index normalization.
    """

    def __init__(self, csv_file: Union[Path, pd.DataFrame]) -> None:
        self.df: pd.DataFrame = self._load_csv(csv_file)

    @staticmethod
    def _load_csv(csv_file: Union[Path, pd.DataFrame]) -> pd.DataFrame:
        """Load a CSV from disk or pass through an existing DataFrame.

        A Path is read with pandas (with typed errors for missing/empty/parsed
        files); a DataFrame is index-reset and used as-is.
        """
        if isinstance(csv_file, Path):
            try:
                return pd.read_csv(csv_file)
            except FileNotFoundError:
                logger.error("Dataset CSV file not found: %s", csv_file)
                raise
            except pd.errors.EmptyDataError:
                logger.error("Dataset CSV file is empty: %s", csv_file)
                raise AutoMLDataError(f"Dataset CSV file is empty: {csv_file}")
            except pd.errors.ParserError as e:
                logger.error("Failed to parse dataset CSV file: %s", e)
                raise
            except Exception as e:
                logger.error("Unexpected error reading dataset CSV file: %s", e)
                raise
        elif isinstance(csv_file, pd.DataFrame):
            return csv_file.reset_index(drop=True)
        else:
            raise AutoMLValidationError("csv_file must be a Path or DataFrame")

    def __len__(self) -> int:
        """Return the number of rows in the underlying DataFrame."""
        return len(self.df)

    @staticmethod
    def _normalize_idx(idx: int) -> int:
        """Convert a tensor index to a plain Python int."""
        if torch.is_tensor(idx):
            idx = idx.item()
        return idx

    @abstractmethod
    def __getitem__(self, idx: int): ...


class ImageClassificationFromCSVDataset(BaseCSVDataset):
    """Torch dataset that reads image paths and labels from a CSV/DataFrame."""

    def __init__(
        self,
        csv_file: Union[Path, pd.DataFrame],
        root_dir: Path,
        img_col: str = "image",
        label_col: str = "label",
        transform: Optional[T.Compose] = None,
        decode_size: Optional[tuple[int, int]] = None,
    ):
        super().__init__(csv_file)
        self.root_dir = Path(root_dir)
        self.img_col = img_col
        self.label_col = label_col
        self.transform = transform
        self.decode_size = decode_size

        label_series = self.df[self.label_col]
        self._use_label_subdir: bool = not pd.api.types.is_numeric_dtype(label_series)

        if self._use_label_subdir:
            self.classes = sorted(label_series.unique().tolist())
            self.class_to_idx = {
                cls_name: idx for idx, cls_name in enumerate(self.classes)
            }
            self.idx_to_class = {
                idx: cls_name for cls_name, idx in self.class_to_idx.items()
            }
            self.df[self.label_col] = self.df[self.label_col].map(self.class_to_idx)
        else:
            raw_vals = sorted(label_series.dropna().unique().tolist())
            self.classes = raw_vals
            self.class_to_idx = {v: i for i, v in enumerate(raw_vals)}
            self.idx_to_class = {i: v for v, i in self.class_to_idx.items()}
            self.df[self.label_col] = self.df[self.label_col].map(self.class_to_idx)

    def __getitem__(self, idx):
        idx = self._normalize_idx(idx)

        row = self.df.iloc[idx]
        label_idx = int(row[self.label_col])

        filename = str(row[self.img_col]).strip().replace("\\", "/")

        if self._use_label_subdir:
            label_name = self.idx_to_class[label_idx]
            img_path = self.root_dir / str(label_name) / filename
        else:
            img_path = self.root_dir / filename

        if not img_path.exists():
            logger.error(
                "Image not found: root_dir=%s, use_label_subdir=%s, filename=%s",
                self.root_dir,
                self._use_label_subdir,
                filename,
            )
            logger.debug("root_dir contents: %s", os.listdir(self.root_dir))
            if self._use_label_subdir:
                logger.debug(
                    "label subdir contents: %s",
                    os.listdir(self.root_dir / str(label_name)),
                )

            raise AutoMLDataError(
                f"Image not found\n"
                f"Expected path: {img_path}\n"
                f"root_dir: {self.root_dir}\n"
                f"use_label_subdir: {self._use_label_subdir}\n"
                f"filename: {repr(filename)}"
            )

        try:
            img = Image.open(img_path)
            if self.decode_size is not None and img.format == "JPEG":
                # Draft mode lets the JPEG decoder downscale on decode
                # (1/2, 1/4, ...) — much cheaper than decoding full size
                # when images are far larger than the processor target.
                img.draft("RGB", self.decode_size)
            img = img.convert("RGB")
        except Exception as e:
            logger.error("Failed to open or convert image %s: %s", img_path, e)
            raise

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label_idx, dtype=torch.long)


class TextClassificationFromCSVDataset(BaseCSVDataset):
    """Torch dataset that reads text and labels from a CSV/DataFrame.

    Expected columns: ``text`` (str) and ``label`` (str or int).
    Returns ``(text, label_idx)`` tuples — the collate function in the
    datamodule applies the tokeniser.
    """

    def __init__(
        self,
        csv_file: Union[Path, pd.DataFrame],
        text_col: str = "text",
        label_col: str = "label",
    ):
        super().__init__(csv_file)
        self.text_col = text_col
        self.label_col = label_col

        if self.df[self.label_col].dtype == object:
            self.classes = sorted(self.df[self.label_col].unique().tolist())
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.df = self.df.copy()
            self.df[self.label_col] = self.df[self.label_col].map(self.class_to_idx)
        else:
            self.classes = sorted(self.df[self.label_col].unique().tolist())
            self.class_to_idx = {c: c for c in self.classes}

    def __getitem__(self, idx: int) -> tuple[str, int]:
        idx = self._normalize_idx(idx)
        row = self.df.iloc[idx]
        return str(row[self.text_col]), int(row[self.label_col])


class QuestionAnsweringFromCSVDataset(BaseCSVDataset):
    """Dataset for extractive QA tasks.

    Expected CSV columns: ``question``, ``context``, ``answer_start`` (int),
    ``answer_text`` (str).  Returns raw strings; the datamodule tokenises them.
    """

    def __init__(
        self,
        csv_file: Union[Path, pd.DataFrame],
        question_col: str = "question",
        context_col: str = "context",
        answer_start_col: str = "answer_start",
        answer_text_col: str = "answer_text",
    ):
        super().__init__(csv_file)
        self.question_col = question_col
        self.context_col = context_col
        self.answer_start_col = answer_start_col
        self.answer_text_col = answer_text_col

    def __getitem__(self, idx: int) -> dict:
        idx = self._normalize_idx(idx)
        row = self.df.iloc[idx]
        return {
            "question": str(row[self.question_col]),
            "context": str(row[self.context_col]),
            "answer_start": int(row[self.answer_start_col]),
            "answer_text": str(row[self.answer_text_col]),
        }


class Seq2SeqFromCSVDataset(BaseCSVDataset):
    """Dataset for sequence-to-sequence tasks.

    Expected CSV columns: ``input_text`` and ``target_text``.
    """

    def __init__(
        self,
        csv_file: Union[Path, pd.DataFrame],
        input_col: str = "input_text",
        target_col: str = "target_text",
    ):
        super().__init__(csv_file)
        self.input_col = input_col
        self.target_col = target_col

    def __getitem__(self, idx: int) -> tuple[str, str]:
        idx = self._normalize_idx(idx)
        row = self.df.iloc[idx]
        return str(row[self.input_col]), str(row[self.target_col])


class CausalLMFromCSVDataset(BaseCSVDataset):
    """Dataset for causal language modelling tasks.

    Expected CSV column: ``text``.  The datamodule tokenises and shifts
    labels automatically.
    """

    def __init__(
        self,
        csv_file: Union[Path, pd.DataFrame],
        text_col: str = "text",
    ):
        super().__init__(csv_file)
        self.text_col = text_col

    def __getitem__(self, idx: int) -> str:
        idx = self._normalize_idx(idx)
        return str(self.df.iloc[idx][self.text_col])


class MultimodalClassificationDataset(BaseCSVDataset):
    """Torch dataset for multimodal image classification with auxiliary tabular features.

    In addition to image + label (like ``ImageClassificationFromCSVDataset``),
    this dataset also returns auxiliary feature values from extra CSV columns.
    Raw values are returned here; encoding/scaling is handled by the datamodule.

    Returns ``(PIL.Image, aux_array, int_label)`` per sample.
    """

    def __init__(
        self,
        csv_file: Union[Path, pd.DataFrame],
        root_dir: Path,
        img_col: str = "filename",
        label_col: str = "label",
        auxiliary_columns: list[str] | None = None,
        transform: Optional[T.Compose] = None,
        decode_size: Optional[tuple[int, int]] = None,
    ):
        super().__init__(csv_file)
        self.root_dir = root_dir
        self.img_col = img_col
        self.label_col = label_col
        self.auxiliary_columns = auxiliary_columns or []
        self.transform = transform
        self.decode_size = decode_size

        if self.df[self.label_col].dtype not in [int, float]:
            self.classes = sorted(self.df[self.label_col].unique().tolist())
            self.class_to_idx = {
                cls_name: idx for idx, cls_name in enumerate(self.classes)
            }
            self.idx_to_class = {
                idx: cls_name for cls_name, idx in self.class_to_idx.items()
            }
            self.df[self.label_col] = self.df[self.label_col].map(self.class_to_idx)
        else:
            self.classes = sorted(self.df[self.label_col].unique().tolist())
            self.class_to_idx = {cls: cls for cls in self.classes}
            self.idx_to_class = {cls: cls for cls in self.classes}

    def __getitem__(self, idx):
        idx = self._normalize_idx(idx)

        row = self.df.iloc[idx]
        label_idx = int(row[self.label_col])
        label_name = str(self.idx_to_class[label_idx])

        filename = str(row[self.img_col]).strip()

        img_path = self.root_dir / label_name / filename
        if not img_path.exists():
            logger.error(
                "Image not found: root_dir=%s, label_name=%s, filename=%s",
                self.root_dir,
                label_name,
                filename,
            )
            raise AutoMLDataError(
                f"Image not found\n"
                f"Expected path: {img_path}\n"
                f"root_dir: {self.root_dir}\n"
                f"label_name: {label_name}\n"
                f"filename: {repr(filename)}"
            )

        try:
            img = Image.open(img_path)
            if self.decode_size is not None and img.format == "JPEG":
                img.draft("RGB", self.decode_size)
            img = img.convert("RGB")
        except Exception as e:
            logger.error("Failed to open or convert image %s: %s", img_path, e)
            raise

        if self.transform:
            img = self.transform(img)

        if self.auxiliary_columns:
            aux_values = np.array(
                [row[col] for col in self.auxiliary_columns], dtype=np.float32
            )
        else:
            aux_values = np.array([], dtype=np.float32)

        return img, aux_values, torch.tensor(label_idx, dtype=torch.long)
