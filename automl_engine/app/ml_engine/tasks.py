"""Task configuration models and supported task-type slugs for the ML engine.

The generic engine supports image, video, audio, and text tasks; the tabular
(AutoGluon) task models live in ``app.ml_engine.model`` alongside the HF
model wrappers. Endpoints import these slugs to validate their ``task_type``
form parameters.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ImageTask(BaseModel):
    """Base Pydantic model describing common image task inputs."""

    train_dir: Path
    test_dir: Path | None = None
    label_format: Literal["folder", "csv"] = "folder"
    labels_file: Path | None = None  # used if label_format != 'folder'

    class Config:
        arbitrary_types_allowed = True


class TextTask(BaseModel):
    """Base Pydantic model for text-based tasks."""

    data_file: Path  # CSV with the required columns for the task type

    class Config:
        arbitrary_types_allowed = True


class ImageClassificationTask(ImageTask):
    """Configuration for single-label image classification tasks."""

    task_type: Literal["image_classification"] = "image_classification"


class ImageSegmentationTask(ImageTask):
    """Configuration for semantic/panoptic image segmentation tasks."""

    task_type: Literal["image_segmentation"] = "image_segmentation"


class ObjectDetectionTask(ImageTask):
    """Configuration for object detection tasks.

    The labels CSV must include ``boxes`` and ``class_labels`` columns
    (JSON-encoded lists per row).
    """

    task_type: Literal["object_detection"] = "object_detection"
    label_format: Literal["csv"] = "csv"


class VideoClassificationTask(ImageTask):
    """Configuration for video classification tasks.

    The labels CSV must include a ``video_path`` column pointing to video
    files relative to ``train_dir``.
    """

    task_type: Literal["video_classification"] = "video_classification"
    label_format: Literal["csv"] = "csv"


class KeypointDetectionTask(ImageTask):
    """Configuration for keypoint detection tasks.

    The labels CSV must include a ``keypoints`` column with JSON-encoded
    ``[x, y, visibility]`` lists.
    """

    task_type: Literal["keypoint_detection"] = "keypoint_detection"


class AudioClassificationTask(BaseModel):
    """Configuration for audio classification tasks.

    ``audio_dir`` is the root directory containing audio files.
    ``labels_file`` is a CSV with ``audio_path`` and ``label`` columns.
    """

    task_type: Literal["audio_classification"] = "audio_classification"
    audio_dir: Path
    labels_file: Path

    class Config:
        arbitrary_types_allowed = True


class SequenceClassificationTask(TextTask):
    """Configuration for text sequence classification tasks.

    CSV must have ``text`` and ``label`` columns.
    """

    task_type: Literal["text_classification"] = "text_classification"


class QuestionAnsweringTask(TextTask):
    """Configuration for extractive question answering tasks.

    CSV must have ``question``, ``context``, ``answer_start``, and
    ``answer_text`` columns.
    """

    task_type: Literal["question_answering"] = "question_answering"


class CausalLMTask(TextTask):
    """Configuration for causal language modelling tasks.

    CSV must have a ``text`` column.
    """

    task_type: Literal["causal_lm"] = "causal_lm"


class Seq2SeqLMTask(TextTask):
    """Configuration for sequence-to-sequence tasks.

    CSV must have ``input_text`` and ``target_text`` columns.
    """

    task_type: Literal["seq2seq_lm"] = "seq2seq_lm"


class MaskedLMTask(TextTask):
    """Configuration for masked language modelling tasks.

    CSV must have a ``text`` column.
    """

    task_type: Literal["masked_lm"] = "masked_lm"


SUPPORTED_VISION_TASK_TYPES: frozenset[str] = frozenset(
    {
        "image_classification",
        "image_segmentation",
        "object_detection",
        "video_classification",
        "keypoint_detection",
    }
)

SUPPORTED_AUDIO_TASK_TYPES: frozenset[str] = frozenset(
    {
        "audio_classification",
    }
)

SUPPORTED_TEXT_TASK_TYPES: frozenset[str] = frozenset(
    {
        "text_classification",
        "question_answering",
        "causal_lm",
        "seq2seq_lm",
        "masked_lm",
    }
)
