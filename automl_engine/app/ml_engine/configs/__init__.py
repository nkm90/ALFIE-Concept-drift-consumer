"""Per-task hyperparameter and model config loader."""

import json
from pathlib import Path

_CONFIGS_DIR = Path(__file__).parent

SUPPORTED_TASK_TYPES: frozenset[str] = frozenset(
    {
        "image_classification",
        "image_classification_multimodal",
        "image_segmentation",
        "object_detection",
        "video_classification",
        "keypoint_detection",
        "audio_classification",
        "text_classification",
        "question_answering",
        "causal_lm",
        "seq2seq_lm",
        "masked_lm",
    }
)


def load_task_config(task_type: str) -> dict:
    """Load and return the JSON config for the given task type.

    Args:
        task_type: One of the supported task type slugs.

    Returns:
        Dict with keys: small_models, medium_models, large_models,
        lr_low, lr_high, batch_sizes, weight_decay_low, weight_decay_high,
        max_epochs, early_stopping_patience.

    Raises:
        ValueError: If the task type is not supported.
    """
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"Unknown task type '{task_type}'. "
            f"Supported: {sorted(SUPPORTED_TASK_TYPES)}"
        )
    config_path = _CONFIGS_DIR / f"{task_type}.json"
    with open(config_path) as f:
        return json.load(f)
