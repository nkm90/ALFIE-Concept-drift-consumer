"""Feature / embedding mapping extraction for trained models.

``extract_feature_mapping`` captures the fitted preprocessing state of a
datamodule (plus model-side state where relevant) so the exact feature
pipeline can be rebuilt at inference time. The HPO objectives persist one
mapping per trial as ``feature_mapping.json`` next to the saved model.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_TEXT_TASK_TYPES: frozenset[str] = frozenset(
    {
        "text_classification",
        "question_answering",
        "causal_lm",
        "seq2seq_lm",
        "masked_lm",
    }
)


def extract_feature_mapping(datamodule: Any, task_type: str, model: Any = None) -> dict:
    """Extract a task-appropriate feature/embedding mapping from a fitted datamodule.

    Captures:
      * ``label_map`` (``id2label`` / ``label2id``) for any classification
        datamodule that exposes them.
      * ``tokenizer`` vocab (token -> id) plus the underlying ``hf_model_id``
        for text tasks.
      * ``auxiliary_features`` for multimodal: numeric/categorical column
        split, fitted ``StandardScaler`` and ``OrdinalEncoder`` state,
        ``aux_feature_dim``, and ``vision_embed_dim`` when *model* exposes
        ``_get_vision_embed_dim``.

    Sections are omitted (not empty) when the corresponding state is missing,
    so consumers can branch on key presence rather than truthiness.
    """
    mapping: dict = {
        "task_type": task_type,
        "label_map": _extract_label_map(datamodule),
    }

    if task_type in _TEXT_TASK_TYPES:
        mapping["tokenizer"] = _extract_tokenizer_vocab(datamodule)

    if task_type == "image_classification_multimodal":
        mapping["auxiliary_features"] = _extract_auxiliary_features(datamodule, model)

    return mapping


def _extract_label_map(datamodule: Any) -> dict:
    """Return the id<->label maps of a datamodule with string keys for JSON."""
    id2label = getattr(datamodule, "id2label", None) or {}
    label2id = getattr(datamodule, "label2id", None) or {}
    if not id2label and not label2id:
        return {}
    # JSON requires string keys.
    return {
        "id2label": {str(k): v for k, v in id2label.items()},
        "label2id": {str(k): v for k, v in label2id.items()},
    }


def _extract_tokenizer_vocab(datamodule: Any) -> dict:
    """Return the tokenizer vocabulary and source model id for text tasks."""
    tokenizer = getattr(datamodule, "tokenizer", None)
    if tokenizer is None:
        logger.warning(
            "Text-task datamodule has no fitted tokenizer; skipping vocab extraction"
        )
        return {}
    try:
        vocab = tokenizer.get_vocab()
    except Exception as e:
        logger.warning(f"Failed to read tokenizer vocab: {e}")
        return {}
    return {
        "hf_model_id": getattr(datamodule, "hf_model_id", None),
        "vocab": dict(vocab),
    }


def _extract_auxiliary_features(datamodule: Any, model: Any) -> dict:
    """Capture fitted preprocessing state for multimodal auxiliary features.

    Records the column splits, scaler/encoder parameters, and (when the model
    exposes it) the vision embedding dimension, so consumers can rebuild the
    exact feature pipeline at inference time.
    """
    out: dict = {
        "auxiliary_columns": list(getattr(datamodule, "auxiliary_columns", []) or []),
        "numeric_columns": list(getattr(datamodule, "numeric_cols", []) or []),
        "categorical_columns": list(getattr(datamodule, "categorical_cols", []) or []),
        "aux_feature_dim": int(getattr(datamodule, "aux_feature_dim", 0) or 0),
        "scaler": _extract_scaler_state(getattr(datamodule, "scaler", None)),
        "ordinal_encoder": _extract_encoder_state(getattr(datamodule, "encoder", None)),
    }
    if model is not None and hasattr(model, "_get_vision_embed_dim"):
        try:
            out["vision_embed_dim"] = int(model._get_vision_embed_dim())
        except Exception as e:
            logger.warning(f"Failed to read vision_embed_dim from model: {e}")
    return out


def _extract_scaler_state(scaler: Any) -> dict:
    """Serialize the fitted StandardScaler's learned statistics."""
    if scaler is None:
        return {}
    out: dict = {"n_features_in": int(getattr(scaler, "n_features_in_", 0) or 0)}
    for attr in ("mean_", "scale_", "var_"):
        arr = getattr(scaler, attr, None)
        if arr is None:
            continue
        try:
            out[attr.removesuffix("_")] = list(arr.tolist())
        except Exception:
            out[attr.removesuffix("_")] = list(arr)
    return out


def _extract_encoder_state(encoder: Any) -> dict:
    """Serialize the fitted OrdinalEncoder's learned categories."""
    if encoder is None:
        return {}
    categories = getattr(encoder, "categories_", None) or []
    encoded: list[list[str]] = []
    for arr in categories:
        try:
            encoded.append([str(c) for c in arr.tolist()])
        except Exception:
            encoded.append([str(c) for c in list(arr)])
    return {"categories": encoded}
