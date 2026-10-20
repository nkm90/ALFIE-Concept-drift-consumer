"""Per-task Optuna objective functions for the generic ML engine.

Each objective trains one candidate model for a trial (sampling model id,
learning rate, batch size, and weight decay from the task config) and returns
its best validation loss. ``OBJECTIVE_REGISTRY`` maps task type slugs to their
objective so ``run_optuna_search`` can dispatch dynamically.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable

import optuna
from optuna.distributions import CategoricalChoiceType
from torch import nn

from app.ml_engine import (
    EarlyStopping,
    FabricTrainer,
    MultimodalClassificationDataModule,
    MultimodalClassificationModel,
)
from app.ml_engine.datamodule import (
    AudioClassificationDataModule,
    CausalLMDataModule,
    ImageClassificationDataModule,
    ImageSegmentationDataModule,
    KeypointDetectionDataModule,
    MaskedLMDataModule,
    ObjectDetectionDataModule,
    QuestionAnsweringDataModule,
    Seq2SeqLMDataModule,
    SequenceClassificationDataModule,
    VideoClassificationDataModule,
    auto_num_workers,
)
from app.ml_engine.model import (
    AudioClassificationModel,
    CausalLMModel,
    ImageClassificationModel,
    ImageSegmentationModel,
    KeypointDetectionModel,
    MaskedLMModel,
    ObjectDetectionModel,
    QuestionAnsweringModel,
    Seq2SeqLMModel,
    SequenceClassificationModel,
    VideoClassificationModel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-task Optuna objective functions
# ---------------------------------------------------------------------------


def _trial_dir(workdir: Path | None, trial: optuna.Trial) -> Path | None:
    """Return this trial's artifact directory under workdir/optuna, if any."""
    if workdir is None:
        return None
    return Path(workdir) / "optuna" / f"trial_{trial.number}"


def _save_trial_feature_mapping(
    trial_dir: Path, datamodule: Any, task_type: str, model: Any
) -> None:
    """Persist feature_mapping.json for one trial. Best-effort; never raises."""
    try:
        # Lazy import keeps the HPO objectives decoupled from the engine's
        # feature-matching module at load time.
        from app.ml_engine.feature_mapping import extract_feature_mapping

        mapping = extract_feature_mapping(datamodule, task_type, model=model)
        trial_dir.mkdir(parents=True, exist_ok=True)
        with open(trial_dir / "feature_mapping.json", "w") as f:
            json.dump(mapping, f, indent=2, sort_keys=True)
        logger.debug("Wrote feature_mapping.json to %s", trial_dir)
    except Exception as e:
        logger.warning("Failed to save feature mapping for trial: %s", e)


def _save_trial_model(trial_dir: Path, model: Any) -> None:
    """Persist model.pt for one trial. Best-effort; never raises."""
    try:
        import torch

        trial_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model, trial_dir / "model.pt")
        logger.debug("Wrote model.pt to %s", trial_dir)
    except Exception as e:
        logger.warning("Failed to save model for trial: %s", e)


def _optuna_objective_base(
    trial: optuna.Trial,
    *,
    model_size: str,
    config: dict[str, Any],
    datamodule_class: type,
    dm_kwargs: dict[str, Any],
    model_class: type[nn.Module],
    build_model_kwargs: Callable[[Any, str], dict[str, Any]],
    model_computes_loss: bool = False,
    loss_fn: nn.Module | None = None,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
    workdir: Path | None = None,
    task_type: str | None = None,
) -> float:
    """Shared trial body used by every per-task objective.

    Suggests hyperparameters (model id, learning rate, batch size, weight
    decay) from the task config, builds the given datamodule/model pair, saves
    the trial's feature mapping before training (so it survives pruning), and
    saves the model after a successful fit. Returns the best validation loss
    Optuna minimizes; the test split is never touched during the search.
    """
    models: list[str] = config[f"{model_size}_models"]
    model_id: str = str(trial.suggest_categorical("model_id", models))
    lr: float = trial.suggest_float("lr", config["lr_low"], config["lr_high"], log=True)
    batch_size: CategoricalChoiceType = trial.suggest_categorical(
        "batch_size", config["batch_sizes"]
    )
    weight_decay: float = trial.suggest_float(
        "weight_decay",
        config["weight_decay_low"],
        config["weight_decay_high"],
        log=True,
    )

    resolved_cpus: int | None = num_cpus if isinstance(num_cpus, int) else None
    # "auto" CPUs still get parallel data loading — image decode / feature
    # extraction otherwise runs inline with training and starves the device.
    num_workers: int = (
        resolved_cpus if resolved_cpus is not None else auto_num_workers()
    )

    dm_kwargs_full: dict[str, Any] = dict(dm_kwargs)
    dm_kwargs_full["num_workers"] = num_workers

    datamodule = datamodule_class(
        **dm_kwargs_full,
        batch_size=batch_size,
        hf_model_id=model_id,
    )

    trainer_kwargs: dict[str, Any] = {
        "datamodule": datamodule,
        "model_class": model_class,
        "model_kwargs": build_model_kwargs(datamodule, model_id),
        "optimizer_kwargs": {"lr": lr, "weight_decay": weight_decay},
        "epochs": config["max_epochs"],
        "callbacks": [EarlyStopping(patience=config["early_stopping_patience"])],
        "model_computes_loss": model_computes_loss,
        "device": num_gpus,
        "num_threads": resolved_cpus,
    }
    if loss_fn is not None:
        trainer_kwargs["loss_fn"] = loss_fn

    trainer = FabricTrainer(**trainer_kwargs)

    # Feature mapping needs both datamodule (preprocessing state) and the model
    # (for vision_embed_dim on multimodal). Save before fit so it's available
    # even if the trial is later pruned.
    trial_dir = _trial_dir(workdir, trial)
    if trial_dir is not None and task_type is not None:
        _save_trial_feature_mapping(trial_dir, datamodule, task_type, trainer.model)

    best_val_loss: float
    best_val_loss, _ = trainer.fit(trial=trial)

    # Reached only on successful completion (TrialPruned raises mid-fit).
    if trial_dir is not None:
        _save_trial_model(trial_dir, trainer.model)

    return best_val_loss


def optuna_objective_image_classification(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
    workdir: Path | None = None,
    task_type: str | None = None,
) -> float:
    """Optuna objective for image classification; returns the best validation loss."""
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        config=config,
        datamodule_class=ImageClassificationDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "root_dir": images_dir,
            "img_col": filename_column,
            "label_col": label_column,
        },
        model_class=ImageClassificationModel,
        build_model_kwargs=lambda dm, mid: {
            "model_id": mid,
            "num_classes": dm.num_classes,
            "id2label": dm.id2label,
            "label2id": dm.label2id,
        },
        model_computes_loss=False,
        loss_fn=nn.CrossEntropyLoss(),
        workdir=workdir,
        task_type=task_type,
    )


def optuna_objective_image_classification_multimodal(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    auxiliary_columns: list[str],
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
    workdir: Path | None = None,
    task_type: str | None = None,
) -> float:
    """Optuna objective for multimodal image classification; returns the best validation loss."""
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        config=config,
        datamodule_class=MultimodalClassificationDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "root_dir": images_dir,
            "img_col": filename_column,
            "label_col": label_column,
            "auxiliary_columns": auxiliary_columns,
        },
        model_class=MultimodalClassificationModel,
        build_model_kwargs=lambda dm, mid: {
            "model_id": mid,
            "num_classes": dm.num_classes,
            "aux_feature_dim": dm.aux_feature_dim,
            "id2label": dm.id2label,
            "label2id": dm.label2id,
        },
        model_computes_loss=False,
        loss_fn=nn.CrossEntropyLoss(),
        workdir=workdir,
        task_type=task_type,
    )


def optuna_objective_image_segmentation(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
    workdir: Path | None = None,
    task_type: str | None = None,
) -> float:
    """Optuna objective for image segmentation; returns the best validation loss."""
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        config=config,
        datamodule_class=ImageSegmentationDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "root_dir": images_dir,
            "img_col": filename_column,

            # For image segmentation the generic
            # Vision API label_column identifies
            # the pixel-mask filename column.
            "mask_col": label_column,
        },
        model_class=ImageSegmentationModel,
        build_model_kwargs=lambda dm, mid: {
            "model_id": mid,
            "num_classes": dm.num_classes,
            "id2label": dm.id2label,
            "label2id": dm.label2id,
        },
        model_computes_loss=True,
        workdir=workdir,
        task_type=task_type,
    )


def optuna_objective_object_detection(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
    workdir: Path | None = None,
    task_type: str | None = None,
) -> float:
    """Optuna objective for object detection; returns the best validation loss."""
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        config=config,
        datamodule_class=ObjectDetectionDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "root_dir": images_dir,
            "img_col": filename_column,
        },
        model_class=ObjectDetectionModel,
        build_model_kwargs=lambda dm, mid: {
            "model_id": mid,
            "num_classes": dm.num_classes,
        },
        model_computes_loss=True,
        workdir=workdir,
        task_type=task_type,
    )


def optuna_objective_video_classification(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
    workdir: Path | None = None,
    task_type: str | None = None,
) -> float:
    """Optuna objective for video classification; returns the best validation loss."""
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        config=config,
        datamodule_class=VideoClassificationDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "root_dir": images_dir,
            "video_col": filename_column,
            "label_col": label_column,
        },
        model_class=VideoClassificationModel,
        build_model_kwargs=lambda dm, mid: {
            "model_id": mid,
            "num_classes": dm.num_classes,
            "id2label": dm.id2label,
            "label2id": dm.label2id,
        },
        model_computes_loss=False,
        loss_fn=nn.CrossEntropyLoss(),
        workdir=workdir,
        task_type=task_type,
    )


def optuna_objective_keypoint_detection(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    images_dir: Path,
    filename_column: str,
    label_column: str,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
    workdir: Path | None = None,
    task_type: str | None = None,
) -> float:
    """Optuna objective for keypoint detection; returns the best validation loss."""
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        config=config,
        datamodule_class=KeypointDetectionDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "root_dir": images_dir,
            "img_col": filename_column,
            "label_col": label_column,
        },
        model_class=KeypointDetectionModel,
        build_model_kwargs=lambda dm, mid: {"model_id": mid},
        model_computes_loss=True,
        workdir=workdir,
        task_type=task_type,
    )


def optuna_objective_audio_classification(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    audio_dir: Path | None = None,
    filename_column: str,
    label_column: str,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
    workdir: Path | None = None,
    task_type: str | None = None,
    **_kwargs: Any,
) -> float:
    """Optuna objective for audio classification; returns the best validation loss.

    ``audio_dir`` may also arrive via the generic ``images_dir`` kwarg that
    ``run_optuna_search`` passes for every media-backed task.
    """
    resolved_audio_dir = audio_dir or _kwargs.get("images_dir")
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        config=config,
        datamodule_class=AudioClassificationDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "root_dir": resolved_audio_dir,
            "audio_col": filename_column,
            "label_col": label_column,
        },
        model_class=AudioClassificationModel,
        build_model_kwargs=lambda dm, mid: {
            "model_id": mid,
            "num_classes": dm.num_classes,
            "id2label": dm.id2label,
            "label2id": dm.label2id,
        },
        model_computes_loss=False,
        loss_fn=nn.CrossEntropyLoss(),
        workdir=workdir,
        task_type=task_type,
    )


def optuna_objective_text_classification(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    text_column: str,
    label_column: str,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
    workdir: Path | None = None,
    task_type: str | None = None,
    **_kwargs: Any,
) -> float:
    """Optuna objective for text classification; returns the best validation loss."""
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        config=config,
        datamodule_class=SequenceClassificationDataModule,
        dm_kwargs={
            "csv_file": csv_path,
            "text_col": text_column,
            "label_col": label_column,
        },
        model_class=SequenceClassificationModel,
        build_model_kwargs=lambda dm, mid: {
            "model_id": mid,
            "num_classes": dm.num_classes,
            "id2label": dm.id2label,
            "label2id": dm.label2id,
        },
        model_computes_loss=False,
        loss_fn=nn.CrossEntropyLoss(),
        workdir=workdir,
        task_type=task_type,
    )


def optuna_objective_question_answering(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
    workdir: Path | None = None,
    task_type: str | None = None,
    **_kwargs: Any,
) -> float:
    """Optuna objective for extractive question answering; returns the best validation loss."""
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        config=config,
        datamodule_class=QuestionAnsweringDataModule,
        dm_kwargs={"csv_file": csv_path},
        model_class=QuestionAnsweringModel,
        build_model_kwargs=lambda dm, mid: {"model_id": mid},
        model_computes_loss=True,
        workdir=workdir,
        task_type=task_type,
    )


def optuna_objective_causal_lm(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
    workdir: Path | None = None,
    task_type: str | None = None,
    **_kwargs: Any,
) -> float:
    """Optuna objective for causal language modelling; returns the best validation loss."""
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        config=config,
        datamodule_class=CausalLMDataModule,
        dm_kwargs={"csv_file": csv_path},
        model_class=CausalLMModel,
        build_model_kwargs=lambda dm, mid: {"model_id": mid},
        model_computes_loss=True,
        workdir=workdir,
        task_type=task_type,
    )


def optuna_objective_seq2seq_lm(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
    workdir: Path | None = None,
    task_type: str | None = None,
    **_kwargs: Any,
) -> float:
    """Optuna objective for seq2seq language modelling; returns the best validation loss."""
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        config=config,
        datamodule_class=Seq2SeqLMDataModule,
        dm_kwargs={"csv_file": csv_path},
        model_class=Seq2SeqLMModel,
        build_model_kwargs=lambda dm, mid: {"model_id": mid},
        model_computes_loss=True,
        workdir=workdir,
        task_type=task_type,
    )


def optuna_objective_masked_lm(
    trial: optuna.Trial,
    *,
    csv_path: Path,
    model_size: str,
    timeout_per_trial: float | None,
    config: dict,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
    workdir: Path | None = None,
    task_type: str | None = None,
    **_kwargs: Any,
) -> float:
    """Optuna objective for masked language modelling; returns the best validation loss."""
    return _optuna_objective_base(
        trial=trial,
        model_size=model_size,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        config=config,
        datamodule_class=MaskedLMDataModule,
        dm_kwargs={"csv_file": csv_path},
        model_class=MaskedLMModel,
        build_model_kwargs=lambda dm, mid: {"model_id": mid},
        model_computes_loss=True,
        workdir=workdir,
        task_type=task_type,
    )


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

# Maps task type slug -> objective function; used by run_optuna_search.
OBJECTIVE_REGISTRY: dict[str, Callable] = {
    "image_classification": optuna_objective_image_classification,
    "image_classification_multimodal": optuna_objective_image_classification_multimodal,
    "image_segmentation": optuna_objective_image_segmentation,
    "object_detection": optuna_objective_object_detection,
    "video_classification": optuna_objective_video_classification,
    "keypoint_detection": optuna_objective_keypoint_detection,
    "audio_classification": optuna_objective_audio_classification,
    "text_classification": optuna_objective_text_classification,
    "question_answering": optuna_objective_question_answering,
    "causal_lm": optuna_objective_causal_lm,
    "seq2seq_lm": optuna_objective_seq2seq_lm,
    "masked_lm": optuna_objective_masked_lm,
}
optuna_objective = optuna_objective_image_classification
