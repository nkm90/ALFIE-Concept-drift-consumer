"""Training loops for the consolidated ML engine.

Contains the generic engine's Lightning Fabric training loop and Optuna
search (``FabricTrainer`` / ``run_optuna_search``) plus the AutoGluon-based
tabular trainer (``AutoGluonTrainer``).
"""

import functools
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any, cast

import lightning as L
import optuna
import pandas as pd
import torch
from autogluon.tabular import TabularDataset, TabularPredictor
from dotenv import find_dotenv, load_dotenv
from torch import nn, optim
from tqdm import tqdm

from app.core.config import get_settings
from app.core.exceptions import (
    AutoMLConfigError,
    AutoMLDataError,
    AutoMLLeaderboardError,
    AutoMLTrainingError,
)
from app.ml_engine.configs import load_task_config

# Backward-compat import alias kept for external code that references
# ClassificationData / ClassificationModel directly.
from app.ml_engine.model import ImageClassificationModel  # noqa: F401

logger = logging.getLogger(__name__)

# Keys whose tensors should be treated as target/label dtype
_TARGET_KEYS: frozenset[str] = frozenset({"labels", "start_positions", "end_positions"})


def _resolve_precision(precision: str) -> str:
    """Resolve the Fabric precision string, honouring "auto".

    "auto" picks bf16-mixed on CUDA hardware that supports it, 16-mixed on
    older CUDA GPUs, and full precision everywhere else (CPU / MPS).
    """
    if precision != "auto":
        return precision
    if torch.cuda.is_available():
        return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    return "32-true"


class EarlyStopping:
    """Simple early stopping callback based on monitored metric."""

    def __init__(
        self, monitor: str = "val_loss", patience: int = 3, min_delta: float = 0.0
    ) -> None:
        self.monitor: str = monitor
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.best: float = float("inf")
        self.counter: int = 0

    def on_epoch_end(
        self, trainer: "FabricTrainer", epoch: int, logs: dict[str, float]
    ) -> None:
        """Update state after epoch; may signal stopping on trainer."""
        current: float | None = logs.get(self.monitor)
        if current is None:
            logger.warning(
                f"Metric '{self.monitor}' not found in logs. Skipping early stopping check."
            )
            return

        if current < self.best - self.min_delta:
            self.best = current
            self.counter = 0
            logger.info(f"New best {self.monitor}: {self.best:.4f}")
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                logger.info("Early stopping triggered!")
                trainer.epochs = epoch + 1


class FabricTrainer:
    """Minimal trainer using Lightning Fabric.

    Supports both:
    - Classification tasks where the model returns logits and the trainer
      computes the loss via ``loss_fn`` (``model_computes_loss=False``).
    - Generative / structured-prediction tasks where the model computes
      its own loss internally and returns a scalar tensor
      (``model_computes_loss=True``).
    """

    def __init__(
        self,
        datamodule: Any,
        model_class: type[nn.Module],
        model_kwargs: dict[str, Any] | None = None,
        optimizer_class: type[optim.Optimizer] = optim.AdamW,
        optimizer_kwargs: dict[str, Any] | None = None,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        lr: float = 0.001,
        epochs: int = 1,
        time_limit: float | None = None,
        device: str | int = "auto",
        num_threads: int | None = None,
        callbacks: list[Any] | None = None,
        input_dtype: torch.dtype = torch.float32,
        target_dtype: torch.dtype = torch.long,
        model_computes_loss: bool = False,
        precision: str | None = None,
    ) -> None:
        self.datamodule: Any = datamodule
        self.model_class: type[nn.Module] = model_class
        self.model_kwargs: dict[str, Any] = model_kwargs or {}
        self.optimizer_class: type[optim.Optimizer] = optimizer_class
        self.optimizer_kwargs: dict[str, Any] = optimizer_kwargs or {"lr": lr}
        self.loss_fn: nn.Module = loss_fn
        self.epochs: int = epochs
        self.time_limit: float | None = time_limit
        self.device: str | int = device
        self.callbacks: list[Any] = callbacks or []
        self.input_dtype: torch.dtype = input_dtype
        self.target_dtype: torch.dtype = target_dtype
        self.model_computes_loss: bool = model_computes_loss

        if num_threads is not None:
            torch.set_num_threads(num_threads)

        self._configure_backend()
        self.precision: str = _resolve_precision(
            precision if precision is not None else get_settings().training_precision
        )
        logger.info("Training with precision: %s", self.precision)
        self.fabric: L.Fabric = L.Fabric(
            devices=self.device, precision=cast(Any, self.precision)
        )
        self._setup_model_optimizer()

    @staticmethod
    def _configure_backend() -> None:
        """Enable fast matmul paths (TF32 on Ampere+, cuDNN autotuning)."""
        try:
            torch.set_float32_matmul_precision("high")
        except Exception as e:  # pragma: no cover - very old torch versions
            logger.debug("Could not set float32 matmul precision: %s", e)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def _setup_model_optimizer(self) -> None:
        """Instantiate model and optimizer and prepare loaders with Fabric."""
        logger.info("Setting up model and optimizer.")
        self.model: nn.Module = self.model_class(**self.model_kwargs)
        self.optimizer: optim.Optimizer = self.optimizer_class(
            self.model.parameters(), **self.optimizer_kwargs
        )

        train_loader: Any = self.datamodule.train_dataloader()
        val_loader: Any = self.datamodule.val_dataloader()
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
        ) = self.fabric.setup(self.model, self.optimizer, train_loader, val_loader)
        self.test_loader: Any = self.datamodule.test_dataloader()
        logger.info("Model and optimizer setup complete.")

    def _move_batch(self, batch: Any) -> dict[str, Any]:
        """Move batch to the Fabric device.

        Handles arbitrary dict batches (all modalities) and legacy
        ``(images, labels)`` tuple batches.  Non-tensor values (e.g. list
        of annotation dicts for object detection) are passed through as-is.
        Integer tensors (``input_ids``, etc.) are moved without dtype coercion.
        """
        if isinstance(batch, dict):
            moved: dict[str, Any] = {}
            for k, v in batch.items():
                if not isinstance(v, torch.Tensor):
                    moved[k] = v  # keep non-tensors (e.g. list of dicts)
                elif k in _TARGET_KEYS:
                    moved[k] = v.to(
                        self.fabric.device, dtype=self.target_dtype, non_blocking=True
                    )
                elif v.dtype.is_floating_point:
                    moved[k] = v.to(
                        self.fabric.device, dtype=self.input_dtype, non_blocking=True
                    )
                else:
                    # int/long tensors (input_ids, etc.) — preserve dtype
                    moved[k] = v.to(self.fabric.device, non_blocking=True)
            return moved
        else:
            imgs, batch_labels = batch
            return {
                "pixel_values": imgs.to(
                    self.fabric.device, dtype=self.input_dtype, non_blocking=True
                ),
                "labels": batch_labels.to(
                    self.fabric.device, dtype=self.target_dtype, non_blocking=True
                ),
            }

    def _check_time_limit(self, start_time: float) -> bool:
        """Return True if configured time limit has been exceeded."""
        elapsed: float = time.time() - start_time
        if self.time_limit and elapsed > self.time_limit:
            logger.warning(f"Time limit reached ({elapsed:.2f}s). Stopping training.")
            return True
        return False

    def _compute_loss_and_logits(
        self, moved: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run forward pass, return (loss, logits_or_None).

        Under mixed precision the module forward runs inside Fabric's
        autocast context, but this loss computation does not — upcast the
        logits so the loss is computed in float32.
        """
        if self.model_computes_loss:
            outputs = self.model(**moved)
            loss = outputs if isinstance(outputs, torch.Tensor) else outputs.loss
            return loss, None
        else:
            labels = moved.pop("labels")
            outputs = self.model(**moved)
            loss = self.loss_fn(outputs.float(), labels)
            return loss, outputs

    def train_epoch(self, epoch: int, start_time: float) -> float:
        """Train for a single epoch and return average training loss."""
        self.model.train()
        # Accumulate on-device and sync once per epoch (per-batch .item()
        # would force a device synchronization every step).
        loss_sum = torch.zeros((), device=self.fabric.device)
        batch_count: int = 0

        for batch in tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1} Training", leave=False
        ):
            if self._check_time_limit(start_time):
                break

            moved = self._move_batch(batch)
            self.optimizer.zero_grad()
            loss, _ = self._compute_loss_and_logits(moved)
            self.fabric.backward(loss)
            self.optimizer.step()
            loss_sum += loss.detach().float()
            batch_count += 1

        avg_loss: float = (loss_sum / max(1, batch_count)).item()
        if batch_count:
            logger.info(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, start_time: float) -> tuple[float, float]:
        """Evaluate on validation set; return (avg_loss, accuracy)."""
        self.model.eval()
        val_loss = torch.zeros((), device=self.fabric.device)
        correct = torch.zeros((), device=self.fabric.device)
        total: int = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                if self._check_time_limit(start_time):
                    break

                moved = self._move_batch(batch)

                if self.model_computes_loss:
                    outputs = self.model(**moved)
                    loss = (
                        outputs if isinstance(outputs, torch.Tensor) else outputs.loss
                    )
                    val_loss += loss.detach().float()
                else:
                    labels = moved.pop("labels")
                    outputs = self.model(**moved)
                    loss = self.loss_fn(outputs.float(), labels)
                    val_loss += loss.detach().float()
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum()
                    total += labels.size(0)

        avg_loss: float = (val_loss / max(1, len(self.val_loader))).item()
        accuracy: float = (correct / max(1, total)).item() if total else 0.0
        logger.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def test(self) -> tuple[float, float]:
        """Evaluate on test set; return (avg_loss, accuracy)."""
        self.model.eval()
        test_loss = torch.zeros((), device=self.fabric.device)
        correct = torch.zeros((), device=self.fabric.device)
        total: int = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                moved = self._move_batch(batch)

                if self.model_computes_loss:
                    outputs = self.model(**moved)
                    loss = (
                        outputs if isinstance(outputs, torch.Tensor) else outputs.loss
                    )
                    test_loss += loss.detach().float()
                else:
                    labels = moved.pop("labels")
                    outputs = self.model(**moved)
                    loss = self.loss_fn(outputs.float(), labels)
                    test_loss += loss.detach().float()
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum()
                    total += labels.size(0)

        avg_loss: float = (test_loss / len(self.test_loader)).item()
        accuracy: float = (correct / max(1, total)).item() if total else 0.0
        logger.info(f"Test Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def fit(self, trial: optuna.Trial | None = None) -> tuple[float, float]:
        """Run the full train/validate loop; return best (val_loss, val_acc).

        Each epoch reports validation loss back to the Optuna trial (so bad
        trials get pruned early), gives callbacks a chance to stop training,
        and checks the wall-clock time limit. The best validation metrics are
        returned so hyperparameter search selects on validation performance
        (the test split stays untouched for final evaluation via ``test()``).
        """
        logger.info("Starting training loop.")
        start_time: float = time.time()

        best_val_loss: float = float("inf")
        best_val_acc: float = 0.0
        last_val: tuple[float, float] = (float("inf"), 0.0)

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch, start_time)
            val_loss, val_acc = self.validate(start_time)
            last_val = (val_loss, val_acc)

            if val_loss < best_val_loss:
                best_val_loss, best_val_acc = val_loss, val_acc

            if trial is not None:
                trial.report(val_loss, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            logs = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)

            if self._check_time_limit(start_time):
                break

        if not math.isfinite(best_val_loss):
            best_val_loss, best_val_acc = last_val
        return best_val_loss, best_val_acc


# ---------------------------------------------------------------------------
# run_optuna_search
# ---------------------------------------------------------------------------
def run_optuna_search(
    *,
    task_type: str = "image_classification",
    csv_path: Path,
    images_dir: Path | None = None,
    filename_column: str = "filename",
    label_column: str = "label",
    n_trials: int = 3,
    timeout: int | None = None,
    model_size: str = "small",
    workdir: Path,
    num_gpus: str | int = "auto",
    num_cpus: str | int = "auto",
    **extra_kwargs,
) -> dict:
    """Run an Optuna hyperparameter search for the given task type.

    Dispatches to the appropriate per-task objective via ``OBJECTIVE_REGISTRY``.
    ``extra_kwargs`` are forwarded to the objective (e.g. ``text_column`` for
    text tasks).

    Raises:
        ValueError: If ``task_type`` is not in ``OBJECTIVE_REGISTRY``.
    """
    from app.ml_engine.hpo.optuna_objectives import OBJECTIVE_REGISTRY

    if task_type not in OBJECTIVE_REGISTRY:
        raise AutoMLConfigError(
            f"Unknown task type '{task_type}'. Supported: {sorted(OBJECTIVE_REGISTRY)}"
        )

    config = load_task_config(task_type)
    objective_fn = OBJECTIVE_REGISTRY[task_type]

    run_dir = workdir / "optuna"
    run_dir.mkdir(exist_ok=True)

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=10,
        reduction_factor=3,
        min_early_stopping_rate=0,
    )
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    timeout_per_trial = timeout / max(n_trials, 1) if timeout else None

    # Build keyword arguments for the objective
    objective_kwargs: dict = {
        "csv_path": csv_path,
        "images_dir": images_dir,
        "filename_column": filename_column,
        "label_column": label_column,
        "model_size": model_size,
        "timeout_per_trial": timeout_per_trial,
        "config": config,
        "num_gpus": num_gpus,
        "num_cpus": num_cpus,
        "workdir": workdir,
        "task_type": task_type,
        **extra_kwargs,
    }

    study.optimize(
        functools.partial(objective_fn, **objective_kwargs),
        n_trials=n_trials,
        timeout=timeout,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise AutoMLTrainingError(
            f"All {len(study.trials)} Optuna trial(s) failed or were pruned. "
            "Check your dataset, model IDs, and time budget."
        )

    best_trial_dir = run_dir / f"trial_{study.best_trial.number}"
    _copy_best_trial_artifacts(best_trial_dir, workdir)

    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "model_dir": best_trial_dir,
    }


def _copy_best_trial_artifacts(best_trial_dir: Path, workdir: Path) -> None:
    """Copy the best trial's feature_mapping.json + model.pt into workdir/model/.

    ``serialize_and_zip_model`` zips ``workdir/model/``, so anything we want in
    the uploaded zip needs to land there. Missing sources are skipped silently
    (per-trial saves are best-effort and may not have run).
    """
    model_dir = Path(workdir) / "model"
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("Failed to create model dir %s: %s", model_dir, e)
        return

    if not best_trial_dir.exists():
        logger.warning("Best trial artifact dir does not exist: %s", best_trial_dir)
        return

    for fname in ("feature_mapping.json", "model.pt"):
        src = best_trial_dir / fname
        if not src.exists():
            logger.debug("Best trial did not produce %s; skipping", fname)
            continue
        try:
            shutil.copy2(src, model_dir / fname)
            logger.debug("Copied %s -> %s", src, model_dir / fname)
        except Exception as e:
            logger.warning("Failed to copy %s into model dir: %s", src, e)


# ---------------------------------------------------------------------------
# AutoGluon tabular trainer
# ---------------------------------------------------------------------------

load_dotenv(find_dotenv())

_tabular_settings = get_settings()
DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE: float = (
    _tabular_settings.default_tabular_train_test_split_size
)

DEFAULT_TIME_LIMIT: int = _tabular_settings.default_time_limit

if not 0 < DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE < 1:
    raise AutoMLConfigError(
        f"DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE must be in (0, 1), "
        f"got {DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE}"
    )


class AutoGluonTrainer:
    """Wrapper around AutoGluon Tabular training routines."""

    def __init__(
        self,
        save_model_path: Path,
        DatasetClass=TabularDataset,
        PredictorClass=TabularPredictor,
    ):
        if save_model_path == "":
            raise AutoMLConfigError("save_model_path cannot be None or empty")

        self.save_model_path: Path = Path(save_model_path)

        if self.save_model_path.exists() and not self.save_model_path.is_dir():
            raise AutoMLConfigError(
                f"save_model_path must be a directory, got: {self.save_model_path}"
            )

        self.DatasetClass = DatasetClass
        self.PredictorClass = PredictorClass
        logger.debug(f"AutoGluon trainer, model path {self.save_model_path}")

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame | None,
        target_column: str,
        time_limit: int,
        num_cpus: str | int = "auto",
        num_gpus: str | int = "auto",
    ) -> tuple[pd.DataFrame, TabularPredictor]:
        """Train an AutoGluon model on the given DataFrames.

        Validates the target column and time limit, splits train/test if no
        test set was supplied, fits the predictor within the time limit, then
        tries to clone the predictor for deployment (falling back to the
        original on failure) and produces the leaderboard on the test set.
        """
        if not isinstance(train_df, pd.DataFrame):
            raise AutoMLDataError("train_df is not a DataFrame")

        if train_df is None or train_df.empty:
            raise AutoMLDataError("train_df cannot be None or empty")

        if not target_column or not isinstance(target_column, str):
            raise AutoMLDataError("target_column must be a non-empty string")

        if target_column not in train_df.columns:
            raise AutoMLDataError(
                f"target_column '{target_column}' not found in train_df columns: {train_df.columns.tolist()}"
            )

        if test_df is not None and target_column not in test_df.columns:
            raise AutoMLDataError(
                f"target_column '{target_column}' not found in test_df columns: {test_df.columns.tolist()}"
            )

        if not isinstance(time_limit, int) or time_limit <= 0:
            raise AutoMLConfigError(
                f"time_limit must be a positive integer, got {time_limit}"
            )

        try:
            final_train_df, final_test_df = self.train_test_split(
                test_df=test_df, train_df=train_df
            )
        except Exception as e:
            logger.exception("Failed to split train/test data")
            raise AutoMLDataError(f"Train/test split failed: {e}") from e

        try:
            train_dataset = self.DatasetClass(final_train_df)
            test_dataset = self.DatasetClass(final_test_df)
        except Exception as e:
            logger.exception("Failed to create TabularDataset")
            raise AutoMLDataError(f"Dataset creation failed: {e}") from e

        try:
            predictor = self.PredictorClass(
                label=target_column, path=str(self.save_model_path)
            ).fit(
                train_data=train_dataset,
                time_limit=time_limit,
                num_gpus=num_gpus,
                num_cpus=num_cpus,
            )
        except Exception as e:
            logger.exception("AutoGluon training failed")
            raise AutoMLTrainingError(f"Model training failed: {e}") from e

        try:
            save_path_clone_opt = self.save_model_path / "-clone-opt"
            path_clone_opt = predictor.clone_for_deployment(
                path=str(save_path_clone_opt)
            )
            predictor_clone_opt = self.PredictorClass.load(path=str(path_clone_opt))
        except Exception as e:
            logger.warning(
                f"Failed to clone model for deployment, falling back to base predictor: {e}"
            )
            predictor_clone_opt = predictor

        try:
            leaderboard = predictor.leaderboard(test_dataset)
            return leaderboard, predictor_clone_opt
        except Exception as e:
            logger.exception("Failed to generate leaderboard")
            raise AutoMLLeaderboardError(f"Leaderboard generation failed: {e}") from e

    def train_test_split(
        self, test_df: pd.DataFrame | None, train_df: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the train/test pair to train on.

        When no test DataFrame is given, a fraction of the training data is
        sampled out to serve as the test set; otherwise both frames are used
        unchanged.
        """
        if train_df is None:
            raise AutoMLDataError("train_df cannot be None")

        if train_df.empty:
            raise AutoMLDataError("train_df cannot be empty")

        if test_df is None:
            logger.debug("Test dataset not found, creating split")
            final_train_df = train_df.sample(
                frac=DEFAULT_TABULAR_TRAIN_TEST_SPLIT_SIZE, random_state=42
            )
            final_test_df = train_df.drop(index=final_train_df.index.tolist())
        else:
            logger.debug("Test dataset found")

            if test_df.empty:
                raise AutoMLDataError("test_df cannot be empty")

            final_train_df = train_df
            final_test_df = test_df

        if final_train_df.empty:
            raise AutoMLDataError("Final training DataFrame is empty after split")

        if final_test_df.empty:
            raise AutoMLDataError("Final test DataFrame is empty after split")

        return final_train_df, final_test_df
