"""Model wrappers and task models for the consolidated ML engine.

Each wrapper class loads a Hugging Face checkpoint for its task with a
replaced classification head (sizes ignored so heads can be re-initialized)
and exposes a plain ``forward`` that the FabricTrainer can call. The tabular
task models at the bottom describe the AutoGluon-backed tabular tasks.
"""

import logging
from pathlib import Path

import torch
from pydantic import BaseModel
from torch import nn
from transformers import (
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForSemanticSegmentation,
    AutoModelForKeypointDetection,
    AutoModelForMaskedLM,
    AutoModelForObjectDetection,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForVideoClassification,
)

from app.core.exceptions import AutoMLConfigError, AutoMLTrainingError

logger = logging.getLogger(__name__)


class ImageClassificationModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForImageClassification. This module is responsible for Image classification!!!"""

    def __init__(
        self,
        model_id: str = "google/vit-base-patch16-224",
        num_classes: int = 2,
        freeze_backbone: bool = True,
        id2label: dict | None = None,
        label2id: dict | None = None,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        config_kwargs = {
            "num_labels": num_classes,
            "id2label": id2label or {i: str(i) for i in range(num_classes)},
            "label2id": label2id or {str(i): i for i in range(num_classes)},
        }
        try:
            self.model = AutoModelForImageClassification.from_pretrained(
                model_id,
                ignore_mismatched_sizes=True,
                **config_kwargs,
            )
        except Exception as e:
            logger.error(
                "Failed to load image classification model from %s: %s", model_id, e
            )
            raise AutoMLTrainingError(
                f"Failed to load image classification model from {model_id}: {e}"
            ) from e
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            if hasattr(self.model, "classifier"):
                for param in self.model.classifier.parameters():
                    param.requires_grad = True

    def train(self, mode: bool = True) -> "ImageClassificationModel":
        """Keep a frozen backbone in eval mode (BatchNorm stats, dropout)."""
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values).logits


# Backward-compatibility alias
ClassificationModel = ImageClassificationModel


class MultimodalClassificationModel(nn.Module):
    """Multimodal image classification model that fuses vision embeddings
    with auxiliary tabular features via early (concatenation) fusion.

    Architecture:
        1. HF vision backbone (without classification head) produces image embeddings.
        2. A small MLP processes tabular auxiliary features.
        3. Image and tabular embeddings are concatenated and passed through a
           fusion classifier head.
    """

    def __init__(
        self,
        model_id: str = "google/vit-base-patch16-224",
        num_classes: int = 2,
        aux_feature_dim: int = 0,
        freeze_backbone: bool = True,
        id2label: dict | None = None,
        label2id: dict | None = None,
        fusion_hidden_dim: int = 128,
    ):
        super().__init__()
        self.aux_feature_dim = aux_feature_dim
        self.freeze_backbone = freeze_backbone

        from transformers import AutoConfig

        config_kwargs = {
            "num_labels": num_classes,
            "id2label": id2label or {i: str(i) for i in range(num_classes)},
            "label2id": label2id or {str(i): i for i in range(num_classes)},
        }
        try:
            hf_config = AutoConfig.from_pretrained(model_id, **config_kwargs)
            self.backbone = AutoModelForImageClassification.from_pretrained(
                model_id,
                config=hf_config,
                ignore_mismatched_sizes=True,
            )
        except Exception as e:
            logger.error("Failed to load vision backbone from %s: %s", model_id, e)
            raise AutoMLTrainingError(
                f"Failed to load vision backbone from {model_id}: {e}"
            ) from e

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if hasattr(self.backbone, "classifier"):
                for param in self.backbone.classifier.parameters():
                    param.requires_grad = True

        vision_dim = self._get_vision_embed_dim()
        tabular_dim = max(aux_feature_dim, 1)

        self.tabular_mlp = (
            nn.Sequential(
                nn.Linear(tabular_dim, fusion_hidden_dim),
                nn.ReLU(),
                nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
                nn.ReLU(),
            )
            if aux_feature_dim > 0
            else nn.Identity()
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(
                vision_dim + fusion_hidden_dim if aux_feature_dim > 0 else vision_dim,
                fusion_hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def train(self, mode: bool = True) -> "MultimodalClassificationModel":
        """Keep the frozen vision backbone in eval mode (BatchNorm, dropout)."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _get_vision_embed_dim(self) -> int:
        model = self.backbone
        if hasattr(model, "classifier") and hasattr(model.classifier, "in_features"):
            return model.classifier.in_features
        if hasattr(model, "fc") and hasattr(model.fc, "in_features"):
            return model.fc.in_features
        config = getattr(model, "config", None)
        if config is not None:
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is not None:
                return hidden_size
        raise AutoMLConfigError(
            "Cannot determine vision embedding dimension from model config"
        )

    def _extract_vision_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "classifier"):
            original_classifier = self.backbone.classifier
            self.backbone.classifier = nn.Identity()
            try:
                embeddings = self.backbone(pixel_values)
                if hasattr(embeddings, "logits"):
                    embeddings = embeddings.logits
            finally:
                self.backbone.classifier = original_classifier
            return embeddings
        output = self.backbone(pixel_values)
        return output.logits if hasattr(output, "logits") else output

    def forward(
        self, pixel_values: torch.Tensor, aux_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        vision_embeds = self._extract_vision_embeddings(pixel_values)

        if aux_features is not None and self.aux_feature_dim > 0:
            tabular_embeds = self.tabular_mlp(aux_features)
            combined = torch.cat([vision_embeds, tabular_embeds], dim=-1)
        else:
            combined = vision_embeds

        return self.fusion_head(combined)


class ImageSegmentationModel(nn.Module):
    """
    Thin nn.Module wrapping a Hugging Face
    semantic-segmentation model.
    """

    def __init__(
        self,
        model_id: str,
        num_classes: int = 2,
        id2label: dict | None = None,
        label2id: dict | None = None,
    ):
        super().__init__()

        try:
            self.model = (
                AutoModelForSemanticSegmentation
                .from_pretrained(
                    model_id,
                    ignore_mismatched_sizes=True,
                    num_labels=num_classes,
                    id2label=(
                        id2label
                        or {
                            i: str(i)
                            for i
                            in range(num_classes)
                        }
                    ),
                    label2id=(
                        label2id
                        or {
                            str(i): i
                            for i
                            in range(num_classes)
                        }
                    ),
                )
            )

        except Exception as e:
            logger.error(
                "Failed to load semantic "
                "segmentation model from %s: %s",
                model_id,
                e,
            )

            raise AutoMLTrainingError(
                "Failed to load semantic "
                f"segmentation model from "
                f"{model_id}: {e}"
            ) from e

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """
        Return scalar loss when labels are supplied;
        otherwise return semantic-segmentation logits.
        """

        output = self.model(
            pixel_values=pixel_values,
            labels=labels,
        )

        return (
            output.loss
            if labels is not None
            else output.logits
        )


class ObjectDetectionModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForObjectDetection."""

    def __init__(self, model_id: str, num_classes: int = 2):
        super().__init__()
        try:
            self.model = AutoModelForObjectDetection.from_pretrained(
                model_id,
                ignore_mismatched_sizes=True,
                num_labels=num_classes,
            )
        except Exception as e:
            logger.error(
                "Failed to load object detection model from %s: %s", model_id, e
            )
            raise AutoMLTrainingError(
                f"Failed to load object detection model from {model_id}: {e}"
            ) from e

    def forward(self, pixel_values: torch.Tensor, labels=None):
        """Returns loss when labels provided (list of dicts), else raw output."""
        output = self.model(pixel_values=pixel_values, labels=labels)
        return output.loss if labels is not None else output


class VideoClassificationModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForVideoClassification."""

    def __init__(
        self,
        model_id: str,
        num_classes: int = 2,
        id2label: dict | None = None,
        label2id: dict | None = None,
    ):
        super().__init__()
        try:
            self.model = AutoModelForVideoClassification.from_pretrained(
                model_id,
                ignore_mismatched_sizes=True,
                num_labels=num_classes,
                id2label=id2label or {i: str(i) for i in range(num_classes)},
                label2id=label2id or {str(i): i for i in range(num_classes)},
            )
        except Exception as e:
            logger.error(
                "Failed to load video classification model from %s: %s", model_id, e
            )
            raise AutoMLTrainingError(
                f"Failed to load video classification model from {model_id}: {e}"
            ) from e

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


class KeypointDetectionModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForKeypointDetection."""

    def __init__(self, model_id: str):
        super().__init__()
        try:
            self.model = AutoModelForKeypointDetection.from_pretrained(
                model_id,
                ignore_mismatched_sizes=True,
            )
        except Exception as e:
            logger.error(
                "Failed to load keypoint detection model from %s: %s", model_id, e
            )
            raise AutoMLTrainingError(
                f"Failed to load keypoint detection model from {model_id}: {e}"
            ) from e

    def forward(self, pixel_values: torch.Tensor, labels=None):
        """Returns loss when labels provided, else raw output."""
        output = self.model(pixel_values=pixel_values, labels=labels)
        return output.loss if labels is not None else output


class AudioClassificationModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForAudioClassification."""

    def __init__(
        self,
        model_id: str,
        num_classes: int = 2,
        id2label: dict | None = None,
        label2id: dict | None = None,
    ):
        super().__init__()
        try:
            self.model = AutoModelForAudioClassification.from_pretrained(
                model_id,
                ignore_mismatched_sizes=True,
                num_labels=num_classes,
                id2label=id2label or {i: str(i) for i in range(num_classes)},
                label2id=label2id or {str(i): i for i in range(num_classes)},
            )
        except Exception as e:
            logger.error(
                "Failed to load audio classification model from %s: %s", model_id, e
            )
            raise AutoMLTrainingError(
                f"Failed to load audio classification model from {model_id}: {e}"
            ) from e

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.model(input_values=input_values).logits


class SequenceClassificationModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForSequenceClassification."""

    def __init__(
        self,
        model_id: str,
        num_classes: int = 2,
        id2label: dict | None = None,
        label2id: dict | None = None,
    ):
        super().__init__()
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                ignore_mismatched_sizes=True,
                num_labels=num_classes,
                id2label=id2label or {i: str(i) for i in range(num_classes)},
                label2id=label2id or {str(i): i for i in range(num_classes)},
            )
        except Exception as e:
            logger.error(
                "Failed to load sequence classification model from %s: %s", model_id, e
            )
            raise AutoMLTrainingError(
                f"Failed to load sequence classification model from {model_id}: {e}"
            ) from e

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class QuestionAnsweringModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForQuestionAnswering."""

    def __init__(self, model_id: str):
        super().__init__()
        try:
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        except Exception as e:
            logger.error(
                "Failed to load question answering model from %s: %s", model_id, e
            )
            raise AutoMLTrainingError(
                f"Failed to load question answering model from {model_id}: {e}"
            ) from e

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
    ):
        """Returns loss scalar when start/end positions provided, else raw output."""
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        if start_positions is not None and end_positions is not None:
            return output.loss
        return output


class CausalLMModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForCausalLM."""

    def __init__(self, model_id: str):
        super().__init__()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
        except Exception as e:
            logger.error("Failed to load causal LM model from %s: %s", model_id, e)
            raise AutoMLTrainingError(
                f"Failed to load causal LM model from {model_id}: {e}"
            ) from e

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Always returns the scalar language modelling loss."""
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss


class Seq2SeqLMModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForSeq2SeqLM."""

    def __init__(self, model_id: str):
        super().__init__()
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        except Exception as e:
            logger.error("Failed to load seq2seq LM model from %s: %s", model_id, e)
            raise AutoMLTrainingError(
                f"Failed to load seq2seq LM model from {model_id}: {e}"
            ) from e

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns the scalar seq2seq loss."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        ).loss


class MaskedLMModel(nn.Module):
    """Thin nn.Module wrapping HF AutoModelForMaskedLM."""

    def __init__(self, model_id: str):
        super().__init__()
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(model_id)
        except Exception as e:
            logger.error("Failed to load masked LM model from %s: %s", model_id, e)
            raise AutoMLTrainingError(
                f"Failed to load masked LM model from {model_id}: {e}"
            ) from e

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns the scalar masked language modelling loss."""
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss


# ---------------------------------------------------------------------------
# Tabular task models (AutoGluon engine)
# ---------------------------------------------------------------------------

SUPPORTED_TABULAR_TASK_TYPES = [
    "tabular_classification",
    "tabular_regression",
    "tabular_time_series",
]


class TabularTask(BaseModel):
    """Base Pydantic model describing common tabular task inputs."""

    target_feature: str
    time_stamp_col: str | None = None
    train_file_path: Path
    test_file_path: Path | None = None

    class Config:
        arbitrary_types_allowed: bool = True


class TabularSupervisedClassificationTask(TabularTask):
    """Tabular classification task configuration.

    Typical use-cases: churn prediction, loan approval, disease type, etc.
    """

    task_type: str = "tabular_classification"


class TabularSupervisedRegressionTask(TabularTask):
    """Tabular regression task configuration.

    Predicts continuous numeric values (e.g., price, salary, demand).
    """

    task_type: str = "tabular_regression"


class TabularSupervisedTimeSeriesTask(TabularTask):
    """Time-series forecasting task configuration for tabular data."""

    task_type: str = "tabular_time_series"
    time_stamp_col: str = "timestamp"
