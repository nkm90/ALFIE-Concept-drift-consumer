"""Shared schema primitives: API response models.

The task/dataset/datamodule base classes previously living here moved into
the consolidated ML engine (``app.ml_engine.tasks``, ``app.ml_engine.dataset``,
``app.ml_engine.datamodule``).
"""

from .responses import (
    AltTextCheckResponse,
    AudioTrainingSuccessResponse,
    ErrorResponse,
    ImagePromptResponse,
    InstructionsResponse,
    MultimodalTrainingSuccessResponse,
    TextTrainingSuccessResponse,
    TrainingSuccessResponse,
    VisionTrainingSuccessResponse,
    WebAccessibilityResponse,
)

__all__ = [
    "ErrorResponse",
    "InstructionsResponse",
    "AltTextCheckResponse",
    "ImagePromptResponse",
    "WebAccessibilityResponse",
    "TrainingSuccessResponse",
    "VisionTrainingSuccessResponse",
    "MultimodalTrainingSuccessResponse",
    "AudioTrainingSuccessResponse",
    "TextTrainingSuccessResponse",
]
