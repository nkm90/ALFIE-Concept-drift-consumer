"""Route definitions for the tabular AutoML service.

The ``best_model`` endpoint is deliberately thin: it binds HTTP form parameters,
hands them to ``run_training_pipeline`` in the orchestrator layer, and
translates the domain exceptions it raises into HTTP responses via
``app.core.api_errors.automl_exception_to_response``. None of the fetch →
resolve → download → validate → train → serialize → upload orchestration lives
here.
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse

from app.core.api_errors import automl_exception_to_response
from app.core.exceptions import AutoMLError
from app.core.process_log import get_process_log, start_process_log
from app.core.utils import render_deployment_instructions
from app.core.schemas.responses import (
    ErrorResponse,
    InstructionsResponse,
    TrainingSuccessResponse,
)
from app.ml_engine.model import SUPPORTED_TABULAR_TASK_TYPES
from app.tabular_automl.orchestrator import (
    TabularTrainingRequest,
    run_training_pipeline,
)
from app.tabular_automl.services import (
    deployment_instructions,
    tabular_data_instructions,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tabular"])

_COMMON_RESPONSES: dict[int | str, dict[str, Any]] = {
    500: {"description": "Internal server error", "model": ErrorResponse},
}


@router.post(
    "/deployment_instructions/",
    response_model=InstructionsResponse,
    responses=_COMMON_RESPONSES,
)
async def show_deployment_instructions() -> JSONResponse:
    """Show deployment instructions from a template"""
    try:
        return JSONResponse(
            content={"instructions": deployment_instructions()}, status_code=200
        )
    except Exception as e:
        logger.exception(
            "Unexpected error in finding deployment instructions in tabular"
        )
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post(
    "/accepted_format/",
    response_model=InstructionsResponse,
    responses=_COMMON_RESPONSES,
)
async def show_accepted_format_instructions() -> JSONResponse:
    """Show accepted format instructions from a template"""
    try:
        return JSONResponse(
            content={"instructions": tabular_data_instructions()}, status_code=200
        )
    except Exception as e:
        logger.exception(
            "Unexpected error in finding data format instructions in tabular"
        )
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post(
    "/best_model/",
    response_model=TrainingSuccessResponse,
    responses={
        400: {"description": "Validation error", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
        502: {"description": "AutoDW communication failure", "model": ErrorResponse},
    },
)
async def find_best_model_for_mvp(
    request: Request,
    user_id: Annotated[str, Form(..., description="User id from AutoDW")],
    dataset_id: Annotated[str, Form(..., description="Dataset id from AutoDW")],
    dataset_version: Annotated[
        str | None, Form(description="Dataset version (e.g., 'v1', 'v2')")
    ] = "v1",
    target_column_name: Annotated[
        str, Form(..., description="Name of the target column")
    ] = "",
    time_stamp_column_name: Annotated[
        str | None,
        Form(..., description="Timestamp column (required for time-series tasks)"),
    ] = None,
    task_type: Annotated[
        str,
        Form(
            ...,
            description="Type of ML task",
            examples=SUPPORTED_TABULAR_TASK_TYPES,
        ),
    ] = "classification",
    time_budget: Annotated[int, Form(..., description="Time budget in seconds")] = 10,
    num_cpus: Annotated[
        int | str,
        Form(
            ...,
            description="Number of CPUs to use for AutoML. Can be a number or 'auto' ",
        ),
    ] = "auto",
    num_gpus: Annotated[
        int | str,
        Form(
            ...,
            description="Number of GPUs to use for AutoML. Can be a number or 'auto' ",
        ),
    ] = "auto",
    dataset_split: Annotated[
        str | None,
        Form(description="Dataset split to use for training (e.g., 'train')."),
    ] = None,
) -> JSONResponse:
    """
    Fetch a tabular dataset from AutoDW, run AutoML training, and upload the best model.

    Steps (performed in the orchestrator):
      1. Fetch dataset metadata from AutoDW.
      2. Resolve the correct download URL (respecting splits if present).
      3. Download the dataset file to a temporary directory.
      4. Validate user-supplied parameters against the dataset.
      5. Train an AutoML model within the given time budget.
      6. Serialize and zip the best predictor.
      7. Upload the model and leaderboard back to AutoDW.

    Returns:
        200 – success message and leaderboard summary.
        400 – validation error (bad inputs or unsupported file type).
        502 – AutoDW communication failure.
        500 – unexpected runtime error.
    """
    try:
        start_process_log(request.headers.get("X-Task-ID"))
        req = TabularTrainingRequest(
            user_id=user_id,
            dataset_id=dataset_id,
            target_column_name=target_column_name,
            task_type=task_type,
            time_budget=time_budget,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            dataset_version=dataset_version,
            time_stamp_column_name=time_stamp_column_name,
            dataset_split=dataset_split,
        )
        result = await run_training_pipeline(
            req, task_id=request.headers.get("X-Task-ID")
        )
        return JSONResponse(
            status_code=200,
            content={
                "message": result.message,
                "leaderboard": result.leaderboard,
                "deployment_instructions": render_deployment_instructions(
                    deployment_instructions
                ),
                "process_log": get_process_log(),
            },
        )
    except Exception as e:
        if not isinstance(e, AutoMLError):
            logger.exception("Unexpected error during tabular AutoML")
        return automl_exception_to_response(e)
