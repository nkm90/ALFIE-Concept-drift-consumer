"""Unified API router for all ALFIE AutoML engine services.

Every service router is mounted under a single ``/automl`` prefix so the
combined app exposes one URL scheme:

- ``/automl/tabular/...``     — tabular AutoML
- ``/automl/vision/...``      — vision AutoML
- ``/automl/audio/...``       — audio AutoML
- ``/automl/text/...``        — text AutoML
- ``/automl/automl_plus/...`` — AutoML+ (LLM/VLM tools)

Plus one meta endpoint, ``GET /automl/endpoints``, that lists every endpoint
in the running app (path, methods, docs) in an LLM-friendly JSON format.
"""

import inspect

from fastapi import APIRouter, Request
from fastapi.routing import APIRoute
from pydantic import BaseModel

from app.audio_automl.router import router as audio_router
from app.automlplus.router import router as automl_plus_router
from app.tabular_automl.router import router as tabular_router
from app.text_automl.router import router as text_router
from app.vision_automl.router import router as vision_router

router = APIRouter(prefix="/automl")

router.include_router(tabular_router, prefix="/tabular")
router.include_router(vision_router, prefix="/vision")
router.include_router(audio_router, prefix="/audio")
router.include_router(text_router, prefix="/text")
router.include_router(automl_plus_router, prefix="/automl_plus")


class EndpointInfo(BaseModel):
    """One API endpoint, described for LLM consumption."""

    path: str
    methods: list[str]
    name: str
    summary: str
    description: str
    tags: list[str]


@router.get("/endpoints", response_model=list[EndpointInfo], tags=["meta"])
async def list_endpoints(request: Request) -> list[EndpointInfo]:
    """List every endpoint in this service as structured JSON.

    Call this endpoint if the user wants to know what the AutoML engine does.
    Introspects the running FastAPI app's routes and returns each endpoint's
    path, HTTP methods, function name, and docstring-derived summary and
    description — a compact, LLM-readable alternative to the raw OpenAPI spec.
    """
    endpoints: list[EndpointInfo] = []
    for route in request.app.routes:
        if not isinstance(route, APIRoute):
            continue
        summary = route.summary or route.name.replace("_", " ").title()
        description = route.description
        if not description:
            description = inspect.getdoc(route.endpoint) or ""
        endpoints.append(
            EndpointInfo(
                path=route.path,
                methods=sorted(
                    m for m in route.methods if m not in ("HEAD", "OPTIONS")
                ),
                name=route.name,
                summary=summary,
                description=description,
                tags=[str(tag) for tag in route.tags],
            )
        )
    endpoints.sort(key=lambda e: (e.path, e.name))
    return endpoints
