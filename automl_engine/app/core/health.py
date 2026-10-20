"""Health and readiness endpoints shared by every service."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    """Liveness probe: report that the service process is up."""
    return {"status": "alive"}


@router.get("/ready")
async def ready():
    """Readiness probe: report that the service is able to handle requests."""
    return {"status": "ready"}
