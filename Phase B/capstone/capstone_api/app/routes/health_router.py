from fastapi import APIRouter

from app.models.health import HealthResponse

router = APIRouter()


@router.get("/health-check")
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/readiness")
async def readiness() -> HealthResponse:
    return HealthResponse(status="ok")
