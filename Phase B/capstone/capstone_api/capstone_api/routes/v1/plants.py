"""Plant routes for managing plants in the user's orchard."""

from uuid import UUID

from fastapi import APIRouter, Query, status

from capstone_api.core.dependencies import CurrentUser
from capstone_api.models.analysis import AnalysisListResponse
from capstone_api.models.plant import (
    PlantCreate,
    PlantListResponse,
    PlantResponse,
    PlantUpdate,
)

router = APIRouter(prefix="/plants", tags=["plants"])


@router.post(
    "",
    response_model=PlantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Plant",
    description="Create a new plant in the user's orchard.",
)
async def create_plant(
    plant_data: PlantCreate,
    current_user: CurrentUser,
) -> PlantResponse:
    """Create a new plant."""
    raise NotImplementedError()


@router.get(
    "",
    response_model=PlantListResponse,
    summary="List Plants",
    description="List all plants for the authenticated user with pagination.",
)
async def list_plants(
    current_user: CurrentUser,
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
) -> PlantListResponse:
    """List all plants for the user."""
    raise NotImplementedError()


@router.get(
    "/{plant_id}",
    response_model=PlantResponse,
    summary="Get Plant",
    description="Get details of a specific plant by ID.",
)
async def get_plant(
    plant_id: UUID,
    current_user: CurrentUser,
) -> PlantResponse:
    """Get a plant by ID."""
    raise NotImplementedError()


@router.put(
    "/{plant_id}",
    response_model=PlantResponse,
    summary="Update Plant",
    description="Update a plant's information.",
)
async def update_plant(
    plant_id: UUID,
    plant_data: PlantUpdate,
    current_user: CurrentUser,
) -> PlantResponse:
    """Update a plant."""
    raise NotImplementedError()


@router.delete(
    "/{plant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Plant",
    description="Delete a plant and all its associated analyses.",
)
async def delete_plant(
    plant_id: UUID,
    current_user: CurrentUser,
) -> None:
    """Delete a plant."""
    raise NotImplementedError()


@router.get(
    "/{plant_id}/analysis",
    response_model=AnalysisListResponse,
    summary="List Plant Analyses",
    description="List all analyses for a specific plant.",
)
async def list_plant_analyses(
    plant_id: UUID,
    current_user: CurrentUser,
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
) -> AnalysisListResponse:
    """List analyses for a plant."""
    raise NotImplementedError()
