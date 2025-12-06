"""Plant-related Pydantic models."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PlantCreate(BaseModel):
    """Request model for creating a new plant."""

    name: str = Field(min_length=1, max_length=100, description="Plant name")
    plant_number: str = Field(min_length=1, max_length=50, description="Plant identification number in orchard")
    location: str | None = Field(default=None, max_length=200, description="Plant location in orchard")
    notes: str | None = Field(default=None, max_length=1000, description="Additional notes about the plant")


class PlantUpdate(BaseModel):
    """Request model for updating a plant."""

    name: str | None = Field(default=None, min_length=1, max_length=100, description="Plant name")
    plant_number: str | None = Field(
        default=None, min_length=1, max_length=50, description="Plant identification number"
    )
    location: str | None = Field(default=None, max_length=200, description="Plant location in orchard")
    notes: str | None = Field(default=None, max_length=1000, description="Additional notes about the plant")


class PlantResponse(BaseModel):
    """Response model for a plant."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    name: str
    plant_number: str
    location: str | None = None
    notes: str | None = None
    created_at: datetime
    updated_at: datetime | None = None


class PlantListResponse(BaseModel):
    """Response model for listing plants."""

    model_config = ConfigDict(from_attributes=True)

    items: list[PlantResponse]
    total: int = Field(description="Total number of plants")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")
    total_pages: int = Field(description="Total number of pages")


class PlantSummary(BaseModel):
    """Summary view of a plant for use in other responses."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    plant_number: str

