"""Common Pydantic models shared across the application."""

from datetime import datetime
from typing import Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """Generic message response."""

    message: str


class ErrorResponse(BaseModel):
    """Standard error response format."""

    detail: str
    error_code: str | None = None


class TimestampMixin(BaseModel):
    """Mixin for models with timestamp fields."""

    created_at: datetime
    updated_at: datetime | None = None


# Generic type for paginated responses
T = TypeVar("T")


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""

    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    items: list[T]
    total: int = Field(description="Total number of items")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")
    total_pages: int = Field(description="Total number of pages")


class HealthStatus(BaseModel):
    """Health check status."""

    status: str = Field(default="ok", description="Service health status")
    version: str | None = Field(default=None, description="API version")


class BaseEntityResponse(BaseModel):
    """Base response model for entities with ID and timestamps."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime | None = None

