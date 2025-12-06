"""Camera and IoT device Pydantic models."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CameraStatus(str, Enum):
    """Status of a connected camera."""

    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    UNKNOWN = "unknown"


class CameraCreate(BaseModel):
    """Request model for registering a new camera."""

    name: str = Field(min_length=1, max_length=100, description="Camera display name")
    home_assistant_entity_id: str = Field(
        min_length=1, max_length=200, description="Home Assistant entity ID for the camera"
    )
    description: str | None = Field(default=None, max_length=500, description="Camera description")


class CameraUpdate(BaseModel):
    """Request model for updating a camera."""

    name: str | None = Field(default=None, min_length=1, max_length=100, description="Camera display name")
    home_assistant_entity_id: str | None = Field(
        default=None, min_length=1, max_length=200, description="Home Assistant entity ID"
    )
    description: str | None = Field(default=None, max_length=500, description="Camera description")


class CameraResponse(BaseModel):
    """Response model for a camera."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    name: str
    home_assistant_entity_id: str
    description: str | None = None
    status: CameraStatus
    last_capture_at: datetime | None = Field(default=None, description="Timestamp of last image capture")
    created_at: datetime
    updated_at: datetime | None = None


class CameraListResponse(BaseModel):
    """Response model for listing cameras."""

    items: list[CameraResponse]
    total: int = Field(description="Total number of cameras")


class CameraCaptureResponse(BaseModel):
    """Response model for camera capture action."""

    message: str = Field(description="Status message")
    camera_id: UUID = Field(description="ID of the camera")
    image_url: str | None = Field(default=None, description="URL of the captured image")
    captured_at: datetime = Field(description="Timestamp of the capture")


class CameraTestResponse(BaseModel):
    """Response model for testing camera connection."""

    camera_id: UUID
    status: CameraStatus
    message: str = Field(description="Connection test result message")
    response_time_ms: float | None = Field(default=None, description="Response time in milliseconds")

