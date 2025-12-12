"""User-related Pydantic models."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserSettings(BaseModel):
    """User settings configuration."""

    language: str = Field(default="en", description="Preferred language code")
    notifications_enabled: bool = Field(default=True, description="Enable notifications")
    dark_mode: bool = Field(default=False, description="Enable dark mode")
    default_monitoring_interval: int = Field(
        default=60, ge=5, le=1440, description="Default monitoring interval in minutes"
    )


class UserSettingsUpdate(BaseModel):
    """Request model for updating user settings."""

    language: str | None = Field(default=None, description="Preferred language code")
    notifications_enabled: bool | None = Field(default=None, description="Enable notifications")
    dark_mode: bool | None = Field(default=None, description="Enable dark mode")
    default_monitoring_interval: int | None = Field(
        default=None, ge=5, le=1440, description="Default monitoring interval in minutes"
    )


class UserResponse(BaseModel):
    """User response model."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    user_name: str
    password: str


class UserSettingsResponse(BaseModel):
    """Response model for user settings."""

    model_config = ConfigDict(from_attributes=True)

    settings: UserSettings
    updated_at: datetime | None = None

