"""Monitoring configuration Pydantic models."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MonitoringConfigCreate(BaseModel):
    """Request model for creating a monitoring configuration."""

    camera_id: UUID = Field(description="ID of the camera to use for monitoring")
    plant_id: UUID = Field(description="ID of the plant to monitor")
    interval_minutes: int = Field(
        ge=5, le=1440, default=60, description="Monitoring interval in minutes (5 min to 24 hours)"
    )
    enabled: bool = Field(default=True, description="Whether monitoring is enabled")


class MonitoringConfigUpdate(BaseModel):
    """Request model for updating a monitoring configuration."""

    camera_id: UUID | None = Field(default=None, description="ID of the camera to use")
    plant_id: UUID | None = Field(default=None, description="ID of the plant to monitor")
    interval_minutes: int | None = Field(
        default=None, ge=5, le=1440, description="Monitoring interval in minutes"
    )
    enabled: bool | None = Field(default=None, description="Whether monitoring is enabled")


class MonitoringConfigResponse(BaseModel):
    """Response model for a monitoring configuration."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    camera_id: UUID
    plant_id: UUID
    interval_minutes: int
    enabled: bool
    last_run_at: datetime | None = Field(default=None, description="Timestamp of last monitoring run")
    next_run_at: datetime | None = Field(default=None, description="Scheduled timestamp for next run")
    created_at: datetime
    updated_at: datetime | None = None


class MonitoringConfigListResponse(BaseModel):
    """Response model for listing monitoring configurations."""

    items: list[MonitoringConfigResponse]
    total: int = Field(description="Total number of configurations")


class MonitoringTriggerResponse(BaseModel):
    """Response model for manually triggering monitoring."""

    message: str = Field(description="Status message")
    analysis_id: UUID | None = Field(default=None, description="ID of the triggered analysis")
    triggered_at: datetime = Field(description="Timestamp when monitoring was triggered")


class MonitoringStatus(BaseModel):
    """Status of monitoring system."""

    active_configs: int = Field(description="Number of active monitoring configurations")
    total_configs: int = Field(description="Total number of monitoring configurations")
    last_global_run: datetime | None = Field(default=None, description="Last global monitoring run")

