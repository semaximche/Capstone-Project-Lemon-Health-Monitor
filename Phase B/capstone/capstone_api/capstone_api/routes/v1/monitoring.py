"""Monitoring routes for automated interval-based analysis configuration."""

from uuid import UUID

from fastapi import APIRouter, status

from capstone_api.core.dependencies import CurrentUser
from capstone_api.models.monitoring import (
    MonitoringConfigCreate,
    MonitoringConfigListResponse,
    MonitoringConfigResponse,
    MonitoringConfigUpdate,
    MonitoringTriggerResponse,
)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get(
    "/configs",
    response_model=MonitoringConfigListResponse,
    summary="List Monitoring Configs",
    description="List all monitoring configurations for the authenticated user.",
)
async def list_configs(current_user: CurrentUser) -> MonitoringConfigListResponse:
    """List all monitoring configurations."""
    raise NotImplementedError()


@router.post(
    "/configs",
    response_model=MonitoringConfigResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Monitoring Config",
    description="Create a new monitoring configuration for automated analysis.",
)
async def create_config(
    config_data: MonitoringConfigCreate,
    current_user: CurrentUser,
) -> MonitoringConfigResponse:
    """Create a new monitoring configuration."""
    raise NotImplementedError()


@router.get(
    "/configs/{config_id}",
    response_model=MonitoringConfigResponse,
    summary="Get Monitoring Config",
    description="Get details of a specific monitoring configuration.",
)
async def get_config(
    config_id: UUID,
    current_user: CurrentUser,
) -> MonitoringConfigResponse:
    """Get a monitoring configuration by ID."""
    raise NotImplementedError()


@router.put(
    "/configs/{config_id}",
    response_model=MonitoringConfigResponse,
    summary="Update Monitoring Config",
    description="Update a monitoring configuration.",
)
async def update_config(
    config_id: UUID,
    config_data: MonitoringConfigUpdate,
    current_user: CurrentUser,
) -> MonitoringConfigResponse:
    """Update a monitoring configuration."""
    raise NotImplementedError()


@router.delete(
    "/configs/{config_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Monitoring Config",
    description="Delete a monitoring configuration.",
)
async def delete_config(
    config_id: UUID,
    current_user: CurrentUser,
) -> None:
    """Delete a monitoring configuration."""
    raise NotImplementedError()


@router.post(
    "/configs/{config_id}/trigger",
    response_model=MonitoringTriggerResponse,
    summary="Trigger Monitoring",
    description="Manually trigger monitoring for a configuration.",
)
async def trigger_monitoring(
    config_id: UUID,
    current_user: CurrentUser,
) -> MonitoringTriggerResponse:
    """Manually trigger monitoring."""
    raise NotImplementedError()
