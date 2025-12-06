"""Monitoring service for automated interval-based analysis."""

from uuid import UUID

from capstone_api.db.models import User
from capstone_api.models.monitoring import (
    MonitoringConfigCreate,
    MonitoringConfigListResponse,
    MonitoringConfigResponse,
    MonitoringConfigUpdate,
    MonitoringTriggerResponse,
)


class MonitoringService:
    """Service for handling monitoring configuration and execution."""

    async def create_config(
        self,
        user: User,
        config_data: MonitoringConfigCreate,
    ) -> MonitoringConfigResponse:
        """
        Create a new monitoring configuration.

        Args:
            user: The authenticated user
            config_data: Monitoring configuration data

        Returns:
            MonitoringConfigResponse: The created configuration

        TODO: Implement configuration creation.
        """
        raise NotImplementedError("Create monitoring config not implemented")

    async def get_config(
        self,
        user: User,
        config_id: UUID,
    ) -> MonitoringConfigResponse | None:
        """
        Get a monitoring configuration by ID.

        Args:
            user: The authenticated user
            config_id: The configuration UUID

        Returns:
            MonitoringConfigResponse | None: Configuration or None if not found

        TODO: Implement configuration retrieval.
        """
        raise NotImplementedError("Get monitoring config not implemented")

    async def list_configs(self, user: User) -> MonitoringConfigListResponse:
        """
        List all monitoring configurations for the user.

        Args:
            user: The authenticated user

        Returns:
            MonitoringConfigListResponse: List of configurations

        TODO: Implement configuration listing.
        """
        raise NotImplementedError("List monitoring configs not implemented")

    async def update_config(
        self,
        user: User,
        config_id: UUID,
        config_data: MonitoringConfigUpdate,
    ) -> MonitoringConfigResponse | None:
        """
        Update a monitoring configuration.

        Args:
            user: The authenticated user
            config_id: The configuration UUID
            config_data: Configuration update data

        Returns:
            MonitoringConfigResponse | None: Updated config or None if not found

        TODO: Implement configuration update.
        """
        raise NotImplementedError("Update monitoring config not implemented")

    async def delete_config(self, user: User, config_id: UUID) -> bool:
        """
        Delete a monitoring configuration.

        Args:
            user: The authenticated user
            config_id: The configuration UUID

        Returns:
            bool: True if deleted, False if not found

        TODO: Implement configuration deletion.
        """
        raise NotImplementedError("Delete monitoring config not implemented")

    async def trigger_monitoring(
        self,
        user: User,
        config_id: UUID,
    ) -> MonitoringTriggerResponse:
        """
        Manually trigger monitoring for a configuration.

        This captures an image from the configured camera and runs
        the analysis pipeline.

        Args:
            user: The authenticated user
            config_id: The configuration UUID

        Returns:
            MonitoringTriggerResponse: Result of the trigger action

        TODO: Implement manual trigger.
        """
        raise NotImplementedError("Trigger monitoring not implemented")

    async def run_scheduled_monitoring(self) -> int:
        """
        Run all due scheduled monitoring tasks.

        This is called by a background scheduler.

        Returns:
            int: Number of monitoring tasks executed

        TODO: Implement scheduled monitoring execution.
        """
        raise NotImplementedError("Scheduled monitoring not implemented")


# Singleton instance
monitoring_service = MonitoringService()

