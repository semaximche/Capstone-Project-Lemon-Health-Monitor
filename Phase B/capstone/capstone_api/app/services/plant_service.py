"""Plant service for managing plants in the user's orchard."""

from uuid import UUID

from app.db.models import User
from app.models.plant import (
    PlantCreate,
    PlantListResponse,
    PlantResponse,
    PlantUpdate,
)


class PlantService:
    """Service for handling plant operations."""

    async def create_plant(self, user: User, plant_data: PlantCreate) -> PlantResponse:
        """
        Create a new plant for the user.

        Args:
            user: The authenticated user
            plant_data: Plant creation data

        Returns:
            PlantResponse: The created plant

        TODO: Implement plant creation.
        """
        raise NotImplementedError("Create plant not implemented")

    async def get_plant(self, user: User, plant_id: UUID) -> PlantResponse | None:
        """
        Get a plant by ID.

        Args:
            user: The authenticated user
            plant_id: The plant's UUID

        Returns:
            PlantResponse | None: Plant data or None if not found

        TODO: Implement plant retrieval.
        """
        raise NotImplementedError("Get plant not implemented")

    async def list_plants(
        self,
        user: User,
        page: int = 1,
        page_size: int = 20,
    ) -> PlantListResponse:
        """
        List all plants for the user.

        Args:
            user: The authenticated user
            page: Page number (1-indexed)
            page_size: Number of items per page

        Returns:
            PlantListResponse: Paginated list of plants

        TODO: Implement plant listing with pagination.
        """
        raise NotImplementedError("List plants not implemented")

    async def update_plant(
        self,
        user: User,
        plant_id: UUID,
        plant_data: PlantUpdate,
    ) -> PlantResponse | None:
        """
        Update a plant.

        Args:
            user: The authenticated user
            plant_id: The plant's UUID
            plant_data: Plant update data

        Returns:
            PlantResponse | None: Updated plant or None if not found

        TODO: Implement plant update.
        """
        raise NotImplementedError("Update plant not implemented")

    async def delete_plant(self, user: User, plant_id: UUID) -> bool:
        """
        Delete a plant.

        Args:
            user: The authenticated user
            plant_id: The plant's UUID

        Returns:
            bool: True if deleted, False if not found

        TODO: Implement plant deletion.
        """
        raise NotImplementedError("Delete plant not implemented")

    async def get_plant_by_number(
        self,
        user: User,
        plant_number: str,
    ) -> PlantResponse | None:
        """
        Get a plant by its number in the orchard.

        Args:
            user: The authenticated user
            plant_number: The plant's identification number

        Returns:
            PlantResponse | None: Plant data or None if not found

        TODO: Implement plant lookup by number.
        """
        raise NotImplementedError("Get plant by number not implemented")


# Singleton instance
plant_service = PlantService()

