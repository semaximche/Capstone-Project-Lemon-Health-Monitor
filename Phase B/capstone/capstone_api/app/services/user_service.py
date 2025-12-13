"""User service for user profile and settings management."""
from sqlalchemy.orm import Session
from uuid import UUID
from app.crud.user import user_crud
from app.db.models import User
from app.models.user import (
    UserResponse,
    UserSettings,
    UserSettingsResponse,
    UserSettingsUpdate,
)


class UserService:
    """Service for handling user operations."""


    def get_user_by_username(self,db: Session,user_name:str) -> UserResponse | None:
            current_user = user_crud.get_user_by_username(db,user_name)
            if current_user:
                user_response = UserResponse(id=current_user.id, user_name=current_user.user_name,
                                             password=current_user.password)
                return user_response
            return None

    def get_user_by_id(self,db: Session, user_id: str) -> UserResponse | None:
        """
        Get a user by their ID.
        """
        current_user = user_crud.get_user_by_id(db, user_id)
        if current_user:
            user_response = UserResponse(id=current_user.id, user_name=current_user.user_name,
                                         password=current_user.password)
            return user_response
        return None


    # async def get_user_by_email(self, email: str) -> UserResponse | None:
    #     """
    #     Get a user by their email address.
    #
    #     Args:
    #         email: The user's email
    #
    #     Returns:
    #         UserResponse | None: User data or None if not found
    #
    #     TODO: Implement database lookup.
    #     """
    #     raise NotImplementedError("Get user by email not implemented")
    #
    # async def get_user_by_google_id(self, google_id: str) -> UserResponse | None:
    #     """
    #     Get a user by their Google ID.
    #
    #     Args:
    #         google_id: The user's Google ID
    #
    #     Returns:
    #         UserResponse | None: User data or None if not found
    #
    #     TODO: Implement database lookup.
    #     """
    #     raise NotImplementedError("Get user by Google ID not implemented")
    #
    # async def create_user(
    #     self,
    #     email: str,
    #     name: str,
    #     google_id: str,
    #     picture: str | None = None,
    # ) -> UserResponse:
    #     """
    #     Create a new user account.
    #
    #     Args:
    #         email: User's email address
    #         name: User's display name
    #         google_id: User's Google ID
    #         picture: User's profile picture URL
    #
    #     Returns:
    #         UserResponse: The created user
    #
    #     TODO: Implement user creation.
    #     """
    #     raise NotImplementedError("Create user not implemented")
    #
    # async def get_settings(self, user: User) -> UserSettingsResponse:
    #     """
    #     Get user settings.
    #
    #     Args:
    #         user: The authenticated user
    #
    #     Returns:
    #         UserSettingsResponse: User's settings
    #
    #     TODO: Implement settings retrieval.
    #     """
    #     raise NotImplementedError("Get settings not implemented")
    #
    # async def update_settings(
    #     self,
    #     user: User,
    #     settings_update: UserSettingsUpdate,
    # ) -> UserSettingsResponse:
    #     """
    #     Update user settings.
    #
    #     Args:
    #         user: The authenticated user
    #         settings_update: The settings to update
    #
    #     Returns:
    #         UserSettingsResponse: Updated settings
    #
    #     TODO: Implement settings update.
    #     """
    #     raise NotImplementedError("Update settings not implemented")
    #
    # async def delete_user(self, user: User) -> None:
    #     """
    #     Delete a user account and all associated data.
    #
    #     Args:
    #         user: The user to delete
    #
    #     TODO: Implement user deletion with cascade.
    #     """
    #     raise NotImplementedError("Delete user not implemented")


# Singleton instance
user_service = UserService()

