"""User routes for profile and settings management."""

# from fastapi import APIRouter
# from app.models.user import UserSettingsResponse, UserSettingsUpdate
# from app.db.models import User
# router = APIRouter(prefix="/users", tags=["users"])
#
#
# @router.get(
#     "/settings",
#     response_model=UserSettingsResponse,
#     summary="Get User Settings",
#     description="Retrieve the current user's settings including language and notification preferences.",
# )
# async def get_settings(current_user: User) -> UserSettingsResponse:
#     """Get the current user's settings."""
#     raise NotImplementedError()
#
#
# @router.put(
#     "/settings",
#     response_model=UserSettingsResponse,
#     summary="Update User Settings",
#     description="Update the current user's settings. Only provided fields will be updated.",
# )
# async def update_settings(
#     settings_update: UserSettingsUpdate,
#     current_user: User,
# ) -> UserSettingsResponse:
#     """Update the current user's settings."""
#     raise NotImplementedError()
