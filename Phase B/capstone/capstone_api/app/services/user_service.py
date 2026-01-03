"""User service for user profile and settings management."""
from sqlalchemy.orm import Session
from uuid import UUID

# from app.core.security import hash_password
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

    def get_user_by_email(self, db: Session, email: str) -> UserResponse | None:
        """
        Get a user by their email.
        """
        current_user = user_crud.get_user_by_email(db, email)
        if current_user:
            user_response = UserResponse(
                id=current_user.id,
                user_name=current_user.user_name,
                password=current_user.password
            )
            return user_response
        return None

    def create_user(self, db: Session, user_name: str, email: str, password: str) -> UserResponse:
        """
        Create a new user.
        """
        # Call the user_crud to create the user
        # hashed_password = hash_password(password)

        current_user = user_crud.create_user(db, username=user_name, email=email, password=password)

        if current_user:
            user_response = UserResponse(
                id=current_user.id,
                user_name=current_user.user_name,
                password=current_user.password  # Consider hashing before returning in production
            )
            return user_response

        # If creation failed for some reason
        return None
# Singleton instance
user_service = UserService()

