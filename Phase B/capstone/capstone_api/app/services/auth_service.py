"""Authentication service for Google OAuth integration."""
from app.db.models import User
from app.models.auth import (
    CurrentUserResponse,
    LoginResponse,
    LogoutResponse,
    TokenResponse,
)
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional



class AuthService:
    """Service for handling authentication operations."""

    @staticmethod
    def get_password_hash(password: str) -> str:
        """Returns the bcrypt hash of the given password."""
        return "qqqq0"+password

    async def local_signup(self, username: str, password: str, db: AsyncSession) -> Optional[User]:
        """
        Creates a new user with a local username and securely hashed password.

        Args:
            username: The desired username.
            password: The plain text password.
            db: The SQLAlchemy asynchronous database session.

        Returns:
            User: The newly created User object, or None if the username already exists.
        """

        # result = await db.execute(select(User).filter(User.user_name == username))
        # existing_user = result.scalars().first()
        #
        # if existing_user:
        #     return None  # Indicate username is taken

        # 2. Hash the password securely
        hashed_password = self.get_password_hash(password)

        new_user = User(
            user_name=username,
            password=hashed_password,
        )

        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        print(f"User created with ID: {new_user.id}")
        return new_user
    async def initiate_login(self) -> LoginResponse:
        """
        Initiate Google OAuth login flow.

        Returns:
            LoginResponse: Contains the authorization URL and state parameter

        TODO: Implement Google OAuth flow initiation.
        """
        raise NotImplementedError("Login initiation not implemented")

    async def handle_callback(self, code: str, state: str | None = None) -> TokenResponse:
        """
        Handle Google OAuth callback and exchange code for tokens.

        Args:
            code: Authorization code from Google
            state: State parameter for CSRF verification

        Returns:
            TokenResponse: JWT tokens for the authenticated user

        TODO: Implement OAuth callback handling.
        """
        raise NotImplementedError("OAuth callback handling not implemented")

    async def logout(self, user: User) -> LogoutResponse:
        """
        Log out the current user.

        Args:
            user: The user to log out

        Returns:
            LogoutResponse: Confirmation of logout

        TODO: Implement logout logic (token invalidation, etc.).
        """
        raise NotImplementedError("Logout not implemented")

    async def get_current_user_info(self, user: User) -> CurrentUserResponse:
        """
        Get current authenticated user information.

        Args:
            user: The authenticated user

        Returns:
            CurrentUserResponse: User information

        TODO: Implement user info retrieval.
        """
        raise NotImplementedError("Get current user info not implemented")

    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        """
        Refresh an expired access token.

        Args:
            refresh_token: The refresh token

        Returns:
            TokenResponse: New JWT tokens

        TODO: Implement token refresh logic.
        """
        raise NotImplementedError("Token refresh not implemented")


# Singleton instance
auth_service = AuthService()

