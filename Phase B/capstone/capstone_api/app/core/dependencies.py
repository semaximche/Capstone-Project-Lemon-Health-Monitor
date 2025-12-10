"""FastAPI dependency injection functions."""

from typing import Annotated, Generator

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer

from app.db.models import User

# OAuth2 scheme for bearer token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/token", auto_error=False)


def get_db() -> Generator:
    """
    Database session dependency.

    Yields a database session and ensures proper cleanup.
    """
    raise NotImplementedError()


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
) -> User:
    """
    Get the current authenticated user from the JWT token.

    Args:
        token: JWT bearer token from request header

    Returns:
        User: The authenticated user object

    Raises:
        HTTPException: If token is invalid or user not found
    """
    raise NotImplementedError()


async def get_current_user_optional(
    token: Annotated[str | None, Depends(oauth2_scheme)],
) -> User | None:
    """
    Optionally get the current authenticated user.

    Returns None if no valid token is provided instead of raising an exception.

    Args:
        token: JWT bearer token from request header

    Returns:
        User | None: The authenticated user object or None
    """
    raise NotImplementedError()


# Type aliases for dependency injection
CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[User | None, Depends(get_current_user_optional)]
DBSession = Annotated[Generator, Depends(get_db)]
