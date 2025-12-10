"""Authentication routes for Google OAuth integration."""
from http.client import HTTPException

from fastapi import APIRouter, Query

from app.core.dependencies import CurrentUser
from app.db.models import User
from app.db.session import SessionDep
from app.models.auth import (
    CurrentUserResponse,
    LoginResponse,
    LogoutResponse,
    TokenResponse, UserCreate,
)
from app.services.auth_service import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get(
    "/login",
    response_model=LoginResponse,
    summary="Initiate Google OAuth Login",
    description="Returns the Google OAuth authorization URL for user authentication.",
)
async def login() -> LoginResponse:
    """Initiate the Google OAuth login flow."""
    raise NotImplementedError()


@router.post(
    "/signin",
    response_model=LoginResponse,
    summary="Sign Up (Registration)",
    description="Creates a new user with username and password and returns a token.",
)
async def signin(
        user_data: UserCreate,
        db: SessionDep,
) -> LoginResponse:
    new_user: User | None = await auth_service.local_signup(
        username=user_data.username,
        password=user_data.password,
        db=db
    )
    if new_user is None:
        raise HTTPException(
            detail="Username already registered."
        )
    # Placeholder for Token Generation
    fake_token = f"auth_token_for_user_{new_user.id}"
    return LoginResponse(
        token=fake_token,
        user_id=new_user.id
    )






@router.get(
    "/callback",
    response_model=TokenResponse,
    summary="OAuth Callback",
    description="Handle the Google OAuth callback and exchange the authorization code for tokens.",
)
async def oauth_callback(
    code: str = Query(..., description="Authorization code from Google"),
    state: str | None = Query(default=None, description="State parameter for CSRF verification"),
) -> TokenResponse:
    """Handle the Google OAuth callback."""
    raise NotImplementedError()


@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Logout",
    description="Log out the current user and invalidate their session.",
)
async def logout(current_user: CurrentUser) -> LogoutResponse:
    """Log out the current user."""
    raise NotImplementedError()


@router.get(
    "/me",
    response_model=CurrentUserResponse,
    summary="Get Current User",
    description="Get information about the currently authenticated user.",
)
async def get_current_user(current_user: CurrentUser) -> CurrentUserResponse:
    """Get the current authenticated user's information."""
    raise NotImplementedError()


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh Token",
    description="Refresh an expired access token using a refresh token.",
)
async def refresh_token(
    refresh_token: str = Query(..., description="Refresh token"),
) -> TokenResponse:
    """Refresh an expired access token."""
    raise NotImplementedError()
