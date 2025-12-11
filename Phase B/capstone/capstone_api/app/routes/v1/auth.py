"""Authentication routes for Google OAuth integration."""
from http.client import HTTPException

from sqlalchemy.orm import Session
from fastapi import APIRouter, Query , Depends
from app.db.db import get_db
from app.core.dependencies import CurrentUser
from app.db.models import User
from app.models.auth import (
    CurrentUserResponse,
    LoginResponse,
    LogoutResponse,
    TokenResponse, UserCreate,LoginRequest
)
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login",
    description="Check user credentials",
)
async def login(request: LoginRequest, db: Session = Depends(get_db)) -> LoginResponse:
    """
    Validate user credentials using SQLAlchemy session.
    """
    user  = db.query(User).filter(User.user_name == request.user_name).first()
    if not user:
        raise HTTPException()

    if not (user.password == request.password):
        raise HTTPException()

    # Login successful
    return LoginResponse(user_name=request.user_name,auth_url="logged_in", state="checked")

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
