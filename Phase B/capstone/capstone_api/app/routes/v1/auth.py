"""Authentication routes for Google OAuth integration."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.services.user_service import user_service
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends
from app.db.db import get_db
from app.core.security import verify_password,create_access_token
from app.models.auth import (
    LoginResponse, SignupRequest,
)

from app.settings import settings
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login",
    description="Check user credentials",
)
def login(request: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> LoginResponse:
    """
    Validate user credentials using SQLAlchemy session.
    """
    user = user_service.get_user_by_username(db, request.username)
    if not user or not verify_password(request.password, user.password):
        raise HTTPException(
        )

    access_token = create_access_token(data={"sub": user.id},expires_minutes=settings.jwt_access_token_expire_minutes)

    return LoginResponse(access_token=access_token)


@router.post(
    "/signup",
    response_model=LoginResponse,
    summary="Signup",
    description="Create a new user account",
    status_code=status.HTTP_201_CREATED,
)
def signup(
    request: SignupRequest,
    db: Session = Depends(get_db),
) -> LoginResponse:
    # Check if username already exists
    if user_service.get_user_by_username(db, request.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )

    # Check if email already exists
    if user_service.get_user_by_email(db, request.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists",
        )

    # Create user
    user = user_service.create_user(
        db=db,
        user_name=request.username,
        email=request.email,
        password=request.password,
    )

    # Create JWT
    access_token = create_access_token(
        data={"sub": user.id},
        expires_minutes=settings.jwt_access_token_expire_minutes,
    )

    return LoginResponse(access_token=access_token)