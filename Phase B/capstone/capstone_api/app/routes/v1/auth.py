"""Authentication routes for Google OAuth integration."""
from http.client import HTTPException

from fastapi.security import OAuth2PasswordRequestForm

from app.services.user_service import user_service
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends
from app.db.db import get_db
from app.core.security import verify_password,create_access_token
from app.models.auth import (
    LoginResponse,
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


