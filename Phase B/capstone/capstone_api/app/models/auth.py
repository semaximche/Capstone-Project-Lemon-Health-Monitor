"""Authentication-related Pydantic models."""

from pydantic import BaseModel, Field

class UserCreate(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    """OAuth token response."""

    access_token: str = Field(description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(description="Token expiration time in seconds")
    refresh_token: str | None = Field(default=None, description="Refresh token")


class GoogleAuthCallback(BaseModel):
    """Google OAuth callback data."""

    code: str = Field(description="Authorization code from Google")
    state: str | None = Field(default=None, description="State parameter for CSRF protection")


class LoginResponse(BaseModel):
    """Login initiation response."""

    auth_url: str = Field(description="URL to redirect user for authentication")
    state: str = Field(description="State parameter for CSRF protection")


class LogoutResponse(BaseModel):
    """Logout response."""

    message: str = Field(default="Successfully logged out")


class CurrentUserResponse(BaseModel):
    """Current authenticated user information."""

    id: str = Field(description="User ID")
    email: str = Field(description="User email")
    name: str = Field(description="User display name")
    picture: str | None = Field(default=None, description="User profile picture URL")

