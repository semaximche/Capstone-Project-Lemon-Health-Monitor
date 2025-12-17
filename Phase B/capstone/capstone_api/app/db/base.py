"""SQLModel base configuration."""
from datetime import datetime
from uuid import UUID, uuid4
from sqlmodel import Field, SQLModel
from app.db import Base

class BaseModel(SQLModel):
    """Base model with UUID primary key."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)


class TimestampModel(BaseModel):
    """Base model with UUID primary key and timestamps."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = Field(default=None)
