from sqlalchemy.orm import Session
from fastapi import Depends
from typing import Generator

from app.db.db import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """Dependency function that yields a new SQLAlchemy session for each request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from typing import Annotated
SessionDep = Annotated[Session, Depends(get_db)]