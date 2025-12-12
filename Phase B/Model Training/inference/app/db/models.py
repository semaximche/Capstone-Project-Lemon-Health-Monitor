from sqlalchemy import Column, Integer, String,ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from inference.app.db.db import Base
import uuid



class User(Base):
    __tablename__ = "users"
    # UUID primary key stored as string
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(100), nullable=False)


class Analysis(Base):
    __tablename__ = "analysis"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    presigned_url: Mapped[str] = mapped_column(String(255), nullable=True)
    description: Mapped[str] = mapped_column(String(500), nullable=True)
    # user = relationship("User", back_populates="analyses")


# Add back-populates in User
# User.analyses = relationship("Analysis", back_populates="user", cascade="all, delete-orphan")