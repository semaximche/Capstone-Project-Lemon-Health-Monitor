"""SQLModel ORM models for all database entities."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import JSON
from sqlmodel import Field, Relationship, SQLModel

from capstone_api.db.base import TimestampModel


class User(TimestampModel, table=True):
    """User account model."""

    __tablename__ = "users"

    email: str = Field(max_length=255, unique=True, index=True)
    name: str = Field(max_length=255)
    google_id: str = Field(max_length=255, unique=True, index=True)
    picture: str | None = Field(default=None, max_length=500)
    settings: dict[str, Any] = Field(default_factory=dict, sa_type=JSON)

    # Relationships
    plants: list["Plant"] = Relationship(back_populates="user", cascade_delete=True)
    cameras: list["Camera"] = Relationship(back_populates="user", cascade_delete=True)
    monitoring_configs: list["MonitoringConfig"] = Relationship(back_populates="user", cascade_delete=True)


class Plant(TimestampModel, table=True):
    """Plant entity model representing a plant in the user's orchard."""

    __tablename__ = "plants"

    user_id: UUID = Field(foreign_key="users.id", index=True)
    name: str = Field(max_length=100)
    plant_number: str = Field(max_length=50)
    location: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None)

    # Relationships
    user: User = Relationship(back_populates="plants")
    analyses: list["ImageAnalysis"] = Relationship(back_populates="plant", cascade_delete=True)
    monitoring_configs: list["MonitoringConfig"] = Relationship(back_populates="plant", cascade_delete=True)


class ImageAnalysis(TimestampModel, table=True):
    """Image analysis model containing results from the identification pipeline."""

    __tablename__ = "image_analyses"

    plant_id: UUID = Field(foreign_key="plants.id", index=True)
    image_url: str = Field(max_length=500)
    status: str = Field(default="pending", max_length=20)
    recommendation: str | None = Field(default=None)
    notes: str | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)

    # Relationships
    plant: Plant = Relationship(back_populates="analyses")
    leaf_detections: list["LeafDetection"] = Relationship(back_populates="analysis", cascade_delete=True)


class LeafDetection(SQLModel, table=True):
    """Individual leaf detection result from the YOLOv11 and disease classification models."""

    __tablename__ = "leaf_detections"

    id: UUID = Field(primary_key=True)
    analysis_id: UUID = Field(foreign_key="image_analyses.id", index=True)

    # Bounding box coordinates
    bbox_x_min: float
    bbox_y_min: float
    bbox_x_max: float
    bbox_y_max: float

    # Diagnosis results
    diagnosis: str = Field(max_length=50)
    confidence: float
    disease_details: str | None = Field(default=None)

    # Relationships
    analysis: ImageAnalysis = Relationship(back_populates="leaf_detections")


class Camera(TimestampModel, table=True):
    """IoT camera model for Home Assistant integration."""

    __tablename__ = "cameras"

    user_id: UUID = Field(foreign_key="users.id", index=True)
    name: str = Field(max_length=100)
    home_assistant_entity_id: str = Field(max_length=200)
    description: str | None = Field(default=None)
    status: str = Field(default="unknown", max_length=20)
    last_capture_at: datetime | None = Field(default=None)

    # Relationships
    user: User = Relationship(back_populates="cameras")
    monitoring_configs: list["MonitoringConfig"] = Relationship(back_populates="camera", cascade_delete=True)


class MonitoringConfig(TimestampModel, table=True):
    """Monitoring configuration for automated interval-based analysis."""

    __tablename__ = "monitoring_configs"

    user_id: UUID = Field(foreign_key="users.id", index=True)
    camera_id: UUID = Field(foreign_key="cameras.id", index=True)
    plant_id: UUID = Field(foreign_key="plants.id", index=True)
    interval_minutes: int = Field(default=60)
    enabled: bool = Field(default=True)
    last_run_at: datetime | None = Field(default=None)
    next_run_at: datetime | None = Field(default=None)

    # Relationships
    user: User = Relationship(back_populates="monitoring_configs")
    camera: Camera = Relationship(back_populates="monitoring_configs")
    plant: Plant = Relationship(back_populates="monitoring_configs")
