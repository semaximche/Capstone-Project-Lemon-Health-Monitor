"""Analysis-related Pydantic models for image analysis pipeline."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AnalysisStatus(str, Enum):
    """Status of an image analysis."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DiseaseType(str, Enum):
    """Types of diseases that can be detected."""

    HEALTHY = "healthy"
    DEHYDRATED = "dehydrated"
    NUTRIENT_DEFICIENT = "nutrient_deficient"
    FUNGAL_INFECTION = "fungal_infection"
    BACTERIAL_INFECTION = "bacterial_infection"
    PEST_DAMAGE = "pest_damage"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected leaves."""

    x_min: float = Field(ge=0, description="Left edge coordinate")
    y_min: float = Field(ge=0, description="Top edge coordinate")
    x_max: float = Field(ge=0, description="Right edge coordinate")
    y_max: float = Field(ge=0, description="Bottom edge coordinate")


class LeafDetectionResult(BaseModel):
    """Result of leaf detection and disease classification for a single leaf."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    bounding_box: BoundingBox
    diagnosis: DiseaseType
    confidence: float = Field(ge=0, le=1, description="Confidence score of the diagnosis")
    disease_details: str | None = Field(default=None, description="Additional details about the disease")


class AnalysisCreate(BaseModel):
    """Request model for creating a new analysis."""

    plant_id: UUID = Field(description="ID of the plant being analyzed")
    notes: str | None = Field(default=None, max_length=500, description="Optional notes about the analysis")


class AnalysisResponse(BaseModel):
    """Response model for an image analysis."""
    status : str
    # model_config = ConfigDict(from_attributes=True)
    #
    # id: UUID
    # plant_id: UUID
    # image_url: str = Field(description="URL of the analyzed image")
    # status: AnalysisStatus
    # leaf_detections: list[LeafDetectionResult] = Field(default_factory=list)
    # recommendation: str | None = Field(default=None, description="LLM-generated recommendation")
    # notes: str | None = None
    # created_at: datetime
    # completed_at: datetime | None = None


class AnalysisSummary(BaseModel):
    """Summary view of an analysis for list responses."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    plant_id: UUID
    status: AnalysisStatus
    disease_count: int = Field(description="Number of diseases detected")
    created_at: datetime


class AnalysisListResponse(BaseModel):
    """Response model for listing analyses."""

    items: list[AnalysisSummary]
    total: int = Field(description="Total number of analyses")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")
    total_pages: int = Field(description="Total number of pages")


class AnalysisHistoryResponse(BaseModel):
    """Response model for analysis history with trends."""

    analyses: list[AnalysisSummary]
    total_analyses: int = Field(description="Total number of analyses")
    disease_trends: dict[str, int] = Field(
        default_factory=dict, description="Count of each disease type over time"
    )
    period_start: datetime | None = Field(default=None, description="Start of the analysis period")
    period_end: datetime | None = Field(default=None, description="End of the analysis period")

