"""Analysis-related Pydantic models for image analysis pipeline."""

from pydantic import BaseModel

from app.models.queue import QueueMessage


class AnalysisResponse(BaseModel):
    """Response model for an image analysis."""
    analysis_id: str = ""
    status : str
    description : str =""

class AnalysisCreate(QueueMessage):
    """Request model for creating a new analysis."""

    image: str  # Base64 encoded image
