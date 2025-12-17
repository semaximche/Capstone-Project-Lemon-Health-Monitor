"""Analysis routes for image analysis pipeline operations."""
from sqlalchemy.orm import Session
from fastapi import  Depends
from fastapi import APIRouter, File, UploadFile, status
from app.services.analysis_service import analysis_service
from app.models.analysis import (
    AnalysisResponse,
)
from app.db.models import User
from app.db.db import get_db
from app.utils.jwt_validation import get_current_user

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post(
    "",
    response_model=AnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Analysis",
    description="Upload an image and create a new analysis for a plant.",
)
async def create_analysis(
        image: UploadFile = File(..., description="Image file to analyze"),
        current_user: User = Depends(get_current_user),
) -> AnalysisResponse:
    """Create a new image analysis."""
    return await analysis_service.create_analysis(current_user,image)





@router.get(
    "/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get Analysis",
    description="Get details of a specific analysis by ID.",
)
async def get_analysis(
    analysis_id: str,
    db: Session = Depends(get_db),
    # current_user: User = Depends(get_current_user),

) -> AnalysisResponse:
    return await analysis_service.get_analysis(analysis_id, db)




@router.delete(
    "/{analysis_id}",
    response_model=bool,
    summary="Delete Analysis",
    description="Delete an analysis and its associated data.",
)
async def delete_analysis(
    analysis_id: str,
    db: Session = Depends(get_db),
    # current_user: User = Depends(get_current_user),
) -> bool:
    """Delete an analysis."""
    return await analysis_service.delete_analysis(analysis_id, db)

