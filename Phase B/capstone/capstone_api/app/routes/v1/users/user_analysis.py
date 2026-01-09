"""User analysis routes for retrieving analyses for a specific user."""
from fastapi import HTTPException

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from typing import List
from app.services.user_service import user_service
from app.db.db import get_db
from app.db.models import User, Analysis
from app.models.analysis import AnalysisResponse
from app.services.analysis_service import analysis_service
from app.utils.jwt_validation import get_current_user

router = APIRouter(prefix="/me/analysis", tags=["user-analysis"])


@router.get(
    "",
    response_model=List[AnalysisResponse],
    summary="Get Current User Analyses",
    description="Get all analyses created by the currently logged-in user.",
)
async def get_current_user_analyses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[AnalysisResponse]:
    """Get all analyses for the current logged-in user."""

    # TODO:need to add pagination param to the request
    return await analysis_service.get_analyses_by_user(current_user.id, db)




@router.get(
    "/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get specific analyses for a specific user",
    description="Get specific analysis created by the currently logged-in user.",
)
async def get_specific_analyses_for_current_user(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisResponse:

    """Get all analyses for the current logged-in user."""
    analysis = await analysis_service.get_analysis(analysis_id,db)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if analysis.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="this uer cannot access this analysis",
        )

    analysis_response = AnalysisResponse(id=analysis.id, description=analysis.description, summary=analysis.summary)
    return analysis_response