"""Analysis service for image analysis pipeline operations."""
import base64
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from app.db.models import Analysis
from app.utils.rabbitmq import publisher
from fastapi import  UploadFile
from app.models.analysis import (
    AnalysisResponse,
)
from app.settings import settings
from app.crud.analysis import analysis_crud
from sqlalchemy.orm import Session
from app.db.models import User
class AnalysisService:
    """Service for handling image analysis operations."""

    async def create_analysis(
        self,
        user: User,
        image: UploadFile,

    ) -> AnalysisResponse:
        """
        Create a new image analysis.
        """

        image_bytes = await image.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        job_payload = {
            "type":"analysis.requested",
            "user_id": user.id,
            "image": image_base64,
        }
        try:
            print("start publish - request to start analysis")
            publisher.publish_event(routing_key="analysis.requested", payload=job_payload)

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

        return AnalysisResponse(description="analysis sent to queue")


    async def get_analysis(
        self,
        # user: User,
        analysis_id: str,
        db: Session
    ) -> Analysis | None:
        """
        Get an analysis by ID.
        """

        response = analysis_crud.get(db,str(analysis_id))
        if response:
            return response
        return None

    async def delete_analysis(
        self,
        analysis_id: str,
        db: Session)-> bool | None:
        """
        Delete analysis by ID.
        """
        try:
            response = analysis_crud.delete(db, str(analysis_id))
            return response
        except:
            return False


    async def get_analyses_by_user(self,user_id:str, db: Session) -> list[type[Analysis]]:

        try:
            reponse= analysis_crud.get_by_user_id(db,user_id)
            return reponse
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            )





analysis_service = AnalysisService()

