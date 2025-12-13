"""Analysis service for image analysis pipeline operations."""
import base64
from http.client import HTTPException
from app.utils.rabbitmq import publisher
from fastapi import  UploadFile
from app.models.analysis import (
    AnalysisResponse,
)
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
            "user_id": user.id,
            "image": image_base64,
        }
        try:
            print("start publish")
            publisher.publish_job(job_payload)

        except Exception as e:
            raise HTTPException(

            )

        return AnalysisResponse(status="analysis sent to queue")


    async def get_analysis(
        self,
        # user: User,
        analysis_id: str,
        db: Session
    ) -> AnalysisResponse | None:
        """
        Get an analysis by ID.
        """

        response = analysis_crud.get(db,str(analysis_id))
        if response:
            analysis_response = AnalysisResponse(analysis_id=response.id,status="ok",description=response.description)
            return analysis_response
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


analysis_service = AnalysisService()

