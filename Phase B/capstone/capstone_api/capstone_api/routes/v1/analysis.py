"""Analysis routes for image analysis pipeline operations."""
import base64
import os
from datetime import datetime
from http.client import HTTPException
from uuid import UUID
from capstone_api.utils.rabbitmq import RabbitMQPublisher
from fastapi import APIRouter, File, Form, Query, UploadFile, status
from capstone_api.utils.rabbitmq import publisher
from capstone_api.core.dependencies import CurrentUser
from capstone_api.models.analysis import (
    AnalysisHistoryResponse,
    AnalysisResponse,
)

router = APIRouter(prefix="/analysis", tags=["analysis"])


def save_uploaded_image(image: UploadFile, analysis_id: str) -> str:
    """Saves the uploaded image and returns the path."""
    # Define a target directory (adjust as needed)
    UPLOAD_DIR = "uploaded_images"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Create a unique filename
    file_extension = os.path.splitext(image.filename)[1]
    filename = f"{analysis_id}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(file_path, "wb") as f:
            # Note: A real implementation should use a more efficient async method
            f.write(image.file.read())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save image file: {e}"
        )

    return file_path

# @router.post(
#     "",
#     response_model=AnalysisResponse,
#     status_code=status.HTTP_201_CREATED,
#     summary="Create Analysis",
#     description="Upload an image and create a new analysis for a plant.",
# )
# async def create_analysis(
#
#     image: UploadFile = File(..., description="Image file to analyze"),
#     plant_id: UUID = Form(..., description="ID of the plant being analyzed"),
#     notes: str | None = Form(default=None, description="Optional notes about the analysis"),
# ) -> None:
#     """Create a new image analysis."""
#     rabbitmq_publisher.publish_job()

@router.post(
    "",
    response_model=AnalysisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Analysis",
    description="Upload an image and create a new analysis for a plant.",
)
async def create_analysis(
        image: UploadFile = File(..., description="Image file to analyze"),
        notes: str | None = Form(default=None, description="Optional notes about the analysis"),
) -> str:
    """Create a new image analysis."""
    # 1. Generate a unique ID for this analysis
    analysis_id = "12345"
    image_bytes = await image.read()

    # Encode the bytes into a Base64 string
    # .decode('utf-8') converts the resulting bytes object to a string
    # for JSON serialization.
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # //image_path = save_uploaded_image(image, analysis_id)

    # 3. Define the job payload for the worker
    job_payload = {
        "analysis_id": str(analysis_id),
        "image_path": image_base64,

    }

    # 4. Publish the job using the context manager (ensures connection is closed)
    try:
        publisher.publish_job(job_payload)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RabbitMQ service is unavailable. The job could not be queued."
        )

    return str

@router.get(
    "/history",
    response_model=AnalysisHistoryResponse,
    summary="Get Analysis History",
    description="Get historical analysis data with disease trends.",
)
async def get_analysis_history(
    current_user: CurrentUser,
    start_date: datetime | None = Query(default=None, description="Start date filter"),
    end_date: datetime | None = Query(default=None, description="End date filter"),
    plant_id: UUID | None = Query(default=None, description="Filter by plant ID"),
) -> AnalysisHistoryResponse:
    """Get analysis history with trends."""
    raise NotImplementedError()


@router.get(
    "/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get Analysis",
    description="Get details of a specific analysis by ID.",
)
async def get_analysis(
    analysis_id: UUID,
    current_user: CurrentUser,
) -> AnalysisResponse:
    """Get an analysis by ID."""
    raise NotImplementedError()


@router.delete(
    "/{analysis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Analysis",
    description="Delete an analysis and its associated data.",
)
async def delete_analysis(
    analysis_id: UUID,
    current_user: CurrentUser,
) -> None:
    """Delete an analysis."""
    raise NotImplementedError()
