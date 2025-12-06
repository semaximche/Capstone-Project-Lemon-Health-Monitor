"""Camera routes for IoT camera management via Home Assistant."""

from uuid import UUID

from fastapi import APIRouter, status

from capstone_api.core.dependencies import CurrentUser
from capstone_api.models.camera import (
    CameraCaptureResponse,
    CameraCreate,
    CameraListResponse,
    CameraResponse,
    CameraTestResponse,
)

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get(
    "",
    response_model=CameraListResponse,
    summary="List Cameras",
    description="List all registered cameras for the authenticated user.",
)
async def list_cameras(current_user: CurrentUser) -> CameraListResponse:
    """List all cameras."""
    raise NotImplementedError()


@router.post(
    "",
    response_model=CameraResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register Camera",
    description="Register a new camera from Home Assistant.",
)
async def register_camera(
    camera_data: CameraCreate,
    current_user: CurrentUser,
) -> CameraResponse:
    """Register a new camera."""
    raise NotImplementedError()


@router.get(
    "/{camera_id}",
    response_model=CameraResponse,
    summary="Get Camera",
    description="Get details of a specific camera by ID.",
)
async def get_camera(
    camera_id: UUID,
    current_user: CurrentUser,
) -> CameraResponse:
    """Get a camera by ID."""
    raise NotImplementedError()


@router.delete(
    "/{camera_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Camera",
    description="Remove a registered camera.",
)
async def delete_camera(
    camera_id: UUID,
    current_user: CurrentUser,
) -> None:
    """Delete a camera."""
    raise NotImplementedError()


@router.post(
    "/{camera_id}/capture",
    response_model=CameraCaptureResponse,
    summary="Capture Image",
    description="Capture an image from the camera via Home Assistant.",
)
async def capture_image(
    camera_id: UUID,
    current_user: CurrentUser,
) -> CameraCaptureResponse:
    """Capture an image from the camera."""
    raise NotImplementedError()


@router.post(
    "/{camera_id}/test",
    response_model=CameraTestResponse,
    summary="Test Camera Connection",
    description="Test the connection to a camera via Home Assistant.",
)
async def test_camera_connection(
    camera_id: UUID,
    current_user: CurrentUser,
) -> CameraTestResponse:
    """Test camera connection."""
    raise NotImplementedError()
