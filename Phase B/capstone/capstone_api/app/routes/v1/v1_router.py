"""API v1 router aggregating all v1 endpoints."""
from fastapi import APIRouter
from app.routes.v1.analysis import router as analysis_router
from app.routes.v1.auth import router as auth_router
# from app.routes.v1.users import router as users_router

router = APIRouter(prefix="/v1")

# Include all v1 routers
router.include_router(auth_router)
# router.include_router(users_router)
router.include_router(analysis_router)