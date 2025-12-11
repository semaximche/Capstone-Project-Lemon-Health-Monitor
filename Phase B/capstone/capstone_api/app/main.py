"""Main FastAPI application entry point."""

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.core.o11y.logger import init_logger
from app.middlewares.logging_middleware import LoggingMiddleware
from app.models.common import Message
from app.routes.health_router import router as health_router
from app.routes.v1.v1_router import router as v1_router

# Initialize logger
init_logger(service_name="Lemon Health")

# Create FastAPI application
app = FastAPI(
    title="Capstone API",
    description="Lemon Tree Disease Identification API - A plant disease identification system using computer vision and LLM",
    version="0.1.0",
)

# Add middlewares
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router=v1_router)
app.include_router(router=health_router, tags=["health"])


@app.get("/", tags=["root"])
def root() -> Message:
    """Root endpoint returning a welcome message."""
    return Message(message="Welcome to Capstone API - Lemon Tree Disease Identification System")


if __name__ == "__main__":
    from uvicorn import run
    run(app=app, host="127.0.0.1", port=8000)
