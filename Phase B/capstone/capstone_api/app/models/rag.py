"""RAG-related Pydantic models."""

from uuid import UUID
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat query."""

    query: str = Field(min_length=1, max_length=2000, description="User query string")


class ChatResponse(BaseModel):
    """Response model for chat query."""

    response: str = Field(description="Generated response from RAG system")
    query: str = Field(description="Original user query")


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""

    file_path: str = Field(description="Path to the document file")
    metadata: dict | None = Field(default=None, description="Optional document metadata")


class DocumentResponse(BaseModel):
    """Response model for document information."""

    id: str = Field(description="Document ID")
    source: str = Field(description="Document source/path")
    chunk_count: int = Field(description="Number of chunks created from document")
    metadata: dict | None = Field(default=None, description="Document metadata")
