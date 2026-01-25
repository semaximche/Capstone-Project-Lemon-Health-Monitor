"""Chatbot routes for RAG service."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.db.models import User
from app.utils.jwt_validation import get_current_user
from app.models.rag import ChatRequest, ChatResponse
from app.rag_service.services.rag_service import rag_service

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


@router.post(
    "/query",
    response_model=ChatResponse,
    summary="Query Chatbot",
    description="Send a query to the RAG chatbot and get a response.",
)
def query_chatbot(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    """Process a chat query using RAG."""
    try:
        response = rag_service.query(query=request.query)
        print(response)
        return ChatResponse(response=response, query=request.query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}",
        )


