"""RAG-specific exceptions."""


class RAGException(Exception):
    """Base exception for RAG service errors."""
    pass


class VectorStoreException(RAGException):
    """Exception raised for vector store operations."""
    pass


class LLMException(RAGException):
    """Exception raised for LLM operations."""
    pass


class DocumentException(RAGException):
    """Exception raised for document operations."""
    pass
