"""Custom exception classes for the application."""

from fastapi import HTTPException, status


class CapstoneException(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, error_code: str | None = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class EntityNotFoundError(CapstoneException):
    """Raised when a requested entity is not found."""

    def __init__(self, entity_type: str, entity_id: str):
        super().__init__(
            message=f"{entity_type} with ID {entity_id} not found",
            error_code="ENTITY_NOT_FOUND",
        )
        self.entity_type = entity_type
        self.entity_id = entity_id


class AuthenticationError(CapstoneException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message=message, error_code="AUTHENTICATION_ERROR")


class AuthorizationError(CapstoneException):
    """Raised when user is not authorized to perform an action."""

    def __init__(self, message: str = "Not authorized to perform this action"):
        super().__init__(message=message, error_code="AUTHORIZATION_ERROR")


class ValidationError(CapstoneException):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message=message, error_code="VALIDATION_ERROR")
        self.field = field


class ExternalServiceError(CapstoneException):
    """Raised when an external service (e.g., Home Assistant, Google) fails."""

    def __init__(self, service_name: str, message: str):
        super().__init__(
            message=f"{service_name} error: {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
        )
        self.service_name = service_name


class AnalysisPipelineError(CapstoneException):
    """Raised when the analysis pipeline fails."""

    def __init__(self, message: str, stage: str | None = None):
        super().__init__(message=message, error_code="ANALYSIS_PIPELINE_ERROR")
        self.stage = stage


class StorageError(CapstoneException):
    """Raised when object storage operations fail."""

    def __init__(self, message: str, operation: str | None = None):
        super().__init__(message=message, error_code="STORAGE_ERROR")
        self.operation = operation


# HTTP Exception helpers
def not_found_exception(entity_type: str, entity_id: str) -> HTTPException:
    """Create a 404 Not Found HTTPException."""
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"{entity_type} with ID {entity_id} not found",
    )


def unauthorized_exception(message: str = "Not authenticated") -> HTTPException:
    """Create a 401 Unauthorized HTTPException."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=message,
        headers={"WWW-Authenticate": "Bearer"},
    )


def forbidden_exception(message: str = "Not authorized") -> HTTPException:
    """Create a 403 Forbidden HTTPException."""
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=message,
    )


def bad_request_exception(message: str) -> HTTPException:
    """Create a 400 Bad Request HTTPException."""
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=message,
    )

