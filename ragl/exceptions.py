__all__ = (
    'ConfigurationError',
    'DataError',
    'QueryError',
    'RAGLException',
    'StorageCapacityError',
    'StorageConnectionError',
    'StorageError',
    'ValidationError',
)


class RAGLException(Exception):
    """Base exception for all ragl errors."""


class ConfigurationError(RAGLException):
    """Raised when setup fails."""


class StorageError(RAGLException):
    """Base exception for vector store errors."""


class StorageCapacityError(StorageError):
    """Raised when vector store capacity is exceeded."""


class StorageConnectionError(StorageError):
    """Raised when a vector store connection fails."""


class DataError(RAGLException):
    """Raised when data operations fail due to invalid data."""


class QueryError(DataError):
    """Raised when a retrieval operation fails."""


class ValidationError(RAGLException):
    """Raised when input validation fails."""
