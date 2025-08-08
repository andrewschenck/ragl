__all__ = (
    'ConfigurationError',
    'DataError',
    'QueryError',
    'RaglException',
    'StorageCapacityError',
    'StorageConnectionError',
    'StorageError',
    'ValidationError',
)


class RaglException(Exception):
    """Base exception for all ragl errors."""


class ConfigurationError(RaglException):
    """Raised when setup fails."""


class StorageError(RaglException):
    """Base exception for vector store errors."""


class StorageCapacityError(StorageError):
    """Raised when vector store capacity is exceeded."""


class StorageConnectionError(StorageError):
    """Raised when a vector store connection fails."""


class DataError(RaglException):
    """Raised when data operations fail due to invalid data."""


class QueryError(DataError):
    """Raised when a retrieval operation fails."""


class ValidationError(RaglException):
    """Raised when input validation fails."""
