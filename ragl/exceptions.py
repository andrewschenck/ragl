__all__ = (
    'ConfigurationError',
    'DataError',
    'QueryError',
    'RedisCapacityError',
    'StorageConnectionError',
    'StorageError',
    'ValidationError',
)


class StorageError(Exception):  # todo exception hierarchy / taxonomy
    """Base exception for all StorageStrategy errors."""


class ConfigurationError(StorageError):
    """Raised when storage backend setup fails."""


class DataError(StorageError):
    """Raised when data operations fail due to invalid data."""


class QueryError(StorageError):
    """Raised when a retrieval operation fails."""


class RedisCapacityError(StorageError):
    """Raised when Redis storage capacity is exceeded."""


class StorageConnectionError(StorageError):
    """Raised when a storage backend connection fails."""


class ValidationError(StorageError):
    """Raised when input validation fails."""
