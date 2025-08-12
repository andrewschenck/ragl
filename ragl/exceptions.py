"""
Exception classes for the ragl library.

This module defines a hierarchy of custom exceptions used throughout
the ragl library. All exceptions inherit from RaglException.

The exception hierarchy is organized by functional area:
- Configuration errors for setup and validation issues
- Storage errors for vector store operations and connections
- Data errors for invalid data and query operations

Classes:
    RaglException:
        Base exception for all ragl errors
    ConfigurationError:
        Setup and configuration failures
    StorageError:
        Base exception for VectorStore errors
    StorageCapacityError:
        VectorStore capacity exceeded
    StorageConnectionError:
        VectorStore connection failures
    DataError:
        General data operation failures
    QueryError:
        Retrieval operation failures
    ValidationError:
        Input validation failures
"""

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
    """Base exception for VectorStore errors."""


class StorageCapacityError(StorageError):
    """Raised when VectorStore capacity is exceeded."""


class StorageConnectionError(StorageError):
    """Raised when a VectorStore connection fails."""


class DataError(RaglException):
    """Raised when data operations fail due to invalid data."""


class QueryError(DataError):
    """Raised when a retrieval operation fails."""


class ValidationError(RaglException):
    """Raised when input validation fails."""
