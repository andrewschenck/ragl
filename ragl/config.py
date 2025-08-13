"""
Configuration classes for the ragl library.

This module defines dataclass-based configuration objects for ragl
system components including embedders, vector stores, and the RAG
manager. All configurations inherit from RaglConfig and include
validation.

Classes:
    RaglConfig:
        Base class for all ragl configurations.
    EmbedderConfig:
        Base Embedder configuration
    VectorStoreConfig:
        Base VectorStore configuration
    SentenceTransformerConfig:
        SentenceTransformer concrete embedder config
    RedisConfig:
        RedisVectorStore concrete VectorStore configuration
    ManagerConfig:
        RAGManager configuration
"""

import re
from dataclasses import dataclass
from typing import Any

from ragl.exceptions import ConfigurationError


__all__ = (
    'EmbedderConfig',
    'ManagerConfig',
    'RaglConfig',
    'RedisConfig',
    'SentenceTransformerConfig',
    'VectorStoreConfig',
)


@dataclass
class RaglConfig:
    """
    Base configuration class for ragl.

    This class serves as a base for all configuration classes in ragl.
    """

    def __str__(self) -> str:
        """Return a string representation of the configuration."""
        class_name = self.__class__.__name__
        fields = []

        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, str):
                fields.append(f'{field_name}="{field_value}"')
            else:
                fields.append(f'{field_name}={field_value}')

        return f'{class_name}({", ".join(fields)})'

    def __repr__(self) -> str:
        """Return a detailed string representation of the configuration."""
        return self.__str__()


@dataclass
class EmbedderConfig(RaglConfig):
    """Base configuration for EmbedderProtocol implementations."""


@dataclass
class VectorStoreConfig(RaglConfig):
    """Base configuration for VectorStore implementations."""


@dataclass
class SentenceTransformerConfig(EmbedderConfig):
    """
    Configuration for SentenceTransformer.

    This class provides configuration options for the
    SentenceTransformer embedder used in the ragl system.

    It exposes parameters which are injected into the
    SentenceTransformer class, allowing for flexible
    configuration of the embedding model used in the ragl
    system.

    Parameters are validated during initialization to ensure
    that they meet the requirements for the underlying
    SentenceTransformer class.

    Attributes:
        model_name_or_path:
            Name of the Hugging Face model to use for embedding.
            This parameter is passed directly to the underlying
            SentenceTransformer class and can be any compatible
            model name from Hugging Face's model hub or a path
            to a model on disk.

        cache_maxsize:
            Maximum number of embeddings to cache in memory.
        device:
            Device to use for embedding.
        auto_clear_cache:
            Whether to automatically clean up the cache when
            memory usage exceeds the threshold.
        show_progress:
            Whether to show progress bars during embedding.
        memory_threshold:
            Threshold for memory usage before cleaning up cache.
            This is a float between 0.0 and 1.0, where 1.0 means
            100% memory usage.
    """

    model_name_or_path: str = 'all-mpnet-base-v2'
    cache_maxsize: int = 10_000  # set this to 0 to disable caching
    device: str | None = None
    auto_clear_cache: bool = True
    show_progress: bool = False
    memory_threshold: float = 0.9

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_model_name()
        self._validate_cache_settings()

    def _validate_model_name(self) -> None:
        """
        Validate model name.

        Raises:
            ConfigurationError:
                If model_name_or_path is empty.
        """
        model_name_or_path = str(self.model_name_or_path)
        if not model_name_or_path or not model_name_or_path.strip():
            raise ConfigurationError('model_name_or_path cannot be empty')

    def _validate_cache_settings(self) -> None:
        """
        Validate cache settings.

        Raises:
            ConfigurationError:
                If cache_maxsize is not a non-negative integer,
        """
        if not isinstance(self.cache_maxsize, int) or self.cache_maxsize < 0:
            raise ConfigurationError(f'{self.cache_maxsize} must be '
                                     'non-negative')

        if not 0.0 <= self.memory_threshold <= 1.0:
            raise ConfigurationError(f'{self.memory_threshold=} must be '
                                     'between 0.0 and 1.0')


@dataclass
class RedisConfig(VectorStoreConfig):
    # pylint: disable=too-many-instance-attributes
    """
    Configuration for RedisVectorStore.

    This class provides configuration options for the
    RedisVectorStore backend. Various parameters are exposed to permit
    configuration of the Redis connection and behavior.

    Parameters are validated during initialization to ensure
    that they meet basic functional requirements.

    Attributes:
        host:
            Hostname of the Redis server.
        port:
            Port number of the Redis server.
        db:
            Database index to use in Redis.
        socket_timeout:
            Timeout for socket operations in seconds.
        socket_connect_timeout:
            Timeout for establishing a connection in seconds.
        max_connections:
            Maximum number of connections to Redis.
        health_check_interval:
            Interval for health checks in seconds.
        decode_responses:
            Whether to decode responses from Redis as strings. This
            should be set to True unless you are working with
            binary data.
        retry_on_timeout:
            Whether to retry on timeout errors.
    """

    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    max_connections: int = 50
    health_check_interval: int = 30
    decode_responses: bool = True
    retry_on_timeout: bool = True

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Redis configuration to a dictionary.

        This is useful for passing the configuration to a Redis
        client or for serialization.

        Returns:
            A dictionary representation of the Redis configuration.
        """
        return {
            'host':                     self.host,
            'port':                     self.port,
            'db':                       self.db,
            'socket_timeout':           self.socket_timeout,
            'socket_connect_timeout':   self.socket_connect_timeout,
            'max_connections':          self.max_connections,
            'health_check_interval':    self.health_check_interval,
            'decode_responses':         self.decode_responses,
            'retry_on_timeout':         self.retry_on_timeout,
        }

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_connection_settings()
        self._validate_timeouts()

    def _validate_connection_settings(self) -> None:
        """
        Validate Redis connection settings.

        Raises:
            ConfigurationError:
                If host is empty, port is not in the range 1-65535,
        """
        if not self.host or not self.host.strip():
            raise ConfigurationError('host cannot be empty')

        if not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ConfigurationError(f'{self.port=} must be between 1-65535')

        if not isinstance(self.db, int) or self.db < 0:
            raise ConfigurationError(f'{self.db=} must be non-negative')

    def _validate_timeouts(self) -> None:
        """
        Validate timeout settings.

        Raises:
            ConfigurationError:
                If socket_timeout, socket_connect_timeout, or
                health_check_interval are not positive integers.
        """
        if self.socket_timeout < 1:
            raise ConfigurationError(f'{self.socket_timeout=} must be '
                                     'positive')

        if self.socket_connect_timeout < 1:
            raise ConfigurationError(f'{self.socket_connect_timeout=} must be '
                                     'positive')

        if self.health_check_interval < 1:
            raise ConfigurationError(f'{self.health_check_interval=} must be '
                                     'positive')


@dataclass
class ManagerConfig:
    # pylint: disable=too-many-instance-attributes
    """
    Configuration for RAGManager.

    This class provides configuration options for the RAGManager,
    which manages the RAG process. It includes settings which modify
    document chunking behavior, query limits, and index management.

    Parameters are validated during initialization to ensure
    that they meet the requirements for the RAG system.

    Attributes:
        index_name:
            Name of the index in the vector store.
        chunk_size:
            Size of text chunks to create from documents.
        min_chunk_size:
            Minimum size of text chunks. If None, defaults to
            overlap // 2.
        overlap:
            Number of overlapping tokens between chunks.
        max_query_length:
            Maximum length of queries for the retriever.
        max_input_length:
            Maximum length of input text for the embed.
        default_base_id:
            Default base ID for documents in the index.
        paranoid:
            Whether to perform additional checks and sanitization
            of input data.
    """

    index_name: str = 'rag_index'
    chunk_size: int = 512
    min_chunk_size: int | None = None
    overlap: int = 64
    max_query_length: int = 8192
    max_input_length: int = (1024 * 1024) * 10
    default_base_id: str = 'doc'
    paranoid: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_chunk_settings()
        self._validate_limits()
        self._validate_names()

    def _validate_chunk_settings(self) -> None:
        """
        Validate chunking settings.

        Raises:
            ConfigurationError:
                If chunk_size is not positive, overlap is negative,
        """
        if self.chunk_size < 1:
            raise ConfigurationError(f'{self.chunk_size=} must be positive')

        if self.overlap < 0:
            raise ConfigurationError(f'{self.overlap=} must be non-negative')

        if self.overlap >= self.chunk_size:
            raise ConfigurationError(f'({self.overlap=}) must be less than '
                                     f'({self.chunk_size=})')

    def _validate_limits(self) -> None:
        """
        Validate query and input length limits.

        Raises:
            ConfigurationError:
                If max_query_length or max_input_length are not
                positive, or if max_query_length exceeds
                max_input_length.
        """
        if self.max_query_length < 1:
            raise ConfigurationError(f'{self.max_query_length=} must be '
                                     'positive')

        if self.max_input_length < 1:
            raise ConfigurationError(f'{self.max_input_length=} must be '
                                     'positive')

        if self.max_query_length > self.max_input_length:
            raise ConfigurationError(f'{self.max_query_length=} cannot exceed '
                                     f'{self.max_input_length=}')

    def _validate_names(self) -> None:
        """
        Validate index and base ID names.

        Raises:
            ConfigurationError:
                If index_name is empty, contains invalid characters,
                or if default_base_id is empty.
        """
        if not self.index_name or not self.index_name.strip():
            raise ConfigurationError('index_name cannot be empty')

        if not re.match(r'^[a-zA-Z0-9_-]+$', self.index_name):
            raise ConfigurationError('index_name contains invalid characters')

        if not self.default_base_id or not self.default_base_id.strip():
            raise ConfigurationError('default_base_id cannot be empty')
