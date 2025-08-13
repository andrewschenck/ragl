"""
SentenceTransformer-based text embedding.

This module provides an interface to the SentenceTransformer library for
converting text into high-dimensional vector embeddings. It includes
intelligent caching to improve performance and automatic memory management
to prevent resource exhaustion.

Key Features:
    - Text-to-vector embedding using pre-trained transformer models.
    - LRU caching for repeated embeddings with configurable cache size.
    - Automatic cache cleanup based on system memory thresholds.
    - Memory usage monitoring and reporting.
    - Support for custom device placement (CPU/GPU).

Classes:
    SentenceTransformerEmbedder:
        Main embedding class with caching and memory management
        capabilities.
"""

import gc
import logging
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from typing import Any

import psutil
import numpy as np
from sentence_transformers import SentenceTransformer

from ragl.config import SentenceTransformerConfig


__all__ = ('SentenceTransformerEmbedder',)


_LOG = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """
    Embed text using the SentenceTransformer model.

    This class provides methods to embed text into high-dimensional
    vector representations using pre-trained transformer models.

    It includes intelligent caching to improve performance and
    automatic memory management to prevent resource exhaustion.

    It supports configurable cache size and memory thresholds,
    allowing for efficient embedding of large datasets without
    overwhelming system resources.

    It also provides methods to monitor cache statistics and
    memory usage, making it suitable for production environments
    where resource management is critical.

    Attributes:
        model:
            Model to use for embedding.
        _cache_size:
            Maximum size of the embedding cache.
        _memory_threshold:
            Memory usage threshold for automatic cache cleanup.
        _auto_cleanup:
            Whether to automatically clear the cache based on
            memory usage.
        _show_progress:
            Whether to show progress bars during embedding.

    Example:
        >>> config = SentenceTransformerConfig(
        ...     model_name_or_path='intfloat/e5-large-v2',
        ...     cache_maxsize=1000,
        ... )
        >>> embedder = SentenceTransformerEmbedder(config)
        >>> embedding = embedder.embed('Hello, world!')
        >>> print(embedding.shape)
    """

    model: SentenceTransformer
    _cache_size: int
    _memory_threshold: float
    _auto_cleanup: bool
    _show_progress: bool
    _embed_cached: Any

    @property
    def dimensions(self) -> int:
        """Retrieve the embedding dimension count."""
        dimensions = self.model.get_sentence_embedding_dimension()
        assert isinstance(dimensions, int)
        return dimensions

    def __init__(self, config: SentenceTransformerConfig) -> None:
        """
        Initialize the SentenceTransformerEmbedder.

        This constructor sets up the embedding model and configures
        caching and memory management based on the provided
        configuration.

        Args:
            config: Configuration object with model and cache settings
        """
        model_path = Path(config.model_name_or_path)
        self.model = SentenceTransformer(str(model_path), device=config.device)
        self._cache_size = config.cache_maxsize
        self._memory_threshold = config.memory_threshold
        self._auto_cleanup = config.auto_clear_cache
        self._show_progress = config.show_progress
        self._embed_cached = lru_cache(self._cache_size)(self._embed_impl)

    def cache_info(self):
        """
        Retrieve cache statistics.

        Returns:
            Cache statistics including hits, misses, max size,
            and current size.
        """
        return self._embed_cached.cache_info()

    def clear_cache(self) -> None:
        """Clear the embedding cache and force garbage collection."""
        self._embed_cached.cache_clear()
        gc.collect()

    def embed(self, text: str) -> np.ndarray:
        """
        Embed text into a high-dimensional vector representation.

        This method uses caching to improve performance for repeated
        embeddings. If the cache exceeds the configured size or if
        memory usage exceeds the threshold, it will automatically clear
        the cache to free up resources.

        It returns the embedding as a numpy array of float32 type.

        Args:
            text:
                Text to embed.

        Returns:
            Embedding as a numpy array.
        """
        _LOG.debug('Embedding text: %s', text)
        if self._auto_cleanup and self._should_clear_cache():
            _LOG.info('%s: Clearing embed cache due to memory threshold',
                      self.__class__.__name__)
            self.clear_cache()
        return self._embed_cached(text)

    def _embed_impl(self, text: str) -> np.ndarray:
        """
        Embed text into a high-dimensional vector representation.

        (Internal implementation for embedding without cache exposure.)

        Args:
            text:
                Text to embed.

        Returns:
            Embedding as a numpy array of float32 type.
        """
        array = self.model.encode(
            sentences=text,
            show_progress_bar=self._show_progress,
        )
        assert isinstance(array, np.ndarray)
        return array.astype(np.float32)

    def get_memory_usage(self) -> dict[str, Any]:
        """
        Retrieve detailed memory usage information.

        This method provides statistics about the embedding cache,
        including hits, misses, size, and estimated memory usage.
        It also retrieves system memory usage statistics, such as
        process memory and system memory percentage.

        Returns:
            Dictionary with cache and system memory statistics
        """
        _LOG.debug('Retrieving memory usage statistics')
        cache_info = self.cache_info()
        cache_total = cache_info.hits + cache_info.misses
        if cache_total > 0:
            cache_hit_rate = cache_info.hits / cache_total
        else:
            cache_hit_rate = 0.0

        cache_hit_rate = round(cache_hit_rate, 2)

        # Estimate cache memory usage (rough approximation).
        #
        # This is a rough estimate based on the number of cached
        # embeddings and their dimensions.
        #
        # We assume 4 bytes per vector because self._embed_impl()
        # explicitly returns float32 arrays.
        #
        # embedding count * dimensions per embedding * 4 bytes (float32)

        estimated_cache_size = (
            cache_info.currsize *     # Number of cached embeddings
            self.dimensions *         # Dimensions per embedding (float32)
            4 /                       # Bytes per float32
            (1024 * 1024)             # Convert to MB
        )
        estimated_cache_size = round(estimated_cache_size, 2)

        usage = {
            'cache_hits':                   cache_info.hits,
            'cache_misses':                 cache_info.misses,
            'cache_size':                   cache_info.currsize,
            'cache_maxsize':                cache_info.maxsize,
            'cache_hit_rate':               cache_hit_rate,
            'estimated_cache_memory_mb':    estimated_cache_size,
            'process_memory_mb':            None,
            'system_memory_percent':        None,
            'system_memory_threshold':      self._memory_threshold,
            'auto_cleanup_enabled':         self._auto_cleanup,
        }

        with suppress(Exception):
            process = psutil.Process()
            memory_info = process.memory_info()
            process_memory_mb = memory_info.rss / (1024 * 1024)
            system_memory_percent = psutil.virtual_memory().percent
            usage.update({
                'process_memory_mb':        process_memory_mb,
                'system_memory_percent':    system_memory_percent,
            })

        return usage

    def _should_clear_cache(self) -> bool:
        """
        Check if cache should be cleared based on memory usage.

        Checks system memory usage against the configured threshold
        to determine if automatic cache cleanup should be performed.


        Returns:
            True if cleanup is needed
        """
        _LOG.debug('Checking if cache should be cleared')
        with suppress(Exception):
            memory_percent = psutil.virtual_memory().percent / 100.0
            return memory_percent > self._memory_threshold
        return False

    def __repr__(self) -> str:
        """Return detailed string representation for debugging."""
        return (
            f'{self.__class__.__name__}('
            f'model="{self.model.model_name}", '
            f'dims={self.dimensions}, '
            f'cache_size={self._cache_size}, '
            f'device="{self.model.device}", '
            f'auto_cleanup={self._auto_cleanup})'
        )

    def __str__(self) -> str:
        """Return user-friendly string representation."""
        return (
            f'SentenceTransformerEmbedder('
            f'{self.dimensions}D, '
            f'cache_size={self._cache_size})'
        )
