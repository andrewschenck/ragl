"""
Core RAG management functionality for text storage and retrieval.

This module provides the primary interface for managing text chunks in
a retrieval-augmented generation system. It handles text splitting,
storage with metadata, and semantic retrieval operations.

Classes:
    RAGTelemetry:
        Performance monitoring and metrics collection
    RAGManager:
        Main class for managing RAG operations

Features:
    - Text chunking with configurable size and overlap
    - Metadata-rich storage (source, timestamp, tags, etc.)
    - Semantic similarity retrieval
    - Performance metrics and health monitoring
    - Configurable text sanitization and validation
    - Parent-child document relationships
"""

import logging
import statistics
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import bleach

from ragl.config import ManagerConfig
from ragl.constants import TEXT_ID_PREFIX
from ragl.exceptions import DataError, ValidationError
from ragl.protocols import RAGStoreProtocol, TokenizerProtocol
from ragl.textunit import TextUnit
from ragl.tokenizer import TiktokenTokenizer


__all__ = (
    'RAGManager',
    'RAGTelemetry',
)


_LOG = logging.getLogger(__name__)


@dataclass
class RAGTelemetry:
    """
    Telemetry for RAG operations.

    This class is used internally by RAGManager to record the
    performance of text chunking and retrieval operations.

    It maintains statistics such as total calls, average duration,
    minimum and maximum durations, and failure counts.

    It provides methods to record both successful and failed
    operations, updating the relevant metrics accordingly. It also
    includes a method to compute and return all metrics as a dictionary
    for easy access and logging.


    Attributes:
        total_calls:
            Total number of calls made to the operation.
        total_duration:
            Total duration of all calls in seconds.
        avg_duration:
            Average duration of calls in seconds.
        min_duration:
            Minimum duration of a single call in seconds.
        max_duration:
            Maximum duration of a single call in seconds.
        failure_count:
            Number of failed calls.
        recent_durations:
            A deque to store the most recent durations for
            calculating average and median durations.
    """

    total_calls: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    failure_count: int = 0
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=100))

    def record_failure(self, duration: float) -> None:
        """
        Record a failed operation.

        Updates the telemetry with the duration of a failed
        operation, incrementing the failure count and updating
        the total duration and other metrics.

        Records the duration in the recent durations deque for
        calculating recent average and median durations.

        Args:
            duration:
                Duration of the operation in seconds.
        """
        _LOG.debug('Recording failed operation')
        self.total_calls += 1
        self.failure_count += 1
        self.total_duration += duration
        self.avg_duration = self.total_duration / self.total_calls
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        self.recent_durations.append(duration)

    def record_success(self, duration: float) -> None:
        """
        Record a successful operation.

        Updates the telemetry with the duration of a successful
        operation, incrementing the total calls and updating the
        total duration, average, minimum, and maximum durations.

        Records the duration in the recent durations deque for
        calculating recent average and median durations.

        Args:
            duration:
                Duration of the operation in seconds.
        """
        _LOG.debug('Recording successful operation')
        self.total_calls += 1
        self.total_duration += duration
        self.avg_duration = self.total_duration / self.total_calls
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        self.recent_durations.append(duration)

    def compute_metrics(self) -> dict[str, Any]:
        """
        Calculate and return metrics as a dictionary.

        Computes the operational metrics including total calls,
        failure count, success rate, minimum, maximum, and average
        durations, as well as recent average and median durations.

        Aggregates the recorded data and formats it into
        a dictionary for easy access and logging.

        If no calls have been made, it returns default values.

        If no durations have been recorded, it returns zero for
        minimum and average durations.

        Returns:
            A dictionary containing operational metrics.
        """
        _LOG.debug('Computing metrics')
        # Total / Failed / Successful Calls
        total_calls = self.total_calls
        failure_count = self.failure_count
        success_rate = (
            (self.total_calls - self.failure_count) / self.total_calls
            if self.total_calls > 0 else 0.0
        )
        success_rate = round(success_rate, 4)

        # Min / Max / Avg Durations
        min_duration = (
            round(self.min_duration, 4)
            if self.min_duration != float('inf') else 0.0
        )
        max_duration = round(self.max_duration, 4)
        avg_duration = round(self.avg_duration, 4)

        # Recent Avg / Med Durations
        recent = list(self.recent_durations)
        recent_avg = round(statistics.mean(recent), 4) if recent else 0.0
        recent_med = round(statistics.median(recent), 4) if recent else 0.0

        return {
            'total_calls':      total_calls,
            'failure_count':    failure_count,
            'success_rate':     success_rate,
            'min_duration':     min_duration,
            'max_duration':     max_duration,
            'avg_duration':     avg_duration,
            'recent_avg':       recent_avg,
            'recent_med':       recent_med,
        }


class RAGManager:
    """
    Manage text chunks for retrieval-augmented generation.

    RAGManager user the user-facing orchestrator which
    handles vector-based storage and retrieval of text chunks.
    It provides an interface to basic operations like adding
    text, deleting text, and retrieving context based on queries
    and interfaces with a RAGStoreProtocol-compliant backend.

    RAGManager supports both string text and TextUnit objects,
    automatically generating unique identifiers and maintaining
    relationships between chunks and their parent documents.

    Metadata includes optional fields like source, timestamp, tags,
    confidence, language, section, author, and parent_id.

    The parent_id groups chunks and is auto-generated if base_id
    is unset. For heavy deletion use cases relying on unique
    parent_id, always specify base_id to avoid collisions.

    RAGManager requires a class which implements RAGStoreProtocol
    for storage and retrieval operations, and a tokenizer
    implementing TokenizerProtocol for text splitting.

    Example:
        >>> from ragl.config import ManagerConfig
        >>>
        >>> config = ManagerConfig(chunk_size=512, overlap=50)
        >>> manager = RAGManager(config, ragstore)
        >>> chunks = manager.add_text('Your text here')
        >>> results = manager.get_context('query text', top_k=5)

    Attributes:
        ragstore:
            RagstoreProtocol-conforming object for store
            operations.
        tokenizer:
            TokenizerProtocol-conforming object for text splitting.
        chunk_size:
            Size of text chunks.
        overlap:
            Overlap between chunks.
        min_chunk_size:
            Minimum size of a chunk, if specified.
        paranoid:
            Take extra measures when sanitizing text input, aimed
            at preventing injection attacks.
        _metrics:
            Dictionary of operation names to RAGTelemetry instances
            for performance tracking.
    """

    DEFAULT_BASE_ID = 'doc'
    MAX_QUERY_LENGTH = 8192
    MAX_INPUT_LENGTH = (1024 * 1024) * 10

    ragstore: RAGStoreProtocol
    tokenizer: TokenizerProtocol
    chunk_size: int
    overlap: int
    min_chunk_size: int | None
    paranoid: bool
    _metrics: dict[str, RAGTelemetry]

    def __init__(
            self,
            config: ManagerConfig,
            ragstore: RAGStoreProtocol,
            *,
            tokenizer: TokenizerProtocol = TiktokenTokenizer(),
    ):
        """
        Initialize RAG store with configuration.

        Initializes the RAGManager with a configuration object,
        a RAGStoreProtocol-compliant store for text storage and
        retrieval, and a TokenizerProtocol-compliant tokenizer for
        text splitting.

        Args:
            config:
                Configuration object with RAG parameters.
            ragstore:
                Manages embedding for store and retrieval.
            tokenizer:
                Tokenizer for text splitting.
        """
        if not isinstance(ragstore, RAGStoreProtocol):
            msg = 'ragstore must implement RAGStoreProtocol'
            _LOG.critical(msg)
            raise TypeError(msg)
        if not isinstance(tokenizer, TokenizerProtocol):
            msg = 'tokenizer must implement TokenizerProtocol'
            _LOG.critical(msg)
            raise TypeError(msg)

        self._validate_chunking(config.chunk_size, config.overlap)

        self.ragstore = ragstore
        self.tokenizer = tokenizer
        self.chunk_size = config.chunk_size
        self.overlap = config.overlap
        self.min_chunk_size = config.min_chunk_size
        self.paranoid = config.paranoid
        self._metrics = defaultdict(RAGTelemetry)

    def add_text(
            self,
            text_or_doc: str | TextUnit,
            *,
            base_id: str | None = None,
            chunk_size: int | None = None,
            overlap: int | None = None,
            split: bool = True,
    ) -> list[TextUnit]:
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        """
        Add text to the store.

        Splits text into chunks, stores with metadata, and
        returns stored TextUnit instances.

        Args:
            text_or_doc:
                Text or TextUnit to add.
            base_id:
                Optional base ID for chunks, sets parent_id.
                If unset, parent_id is auto-generated and may
                collide after deletes; specify for uniqueness
                (e.g., UUID) if critical for grouping.
            chunk_size:
                Optional chunk size override.
            overlap:
                Optional overlap override.
            split:
                Whether to split the text.

        Raises:
            ValidationError:
                If text is empty or params invalid.
            DataError:
                If no chunks are stored.

        Returns:
            List of stored TextUnit instances.
        """
        _LOG.debug('Adding text: %s', text_or_doc)
        with self.track_operation('add_text'):
            cs = chunk_size if chunk_size is not None else self.chunk_size
            ov = overlap if overlap is not None else self.overlap
            self._validate_chunking(cs, ov)

            if isinstance(text_or_doc, str):
                if not text_or_doc.strip():
                    msg = 'Text cannot be empty'
                    _LOG.critical(msg)
                    raise ValidationError(msg)
                text_or_doc = self._sanitize_query(text_or_doc)

            elif isinstance(text_or_doc, TextUnit):
                text_or_doc.text = self._sanitize_query(text_or_doc.text)

            if base_id:
                parent_id = base_id
            else:
                current_text_count = len(self.ragstore.list_texts())
                parent_id = f'{TEXT_ID_PREFIX}{current_text_count + 1}'
            _LOG.debug('Using parent_id: %s', parent_id)

            chunks = self._get_chunks(text_or_doc, cs, ov, split)
            base_data = self._prepare_base_data(text_or_doc, parent_id)

            stored_docs: list[TextUnit] = []
            for i, chunk in enumerate(chunks):
                _base = base_id or self.DEFAULT_BASE_ID
                text_id = f'{TEXT_ID_PREFIX}{_base}-{i}'

                stored_doc = self._store_chunk(
                    chunk=chunk,
                    base_data=base_data,
                    text_id=text_id,
                    i=i,
                    parent_id=parent_id,
                )
                stored_docs.append(stored_doc)
                _LOG.info('Stored text chunk: %s', stored_doc.text_id)

            if not stored_docs:
                msg = 'No valid chunks stored'
                _LOG.critical(msg)
                raise DataError(msg)

            _doc_count = len(stored_docs)
            _doc_id = [doc.text_id for doc in stored_docs]
            _LOG.debug('Added %s text chunks with IDs: %s',
                       _doc_count, _doc_id)

            return stored_docs

    def delete_text(self, text_id: str) -> None:
        """
        Delete a text from the store.

        Deletes a text chunk by its ID, removing it and any
        associated metadata from the store.

        Args:
            text_id:
                ID of text to delete.
        """
        _LOG.debug('Deleting text %s', text_id)
        with self.track_operation('delete_text'):
            existing_texts = self.ragstore.list_texts()
            if text_id not in existing_texts:
                _LOG.warning('Text ID %s not found, skipping deletion',
                             text_id)
                return
            self.ragstore.delete_text(text_id)

    def get_context(
            self,
            query: str,
            top_k: int = 10,
            *,
            min_time: int | None = None,
            max_time: int | None = None,
            sort_by_time: bool = False,
    ) -> list[TextUnit]:
        # pylint: disable=too-many-arguments
        """
        Retrieve relevant text chunks for a query.

        Retrieves text chunks based on semantic similarity
        to the query, optionally filtering by time range and sorting.

        Args:
            query:
                Query text.
            top_k:
                Number of results to return.
            min_time:
                Minimum timestamp filter.
            max_time:
                Maximum timestamp filter.
            sort_by_time:
                Sort by time instead of distance.

        Returns:
            List of TextUnit instances, possibly fewer than top_k
            if backend filtering reduces results. See relevant
            backend documentation for details.
        """
        _LOG.debug('Retrieving context for query: %s', query)

        if query.strip() == '':
            return []

        with self.track_operation('get_context'):
            self._sanitize_query(query)
            self._validate_query(query)
            self._validate_top_k(top_k)

            results = self.ragstore.get_relevant(
                query=query,
                top_k=top_k,
                min_time=min_time,
                max_time=max_time,
            )

            if sort_by_time:
                results = sorted(results, key=lambda x: x['timestamp'])
            else:
                results = sorted(results, key=lambda x: x['distance'])

            _LOG.info('Retrieved %s contexts for query: %s',
                      len(results), query)

            return [TextUnit.from_dict(result) for result in results]

    def get_health_status(self) -> dict[str, Any]:
        """
        Return the health status of the backend, if available.

        Determines whether the storage backend supports health checks
        and returns the health check response.

        If not supported, returns a default message indicating
        health checks are not available.

        Returns:
            Health status dictionary.
        """
        _LOG.debug('Retrieving health status')
        with self.track_operation('health_check'):
            if hasattr(self.ragstore.storage, 'health_check'):
                return self.ragstore.storage.health_check()
            return {'status': 'health_check_not_supported'}

    def get_performance_metrics(
            self,
            operation_name: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Return performance metrics for methods which are tracked.

        Retrieves performance metrics for specific operations or
        all operations if no specific name is provided.

        Args:
            operation_name:
                Specific operation to get metrics for, or None for
                all.

        Returns:
            Dictionary of operation metrics.
        """
        _LOG.debug('Retrieving performance metrics')
        if operation_name:
            if operation_name not in self._metrics:
                return {}
            return {
                operation_name: self._metrics[operation_name].compute_metrics()
            }

        return {
            name: metrics.compute_metrics()
            for name, metrics in self._metrics.items()
        }

    def list_texts(self) -> list[str]:
        """
        Return a list of all text IDs in the store.

        Retrieves all text IDs stored in the backend. This is useful
        for tracking stored texts and managing deletions.

        Returns:
            Sorted list of text IDs.
        """
        _LOG.debug('Listing texts')
        with self.track_operation('list_texts'):
            text_ids = self.ragstore.list_texts()
            _LOG.debug('text count: %d', len(text_ids))
            return text_ids

    def reset(self, *, reset_metrics: bool = True) -> None:
        """
        Reset the store to empty state.

        Clears all stored texts and metadata, optionally resetting
        performance metrics. Does not ensure the underlying storage
        layer is 100% empty, as some backends may retain schema or
        configuration metadata.

        Use with caution, as this will remove all stored data.

        Args:
            reset_metrics:
                Whether to reset performance metrics as well.
        """
        _LOG.debug('Resetting store')

        if reset_metrics:
            self.reset_metrics()
            self.ragstore.clear()
        else:
            with self.track_operation('reset'):
                self.ragstore.clear()

        _LOG.info('Store reset')

    def reset_metrics(self) -> None:
        """
        Clear all collected metrics.

        Resets the performance metrics for all tracked operations.
        This is useful for starting fresh without historical data.
        """
        _LOG.debug('Resetting metrics')
        self._metrics.clear()
        _LOG.info('Metrics reset')

    @contextmanager
    def track_operation(
            self,
            operation_name: str,
    ) -> Iterator[None]:
        """
        Return a context manager which tracks RAG performance metrics.

        Uses the RAGTelemetry class to track the performance of RAG
        operations within a context. It allows for easy tracking of
        operation duration and success/failure rates.

        Args:
            operation_name:
                Name of the operation being tracked.
        """
        start = time.time()
        _LOG.debug('Starting operation: %s', operation_name)

        try:
            yield
            duration = time.time() - start
            record_success = self._metrics[operation_name].record_success
            record_success(duration)
            _LOG.debug('Operation completed: %s (%.3fs)',
                       operation_name, duration)

        except Exception as e:  # pylint: disable=broad-except
            duration = time.time() - start
            record_failure = self._metrics[operation_name].record_failure
            record_failure(duration)
            _LOG.critical('Operation failed: %s (%.3fs) - %s', operation_name,
                          duration, e)
            raise

    @staticmethod
    def _format_context(
            chunks: list[TextUnit],
            separator: str = '\n\n',
    ) -> str:
        """
        Format text chunks into a string.

        Formats a list of TextUnit instances into a single string
        with a specified separator between chunks. This is useful
        for preparing context for queries or responses.

        Args:
            chunks:
                List of TextUnit instances.
            separator:
                Separator between chunks.

        Returns:
            Formatted context string.
        """
        _LOG.debug('Formatting chunks')
        return separator.join(str(chunk) for chunk in chunks)

    def _get_chunks(
            self,
            text_or_doc: str | TextUnit,
            cs: int,
            ov: int,
            split: bool,
    ) -> list[str]:
        """
        Get text chunks based on split option.

        Args:
            text_or_doc:
                Text or TextUnit to chunk.
            cs:
                Chunk size.
            ov:
                Overlap size.
            split:
                Whether to split the text.

        Returns:
            List of text chunks.
        """
        _LOG.debug('Getting chunks')
        if split:
            if isinstance(text_or_doc, TextUnit):
                return self._split_text(text_or_doc.text, cs, ov)
            return self._split_text(text_or_doc, cs, ov)

        if isinstance(text_or_doc, TextUnit):
            return [text_or_doc.text]
        return [text_or_doc]

    def _sanitize_query(self, text: str) -> str:
        """
        Validate and sanitize text input to prevent injection attacks.

        Validate the input text by ensuring it does not exceed the
        maximum length and sanitize it by removing dangerous characters.

        Args:
            text:
                Text to sanitize.

        Raises:
            ValidationError:
                If text is too large.

        Returns:
            Sanitized text string.
        """
        _LOG.debug('Sanitizing text')
        limit = self.MAX_INPUT_LENGTH
        if len(text.encode('utf-8')) > limit:
            msg = 'text too long'
            _LOG.critical(msg)
            raise ValidationError(msg)

        if self.paranoid:
            text = bleach.clean(text=text, strip=True)

        return text

    def _split_text(
            self,
            text: str,
            chunk_size: int,
            overlap: int,
    ) -> list[str]:
        """
        Split text into chunks.

        Splits the input text into smaller chunks of specified size
        with a defined overlap. This is useful for processing large
        texts in manageable pieces for storage and retrieval.

        Args:
            text:
                Text to split.
            chunk_size:
                Size of each chunk.
            overlap:
                Overlap between chunks.

        Returns:
            List of text chunks.
        """
        _LOG.debug('Splitting text')

        # tokens = self.tokenizer.encode(text)
        # chunks = []
        # step = chunk_size - overlap
        # for i in range(0, len(tokens), step):
        #     chunk_tokens = tokens[i:min(i + chunk_size, len(tokens))]
        #     chunk_text = self.tokenizer.decode(chunk_tokens)
        #
        #     if chunk_text.strip():
        #         chunks.append(chunk_text)
        #
        # return chunks

        min_chunk_size = (
            self.min_chunk_size
            if self.min_chunk_size is not None
            else overlap // 2
        )
        tokens = self.tokenizer.encode(text)
        chunks = []
        step = chunk_size - overlap

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:min(i + chunk_size, len(tokens))]
            chunk_text = self.tokenizer.decode(chunk_tokens).strip()
            if chunk_text:
                chunks.append(chunk_text)

        # Merge the last chunk if it's too short
        if len(chunks) > 1:
            last_tokens = self.tokenizer.encode(chunks[-1])
            if len(last_tokens) < min_chunk_size:
                _LOG.debug('Merging last chunk due to short length')
                chunks[-2] += ' ' + chunks[-1]
                chunks.pop()

        return chunks

    def _store_chunk(
            self,
            *,
            chunk: str,
            base_data: dict[str, Any],
            text_id: str,
            i: int,
            parent_id: str,
    ) -> TextUnit:
        # pylint: disable=too-many-arguments
        """
        Store a single text chunk.

        Stores a text chunk with metadata in the RAG store.

        Args:
            chunk:
                Text chunk to store.
            base_data:
                Base metadata dict.
            text_id:
                ID for the chunk.
            i:
                Position of the chunk.
            parent_id:
                ID of parent document.

        Returns:
            Stored TextUnit instance.
        """
        _LOG.debug('Storing chunk')
        chunk_data = base_data.copy()
        chunk_data.update({
            'text_id':          text_id,
            'text':             chunk,
            'chunk_position':   i,
            'parent_id':        parent_id,
            'distance':         0.0,
        })

        metadata = {k: v for k, v in chunk_data.items()
                    if k not in {'text_id', 'text', 'distance'}}

        text_id = self.ragstore.store_text(
            text=chunk,
            text_id=text_id,
            metadata=metadata
        )
        chunk_data['text_id'] = text_id

        return TextUnit.from_dict(chunk_data)

    @staticmethod
    def _prepare_base_data(
            text_or_doc: str | TextUnit,
            parent_id: str,
    ) -> dict[str, Any]:
        """
        Prepare base metadata for store.

        Creates a base metadata dictionary for a text or TextUnit,
        including source, timestamp, tags, and other fields.


        Args:
            text_or_doc:
                Text or TextUnit to process.
            parent_id:
                ID of parent document.

        Returns:
            Base metadata dict.
        """
        _LOG.debug('Preparing base metadata')
        if isinstance(text_or_doc, TextUnit):
            return text_or_doc.to_dict()

        return {
            'source':           'unknown',
            'timestamp':        int(time.time()),
            'tags':             [],
            'confidence':       None,
            'language':         'unknown',
            'section':          'unknown',
            'author':           'unknown',
            'parent_id':        parent_id,
        }

    @staticmethod
    def _validate_chunking(
            chunk_size: int,
            overlap: int,
    ) -> None:
        """
        Validate chunk size and overlap.

        Validates the chunk size and overlap parameters to ensure
        they're logically consistent and within acceptable limits.

        Args:
            chunk_size:
                Size of text chunks.
            overlap:
                Overlap between chunks.

        Raises:
            ValidationError:
                If params are invalid.
        """
        _LOG.debug('Validating chunking parameters')
        cs = chunk_size
        ov = overlap

        if cs <= 0:
            msg = 'Chunk_size must be positive'
            _LOG.critical(msg)
            raise ValidationError(msg)
        if ov < 0:
            msg = 'Overlap must be non-negative'
            _LOG.critical(msg)
            raise ValidationError(msg)
        if ov >= cs:
            msg = 'Overlap must be less than chunk_size'
            _LOG.critical(msg)
            raise ValidationError(msg)

    def _validate_query(self, query: str) -> None:
        """
        Validate the query string.

        Validates the query string to ensure it is not empty and does
        not exceed the maximum allowed length. This is important to
        prevent unnecessary load on the system and ensure meaningful
        queries.

        Args:
            query:
                Query string to validate.

        Raises:
            ValidationError:
                If query is invalid.
        """
        _LOG.debug('Validating query')
        if not query or not query.strip():
            msg = 'Query cannot be empty'
            _LOG.critical(msg)
            raise ValidationError(msg)

        if len(query) > self.MAX_QUERY_LENGTH:
            msg = f'Query too long: {len(query)} > {self.MAX_QUERY_LENGTH}'
            _LOG.critical(msg)
            raise ValidationError(msg)

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        """
        Validate top_k parameter.

        Args:
            top_k:
                Number of results to return.

        Raises:
            ValidationError:
                If top_k is invalid.
        """
        _LOG.debug('Validating top_k parameter')
        if not isinstance(top_k, int) or top_k < 1:
            msg = 'top_k must be a positive integer'
            _LOG.critical(msg)
            raise ValidationError(msg)

    def __str__(self) -> str:
        """Human-readable summary showing current state."""
        text_count = len(self.ragstore.list_texts())
        return (
            f'RAGManager(texts={text_count}, '
            f'chunk_size={self.chunk_size}, '
            f'overlap={self.overlap})'
        )

    def __repr__(self) -> str:
        """Developer representation showing object construction."""
        return (
            f'RAGManager('
            f'config=ManagerConfig('
            f'chunk_size={self.chunk_size}, '
            f'overlap={self.overlap}, '
            f'min_chunk_size={self.min_chunk_size}, '
            f'paranoid={self.paranoid}), '
            f'ragstore={self.ragstore!r}, '
            f'tokenizer={self.tokenizer!r})'
        )
