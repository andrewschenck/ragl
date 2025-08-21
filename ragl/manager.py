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
    TextUnitChunker:
        Utility for chunking TextUnit instances

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
from typing import Any, ClassVar, Iterator

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
    'TextUnitChunker',
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


class TextUnitChunker:
    # pylint: disable=too-few-public-methods
    """
    Chunk a TextUnit into smaller pieces.

    This class handles the splitting of a TextUnit's text payload
    into smaller TextUnit instances based on specified chunk size
    and overlap.
    """

    def __init__(
            self,
            *,
            tokenizer: TokenizerProtocol,
            chunk_size: int,
            overlap: int,
            min_chunk_size: int | None = None,
            split: bool = True,
            batch_timestamp: int | None = None,
            text_index: int = 0,
    ):
        # pylint: disable=too-many-arguments
        """
        Initialize the TextUnitChunker.

        Args:
            tokenizer:
                Tokenizer for text splitting.
            chunk_size:
                Size of text chunks.
            overlap:
                Overlap between chunks.
            min_chunk_size:
                Minimum size of a chunk, if specified.
            split:
                Whether to split the text into chunks.
            batch_timestamp:
                Timestamp for the batch, used in text_id generation.
            text_index:
                Index of the text in the batch, used in text_id generation.
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.split = split
        self.timestamp = batch_timestamp or int(time.time_ns() // 1000)
        self.text_index = text_index

    def chunk_text_unit(self, unit: TextUnit) -> Iterator[TextUnit]:
        """
        Chunk a TextUnit into smaller TextUnits.

        If self.split is False, yields the original TextUnit with
        updated text_id and chunk_position set to 0.

        If self.split is True, splits the text of the given TextUnit
        into smaller chunks based on the specified chunk size and overlap.
        Each chunk is yielded as a new TextUnit instance with updated
        metadata.

        Args:
            unit:
                TextUnit to chunk.

        Yields:
            Chunked TextUnit instances.
        """
        if not self.split:
            # Create a copy with proper chunk_position and text_id
            text_id = (f'{TEXT_ID_PREFIX}{unit.parent_id}-'
                       f'{self.timestamp}-{self.text_index}-0')
            unit_data = unit.to_dict()
            unit_data.update({
                'text_id':        text_id,
                'chunk_position': 0,
                'distance':       0.0,
            })

            yield TextUnit.from_dict(unit_data)
            return

        chunks = self._split_text(
            tokenizer=self.tokenizer,
            text=unit.text,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            min_chunk_size=self.min_chunk_size,
        )
        base_data = unit.to_dict()
        for chunk_position, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            text_id = (f'{TEXT_ID_PREFIX}{unit.parent_id}-'
                       f'{self.timestamp}-{self.text_index}-{chunk_position}')
            chunk_data = base_data.copy()
            chunk_data.update({
                'text_id':        text_id,
                'text':           chunk,
                'chunk_position': chunk_position,
                'distance':       0.0,
            })

            yield TextUnit.from_dict(chunk_data)

    @staticmethod
    def _split_text(
            *,
            tokenizer: TokenizerProtocol,
            text: str,
            chunk_size: int,
            overlap: int,
            min_chunk_size: int | None = None,
    ) -> list[str]:
        """
        Split text into chunks based on chunk size and overlap.

        Args:
            tokenizer:
                Tokenizer for text splitting.
            text:
                Text to split.
            chunk_size:
                Size of text chunks.
            overlap:
                Overlap between chunks.
            min_chunk_size:
                Minimum size of a chunk, if specified.
        """
        min_chunk_size = min_chunk_size or overlap // 2
        tokens = tokenizer.encode(text)

        # If text is shorter than chunk size, return as single chunk
        if len(tokens) <= chunk_size:
            return [text]

        chunks = []
        step = chunk_size - overlap

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:min(i + chunk_size, len(tokens))]
            chunk_text = tokenizer.decode(chunk_tokens).strip()
            if chunk_text:
                chunks.append(chunk_text)

        # Merge the last chunk if it's too short
        if len(chunks) > 1:
            last_tokens = tokenizer.encode(chunks[-1])
            if len(last_tokens) < min_chunk_size:
                chunks[-2] += ' ' + chunks[-1]
                chunks.pop()

        return chunks


class RAGManager:
    # pylint: disable=too-many-instance-attributes
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

    DEFAULT_PARENT_ID: ClassVar[str] = 'doc'
    MAX_QUERY_LENGTH: ClassVar[int] = 8192
    MAX_INPUT_LENGTH: ClassVar[int] = (1024 * 1024) * 10

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
            _LOG.error(msg)
            raise TypeError(msg)
        if not isinstance(tokenizer, TokenizerProtocol):
            msg = 'tokenizer must implement TokenizerProtocol'
            _LOG.error(msg)
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
            text_or_unit: str | TextUnit,
            *,
            chunk_size: int | None = None,
            overlap: int | None = None,
            split: bool = True,
    ) -> list[TextUnit]:
        """
        Add text to the store.

        Splits text into chunks, stores with metadata, and
        returns stored TextUnit instances.

        Args:
            text_or_unit:
                Text or TextUnit to add.
            chunk_size:
                Optional chunk size override.
            overlap:
                Optional overlap override.
            split:
                Whether to split the text into chunks.

        Raises:
            ValidationError:
                If text is empty or params invalid.
            DataError:
                If no chunks are stored.

        Returns:
            List of stored TextUnit instances.
        """
        _LOG.debug('Adding text: %s', text_or_unit)

        with self.track_operation('add_text'):
            results = self.add_texts(
                texts_or_units=[text_or_unit],
                chunk_size=chunk_size,
                overlap=overlap,
                split=split,
            )

        return results

    def add_texts(
            self,
            texts_or_units: list[str | TextUnit],
            *,
            chunk_size: int | None = None,
            overlap: int | None = None,
            split: bool = True,
    ) -> list[TextUnit]:
        """
        Add multiple texts to the store.

        Splits texts into chunks, stores with metadata in batch, and
        returns stored TextUnit instances.

        Args:
            texts_or_units:
                List of texts or TextUnit objects to add.
            chunk_size:
                Optional chunk size override.
            overlap:
                Optional overlap override.
            split:
                Whether to split the text into chunks.

        Raises:
            ValidationError:
                If texts are empty or params invalid.
            DataError:
                If no chunks are stored.

        Returns:
            List of stored TextUnit instances.
        """
        with self.track_operation('add_texts'):
            _LOG.debug('Adding texts: %d items', len(texts_or_units))

            self._validate_texts_input(texts_or_units)

            _chunk_size, _overlap = self._resolve_chunk_params(
                chunk_size=chunk_size,
                overlap=overlap,
            )
            text_units_to_store = self._prepare_text_units_for_storage(
                texts_or_units=texts_or_units,
                chunk_size=_chunk_size,
                overlap=_overlap,
                split=split,
            )

            stored_units = self.ragstore.store_texts(text_units_to_store)
            _LOG.info('Added %d texts resulting in %d chunks',
                      len(texts_or_units), len(stored_units))

            return stored_units

    @staticmethod
    def _validate_texts_input(texts_or_units: list[str | TextUnit]) -> None:
        """
        Validate input for add_texts method.

        Args:
            texts_or_units:
                List of texts or TextUnit objects to validate.

        Raises:
            ValidationError:
                If texts_or_units is empty.
        """
        if not texts_or_units:
            _LOG.error('texts_or_units cannot be empty')
            raise ValidationError('texts_or_units cannot be empty')

    def _resolve_chunk_params(
            self,
            chunk_size: int | None,
            overlap: int | None
    ) -> tuple[int, int]:
        """
        Resolve effective chunk size and overlap.

        Determines the effective chunk size and overlap to use,
        prioritizing method parameters over instance defaults.

        Args:
            chunk_size:
                Optional chunk size override.
            overlap:
                Optional overlap override.

        Returns:
            Tuple of effective chunk size and overlap.
        """
        effective_chunk_size = chunk_size or self.chunk_size
        effective_overlap = overlap or self.overlap
        self._validate_chunking(effective_chunk_size, effective_overlap)

        return effective_chunk_size, effective_overlap

    def _prepare_text_units_for_storage(
            self,
            *,
            texts_or_units: list[str | TextUnit],
            chunk_size: int,
            overlap: int,
            split: bool,
    ) -> list[TextUnit]:
        """
        Prepare TextUnit instances for storage.

        Converts input texts or TextUnits to sanitized TextUnits,
        chunks them, and returns a list of TextUnits ready for storage.

        Args:
            texts_or_units:
                List of texts or TextUnit objects to add.
            chunk_size:
                Chunk size to use.
            overlap:
                Overlap to use.
            split:
                Whether to split the text into chunks.

        Returns:
            List of TextUnit instances ready for storage.
        """
        text_units_to_store: list[TextUnit] = []
        batch_timestamp = int(time.time_ns() // 1000)

        for text_index, item in enumerate(texts_or_units):
            unit = self._convert_to_text_unit(item)
            chunker = self._create_chunker(
                chunk_size=chunk_size,
                overlap=overlap,
                split=split,
                batch_timestamp=batch_timestamp,
                text_index=text_index,
            )
            text_units_to_store.extend(chunker.chunk_text_unit(unit))

        if not text_units_to_store:
            msg = 'No valid chunks stored'
            _LOG.error(msg)
            raise DataError(msg)

        return text_units_to_store

    def _convert_to_text_unit(self, item: str | TextUnit) -> TextUnit:
        """
        Convert input to a sanitized TextUnit.

        Args:
            item:
                Text or TextUnit to convert.

        Returns:
            Sanitized TextUnit instance.
        """
        if isinstance(item, str):
            return TextUnit(
                text=self._sanitize_text(item),
                parent_id=self.DEFAULT_PARENT_ID,
            )
        if isinstance(item, TextUnit):
            return TextUnit(
                text=self._sanitize_text(item.text),
                text_id=item.text_id,
                parent_id=item.parent_id or self.DEFAULT_PARENT_ID,
                chunk_position=item.chunk_position,
                distance=item.distance,
                source=item.source,
                confidence=item.confidence,
                language=item.language,
                section=item.section,
                author=item.author,
                tags=item.tags,
                timestamp=item.timestamp,
            )
        msg = 'Invalid text type, must be str or TextUnit'
        _LOG.error(msg)
        raise ValidationError(msg)

    def _create_chunker(
            self,
            *,
            chunk_size: int,
            overlap: int,
            split: bool,
            batch_timestamp: int,
            text_index: int,
    ) -> TextUnitChunker:
        # pylint: disable=too-many-arguments
        """
        Create a TextUnitChunker instance.

        Args:
            chunk_size:
                Chunk size to use.
            overlap:
                Overlap to use.
            split:
                Whether to split the text into chunks.
            batch_timestamp:
                Timestamp for the batch, used in text_id generation.
            text_index:
                Index of the text in the batch, used in text_id generation.

        Returns:
            TextUnitChunker instance.
        """
        return TextUnitChunker(
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=self.min_chunk_size,
            split=split,
            batch_timestamp=batch_timestamp,
            text_index=text_index,
        )

    # def add_texts(
    #         self,
    #         texts_or_units: list[str | TextUnit],
    #         *,
    #         chunk_size: int | None = None,
    #         overlap: int | None = None,
    #         split: bool = True,
    # ) -> list[TextUnit]:
    #     # pylint: disable=too-many-locals
    #     """
    #     Add multiple texts to the store.
    #
    #     Splits texts into chunks, stores with metadata in batch, and
    #     returns stored TextUnit instances.
    #
    #     Args:
    #         texts_or_units:
    #             List of texts or TextUnit objects to add.
    #         chunk_size:
    #             Optional chunk size override.
    #         overlap:
    #             Optional overlap override.
    #         split:
    #             Whether to split the text into chunks.
    #
    #     Raises:
    #         ValidationError:
    #             If texts are empty or params invalid.
    #         DataError:
    #             If no chunks are stored.
    #
    #     Returns:
    #         List of stored TextUnit instances.
    #     """
    #     with self.track_operation('add_texts'):
    #         _LOG.debug('Adding texts: %d items', len(texts_or_units))
    #
    #         if not texts_or_units:
    #             _LOG.error('texts_or_units cannot be empty')
    #             raise ValidationError('texts_or_units cannot be empty')
    #
    #         effective_chunk_size = chunk_size or self.chunk_size
    #         effective_overlap = overlap or self.overlap
    #         self._validate_chunking(effective_chunk_size, effective_overlap)
    #
    #         text_units_to_store: list[TextUnit] = []
    #         batch_timestamp = int(time.time_ns() // 1000)
    #
    #         for text_index, item in enumerate(texts_or_units):
    #
    #             if isinstance(item, str):
    #                 sanitized_text = self._sanitize_text(item)
    #                 unit = TextUnit(
    #                     text=sanitized_text,
    #                     parent_id=self.DEFAULT_PARENT_ID,
    #                 )
    #
    #             elif isinstance(item, TextUnit):
    #                 # Create a copy to avoid modifying the original
    #                 unit = TextUnit(
    #                     text=self._sanitize_text(item.text),
    #                     text_id=item.text_id,
    #                     parent_id=item.parent_id or self.DEFAULT_PARENT_ID,
    #                     chunk_position=item.chunk_position,
    #                     distance=item.distance,
    #                     source=item.source,
    #                     confidence=item.confidence,
    #                     language=item.language,
    #                     section=item.section,
    #                     author=item.author,
    #                     tags=item.tags,
    #                     timestamp=item.timestamp,
    #                 )
    #
    #             else:
    #                 msg = 'Invalid text type, must be str or TextUnit'
    #                 _LOG.error(msg)
    #                 raise ValidationError(msg)
    #
    #             chunker = TextUnitChunker(
    #                 tokenizer=self.tokenizer,
    #                 chunk_size=effective_chunk_size,
    #                 overlap=effective_overlap,
    #                 min_chunk_size=self.min_chunk_size,
    #                 split=split,
    #                 batch_timestamp=batch_timestamp,
    #                 text_index=text_index,
    #             )
    #             text_units_to_store.extend(chunker.chunk_text_unit(unit))
    #
    #         if not text_units_to_store:
    #             msg = 'No valid chunks stored'
    #             _LOG.error(msg)
    #             raise DataError(msg)
    #
    #         stored_units = self.ragstore.store_texts(text_units_to_store)
    #
    #         _LOG.info('Added %d texts resulting in %d chunks',
    #                   len(texts_or_units), len(stored_units))
    #         return stored_units

    def delete_text(self, text_id: str) -> bool | None:
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
                return None
            deleted = self.ragstore.delete_text(text_id)
            _LOG.info('Deleted text %s', text_id)
            return deleted

    def delete_texts(self, text_ids: list[str]) -> int:
        """
        Delete multiple texts from the store.

        Deletes a list of text chunks by their IDs, removing them
        and any associated metadata from the store.

        Args:
            text_ids:
                List of text IDs to delete.

        Returns:
            Number of texts deleted.
        """
        _LOG.debug('Deleting texts: %s', text_ids)
        with self.track_operation('delete_texts'):
            valid_ids = self._filter_valid_text_ids(text_ids)

            if not valid_ids:
                _LOG.warning('No valid text IDs found for deletion')
                return 0

            deleted_count = self.ragstore.delete_texts(valid_ids)
            _LOG.info('Deleted %d texts', deleted_count)
            return deleted_count

    def _filter_valid_text_ids(self, text_ids: list[str]) -> list[str]:
        """
        Filter text IDs to only include those that exist in the store.

        Args:
            text_ids: List of text IDs to filter.

        Returns:
            List of valid text IDs that exist in the store.
        """
        existing_texts = self.ragstore.list_texts()
        return [tid for tid in text_ids if tid in existing_texts]

    # def delete_texts(self, text_ids: list[str]) -> int:
    #     """
    #     Delete multiple texts from the store.
    #
    #     Deletes a list of text chunks by their IDs, removing them
    #     and any associated metadata from the store.
    #
    #     Args:
    #         text_ids:
    #             List of text IDs to delete.
    #
    #     Returns:
    #         Number of texts deleted.
    #     """
    #     _LOG.debug('Deleting texts: %s', text_ids)
    #     with self.track_operation('delete_texts'):
    #         existing_texts = self.ragstore.list_texts()
    #         valid_ids = [tid for tid in text_ids if tid in existing_texts]
    #         if not valid_ids:
    #             _LOG.warning('No valid text IDs found for deletion')
    #             return 0
    #         deleted_count = self.ragstore.delete_texts(valid_ids)
    #         _LOG.info('Deleted %d texts', deleted_count)
    #         return deleted_count

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
            if backend filtering reduces results.
        """
        _LOG.debug('Retrieving context for query: %s', query)

        if query.strip() == '':
            return []

        with self.track_operation('get_context'):
            self._validate_and_sanitize_query(query, top_k)

            results = self._retrieve_relevant_texts(
                query, top_k, min_time, max_time
            )

            sorted_results = self._sort_results(results, sort_by_time)

            _LOG.info('Retrieved %s contexts for query: %s',
                      len(sorted_results), query)

            return sorted_results

    def _validate_and_sanitize_query(self, query: str, top_k: int) -> None:
        """
        Validate and sanitize the query and top_k parameters.

        Args:
            query: Query string to validate and sanitize.
            top_k: Number of results to validate.
        """
        self._sanitize_text(query)
        self._validate_query(query)
        self._validate_top_k(top_k)

    def _retrieve_relevant_texts(
            self,
            query: str,
            top_k: int,
            min_time: int | None,
            max_time: int | None,
    ) -> list[TextUnit]:
        """
        Retrieve relevant texts from the RAG store.

        Args:
            query: Query text.
            top_k: Number of results to return.
            min_time: Minimum timestamp filter.
            max_time: Maximum timestamp filter.

        Returns:
            List of relevant TextUnit instances.
        """
        return self.ragstore.get_relevant(
            query=query,
            top_k=top_k,
            min_time=min_time,
            max_time=max_time,
        )

    @staticmethod
    def _sort_results(
            results: list[TextUnit],
            sort_by_time: bool,
    ) -> list[TextUnit]:
        """
        Sort results by time or distance.

        Args:
            results: List of TextUnit instances to sort.
            sort_by_time: Whether to sort by time instead of distance.

        Returns:
            Sorted list of TextUnit instances.
        """
        if sort_by_time:
            return sorted(results, key=lambda x: x.timestamp)
        return sorted(results, key=lambda x: x.distance)

    # def get_context(
    #         self,
    #         query: str,
    #         top_k: int = 10,
    #         *,
    #         min_time: int | None = None,
    #         max_time: int | None = None,
    #         sort_by_time: bool = False,
    # ) -> list[TextUnit]:
    #     # pylint: disable=too-many-arguments
    #     """
    #     Retrieve relevant text chunks for a query.
    #
    #     Retrieves text chunks based on semantic similarity
    #     to the query, optionally filtering by time range and sorting.
    #
    #     Args:
    #         query:
    #             Query text.
    #         top_k:
    #             Number of results to return.
    #         min_time:
    #             Minimum timestamp filter.
    #         max_time:
    #             Maximum timestamp filter.
    #         sort_by_time:
    #             Sort by time instead of distance.
    #
    #     Returns:
    #         List of TextUnit instances, possibly fewer than top_k
    #         if backend filtering reduces results. See relevant
    #         backend documentation for details.
    #     """
    #     _LOG.debug('Retrieving context for query: %s', query)
    #
    #     if query.strip() == '':
    #         return []
    #
    #     with self.track_operation('get_context'):
    #         self._sanitize_text(query)
    #         self._validate_query(query)
    #         self._validate_top_k(top_k)
    #
    #         results = self.ragstore.get_relevant(
    #             query=query,
    #             top_k=top_k,
    #             min_time=min_time,
    #             max_time=max_time,
    #         )
    #
    #         if sort_by_time:
    #             results = sorted(results, key=lambda x: x.timestamp)
    #         else:
    #             results = sorted(results, key=lambda x: x.distance)
    #
    #         _LOG.info('Retrieved %s contexts for query: %s',
    #                   len(results), query)
    #
    #         return results

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
    def track_operation(self, operation_name: str) -> Iterator[None]:
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

    def _sanitize_text(self, text: str) -> str:
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

    @staticmethod
    def _validate_chunking(chunk_size: int, overlap: int) -> None:
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
