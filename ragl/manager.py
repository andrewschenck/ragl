import logging
import re
import statistics
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from ragl.config import ManagerConfig
from ragl.constants import TEXT_ID_PREFIX
from ragl.exceptions import DataError, ValidationError
from ragl.protocols import RAGStoreProtocol, TokenizerProtocol
from ragl.textunit import TextUnit
from ragl.tokenizer import TiktokenTokenizer


__all__ = (
    'RAGTelemetry',
    'RAGManager',
)


_LOG = logging.getLogger(__name__)


@dataclass
class RAGTelemetry:
    """Telemetry for RAG operations."""

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

        Args:
            duration:
                Duration of the operation in seconds.
        """
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

        Args:
            duration:
                Duration of the operation in seconds.
        """
        self.total_calls += 1
        self.total_duration += duration
        self.avg_duration = self.total_duration / self.total_calls
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        self.recent_durations.append(duration)

    def compute_metrics(self) -> dict[str, Any]:
        """
        Calculate metrics and dump as a dictionary.

        Returns:
            A dictionary containing operational metrics.
        """
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

    Handles storage and retrieval of text chunks with a retriever
    and tokenizer. Splits text, stores with metadata, and retrieves
    relevant chunks for queries.

    Metadata includes optional fields like source, timestamp, tags,
    confidence, language, section, author, and parent_id.

    The parent_id groups chunks and is auto-generated if base_id
    is unset. For heavy deletion use cases relying on unique
    parent_id, always specify base_id to avoid collisions.

    Attributes:
        ragstore:
            RagstoreProtocol-conforming object for storage
            operations.
        tokenizer:
            TokenizerProtocol-conforming object for text splitting.
        chunk_size:
            Size of text chunks.
        overlap:
            Overlap between chunks.
    """

    DEFAULT_BASE_ID = 'doc'
    MAX_QUERY_LENGTH = 8192
    MAX_INPUT_LENGTH = (1024 * 1024) * 10

    ragstore: RAGStoreProtocol
    tokenizer: TokenizerProtocol
    chunk_size: int
    overlap: int
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

        Args:
            config:
                Configuration object with RAG parameters. # todo
            ragstore:
                Manages embedding for storage and retrieval.
            tokenizer:
                Tokenizer for text splitting.
        """
        if not isinstance(ragstore, RAGStoreProtocol):
            raise TypeError('retriever must implement RAGStoreProtocol')
        if not isinstance(tokenizer, TokenizerProtocol):
            raise TypeError('tokenizer must implement TokenizerProtocol')

        chunk_size, overlap = self._validate_params(config.chunk_size,
                                                    config.overlap)

        self.ragstore = ragstore
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
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
        with self.track_operation('add_text'):
            cs = chunk_size if chunk_size is not None else self.chunk_size
            ov = overlap if overlap is not None else self.overlap

            cs, ov = self._validate_params(cs, ov)

            if isinstance(text_or_doc, str):
                if not text_or_doc.strip():
                    raise ValidationError('text cannot be empty')
                text_or_doc = self._sanitize_text_input(text_or_doc)

            elif isinstance(text_or_doc, TextUnit):
                text_or_doc.text = self._sanitize_text_input(text_or_doc.text)

            if base_id:
                parent_id = base_id
            else:
                current_text_count = len(self.ragstore.list_texts())
                parent_id = f'{TEXT_ID_PREFIX}{current_text_count + 1}'

            # parent_id = base_id or default_id
            stored_docs: list[TextUnit] = []

            chunks = self._get_chunks(text_or_doc, cs, ov, split)
            base_data = self._prepare_base_data(text_or_doc, parent_id)

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

            if not stored_docs:
                raise DataError('no valid chunks stored')

            _doc_count = len(stored_docs)
            _doc_id = [doc.text_id for doc in stored_docs]
            _LOG.info('added %s text chunks with IDs: %s', _doc_count, _doc_id)

            return stored_docs

    def delete_text(self, text_id: str) -> None:
        """
        Delete a text from the store.

        Args:
            text_id: ID of text to delete.
        """
        with self.track_operation('delete_text'):
            self.ragstore.delete_text(text_id)

    def get_context(
            self,
            query: str,
            top_k: int = 1,
            *,
            min_time: int | None = None,
            max_time: int | None = None,
            sort_by_time: bool = False,
    ) -> list[TextUnit]:
        # pylint: disable=too-many-arguments
        """
        Get relevant text chunks for a query.

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
        with self.track_operation('get_context'):
            self._sanitize_text_input(query)
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

            _LOG.info('retrieved %s contexts for query: %s',
                      len(results), query)

            return [TextUnit.from_dict(result) for result in results]

    def get_health_status(self) -> dict[str, Any]:
        """
        Get health status of the storage backend.

        Returns:
            Health status dictionary.
        """
        with self.track_operation('health_check'):
            if hasattr(self.ragstore.storage, 'health_check'):
                return self.ragstore.storage.health_check()
            return {'status': 'health_check_not_supported'}

    def get_performance_metrics(
            self,
            operation_name: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Get performance metrics for operations.

        Args:
            operation_name:
                Specific operation to get metrics for, or None for
                all.

        Returns:
            Dictionary of operation metrics.
        """
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
        List all text IDs in the store.

        Returns:
            Sorted list of text IDs.
        """
        with self.track_operation('list_texts'):
            text_ids = self.ragstore.list_texts()
            _LOG.info('retrieved %s texts', len(text_ids))
            return text_ids

    def reset(self, *, reset_metrics: bool = True) -> None:
        """
        Reset the store to empty state.
        """
        with self.track_operation('reset'):
            self.ragstore.clear()

        if reset_metrics:
            self.reset_metrics()

        _LOG.info('store reset successfully')

    def reset_metrics(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()
        _LOG.info('metrics reset')

    @contextmanager
    def track_operation(
            self,
            operation_name: str,
    ) -> Iterator[None]:
        """
        A context manager to track operation performance metrics.

        Args:
            operation_name: Name of the operation being tracked.
        """
        start = time.time()
        _LOG.info('starting operation: %s', operation_name)  # todo permit other log level

        try:
            yield

        except Exception as e:  # pylint: disable=broad-except
            duration = time.time() - start
            record_failure = self._metrics[operation_name].record_failure
            record_failure(duration)
            _LOG.error('operation failed: %s (%.3fs) - %s', operation_name,
                       duration, e)
            raise

        duration = time.time() - start
        record_success = self._metrics[operation_name].record_success
        record_success(duration)
        _LOG.info('operation completed: %s (%.3fs)', operation_name, duration)

    @staticmethod
    def _format_context(
        chunks: list[TextUnit],
        separator: str = '\n\n',
    ) -> str:
        """
        Format text chunks into a string.

        Args:
            chunks:
                List of TextUnit instances.
            separator:
                Separator between chunks.

        Returns:
            Formatted context string.
        """
        return separator.join(str(chunk) for chunk in chunks)

    def _get_chunks(
            self,
            text_or_doc: str | TextUnit,
            cs: int,
            ov: int,
            split: bool,
    ) -> list[str]:
        """Get text chunks based on split option.

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
        if split:
            if isinstance(text_or_doc, TextUnit):
                return self._split_text(text_or_doc.text, cs, ov)
            return self._split_text(text_or_doc, cs, ov)

        if isinstance(text_or_doc, TextUnit):
            return [text_or_doc.text]
        return [text_or_doc]

    def _sanitize_text_input(self, text: str) -> str:
        """
        Sanitize text input to prevent injection attacks.

        Args:
            text:
                Text to sanitize.

        Raises:
            ValidationError:
                If text is too large.

        Returns:
            Sanitized text string.
        """
        limit = self.MAX_INPUT_LENGTH
        if len(text.encode('utf-8')) > limit:
            raise ValidationError('text too large')

        # Remove potentially dangerous characters
        if self.paranoid:
            text = re.sub(r'[^\w\s.,!?-]', '', text)

        return text

    def _split_text(
            self,
            text: str,
            chunk_size: int,
            overlap: int,
    ) -> list[str]:
        """
        Split text into chunks.

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
        tokens = self.tokenizer.encode(text)
        chunks = []
        step = chunk_size - overlap
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:min(i + chunk_size, len(tokens))]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)
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
        Prepare base metadata for storage.

        Args:
            text_or_doc:
                Text or TextUnit to process.
            parent_id:
                ID of parent document.

        Returns:
            Base metadata dict.
        """
        if isinstance(text_or_doc, TextUnit):
            return text_or_doc.as_dict()

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
    def _validate_params(
            chunk_size: int,
            overlap: int,
    ) -> tuple[int, int]:
        """
        Validate chunk size and overlap.

        Args:
            chunk_size:
                Size of text chunks.
            overlap:
                Overlap between chunks.

        Raises:
            ValidationError:
                If params are invalid.

        Returns:
            Tuple of validated chunk_size and overlap.
        """
        cs = chunk_size
        ov = overlap

        if cs <= 0:
            raise ValidationError('chunk_size must be positive')
        if ov < 0:
            raise ValidationError('overlap must be non-negative')
        if ov >= cs:
            raise ValidationError('overlap must be less than chunk_size')

        return cs, ov

    def _validate_query(self, query: str) -> None:
        """
        Validate query string.

        Args:
            query:
                Query string to validate.

        Raises:
            ValidationError:
                If query is invalid.
        """
        if not query or not query.strip():
            raise ValidationError('query cannot be empty')

        if len(query) > self.MAX_QUERY_LENGTH:
            raise ValidationError(f'query too long: {len(query)} > '
                                  f'{self.MAX_QUERY_LENGTH}')

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
        if not isinstance(top_k, int) or top_k < 1:
            raise ValidationError('top_k must be a positive integer')
