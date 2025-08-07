import logging
import time
from contextlib import contextmanager
from typing import (
    Any,
    Iterator,
    Mapping,
    cast,
    overload,
)

import numpy as np
import redis  # type: ignore[import-untyped]
import redisvl.exceptions  # type: ignore[import-untyped]
from redisvl.index import SearchIndex  # type: ignore[import-untyped]
from redisvl.query import VectorQuery  # type: ignore[import-untyped]
from redisvl.schema import IndexSchema  # type: ignore[import-untyped]

from ragl.config import RedisConfig
from ragl.constants import MAX_TEXT_ID_LENGTH, TEXT_ID_PREFIX
from ragl.exceptions import (
    ConfigurationError,
    DataError,
    QueryError,
    RedisCapacityError,
    StorageConnectionError,
    ValidationError,
)
from ragl.schema import SchemaField, sanitize_metadata


_LOG = logging.getLogger(__name__)


__all__ = ('RedisStorage',)


class RedisStorage:
    """
    Store and retrieve vectors in Redis.

    Attributes:
        index:
            Redis SearchIndex instance.
        redis_client:
            Redis client instance.
        dimensions:
            Embedding dimension size.
        metadata_schema:
            Schema for metadata sanitization.

    Note:
        Metadata fields like `tags` are stored as strings in Redis
        (e.g., comma-separated for `tags`), but are returned as their
        expected types (e.g., list for `tags`) in method results,
        such as `get_relevant`.
    """
    SCHEMA_VERSION = 1

    MAX_FIELD_SIZE = (1024 * 1024) * 32
    MAX_METADATA_SIZE = (1024 * 1024) * 64
    MAX_TEXT_SIZE = (1024 * 1024) * 512

    POOL_DEFAULTS = {
        'socket_timeout':           5,
        'socket_connect_timeout':   5,
        'retry_on_timeout':         True,
        'max_connections':          50,
        'health_check_interval':    30,
    }

    TAG_SEPARATOR = ','
    TEXT_COUNTER_KEY = 'text_counter'

    index: SearchIndex
    index_name: str
    redis_client: redis.Redis
    dimensions: int
    metadata_schema: dict[str, SchemaField]

    @overload
    def __init__(
            self,
            *,
            redis_client: redis.Redis,
            dimensions: int | None,
            index_name: str,
    ) -> None:
        ...  # pragma: no cover

    @overload
    def __init__(
            self,
            *,
            redis_config: RedisConfig,
            dimensions: int | None,
            index_name: str,
    ) -> None:
        ...  # pragma: no cover

    def __init__(
            self,
            *,
            redis_client: redis.Redis | None = None,
            redis_config: RedisConfig | None = None,
            dimensions: int | None = None,
            index_name: str,
    ):
        """
        Initialize Redis storage.

        Args:
            redis_client:
                Optional Redis client instance. If provided,
                config is ignored.
            redis_config:
                Optional Redis configuration object. If provided,
                redis_client is ignored.
            dimensions:
                Size of embedding vectors, required for schema
                creation.
            index_name:
                Name of the Redis search index.

        Raises:
            StorageConnectionError:
                If Redis connection or index creation fails.
            ConfigurationError:
                If both redis_client and redis_config are provided,
                or if schema version mismatch occurs.
        """
        if redis_client is None and redis_config is None:
            raise ConfigurationError('either redis_client or redis_config '
                                     'must be provided')

        if redis_client is not None and redis_config is not None:
            raise ConfigurationError('both redis_client and redis_config '
                                     'were provided; only one is permitted')

        if dimensions is None:
            raise ConfigurationError('dimensions required for schema creation')

        if redis_client is not None:
            self.redis_client = redis_client
            try:
                self.redis_client.ping()
                _LOG.info('successfully connected to injected Redis client')
            except redis.ConnectionError as e:
                msg = f'injected Redis client connection failed: {e}'
                _LOG.error(msg)
                raise StorageConnectionError(msg) from e
        else:
            assert redis_config is not None

            pool_config = {**self.POOL_DEFAULTS, **redis_config.to_dict()}
            try:
                pool = redis.BlockingConnectionPool(**pool_config)
                self.redis_client = redis.Redis(connection_pool=pool)
                self.redis_client.ping()
                _LOG.info('successfully connected to Redis')

            except redis.ConnectionError as e:
                msg = f'failed to connect to Redis: {e}'
                _LOG.error(msg)
                raise StorageConnectionError(msg) from e

        self.dimensions = dimensions
        self.index_name = index_name

        self.metadata_schema = {
            'chunk_position': {
                'type':         int,
                'default':      0,
            },
            'timestamp': {
                'type':         int,
                'default':      0,
            },
            'confidence': {
                'type':         float,
                'default':      0.0,
            },
            'tags': {
                'type':         str,
                'default':      '',
            },
            'parent_id': {
                'type':         str,
                'default':      '',
            },
            'source': {
                'type':         str,
                'default':      '',
            },
            'language': {
                'type':         str,
                'default':      '',
            },
            'section': {
                'type':         str,
                'default':      '',
            },
            'author': {
                'type':         str,
                'default':      '',
            },
        }
        self._enforce_schema_version()
        schema = self._create_redis_schema(index_name)
        self.index = SearchIndex(schema, self.redis_client)

        try:
            self.index.create()
            _LOG.info('Connected to index: %s', index_name)

        except redis.ConnectionError as e:
            msg = f'Failed to connect to Redis: {e}'
            _LOG.error(msg)
            raise StorageConnectionError(msg) from e

    def clear(self) -> None:
        """
        Clear all data from the Redis index.

        This method deletes the existing index and creates a new one,
        resetting the schema version to the current version.
        """
        with self.redis_context() as client:
            self.index.delete(drop=True)
            self.index.create()
            _LOG.info('index cleared successfully')
            version_key = f"schema_version:{self.index_name}"
            client.set(version_key, self.SCHEMA_VERSION)
            _LOG.info('reset schema version to %s for index %s',
                      self.SCHEMA_VERSION, self.index_name)

    def delete_text(self, text_id: str) -> bool:
        """
        Delete a text from Redis.

        Args:
            text_id:
                ID of text to delete.
        """
        self._validate_text_id(text_id)
        with self.redis_context() as client:
            deleted = client.delete(text_id)
            deleted = True

        if deleted == 0:
            deleted = False

        return deleted

    def get_relevant(
            self,
            embedding: np.ndarray,
            top_k: int = 1,
            *,
            min_time: int | None = None,
            max_time: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant texts from Redis.

        Args:
            embedding:
                Query embedding.
            top_k:
                Number of results to return.
            min_time:
                Minimum timestamp filter.
            max_time:
                Maximum timestamp filter.

        Returns:
            List of result dicts, may be fewer than top_k.
        """
        self._validate_embedding_dimensions(embedding)
        self._validate_top_k(top_k)

        vector_query = self._build_vector_query(
            embedding=embedding,
            top_k=top_k,
            min_time=min_time,
            max_time=max_time,
        )
        results = self._search_redis(vector_query)
        return self._transform_redis_results(results)

    def health_check(self) -> dict[str, Any]:
        """
        Check Redis connection and index health.

        Returns:
            Dictionary with health status and diagnostics.
        """
        health_status = {
            'redis_connected':       False,
            'index_exists':          False,
            'index_healthy':         False,
            'memory_info':           {},
            'last_check':            int(time.time()),
            'errors':                []
        }

        with self.redis_context() as client:

            info = client.info()
            if isinstance(info, Mapping):
                health_status['redis_connected'] = True
                health_status['memory_info'] = self._extract_memory_info(info)

        index_info = self.index.info()
        health_status['index_exists'] = bool(index_info)
        health_status['index_healthy'] = 'num_docs' in index_info
        health_status['document_count'] = index_info.get('num_docs', 0)

        return health_status

    def list_texts(self) -> list[str]:
        """
        List all text IDs in Redis.

        Returns:
            Sorted list of text IDs.
        """
        with self.redis_context() as client:
            keys = client.keys(f'{TEXT_ID_PREFIX}*')
            return sorted(cast(list[str], keys))

    @contextmanager
    def redis_context(self) -> Iterator[redis.Redis]:
        """Context manager for Redis connection and error handling."""
        try:
            self.redis_client.ping()
            yield self.redis_client

        except redis.ConnectionError as e:
            _LOG.error('Redis connection failed: %s', e)
            raise StorageConnectionError(
                f'Redis connection failed: {e}') from e

        except redis.TimeoutError as e:
            _LOG.error('Redis operation timed out: %s', e)
            raise StorageConnectionError(f'Redis timeout: {e}') from e

        except redis.ResponseError as e:
            error_msg = str(e).lower()
            if 'oom' in error_msg or 'memory' in error_msg:
                _LOG.error('Redis out of memory: %s', e)
                raise RedisCapacityError(f'Redis out of memory: {e}') from e

            _LOG.error('Redis operation failed: %s', e)
            raise DataError(f'Redis operation failed: {e}') from e

    def store_text(
            self,
            text: str,
            embedding: np.ndarray,
            *,
            text_id: str | None = None,
            metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """
        Store text and embedding in Redis.

        Args:
            text:
                Text to store.
            embedding:
                Vector embedding.
            text_id:
                Optional ID for the text.
            metadata:
                Optional metadata mapping.

        Raises:
            ValidationError:
                If text is empty.
        """
        if not text.strip():
            raise ValidationError('text cannot be empty')

        if text_id is None:
            _LOG.debug('generating text_id')
            text_id = self._generate_text_id(text_id)

        self._validate_input_sizes(text, metadata)
        self._validate_text_id(text_id)
        self._validate_embedding_dimensions(embedding)

        sanitized = sanitize_metadata(
            metadata=metadata,
            schema=self.metadata_schema,
        )
        if 'tags' in sanitized:
            sanitized['tags'] = self._prepare_tags(tags=sanitized['tags'])
        text_data = self._prepare_text_data(
            text=text,
            embedding=embedding,
            metadata=sanitized,
        )
        self._store_to_redis(text_id, text_data)
        return text_id

    def close(self) -> None:
        """Close Redis connection pool."""
        if hasattr(self, 'redis_client'):
            self.redis_client.close()

    def _create_redis_schema(self, index_name: str) -> IndexSchema:
        """
        Create a Redis-specific schema for the vector search index.

        Configures fields like text, tags, and embeddings for Redis
        storage and retrieval using the provided index name and
        dimensions.

        Args:
            index_name:
                Name of the Redis search index.

        Returns:
            Configured IndexSchema object for Redis.
        """
        return IndexSchema.from_dict({
            'index':  {
                'name':   index_name,
                'prefix': TEXT_ID_PREFIX,
            },
            'fields': [
                {
                    'name': 'text',
                    'type': 'text',
                },
                {
                    'name': 'chunk_position',
                    'type': 'numeric',
                },
                {
                    'name': 'parent_id',
                    'type': 'text',
                },
                {
                    'name': 'source',
                    'type': 'text',
                },
                {
                    'name': 'timestamp',
                    'type': 'numeric',
                },
                {
                    'name': 'confidence',
                    'type': 'numeric',
                },
                {
                    'name': 'language',
                    'type': 'text',
                },
                {
                    'name': 'section',
                    'type': 'text',
                },
                {
                    'name': 'author',
                    'type': 'text',
                },
                {
                    'name': 'tags',
                    'type': 'tag',
                    'attrs': {'separator': self.TAG_SEPARATOR},
                },
                {
                    'name': 'embedding',
                    'type': 'vector',
                    'attrs': {
                        'dims':             self.dimensions,
                        'algorithm':        'HNSW',
                        'distance_metric':  'COSINE',
                    },
                },
            ]
        })

    def _enforce_schema_version(self) -> None:
        """Check whether stored schema matches current version."""
        with self.redis_context() as client:
            version_key = f'schema_version:{self.index_name}'
            stored_version = client.get(version_key)

            if stored_version is None:

                client.set(version_key, self.SCHEMA_VERSION)
                _LOG.info('set schema version to %s for index %s',
                          self.SCHEMA_VERSION, self.index_name)
            else:
                if isinstance(stored_version, str):
                    stored_version = int(stored_version)

                if stored_version != self.SCHEMA_VERSION:
                    raise ConfigurationError(
                        f'Schema version mismatch for {self.index_name=}: '
                        f'{stored_version=}, expected={self.SCHEMA_VERSION}. '
                        f'Clear the index or update schema version.'
                    )
                _LOG.debug('schema version %s confirmed for index %s',
                           self.SCHEMA_VERSION, self.index_name)

    @staticmethod
    def _extract_memory_info(info: Mapping) -> dict[str, Any]:
        """
        Extract memory information from Redis meminfo.
        """
        return {
            'used_memory':               info.get('used_memory', 0),
            'used_memory_human':         info.get('used_memory_human',
                                                  'unknown'),
            'used_memory_peak':          info.get('used_memory_peak', 0),
            'used_memory_peak_human':    info.get('used_memory_peak_human',
                                                  'unknown'),
            'maxmemory':                 info.get('maxmemory', 0),
            'maxmemory_human':           info.get('maxmemory_human',
                                                  'not set'),
            'maxmemory_policy':          info.get('maxmemory_policy',
                                                  'noeviction'),
            'total_system_memory':       info.get('total_system_memory', 0),
            'total_system_memory_human': info.get('total_system_memory_human',
                                                  'unknown'),
        }

    def _generate_text_id(self, text_id: str | None) -> str:
        """
        Generate a unique text ID.

        Args:
            text_id:
                Optional provided ID.

        Returns:
            Generated or provided text ID.
        """
        if text_id is None:
            with self.redis_context() as client:
                counter = client.incr(self.TEXT_COUNTER_KEY)
            text_id = f'{TEXT_ID_PREFIX}{counter}'
        return text_id

    def _parse_tags_from_retrieval(
            self,
            tags: str | list[str] | None,
    ) -> list[str]:
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-nested-blocks
        """
        Parse tags from Redis retrieval into a clean list.

        Args:
            tags: Tags from Redis.

        Returns:
            List of tag strings.
        """
        if tags is None:
            return []

        clean_tags: list[str] = []
        strip_chars = "[]'\" \t\n"

        if isinstance(tags, str):
            for tag in tags.split(self.TAG_SEPARATOR):
                tag = tag.strip(strip_chars)
                if tag:
                    clean_tags.append(tag)

        elif isinstance(tags, list):
            for tag in tags:

                if isinstance(tag, str):
                    if self.TAG_SEPARATOR in tag:
                        split_tags = tag.split(self.TAG_SEPARATOR)
                        for t in split_tags:
                            t = t.strip(strip_chars)
                            if t:
                                clean_tags.append(t)

                    else:
                        tag = tag.strip(strip_chars)
                        if tag:
                            clean_tags.append(tag)

        else:
            _LOG.warning('bad tags type: %s', type(tags))

        return clean_tags

    def _prepare_tags(self, tags: Any) -> str:
        """
        Convert tags to a string for Redis storage.

        Args:
            tags:
                Tags to convert (list or other).

        Returns:
            Comma-separated string of tags.
        """
        if isinstance(tags, list):
            return self.TAG_SEPARATOR.join(str(t).strip() for t in tags)
        return str(tags)

    def _search_redis(self, vector_query: VectorQuery) -> Any:
        """
        Execute a vector search in Redis.

        Args:
            vector_query:
                Query to execute.

        Raises:
            StorageConnectionError:
                If Redis connection fails.
            QueryError:
                If search operation fails.

        Returns:
            Redis search results.
        """
        try:
            return self.index.search(
                vector_query.query,
                query_params=vector_query.params,
            )

        except redisvl.exceptions.RedisVLError as e:
            error_msg = str(e).lower()
            if 'oom' in error_msg or 'memory' in error_msg:
                _LOG.error('Redis out of memory during search: %s', e)
                raise RedisCapacityError(
                    f'Redis out of memory during search: {e}') from e

            _LOG.error('Redis search failed: %s', e)
            raise QueryError(f'Redis search failed: {e}') from e

        except redis.ResponseError as e:
            msg = f'Redis operation failed: {e}'
            _LOG.error(msg)
            raise QueryError(msg) from e

        except redis.ConnectionError as e:
            msg = f'connection failed: {e}'
            _LOG.error(msg)
            raise StorageConnectionError(msg) from e

    def _store_to_redis(self, text_id: str, text_data: dict[str, Any]) -> None:
        """
        Store text data in Redis using RedisVL.

        Args:
            text_id:
                ID for the text.
            text_data:
                Data dict to store.
        """
        with self.redis_context():
            self.index.load([text_data], keys=[text_id])

    def _transform_redis_results(self, results: Any) -> list[dict[str, Any]]:
        """
        Transform Redis results into dicts.

        Args:
            results:
                Raw Redis search results.

        Returns:
            List of result dicts.
        """
        def _build_doc_dict(doc):
            return {
                'text_id':          doc.id,
                'text':             doc.text,
                'parent_id':        doc.parent_id,
                'source':           doc.source,
                'language':         doc.language,
                'section':          doc.section,
                'author':           doc.author,
                'distance':         float(doc.vector_distance),
                'tags':             self._parse_tags_from_retrieval(doc.tags),
                'timestamp':        int(doc.timestamp) if doc.timestamp else 0,
                'confidence':       (float(doc.confidence)
                                     if doc.confidence else None),
                'chunk_position':   (int(doc.chunk_position)
                                     if doc.chunk_position else None),
            }

        return [_build_doc_dict(doc) for doc in results.docs]

    @staticmethod
    def _build_vector_query(
            embedding: np.ndarray,
            top_k: int,
            min_time: int | None,
            max_time: int | None,
    ) -> VectorQuery:
        """
        Build a vector query for Redis search.

        Args:
            embedding:
                Query embedding.
            top_k:
                Number of results to return.
            min_time:
                Minimum timestamp filter.
            max_time:
                Maximum timestamp filter.

        Raises:
            ValidationError:
                If top_k is not positive.

        Returns:
            Configured VectorQuery object.
        """
        if top_k < 1:
            raise ValidationError('top_k must be positive')

        _min_time: int | str | None = min_time
        _max_time: int | str | None = max_time

        if _min_time is None:
            _min_time = '-inf'
        if _max_time is None:
            _max_time = '+inf'

        filter_expr = f'@timestamp:[{_min_time} {_max_time}]'

        return_fields = [
            'text', 'chunk_position', 'parent_id', 'source', 'timestamp',
            'tags', 'confidence', 'language', 'section', 'author'
        ]
        return VectorQuery(
            vector=embedding.tobytes(),
            vector_field_name='embedding',
            return_fields=return_fields,
            num_results=top_k,
            filter_expression=filter_expr if filter_expr else None,
        )

    @staticmethod
    def _prepare_text_data(
            text: str,
            embedding: np.ndarray,
            metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Prepare data dict for Redis storage.

        Args:
            text:
                Text to store.
            embedding:
                Vector embedding.
            metadata:
                Sanitized metadata.

        Returns:
            Dict with text, embedding, and metadata.
        """
        return {
            'text':             text,
            'embedding':        embedding.tobytes(),
            **metadata,
        }

    def _validate_embedding_dimensions(self, embedding: np.ndarray) -> None:
        dim = embedding.shape[0]
        if dim != self.dimensions:
            raise ConfigurationError('Embedding dimension mismatch: '
                                     f'{dim} != {self.dimensions}')

    def _validate_input_sizes(
            self,
            text: str,
            metadata: Mapping[str, Any] | None,
    ) -> None:
        """
        Validate input doesn't exceed Redis limits.

        Args:
            text:
                Text content to validate
            metadata:
                Optional metadata to validate

        Raises:
            ValidationError:
                If inputs exceed Redis limits
        """
        text_bytes = len(text.encode('utf-8'))
        if text_bytes > self.MAX_TEXT_SIZE:
            raise ValidationError(f'Text too large: {text_bytes} bytes '
                                  f'(max: {self.MAX_TEXT_SIZE})')

        if metadata:
            total_metadata_size = 0
            for key, value in metadata.items():
                key_bytes = len(str(key).encode('utf-8'))
                value_bytes = len(str(value).encode('utf-8'))

                if key_bytes > self.MAX_FIELD_SIZE:
                    raise ValidationError(f'Metadata key too large: "{key}" '
                                          f'({key_bytes} bytes)')

                if value_bytes > self.MAX_FIELD_SIZE:
                    raise ValidationError('Metadata value too large for '
                                          f'key "{key}": {value_bytes} bytes')

                total_metadata_size += key_bytes + value_bytes

            if total_metadata_size > self.MAX_METADATA_SIZE:
                raise ValidationError('Total metadata too large: '
                                      f'{total_metadata_size} bytes')

    @staticmethod
    def _validate_text_id(text_id: str) -> None:
        """
        Validate text ID format and length.

        Args:
            text_id:
                Text ID to validate.

        Raises:
            ValidationError:
                If text_id is invalid.
        """
        if not text_id or not text_id.strip():
            raise ValidationError('text_id cannot be empty')

        if len(text_id) > MAX_TEXT_ID_LENGTH:
            raise ValidationError('text_id too long: '
                                  f'{len(text_id)} > {MAX_TEXT_ID_LENGTH}')

        if not text_id.startswith(TEXT_ID_PREFIX):
            raise ValidationError('text_id must start with '
                                  f'"{TEXT_ID_PREFIX}"')

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if not isinstance(top_k, int) or top_k < 1:
            raise ValidationError('top_k must be a positive integer')

    def __del__(self):
        """Destructor to ensure Redis connection is closed."""
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
