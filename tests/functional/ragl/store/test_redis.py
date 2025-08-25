import unittest
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import redis
import redisvl.exceptions
from redisvl.query import VectorQuery

from ragl.config import RedisConfig
from ragl.constants import TEXT_ID_PREFIX
from ragl.exceptions import (
    ConfigurationError,
    DataError,
    QueryError,
    StorageCapacityError,
    StorageConnectionError,
    ValidationError,
)
from ragl.store.redis import RedisVectorStore
from ragl.textunit import TextUnit


class TestRedisVectorStore(unittest.TestCase):
    """Test cases for RedisVectorStore class."""

    @patch('ragl.store.redis.SearchIndex')
    @patch('ragl.store.redis.IndexSchema')
    @patch('ragl.store.redis.redis.Redis')
    @patch('redis.BlockingConnectionPool')
    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def setUp(self, mock_validate, mock_pool, mock_redis, mock_schema_class,
              mock_index_class):
        """Set up test fixtures with proper mocking."""
        # Configure mock Redis client
        self.mock_redis_client = mock_redis.return_value
        self.mock_redis_client.ping.return_value = True
        self.mock_redis_client.info.return_value = {
            'used_memory':               1024,
            'used_memory_human':         '1K',
            'used_memory_peak':          2048,
            'used_memory_peak_human':    '2K',
            'maxmemory':                 0,
            'maxmemory_human':           'not set',
            'maxmemory_policy':          'noeviction',
            'total_system_memory':       8589934592,
            'total_system_memory_human': '8G'
        }
        self.mock_redis_client.get.return_value = str(
            RedisVectorStore.SCHEMA_VERSION)
        self.mock_redis_client.set.return_value = True
        self.mock_redis_client.incr.return_value = 1
        self.mock_redis_client.delete.return_value = 1
        self.mock_redis_client.keys.return_value = [f'{TEXT_ID_PREFIX}1',
                                                    f'{TEXT_ID_PREFIX}2']

        # Configure mock connection pool
        mock_pool.return_value = Mock()

        # Configure mock schema
        self.mock_schema = mock_schema_class.from_dict.return_value

        # Configure mock index
        self.mock_index = mock_index_class.return_value
        self.mock_index.create.return_value = None
        self.mock_index.delete.return_value = None
        self.mock_index.info.return_value = {'num_docs': 10}
        self.mock_index.load.return_value = None

        # Mock search results
        mock_doc = Mock()
        mock_doc.id = f'{TEXT_ID_PREFIX}1'
        mock_doc.text = 'test text'
        mock_doc.parent_id = 'parent-1'
        mock_doc.source = 'test'
        mock_doc.language = 'en'
        mock_doc.section = 'intro'
        mock_doc.author = 'test_author'
        mock_doc.vector_distance = 0.5
        mock_doc.tags = 'tag1,tag2'
        mock_doc.timestamp = 12345
        mock_doc.confidence = 0.8
        mock_doc.chunk_position = 0

        mock_search_results = Mock()
        mock_search_results.docs = [mock_doc]
        self.mock_index.search.return_value = mock_search_results

        # Test configuration
        self.redis_config = RedisConfig(host='localhost', port=6379, db=0)
        self.dimensions = 768
        self.index_name = 'test_index'
        self.embedding = np.random.rand(self.dimensions)

        # Create store instance
        self.store = RedisVectorStore(
            redis_config=self.redis_config,
            dimensions=self.dimensions,
            index_name=self.index_name
        )

    @patch('redis.Redis')
    @patch('redisvl.index.SearchIndex')
    @patch('redisvl.schema.IndexSchema')
    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_with_redis_config(self, mock_validate, mock_schema_class,
                                    mock_index_class,
                                    mock_redis_class):
        """Test initialization with RedisConfig."""
        mock_redis_class.return_value = self.mock_redis_client
        mock_schema_class.from_dict.return_value = self.mock_schema
        mock_index_class.return_value = self.mock_index

        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_config=self.redis_config,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        self.assertEqual(store.dimensions, self.dimensions)
        self.assertEqual(store.index_name, self.index_name)
        self.assertEqual(store.redis_client, self.mock_redis_client)

    @patch('redis.Redis')
    @patch('redisvl.index.SearchIndex')
    @patch('redisvl.schema.IndexSchema')
    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_with_redis_client(self, mock_validate, mock_schema_class,
                                    mock_index_class, mock_redis_class):
        """Test initialization with Redis client."""
        mock_schema_class.from_dict.return_value = self.mock_schema
        mock_index_class.return_value = self.mock_index

        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        self.assertEqual(store.redis_client, self.mock_redis_client)
        mock_redis_class.assert_not_called()

    @patch('ragl.store.redis.redis.Redis')
    @patch('ragl.store.redis.SearchIndex')
    @patch('ragl.store.redis.IndexSchema')
    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_clear(self, mock_validate, mock_schema_class, mock_index_class,
                   mock_redis_class):
        """Test clearing the index."""
        mock_redis_class.return_value = self.mock_redis_client
        mock_schema_class.from_dict.return_value = self.mock_schema
        mock_index_class.return_value = self.mock_index

        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_config=self.redis_config,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        store.clear()

        self.mock_index.delete.assert_called_once()
        self.mock_index.create.assert_called()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_delete_text_success(self, mock_validate_sync):
        """Test successful text deletion."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_id = f'{TEXT_ID_PREFIX}123'
        self.mock_redis_client.delete.return_value = 1

        result = store.delete_text(text_id)

        self.assertTrue(result)
        self.mock_redis_client.delete.assert_called_once_with(text_id)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_delete_text_not_found(self, mock_validate_sync):
        """Test deleting non-existent text."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_id = f'{TEXT_ID_PREFIX}123'
        self.mock_redis_client.delete.return_value = 0

        result = store.delete_text(text_id)

        self.assertFalse(result)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_delete_text_invalid_id(self, mock_validate_sync):
        """Test deleting text with invalid ID."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with self.assertRaises(ValidationError):
            store.delete_text('invalid_id')

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_delete_texts_success(self, mock_validate_sync):
        """Test successful deletion of multiple texts."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_ids = [f'{TEXT_ID_PREFIX}1', f'{TEXT_ID_PREFIX}2',
                    f'{TEXT_ID_PREFIX}3']
        self.mock_redis_client.delete.return_value = 3

        result = store.delete_texts(text_ids)

        self.assertEqual(result, 3)
        self.mock_redis_client.delete.assert_called_once_with(*text_ids)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_delete_texts_partial_success(self, mock_validate_sync):
        """Test deletion when some texts don't exist."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_ids = [f'{TEXT_ID_PREFIX}1', f'{TEXT_ID_PREFIX}2',
                    f'{TEXT_ID_PREFIX}999']
        self.mock_redis_client.delete.return_value = 2  # Only 2 existed

        result = store.delete_texts(text_ids)

        self.assertEqual(result, 2)
        self.mock_redis_client.delete.assert_called_once_with(*text_ids)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_delete_texts_empty_list(self, mock_validate_sync):
        """Test deletion with empty list."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        result = store.delete_texts([])

        self.assertEqual(result, 0)
        self.mock_redis_client.delete.assert_not_called()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_delete_texts_single_item(self, mock_validate_sync):
        """Test deletion with single text ID."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_ids = [f'{TEXT_ID_PREFIX}1']
        self.mock_redis_client.delete.return_value = 1

        result = store.delete_texts(text_ids)

        self.assertEqual(result, 1)
        self.mock_redis_client.delete.assert_called_once_with(
            f'{TEXT_ID_PREFIX}1')

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_delete_texts_invalid_text_id_empty(self, mock_validate_sync):
        """Test deletion with invalid empty text ID."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_ids = ['']

        with self.assertRaises(ValidationError) as context:
            store.delete_texts(text_ids)

        self.assertIn('text_id cannot be empty', str(context.exception))
        self.mock_redis_client.delete.assert_not_called()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_delete_texts_invalid_text_id_wrong_prefix(self,
                                                       mock_validate_sync):
        """Test deletion with invalid text ID prefix."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_ids = ['invalid_prefix:123']

        with self.assertRaises(ValidationError) as context:
            store.delete_texts(text_ids)

        self.assertIn('text_id must start with', str(context.exception))
        self.mock_redis_client.delete.assert_not_called()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_delete_texts_invalid_text_id_too_long(self, mock_validate_sync):
        """Test deletion with text ID that's too long."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        long_id = TEXT_ID_PREFIX + 'x' * (
                RedisVectorStore.MAX_TEXT_ID_LENGTH + 1)
        text_ids = [long_id]

        with self.assertRaises(ValidationError) as context:
            store.delete_texts(text_ids)

        self.assertIn('text_id too long', str(context.exception))
        self.mock_redis_client.delete.assert_not_called()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_delete_texts_mixed_valid_invalid(self, mock_validate_sync):
        """Test deletion with mix of valid and invalid text IDs."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_ids = [f'{TEXT_ID_PREFIX}1', 'invalid_prefix:2']

        with self.assertRaises(ValidationError) as context:
            store.delete_texts(text_ids)

        self.assertIn('text_id must start with', str(context.exception))
        self.mock_redis_client.delete.assert_not_called()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_delete_texts_redis_connection_error(self, mock_validate_sync):
        """Test deletion with Redis connection error."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_ids = [f'{TEXT_ID_PREFIX}1']

        # Mock redis_context to raise StorageConnectionError
        with patch.object(store, 'redis_context') as mock_context:
            mock_context.side_effect = StorageConnectionError(
                "Connection failed")

            with self.assertRaises(StorageConnectionError):
                store.delete_texts(text_ids)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_get_relevant_success(self, mock_validate_sync):
        """Test successful relevance search."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        embedding = np.random.rand(self.dimensions)
        mock_results = Mock()
        mock_doc = Mock()
        mock_doc.text = 'Sample text'
        mock_doc.id = f'{TEXT_ID_PREFIX}123'
        mock_doc.parent_id = 'parent_123'
        mock_doc.source = 'test'
        mock_doc.language = 'en'
        mock_doc.section = 'intro'
        mock_doc.author = 'test_author'
        mock_doc.vector_distance = 0.85
        mock_doc.tags = 'tag1,tag2'
        mock_doc.timestamp = 1234567890
        mock_doc.confidence = 0.9
        mock_doc.chunk_position = 0
        mock_results.docs = [mock_doc]

        with patch.object(store, '_search_redis', return_value=mock_results):
            results = store.get_relevant(embedding, top_k=5)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.text, 'Sample text')
        self.assertEqual(result.text_id, f'{TEXT_ID_PREFIX}123')
        self.assertEqual(result.tags, ['tag1', 'tag2'])

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_get_relevant_with_time_filters(self, mock_validate_sync):
        """Test relevance search with time filters."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        embedding = np.random.rand(self.dimensions)
        min_time = 1000
        max_time = 2000

        with patch.object(store, '_build_vector_query') as mock_build_query:
            with patch.object(store, '_search_redis',
                              return_value=Mock(docs=[])):
                with patch.object(store, '_transform_redis_results',
                                  return_value=[]):
                    store.get_relevant(embedding, top_k=5, min_time=min_time,
                                       max_time=max_time)

        call_args = mock_build_query.call_args
        args = {
            'embedding': embedding,
            'top_k': 5,
            'min_time': min_time,
            'max_time': max_time
        }
        self.assertEqual(call_args[1], args)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_get_relevant_invalid_dimensions(self, mock_validate_sync):
        """Test relevance search with invalid embedding dimensions."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        wrong_embedding = np.random.rand(512)  # Wrong dimensions

        with self.assertRaises(ConfigurationError):
            store.get_relevant(wrong_embedding, top_k=5)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_health_check_success(self, mock_validate_sync):
        """Test successful health check."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with patch.object(store, 'index', self.mock_index):
            result = store.health_check()

        self.assertTrue(result['redis_connected'])
        self.assertTrue(result['index_exists'])
        self.assertTrue(result['index_healthy'])
        self.assertEqual(result['document_count'], 10)
        self.assertIn('memory_info', result)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_health_check_redis_connection_error(self, mock_validate_sync):
        """Test health check with Redis connection error."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        self.mock_redis_client.ping.side_effect = redis.ConnectionError(
            "Connection failed")

        with self.assertRaises(StorageConnectionError):
            store.health_check()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_list_texts(self, mock_validate_sync):
        """Test listing all text IDs."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        expected_keys = [f'{TEXT_ID_PREFIX}1', f'{TEXT_ID_PREFIX}2',
                         f'{TEXT_ID_PREFIX}3']
        self.mock_redis_client.keys.return_value = expected_keys

        result = store.list_texts()

        self.assertEqual(result, sorted(expected_keys))
        self.mock_redis_client.keys.assert_called_once_with(
            f'{TEXT_ID_PREFIX}*')

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_redis_context_success(self, mock_validate_sync):
        """Test successful Redis context manager."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with store.redis_context() as client:
            self.assertEqual(client, self.mock_redis_client)

        self.mock_redis_client.ping.assert_called()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_redis_context_connection_error(self, mock_validate_sync):
        """Test Redis context manager with connection error."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        self.mock_redis_client.ping.side_effect = redis.ConnectionError(
            "Connection failed")

        with self.assertRaises(StorageConnectionError):
            with store.redis_context():
                pass

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_redis_context_timeout_error(self, mock_validate_sync):
        """Test Redis context manager with timeout error."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        self.mock_redis_client.ping.side_effect = redis.TimeoutError("Timeout")

        with self.assertRaises(StorageConnectionError):
            with store.redis_context():
                pass

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_redis_context_response_error_oom(self, mock_validate_sync):
        """Test Redis context manager with OOM response error."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        self.mock_redis_client.ping.side_effect = redis.ResponseError(
            "OOM command not allowed")

        with self.assertRaises(StorageCapacityError):
            with store.redis_context():
                pass

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_redis_context_response_error_other(self, mock_validate_sync):
        """Test Redis context manager with other response error."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        self.mock_redis_client.ping.side_effect = redis.ResponseError(
            "Other error")

        with self.assertRaises(DataError):
            with store.redis_context():
                pass

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_text_success(self, mock_validate_sync):
        """Test successful text storage."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text = "Sample text"
        text_unit = TextUnit(
            text=text,
            text_id=None,
            distance=0.0,
        )
        embedding = np.random.rand(self.dimensions)
        metadata = {'tags': ['test'], 'source': 'test_source'}

        self.mock_redis_client.incr.return_value = 123

        # Configure the mock index.load to return the expected text ID
        expected_text_id = f'{TEXT_ID_PREFIX}123'
        self.mock_index.load.return_value = [expected_text_id]

        with patch('ragl.store.redis.sanitize_metadata',
                   return_value=metadata):
            with patch.object(store, 'index', self.mock_index):
                result = store.store_text(text_unit, embedding)

        self.assertEqual(result, expected_text_id)
        self.mock_index.load.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_text_with_custom_id(self, mock_validate_sync):
        """Test storing text with custom text ID."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )
        text = "Sample text"
        custom_id = f'{TEXT_ID_PREFIX}custom'
        text_unit = TextUnit(
            text=text,
            text_id=custom_id,
            distance=0.0,
        )

        embedding = np.random.rand(self.dimensions)

        # Configure the mock index.load to return the expected text ID
        self.mock_index.load.return_value = [custom_id]

        with patch('ragl.store.redis.sanitize_metadata', return_value={}):
            with patch.object(store, 'index', self.mock_index):
                result = store.store_text(text_unit, embedding)

        self.assertEqual(result, custom_id)
        self.mock_index.load.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_text_empty_text(self, mock_validate_sync):
        """Test storing empty text."""

        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        embedding = np.random.rand(self.dimensions)

        with self.assertRaises(ValidationError):
            text_unit = TextUnit(
                text="",
                text_id='test-id',
                distance=0.0,
            )
            store.store_text(text_unit, embedding)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_text_whitespace_only(self, mock_validate_sync):
        """Test storing whitespace-only text."""
        text = "   \n\t   "
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        embedding = np.random.rand(self.dimensions)

        with self.assertRaises(ValidationError):
            text_unit = TextUnit(
                text=text,
                text_id='test-id',
                distance=0.0,
            )
            store.store_text(text_unit, embedding)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_text_invalid_embedding_dimensions(self, mock_validate_sync):
        """Test storing text with invalid embedding dimensions."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )
        text_unit = TextUnit(
            text="txt",
            text_id='txt:100',
            distance=0.0,
        )

        wrong_embedding = np.random.rand(512)  # Wrong dimensions

        with self.assertRaises(ConfigurationError):
            store.store_text(text_unit, wrong_embedding)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_success(self, mock_validate_sync):
        """Test successful batch text storage."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_unit = TextUnit(text="Sample text 1", text_id=None, distance=0.0)
        embedding = np.random.rand(self.dimensions)
        text_embedding_pairs = [(text_unit, embedding)]

        self.mock_redis_client.incr.return_value = 123

        # Configure the mock index.load to return the expected text ID
        expected_ids = [f'{TEXT_ID_PREFIX}123']
        self.mock_index.load.return_value = expected_ids

        with patch('ragl.store.redis.sanitize_metadata', return_value={}):
            with patch.object(store, 'index', self.mock_index):
                results = store.store_texts(text_embedding_pairs)

        self.assertEqual(results, expected_ids)
        self.mock_index.load.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_multiple_pairs(self, mock_validate_sync):
        """Test batch storage with multiple text-embedding pairs."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_embedding_pairs = [
            (TextUnit(text="Sample text 1", text_id=None, distance=0.0),
             np.random.rand(self.dimensions)),
            (TextUnit(text="Sample text 2", text_id=None, distance=0.0),
             np.random.rand(self.dimensions)),
        ]

        self.mock_redis_client.incr.side_effect = [123, 124]

        # Configure the mock index.load to return the expected text IDs
        expected_ids = [f'{TEXT_ID_PREFIX}123', f'{TEXT_ID_PREFIX}124']
        self.mock_index.load.return_value = expected_ids

        with patch('ragl.store.redis.sanitize_metadata', return_value={}):
            with patch.object(store, 'index', self.mock_index):
                results = store.store_texts(text_embedding_pairs)

        self.assertEqual(results, expected_ids)
        self.mock_index.load.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_with_custom_ids(self, mock_validate_sync):
        """Test batch storage with custom text IDs."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        custom_ids = [f'{TEXT_ID_PREFIX}custom1', f'{TEXT_ID_PREFIX}custom2']
        text_embedding_pairs = [
            (TextUnit(text="Sample text 1", text_id=custom_ids[0],
                      distance=0.0), np.random.rand(self.dimensions)),
            (TextUnit(text="Sample text 2", text_id=custom_ids[1],
                      distance=0.0), np.random.rand(self.dimensions)),
        ]

        # Configure the mock index.load to return the expected custom IDs
        self.mock_index.load.return_value = custom_ids

        with patch('ragl.store.redis.sanitize_metadata', return_value={}):
            with patch.object(store, 'index', self.mock_index):
                results = store.store_texts(text_embedding_pairs)

        self.assertEqual(results, custom_ids)
        self.mock_index.load.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_empty_list(self, mock_validate_sync):
        """Test batch storage with empty input list."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        results = store.store_texts([])
        self.assertEqual(results, [])

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_invalid_embedding_dimensions(self,
                                                      mock_validate_sync):
        """Test batch storage with invalid embedding dimensions."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_unit = TextUnit(text="Sample text", text_id=None, distance=0.0)
        wrong_embedding = np.random.rand(512)  # Wrong dimensions
        text_embedding_pairs = [(text_unit, wrong_embedding)]

        with self.assertRaises(ConfigurationError):
            store.store_texts(text_embedding_pairs)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_empty_text_validation(self, mock_validate_sync):
        """Test batch storage with empty text."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        embedding = np.random.rand(self.dimensions)

        with self.assertRaises(ValidationError):
            text_unit = TextUnit(text="", text_id=None, distance=0.0)
            text_embedding_pairs = [(text_unit, embedding)]
            store.store_texts(text_embedding_pairs)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_whitespace_only_text(self, mock_validate_sync):
        """Test batch storage with whitespace-only text."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )


        with self.assertRaises(ValidationError):
            text_unit = TextUnit(text="   \n\t   ", text_id=None, distance=0.0)
            embedding = np.random.rand(self.dimensions)
            text_embedding_pairs = [(text_unit, embedding)]
            store.store_texts(text_embedding_pairs)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_empty_text_after_strip(self, mock_validate_sync):
        """Test that store_texts raises ValidationError for whitespace-only text."""
        text_unit = TextUnit(
            text="valid text",  # Start with valid text
            text_id="test_id",
            distance=0.0
        )
        # Bypass TextUnit validation by setting the private attribute directly
        text_unit._text = "   \n\t   "  # whitespace-only text

        embedding = np.random.rand(self.dimensions).astype(np.float32)

        with self.assertRaises(ValidationError) as cm:
            self.store.store_texts([(text_unit, embedding)])

        self.assertIn('text cannot be empty', str(cm.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_large_batch(self, mock_validate_sync):
        """Test batch storage with large number of texts."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        batch_size = 100
        text_embedding_pairs = [
            (TextUnit(text=f"Sample text {i}", text_id=None, distance=0.0),
             np.random.rand(self.dimensions))
            for i in range(batch_size)
        ]

        # Mock incrementing counter for each text
        self.mock_redis_client.incr.side_effect = range(1, batch_size + 1)

        # Configure the mock index.load to return the expected text IDs
        expected_ids = [f'{TEXT_ID_PREFIX}{i}' for i in
                        range(1, batch_size + 1)]
        self.mock_index.load.return_value = expected_ids

        with patch('ragl.store.redis.sanitize_metadata', return_value={}):
            with patch.object(store, 'index', self.mock_index):
                results = store.store_texts(text_embedding_pairs)

        self.assertEqual(results, expected_ids)
        self.mock_index.load.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_redis_error(self, mock_validate_sync):
        """Test batch storage with Redis error during storage."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_unit = TextUnit(text="Sample text", text_id=None, distance=0.0)
        embedding = np.random.rand(self.dimensions)
        text_embedding_pairs = [(text_unit, embedding)]

        self.mock_redis_client.incr.return_value = 123

        with patch('ragl.store.redis.sanitize_metadata', return_value={}):
            with patch.object(store, 'index', self.mock_index):
                self.mock_index.load.side_effect = redis.ResponseError(
                    "Storage error")

                with self.assertRaises(DataError):
                    store.store_texts(text_embedding_pairs)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_connection_error(self, mock_validate_sync):
        """Test batch storage with Redis connection error."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_unit = TextUnit(text="Sample text", text_id=None, distance=0.0)
        embedding = np.random.rand(self.dimensions)
        text_embedding_pairs = [(text_unit, embedding)]

        self.mock_redis_client.incr.return_value = 123

        with patch('ragl.store.redis.sanitize_metadata', return_value={}):
            with patch.object(store, 'index', self.mock_index):
                self.mock_index.load.side_effect = redis.ConnectionError(
                    "Connection lost")

                with self.assertRaises(StorageConnectionError):
                    store.store_texts(text_embedding_pairs)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_partial_custom_ids(self, mock_validate_sync):
        """Test batch storage with mix of custom and auto-generated IDs."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_embedding_pairs = [
            (TextUnit(text="Sample text 1", text_id=f'{TEXT_ID_PREFIX}custom1',
                      distance=0.0), np.random.rand(self.dimensions)),
            (TextUnit(text="Sample text 2", text_id=None, distance=0.0),
             np.random.rand(self.dimensions)),
            (TextUnit(text="Sample text 3", text_id=f'{TEXT_ID_PREFIX}custom3',
                      distance=0.0), np.random.rand(self.dimensions)),
        ]

        # Only second text should trigger counter increment
        self.mock_redis_client.incr.return_value = 123

        # Configure the mock index.load to return the expected mixed IDs
        expected_ids = [f'{TEXT_ID_PREFIX}custom1', f'{TEXT_ID_PREFIX}123',
                        f'{TEXT_ID_PREFIX}custom3']
        self.mock_index.load.return_value = expected_ids

        with patch('ragl.store.redis.sanitize_metadata', return_value={}):
            with patch.object(store, 'index', self.mock_index):
                results = store.store_texts(text_embedding_pairs)

        self.assertEqual(results, expected_ids)
        self.mock_index.load.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_invalid_text_id(self, mock_validate_sync):
        """Test batch storage with invalid custom text ID."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_unit = TextUnit(text="Sample text", text_id="invalid_id",
                             distance=0.0)
        embedding = np.random.rand(self.dimensions)
        text_embedding_pairs = [(text_unit, embedding)]

        with self.assertRaises(ValidationError):
            store.store_texts(text_embedding_pairs)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_invalid_pair_type(self, mock_validate_sync):
        """Test batch storage with invalid pair types."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        invalid_pairs = ["not_a_tuple", "another_string"]

        with self.assertRaises(ValidationError):
            store.store_texts(invalid_pairs)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_wrong_pair_length(self, mock_validate_sync):
        """Test batch storage with wrong tuple length."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_unit = TextUnit(text="Sample text", text_id=None, distance=0.0)
        wrong_pairs = [(text_unit,)]

        with self.assertRaises(ValidationError):
            store.store_texts(wrong_pairs)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_invalid_text_unit_type(self, mock_validate_sync):
        """Test batch storage with invalid TextUnit type."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        embedding = np.random.rand(self.dimensions)
        invalid_pairs = [("not_a_text_unit", embedding)]

        with self.assertRaises(ValidationError):
            store.store_texts(invalid_pairs)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_invalid_embedding_type(self, mock_validate_sync):
        """Test batch storage with invalid embedding type."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_unit = TextUnit(text="Sample text", text_id=None, distance=0.0)
        invalid_pairs = [(text_unit, "not_an_embedding")]

        with self.assertRaises(ValidationError):
            store.store_texts(invalid_pairs)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_store_texts_non_list_input(self, mock_validate_sync):
        """Test batch storage with non-list input."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        # Test with various non-list types
        non_list_inputs = [
            "not_a_list",  # string
            123,  # integer
            {"key": "value"},  # dictionary
            ("tuple", "input"),  # tuple
            TextUnit(text="Sample", text_id=None, distance=0.0),
            # single TextUnit
        ]

        for non_list_input in non_list_inputs:
            with self.subTest(input_type=type(non_list_input).__name__):
                with self.assertRaises(ValidationError) as context:
                    print(non_list_input)
                    store.store_texts(non_list_input)

                self.assertIn("texts_and_embeddings must be a list",
                              str(context.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_close(self, mock_validate_sync):
        """Test closing Redis connection."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        store.close()
        self.mock_redis_client.close.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_create_redis_schema(self, mock_validate_sync):
        """Test creating Redis schema."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with patch('ragl.store.redis.IndexSchema') as mock_schema_class:
            mock_schema_class.from_dict.return_value = self.mock_schema
            result = store._create_index_schema(self.index_name)

        self.assertEqual(result, self.mock_schema)
        mock_schema_class.from_dict.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_enforce_schema_version_new(self, mock_validate_sync):
        """Test enforcing schema version for new index."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        self.mock_redis_client.get.return_value = None

        store._enforce_schema_version()

        version_key = f'schema_version:{self.index_name}'
        self.mock_redis_client.get.assert_called_with(version_key)
        self.mock_redis_client.set.assert_called_with(version_key,
                                                      RedisVectorStore.SCHEMA_VERSION)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_enforce_schema_version_mismatch(self, mock_validate_sync):
        """Test enforcing schema version with mismatch."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        self.mock_redis_client.get.return_value = b'999'  # Different version

        with self.assertRaises(ConfigurationError):
            store._enforce_schema_version()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_extract_memory_info(self, mock_validate_sync):
        """Test extracting memory information."""
        info = {
            'used_memory':               1000000,
            'used_memory_human':         '1.0M',
            'used_memory_peak':          2000000,
            'used_memory_peak_human':    '2.0M',
            'maxmemory':                 10000000,
            'maxmemory_human':           '10.0M',
            'maxmemory_policy':          'allkeys-lru',
            'total_system_memory':       16000000000,
            'total_system_memory_human': '16.0G'
        }

        result = RedisVectorStore._extract_memory_info(info)

        self.assertEqual(result['used_memory'], 1000000)
        self.assertEqual(result['used_memory_human'], '1.0M')
        self.assertEqual(result['maxmemory_policy'], 'allkeys-lru')

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_generate_text_id_auto(self, mock_validate_sync):
        """Test generating automatic text ID."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        self.mock_redis_client.incr.return_value = 123

        result = store._generate_text_id()

        self.assertEqual(result, f'{TEXT_ID_PREFIX}123')
        self.mock_redis_client.incr.assert_called_once_with(
            RedisVectorStore.TEXT_COUNTER_KEY)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_parse_tags_from_retrieval_none(self, mock_validate_sync):
        """Test parsing None tags."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        result = store._parse_tags_from_retrieval(None)
        self.assertEqual(result, [])

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_parse_tags_from_retrieval_string(self, mock_validate_sync):
        """Test parsing string tags."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        tags_str = "tag1,tag2,tag3"
        result = store._parse_tags_from_retrieval(tags_str)
        self.assertEqual(result, ['tag1', 'tag2', 'tag3'])

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_parse_tags_from_retrieval_string_with_brackets(self, mock_validate_sync):
        """Test parsing string tags with brackets."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        tags_str = "['tag1', 'tag2', 'tag3']"
        result = store._parse_tags_from_retrieval(tags_str)
        self.assertEqual(result, ['tag1', 'tag2', 'tag3'])

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_parse_tags_from_retrieval_list(self, mock_validate_sync):
        """Test parsing list tags."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        tags_list = ['tag1', 'tag2', 'tag3']
        result = store._parse_tags_from_retrieval(tags_list)
        self.assertEqual(result, ['tag1', 'tag2', 'tag3'])

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_parse_tags_from_retrieval_invalid_type(self, mock_validate_sync):
        """Test parsing invalid tag type."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        result = store._parse_tags_from_retrieval(123)
        self.assertEqual(result, [])

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_prepare_tags_list(self, mock_validate_sync):
        """Test preparing tags from list."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        tags = ['tag1', 'tag2', 'tag3']
        result = store._prepare_tags_for_storage(tags)
        self.assertEqual(result, 'tag1,tag2,tag3')

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_prepare_tags_string(self, mock_validate_sync):
        """Test preparing tags from string."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        tags = 'single_tag'
        result = store._prepare_tags_for_storage(tags)
        self.assertEqual(result, 'single_tag')

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_search_redis_success(self, mock_validate_sync):
        """Test successful Redis search."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        mock_query = Mock(spec=VectorQuery)
        mock_results = Mock()

        with patch.object(store, 'index', self.mock_index):
            self.mock_index.search.return_value = mock_results
            result = store._search_redis(mock_query)

        self.assertEqual(result, mock_results)
        self.mock_index.search.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_search_redis_redisvl_error_oom(self, mock_validate_sync):
        """Test Redis search with RedisVL OOM error."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        mock_query = Mock(spec=VectorQuery)

        with patch.object(store, 'index', self.mock_index):
            self.mock_index.search.side_effect = redisvl.exceptions.RedisVLError(
                "OOM error")

            with self.assertRaises(StorageCapacityError):
                store._search_redis(mock_query)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_search_redis_redisvl_error_other(self, mock_validate_sync):
        """Test Redis search with other RedisVL error."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        mock_query = Mock(spec=VectorQuery)

        with patch.object(store, 'index', self.mock_index):
            self.mock_index.search.side_effect = redisvl.exceptions.RedisVLError(
                "Other error")

            with self.assertRaises(QueryError):
                store._search_redis(mock_query)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_search_redis_response_error(self, mock_validate_sync):
        """Test Redis search with response error."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        mock_query = Mock(spec=VectorQuery)

        with patch.object(store, 'index', self.mock_index):
            self.mock_index.search.side_effect = redis.ResponseError(
                "Response error")

            with self.assertRaises(QueryError):
                store._search_redis(mock_query)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_search_redis_connection_error(self, mock_validate_sync):
        """Test Redis search with connection error."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        mock_query = Mock(spec=VectorQuery)

        with patch.object(store, 'index', self.mock_index):
            self.mock_index.search.side_effect = redis.ConnectionError(
                "Connection error")

            with self.assertRaises(StorageConnectionError):
                store._search_redis(mock_query)

    @patch( 'redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_transform_redis_results(self, mock_validate_sync):
        """Test transforming Redis results."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        mock_results = Mock()
        mock_doc = Mock()
        mock_doc.id = f'{TEXT_ID_PREFIX}123'  # Changed from text_id to id
        mock_doc.text = 'Sample text'
        mock_doc.parent_id = 'parent_123'
        mock_doc.source = 'test'
        mock_doc.language = 'en'
        mock_doc.section = 'intro'
        mock_doc.author = 'test_author'
        mock_doc.vector_distance = 0.85  # Set as actual float, not Mock
        mock_doc.tags = 'tag1,tag2'
        mock_doc.timestamp = 1234567890
        mock_doc.confidence = 0.9
        mock_doc.chunk_position = 0
        mock_results.docs = [mock_doc]

        with patch.object(store, '_parse_tags_from_retrieval',
                          return_value=['tag1', 'tag2']):
            results = store._transform_redis_results(mock_results)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result['text'], 'Sample text')
        self.assertEqual(result['text_id'], f'{TEXT_ID_PREFIX}123')
        self.assertEqual(result['distance'], 0.85)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_build_vector_query(self, mock_validate_sync):
        """Test building vector query."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        embedding = np.random.rand(self.dimensions)
        top_k = 5
        min_time = 1000
        max_time = 2000

        with patch('ragl.store.redis.VectorQuery') as mock_vector_query_class:
            mock_query = Mock()
            mock_vector_query_class.return_value = mock_query

            result = store._build_vector_query(embedding, top_k, min_time,
                                               max_time)

        self.assertEqual(result, mock_query)
        mock_vector_query_class.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_build_vector_query_no_time_filters(self, mock_validate_sync):
        """Test building vector query without time filters."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        embedding = np.random.rand(self.dimensions)
        top_k = 5

        with patch('ragl.store.redis.VectorQuery') as mock_vector_query_class:
            mock_query = Mock()
            mock_vector_query_class.return_value = mock_query

            result = store._build_vector_query(embedding, top_k, None, None)

        self.assertEqual(result, mock_query)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_build_vector_query_invalid_top_k(self, mock_validate_sync):
        """Test building vector query with invalid top_k."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        embedding = np.random.rand(self.dimensions)

        with self.assertRaises(ValidationError):
            store._build_vector_query(embedding, 0, None, None)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_prepare_text_data(self, mock_validate_sync):
        """Test preparing text data for storage."""
        text = "Sample text"
        embedding = np.random.rand(768)
        metadata = {'source': 'test', 'tags': 'tag1,tag2'}

        result = RedisVectorStore._prepare_text_data(text, embedding, metadata)

        self.assertEqual(result['text'], text)
        self.assertEqual(result['source'], 'test')
        self.assertEqual(result['tags'], 'tag1,tag2')
        self.assertIsInstance(result['embedding'], bytes)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_validate_embedding_dimensions_valid(self, mock_validate_sync):
        """Test validating valid embedding dimensions."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        embedding = np.random.rand(self.dimensions)
        store._validate_dimensions_match(embedding)  # Should not raise

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_validate_embedding_dimensions_invalid(self, mock_validate_sync):
        """Test validating invalid embedding dimensions."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        embedding = np.random.rand(512)  # Wrong dimensions

        with self.assertRaises(ConfigurationError):
            store._validate_dimensions_match(embedding)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_validate_input_sizes_valid(self, mock_validate_sync):
        """Test validating valid input sizes."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text = "Sample text"
        metadata = {'source': 'test'}

        store._validate_input_sizes(text, metadata)  # Should not raise

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_validate_input_sizes_text_too_large(self, mock_validate_sync):
        """Test validating text that's too large."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        # Create text larger than MAX_TEXT_SIZE
        large_text = "x" * (RedisVectorStore.MAX_TEXT_SIZE + 1)

        with self.assertRaises(ValidationError):
            store._validate_input_sizes(large_text, None)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_validate_input_sizes_metadata_too_large(self, mock_validate_sync):
        """Test validating metadata that's too large."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text = "Sample text"
        # Create metadata larger than MAX_METADATA_SIZE
        large_value = "x" * (RedisVectorStore.MAX_METADATA_SIZE + 1)
        metadata = {'large_field': large_value}

        with self.assertRaises(ValidationError):
            store._validate_input_sizes(text, metadata)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_validate_input_sizes_field_too_large(self, mock_validate_sync):
        """Test validating metadata field that's too large."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text = "Sample text"
        # Create field larger than MAX_FIELD_SIZE
        large_value = "x" * (RedisVectorStore.MAX_FIELD_SIZE + 1)
        metadata = {'large_field': large_value}

        with self.assertRaises(ValidationError):
            store._validate_input_sizes(text, metadata)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_validate_text_id_valid(self, mock_validate_sync):
        """Test validating valid text ID."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_id = f'{TEXT_ID_PREFIX}123'
        store._validate_text_id(text_id)  # Should not raise

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_validate_text_id_empty(self, mock_validate_sync):
        """Test validating empty text ID."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with self.assertRaises(ValidationError):
            store._validate_text_id("")

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_validate_text_id_too_long(self, mock_validate_sync):
        """Test validating text ID that's too long."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        long_id = TEXT_ID_PREFIX + "x" * (
                RedisVectorStore.MAX_TEXT_ID_LENGTH + 1)

        with self.assertRaises(ValidationError):
            store._validate_text_id(long_id)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_validate_text_id_wrong_prefix(self, mock_validate_sync):
        """Test validating text ID with wrong prefix."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        wrong_id = "wrong_prefix_123"

        with self.assertRaises(ValidationError):
            store._validate_text_id(wrong_id)

    def test_validate_top_k_valid(self):
        """Test validating valid top_k."""
        RedisVectorStore._validate_top_k(5)  # Should not raise

    def test_validate_top_k_zero(self):
        """Test validating zero top_k."""
        with self.assertRaises(ValidationError):
            RedisVectorStore._validate_top_k(0)

    def test_validate_top_k_negative(self):
        """Test validating negative top_k."""
        with self.assertRaises(ValidationError):
            RedisVectorStore._validate_top_k(-1)

    def test_validate_top_k_non_integer(self):
        """Test validating non-integer top_k."""
        with self.assertRaises(ValidationError):
            RedisVectorStore._validate_top_k(5.5)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_del_method(self, mock_validate_sync):
        """Test destructor method."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with patch.object(store, 'close') as mock_close:
            store.__del__()
            mock_close.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_context_manager_enter(self, mock_validate_sync):
        """Test context manager entry."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        result = store.__enter__()
        self.assertEqual(result, store)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_context_manager_exit(self, mock_validate_sync):
        """Test context manager exit."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with patch.object(store, 'close') as mock_close:
            store.__exit__(None, None, None)
            mock_close.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_str_representation_connected(self, mock_validate_sync):
        """Test string representation when connected."""
        # Mock connection pool with connection kwargs
        mock_pool = Mock()
        mock_pool.connection_kwargs = {'host': 'localhost', 'port': 6379}
        self.mock_redis_client.connection_pool = mock_pool
        self.mock_redis_client.ping.return_value = True

        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        result = str(store)
        expected = (f"RedisVectorStore(index='{self.index_name}', "
                    f"dimensions={self.dimensions}, "
                    f"host=localhost, "
                    f"port=6379, "
                    f"status=connected)")
        self.assertEqual(result, expected)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_str_representation_disconnected(self, mock_validate_sync):
        """Test string representation when disconnected."""
        # Mock connection pool with connection kwargs
        mock_pool = Mock()
        mock_pool.connection_kwargs = {'host': 'localhost', 'port': 6379}

        # Configure mock to allow initialization but fail during str()
        self.mock_redis_client.connection_pool = mock_pool
        self.mock_redis_client.ping.return_value = True  # Allow init to succeed

        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        # Now set ping to fail for the str() method test
        self.mock_redis_client.ping.side_effect = redis.RedisError(
            "Connection failed")

        result = str(store)
        expected = (f"RedisVectorStore(index='{self.index_name}', "
                    f"dimensions={self.dimensions}, "
                    f"host=localhost, "
                    f"port=6379, "
                    f"status=disconnected)")
        self.assertEqual(result, expected)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_str_representation_unknown(self, mock_validate_sync):
        """Test string representation with unknown connection info."""
        # Create a mock client that will work with RedisVL but not have connection_pool
        mock_client = Mock(spec=redis.Redis)
        mock_client.ping.return_value = True

        # Mock the index creation process
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            with patch('ragl.store.redis.SearchIndex') as mock_index_class:
                with patch(
                    'ragl.store.redis.IndexSchema') as mock_schema_class:
                    # Configure mocks
                    mock_schema_class.from_dict.return_value = Mock()
                    mock_index = Mock()
                    mock_index.exists.return_value = False
                    mock_index.create.return_value = None
                    mock_index_class.return_value = mock_index

                    store = RedisVectorStore(
                        redis_client=mock_client,
                        dimensions=self.dimensions,
                        index_name=self.index_name
                    )

                    # Remove connection_pool after initialization
                    del mock_client.connection_pool

        result = str(store)
        expected = (f"RedisVectorStore(index='{self.index_name}', "
                    f"dimensions={self.dimensions}, "
                    f"host=unknown, "
                    f"port=unknown, "
                    f"status=unknown)")
        self.assertEqual(result, expected)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_repr_representation(self, mock_validate_sync):
        """Test repr representation."""
        # Mock connection pool with connection kwargs
        mock_pool = Mock()
        mock_pool.connection_kwargs = {'host': 'localhost', 'port': 6379}
        self.mock_redis_client.connection_pool = mock_pool

        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        result = repr(store)
        # The repr should contain class name and key attributes
        self.assertIn('RedisVectorStore', result)
        self.assertIn(str(self.dimensions), result)
        self.assertIn(self.index_name, result)

    def test_repr_with_missing_connection_pool(self):
        """Test __repr__ when redis_client has no connection_pool attribute."""
        # Create a mock redis client without connection_pool attribute
        mock_client = Mock(spec=[])  # Empty spec means no attributes

        # Replace the redis_client temporarily
        original_client = self.store.redis_client
        self.store.redis_client = mock_client

        try:
            result = repr(self.store)

            # Should fall back to 'unknown' values
            self.assertIn("host='unknown'", result)
            self.assertIn("port=unknown", result)
            self.assertIn("db=unknown", result)
            self.assertIn("max_connections=unknown", result)
            self.assertIn(f"index_name='{self.store.index_name}'", result)
            self.assertIn(f"dimensions={self.store.dimensions}", result)
        finally:
            # Restore original client
            self.store.redis_client = original_client

    def test_repr_with_connection_pool_none(self):
        """Test __repr__ when connection_pool is None."""
        # Create a mock redis client with connection_pool = None
        mock_client = Mock()
        mock_client.connection_pool = None

        original_client = self.store.redis_client
        self.store.redis_client = mock_client

        try:
            result = repr(self.store)

            # Should fall back to 'unknown' values via else block
            self.assertIn("host='unknown'", result)
            self.assertIn("port=unknown", result)
            self.assertIn("db=unknown", result)
            self.assertIn("max_connections=unknown", result)
        finally:
            self.store.redis_client = original_client

    @patch('ragl.store.redis.redis.Redis')
    @patch('ragl.store.redis.SearchIndex')
    @patch('ragl.store.redis.IndexSchema')
    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_repr_with_attribute_error(self, mock_validate, mock_schema_class,
                                       mock_index_class, mock_redis_class):
        """Test __repr__ when AttributeError is raised during inspection."""
        # Create a mock connection pool with connection_kwargs as a string (not dict)
        mock_connection_pool = Mock()
        mock_connection_pool.connection_kwargs = "not_a_dict"  # This will cause AttributeError on .get()
        mock_connection_pool.max_connections = 50

        # Create a mock Redis client
        mock_redis_client = Mock()
        mock_redis_client.connection_pool = mock_connection_pool
        mock_redis_client.ping.return_value = True
        mock_redis_client.get.return_value = None  # For schema version check
        mock_redis_client.set.return_value = True

        # Configure other mocks
        mock_schema_class.from_dict.return_value = Mock()
        mock_index = Mock()
        mock_index.exists.return_value = False
        mock_index.create.return_value = None
        mock_index_class.return_value = mock_index

        # Create store with the mock client
        store = RedisVectorStore(
            redis_client=mock_redis_client,
            dimensions=768,
            index_name="test_index"
        )

        # Call __repr__ - should handle AttributeError gracefully
        repr_str = repr(store)

        # Verify the repr contains expected fallback values
        self.assertIn("RedisVectorStore(", repr_str)
        self.assertIn("index_name='test_index'", repr_str)
        self.assertIn("dimensions=768", repr_str)
        self.assertIn("host='unknown'", repr_str)
        self.assertIn("port=unknown", repr_str)
        self.assertIn("db=unknown", repr_str)
        self.assertIn("max_connections=unknown", repr_str)
        self.assertIn(f"schema_version={store.SCHEMA_VERSION}", repr_str)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_no_redis_config_or_client(self, mock_validate):
        """Test initialization with neither redis_config nor redis_client."""
        with self.assertRaises(ConfigurationError) as context:
            RedisVectorStore(
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        self.assertIn("Either redis_client or redis_config must be provided",
                      str(context.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_both_redis_config_and_client(self, mock_validate):
        """Test initialization with both redis_config and redis_client."""
        with self.assertRaises(ConfigurationError) as context:
            RedisVectorStore(
                redis_config=self.redis_config,
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        self.assertIn("Both redis_client and redis_config were provided", str(context.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_invalid_dimensions_zero(self, mock_validate):
        """Test initialization with zero dimensions."""
        with self.assertRaises(ConfigurationError) as context:
            RedisVectorStore(
                redis_config=self.redis_config,
                dimensions=0,
                index_name=self.index_name
            )

        self.assertIn("Dimensions must be positive",
                      str(context.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_invalid_dimensions_negative(self, mock_validate):
        """Test initialization with negative dimensions."""
        with self.assertRaises(ConfigurationError) as context:
            RedisVectorStore(
                redis_config=self.redis_config,
                dimensions=-100,
                index_name=self.index_name
            )

        self.assertIn("Dimensions must be positive",
                      str(context.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_invalid_dimensions_float(self, mock_validate):
        """Test initialization with float dimensions."""
        with self.assertRaises(ConfigurationError) as context:
            RedisVectorStore(
                redis_config=self.redis_config,
                dimensions=768.5,
                index_name=self.index_name
            )

        self.assertIn("Dimensions must be positive",
                      str(context.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_invalid_dimensions_none(self, mock_validate):
        """Test initialization with float dimensions."""
        with self.assertRaises(ConfigurationError) as context:
            RedisVectorStore(
                redis_config=self.redis_config,
                dimensions=None,
                index_name=self.index_name
            )

        self.assertIn("Dimensions required for Schema creation",
                      str(context.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_empty_index_name(self, mock_validate):
        """Test initialization with empty index name."""
        with self.assertRaises(ConfigurationError) as context:
            RedisVectorStore(
                redis_config=self.redis_config,
                dimensions=self.dimensions,
                index_name=""
            )

        self.assertIn("index_name cannot be empty", str(context.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_whitespace_only_index_name(self, mock_validate):
        """Test initialization with whitespace-only index name."""
        with self.assertRaises(ConfigurationError) as context:
            RedisVectorStore(
                redis_config=self.redis_config,
                dimensions=self.dimensions,
                index_name="   \n\t   "
            )

        self.assertIn("index_name cannot be empty", str(context.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_none_index_name(self, mock_validate):
        """Test initialization with None index name."""
        with self.assertRaises(ConfigurationError) as context:
            RedisVectorStore(
                redis_config=self.redis_config,
                dimensions=self.dimensions,
                index_name=None
            )

        self.assertIn("index_name cannot be empty", str(context.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_long_index_name(self, mock_validate):
        """Test initialization with None index name."""
        max_index_len = RedisVectorStore.MAX_TEXT_ID_LENGTH
        with self.assertRaises(ConfigurationError) as context:
            RedisVectorStore(
                redis_config=self.redis_config,
                dimensions=self.dimensions,
                index_name=''.join(str(i % 10) for i in range(max_index_len + 1)),
            )

        self.assertIn("index_name too long: 257 > 256", str(context.exception))

    @patch('ragl.store.redis.redis.Redis')
    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_redis_connection_error(self, mock_validate,
                                         mock_redis_class):
        """Test initialization with Redis connection error."""
        mock_redis_client = Mock()
        mock_redis_client.ping.side_effect = redis.ConnectionError(
            "Connection failed")
        mock_redis_class.return_value = mock_redis_client

        with self.assertRaises(StorageConnectionError) as context:
            RedisVectorStore(
                redis_config=self.redis_config,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        self.assertIn("Redis connection failed: Connection failed", str(context.exception))

    @patch('ragl.store.redis.redis.Redis')
    @patch('ragl.store.redis.SearchIndex')
    @patch('ragl.store.redis.IndexSchema')
    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_index_creation_error(self, mock_validate, mock_schema_class,
                                       mock_index_class, mock_redis_class):
        """Test initialization with index creation error."""
        mock_redis_class.return_value = self.mock_redis_client
        mock_schema_class.from_dict.return_value = self.mock_schema

        mock_index = Mock()
        mock_index.exists.return_value = False
        mock_index.create.side_effect = redis.ResponseError(
            "Index creation failed")
        mock_index_class.return_value = mock_index

        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            with self.assertRaises(DataError) as context:
                RedisVectorStore(
                    redis_config=self.redis_config,
                    dimensions=self.dimensions,
                    index_name=self.index_name,
                )

        self.assertIn("Failed to create Redis index", str(context.exception))

    @patch('ragl.store.redis.redis.Redis')
    @patch('ragl.store.redis.SearchIndex')
    @patch('ragl.store.redis.IndexSchema')
    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_index_creation_connection_error(self, mock_validate,
                                                  mock_schema_class,
                                                  mock_index_class,
                                                  mock_redis_class):
        """Test initialization with index creation connection error."""
        mock_redis_class.return_value = self.mock_redis_client
        mock_schema_class.from_dict.return_value = self.mock_schema

        mock_index = Mock()
        mock_index.exists.return_value = False
        mock_index.create.side_effect = redis.ConnectionError(
            "Connection lost")
        mock_index_class.return_value = mock_index

        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            with self.assertRaises(StorageConnectionError) as context:
                RedisVectorStore(
                    redis_config=self.redis_config,
                    dimensions=self.dimensions,
                    index_name=self.index_name
                )

        self.assertIn("Failed to connect to Redis: Connection lost",
                      str(context.exception))

    def test_init_handles_unexpected_exception_during_index_creation(self):
        """Test that unexpected exceptions during index creation are handled properly."""

        # Mock Redis client that connects successfully
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None  # No existing schema version
        mock_client.set.return_value = True  # Schema version set successfully

        # Mock SearchIndex and IndexSchema
        with patch('ragl.store.redis.SearchIndex') as mock_search_index_class:
            with patch('ragl.store.redis.IndexSchema') as mock_schema_class:
                # Configure schema mock
                mock_schema_class.from_dict.return_value = MagicMock()

                # Configure index mock to raise unexpected exception during exists() check
                mock_index = MagicMock()
                mock_index.exists.side_effect = RuntimeError(
                    "Unexpected error during index operation")
                mock_search_index_class.return_value = mock_index

                # Should raise DataError wrapping the unexpected exception
                with self.assertRaises(DataError) as context:
                    RedisVectorStore(
                        redis_client=mock_client,
                        dimensions=768,
                        index_name='test_index'
                    )

                self.assertIn('Unexpected error creating index',
                              str(context.exception))

    def test_validate_input_sizes_metadata_key_too_large(self):
        """Test that _validate_input_sizes raises ValidationError when metadata key exceeds MAX_FIELD_SIZE."""
        # Create a key that exceeds MAX_FIELD_SIZE (32MB)
        large_key = "x" * (RedisVectorStore.MAX_FIELD_SIZE + 1)
        metadata = {large_key: "small_value"}

        with self.assertRaises(ValidationError) as cm:
            self.store._validate_input_sizes("test text", metadata)

        self.assertIn("Metadata key too large", str(cm.exception))
        self.assertIn(large_key[:50],
                      str(cm.exception))  # Check partial key is in error

    def test_validate_input_sizes_total_metadata_too_large(self):
        """Test that _validate_input_sizes raises ValidationError when total metadata exceeds MAX_METADATA_SIZE."""
        # Create metadata that totals more than MAX_METADATA_SIZE (64MB)
        # Use multiple keys and values that together exceed the limit
        large_value = "x" * (RedisVectorStore.MAX_METADATA_SIZE // 2)
        metadata = {
            "key1": large_value,
            "key2": large_value,
            "key3": "extra"  # This pushes it over the limit
        }

        with self.assertRaises(ValidationError) as cm:
            self.store._validate_input_sizes("test text", metadata)

        self.assertIn("Total metadata too large", str(cm.exception))

    def test_validate_input_sizes_metadata_key_at_limit(self):
        """Test that _validate_input_sizes accepts metadata key exactly at MAX_FIELD_SIZE."""
        # Create a key exactly at MAX_FIELD_SIZE
        key_at_limit = "x" * RedisVectorStore.MAX_FIELD_SIZE
        metadata = {key_at_limit: "small_value"}

        # Should not raise an exception
        try:
            self.store._validate_input_sizes("test text", metadata)
        except ValidationError:
            self.fail(
                "_validate_input_sizes raised ValidationError for key at size limit")

    def test_validate_input_sizes_total_metadata_at_limit(self):
        """Test that _validate_input_sizes accepts total metadata exactly at MAX_METADATA_SIZE."""
        # Create metadata that totals exactly MAX_METADATA_SIZE
        # Use multiple keys/values that individually don't exceed MAX_FIELD_SIZE
        # but together total exactly MAX_METADATA_SIZE

        # Create two fields that together equal MAX_METADATA_SIZE
        key1 = "key1"  # 4 bytes
        key2 = "key2"  # 4 bytes

        # Split the remaining bytes between two values
        remaining_bytes = RedisVectorStore.MAX_METADATA_SIZE - len(
            key1.encode('utf-8')) - len(key2.encode('utf-8'))
        value1_size = remaining_bytes // 2
        value2_size = remaining_bytes - value1_size

        value1 = "x" * value1_size
        value2 = "y" * value2_size

        metadata = {key1: value1, key2: value2}

        # Should not raise an exception
        try:
            self.store._validate_input_sizes("test text", metadata)
        except ValidationError:
            self.fail(
                "_validate_input_sizes raised ValidationError for metadata at size limit")

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_parse_tags_from_retrieval_splits_tags_with_separator(self, mock_validate_sync):
        """Test that tags containing TAG_SEPARATOR are properly split."""
        store = RedisVectorStore(
            redis_config=self.redis_config,
            dimensions=768,
            index_name='test_index'
        )

        # Test with a list containing a tag with separator
        tags_with_separator = ['tag1,tag2,tag3', 'single_tag']
        result = store._parse_tags_from_retrieval(tags_with_separator)

        # Should split the first element and keep the second as-is
        expected = ['tag1', 'tag2', 'tag3', 'single_tag']
        self.assertEqual(result, expected)

        # Test with multiple tags containing separators
        complex_tags = ['alpha,beta', 'gamma,delta,epsilon', 'zeta']
        result = store._parse_tags_from_retrieval(complex_tags)

        expected = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
        self.assertEqual(result, expected)

    @patch('ragl.store.redis.redis.Redis')
    @patch('ragl.store.redis.SearchIndex')
    @patch('ragl.store.redis.IndexSchema')
    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_schema_version_enforcement_error(self, mock_validate,
                                                   mock_schema_class,
                                                   mock_index_class,
                                                   mock_redis_class):
        """Test initialization with schema version enforcement error."""
        mock_redis_class.return_value = self.mock_redis_client
        mock_schema_class.from_dict.return_value = self.mock_schema
        mock_index_class.return_value = self.mock_index

        with patch.object(RedisVectorStore,
                          '_enforce_schema_version') as mock_enforce:
            mock_enforce.side_effect = ConfigurationError(
                "Schema version mismatch")

            with self.assertRaises(ConfigurationError) as context:
                RedisVectorStore(
                    redis_config=self.redis_config,
                    dimensions=self.dimensions,
                    index_name=self.index_name
                )

            self.assertIn("Schema version mismatch", str(context.exception))

    @patch('ragl.store.redis.redis.Redis')
    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_invalid_redis_config_type(self, mock_validate,
                                            mock_redis_class):
        """Test initialization with invalid redis_config type."""
        with self.assertRaises(ConfigurationError) as context:
            RedisVectorStore(
                redis_config="invalid_config",  # Should be RedisConfig object
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        self.assertIn("redis_config must be an instance of RedisConfig",
                      str(context.exception))

    @patch('ragl.store.redis.redis.Redis')
    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_init_with_injected_redis_client_connection_error(self, mock_validate, mock_redis_class):
        """Test initialization with injected Redis client that fails ping."""
        # Create a mock Redis client that raises ConnectionError on ping
        mock_redis_client = Mock()
        mock_redis_client.ping.side_effect = redis.ConnectionError(
            "Connection failed")

        # Attempt to initialize RedisVectorStore with the failing client
        with self.assertRaises(StorageConnectionError) as context:
            RedisVectorStore(
                redis_client=mock_redis_client,
                dimensions=768,
                index_name="test_index"
            )

        # Verify the error message contains expected information
        self.assertIn("Injected Redis client connection failed",
                      str(context.exception))
        self.assertIn("Connection failed", str(context.exception))

        # Verify ping was called
        mock_redis_client.ping.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_initialize_redis_with_client_success(self, mock_validate_sync):
        """Test _initialize_redis with valid Redis client."""
        mock_client = Mock(spec=redis.Redis)
        mock_client.ping.return_value = True

        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        result = store._initialize_redis(mock_client, None)

        self.assertEqual(result, mock_client)
        mock_client.ping.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_initialize_redis_with_client_connection_error(self,
                                                           mock_validate_sync):
        """Test _initialize_redis with Redis client connection failure."""
        mock_client = Mock(spec=redis.Redis)
        mock_client.ping.side_effect = redis.ConnectionError(
            "Connection failed")

        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with self.assertRaises(StorageConnectionError) as cm:
            store._initialize_redis(mock_client, None)

        self.assertIn("Injected Redis client connection failed",
                      str(cm.exception))

    @patch('redis.Redis')
    @patch('redis.BlockingConnectionPool')
    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_initialize_redis_with_config_success(self, mock_validate_sync,
                                                  mock_pool_class,
                                                  mock_redis_class):
        """Test _initialize_redis with valid Redis config."""
        mock_config = RedisConfig(host='localhost', port=6379)
        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool

        mock_redis_instance = Mock()
        mock_redis_class.return_value = mock_redis_instance

        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        result = store._initialize_redis(None, mock_config)

        self.assertEqual(result, mock_redis_instance)
        mock_pool_class.assert_called_once()
        mock_redis_class.assert_called_once_with(connection_pool=mock_pool)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_initialize_redis_no_client_or_config(self, mock_validate_sync):
        """Test _initialize_redis with neither client nor config provided."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with self.assertRaises(ConfigurationError) as cm:
            store._initialize_redis(None, None)

        self.assertIn("Either redis_client or redis_config must be provided",
                      str(cm.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_initialize_redis_both_client_and_config(self, mock_validate_sync):
        """Test _initialize_redis with both client and config provided."""
        mock_client = Mock(spec=redis.Redis)
        mock_config = RedisConfig(host='localhost', port=6379)

        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with self.assertRaises(ConfigurationError) as cm:
            store._initialize_redis(mock_client, mock_config)

        self.assertIn("Both redis_client and redis_config were provided",
                      str(cm.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_initialize_redis_invalid_config_type(self, mock_validate_sync):
        """Test _initialize_redis with invalid config type."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with self.assertRaises(ConfigurationError) as cm:
            store._initialize_redis(None, "invalid_config")

        self.assertIn("redis_config must be an instance of RedisConfig",
                      str(cm.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_create_validation_schema(self, mock_validate_sync):
        """Test _create_validation_schema returns correct schema structure."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        schema = store._create_validation_schema()

        # Verify schema structure
        self.assertIsInstance(schema, dict)

        # Check required fields exist
        expected_fields = [
            'chunk_position', 'timestamp', 'confidence', 'tags',
            'parent_id', 'source', 'language', 'section', 'author'
        ]
        for field in expected_fields:
            self.assertIn(field, schema)

        # Check specific field types and defaults
        self.assertEqual(schema['chunk_position']['type'], int)
        self.assertEqual(schema['chunk_position']['default'], 0)
        self.assertEqual(schema['timestamp']['type'], int)
        self.assertEqual(schema['timestamp']['default'], 0)
        self.assertEqual(schema['confidence']['type'], float)
        self.assertEqual(schema['confidence']['default'], 0.0)
        self.assertEqual(schema['tags']['type'], str)
        self.assertEqual(schema['tags']['default'], '')

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_initialize_search_index_create_new(self, mock_validate_sync):
        """Test _initialize_search_index creating new index."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with patch.object(store.index, 'exists', return_value=False), \
            patch.object(store.index, 'create') as mock_create:
            store._initialize_search_index()

            mock_create.assert_called_once()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_initialize_search_index_existing(self, mock_validate_sync):
        """Test _initialize_search_index with existing index."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with patch.object(store.index, 'exists', return_value=True), \
            patch.object(store.index, 'create') as mock_create:
            store._initialize_search_index()

            mock_create.assert_not_called()

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_initialize_search_index_response_error(self, mock_validate_sync):
        """Test _initialize_search_index with Redis response error."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with patch.object(store.index, 'exists', return_value=False), \
            patch.object(store.index, 'create',
                         side_effect=redis.ResponseError(
                             "Index creation failed")):
            with self.assertRaises(DataError) as cm:
                store._initialize_search_index()

            self.assertIn("Failed to create Redis index", str(cm.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_initialize_search_index_connection_error(self,
                                                      mock_validate_sync):
        """Test _initialize_search_index with connection error."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with patch.object(store.index, 'exists',
                          side_effect=redis.ConnectionError(
                              "Connection failed")):
            with self.assertRaises(StorageConnectionError) as cm:
                store._initialize_search_index()

            self.assertIn("Failed to connect to Redis", str(cm.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_initialize_search_index_unexpected_error(self,
                                                      mock_validate_sync):
        """Test _initialize_search_index with unexpected error."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        with patch.object(store.index, 'exists',
                          side_effect=Exception("Unexpected error")):
            with self.assertRaises(DataError) as cm:
                store._initialize_search_index()

            self.assertIn("Unexpected error creating index", str(cm.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_validate_and_prepare_batch_data_success(self, mock_validate_sync):
        """Test successful validation and preparation of batch data."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_unit1 = TextUnit(text="Sample text 1",
                              text_id=f'{TEXT_ID_PREFIX}1', distance=0.0)
        text_unit2 = TextUnit(text="Sample text 2", text_id=None, distance=0.0)
        embedding1 = np.random.rand(self.dimensions).astype(np.float32)
        embedding2 = np.random.rand(self.dimensions).astype(np.float32)

        texts_and_embeddings = [(text_unit1, embedding1),
                                (text_unit2, embedding2)]

        self.mock_redis_client.incr.return_value = 123

        with patch('ragl.store.redis.sanitize_metadata', return_value={}):
            result = store._validate_and_prepare_batch_data(
                texts_and_embeddings)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertIn(f'{TEXT_ID_PREFIX}1', result)
        self.assertIn(f'{TEXT_ID_PREFIX}123', result)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_validate_and_prepare_batch_data_invalid_structure(self,
                                                               mock_validate_sync):
        """Test batch data validation with invalid structure."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        # Test with non-list input
        with self.assertRaises(ValidationError):
            store._validate_and_prepare_batch_data("not_a_list")

        # Test with invalid tuple structure
        invalid_pairs = [("not_a_text_unit", "not_an_embedding")]
        with self.assertRaises(ValidationError):
            store._validate_and_prepare_batch_data(invalid_pairs)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_prepare_single_text_entry_success(self, mock_validate_sync):
        """Test successful preparation of single text entry."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_unit = TextUnit(text="Sample text",
                             text_id=f'{TEXT_ID_PREFIX}123', distance=0.0)
        embedding = np.random.rand(self.dimensions).astype(np.float32)

        with patch('ragl.store.redis.sanitize_metadata',
                   return_value={'source': 'test'}):
            text_id, prepared_data = store._prepare_single_text_entry(
                text_unit, embedding)

        self.assertEqual(text_id, f'{TEXT_ID_PREFIX}123')
        self.assertIsInstance(prepared_data, dict)
        self.assertEqual(prepared_data['text'], "Sample text")
        self.assertIsInstance(prepared_data['embedding'], bytes)
        self.assertEqual(prepared_data['source'], 'test')

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_prepare_single_text_entry_auto_id(self, mock_validate_sync):
        """Test preparation of single text entry with auto-generated ID."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_unit = TextUnit(text="Sample text", text_id=None, distance=0.0)
        embedding = np.random.rand(self.dimensions).astype(np.float32)

        self.mock_redis_client.incr.return_value = 456

        with patch('ragl.store.redis.sanitize_metadata', return_value={}):
            text_id, prepared_data = store._prepare_single_text_entry(
                text_unit, embedding)

        self.assertEqual(text_id, f'{TEXT_ID_PREFIX}456')
        self.assertEqual(text_unit.text_id,
                         f'{TEXT_ID_PREFIX}456')  # Should be set on the unit

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_prepare_single_text_entry_empty_text(self, mock_validate_sync):
        """Test preparation with empty text raises ValidationError."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        embedding = np.random.rand(self.dimensions).astype(np.float32)

        with self.assertRaises(ValidationError) as cm:
            text_unit = TextUnit(text="", text_id=None, distance=0.0)
            store._prepare_single_text_entry(text_unit, embedding)

        self.assertIn('text cannot be whitespace-only or zero-length', str(cm.exception))

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_prepare_single_text_entry_with_tags(self, mock_validate_sync):
        """Test preparation of single text entry with tags."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_unit = TextUnit(text="Sample text",
                             text_id=f'{TEXT_ID_PREFIX}123', distance=0.0)
        embedding = np.random.rand(self.dimensions).astype(np.float32)

        with patch('ragl.store.redis.sanitize_metadata',
                   return_value={'tags': ['tag1', 'tag2']}):
            text_id, prepared_data = store._prepare_single_text_entry(
                text_unit, embedding)

        self.assertEqual(prepared_data['tags'], 'tag1,tag2')

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_prepare_single_text_entry_invalid_dimensions(self,
                                                          mock_validate_sync):
        """Test preparation with invalid embedding dimensions."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        text_unit = TextUnit(text="Sample text", text_id=None, distance=0.0)
        wrong_embedding = np.random.rand(512).astype(
            np.float32)  # Wrong dimensions

        with self.assertRaises(ConfigurationError):
            store._prepare_single_text_entry(text_unit, wrong_embedding)

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_execute_batch_storage_success(self, mock_validate_sync):
        """Test successful execution of batch storage."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        batch_data = {
            f'{TEXT_ID_PREFIX}1': {'text': 'text1', 'embedding': b'data1'},
            f'{TEXT_ID_PREFIX}2': {'text': 'text2', 'embedding': b'data2'}
        }

        expected_ids = [f'{TEXT_ID_PREFIX}1', f'{TEXT_ID_PREFIX}2']
        self.mock_index.load.return_value = expected_ids

        with patch.object(store, 'index', self.mock_index):
            result = store._execute_batch_storage(batch_data)

        self.assertEqual(result, expected_ids)
        self.mock_index.load.assert_called_once_with(
            data=list(batch_data.values()),
            keys=list(batch_data.keys())
        )

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_execute_batch_storage_empty_batch(self, mock_validate_sync):
        """Test execution of batch storage with empty batch."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        batch_data = {}
        self.mock_index.load.return_value = []

        with patch.object(store, 'index', self.mock_index):
            result = store._execute_batch_storage(batch_data)

        self.assertEqual(result, [])
        self.mock_index.load.assert_called_once_with(data=[], keys=[])

    @patch('redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis')
    def test_execute_batch_storage_redis_error(self, mock_validate_sync):
        """Test batch storage execution with Redis error."""
        with patch.object(RedisVectorStore, '_enforce_schema_version'):
            store = RedisVectorStore(
                redis_client=self.mock_redis_client,
                dimensions=self.dimensions,
                index_name=self.index_name
            )

        batch_data = {
            f'{TEXT_ID_PREFIX}1': {'text': 'text1', 'embedding': b'data1'}
        }

        with patch.object(store, 'index', self.mock_index):
            self.mock_index.load.side_effect = redis.ResponseError(
                "Redis error")

            with self.assertRaises(DataError):
                store._execute_batch_storage(batch_data)


if __name__ == '__main__':
    unittest.main()
