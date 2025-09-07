from dataclasses import dataclass
import unittest

from ragl.config import (
    RaglConfig,
    EmbedderConfig,
    VectorStoreConfig,
    SentenceTransformerConfig,
    RedisConfig,
    ManagerConfig
)
from ragl.exceptions import ConfigurationError


class TestRaglConfig(unittest.TestCase):
    """Test cases for RaglConfig base class."""

    def test_str_representation(self):
        """Test __str__ method with different field types."""
        config = RaglConfig()
        expected = "RaglConfig()"
        self.assertEqual(str(config), expected)

    def test_str_representation_with_fields(self):
        """Test __str__ with string and non-string fields."""

        # Create a subclass to test with fields
        @dataclass
        class TestConfig(RaglConfig):
            name: str = "test"
            count: int = 42
            flag: bool = True

        config = TestConfig()
        result = str(config)
        self.assertIn('name="test"', result)
        self.assertIn('count=42', result)
        self.assertIn('flag=True', result)
        self.assertTrue(result.startswith('TestConfig('))
        self.assertTrue(result.endswith(')'))

    def test_repr_representation(self):
        """Test __repr__ method returns same as __str__."""
        config = RaglConfig()
        self.assertEqual(repr(config), str(config))


class TestEmbedderConfig(unittest.TestCase):
    """Test cases for EmbedderConfig base class."""

    def test_inheritance(self):
        """Test that EmbedderConfig inherits from RaglConfig."""
        config = EmbedderConfig()
        self.assertIsInstance(config, RaglConfig)

    def test_str_representation(self):
        """Test string representation."""
        config = EmbedderConfig()
        expected = "EmbedderConfig()"
        self.assertEqual(str(config), expected)


class TestVectorStoreConfig(unittest.TestCase):
    """Test cases for VectorStoreConfig base class."""

    def test_inheritance(self):
        """Test that VectorStoreConfig inherits from RaglConfig."""
        config = VectorStoreConfig()
        self.assertIsInstance(config, RaglConfig)

    def test_str_representation(self):
        """Test string representation."""
        config = VectorStoreConfig()
        expected = "VectorStoreConfig()"
        self.assertEqual(str(config), expected)


class TestSentenceTransformerConfig(unittest.TestCase):
    """Test cases for SentenceTransformerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SentenceTransformerConfig()
        self.assertEqual(config.model_name_or_path, 'all-mpnet-base-v2')
        self.assertEqual(config.cache_maxsize, 10_000)
        self.assertIsNone(config.device)
        self.assertTrue(config.auto_clear_cache)
        self.assertEqual(config.memory_threshold, 0.9)

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = SentenceTransformerConfig(
            model_name_or_path='custom-model',
            cache_maxsize=5000,
            device='cuda',
            auto_clear_cache=False,
            memory_threshold=0.8
        )
        self.assertEqual(config.model_name_or_path, 'custom-model')
        self.assertEqual(config.cache_maxsize, 5000)
        self.assertEqual(config.device, 'cuda')
        self.assertFalse(config.auto_clear_cache)
        self.assertEqual(config.memory_threshold, 0.8)

    def test_validate_model_name_empty_string(self):
        """Test validation fails for empty model name."""
        with self.assertRaises(ConfigurationError) as cm:
            SentenceTransformerConfig(model_name_or_path='')
        self.assertIn('model_name_or_path cannot be empty', str(cm.exception))

    def test_validate_model_name_whitespace_only(self):
        """Test validation fails for whitespace-only model name."""
        with self.assertRaises(ConfigurationError) as cm:
            SentenceTransformerConfig(model_name_or_path='   ')
        self.assertIn('model_name_or_path cannot be empty', str(cm.exception))

    def test_validate_cache_maxsize_negative(self):
        """Test validation fails for negative cache size."""
        with self.assertRaises(ConfigurationError) as cm:
            SentenceTransformerConfig(cache_maxsize=-1)
        self.assertIn('cache_maxsize must be a non-negative integer', str(cm.exception))

    def test_validate_cache_maxsize_non_integer(self):
        """Test validation fails for non-integer cache size."""
        with self.assertRaises(ConfigurationError) as cm:
            SentenceTransformerConfig(cache_maxsize=10.5)
        self.assertIn('cache_maxsize must be a non-negative integer', str(cm.exception))

    def test_validate_cache_maxsize_zero(self):
        """Test validation passes for zero cache size."""
        config = SentenceTransformerConfig(cache_maxsize=0)
        self.assertEqual(config.cache_maxsize, 0)

    def test_validate_memory_threshold_below_range(self):
        """Test validation fails for memory threshold below 0.0."""
        with self.assertRaises(ConfigurationError) as cm:
            SentenceTransformerConfig(memory_threshold=-0.1)
        self.assertIn('must be between 0.0 and 1.0', str(cm.exception))

    def test_validate_memory_threshold_above_range(self):
        """Test validation fails for memory threshold above 1.0."""
        with self.assertRaises(ConfigurationError) as cm:
            SentenceTransformerConfig(memory_threshold=1.1)
        self.assertIn('must be between 0.0 and 1.0', str(cm.exception))

    def test_validate_memory_threshold_boundary_values(self):
        """Test validation passes for boundary values."""
        config1 = SentenceTransformerConfig(memory_threshold=0.0)
        self.assertEqual(config1.memory_threshold, 0.0)

        config2 = SentenceTransformerConfig(memory_threshold=1.0)
        self.assertEqual(config2.memory_threshold, 1.0)

    def test_inheritance(self):
        """Test that SentenceTransformerConfig inherits from EmbedderConfig."""
        config = SentenceTransformerConfig()
        self.assertIsInstance(config, EmbedderConfig)
        self.assertIsInstance(config, RaglConfig)


class TestRedisConfig(unittest.TestCase):
    """Test cases for RedisConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RedisConfig()
        self.assertEqual(config.host, 'localhost')
        self.assertEqual(config.port, 6379)
        self.assertEqual(config.db, 0)
        self.assertEqual(config.socket_timeout, 5)
        self.assertEqual(config.socket_connect_timeout, 5)
        self.assertEqual(config.max_connections, 50)
        self.assertEqual(config.health_check_interval, 30)
        self.assertTrue(config.decode_responses)
        self.assertTrue(config.retry_on_timeout)

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = RedisConfig(
            host='redis.example.com',
            port=6380,
            db=1,
            socket_timeout=10,
            socket_connect_timeout=15,
            max_connections=100,
            health_check_interval=60,
            decode_responses=False,
            retry_on_timeout=False
        )
        self.assertEqual(config.host, 'redis.example.com')
        self.assertEqual(config.port, 6380)
        self.assertEqual(config.db, 1)
        self.assertEqual(config.socket_timeout, 10)
        self.assertEqual(config.socket_connect_timeout, 15)
        self.assertEqual(config.max_connections, 100)
        self.assertEqual(config.health_check_interval, 60)
        self.assertFalse(config.decode_responses)
        self.assertFalse(config.retry_on_timeout)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = RedisConfig()
        result = config.to_dict()
        expected = {
            'host':                   'localhost',
            'port':                   6379,
            'db':                     0,
            'socket_timeout':         5,
            'socket_connect_timeout': 5,
            'max_connections':        50,
            'health_check_interval':  30,
            'decode_responses':       True,
            'retry_on_timeout':       True,
        }
        self.assertEqual(result, expected)

    def test_validate_host_empty(self):
        """Test validation fails for empty host."""
        with self.assertRaises(ConfigurationError) as cm:
            RedisConfig(host='')
        self.assertIn('host cannot be empty', str(cm.exception))

    def test_validate_host_whitespace_only(self):
        """Test validation fails for whitespace-only host."""
        with self.assertRaises(ConfigurationError) as cm:
            RedisConfig(host='   ')
        self.assertIn('host cannot be empty', str(cm.exception))

    def test_validate_port_below_range(self):
        """Test validation fails for port below 1."""
        with self.assertRaises(ConfigurationError) as cm:
            RedisConfig(port=0)
        self.assertIn('must be between 1-65535', str(cm.exception))

    def test_validate_port_above_range(self):
        """Test validation fails for port above 65535."""
        with self.assertRaises(ConfigurationError) as cm:
            RedisConfig(port=65536)
        self.assertIn('must be between 1-65535', str(cm.exception))

    def test_validate_port_non_integer(self):
        """Test validation fails for non-integer port."""
        with self.assertRaises(ConfigurationError) as cm:
            RedisConfig(port=6379.5)
        self.assertIn('must be between 1-65535', str(cm.exception))

    def test_validate_port_boundary_values(self):
        """Test validation passes for boundary port values."""
        config1 = RedisConfig(port=1)
        self.assertEqual(config1.port, 1)

        config2 = RedisConfig(port=65535)
        self.assertEqual(config2.port, 65535)

    def test_validate_db_negative(self):
        """Test validation fails for negative db."""
        with self.assertRaises(ConfigurationError) as cm:
            RedisConfig(db=-1)
        self.assertIn('db must be a non-negative integer', str(cm.exception))

    def test_validate_db_non_integer(self):
        """Test validation fails for non-integer db."""
        with self.assertRaises(ConfigurationError) as cm:
            RedisConfig(db=1.5)
        self.assertIn('db must be a non-negative integer', str(cm.exception))

    def test_validate_socket_timeout_negative(self):
        """Test validation fails for negative socket timeout."""
        with self.assertRaises(ConfigurationError) as cm:
            RedisConfig(socket_timeout=0)
        self.assertIn('socket_timeout must be a positive integer', str(cm.exception))

    def test_validate_socket_connect_timeout_negative(self):
        """Test validation fails for negative socket connect timeout."""
        with self.assertRaises(ConfigurationError) as cm:
            RedisConfig(socket_connect_timeout=0)
        self.assertIn('socket_connect_timeout must be a positive integer', str(cm.exception))

    def test_validate_health_check_interval_negative(self):
        """Test validation fails for negative health check interval."""
        with self.assertRaises(ConfigurationError) as cm:
            RedisConfig(health_check_interval=0)
        self.assertIn('health_check_interval must be a positive integer', str(cm.exception))

    def test_inheritance(self):
        """Test that RedisConfig inherits from VectorStoreConfig."""
        config = RedisConfig()
        self.assertIsInstance(config, VectorStoreConfig)
        self.assertIsInstance(config, RaglConfig)


class TestManagerConfig(unittest.TestCase):
    """Test cases for ManagerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ManagerConfig()
        self.assertEqual(config.index_name, 'rag_index')
        self.assertEqual(config.chunk_size, 512)
        self.assertEqual(config.overlap, 64)
        self.assertEqual(config.max_query_length, 8192)
        self.assertEqual(config.max_input_length, (1024 * 1024) * 10)
        self.assertEqual(config.default_base_id, 'doc')
        self.assertTrue(config.paranoid)

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = ManagerConfig(
            index_name='custom_index',
            chunk_size=1024,
            overlap=128,
            max_query_length=4096,
            max_input_length=5000000,
            default_base_id='document',
            paranoid=False
        )
        self.assertEqual(config.index_name, 'custom_index')
        self.assertEqual(config.chunk_size, 1024)
        self.assertEqual(config.overlap, 128)
        self.assertEqual(config.max_query_length, 4096)
        self.assertEqual(config.max_input_length, 5000000)
        self.assertEqual(config.default_base_id, 'document')
        self.assertFalse(config.paranoid)

    def test_validate_chunk_size_negative(self):
        """Test validation fails for negative chunk size."""
        with self.assertRaises(ConfigurationError) as cm:
            ManagerConfig(chunk_size=0)
        self.assertIn('chunk_size must be a positive integer', str(cm.exception))

    def test_validate_overlap_negative(self):
        """Test validation fails for negative overlap."""
        with self.assertRaises(ConfigurationError) as cm:
            ManagerConfig(overlap=-1)
        self.assertIn('overlap must be a non-negative integer', str(cm.exception))

    def test_validate_overlap_greater_than_chunk_size(self):
        """Test validation fails when overlap >= chunk_size."""
        with self.assertRaises(ConfigurationError) as cm:
            ManagerConfig(chunk_size=100, overlap=100)
        self.assertIn('must be less than', str(cm.exception))

        with self.assertRaises(ConfigurationError) as cm:
            ManagerConfig(chunk_size=100, overlap=150)
        self.assertIn('must be less than', str(cm.exception))

    def test_validate_overlap_boundary_value(self):
        """Test validation passes when overlap = chunk_size - 1."""
        config = ManagerConfig(chunk_size=100, overlap=99)
        self.assertEqual(config.overlap, 99)

    def test_validate_max_query_length_negative(self):
        """Test validation fails for negative max query length."""
        with self.assertRaises(ConfigurationError) as cm:
            ManagerConfig(max_query_length=0)
        self.assertIn('max_query_length must be a positive integer', str(cm.exception))

    def test_validate_max_input_length_negative(self):
        """Test validation fails for negative max input length."""
        with self.assertRaises(ConfigurationError) as cm:
            ManagerConfig(max_input_length=0)
        self.assertIn('max_input_length must be a positive integer', str(cm.exception))

    def test_validate_max_query_length_exceeds_max_input_length(self):
        """Test validation fails when max_query_length > max_input_length."""
        with self.assertRaises(ConfigurationError) as cm:
            ManagerConfig(max_query_length=1000, max_input_length=500)
        self.assertIn('cannot exceed', str(cm.exception))

    def test_validate_max_query_length_equals_max_input_length(self):
        """Test validation passes when max_query_length = max_input_length."""
        config = ManagerConfig(max_query_length=1000, max_input_length=1000)
        self.assertEqual(config.max_query_length, 1000)
        self.assertEqual(config.max_input_length, 1000)

    def test_validate_index_name_empty(self):
        """Test validation fails for empty index name."""
        with self.assertRaises(ConfigurationError) as cm:
            ManagerConfig(index_name='')
        self.assertIn('index_name cannot be empty', str(cm.exception))

    def test_validate_index_name_whitespace_only(self):
        """Test validation fails for whitespace-only index name."""
        with self.assertRaises(ConfigurationError) as cm:
            ManagerConfig(index_name='   ')
        self.assertIn('index_name cannot be empty', str(cm.exception))

    def test_validate_index_name_invalid_characters(self):
        """Test validation fails for index name with invalid characters."""
        invalid_names = ['test!', 'test@name', 'test name', 'test.name',
                         'test#name']
        for name in invalid_names:
            with self.assertRaises(ConfigurationError) as cm:
                ManagerConfig(index_name=name)
            self.assertIn('contains invalid characters', str(cm.exception))

    def test_validate_index_name_valid_characters(self):
        """Test validation passes for index name with valid characters."""
        valid_names = ['test', 'test_name', 'test-name', 'test123',
                       'TEST_NAME']
        for name in valid_names:
            config = ManagerConfig(index_name=name)
            self.assertEqual(config.index_name, name)

    def test_validate_default_base_id_empty(self):
        """Test validation fails for empty default base ID."""
        with self.assertRaises(ConfigurationError) as cm:
            ManagerConfig(default_base_id='')
        self.assertIn('default_base_id cannot be empty', str(cm.exception))

    def test_validate_default_base_id_whitespace_only(self):
        """Test validation fails for whitespace-only default base ID."""
        with self.assertRaises(ConfigurationError) as cm:
            ManagerConfig(default_base_id='   ')
        self.assertIn('default_base_id cannot be empty', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
