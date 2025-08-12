"""
Unit tests for the ragl.registry module.

This module contains comprehensive tests for the factory registry system,
including all factory classes, registration mechanisms, and the
create_rag_manager function.
"""

import logging
import unittest
from unittest.mock import Mock, MagicMock, patch
from typing import Any

from ragl.registry import (
    AbstractFactory,
    EmbedderFactory,
    VectorStoreFactory,
    SentenceTransformerFactory,
    RedisVectorStoreFactory,
    create_rag_manager,
)
from ragl.config import (
    EmbedderConfig,
    ManagerConfig,
    RaglConfig,
    RedisConfig,
    SentenceTransformerConfig,
    VectorStoreConfig,
)
from ragl.exceptions import ConfigurationError


class MockConfig(RaglConfig):
    """Mock configuration class for testing."""
    pass


class MockFactory(AbstractFactory):
    """Mock factory class for testing."""
    _config_cls = MockConfig

    def __call__(self, *args, **kwargs) -> Any:
        return Mock()


class InvalidConfig:
    """Invalid config class that doesn't inherit from RaglConfig."""
    pass


class TestAbstractFactory(unittest.TestCase):
    """Test cases for AbstractFactory class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Clear factory map before each test
        AbstractFactory._factory_map = {}
        EmbedderFactory._factory_map = {}
        VectorStoreFactory._factory_map = {}

    def test_init_subclass_creates_factory_map_for_direct_subclass(self):
        """Test that direct subclasses get their own factory map."""

        class TestDirectFactory(AbstractFactory):
            _config_cls = MockConfig

        self.assertIsInstance(TestDirectFactory._factory_map, dict)
        self.assertIn(MockConfig, TestDirectFactory._factory_map)

    def test_init_subclass_with_config_cls_kwarg(self):
        """Test __init_subclass__ with config_cls passed as keyword argument."""

        class TestFactory(AbstractFactory, config_cls=MockConfig):
            pass

        self.assertIn(MockConfig, TestFactory._factory_map)

    def test_register_cls_success(self):
        """Test successful registration of a configuration class."""
        factory_map_before = len(AbstractFactory._factory_map)
        AbstractFactory.register_cls(MockConfig, MockFactory)

        self.assertEqual(len(AbstractFactory._factory_map),
                         factory_map_before + 1)
        self.assertIn(MockConfig, AbstractFactory._factory_map)
        self.assertEqual(AbstractFactory._factory_map[MockConfig],
                         MockFactory)

    def test_register_cls_invalid_config_type(self):
        """Test registration with invalid config class type."""
        with self.assertRaises(TypeError) as context:
            AbstractFactory.register_cls(InvalidConfig, MockFactory)

        self.assertIn('config_cls must be a subclass of RaglConfig',
                      str(context.exception))

    def test_register_cls_invalid_factory_type(self):
        """Test registration with invalid factory type."""

        class InvalidFactory:
            pass

        with self.assertRaises(TypeError) as context:
            AbstractFactory.register_cls(MockConfig, InvalidFactory)

        self.assertIn('cls must be a subclass of AbstractFactory',
                      str(context.exception))

    def test_unregister_cls_existing(self):
        """Test unregistering an existing configuration class."""
        AbstractFactory.register_cls(MockConfig, MockFactory)
        self.assertIn(MockConfig, AbstractFactory._factory_map)

        AbstractFactory.unregister_cls(MockConfig)
        self.assertNotIn(MockConfig, AbstractFactory._factory_map)

    def test_unregister_cls_nonexistent(self):
        """Test unregistering a non-existent configuration class doesn't raise error."""
        # Should not raise any exception
        AbstractFactory.unregister_cls(MockConfig)

    def test_call_not_allowed_for_abstract_factory(self):
        """Test that AbstractFactory cannot be called directly."""
        factory = AbstractFactory()

        with self.assertRaises(TypeError) as context:
            factory(config=MockConfig())

        self.assertIn('AbstractFactory cannot be called directly',
                      str(context.exception))

    def test_call_missing_config(self):
        """Test __call__ without config parameter."""
        factory = EmbedderFactory()

        with self.assertRaises(ConfigurationError) as context:
            factory()

        self.assertIn('config must be provided', str(context.exception))

    def test_call_no_factory_found(self):
        """Test __call__ with unregistered configuration type."""
        factory = EmbedderFactory()
        config = MockConfig()

        with self.assertRaises(ConfigurationError) as context:
            factory(config=config)

        self.assertIn('No factory found for configuration type',
                      str(context.exception))

    def test_call_success(self):
        """Test successful factory call."""
        # Register a mock factory
        EmbedderFactory.register_cls(MockConfig, MockFactory)

        factory = EmbedderFactory()
        config = MockConfig()
        result = factory(config=config)

        self.assertIsNotNone(result)

    def test_is_direct_subclass_true(self):
        """Test _is_direct_subclass returns True for direct subclass."""

        class DirectSubclass(AbstractFactory):
            pass

        self.assertTrue(DirectSubclass._is_direct_subclass())

    def test_is_direct_subclass_false(self):
        """Test _is_direct_subclass returns False for indirect subclass."""

        class IndirectSubclass(EmbedderFactory):
            pass

        self.assertFalse(IndirectSubclass._is_direct_subclass())

    def test_str_representation(self):
        """Test string representation of factory."""
        factory = MockFactory()
        MockFactory.register_cls(MockConfig, MockFactory)

        str_repr = str(factory)
        self.assertIn('MockFactory', str_repr)
        self.assertIn('registered_configs', str_repr)
        self.assertIn(MockConfig.__name__, str_repr)

    def test_repr_representation_with_config_cls(self):
        """Test detailed representation with config class."""
        factory = MockFactory()

        repr_str = repr(factory)
        self.assertIn('MockFactory', repr_str)
        self.assertIn('config_cls=MockConfig', repr_str)
        self.assertIn('factory_map=', repr_str)

    def test_repr_representation_without_config_cls(self):
        """Test detailed representation without config class."""

        class NoConfigFactory(AbstractFactory):
            pass

        factory = NoConfigFactory()
        repr_str = repr(factory)
        self.assertIn('NoConfigFactory', repr_str)
        self.assertIn('config_cls=None', repr_str)


class TestEmbedderFactory(unittest.TestCase):
    """Test cases for EmbedderFactory class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        EmbedderFactory._factory_map = {}

    def test_embedder_factory_inheritance(self):
        """Test that EmbedderFactory inherits from AbstractFactory."""
        self.assertTrue(issubclass(EmbedderFactory, AbstractFactory))

    def test_embedder_factory_has_own_factory_map(self):
        """Test that EmbedderFactory has its own factory map."""
        self.assertIsInstance(EmbedderFactory._factory_map, dict)


class TestVectorStoreFactory(unittest.TestCase):
    """Test cases for VectorStoreFactory class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        VectorStoreFactory._factory_map = {}

    def test_vector_store_factory_inheritance(self):
        """Test that VectorStoreFactory inherits from AbstractFactory."""
        self.assertTrue(issubclass(VectorStoreFactory, AbstractFactory))

    def test_vector_store_factory_has_own_factory_map(self):
        """Test that VectorStoreFactory has its own factory map."""
        self.assertIsInstance(VectorStoreFactory._factory_map, dict)


class TestSentenceTransformerFactory(unittest.TestCase):
    """Test cases for SentenceTransformerFactory class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        EmbedderFactory._factory_map = {}

    def test_sentence_transformer_factory_inheritance(self):
        """Test that SentenceTransformerFactory inherits from EmbedderFactory."""
        self.assertTrue(
            issubclass(SentenceTransformerFactory, EmbedderFactory))

    def test_sentence_transformer_factory_config_cls(self):
        """Test that SentenceTransformerFactory has correct config class."""
        self.assertEqual(SentenceTransformerFactory._config_cls,
                         SentenceTransformerConfig)

    def test_call_missing_config(self):
        """Test __call__ without config parameter."""
        factory = SentenceTransformerFactory()

        with self.assertRaises(ConfigurationError) as context:
            factory()

        self.assertIn('config parameter must be provided',
                      str(context.exception))

    @patch('ragl.embed.sentencetransformer.SentenceTransformerEmbedder')
    def test_call_success(self, mock_embedder_class):
        """Test successful SentenceTransformerEmbedder creation."""
        mock_embedder = Mock()
        mock_embedder_class.return_value = mock_embedder

        factory = SentenceTransformerFactory()
        config = SentenceTransformerConfig()

        result = factory(config=config)

        mock_embedder_class.assert_called_once_with(config=config)
        self.assertEqual(result, mock_embedder)

    def test_call_import_error(self):
        """Test __call__ when SentenceTransformerEmbedder import fails."""
        factory = SentenceTransformerFactory()
        config = SentenceTransformerConfig()

        with patch('ragl.embed.sentencetransformer.SentenceTransformerEmbedder',
                   side_effect=ImportError()):
            with self.assertRaises(ConfigurationError) as context:
                factory(config=config)

            self.assertIn('SentenceTransformerEmbedder not available',
                          str(context.exception))


class TestRedisVectorStoreFactory(unittest.TestCase):
    """Test cases for RedisVectorStoreFactory class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        VectorStoreFactory._factory_map = {}

    def test_redis_vector_store_factory_inheritance(self):
        """Test that RedisVectorStoreFactory inherits from VectorStoreFactory."""
        self.assertTrue(
            issubclass(RedisVectorStoreFactory, VectorStoreFactory))

    def test_redis_vector_store_factory_config_cls(self):
        """Test that RedisVectorStoreFactory has correct config class."""
        self.assertEqual(RedisVectorStoreFactory._config_cls, RedisConfig)

    def test_call_missing_config(self):
        """Test __call__ without config parameter."""
        factory = RedisVectorStoreFactory()

        with self.assertRaises(ConfigurationError) as context:
            factory()

        self.assertIn('config, dimensions, and index_name must be provided',
                      str(context.exception))

    def test_call_missing_dimensions(self):
        """Test __call__ without dimensions parameter."""
        factory = RedisVectorStoreFactory()
        config = RedisConfig()

        with self.assertRaises(ConfigurationError) as context:
            factory(config=config)

        self.assertIn('config, dimensions, and index_name must be provided',
                      str(context.exception))

    def test_call_missing_index_name(self):
        """Test __call__ without index_name parameter."""
        factory = RedisVectorStoreFactory()
        config = RedisConfig()

        with self.assertRaises(ConfigurationError) as context:
            factory(config=config, dimensions=128)

        self.assertIn('config, dimensions, and index_name must be provided',
                      str(context.exception))

    @patch('ragl.store.redis.RedisVectorStore')
    def test_call_success(self, mock_vector_store_class):
        """Test successful RedisVectorStore creation."""
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        factory = RedisVectorStoreFactory()
        config = RedisConfig()
        dimensions = 128
        index_name = "test_index"

        result = factory(config=config, dimensions=dimensions,
                         index_name=index_name)

        mock_vector_store_class.assert_called_once_with(
            redis_config=config,
            dimensions=dimensions,
            index_name=index_name,
        )
        self.assertEqual(result, mock_vector_store)

    def test_call_import_error(self):
        """Test __call__ when RedisVectorStore import fails."""
        factory = RedisVectorStoreFactory()
        config = RedisConfig()

        with patch('ragl.store.redis.RedisVectorStore',
                   side_effect=ImportError()):
            with self.assertRaises(ConfigurationError) as context:
                factory(config=config, dimensions=128, index_name="test")

            self.assertIn('RedisVectorStore not available',
                          str(context.exception))


class TestCreateRagManager(unittest.TestCase):
    """Test cases for create_rag_manager function."""

    @patch('ragl.registry.RAGManager')
    @patch('ragl.registry.RAGStore')
    @patch('ragl.registry.VectorStoreFactory')
    @patch('ragl.registry.EmbedderFactory')
    def test_create_rag_manager_success(self, mock_embedder_factory_class,
                                        mock_vector_store_factory_class,
                                        mock_rag_store_class,
                                        mock_rag_manager_class):
        """Test successful RAGManager creation."""
        # Set up mocks
        mock_embedder = Mock()
        mock_embedder.dimensions = 128
        mock_embedder_factory = Mock()
        mock_embedder_factory.return_value = mock_embedder
        mock_embedder_factory_class.return_value = mock_embedder_factory

        mock_storage = Mock()
        mock_storage_factory = Mock()
        mock_storage_factory.return_value = mock_storage
        mock_vector_store_factory_class.return_value = mock_storage_factory

        mock_ragstore = Mock()
        mock_rag_store_class.return_value = mock_ragstore

        mock_manager = Mock()
        mock_rag_manager_class.return_value = mock_manager

        # Set up configurations
        index_name = "test_index"
        storage_config = VectorStoreConfig()
        embedder_config = EmbedderConfig()
        manager_config = ManagerConfig()

        # Call function
        result = create_rag_manager(
            index_name=index_name,
            storage_config=storage_config,
            embedder_config=embedder_config,
            manager_config=manager_config,
        )

        # Verify calls
        mock_embedder_factory_class.assert_called_once()
        mock_embedder_factory.assert_called_once_with(config=embedder_config)

        mock_vector_store_factory_class.assert_called_once()
        mock_storage_factory.assert_called_once_with(
            config=storage_config,
            dimensions=128,
            index_name=index_name,
        )

        mock_rag_store_class.assert_called_once_with(
            embedder=mock_embedder,
            storage=mock_storage,
        )

        mock_rag_manager_class.assert_called_once_with(
            config=manager_config,
            ragstore=mock_ragstore,
        )

        self.assertEqual(result, mock_manager)


class TestLogging(unittest.TestCase):
    """Test cases for logging functionality."""

    def test_debug_logging_in_call(self):
        """Test debug logging in factory __call__ method."""
        EmbedderFactory.register_cls(MockConfig, MockFactory)
        factory = EmbedderFactory()
        config = MockConfig()

        with self.assertLogs('ragl.registry', level='DEBUG') as log:
            factory(config=config)

        # Verify debug messages are logged
        debug_messages = [record.levelname for record in log.records if
                          record.levelname == 'DEBUG']
        self.assertTrue(len(debug_messages) > 0)

    def test_critical_logging_on_error(self):
        """Test critical logging when errors occur."""
        factory = EmbedderFactory()

        with self.assertLogs('ragl.registry', level='CRITICAL') as log:
            try:
                factory()  # Missing config
            except ConfigurationError:
                pass

        # Verify critical messages are logged
        critical_messages = [record.levelname for record in log.records if
                             record.levelname == 'CRITICAL']
        self.assertTrue(len(critical_messages) > 0)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)

    # Run all tests
    unittest.main()
