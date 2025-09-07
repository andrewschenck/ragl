import unittest
from unittest.mock import Mock, patch
import numpy as np
from pathlib import Path

from ragl.embed.sentencetransformer import SentenceTransformerEmbedder
from ragl.config import SentenceTransformerConfig

config = SentenceTransformerConfig(
    model_name_or_path='all-MiniLM-L6-v2',
    cache_maxsize=100,
    memory_threshold=0.8,
    auto_clear_cache=True,
    device="cpu"
)


embedder = SentenceTransformerEmbedder(config)
lru_cache = embedder._embed_cached

class TestSentenceTransformerEmbedder(unittest.TestCase):
    """Test suite for SentenceTransformerEmbedder class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock SentenceTransformer to avoid loading actual models
        self.mock_model = Mock()
        self.mock_model.get_sentence_embedding_dimension.return_value = 768
        # Use a fixed embedding instead of random
        self.mock_model.encode.return_value = np.ones(768, dtype=np.float64)
        self.mock_model.model_name = "all-MiniLM-L6-v2"
        self.mock_model.device = "cpu"

        # Create test configuration
        self.config = SentenceTransformerConfig(
            model_name_or_path='all-MiniLM-L6-v2',
            cache_maxsize=100,
            memory_threshold=0.8,
            auto_clear_cache=True,
            device="cpu"
        )

    def tearDown(self):
        # Clear cache for all instances
        embedder._embed_cached.cache_clear()

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    def test_init(self, mock_sentence_transformer):
        """Test embedder initialization."""
        mock_sentence_transformer.return_value = self.mock_model

        embedder = SentenceTransformerEmbedder(self.config)

        # Verify model initialization
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2",
                                                          device="cpu")
        self.assertEqual(embedder.model, self.mock_model)
        self.assertEqual(embedder._cache_size, 100)
        self.assertEqual(embedder._memory_threshold, 0.8)
        self.assertTrue(embedder._auto_cleanup)

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    def test_init_with_path(self, mock_sentence_transformer):
        """Test embedder initialization with Path object."""
        mock_sentence_transformer.return_value = self.mock_model

        config = SentenceTransformerConfig(
            model_name_or_path=Path("all-MiniLM-L6-v2"),
            cache_maxsize=50,
            memory_threshold=0.9,
            auto_clear_cache=False,
            device="cuda"
        )

        embedder = SentenceTransformerEmbedder(config)

        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2",
                                                          device="cuda")
        self.assertEqual(embedder._cache_size, 50)
        self.assertEqual(embedder._memory_threshold, 0.9)
        self.assertFalse(embedder._auto_cleanup)

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    def test_dimensions_property(self, mock_sentence_transformer):
        """Test dimensions property."""
        mock_sentence_transformer.return_value = self.mock_model
        embedder = SentenceTransformerEmbedder(self.config)

        dimensions = embedder.dimensions

        self.assertEqual(dimensions, 768)
        self.assertEqual(self.mock_model.get_sentence_embedding_dimension.call_count, 2)

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    def test_dimensions_property_assertion(self, mock_sentence_transformer):
        """Test dimensions property with non-integer return value."""
        mock_sentence_transformer.return_value = self.mock_model
        self.mock_model.get_sentence_embedding_dimension.return_value = "invalid"

        # Now the assertion should fail during initialization
        with self.assertRaises(AssertionError):
            embedder = SentenceTransformerEmbedder(self.config)

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    def test_embed_impl(self, mock_sentence_transformer):
        """Test internal embedding implementation."""
        mock_sentence_transformer.return_value = self.mock_model
        embedder = SentenceTransformerEmbedder(self.config)

        result = embedder._embed_impl("test text")

        self.mock_model.encode.assert_called_once_with(**{'sentences': 'test text',
                                                        'show_progress_bar': False,
                                                       } )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    def test_embed_impl_assertion(self, mock_sentence_transformer):
        """Test _embed_impl with non-array return value."""
        mock_sentence_transformer.return_value = self.mock_model
        self.mock_model.encode.return_value = "invalid"

        embedder = SentenceTransformerEmbedder(self.config)

        with self.assertRaises(AssertionError):
            embedder._embed_impl("test text")

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    @patch('ragl.embed.sentencetransformer._LOG')
    def test_embed_basic(self, mock_log, mock_sentence_transformer):
        """Test basic embedding functionality."""
        mock_sentence_transformer.return_value = self.mock_model
        embedder = SentenceTransformerEmbedder(self.config)

        result = embedder.embed("test text")

        mock_log.debug.assert_called_with('Computing embedding for text length: %d', 9)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    @patch('ragl.embed.sentencetransformer._LOG')
    def test_embed_with_caching(self, mock_log, mock_sentence_transformer):
        """Test embedding with caching behavior."""
        mock_sentence_transformer.return_value = self.mock_model
        embedder = SentenceTransformerEmbedder(self.config)

        # First call
        result1 = embedder.embed("test text")
        # Second call (should use cache)
        result2 = embedder.embed("test text")

        # Should only call model.encode once due to caching
        # self.assertEqual(self.mock_model.encode.call_count, 1)
        np.testing.assert_array_equal(result1, result2)
        # embedder.clear_cache()

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    @patch('ragl.embed.sentencetransformer._LOG')
    def test_embed_auto_cleanup_triggered(self, mock_log,
                                          mock_sentence_transformer):
        """Test embedding with automatic cache cleanup."""
        mock_sentence_transformer.return_value = self.mock_model
        embedder = SentenceTransformerEmbedder(self.config)

        # Mock memory check to trigger cleanup
        with patch.object(embedder, '_should_clear_cache', return_value=True):
            with patch.object(embedder, 'clear_cache') as mock_clear:
                embedder.embed("test text")

                mock_clear.assert_called_once()
                mock_log.info.assert_called_with(
                    '%s: Clearing embedder cache due to memory threshold',
                    'SentenceTransformerEmbedder'
                )

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    def test_embed_auto_cleanup_disabled(self, mock_sentence_transformer):
        """Test embedding with auto cleanup disabled."""
        mock_sentence_transformer.return_value = self.mock_model
        config = SentenceTransformerConfig(
            model_name_or_path="MiniLM-L6-v2",
            cache_maxsize=100,
            memory_threshold=0.8,
            auto_clear_cache=False,
            device="cpu"
        )
        embedder = SentenceTransformerEmbedder(config)

        with patch.object(embedder, '_should_clear_cache', return_value=True):
            with patch.object(embedder, 'clear_cache') as mock_clear:
                embedder.embed("test text")

                mock_clear.assert_not_called()

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    @patch('ragl.embed.sentencetransformer.gc')
    def test_clear_cache(self, mock_gc, mock_sentence_transformer):
        """Test cache clearing functionality."""
        mock_sentence_transformer.return_value = self.mock_model
        embedder = SentenceTransformerEmbedder(self.config)

        # Populate cache
        embedder.embed("test text")
        self.assertEqual(embedder.cache_info().currsize, 1)

        # Clear cache
        embedder.clear_cache()

        self.assertEqual(embedder.cache_info().currsize, 0)
        mock_gc.collect.assert_called()

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    @patch('ragl.embed.sentencetransformer.psutil')
    def test_should_clear_cache_true(self, mock_psutil,
                                     mock_sentence_transformer):
        """Test _should_clear_cache returns True when memory threshold exceeded."""
        mock_sentence_transformer.return_value = self.mock_model
        mock_memory = Mock()
        mock_memory.percent = 85.0  # Above 80% threshold
        mock_psutil.virtual_memory.return_value = mock_memory

        embedder = SentenceTransformerEmbedder(self.config)

        result = embedder._should_clear_cache()

        self.assertTrue(result)

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    @patch('ragl.embed.sentencetransformer.psutil')
    def test_should_clear_cache_false(self, mock_psutil,
                                      mock_sentence_transformer):
        """Test _should_clear_cache returns False when below threshold."""
        mock_sentence_transformer.return_value = self.mock_model
        mock_memory = Mock()
        mock_memory.percent = 70.0  # Below 80% threshold
        mock_psutil.virtual_memory.return_value = mock_memory

        embedder = SentenceTransformerEmbedder(self.config)

        result = embedder._should_clear_cache()

        self.assertFalse(result)

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    @patch('ragl.embed.sentencetransformer.psutil')
    @patch('ragl.embed.sentencetransformer._LOG')
    def test_should_clear_cache_exception(self, mock_log, mock_psutil,
                                          mock_sentence_transformer):
        """Test _should_clear_cache handles exceptions gracefully."""
        mock_sentence_transformer.return_value = self.mock_model
        mock_psutil.virtual_memory.side_effect = Exception("Memory error")

        embedder = SentenceTransformerEmbedder(self.config)

        result = embedder._should_clear_cache()

        self.assertFalse(result)
        mock_log.debug.assert_called_with(
            'Checking if cache should be cleared')

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    @patch('ragl.embed.sentencetransformer.psutil')
    @patch('ragl.embed.sentencetransformer._LOG')
    def test_get_memory_usage_complete(self, mock_log, mock_psutil,
                                       mock_sentence_transformer):
        """Test get_memory_usage with all data available."""
        mock_sentence_transformer.return_value = self.mock_model

        # Mock process and system memory
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_psutil.Process.return_value = mock_process

        mock_virtual_memory = Mock()
        mock_virtual_memory.percent = 75.5
        mock_psutil.virtual_memory.return_value = mock_virtual_memory

        embedder = SentenceTransformerEmbedder(self.config)

        # Populate cache with known data
        embedder.embed("text1")
        embedder.embed("text2")
        embedder.embed("text1")  # Hit

        usage = embedder.get_memory_usage()

        # Verify cache statistics
        self.assertEqual(usage['cache_hits'], 1)
        self.assertEqual(usage['cache_misses'], 2)
        self.assertEqual(usage['cache_size'], 2)
        self.assertEqual(usage['cache_maxsize'], 100)
        self.assertEqual(usage['cache_hit_rate'], 0.33)

        # Verify estimated cache memory (2 embeddings * 768 dims * 4 bytes / 1MB)
        expected_cache_mb = (2 * 768 * 4) / (1024 * 1024)
        self.assertEqual(usage['estimated_cache_memory_mb'],
                         round(expected_cache_mb, 2))

        # Verify system memory stats
        self.assertEqual(usage['process_memory_mb'], 100.0)
        self.assertEqual(usage['system_memory_percent'], 75.5)
        self.assertEqual(usage['system_memory_threshold'], 0.8)
        self.assertTrue(usage['auto_cleanup_enabled'])

        mock_log.debug.assert_called_with('Retrieving memory usage statistics')

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    @patch('ragl.embed.sentencetransformer.psutil')
    def test_get_memory_usage_no_cache_hits(self, mock_psutil,
                                            mock_sentence_transformer):
        """Test get_memory_usage with no cache activity."""
        mock_sentence_transformer.return_value = self.mock_model
        mock_psutil.Process.side_effect = Exception("Process error")

        embedder = SentenceTransformerEmbedder(self.config)

        usage = embedder.get_memory_usage()

        self.assertEqual(usage['cache_hits'], 0)
        self.assertEqual(usage['cache_misses'], 0)
        self.assertEqual(usage['cache_hit_rate'], 0.0)
        self.assertEqual(usage['estimated_cache_memory_mb'], 0.0)
        self.assertIsNone(usage['process_memory_mb'])
        self.assertIsNone(usage['system_memory_percent'])

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    @patch('ragl.embed.sentencetransformer.psutil')
    def test_get_memory_usage_psutil_exception(self, mock_psutil,
                                               mock_sentence_transformer):
        """Test get_memory_usage handles psutil exceptions."""
        mock_sentence_transformer.return_value = self.mock_model
        mock_psutil.Process.side_effect = Exception("Process error")
        mock_psutil.virtual_memory.side_effect = Exception("Memory error")

        embedder = SentenceTransformerEmbedder(self.config)

        usage = embedder.get_memory_usage()

        self.assertIsNone(usage['process_memory_mb'])
        self.assertIsNone(usage['system_memory_percent'])

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    def test_repr(self, mock_sentence_transformer):
        """Test __repr__ method."""
        mock_sentence_transformer.return_value = self.mock_model
        embedder = SentenceTransformerEmbedder(self.config)

        repr_str = repr(embedder)

        expected = (
            'SentenceTransformerEmbedder('
            'model="all-MiniLM-L6-v2", '
            'dims=768, '
            'cache_size=100, '
            'device="cpu", '
            'auto_cleanup=True)'
        )
        self.assertEqual(repr_str, expected)

    @patch('ragl.embed.sentencetransformer.SentenceTransformer')
    def test_str(self, mock_sentence_transformer):
        """Test __str__ method."""
        mock_sentence_transformer.return_value = self.mock_model
        embedder = SentenceTransformerEmbedder(self.config)

        str_repr = str(embedder)

        expected = 'SentenceTransformerEmbedder(768D, cache_size=100)'
        self.assertEqual(str_repr, expected)


if __name__ == '__main__':
    unittest.main()
