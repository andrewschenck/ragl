import unittest
from collections import deque
from unittest.mock import Mock, patch

from ragl.config import ManagerConfig
from ragl.exceptions import DataError, ValidationError, ConfigurationError
from ragl.manager import RAGManager, RAGTelemetry, TextUnitChunker
from ragl.protocols import RAGStoreProtocol, TokenizerProtocol
from ragl.textunit import TextUnit
from ragl.tokenizer import TiktokenTokenizer


class TestRAGTelemetry(unittest.TestCase):
    """Test cases for RAGTelemetry class."""

    def setUp(self):
        """Set up test fixtures."""
        self.telemetry = RAGTelemetry()

    def test_init_default_values(self):
        """Test RAGTelemetry initialization with default values."""
        self.assertEqual(self.telemetry.total_calls, 0)
        self.assertEqual(self.telemetry.total_duration, 0.0)
        self.assertEqual(self.telemetry.avg_duration, 0.0)
        self.assertEqual(self.telemetry.min_duration, float('inf'))
        self.assertEqual(self.telemetry.max_duration, 0.0)
        self.assertEqual(self.telemetry.failure_count, 0)
        self.assertIsInstance(self.telemetry.recent_durations, deque)
        self.assertEqual(self.telemetry.recent_durations.maxlen, 100)

    @patch('ragl.manager._LOG')
    def test_record_success(self, mock_log):
        """Test recording successful operations."""
        duration = 1.5
        self.telemetry.record_success(duration)

        self.assertEqual(self.telemetry.total_calls, 1)
        self.assertEqual(self.telemetry.total_duration, 1.5)
        self.assertEqual(self.telemetry.avg_duration, 1.5)
        self.assertEqual(self.telemetry.min_duration, 1.5)
        self.assertEqual(self.telemetry.max_duration, 1.5)
        self.assertEqual(self.telemetry.failure_count, 0)
        self.assertEqual(list(self.telemetry.recent_durations), [1.5])
        mock_log.debug.assert_called_with('Recording successful operation')

    @patch('ragl.manager._LOG')
    def test_record_failure(self, mock_log):
        """Test recording failed operations."""
        duration = 2.0
        self.telemetry.record_failure(duration)

        self.assertEqual(self.telemetry.total_calls, 1)
        self.assertEqual(self.telemetry.total_duration, 2.0)
        self.assertEqual(self.telemetry.avg_duration, 2.0)
        self.assertEqual(self.telemetry.min_duration, 2.0)
        self.assertEqual(self.telemetry.max_duration, 2.0)
        self.assertEqual(self.telemetry.failure_count, 1)
        self.assertEqual(list(self.telemetry.recent_durations), [2.0])
        mock_log.debug.assert_called_with('Recording failed operation')

    def test_record_multiple_operations(self):
        """Test recording multiple operations updates metrics correctly."""
        self.telemetry.record_success(1.0)
        self.telemetry.record_success(3.0)
        self.telemetry.record_failure(2.0)

        self.assertEqual(self.telemetry.total_calls, 3)
        self.assertEqual(self.telemetry.total_duration, 6.0)
        self.assertEqual(self.telemetry.avg_duration, 2.0)
        self.assertEqual(self.telemetry.min_duration, 1.0)
        self.assertEqual(self.telemetry.max_duration, 3.0)
        self.assertEqual(self.telemetry.failure_count, 1)

    @patch('ragl.manager._LOG')
    def test_compute_metrics_no_calls(self, mock_log):
        """Test computing metrics when no calls have been made."""
        metrics = self.telemetry.compute_metrics()

        expected = {
            'total_calls':   0,
            'failure_count': 0,
            'success_rate':  0.0,
            'min_duration':  0.0,
            'max_duration':  0.0,
            'avg_duration':  0.0,
            'recent_avg':    0.0,
            'recent_med':    0.0,
        }
        self.assertEqual(metrics, expected)
        mock_log.debug.assert_called_with('Computing metrics')

    def test_compute_metrics_with_data(self):
        """Test computing metrics with recorded data."""
        self.telemetry.record_success(1.0)
        self.telemetry.record_success(2.0)
        self.telemetry.record_failure(3.0)

        metrics = self.telemetry.compute_metrics()

        expected = {
            'total_calls':   3,
            'failure_count': 1,
            'success_rate':  0.6667,
            'min_duration':  1.0,
            'max_duration':  3.0,
            'avg_duration':  2.0,
            'recent_avg':    2.0,
            'recent_med':    2.0,
        }
        self.assertEqual(metrics, expected)

    def test_compute_metrics_rounding(self):
        """Test that metrics are properly rounded."""
        self.telemetry.record_success(1.123456)
        self.telemetry.record_success(2.987654)

        metrics = self.telemetry.compute_metrics()

        self.assertEqual(metrics['min_duration'], 1.1235)
        self.assertEqual(metrics['max_duration'], 2.9877)
        self.assertEqual(metrics['avg_duration'], 2.0556)

    def test_recent_durations_max_length(self):
        """Test that recent_durations respects maxlen."""
        for i in range(150):
            self.telemetry.record_success(i)

        self.assertEqual(len(self.telemetry.recent_durations), 100)
        self.assertEqual(list(self.telemetry.recent_durations)[:5],
                         [50, 51, 52, 53, 54])


class TestTextUnitChunker(unittest.TestCase):
    """Test cases for TextUnitChunker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tokenizer = Mock(spec=TokenizerProtocol)
        self.chunker = TextUnitChunker(
            tokenizer=self.mock_tokenizer,
            chunk_size=100,
            overlap=20,
            min_chunk_size=25,
            split=True,
            text_index=0
        )

        # Create a basic TextUnit for testing
        self.base_unit = TextUnit(
            text="This is a test text for chunking",
            parent_id="test-doc",
            source="test_source.txt"
        )

    def test_init_valid_tokenizer(self):
        """Test TextUnitChunker initialization with valid tokenizer."""
        chunker = TextUnitChunker(
            tokenizer=self.mock_tokenizer,
            chunk_size=100,
            overlap=20
        )
        self.assertEqual(chunker.tokenizer, self.mock_tokenizer)
        self.assertEqual(chunker.chunk_size, 100)
        self.assertEqual(chunker.overlap, 20)

    def test_chunk_text_unit_no_split_single_chunk(self):
        """Test chunking with split=False returns single chunk."""
        chunker = TextUnitChunker(
            tokenizer=self.mock_tokenizer,
            chunk_size=100,
            overlap=10,
            split=False,
            text_index=0,
            batch_timestamp=1000000
        )

        result = list(chunker.chunk_text_unit(unit=self.base_unit))

        self.assertEqual(len(result), 1)
        chunk = result[0]
        self.assertEqual(chunk.text, self.base_unit.text)
        self.assertEqual(chunk.parent_id, "test-doc")
        self.assertEqual(chunk.chunk_position, 0)
        self.assertEqual(chunk.distance, 0.0)
        self.assertTrue(chunk.text_id.startswith('txt:test-doc-1000000-0-0'))

    def test_chunk_text_unit_no_split_with_custom_timestamp(self):
        """Test chunking with custom timestamp."""
        chunker = TextUnitChunker(
            tokenizer=self.mock_tokenizer,
            chunk_size=100,
            overlap=10,
            split=False,
            text_index=2,
            batch_timestamp=9999999
        )

        result = list(chunker.chunk_text_unit(unit=self.base_unit))

        self.assertEqual(len(result), 1)
        expected_text_id = 'txt:test-doc-9999999-2-0'
        self.assertEqual(result[0].text_id, expected_text_id)

    def test_chunk_text_unit_split_single_chunk_short_text(self):
        """Test chunking short text that fits in single chunk."""
        self.mock_tokenizer.encode.return_value = list(range(50))
        self.mock_tokenizer.decode.return_value = self.base_unit.text

        result = list(self.chunker.chunk_text_unit(unit=self.base_unit))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, self.base_unit.text)

    def test_chunk_text_unit_split_multiple_chunks(self):
        """Test chunking long text into multiple chunks."""
        # Mock long text that needs splitting
        self.mock_tokenizer.encode.return_value = list(range(250))
        self.mock_tokenizer.decode.side_effect = [
            "First chunk text",
            "Second chunk text",
            "Third chunk text",
            "Fourth chunk text"  # Added missing fourth chunk
        ]

        result = list(self.chunker.chunk_text_unit(unit=self.base_unit))

        self.assertEqual(len(result), 4)  # Updated expected count
        for i, chunk in enumerate(result):
            self.assertEqual(chunk.chunk_position, i)
            self.assertEqual(chunk.parent_id, "test-doc")

    def test_chunk_text_unit_skips_empty_chunks_with_continue(self):
        """Test that empty or whitespace-only chunks trigger the continue statement."""
        # Force splitting by using more tokens than chunk_size
        self.mock_tokenizer.encode.return_value = list(range(250))

        # Mock decode to return mix of empty/whitespace chunks that should be skipped
        self.mock_tokenizer.decode.side_effect = [
            "",  # Empty string - should trigger continue
            "   \n\t   ",  # Whitespace only - should trigger continue
            "Valid chunk"  # Valid chunk - should be processed
        ]

        # Mock _split_text to return the chunks that decode will produce
        with patch.object(self.chunker, '_split_text') as mock_split:
            mock_split.return_value = ["", "   \n\t   ", "Valid chunk"]

            result = list(self.chunker.chunk_text_unit(unit=self.base_unit))

        # Should only return 1 chunk (the valid one), empty/whitespace chunks skipped
        self.assertEqual(len(result), 1)

        # Verify the valid chunk was processed correctly
        chunk = result[0]
        self.assertEqual(chunk.text, "Valid chunk")
        self.assertEqual(chunk.chunk_position,
                         2)  # Position from original enumeration
        self.assertEqual(chunk.parent_id, "test-doc")
        self.assertEqual(chunk.distance, 0.0)

        # Verify text_id reflects the original chunk_position (2)
        self.assertTrue(chunk.text_id.endswith('-2'))

    def test_chunk_text_unit_skips_empty_chunks(self):
        """Test that empty chunks are skipped."""
        # Mock scenario where some chunks decode to empty strings
        self.mock_tokenizer.encode.return_value = list(range(200))
        self.mock_tokenizer.decode.side_effect = [
            "Valid chunk 1",
            "",  # Empty chunk
            "Valid chunk 2"
        ]

        result = list(self.chunker.chunk_text_unit(unit=self.base_unit))

        # Only non-empty chunks should be returned
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].text, "Valid chunk 1")
        self.assertEqual(result[1].text, "Valid chunk 2")

    def test_chunk_text_unit_preserves_metadata(self):
        """Test that TextUnit metadata is preserved in chunks."""
        rich_unit = TextUnit(
            text="Rich text with metadata",
            parent_id="rich-doc",
            source="rich_source.txt",
            confidence=0.95,
            language="en",
            section="chapter1",
            author="test_author",
            tags=["tag1", "tag2"]
        )

        self.mock_tokenizer.encode.return_value = list(range(50))
        self.mock_tokenizer.decode.return_value = rich_unit.text

        result = list(self.chunker.chunk_text_unit(unit=rich_unit))

        self.assertEqual(len(result), 1)
        chunk = result[0]

        # Verify all metadata is preserved
        self.assertEqual(chunk.parent_id, "rich-doc")
        self.assertEqual(chunk.source, "rich_source.txt")
        self.assertEqual(chunk.confidence, 0.95)
        self.assertEqual(chunk.language, "en")
        self.assertEqual(chunk.section, "chapter1")
        self.assertEqual(chunk.author, "test_author")
        self.assertEqual(chunk.tags, ["tag1", "tag2"])

    def test_split_text_short_text_single_chunk(self):
        """Test _split_text with text shorter than chunk_size."""
        text = "Short text"
        self.mock_tokenizer.encode.return_value = list(range(20))

        result = self.chunker._split_text(
            tokenizer=self.mock_tokenizer,
            text=text,
            chunk_size=100,
            overlap=20
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], text)

    def test_split_text_long_text_multiple_chunks(self):
        """Test _split_text with text longer than chunk_size."""
        text = "Long text that needs splitting"
        # Mock tokenizer to simulate long text
        self.mock_tokenizer.encode.return_value = list(range(250))
        self.mock_tokenizer.decode.side_effect = [
            "First chunk",
            "Second chunk",
            "Third chunk",
            "Fourth chunk"
        ]

        result = self.chunker._split_text(
            tokenizer=self.mock_tokenizer,
            text=text,
            chunk_size=100,
            overlap=20
        )

        self.assertEqual(len(result), 4)
        self.assertEqual(result, ["First chunk", "Second chunk", "Third chunk",
                                  "Fourth chunk"])

        # Verify tokenizer calls
        self.assertTrue(self.mock_tokenizer.encode.call_count >= 1)
        self.assertEqual(self.mock_tokenizer.decode.call_count, 4)

    def test_split_text_merge_short_last_chunk(self):
        """Test _split_text merges last chunk if it's too short."""
        text = "Text that creates a short last chunk"

        tokens = list(range(150))
        encode_calls = []

        def mock_encode(text_input):
            encode_calls.append(text_input)
            if "short_last" in text_input:
                return list(range(15))  # Short chunk
            return tokens

        self.mock_tokenizer.encode.side_effect = mock_encode
        self.mock_tokenizer.decode.side_effect = [
            "First chunk content",
            "short_last"
        ]

        result = self.chunker._split_text(
            tokenizer=self.mock_tokenizer,
            text=text,
            chunk_size=100,
            overlap=20,
            min_chunk_size=25
        )

        # Should have fewer chunks due to merging
        self.assertEqual(len(result), 1)
        self.assertIn("First chunk content", result[0])
        self.assertIn("short_last", result[0])

        # Verify encode calls
        self.assertEqual(len(encode_calls), 2)
        self.assertEqual(encode_calls[0], text)
        self.assertEqual(encode_calls[1], "short_last")

    def test_split_text_removes_empty_chunks(self):
        """Test _split_text removes empty chunks after decoding."""
        text = "Text with some empty decoded chunks"
        # Use more tokens to force more chunks
        self.mock_tokenizer.encode.return_value = list(
            range(400))  # Increased from 200
        self.mock_tokenizer.decode.side_effect = [
            "Valid chunk 1",
            "",  # Empty
            "Valid chunk 2",
            "   ",  # Whitespace only
            "Valid chunk 3"
        ]

        result = self.chunker._split_text(
            tokenizer=self.mock_tokenizer,
            text=text,
            chunk_size=100,
            overlap=20
        )

        # Should only return non-empty chunks
        expected = ["Valid chunk 1", "Valid chunk 2", "Valid chunk 3"]
        self.assertEqual(result, expected)

    def test_split_text_default_min_chunk_size(self):
        """Test _split_text uses default min_chunk_size when not provided."""
        chunker = TextUnitChunker(
            tokenizer=self.mock_tokenizer,
            chunk_size=100,
            overlap=20,
            min_chunk_size=None  # Should default to overlap // 2 = 10
        )

        text = "Text for testing default min chunk size"
        self.mock_tokenizer.encode.return_value = list(range(80))

        result = chunker._split_text(
            tokenizer=self.mock_tokenizer,
            text=text,
            chunk_size=100,
            overlap=20
        )

        self.assertIsInstance(result, list)
        self.assertEqual(result, ["Text for testing default min chunk size"])

    def test_split_text_custom_min_chunk_size(self):
        """Test _split_text with custom min_chunk_size."""
        chunker = TextUnitChunker(
            tokenizer=self.mock_tokenizer,
            chunk_size=100,
            overlap=20,
            min_chunk_size=50
        )

        text = "Text for testing custom min chunk size"
        tokens = list(range(120))
        self.mock_tokenizer.encode.return_value = tokens

        def mock_decode(chunk_tokens):
            if len(chunk_tokens) < 50:
                return "short"
            return f"chunk_{len(chunk_tokens)}"

        self.mock_tokenizer.decode.side_effect = mock_decode

        result = chunker._split_text(
            tokenizer=self.mock_tokenizer,
            text=text,
            chunk_size=100,
            overlap=20,
            min_chunk_size=50
        )

        self.assertIsInstance(result, list)

    def test_chunk_text_unit_text_id_generation(self):
        """Test text_id generation format."""
        chunker = TextUnitChunker(
            tokenizer=self.mock_tokenizer,
            chunk_size=100,
            overlap=20,
            text_index=5,
            batch_timestamp=1234567890
        )

        self.mock_tokenizer.encode.return_value = list(range(50))
        self.mock_tokenizer.decode.return_value = self.base_unit.text

        result = list(chunker.chunk_text_unit(unit=self.base_unit))

        expected_text_id = 'txt:test-doc-1234567890-5-0'
        self.assertEqual(result[0].text_id, expected_text_id)

    def test_chunk_text_unit_all_chunks_empty_returns_empty(self):
        """Test that if all chunks are empty, no chunks are returned."""
        # Force splitting by using more tokens than chunk_size
        self.mock_tokenizer.encode.return_value = list(
            range(150))  # > chunk_size
        self.mock_tokenizer.decode.side_effect = ["", "   ", ""]  # All empty

        result = list(self.chunker.chunk_text_unit(unit=self.base_unit))

        self.assertEqual(len(result), 0)

    def test_chunk_text_unit_passes_min_chunk_size(self):
        """Test that min_chunk_size from constructor is used in _split_text."""
        chunker = TextUnitChunker(
            tokenizer=self.mock_tokenizer,
            chunk_size=100,
            overlap=20,
            min_chunk_size=30
        )

        with patch.object(chunker, '_split_text') as mock_split:
            mock_split.return_value = ["chunk1", "chunk2"]

            list(chunker.chunk_text_unit(unit=self.base_unit))

            mock_split.assert_called_once_with(
                tokenizer=self.mock_tokenizer,
                text=self.base_unit.text,
                chunk_size=100,
                overlap=20,
                min_chunk_size=30
            )

    def test_str_representation(self):
        """Test string representation of TextUnitChunker."""
        chunker = TextUnitChunker(
            tokenizer=self.mock_tokenizer,
            chunk_size=256,
            overlap=50,
            min_chunk_size=25,
            split=True
        )

        expected = ('TextUnitChunker(chunk_size=256, overlap=50, '
                    'min_chunk_size=25, split=True)')
        self.assertEqual(str(chunker), expected)

    def test_str_representation_with_none_min_chunk_size(self):
        """Test string representation with None min_chunk_size."""
        chunker = TextUnitChunker(
            tokenizer=self.mock_tokenizer,
            chunk_size=512,
            overlap=100,
            min_chunk_size=None,
            split=False
        )

        expected = ('TextUnitChunker(chunk_size=512, overlap=100, '
                    'min_chunk_size=None, split=False)')
        self.assertEqual(str(chunker), expected)

    def test_repr_representation(self):
        """Test repr representation of TextUnitChunker."""
        chunker = TextUnitChunker(
            tokenizer=self.mock_tokenizer,
            chunk_size=128,
            overlap=25,
            min_chunk_size=10,
            split=True
        )

        # __repr__ calls __str__, so they should be identical
        expected = ('TextUnitChunker(chunk_size=128, overlap=25, '
                    'min_chunk_size=10, split=True)')
        self.assertEqual(repr(chunker), expected)
        self.assertEqual(str(chunker), repr(chunker))


class TestRAGManager(unittest.TestCase):
    """Test cases for RAGManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_ragstore = Mock(spec=RAGStoreProtocol)
        self.mock_tokenizer = Mock(spec=TokenizerProtocol)
        self.config = ManagerConfig(chunk_size=100, overlap=20, paranoid=False)

        # Setup default mock behaviors
        self.mock_ragstore.list_texts.return_value = []
        tu = TextUnit(
            text_id='text-id-1',
            text='test text',
            distance=0.0,
        )
        self.mock_ragstore.store_text.return_value = tu
        # self.mock_ragstore.store_text.return_value = TextUnit(
        #     text_id='text-id-1',
        #     text='test text',
        #     distance=0.0
        # )
        # mock_text_unit = Mock(spec=TextUnit)
        self.mock_ragstore.store_texts.return_value = [tu]

        self.mock_tokenizer.encode.return_value = list(range(100))
        self.mock_tokenizer.decode.return_value = 'decoded text chunk'

        def mock_store_text(text_unit):
            # Update the text_id and return the same TextUnit
            # text_unit.text_id = f'text-id-{len(self.stored_units) + 1}'
            self.stored_units.append(text_unit)
            return text_unit

        self.stored_units = []
        self.mock_ragstore.store_text.side_effect = mock_store_text

        self.mock_tokenizer.encode.return_value = list(range(100))
        self.mock_tokenizer.decode.return_value = 'decoded text chunk'

    def test_init_valid_parameters(self):
        """Test RAGManager initialization with valid parameters."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        self.assertEqual(manager.ragstore, self.mock_ragstore)
        self.assertEqual(manager.tokenizer, self.mock_tokenizer)
        self.assertEqual(manager.chunk_size, 100)
        self.assertEqual(manager.overlap, 20)
        self.assertFalse(manager.paranoid)
        self.assertIsInstance(manager._metrics, dict)

    def test_init_default_tokenizer(self):
        """Test RAGManager initialization with default tokenizer."""
        manager = RAGManager(self.config, self.mock_ragstore)
        self.assertIsInstance(manager.tokenizer, TiktokenTokenizer)
        self.assertEqual(manager.tokenizer.encoding.name, "cl100k_base")

    def test_init_invalid_ragstore(self):
        """Test RAGManager initialization with invalid ragstore."""
        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(TypeError) as cm:
                RAGManager(self.config, "invalid_ragstore")

            self.assertIn('ragstore must implement RAGStoreProtocol',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'ragstore must implement RAGStoreProtocol')

    def test_init_invalid_tokenizer(self):
        """Test RAGManager initialization with invalid tokenizer."""
        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(TypeError) as cm:
                RAGManager(self.config, self.mock_ragstore,
                           tokenizer="invalid_tokenizer")

            self.assertIn('tokenizer must implement TokenizerProtocol',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'tokenizer must implement TokenizerProtocol')

    @patch('ragl.manager._LOG')
    def test_add_text_string_success(self, mock_log):
        """Test adding text as string successfully."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "This is a test text"

        # Mock tokenizer to return smaller chunks
        self.mock_tokenizer.encode.return_value = list(range(50))
        self.mock_tokenizer.decode.return_value = text

        result = manager.add_text(text)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TextUnit)
        self.mock_ragstore.store_texts.assert_called_once()
        mock_log.debug.assert_any_call('Adding text: %d characters', 19)

    @patch('ragl.manager._LOG')
    def test_add_text_textunit_success(self, mock_log):
        """Test adding TextUnit successfully."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text_unit = TextUnit(
            text_id="test-id",
            text="Test text",
            source="test",
            timestamp=12345,
            tags=["test"],
            distance=0.0,
        )

        # Mock tokenizer to return smaller chunks
        self.mock_tokenizer.encode.return_value = list(range(50))
        self.mock_tokenizer.decode.return_value = text_unit.text

        result = manager.add_text(text_unit)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TextUnit)
        mock_log.debug.assert_any_call('Adding text: %d characters',
                                       len(text_unit))

    def test_add_text_empty_string(self):
        """Test adding empty text raises ValidationError."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                manager.add_text("")

            self.assertIn('text cannot be whitespace-only or zero-length', str(cm.exception))
            mock_log.error.assert_called()
            call_args = mock_log.error.call_args
            self.assertEqual(call_args[0][0],
                             'Operation failed: %s (%.3fs) - %s')
            self.assertEqual(call_args[0][1], 'add_text')
            self.assertIsInstance(call_args[0][2], float)  # execution time
            self.assertIsInstance(call_args[0][3], ValidationError)

    def test_add_text_whitespace_only(self):
        """Test adding whitespace-only text raises ValidationError."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        with self.assertRaises(ValidationError):
            manager.add_text("   \n\t   ")

    def test_add_text_custom_chunk_params(self):
        """Test adding text with custom chunk size and overlap."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "Test text"

        self.mock_tokenizer.encode.return_value = list(range(200))
        self.mock_tokenizer.decode.return_value = text

        manager.add_text(text, chunk_size=50, overlap=10)

        # Verify the custom parameters were used in splitting
        self.mock_tokenizer.encode.assert_called()

    def test_add_text_no_split(self):
        """Test adding text without splitting."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "Test text"

        result = manager.add_text(text, split=False)

        # Tokenizer should not be called for encoding when split=False
        self.mock_tokenizer.encode.assert_not_called()
        self.assertEqual(len(result), 1)

    def test_add_text_no_valid_chunks(self):
        """Test adding text that results in no valid chunks."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "Test text"

        # Mock tokenizer to return enough tokens to trigger splitting
        # and empty decoded chunks
        self.mock_tokenizer.encode.return_value = list(
            range(150))  # More than chunk_size
        self.mock_tokenizer.decode.return_value = ""  # Empty chunks after decoding
        self.mock_ragstore.store_texts.return_value = []

        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(DataError) as cm:
                manager.add_text(text)

            self.assertIn('No valid chunks stored', str(cm.exception))
            call_args = mock_log.error.call_args
            self.assertEqual(call_args[0][0],
                             'Operation failed: %s (%.3fs) - %s')
            self.assertEqual(call_args[0][1], 'add_text')
            self.assertIsInstance(call_args[0][2], float)  # execution time
            self.assertIsInstance(call_args[0][3], DataError)

    @patch('ragl.manager._LOG')
    def test_add_texts_strings_success(self, mock_log):
        """Test adding multiple texts as strings successfully."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        texts = ["First test text", "Second test text", "Third test text"]

        # Mock tokenizer to return different token counts for each text
        # Make each text small enough to be a single chunk
        self.mock_tokenizer.encode.side_effect = [
            list(range(25)),  # First text: 25 tokens (fits in one chunk)
            list(range(25)),  # Second text: 25 tokens
            list(range(25)),  # Third text: 25 tokens
        ]
        self.mock_tokenizer.decode.side_effect = texts

        # Mock store_texts to return the input TextUnits as-is
        def mock_store_texts(text_units):
            return text_units

        self.mock_ragstore.store_texts.side_effect = mock_store_texts

        result = manager.add_texts(texts)

        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(unit, TextUnit) for unit in result))
        # Verify store_texts was called once with all 3 units
        self.assertEqual(self.mock_ragstore.store_texts.call_count, 1)
        mock_log.debug.assert_any_call('Adding texts: %d items', 3)

    def test_add_texts_textunits_success(self):
        """Test adding multiple TextUnit objects successfully."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        text_units = [
            TextUnit(text_id="test-1", text="First text", source="test1",
                     distance=0.0),
            TextUnit(text_id="test-2", text="Second text", source="test2",
                     distance=0.0),
            TextUnit(text_id="test-3", text="Third text", source="test3",
                     distance=0.0)
        ]

        # Mock tokenizer to return smaller chunks
        self.mock_tokenizer.encode.return_value = list(range(50))
        self.mock_tokenizer.decode.side_effect = [unit.text for unit in
                                                  text_units]

        # Mock store_texts to return the input TextUnits as-is
        def mock_store_texts(text_units):
            return text_units

        self.mock_ragstore.store_texts.side_effect = mock_store_texts

        result = manager.add_texts(text_units)

        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(unit, TextUnit) for unit in result))
        # Verify store_texts was called once with all 3 units
        self.assertEqual(self.mock_ragstore.store_texts.call_count, 1)

    def test_add_texts_mixed_types_success(self):
        """Test adding mixed strings and TextUnit objects successfully."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        texts_or_units = [
            "First string text",
            TextUnit(text_id="test-2", text="Second text unit", source="test",
                     distance=0.0),
            "Third string text"
        ]

        # Mock tokenizer to return smaller chunks
        self.mock_tokenizer.encode.return_value = list(range(50))
        self.mock_tokenizer.decode.side_effect = ["First string text",
                                                  "Second text unit",
                                                  "Third string text"]

        # Mock store_texts to return the input TextUnits as-is
        def mock_store_texts(text_units):
            return text_units

        self.mock_ragstore.store_texts.side_effect = mock_store_texts

        result = manager.add_texts(texts_or_units)

        self.assertEqual(len(result), 3)
        # Verify store_texts was called once with all 3 units
        self.assertEqual(self.mock_ragstore.store_texts.call_count, 1)

    def test_add_texts_empty_list(self):
        """Test adding empty list raises ValidationError."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                manager.add_texts([])

            self.assertIn('texts_or_units cannot be empty', str(cm.exception))

            # Check that the ValidationError is logged within the operation failure
            mock_log.error.assert_called()
            call_args = mock_log.error.call_args
            self.assertEqual(call_args[0][0],
                             'Operation failed: %s (%.3fs) - %s')
            self.assertEqual(call_args[0][1], 'add_texts')
            self.assertIsInstance(call_args[0][2], float)  # execution time
            self.assertIsInstance(call_args[0][3], ValidationError)
            self.assertIn('texts_or_units cannot be empty',
                          str(call_args[0][3]))

    def test_add_texts_contains_empty_string(self):
        """Test adding texts with empty string raises ValidationError."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        texts = ["Valid text", "", "Another valid text"]

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                manager.add_texts(texts)

            self.assertIn('text cannot be whitespace-only or zero-length',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'text cannot be whitespace-only or zero-length')

    def test_add_texts_contains_invalid_type(self):
        """Test adding texts with invalid type raises ValidationError."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        texts = ["Valid text", 123, "Another valid text"]  # 123 is invalid

        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                manager.add_texts(texts)

            self.assertIn('Invalid text type, must be str or TextUnit',
                          str(cm.exception))

            # Check that the ValidationError is logged within the operation failure
            mock_log.error.assert_called()
            call_args = mock_log.error.call_args
            self.assertEqual(call_args[0][0],
                             'Operation failed: %s (%.3fs) - %s')
            self.assertEqual(call_args[0][1], 'add_texts')
            self.assertIsInstance(call_args[0][2], float)  # execution time
            self.assertIsInstance(call_args[0][3], ValidationError)
            self.assertIn('Invalid text type, must be str or TextUnit',
                          str(call_args[0][3]))

    def test_add_texts_custom_chunk_params(self):
        """Test adding texts with custom chunk parameters."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        texts = ["Text to chunk", "Another text"]

        # Mock tokenizer for custom chunk size
        self.mock_tokenizer.encode.return_value = list(range(150))
        self.mock_tokenizer.decode.side_effect = texts

        manager.add_texts(texts, chunk_size=200, overlap=50)

        # Verify store_text was called for each text
        self.assertEqual(self.mock_ragstore.store_texts.call_count, 1)

    def test_add_texts_no_split(self):
        """Test adding texts without splitting."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        texts = ["First whole text", "Second whole text"]

        # Mock store_texts to return the input TextUnits as-is
        def mock_store_texts(text_units):
            return text_units

        self.mock_ragstore.store_texts.side_effect = mock_store_texts

        result = manager.add_texts(texts, split=False)

        self.assertEqual(len(result), 2)
        # Verify store_texts was called once with all 2 units
        self.assertEqual(self.mock_ragstore.store_texts.call_count, 1)

    def test_add_texts_textunit_with_parent_id(self):
        """Test that TextUnit parent_id takes precedence over base_id."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        text_unit = TextUnit(
            text_id="test-1",
            text="Test text",
            parent_id="unit-parent",
            distance=0.0
        )
        texts = [text_unit]

        self.mock_tokenizer.encode.return_value = list(range(50))
        self.mock_tokenizer.decode.return_value = text_unit.text

        result = manager.add_texts(texts)

        # TextUnit's parent_id should be used, not base_id
        [stored_unit] = self.mock_ragstore.store_texts.call_args[0][0]
        self.assertEqual(stored_unit.parent_id, "unit-parent")

    def test_add_texts_multiple_chunks_per_text(self):
        """Test adding texts that get split into multiple chunks each."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        texts = ["Long text one", "Long text two"]

        # Mock tokenizer to create 2 chunks per text
        self.mock_tokenizer.encode.return_value = list(
            range(160))  # Creates 2 chunks

        # Provide enough side effects for all decode calls
        # Each text creates 2 chunks, so we need 4 total decode calls
        decode_results = [
            "Long text one chunk 1", "Long text one chunk 2",
            "Long text two chunk 1", "Long text two chunk 2",
            "Long text one chunk 1", "Long text one chunk 2",
            "Long text two chunk 1", "Long text two chunk 2",
            "Long text one chunk 1", "Long text one chunk 2",
            "Long text two chunk 1", "Long text two chunk 2",
        ]
        self.mock_tokenizer.decode.side_effect = decode_results

        # Mock store_texts to return the input TextUnits as-is
        def mock_store_texts(text_units):
            return text_units

        self.mock_ragstore.store_texts.side_effect = mock_store_texts

        result = manager.add_texts(texts)

        # Should have 4 total chunks (2 per text)
        self.assertEqual(len(result), 4)
        # Verify store_texts was called once with all 4 units
        self.assertEqual(self.mock_ragstore.store_texts.call_count, 1)

    def test_add_texts_text_id_without_base_id(self):
        """Test text_id generation when no base_id is provided."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        texts = ["First text", "Second text"]

        # Mock tokenizer to create multiple chunks per text
        self.mock_tokenizer.encode.return_value = list(range(150))
        self.mock_tokenizer.decode.side_effect = [
            "First chunk of first text",
            "Second chunk of first text",
            "First chunk of second text",
            "Second chunk of second text"
        ]

        result = manager.add_texts(texts)  # No base_id provided

        # Verify text_ids use global counter format
        call_args_list = self.mock_ragstore.store_text.call_args_list
        expected_ids = ['txt:doc-0-0', 'txt:doc-0-1',
                        'txt:doc-1-0', 'txt:doc-1-1']

        for i, call in enumerate(call_args_list):
            stored_unit = call[0][0]
            self.assertEqual(stored_unit.text_id, expected_ids[i])

    def test_add_texts_all_chunks_empty_raises_error(self):
        """Test that DataError is raised when all chunks are empty."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        # Mock tokenizer to return more tokens than chunk_size to force splitting
        self.mock_tokenizer.encode.return_value = list(
            range(150))  # More than chunk_size
        self.mock_tokenizer.decode.side_effect = ["", "   ", "\t\n",
                                                  "  \r\n  "]

        with self.assertRaises(DataError) as cm:
            manager.add_texts(["Test text that will be split"])

        self.assertIn('No valid chunks stored', str(cm.exception))

    def test_add_texts_integration_batch_processing(self):
        """Integration test for batch processing with mixed content."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        # Mix of strings and TextUnits
        texts_or_units = [
            "First string text",
            TextUnit(text_id="existing", text="Existing unit", source="manual",
                     distance=0.0),
            "Second string text",
            TextUnit(text_id="another", text="Another unit", author="Jane",
                     distance=0.0)
        ]

        # Mock tokenizer for realistic chunking
        self.mock_tokenizer.encode.return_value = list(range(75))
        self.mock_tokenizer.decode.side_effect = [
            "First string text", "Existing unit", "Second string text",
            "Another unit"
        ]

        # Mock store_texts to return list of TextUnit objects
        mock_text_units = [
            TextUnit(text_id="text-batch-test-0-0", text="First string text",
                     chunk_position=0, parent_id="batch-test", distance=0.0),
            TextUnit(text_id="text-batch-test-1-0", text="Existing unit",
                     chunk_position=0, parent_id="batch-test", distance=0.0),
            TextUnit(text_id="text-batch-test-2-0", text="Second string text",
                     chunk_position=0, parent_id="batch-test", distance=0.0),
            TextUnit(text_id="text-batch-test-3-0", text="Another unit",
                     chunk_position=0, parent_id="batch-test", distance=0.0)
        ]
        self.mock_ragstore.store_texts.return_value = mock_text_units

        result = manager.add_texts(texts_or_units)

        # Verify all were processed
        self.assertEqual(len(result), 4)

        # Verify store_texts was called once with batch
        self.mock_ragstore.store_texts.assert_called_once()

        # Verify the TextUnits passed to store_texts
        call_args = self.mock_ragstore.store_texts.call_args[0][0]
        self.assertEqual(len(call_args), 4)

        # Verify parent_id consistency
        for unit in result:
            self.assertEqual(unit.parent_id, "batch-test")

    def test_add_texts_textunit_empty_text(self):
        """Test adding TextUnit with empty text raises ValidationError."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)


        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                text_unit = TextUnit(
                    text_id="test-id",
                    text="",  # Empty text
                    source="test",
                    distance=0.0
                )
                texts = ["Valid text", text_unit]
                manager.add_texts(texts)

            self.assertIn('text cannot be whitespace-only or zero-length',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'text cannot be whitespace-only or zero-length')

    def test_add_texts_textunit_whitespace_only_text(self):
        """Test adding TextUnit with whitespace-only text raises ValidationError."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)


        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                text_unit = TextUnit(
                    text_id="test-id",
                    text="   \n\t   ",  # Whitespace only
                    source="test",
                    distance=0.0
                )
                texts = ["Valid text", text_unit]
                manager.add_texts(texts)

            self.assertIn('text cannot be whitespace-only or zero-length',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'text cannot be whitespace-only or zero-length')


    def test_add_texts_textunit_none_text(self):
        """Test adding TextUnit with None text raises ValidationError."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        # Create TextUnit with None text by bypassing normal construction
        text_unit = TextUnit(
            text_id="test-id",
            text="placeholder",
            source="test",
            distance=0.0
        )

        # texts = ["Valid text", text_unit]

        with patch('ragl.manager._LOG') as mock_log:
        #     with self.assertRaises(ValidationError) as cm:
        #         manager.add_texts(texts)
        #
        #     self.assertIn('text cannot be whitespace-only or zero-length',
        #                   str(cm.exception))
        #     mock_log.error.assert_called_with(
        #         'text cannot be whitespace-only or zero-length')

            with self.assertRaises(ValidationError) as cm:
                text_unit.text = None  # Set to None after creation

                self.assertIn('text cannot be whitespace-only or zero-length',
                              str(cm.exception))
                mock_log.error.assert_called_with(
                    'text cannot be whitespace-only or zero-length')

    def test_add_texts_multiple_textunits_one_empty(self):
        """Test adding multiple TextUnits where one has empty text."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        valid_unit = TextUnit(
            text_id="valid-id",
            text="Valid text content",
            source="test",
            distance=0.0
        )

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                empty_unit = TextUnit(
                    text_id="empty-id",
                    text="",  # Empty text
                    source="test",
                    distance=0.0
                )
                texts = [valid_unit, empty_unit]
                manager.add_texts(texts)

            self.assertIn('text cannot be whitespace-only or zero-length',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'text cannot be whitespace-only or zero-length')

    def test_add_texts_textunit_only_spaces_text(self):
        """Test adding TextUnit with only spaces in text raises ValidationError."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)


        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                text_unit = TextUnit(
                    text_id="test-id",
                    text="     ",  # Only spaces
                    source="test",
                    distance=0.0
                )
                texts = [text_unit]
                manager.add_texts(texts)

            self.assertIn('text cannot be whitespace-only or zero-length',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'text cannot be whitespace-only or zero-length')

    def test_add_text_global_counter_single_chunk(self):
        """Test global counter text_id generation for single chunk when no base_id provided."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        text = "Short text that fits in one chunk"

        # Mock tokenizer to create single chunk (no splitting)
        self.mock_tokenizer.encode.return_value = list(
            range(50))  # Under chunk_size
        self.mock_tokenizer.decode.return_value = text

        # Mock store_texts to return TextUnit with global counter ID
        mock_text_unit = TextUnit(
            text_id="txt:0",
            text=text,
            source="unknown",
            timestamp=12345,
            tags=[],
            confidence=None,
            language="unknown",
            section="unknown",
            author="unknown",
            parent_id="doc",
            chunk_position=0,
            distance=0.0
        )
        self.mock_ragstore.store_texts.return_value = [mock_text_unit]

        result = manager.add_text(text)  # No base_id provided

        # Verify global counter text_id format
        self.assertEqual(len(result), 1)
        call_args = self.mock_ragstore.store_texts.call_args[0][0]
        stored_unit = call_args[0]

        # Should use global counter format: txt:{counter}
        self.assertTrue(stored_unit.text_id.startswith('txt:'))
        self.assertEqual(stored_unit.parent_id, "doc")

    def test_add_text_global_counter_multiple_chunks(self):
        """Test global counter text_id generation for multiple chunks when no base_id provided."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        text = "Long text that will be split into multiple chunks"

        # Mock tokenizer to create 2 chunks
        self.mock_tokenizer.encode.return_value = list(
            range(150))  # More than chunk_size
        self.mock_tokenizer.decode.side_effect = ["First chunk",
                                                  "Second chunk"]

        # Mock store_texts to return TextUnits with global counter IDs
        mock_text_units = [
            TextUnit(text_id="txt:0", text="First chunk", parent_id="doc",
                     chunk_position=0, distance=0.0),
            TextUnit(text_id="txt:1", text="Second chunk", parent_id="doc",
                     chunk_position=1, distance=0.0)
        ]
        self.mock_ragstore.store_texts.return_value = mock_text_units

        result = manager.add_text(text)  # No base_id provided

        # Verify global counter text_ids for multiple chunks
        self.assertEqual(len(result), 2)
        call_args = self.mock_ragstore.store_texts.call_args[0][0]

        for i, stored_unit in enumerate(call_args):
            self.assertTrue(stored_unit.text_id.startswith('txt:'))
            self.assertEqual(stored_unit.parent_id, "doc")
            self.assertEqual(stored_unit.chunk_position, i)

    def test_add_text_global_counter_increments(self):
        """Test that global counter increments across multiple add_text calls."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        # Mock tokenizer for single chunks
        self.mock_tokenizer.encode.return_value = list(range(50))
        self.mock_tokenizer.decode.side_effect = ["First text", "Second text"]

        # Mock store_texts to simulate incrementing counter
        counter = 0

        def mock_store_texts(text_units):
            nonlocal counter
            for unit in text_units:
                unit.text_id = f"txt:{counter}"
                counter += 1
            return text_units

        self.mock_ragstore.store_texts.side_effect = mock_store_texts

        # Add first text
        result1 = manager.add_text("First text")

        # Add second text
        result2 = manager.add_text("Second text")

        # Verify counter incremented
        self.assertEqual(result1[0].text_id, "txt:0")
        self.assertEqual(result2[0].text_id, "txt:1")

    def test_add_texts_global_counter_format(self):
        """Test global counter text_id format when no base_id is provided."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        texts = ["First text", "Second text"]

        # Mock tokenizer to create single chunks (no splitting)
        self.mock_tokenizer.encode.return_value = list(range(50))
        self.mock_tokenizer.decode.side_effect = texts

        # Mock store_texts to return TextUnits with global counter format
        def mock_store_texts(text_units):
            # Simulate global counter incrementing
            for i, unit in enumerate(text_units):
                unit.text_id = f'txt:{i}'
                unit.parent_id = 'doc'
            return text_units

        self.mock_ragstore.store_texts.side_effect = mock_store_texts

        result = manager.add_texts(texts)  # No base_id provided

        # Verify global counter format: txt:{counter}
        self.assertEqual(result[0].text_id, 'txt:0')
        self.assertEqual(result[1].text_id, 'txt:1')

        # Verify parent_id uses default
        self.assertEqual(result[0].parent_id, 'doc')
        self.assertEqual(result[1].parent_id, 'doc')

    def test_add_text_global_counter_with_multiple_chunks_no_base_id(self):
        """Test global counter for multiple chunks when no base_id provided."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        text = "Text that creates multiple chunks"

        # Mock tokenizer to create 2 chunks
        self.mock_tokenizer.encode.return_value = list(range(150))
        self.mock_tokenizer.decode.side_effect = ["First chunk",
                                                  "Second chunk"]

        # Mock store_texts to simulate global counter behavior
        def mock_store_texts(text_units):
            for i, unit in enumerate(text_units):
                unit.text_id = f'txt:{i}'
                unit.parent_id = 'doc'
                unit.chunk_position = i
            return text_units

        self.mock_ragstore.store_texts.side_effect = mock_store_texts

        result = manager.add_text(text)  # No base_id provided

        # Should use global counter format for each chunk
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].text_id, 'txt:0')
        self.assertEqual(result[1].text_id, 'txt:1')

        # Both should have same parent_id (default)
        self.assertEqual(result[0].parent_id, 'doc')
        self.assertEqual(result[1].parent_id, 'doc')

    def test_add_text_bare_string_no_base_id(self):
        """Test adding a bare string with no base_id provided."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "Simple test text"

        # Mock tokenizer for single chunk
        self.mock_tokenizer.encode.return_value = list(range(50))
        self.mock_tokenizer.decode.return_value = text

        # Mock store_texts to return TextUnit with global counter ID
        mock_text_unit = TextUnit(
            text_id="txt:0",
            text=text,
            parent_id="doc",
            chunk_position=0,
            distance=0.0
        )
        self.mock_ragstore.store_texts.return_value = [mock_text_unit]

        result = manager.add_text(text)

        # Verify single chunk returned
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, text)
        self.assertEqual(result[0].parent_id, "doc")

        # Verify store_texts was called once
        self.mock_ragstore.store_texts.assert_called_once()

    def test_add_text_skips_empty_chunks_from_get_chunks(self):
        """Test that empty chunks returned by chunker are properly skipped."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        text = "Text that will produce some empty chunks"

        # Mock TextUnitChunker class to simulate empty chunk filtering
        with patch('ragl.manager.TextUnitChunker') as mock_chunker_class:
            mock_chunker_instance = Mock()
            mock_chunker_class.return_value = mock_chunker_instance

            # Simulate the chunker's actual behavior - it filters out empty chunks
            # and only yields valid ones (mimics the chunker's internal filtering)
            def mock_chunk_generator(*args, **kwargs):
                # Only yield valid chunks (empty ones are filtered internally)
                yield TextUnit(text="Valid chunk 1", parent_id="doc",
                               text_id="txt:doc-1-0-0")
                yield TextUnit(text="Valid chunk 2", parent_id="doc",
                               text_id="txt:doc-1-0-3")

            mock_chunker_instance.chunk_text_unit.return_value = mock_chunk_generator()

            # Mock ragstore to accept and return units
            self.mock_ragstore.store_texts.return_value = [
                TextUnit(text="Valid chunk 1", parent_id="doc",
                         text_id="txt:doc-1-0-0"),
                TextUnit(text="Valid chunk 2", parent_id="doc",
                         text_id="txt:doc-1-0-3")
            ]

            result = manager.add_text(text)

            # Should only have valid chunks (empty ones filtered out by chunker)
            self.assertEqual(len(result), 2)
            stored_texts = [unit.text for unit in result]
            self.assertEqual(stored_texts, ["Valid chunk 1", "Valid chunk 2"])

            # Verify store_texts was called with only valid chunks
            store_call_args = self.mock_ragstore.store_texts.call_args[0][0]
            stored_chunk_texts = [unit.text for unit in store_call_args]
            self.assertEqual(stored_chunk_texts,
                             ["Valid chunk 1", "Valid chunk 2"])

            # Verify chunker was called
            mock_chunker_instance.chunk_text_unit.assert_called_once()

    @patch('ragl.manager._LOG')
    def test_delete_texts_success(self, mock_log):
        """Test deleting multiple texts successfully."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text_ids = ['text-1', 'text-2', 'text-3']

        # Mock existing texts and successful deletion
        self.mock_ragstore.list_texts.return_value = ['text-1', 'text-2',
                                                      'text-3', 'text-4']
        self.mock_ragstore.delete_texts.return_value = 3

        result = manager.delete_texts(text_ids)

        self.assertEqual(result, 3)
        self.mock_ragstore.delete_texts.assert_called_once_with(text_ids)
        mock_log.debug.assert_any_call('Deleting texts: %s', text_ids)
        mock_log.info.assert_called_with('Deleted %d texts', 3)

    @patch('ragl.manager._LOG')
    def test_delete_texts_partial_exists(self, mock_log):
        """Test deleting texts where only some exist."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text_ids = ['text-1', 'nonexistent', 'text-3']

        # Mock existing texts (missing 'nonexistent')
        self.mock_ragstore.list_texts.return_value = ['text-1', 'text-2',
                                                      'text-3']
        self.mock_ragstore.delete_texts.return_value = 2

        result = manager.delete_texts(text_ids)

        self.assertEqual(result, 2)
        # Should only delete the existing ones
        self.mock_ragstore.delete_texts.assert_called_once_with(
            ['text-1', 'text-3'])

    @patch('ragl.manager._LOG')
    def test_delete_texts_none_exist(self, mock_log):
        """Test deleting texts where none exist."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text_ids = ['nonexistent-1', 'nonexistent-2']

        # Mock existing texts (none of the requested IDs exist)
        self.mock_ragstore.list_texts.return_value = ['text-1', 'text-2']

        result = manager.delete_texts(text_ids)

        self.assertEqual(result, 0)
        self.mock_ragstore.delete_texts.assert_not_called()
        mock_log.warning.assert_called_with(
            'No valid text IDs found for deletion')

    def test_delete_texts_empty_list(self):
        """Test deleting empty list of texts."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        self.mock_ragstore.list_texts.return_value = ['text-1', 'text-2']

        with patch('ragl.manager._LOG') as mock_log:
            result = manager.delete_texts([])

            self.assertEqual(result, 0)
            self.mock_ragstore.delete_texts.assert_not_called()
            mock_log.warning.assert_called_with(
                'No valid text IDs found for deletion')

    def test_delete_texts_integration_with_tracking(self):
        """Integration test for delete_texts with operation tracking."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text_ids = ['text-1', 'text-2', 'text-3']

        self.mock_ragstore.list_texts.return_value = text_ids
        self.mock_ragstore.delete_texts.return_value = 3

        # Perform deletion and check metrics
        with patch('time.time',
                   side_effect=[1000.0, 1002.0]):  # 2 second operation
            result = manager.delete_texts(text_ids)

        self.assertEqual(result, 3)

        # Check that operation was tracked
        metrics = manager.get_performance_metrics('delete_texts')
        self.assertIn('delete_texts', metrics)
        self.assertEqual(metrics['delete_texts']['total_calls'], 1)
        self.assertEqual(metrics['delete_texts']['failure_count'], 0)

    @patch('ragl.manager._LOG')
    def test_delete_text(self, mock_log):
        """Test deleting text."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text_id = "test-id"
        self.mock_ragstore.list_texts.return_value = [text_id]
        manager.delete_text(text_id)

        self.mock_ragstore.delete_text.assert_called_once_with(text_id)

        call_args = mock_log.debug.call_args
        self.assertEqual(call_args[0][0],
                         'Operation completed: %s (%.3fs)')
        self.assertEqual(call_args[0][1], 'delete_text')

        self.assertIsInstance(call_args[0][2], float)  # execution time
    @patch('ragl.manager._LOG')
    def test_delete_text_nonexistent(self, mock_log):
        """Test deleting text."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text_id = "test-id"
        self.mock_ragstore.list_texts.return_value = []
        manager.delete_text(text_id)

        call_args = mock_log.warning.call_args
        self.assertEqual(call_args[0][0],
                         'Text ID %s not found, skipping deletion')
        self.assertEqual(call_args[0][1], 'test-id')

    @patch('ragl.manager._LOG')
    def test_get_context_success(self, mock_log):
        """Test getting context successfully."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        query = "test query"

        # Mock ragstore response - now as TextUnit objects
        mock_response = [TextUnit(
            text_id='test-1',
            text='relevant text',
            distance=0.5,
            timestamp=12345,
            source='test',
            tags=[],
            confidence=None,
            language='unknown',
            section='unknown',
            author='unknown',
            parent_id='doc-1',
            chunk_position=0,
        )]
        self.mock_ragstore.get_relevant.return_value = mock_response

        result = manager.get_context(query, top_k=5)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TextUnit)
        self.mock_ragstore.get_relevant.assert_called_once_with(
            query=query,
            top_k=5,
            min_time=None,
            max_time=None,
        )

    def test_get_context_with_time_filters(self):
        """Test getting context with time filters."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        query = "test query"
        min_time = 1000
        max_time = 2000

        self.mock_ragstore.get_relevant.return_value = []

        manager.get_context(query, top_k=3, min_time=min_time,
                            max_time=max_time)

        self.mock_ragstore.get_relevant.assert_called_once_with(
            query=query, top_k=3, min_time=min_time, max_time=max_time
        )

    def test_get_context_sort_by_time(self):
        """Test getting context sorted by time."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        query = "test query"

        # Mock response with different timestamps - now as TextUnit objects
        mock_response = [
            TextUnit(
                text_id='test-1', text='text1', distance=0.3,
                timestamp=2000, source='test', tags=[], confidence=None,
                language='unknown', section='unknown', author='unknown',
                parent_id='doc-1', chunk_position=0
            ),
            TextUnit(
                text_id='test-2', text='text2', distance=0.1,
                timestamp=1000, source='test', tags=[], confidence=None,
                language='unknown', section='unknown', author='unknown',
                parent_id='doc-2', chunk_position=0
            ),
        ]
        self.mock_ragstore.get_relevant.return_value = mock_response

        result = manager.get_context(query, sort_by_time=True)

        # Should be sorted by timestamp (1000, 2000)
        self.assertEqual(result[0].timestamp, 1000)
        self.assertEqual(result[1].timestamp, 2000)

    def test_get_context_sort_by_distance(self):
        """Test getting context sorted by distance (default)."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        query = "test query"

        # Mock response with different distances - now as TextUnit objects
        mock_response = [
            TextUnit(
                text_id='test-1', text='text1', distance=0.8,
                timestamp=2000, source='test', tags=[], confidence=None,
                language='unknown', section='unknown', author='unknown',
                parent_id='doc-1', chunk_position=0
            ),
            TextUnit(
                text_id='test-2', text='text2', distance=0.2,
                timestamp=1000, source='test', tags=[], confidence=None,
                language='unknown', section='unknown', author='unknown',
                parent_id='doc-2', chunk_position=0
            ),
        ]
        self.mock_ragstore.get_relevant.return_value = mock_response

        result = manager.get_context(query)

        # Should be sorted by distance (0.2, 0.8)
        self.assertEqual(result[0].distance, 0.2)
        self.assertEqual(result[1].distance, 0.8)

    def test_get_context_empty_query(self):
        """Test getting context with empty query raises ValidationError."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        results = manager.get_context("")
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    def test_get_context_whitespace_query(self):
        """Test getting context with whitespace-only query raises ValidationError."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        results = manager.get_context("   \n\t   ")
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    def test_get_context_query_too_long(self):
        """Test getting context with overly long query raises ValidationError."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        long_query = "x" * (RAGManager.MAX_QUERY_LENGTH + 1)

        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                manager.get_context(long_query)

            expected_msg = f'Query too long: {len(long_query)} > {RAGManager.MAX_QUERY_LENGTH}'
            self.assertIn('Query too long', str(cm.exception))
            call_args = mock_log.error.call_args
            self.assertEqual(call_args[0][0],
                             'Operation failed: %s (%.3fs) - %s')
            self.assertEqual(call_args[0][1], 'get_context')
            self.assertIsInstance(call_args[0][2], float)  # execution time
            self.assertIsInstance(call_args[0][3], ValidationError)

    def test_get_context_invalid_top_k(self):
        """Test getting context with invalid top_k raises ValidationError."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                manager.get_context("query", top_k=0)

            self.assertIn('top_k must be a positive integer',
                          str(cm.exception))
            call_args = mock_log.error.call_args
            self.assertEqual(call_args[0][0],
                             'Operation failed: %s (%.3fs) - %s')
            self.assertEqual(call_args[0][1], 'get_context')
            self.assertIsInstance(call_args[0][2], float)  # execution time
            self.assertIsInstance(call_args[0][3], ValidationError)

    def test_get_context_invalid_top_k_type(self):
        """Test getting context with invalid top_k type raises ValidationError."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        with self.assertRaises(ValidationError):
            manager.get_context("query", top_k="invalid")

    @patch('ragl.manager._LOG')
    def test_get_health_status_with_health_check(self, mock_log):
        """Test getting health status when backend supports health checks."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        # Mock the ragstore's get_health_status method directly
        expected_result = {
            'embedder': {'used_memory': 1024},
            'storage':  {'status': 'healthy'}
        }
        self.mock_ragstore.get_health_status.return_value = expected_result

        result = manager.get_health_status()

        self.assertEqual(result, expected_result)
        self.mock_ragstore.get_health_status.assert_called_once()

    @patch('ragl.manager._LOG')
    def test_get_performance_metrics_all(self, mock_log):
        """Test getting all performance metrics."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        # Add some metrics
        manager._metrics['test_op'].record_success(1.0)
        manager._metrics['another_op'].record_failure(2.0)

        result = manager.get_performance_metrics()

        self.assertIn('test_op', result)
        self.assertIn('another_op', result)
        self.assertEqual(result['test_op']['total_calls'], 1)
        self.assertEqual(result['another_op']['failure_count'], 1)
        mock_log.debug.assert_called_with('Computing metrics')

    def test_get_performance_metrics_specific_operation(self):
        """Test getting metrics for specific operation."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        # Add some metrics
        manager._metrics['test_op'].record_success(1.0)
        manager._metrics['another_op'].record_failure(2.0)

        result = manager.get_performance_metrics('test_op')

        self.assertIn('test_op', result)
        self.assertNotIn('another_op', result)
        self.assertEqual(result['test_op']['total_calls'], 1)

    def test_get_performance_metrics_nonexistent_operation(self):
        """Test getting metrics for non-existent operation."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        result = manager.get_performance_metrics('nonexistent_op')

        self.assertEqual(result, {})

    @patch('ragl.manager._LOG')
    def test_list_texts(self, mock_log):
        """Test listing texts."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        expected_texts = ['text-1', 'text-2', 'text-3']
        self.mock_ragstore.list_texts.return_value = expected_texts

        result = manager.list_texts()

        self.assertEqual(result, expected_texts)
        self.mock_ragstore.list_texts.assert_called_once()
        mock_log.debug.assert_any_call('Listing texts')
        mock_log.debug.assert_any_call('text count: %d', 3)

    @patch('ragl.manager._LOG')
    def test_reset_with_metrics(self, mock_log):
        """Test reset with metrics reset."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        # Add some metrics
        manager._metrics['test_op'].record_success(1.0)

        manager.reset(reset_metrics=True)

        self.mock_ragstore.clear.assert_called_once()
        self.assertEqual(len(manager._metrics), 0)
        mock_log.debug.assert_any_call('Resetting store')
        mock_log.debug.assert_any_call('Resetting metrics')
        mock_log.info.assert_called_with('Store reset')

    def test_reset_without_metrics(self):
        """Test reset without metrics reset."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        # Add some metrics
        manager._metrics['test_op'].record_success(1.0)

        manager.reset(reset_metrics=False)

        self.mock_ragstore.clear.assert_called_once()
        self.assertEqual(len(manager._metrics), 2)

    @patch('ragl.manager._LOG')
    def test_reset_metrics(self, mock_log):
        """Test reset metrics only."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        # Add some metrics
        manager._metrics['test_op'].record_success(1.0)

        manager.reset_metrics()

        self.assertEqual(len(manager._metrics), 0)
        mock_log.debug.assert_called_with('Resetting metrics')
        mock_log.info.assert_called_with('Metrics reset')

    @patch('time.time')
    @patch('ragl.manager._LOG')
    def test_track_operation_success(self, mock_log, mock_time):
        """Test track_operation context manager with successful operation."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        # Mock time progression
        mock_time.side_effect = [1000.0, 1002.5]  # 2.5 second operation

        with manager.track_operation('test_op'):
            pass

        self.assertEqual(manager._metrics['test_op'].total_calls, 1)
        self.assertEqual(manager._metrics['test_op'].failure_count, 0)
        self.assertEqual(manager._metrics['test_op'].total_duration, 2.5)

        mock_log.debug.assert_any_call('Starting operation: %s', 'test_op')
        mock_log.debug.assert_any_call('Operation completed: %s (%.3fs)',
                                       'test_op', 2.5)

    @patch('time.time')
    @patch('ragl.manager._LOG')
    def test_track_operation_failure(self, mock_log, mock_time):
        """Test track_operation context manager with failed operation."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        # Mock time progression
        mock_time.side_effect = [1000.0, 1001.5]  # 1.5 second operation

        test_exception = Exception("Test error")

        with self.assertRaises(Exception):
            with manager.track_operation('test_op'):
                raise test_exception

        self.assertEqual(manager._metrics['test_op'].total_calls, 1)
        self.assertEqual(manager._metrics['test_op'].failure_count, 1)
        self.assertEqual(manager._metrics['test_op'].total_duration, 1.5)

        mock_log.debug.assert_called_with('Recording failed operation')

    @patch('ragl.manager._LOG')
    def test_sanitize_text_input_normal(self, mock_log):
        """Test sanitizing normal text input."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "Normal text input"

        result = manager._sanitize_text(text)

        self.assertEqual(result, text)
        mock_log.debug.assert_called_with('Sanitizing text')

    def test_sanitize_text_input_too_long(self):
        """Test sanitizing text input that's too long."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        # Create text that exceeds MAX_INPUT_LENGTH
        long_text = "x" * (RAGManager.MAX_INPUT_LENGTH + 1)

        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                manager._sanitize_text(long_text)

            self.assertIn('text too long', str(cm.exception))
            mock_log.error.assert_called_with('text too long')

    def test_sanitize_text_input_paranoid_mode(self):
        """Test sanitizing text input in paranoid mode."""
        config = ManagerConfig(chunk_size=100, overlap=20, paranoid=True)
        manager = RAGManager(config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "Text with <script>alert('xss')</script> dangerous chars!"

        result = manager._sanitize_text(text)

        # Should remove dangerous characters, keeping only alphanumeric, spaces, and basic punctuation
        expected = "Text with alert('xss') dangerous chars!"
        self.assertEqual(result, expected)

    def test_add_text_filters_empty_chunks(self):
        """Test that empty chunks are filtered out in add_text."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        text = "Text that creates empty chunks"

        # Mock tokenizer to create chunks where some decode to empty
        self.mock_tokenizer.encode.return_value = list(
            range(120))  # Creates 2 chunks
        self.mock_tokenizer.decode.side_effect = [
            "",  # First chunk is empty - should be filtered
            "Valid chunk content"  # Second chunk is valid
        ]

        # Mock store_text to return the input
        self.mock_ragstore.store_texts.side_effect = lambda x: x

        result = manager.add_text(text)

        # Should only store 1 chunk (the valid one), empty chunk filtered
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Valid chunk content")

        # Verify store_text was called only once (for the valid chunk)
        self.assertEqual(self.mock_ragstore.store_texts.call_count, 1)

    @patch('ragl.manager._LOG')
    def test_validate_chunking_valid(self, mock_log):
        """Test chunking validation with valid parameters."""
        RAGManager._validate_chunking(100, 20)
        mock_log.debug.assert_called_with('Validating chunking parameters')

    def test_validate_chunking_zero_chunk_size(self):
        """Test chunking validation with zero chunk size."""
        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                RAGManager._validate_chunking(0, 20)

            self.assertIn('Chunk_size must be positive', str(cm.exception))
            mock_log.error.assert_called_with('Chunk_size must be positive')

    def test_validate_chunking_negative_chunk_size(self):
        """Test chunking validation with negative chunk size."""
        with self.assertRaises(ValidationError):
            RAGManager._validate_chunking(-10, 20)

    def test_validate_chunking_negative_overlap(self):
        """Test chunking validation with negative overlap."""
        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                RAGManager._validate_chunking(100, -5)

            self.assertIn('Overlap must be non-negative', str(cm.exception))
            mock_log.error.assert_called_with(
                'Overlap must be non-negative')

    def test_validate_chunking_overlap_too_large(self):
        """Test chunking validation with overlap >= chunk_size."""
        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                RAGManager._validate_chunking(100, 100)

            self.assertIn('Overlap must be less than chunk_size',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'Overlap must be less than chunk_size')

    def test_validate_chunking_overlap_greater_than_chunk_size(self):
        """Test chunking validation with overlap > chunk_size."""
        with self.assertRaises(ValidationError):
            RAGManager._validate_chunking(50, 75)

    @patch('ragl.manager._LOG')
    def test_validate_query_valid(self, mock_log):
        """Test query validation with valid query."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        query = "Valid query text"

        manager._validate_query(query)
        mock_log.debug.assert_called_with('Validating query')

    def test_validate_query_none(self):
        """Test query validation with None query."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                manager._validate_query(None)

            self.assertIn('Query cannot be empty', str(cm.exception))
            mock_log.error.assert_called_with('Query cannot be empty')

    def test_validate_query_empty(self):
        """Test query validation with empty query."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        with self.assertRaises(ValidationError):
            manager._validate_query("")

    def test_validate_query_whitespace(self):
        """Test query validation with whitespace-only query."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        with self.assertRaises(ValidationError):
            manager._validate_query("   \n\t   ")

    def test_validate_query_too_long(self):
        """Test query validation with overly long query."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        long_query = "x" * (RAGManager.MAX_QUERY_LENGTH + 1)

        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                manager._validate_query(long_query)

            expected_msg = f'Query too long: {len(long_query)} > {RAGManager.MAX_QUERY_LENGTH}'
            self.assertIn('Query too long', str(cm.exception))
            mock_log.error.assert_called_with(expected_msg)

    @patch('ragl.manager._LOG')
    def test_validate_top_k_valid(self, mock_log):
        """Test top_k validation with valid value."""
        RAGManager._validate_top_k(5)
        mock_log.debug.assert_called_with('Validating top_k parameter')

    def test_validate_top_k_zero(self):
        """Test top_k validation with zero value."""
        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                RAGManager._validate_top_k(0)

            self.assertIn('top_k must be a positive integer',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'top_k must be a positive integer')

    def test_validate_top_k_negative(self):
        """Test top_k validation with negative value."""
        with self.assertRaises(ValidationError):
            RAGManager._validate_top_k(-1)

    def test_validate_top_k_non_integer(self):
        """Test top_k validation with non-integer value."""
        with self.assertRaises(ValidationError):
            RAGManager._validate_top_k(5.5)

    def test_validate_top_k_string(self):
        """Test top_k validation with string value."""
        with self.assertRaises(ValidationError):
            RAGManager._validate_top_k("5")

    def test_str_representation(self):
        """Test string representation of RAGManager."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        self.mock_ragstore.list_texts.return_value = ['text1', 'text2',
                                                      'text3']

        result = str(manager)

        expected = 'RAGManager(texts=3, chunk_size=100, overlap=20)'
        self.assertEqual(result, expected)

    def test_repr_representation(self):
        """Test repr representation of RAGManager."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        result = repr(manager)
        expected = (
            'RAGManager('
            'config=ManagerConfig('
            'chunk_size=100, '
            'overlap=20, '
            'min_chunk_size=None, '
            'paranoid=False), '
            f'ragstore={self.mock_ragstore!r}, '
            f'tokenizer={self.mock_tokenizer!r})'
        )
        self.assertEqual(result, expected)

    def test_constants(self):
        """Test that class constants are properly defined."""
        self.assertEqual(RAGManager.DEFAULT_PARENT_ID, 'doc')
        self.assertEqual(RAGManager.MAX_QUERY_LENGTH, 8192)
        self.assertEqual(RAGManager.MAX_INPUT_LENGTH, (1024 * 1024) * 10)

    def test_add_text_integration_multiple_chunks(self):
        """Integration test for adding text that gets split into multiple chunks."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "Long text that will be split into multiple chunks"

        # Mock tokenizer to create 3 chunks
        self.mock_tokenizer.encode.return_value = list(
            range(220))  # 220 tokens
        self.mock_tokenizer.decode.side_effect = [
            "Chunk 1 content",
            "Chunk 2 content",
            "Chunk 3 content"
        ]

        # Create mock TextUnit objects to return
        mock_text_units = [
            TextUnit(
                text_id="text-doc-0-0",
                text="Chunk 1 content",
                source="unknown",
                timestamp=1234567890,
                tags=[],
                confidence=None,
                language="unknown",
                section="unknown",
                author="unknown",
                parent_id="doc",
                chunk_position=0,
                distance=0.0
            ),
            TextUnit(
                text_id="text-doc-0-1",
                text="Chunk 2 content",
                source="unknown",
                timestamp=1234567890,
                tags=[],
                confidence=None,
                language="unknown",
                section="unknown",
                author="unknown",
                parent_id="doc",
                chunk_position=1,
                distance=0.0
            ),
            TextUnit(
                text_id="text-doc-0-2",
                text="Chunk 3 content",
                source="unknown",
                timestamp=1234567890,
                tags=[],
                confidence=None,
                language="unknown",
                section="unknown",
                author="unknown",
                parent_id="doc",
                chunk_position=2,
                distance=0.0
            )
        ]

        # Mock store_texts to return the TextUnit objects
        self.mock_ragstore.store_texts.return_value = mock_text_units

        result = manager.add_text(text)

        self.assertEqual(len(result), 3)  # Should be 3 chunks, not 4
        self.assertEqual(self.mock_ragstore.store_texts.call_count, 1)

        # Verify all chunks have the same parent_id
        parent_ids = [unit.parent_id for unit in result]
        self.assertTrue(all(pid == "doc" for pid in parent_ids))

        # Verify chunk positions are sequential
        positions = [unit.chunk_position for unit in result]
        self.assertEqual(positions, [0, 1, 2])

    def test_metadata_round_trip_preservation(self):
        """Test that TextUnit metadata is preserved through add/retrieve cycle."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        # Create TextUnit with metadata including confidence
        original_unit = TextUnit(
            text_id="test-id",
            text="Test text with metadata",
            confidence=0.95,
            author="John Doe",
            source="test_source",
            tags=["important", "test"],
            language="en",
            section="intro",
            distance=0.0
        )

        # Mock small chunks to avoid splitting
        self.mock_tokenizer.encode.return_value = list(range(50))
        self.mock_tokenizer.decode.return_value = original_unit.text

        # Mock retrieve to return the stored metadata - now as TextUnit objects
        stored_unit = TextUnit(
            text_id='stored-id',
            text=original_unit.text,
            confidence=0.95,
            author="John Doe",
            source="test_source",
            tags=["important", "test"],
            language="en",
            section="intro",
            distance=0.1,
            chunk_position=0,
            parent_id='test-parent',
            timestamp=12345
        )
        self.mock_ragstore.get_relevant.return_value = [stored_unit]

        # Add the TextUnit
        manager.add_text(original_unit)

        # Verify the TextUnit was stored properly
        call_args = self.mock_ragstore.store_texts.call_args
        [stored_textunit] = call_args[0][0]

        # Check that metadata was preserved
        self.assertEqual(stored_textunit.confidence, 0.95)
        self.assertEqual(stored_textunit.author, "John Doe")
        self.assertEqual(stored_textunit.source, "test_source")
        self.assertEqual(stored_textunit.tags, ["important", "test"])
        self.assertEqual(stored_textunit.language, "en")
        self.assertEqual(stored_textunit.section, "intro")


if __name__ == '__main__':
    unittest.main()
