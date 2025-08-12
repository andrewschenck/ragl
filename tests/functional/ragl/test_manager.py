import logging
import statistics
import time
import unittest
from collections import deque
from contextlib import contextmanager
from unittest.mock import Mock, MagicMock, patch, call

from ragl.config import ManagerConfig
from ragl.constants import TEXT_ID_PREFIX
from ragl.exceptions import DataError, ValidationError, ConfigurationError
from ragl.manager import RAGManager, RAGTelemetry
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


class TestRAGManager(unittest.TestCase):
    """Test cases for RAGManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_ragstore = Mock(spec=RAGStoreProtocol)
        self.mock_tokenizer = Mock(spec=TokenizerProtocol)
        self.config = ManagerConfig(chunk_size=100, overlap=20, paranoid=False)

        # Setup default mock behaviors
        self.mock_ragstore.list_texts.return_value = []
        self.mock_ragstore.store_text.return_value = 'text-id-1'
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
            mock_log.critical.assert_called_with(
                'ragstore must implement RAGStoreProtocol')

    def test_init_invalid_tokenizer(self):
        """Test RAGManager initialization with invalid tokenizer."""
        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(TypeError) as cm:
                RAGManager(self.config, self.mock_ragstore,
                           tokenizer="invalid_tokenizer")

            self.assertIn('tokenizer must implement TokenizerProtocol',
                          str(cm.exception))
            mock_log.critical.assert_called_with(
                'tokenizer must implement TokenizerProtocol')

    def test_init_invalid_chunking_parameters(self):
        """Test RAGManager initialization with invalid chunking parameters."""

        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(ConfigurationError) as cm:
                invalid_config = ManagerConfig(chunk_size=0, overlap=20,
                                               paranoid=False)
                RAGManager(invalid_config, self.mock_ragstore)

            self.assertIn('self.chunk_size=0 must be positive', str(cm.exception))

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
        self.mock_ragstore.store_text.assert_called_once()
        mock_log.debug.assert_any_call('Adding text: %s', text)

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
        mock_log.debug.assert_any_call('Adding text: %s', text_unit)

    def test_add_text_empty_string(self):
        """Test adding empty text raises ValidationError."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                manager.add_text("")

            self.assertIn('Text cannot be empty', str(cm.exception))
            mock_log.critical.assert_called()
            call_args = mock_log.critical.call_args
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

    def test_add_text_with_base_id(self):
        """Test adding text with custom base_id."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "Test text"
        base_id = "custom-base"

        self.mock_tokenizer.encode.return_value = list(range(50))
        self.mock_tokenizer.decode.return_value = text

        result = manager.add_text(text, base_id=base_id)

        # Check that the parent_id is set to base_id
        call_args = self.mock_ragstore.store_text.call_args
        self.assertEqual(call_args[1]['metadata']['parent_id'], base_id)

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

        # Mock tokenizer to return empty decoded text
        self.mock_tokenizer.encode.return_value = list(range(50))
        self.mock_tokenizer.decode.return_value = ""

        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(DataError) as cm:
                manager.add_text(text)

            self.assertIn('No valid chunks stored', str(cm.exception))
            call_args = mock_log.critical.call_args
            self.assertEqual(call_args[0][0],
                             'Operation failed: %s (%.3fs) - %s')
            self.assertEqual(call_args[0][1], 'add_text')
            self.assertIsInstance(call_args[0][2], float)  # execution time
            self.assertIsInstance(call_args[0][3], DataError)

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

        # Mock ragstore response
        mock_response = [{
            'text_id':        'test-1',
            'text':           'relevant text',
            'distance':       0.5,
            'timestamp':      12345,
            'source':         'test',
            'tags':           [],
            'confidence':     None,
            'language':       'unknown',
            'section':        'unknown',
            'author':         'unknown',
            'parent_id':      'doc-1',
            'chunk_position': 0,
        }]
        self.mock_ragstore.get_relevant.return_value = mock_response

        result = manager.get_context(query, top_k=5)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], TextUnit)
        self.mock_ragstore.get_relevant.assert_called_once_with(
            query=query, top_k=5, min_time=None, max_time=None
        )
        call_args = mock_log.debug.call_args
        self.assertEqual(call_args[0][0],
                         'Operation completed: %s (%.3fs)')
        self.assertEqual(call_args[0][1], 'get_context')
        self.assertIsInstance(call_args[0][2], float)  # execution time

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

        # Mock response with different timestamps
        mock_response = [
            {
                'text_id':   'test-1', 'text': 'text1', 'distance': 0.3,
                'timestamp': 2000,
                'source':    'test', 'tags': [], 'confidence': None,
                'language':  'unknown',
                'section':   'unknown', 'author': 'unknown',
                'parent_id': 'doc-1', 'chunk_position': 0
            },
            {
                'text_id':   'test-2', 'text': 'text2', 'distance': 0.1,
                'timestamp': 1000,
                'source':    'test', 'tags': [], 'confidence': None,
                'language':  'unknown',
                'section':   'unknown', 'author': 'unknown',
                'parent_id': 'doc-2', 'chunk_position': 0
            },
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

        # Mock response with different distances
        mock_response = [
            {
                'text_id':   'test-1', 'text': 'text1', 'distance': 0.8,
                'timestamp': 2000,
                'source':    'test', 'tags': [], 'confidence': None,
                'language':  'unknown',
                'section':   'unknown', 'author': 'unknown',
                'parent_id': 'doc-1', 'chunk_position': 0
            },
            {
                'text_id':   'test-2', 'text': 'text2', 'distance': 0.2,
                'timestamp': 1000,
                'source':    'test', 'tags': [], 'confidence': None,
                'language':  'unknown',
                'section':   'unknown', 'author': 'unknown',
                'parent_id': 'doc-2', 'chunk_position': 0
            },
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
            call_args = mock_log.critical.call_args
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
            call_args = mock_log.critical.call_args
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

        # Mock storage with health_check method
        mock_storage = Mock()
        mock_storage.health_check.return_value = {'status': 'healthy'}
        self.mock_ragstore.storage = mock_storage

        result = manager.get_health_status()

        self.assertEqual(result, {'status': 'healthy'})

        call_args = mock_log.debug.call_args
        self.assertEqual(call_args[0][0],
                         'Operation completed: %s (%.3fs)')
        self.assertEqual(call_args[0][1], 'health_check')
        self.assertIsInstance(call_args[0][2], float)  # execution time

    @patch('ragl.manager._LOG')
    def test_get_health_status_without_health_check(self, mock_log):
        """Test getting health status when backend doesn't support health checks."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        # Mock storage without health_check method
        self.mock_ragstore.storage = Mock(spec=[])

        result = manager.get_health_status()

        self.assertEqual(result, {'status': 'health_check_not_supported'})
        call_args = mock_log.debug.call_args
        self.assertEqual(call_args[0][0],
                         'Operation completed: %s (%.3fs)')
        self.assertEqual(call_args[0][1], 'health_check')
        self.assertIsInstance(call_args[0][2], float)  # execution time

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
    def test_format_context(self, mock_log):
        """Test formatting context from text chunks."""
        text_units = [
            TextUnit(text_id="1", text="First chunk", source="test", distance=0.00),
            TextUnit(text_id="2", text="Second chunk", source="test", distance=0.01),
        ]

        result = RAGManager._format_context(text_units)

        expected = "First chunk\n\nSecond chunk"
        self.assertEqual(result, expected)
        mock_log.debug.assert_called_with('Formatting chunks')

    def test_format_context_custom_separator(self):
        """Test formatting context with custom separator."""
        text_units = [
            TextUnit(text_id="1", text="First chunk", source="test", distance=0.00),
            TextUnit(text_id="2", text="Second chunk", source="test", distance=0.01),
        ]

        result = RAGManager._format_context(text_units, separator=" | ")

        expected = "First chunk | Second chunk"
        self.assertEqual(result, expected)

    @patch('ragl.manager._LOG')
    def test_get_chunks_with_split(self, mock_log):
        """Test getting chunks with splitting enabled."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "Test text to split"

        # Mock _split_text method
        with patch.object(manager, '_split_text',
                          return_value=['chunk1', 'chunk2']) as mock_split:
            result = manager._get_chunks(text, 100, 20, True)

        self.assertEqual(result, ['chunk1', 'chunk2'])
        mock_split.assert_called_once_with(text, 100, 20)
        mock_log.debug.assert_called_with('Getting chunks')

    def test_get_chunks_without_split_string(self):
        """Test getting chunks without splitting for string input."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "Test text no split"

        result = manager._get_chunks(text, 100, 20, False)

        self.assertEqual(result, [text])

    def test_get_chunks_without_split_textunit(self):
        """Test getting chunks without splitting for TextUnit input."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text_unit = TextUnit(text_id="test", text="Test text no split",
                             source="test", distance=0.1)

        result = manager._get_chunks(text_unit, 100, 20, False)

        self.assertEqual(result, [text_unit.text])

    def test_get_chunks_with_split_textunit(self):
        """Test getting chunks with splitting for TextUnit input."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text_unit = TextUnit(text_id="test", text="Test text to split",
                             source="test", distance=0.1)

        with patch.object(manager, '_split_text',
                          return_value=['chunk1', 'chunk2']) as mock_split:
            result = manager._get_chunks(text_unit, 100, 20, True)

        # self.assertEqual(result, ['chunk1', 'chunk2'])
        mock_split.assert_called_once_with(text_unit.text, 100, 20)

    @patch('ragl.manager._LOG')
    def test_sanitize_text_input_normal(self, mock_log):
        """Test sanitizing normal text input."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "Normal text input"

        result = manager._sanitize_query(text)

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
                manager._sanitize_query(long_text)

            self.assertIn('text too long', str(cm.exception))
            mock_log.critical.assert_called_with('text too long')

    def test_sanitize_text_input_paranoid_mode(self):
        """Test sanitizing text input in paranoid mode."""
        config = ManagerConfig(chunk_size=100, overlap=20, paranoid=True)
        manager = RAGManager(config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "Text with <script>alert('xss')</script> dangerous chars!"

        result = manager._sanitize_query(text)

        # Should remove dangerous characters, keeping only alphanumeric, spaces, and basic punctuation
        expected = "Text with alert('xss') dangerous chars!"
        self.assertEqual(result, expected)

    @patch('ragl.manager._LOG')
    def test_split_text_last_chunk_too_small(self, mock_log):
        """Test splitting text where last chunk gets merged."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = (
                         "Artificial Intelligence (AI) is a broad field encompassing various techniques. "
                         "Machine Learning (ML) is a subset of AI that focuses on training models with data. "
                         "Deep Learning (DL), a further subset, uses neural networks with many layers."
                     ) * 2

        encode_calls = [
            list(range(64)),
            list(range(10))
        ]
        self.mock_tokenizer.encode.side_effect = encode_calls

        decode_calls = [
            "First chunk content",
            "Small last chunk"
        ]
        self.mock_tokenizer.decode.side_effect = decode_calls

        result = manager._split_text(text, chunk_size=64, overlap=32)

        self.assertEqual(len(result), 1)
        self.assertIn("First chunk content", result[0])
        self.assertIn("Small last chunk", result[0])
        mock_log.debug.assert_called_with('Merging last chunk due to short length')

    @patch('ragl.manager._LOG')
    def test_split_text(self, mock_log):
        """Test splitting text into chunks."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "This is a test text to split"

        # Mock tokenizer behavior
        tokens = list(range(220))  # 220 tokens
        self.mock_tokenizer.encode.return_value = tokens

        result = manager._split_text(text, chunk_size=100, overlap=25)

        # Should create 3 chunks: 0-100, 80-180, 160-220
        self.assertEqual(len(result), 3)
        self.assertEqual(self.mock_tokenizer.decode.call_count, 3)
        mock_log.debug.assert_called_with('Splitting text')

    def test_split_text_empty_chunks_filtered(self):
        """Test that empty chunks are filtered out during splitting."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "Test text"

        # Mock tokenizer to return some empty decoded chunks
        tokens = list(range(100))
        self.mock_tokenizer.encode.return_value = tokens
        self.mock_tokenizer.decode.side_effect = lambda \
            x: "valid chunk" if len(x) > 50 else "   "

        result = manager._split_text(text, chunk_size=60, overlap=10)

        # Only non-empty chunks should be included
        self.assertTrue(all(chunk.strip() for chunk in result))

    @patch('time.time')
    @patch('ragl.manager._LOG')
    def test_store_chunk(self, mock_log, mock_time):
        """Test storing a single chunk."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        mock_time.return_value = 12345

        chunk = "Test chunk text"
        base_data = {
            'source':     'test',
            'timestamp':  12345,
            'tags':       ['test'],
            'confidence': 0.8,
            'language':   'en',
            'section':    'intro',
            'author':     'test_author',
            'parent_id':  'parent-1',
        }
        text_id = "chunk-1"

        result = manager._store_chunk(
            chunk=chunk,
            base_data=base_data,
            text_id=text_id,
            i=0,
            parent_id="parent-1"
        )

        self.assertIsInstance(result, TextUnit)
        self.assertEqual(result.text, chunk)
        self.assertEqual(result.text_id, 'text-id-1')  # Mocked return value
        self.assertEqual(result.chunk_position, 0)
        self.assertEqual(result.parent_id, "parent-1")

        # Verify ragstore.store_text was called correctly
        self.mock_ragstore.store_text.assert_called_once()
        call_args = self.mock_ragstore.store_text.call_args
        self.assertEqual(call_args[1]['text'], chunk)
        self.assertEqual(call_args[1]['text_id'], text_id)

        mock_log.debug.assert_called_with('Storing chunk')

    @patch('time.time')
    @patch('ragl.manager._LOG')
    def test_prepare_base_data_textunit(self, mock_log, mock_time):
        """Test preparing base data from TextUnit."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        text_unit = TextUnit(
            text_id="test-id",
            text="Test text",
            source="test_source",
            timestamp=54321,
            tags=["tag1", "tag2"],
            confidence=0.9,
            language="en",
            section="chapter1",
            author="test_author",
            parent_id="original_parent",
            distance=0.01
        )

        result = manager._prepare_base_data(text_unit, "new_parent")

        # Should return the TextUnit's dict representation
        expected = text_unit.to_dict()
        self.assertEqual(result, expected)
        mock_log.debug.assert_called_with('Preparing base metadata')

    @patch('time.time')
    @patch('ragl.manager._LOG')
    def test_prepare_base_data_string(self, mock_log, mock_time):
        """Test preparing base data from string."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)

        mock_time.return_value = 98765
        text = "Test text string"
        parent_id = "parent-123"

        result = manager._prepare_base_data(text, parent_id)

        expected = {
            'source':     'unknown',
            'timestamp':  98765,
            'tags':       [],
            'confidence': None,
            'language':   'unknown',
            'section':    'unknown',
            'author':     'unknown',
            'parent_id':  parent_id,
        }
        self.assertEqual(result, expected)
        mock_log.debug.assert_called_with('Preparing base metadata')

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
            mock_log.critical.assert_called_with('Chunk_size must be positive')

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
            mock_log.critical.assert_called_with(
                'Overlap must be non-negative')

    def test_validate_chunking_overlap_too_large(self):
        """Test chunking validation with overlap >= chunk_size."""
        with patch('ragl.manager._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                RAGManager._validate_chunking(100, 100)

            self.assertIn('Overlap must be less than chunk_size',
                          str(cm.exception))
            mock_log.critical.assert_called_with(
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
            mock_log.critical.assert_called_with('Query cannot be empty')

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
            mock_log.critical.assert_called_with(expected_msg)

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
            mock_log.critical.assert_called_with(
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
            'config=ManagerConfig(chunk_size=100, overlap=20, paranoid=False), '
            f'ragstore={self.mock_ragstore!r}, '
            f'tokenizer={self.mock_tokenizer!r})'
        )
        self.assertEqual(result, expected)

    def test_constants(self):
        """Test that class constants are properly defined."""
        self.assertEqual(RAGManager.DEFAULT_BASE_ID, 'doc')
        self.assertEqual(RAGManager.MAX_QUERY_LENGTH, 8192)
        self.assertEqual(RAGManager.MAX_INPUT_LENGTH, (1024 * 1024) * 10)

    def test_add_text_integration_multiple_chunks(self):
        """Integration test for adding text that gets split into multiple chunks."""
        manager = RAGManager(self.config, self.mock_ragstore,
                             tokenizer=self.mock_tokenizer)
        text = "Long text that will be split into multiple chunks"

        # Mock tokenizer to create 3 chunks
        self.mock_tokenizer.encode.return_value = list(range(220))  # 250 tokens
        self.mock_tokenizer.decode.side_effect = [
            "Chunk 1 content",
            "Chunk 2 content",
            "Chunk 3 content"
        ]

        # Mock store_text to return different IDs
        self.mock_ragstore.store_text.side_effect = ['id-1', 'id-2', 'id-3']

        result = manager.add_text(text)

        self.assertEqual(len(result), 3)
        self.assertEqual(self.mock_ragstore.store_text.call_count, 3)

        # Verify all chunks have the same parent_id
        parent_ids = [unit.parent_id for unit in result]
        self.assertEqual(len(set(parent_ids)), 1)  # All should be the same

        # Verify chunk positions
        positions = [unit.chunk_position for unit in result]
        self.assertEqual(positions, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
