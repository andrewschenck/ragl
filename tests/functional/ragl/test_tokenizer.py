import unittest
from unittest.mock import patch, MagicMock
import logging

from ragl.tokenizer import TiktokenTokenizer


class TestTiktokenTokenizer(unittest.TestCase):
    """Test cases for TiktokenTokenizer class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tokenizer = TiktokenTokenizer()

    def test_init_default_encoding(self):
        """Test initialization with default encoding."""
        tokenizer = TiktokenTokenizer()
        self.assertEqual(tokenizer.encoding.name, 'cl100k_base')

    def test_init_custom_encoding(self):
        """Test initialization with custom encoding."""
        tokenizer = TiktokenTokenizer('gpt2')
        self.assertEqual(tokenizer.encoding.name, 'gpt2')

    @patch('ragl.tokenizer._LOG')
    def test_init_logging(self, mock_log):
        """Test that initialization logs the encoding name."""
        TiktokenTokenizer('cl100k_base')
        mock_log.debug.assert_called_with(
            'Initializing TiktokenTokenizer with encoding: %s',
            'cl100k_base'
        )

    def test_encode_text(self):
        """Test encoding text into tokens."""
        text = "Hello, world!"
        tokens = self.tokenizer.encode(text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(token, int) for token in tokens))
        self.assertGreater(len(tokens), 0)

    def test_encode_empty_string(self):
        """Test encoding empty string."""
        tokens = self.tokenizer.encode("")
        self.assertEqual(tokens, [])

    @patch('ragl.tokenizer._LOG')
    def test_encode_logging(self, mock_log):
        """Test that encoding logs the input text."""
        text = "test text"
        self.tokenizer.encode(text)
        mock_log.debug.assert_called_with('Encoding text: %s', text)

    def test_decode_tokens(self):
        """Test decoding tokens back to text."""
        original_text = "Hello, world!"
        tokens = self.tokenizer.encode(original_text)
        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(decoded_text, original_text)

    def test_decode_empty_list(self):
        """Test decoding empty token list."""
        result = self.tokenizer.decode([])
        self.assertEqual(result, '')

    @patch('ragl.tokenizer._LOG')
    def test_decode_empty_list_warning(self, mock_log):
        """Test that decoding empty list logs a warning."""
        self.tokenizer.decode([])
        mock_log.warning.assert_called_with(
            'Attempted to decode empty token list')

    @patch('ragl.tokenizer._LOG')
    def test_decode_logging(self, mock_log):
        """Test that decoding logs the input tokens."""
        tokens = [1, 2, 3]
        self.tokenizer.decode(tokens)
        mock_log.debug.assert_called_with('Decoding tokens: %s', tokens)

    def test_decode_valid_tokens(self):
        """Test decoding with valid token IDs."""
        # Use known tokens for cl100k_base encoding
        tokens = [9906, 11, 1917, 0]  # "Hello, world!"
        result = self.tokenizer.decode(tokens)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_repr(self):
        """Test __repr__ method."""
        repr_str = repr(self.tokenizer)
        expected = 'TiktokenTokenizer(encoding_name="cl100k_base")'
        self.assertEqual(repr_str, expected)

    def test_repr_custom_encoding(self):
        """Test __repr__ method with custom encoding."""
        tokenizer = TiktokenTokenizer('gpt2')
        repr_str = repr(tokenizer)
        expected = 'TiktokenTokenizer(encoding_name="gpt2")'
        self.assertEqual(repr_str, expected)

    def test_str(self):
        """Test __str__ method."""
        str_repr = str(self.tokenizer)
        expected = 'TiktokenTokenizer using cl100k_base encoding'
        self.assertEqual(str_repr, expected)

    def test_str_custom_encoding(self):
        """Test __str__ method with custom encoding."""
        tokenizer = TiktokenTokenizer('gpt2')
        str_repr = str(tokenizer)
        expected = 'TiktokenTokenizer using gpt2 encoding'
        self.assertEqual(str_repr, expected)

    def test_round_trip_encoding_decoding(self):
        """Test that encoding then decoding preserves the original text."""
        test_texts = [
            "Simple text",
            "Text with numbers 123",
            "Special chars: !@#$%^&*()",
            "Unicode: 你好世界",
            "Mixed: Hello 世界 123!",
            "Whitespace and\nnewlines\ttabs"
        ]

        for text in test_texts:
            with self.subTest(text=text):
                tokens = self.tokenizer.encode(text)
                decoded = self.tokenizer.decode(tokens)
                self.assertEqual(decoded, text)

    def test_encoding_attribute_access(self):
        """Test that encoding attribute is accessible."""
        self.assertTrue(hasattr(self.tokenizer, 'encoding'))
        self.assertIsNotNone(self.tokenizer.encoding)
        self.assertTrue(hasattr(self.tokenizer.encoding, 'encode'))
        self.assertTrue(hasattr(self.tokenizer.encoding, 'decode'))

    @patch('ragl.tokenizer.tiktoken.get_encoding')
    def test_tiktoken_integration(self, mock_get_encoding):
        """Test integration with tiktoken library."""
        mock_encoding = MagicMock()
        mock_get_encoding.return_value = mock_encoding

        tokenizer = TiktokenTokenizer('test_encoding')

        mock_get_encoding.assert_called_once_with('test_encoding')
        self.assertEqual(tokenizer.encoding, mock_encoding)


class TestModuleConstants(unittest.TestCase):
    """Test module-level constants and imports."""

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from ragl.tokenizer import __all__
        self.assertEqual(__all__, ('TiktokenTokenizer',))

    def test_default_encoding_constant(self):
        """Test that default encoding constant is accessible."""
        self.assertEqual(TiktokenTokenizer._DEFAULT_ENCODING, 'cl100k_base')


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
