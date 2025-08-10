"""
Text tokenization utilities using tiktoken.

This module provides an interface to the tiktoken library for encoding
and decoding text into tokens. The tokenizer is used for text processing
tasks that require token-level operations, such as chunking text by
token count or calculating token-based similarity metrics.

Classes:
- TiktokenTokenizer:
    Encoding/decoding text using tiktoken.
"""

import tiktoken


__all__ = ('TiktokenTokenizer',)


class TiktokenTokenizer:
    """
    Tokenize text using tiktoken.

    Attributes:
        encoding:
            tiktoken Encoding for tokenization.
    """

    _DEFAULT_ENCODING = 'cl100k_base'

    encoding: tiktoken.Encoding

    def __init__(self, encoding_name: str = _DEFAULT_ENCODING):
        """
        Create a tokenizer with the specified encoding.

        Args:
            encoding_name:
                Name of the tiktoken encoding to use. Defaults to
                'cl100k_base'.
        """
        self.encoding = tiktoken.get_encoding(encoding_name)

    def decode(self, tokens: list[int]) -> str:
        """
        Decode tokens into text.

        Args:
            tokens:
                Token IDs to decode.

        Returns:
            Decoded text string.
        """
        return self.encoding.decode(tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode text into tokens.

        Args:
            text:
                Text to encode.

        Returns:
            List of token IDs.
        """
        return self.encoding.encode(text)
