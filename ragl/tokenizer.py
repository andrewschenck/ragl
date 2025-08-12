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

import logging

import tiktoken


__all__ = ('TiktokenTokenizer',)


_LOG = logging.getLogger(__name__)


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
        _LOG.debug('Initializing TiktokenTokenizer with encoding: %s',
                   encoding_name)
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
        _LOG.debug('Decoding tokens: %s', tokens)
        if not tokens:
            _LOG.warning('Attempted to decode empty token list')
            return ''
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
        _LOG.debug('Encoding text: %s', text)
        return self.encoding.encode(text)

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the tokenizer.

        Returns:
            String representation suitable for debugging.
        """
        return f'TiktokenTokenizer(encoding_name="{self.encoding.name}")'

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the tokenizer.

        Returns:
            Human-readable string representation.
        """
        return f'TiktokenTokenizer using {self.encoding.name} encoding'
