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
        """Initialize with an encoding name."""
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
