"""
Protocol definitions for ragl components.

This module defines the core protocols that establish contracts
for different components in the ragl system, including text
embedding, vector storage, text retrieval, and tokenization
operations.

Classes:
    EmbedderProtocol:
        Protocol for text embedding operations.
    VectorStoreProtocol:
        Protocol for vector storage operations.
    RAGStoreProtocol:
        Protocol for text retrieval and storage.
    TokenizerProtocol:
        Protocol for text tokenization operations.
"""

from typing import (
    Any,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

import numpy as np

from ragl.textunit import TextUnit

__all__ = (
    'EmbedderProtocol',
    'RAGStoreProtocol',
    'TokenizerProtocol',
    'TextEmbeddingPair',
    'VectorStoreProtocol',
)


TextEmbeddingPair: TypeAlias = tuple[TextUnit, np.ndarray]


@runtime_checkable
class EmbedderProtocol(Protocol):
    """
    Protocol for text embedding.

    Defines methods for embedding text into vectors.
    """

    @property
    def dimensions(self) -> int:
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover

    def embed(self, text: str) -> np.ndarray:
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """
    Protocol for vector store operations.

    Defines methods for storing and retrieving vectors.
    """

    def clear(self) -> None:
        # pylint: disable=missing-function-docstring
        ...

    def delete_text(self, text_id: str) -> bool:
        # pylint: disable=missing-function-docstring
        ...

    def delete_texts(self, text_ids: list[str]) -> int:
        # pylint: disable=missing-function-docstring
        ...

    def get_relevant(
            self,
            embedding: np.ndarray,
            top_k: int,
            *,
            min_time: int | None,
            max_time: int | None,
    ) -> list[TextUnit]:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...

    def health_check(self) -> dict[str, Any]:
        # pylint: disable=missing-function-docstring
        ...

    def list_texts(self) -> list[str]:
        # pylint: disable=missing-function-docstring
        ...

    def store_text(self, text_unit: TextUnit, embedding: np.ndarray) -> str:
        # pylint: disable=missing-function-docstring
        ...

    def store_texts(
            self,
            texts_and_embeddings: list[TextEmbeddingPair],
    ) -> list[str]:
        # pylint: disable=missing-function-docstring
        ...


@runtime_checkable
class RAGStoreProtocol(Protocol):
    """
    Protocol for text retrieval.

    Defines methods for storing and retrieving text.
    """

    embedder: EmbedderProtocol
    storage: VectorStoreProtocol

    def clear(self) -> None:
        # pylint: disable=missing-function-docstring
        ...

    def delete_text(self, text_id: str) -> bool:
        # pylint: disable=missing-function-docstring
        ...

    def delete_texts(self, text_ids: list[str]) -> int:
        # pylint: disable=missing-function-docstring
        ...

    def get_relevant(
            self,
            query: str,
            top_k: int,
            *,
            min_time: int | None,
            max_time: int | None,
    ) -> list[TextUnit]:
        # pylint: disable=missing-function-docstring
        ...

    def list_texts(self) -> list[str]:
        # pylint: disable=missing-function-docstring
        ...

    def store_text(self, text_unit: TextUnit) -> TextUnit:
        # pylint: disable=missing-function-docstring
        ...

    def store_texts(self, text_units: list[TextUnit]) -> list[TextUnit]:
        # pylint: disable=missing-function-docstring
        ...


@runtime_checkable
class TokenizerProtocol(Protocol):
    """
    Protocol for text tokenization.

    Defines methods for encoding and decoding text.
    """

    def decode(self, tokens: list[int]) -> str:
        # pylint: disable=missing-function-docstring
        ...

    def encode(self, text: str) -> list[int]:
        # pylint: disable=missing-function-docstring
        ...
