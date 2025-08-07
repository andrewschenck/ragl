from typing import (
    Any,
    Mapping,
    Protocol,
    runtime_checkable,
)

import numpy as np


__all__ = (
    'Embedder',
    'Retriever',
    'Tokenizer',
    'VectorStorage',
)


@runtime_checkable
class Embedder(Protocol):
    """
    Protocol for text embedding.

    Defines methods for embedding text into vectors.
    """

    @property
    def dimensions(self) -> int:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover

    def embed(self, text: str) -> np.ndarray:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover


@runtime_checkable
class VectorStorage(Protocol):
    """
    Protocol for vector storage operations.

    Defines methods for storing and retrieving vectors.
    """

    def clear(self) -> None:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover

    def delete_text(self, text_id: str) -> None:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover

    def get_relevant(
            self,
            embedding: np.ndarray,
            top_k: int = 1,
            *,
            min_time: int | None,
            max_time: int | None,
    ) -> list[dict[str, Any]]:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover

    def health_check(self) -> dict[str, Any]:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover

    def list_texts(self) -> list[str]:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover

    def store_text(
            self,
            text: str,
            embedding: np.ndarray,
            *,
            text_id: str | None,
            metadata: Mapping[str, Any] | None = None,
    ) -> str:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover


@runtime_checkable
class Retriever(Protocol):
    """
    Protocol for text retrieval.

    Defines methods for storing and retrieving text.
    """

    storage: VectorStorage

    def clear(self) -> None:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover

    def delete_text(self, text_id: str) -> None:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover

    def get_relevant(
            self,
            query: str,
            top_k: int,
            *,
            min_time: int | None,
            max_time: int | None,
    ) -> list[dict[str, Any]]:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover

    def list_texts(self) -> list[str]:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover

    def store_text(
            self,
            text: str,
            *,
            text_id: str | None,
            metadata: Mapping[str, Any] | None,
    ) -> str:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover


@runtime_checkable
class Tokenizer(Protocol):
    """
    Protocol for text tokenization.

    Defines methods for encoding and decoding text.
    """

    def decode(self, tokens: list[int]) -> str:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover

    def encode(self, text: str) -> list[int]:  # noqa: D102
        # pylint: disable=missing-function-docstring
        ...  # pragma: no cover
