"""
Store and retrieve text documents using semantic similarity search.

This module provides the RAGStore class, which combines text embedding
and vector storage capabilities to enable efficient storage and
retrieval of text documents based on semantic similarity.

The RAGStore acts as a high-level interface that coordinates between
an Embedder (for converting text to vectors) and a VectorStore (for
persistent storage and similarity search).

Classes:
    RAGStore:
        Store and retrieve text using an embedder and vector store.
"""

import logging
from typing import Mapping, Any

from ragl.protocols import EmbedderProtocol, VectorStoreProtocol


__all__ = ('RAGStore',)


_LOG = logging.getLogger(__name__)


class RAGStore:
    """
    Store and retrieve text using an embedder.

    This class provides methods to store text documents, retrieve
    relevant documents based on semantic similarity, and manage the
    contents of the underlying VectorStoreProtocol-conforming class.

    Attributes:
        embedder:
            EmbedderProtocol-conforming object for vectorization.
        storage:
            VectorStoreProtocol-conforming object for backend data
            store and retrieval.

    Example:
        >>> rag_store = RAGStore(embedder=embedder, storage=storage)
        >>> text = "Hello, world!"
        >>> metadata = {"author": "John Doe", 'title': "Sample Document"}
        >>> text_id = rag_store.store_text(text, metadata=metadata)
        >>> results = rag_store.get_relevant("Hello", top_k=5)
    """

    embedder: EmbedderProtocol
    storage: VectorStoreProtocol

    def __init__(
            self,
            embedder: EmbedderProtocol,
            storage: VectorStoreProtocol,
    ):
        """
        Initialize with Embedder and VectorStore.

        This constructor checks that the provided embedder and storage
        objects conform to the required protocol (EmbedderProtocol and
        VectorStoreProtocol, respectively) and raises a TypeError
        if their is a protocol mismatch.

        Args:
            embedder:
                EmbedderProtocol-conforming object for vectorization
                of text.
            storage:
                StorageProtocol-conforming object for backend data
                store and retrieval.

        Raises:
            TypeError: If args don’t implement protocols.
        """
        if not isinstance(embedder, EmbedderProtocol):
            msg = 'embedder must implement EmbedderProtocol'
            _LOG.critical(msg)
            raise TypeError(msg)
        if not isinstance(storage, VectorStoreProtocol):
            msg = 'store must implement VectorStoreProtocol'
            _LOG.critical(msg)
            raise TypeError(msg)
        self.embedder = embedder
        self.storage = storage

    def clear(self) -> None:
        """Clear all data from store."""
        self.storage.clear()

    def delete_text(self, text_id: str) -> bool:
        """
        Delete a text from store.

        Attempts to delete a text document by its ID.

        Args:
            text_id:
                ID of text to delete.

        Returns:
            True if text was deleted, False if it did not exist.
        """
        _LOG.debug('Deleting text %s', text_id)
        return self.storage.delete_text(text_id)

    def get_relevant(
            self,
            query: str,
            top_k: int,
            *,
            min_time: int | None = None,
            max_time: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant texts for a query.

        Retrieves a list of texts that are relevant to the provided
        query based on semantic similarity. It uses the embedder to
        convert the query into a vector, and then queries the storage
        for the most similar texts.

        If `min_time` or `max_time` are provided, only texts within
        that time range will be considered.

        If no time range is specified, all texts are considered.

        Args:
            query:
                Query text.
            top_k:
                Number of results to return.
            min_time:
                Minimum timestamp filter.
            max_time:
                Maximum timestamp filter.

        Returns:
            List of result dicts.
        """
        _LOG.debug('Retrieving relevant texts for query %s', query)
        return self.storage.get_relevant(
            embedding=self.embedder.embed(query),
            top_k=top_k,
            min_time=min_time,
            max_time=max_time,
        )

    def list_texts(self) -> list[str]:
        """
        Return a list of text IDs in the store.

        Returns:
            List of text IDs.
        """
        _LOG.debug('Listing all texts in store')
        return self.storage.list_texts()

    def store_text(
            self,
            text: str,
            *,
            text_id: str | None = None,
            metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """
        Store text in the store backend.

        Stores a text document in the underlying storage system,
        generating an embedding for the text using the embedder.

        If `text_id` is provided, it will be used as the ID for the
        text; otherwise, a new ID will be generated.

        If `metadata` is provided, it will be stored alongside the text.

        This method returns the text ID, which is either the provided
        `text_id` or a newly generated ID if none was provided.

        Args:
            text:
                Text to store.
            text_id:
                Optional ID for the text.
            metadata:
                Optional metadata dict.

        Returns:
                The text ID (generated if not provided, or the provided
                text_id).
        """
        _LOG.debug('Storing text %s', text)
        text_id = self.storage.store_text(
            text=text,
            embedding=self.embedder.embed(text),
            text_id=text_id,
            metadata=metadata,
        )
        return text_id

    def __repr__(self) -> str:
        """Return a detailed string representation of the RAGStore."""
        return (f'RAGStore(embedder={self.embedder!r}, '
                f'storage={self.storage!r})')

    def __str__(self) -> str:
        """Return a user-friendly string representation of the RAGStore."""
        embedder_name = type(self.embedder).__name__
        storage_name = type(self.storage).__name__
        return (f'RAGStore with {embedder_name} embedder '
                f'and {storage_name} storage')
