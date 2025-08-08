from typing import Mapping, Any

from ragl.protocols import EmbedderProtocol, VectorStoreProtocol


__all__ = ('RAGStore',)


class RAGStore:
    """
    Store and retrieve text using an embed.

    Attributes:
        embedder:
            EmbedderProtocol-conforming object for vectorization.
        storage:
            StorageProtocol-conforming object for backend data
            store and retrieval.
    """

    embedder: EmbedderProtocol
    storage: VectorStoreProtocol

    def __init__(
            self,
            embedder: EmbedderProtocol,
            storage: VectorStoreProtocol,
    ):
        """
        Initialize with embed and store.

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
            raise TypeError('embed must implement EmbedderProtocol')
        if not isinstance(storage, VectorStoreProtocol):
            raise TypeError('store must implement StorageProtocol')
        self.embedder = embedder
        self.storage = storage

    def clear(self) -> None:
        """Clear all data from store."""
        self.storage.clear()

    def delete_text(self, text_id: str) -> bool:
        """
        Delete a text from store.

        Args:
            text_id:
                ID of text to delete.
        """
        return self.storage.delete_text(text_id)

    def get_relevant(
            self,
            query: str,
            top_k: int = 1,
            *,
            min_time: int | None = None,
            max_time: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant texts for a query.

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
        return self.storage.get_relevant(
            embedding=self.embedder.embed(query),
            top_k=top_k,
            min_time=min_time,
            max_time=max_time,
        )

    def list_texts(self) -> list[str]:
        """
        List all text IDs in store.

        Returns:
            Sorted list of text IDs.
        """
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

        Args:
            text:
                Text to store.
            text_id:
                Optional ID for the text.
            metadata:
                Optional metadata dict.

        Returns:
            Assigned text ID.
        """
        text_id = self.storage.store_text(
            text=text,
            embedding=self.embedder.embed(text),
            text_id=text_id,
            metadata=metadata,
        )
        return text_id
