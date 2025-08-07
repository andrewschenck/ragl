from typing import Mapping, Any

from ragl.protocols import Embedder, VectorStorage


__all__ =  ('StandardRetriever',)


# todo rename RAGStore RAGEngine (X) (or RAGPipeline / Retrieval)
# todo rename StandardRetriever RAGStore
# todo rename protocols.Retriever to RAGStoreProtocol (name TBD)
# todo rename RAGConfig EngineConfig(?)
# todo update string / doc / log / isinstance references to RAGStore / RAGEngine / Retriever / StandardRetriever


class StandardRetriever:
    """
    Retrieve text using an embedder and storage.

    Attributes:
        embedder:
            Embedder instance for vectorization.
        storage:
            VectorStorage instance for data.
    """

    embedder: Embedder
    storage: VectorStorage

    def __init__(self, embedder: Embedder, storage: VectorStorage):
        """
        Initialize with embedder and storage.

        Args:
            embedder:
                Embedder for text vectorization.
            storage:
                Storage backend for data.

        Raises:
            TypeError: If args don’t implement protocols.
        """
        if not isinstance(embedder, Embedder):
            raise TypeError('embedder must implement Embedder')
        if not isinstance(storage, VectorStorage):
            raise TypeError('storage must implement VectorStorage')
        self.embedder = embedder
        self.storage = storage

    def clear(self) -> None:
        """Clear all data from storage."""
        self.storage.clear()

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

    def delete_text(self, text_id: str) -> None:
        """
        Delete a text from storage.

        Args:
            text_id:
                ID of text to delete.
        """
        self.storage.delete_text(text_id)

    def list_texts(self) -> list[str]:
        """
        List all text IDs in storage.

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
        Store text in the storage backend.

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
