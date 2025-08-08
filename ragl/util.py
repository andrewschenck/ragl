from ragl.manager import RAGManager
from ragl.ragstore import RAGStore
from ragl.config import (
    EmbedderConfig,
    SentencetransformerConfig,
    ManagerConfig,
    RedisConfig,
    StorageConfig,
)
from ragl.exceptions import ConfigurationError


__all__ = ('create_rag_manager',)


def _create_hf_embedder(config: SentencetransformerConfig):
    try:
        # pylint: disable=import-outside-toplevel
        from ragl.embed.sentencetransformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(config=config)
    except ImportError as e:
        raise ConfigurationError('SentenceTransformer not available') from e


def _create_redis_storage(
        config: RedisConfig,
        dimensions: int,
        index_name: str,
):
    try:
        # pylint: disable=import-outside-toplevel
        from ragl.store.redis import RedisStore
        return RedisStore(
            redis_config=config,
            dimensions=dimensions,
            index_name=index_name,
        )
    except ImportError as e:
        raise ConfigurationError('Redis not available') from e


def create_rag_manager(
        *,
        index_name: str,
        storage_config: StorageConfig,
        embedder_config: EmbedderConfig,
        manager_config: ManagerConfig,
) -> RAGManager:
    """
    Convenience function to create a RAGManager with a default embed.

    Args:
        index_name:
            Name of the vector store index.
        storage_config:
            Redis configuration, represented by
            ragl.config.RedisConfig
        embedder_config:
            Configuration for the embed, represented by
            ragl.config.EmbedderConfig
        manager_config:
            Configuration for the RAGManager, represented by
            ragl.config.ManagerConfig

    Returns:
        A new RAGManager instance.
    """
    if isinstance(embedder_config, SentencetransformerConfig):
        embedder = _create_hf_embedder(embedder_config)
    else:
        embedder = None

    if embedder is None:
        raise ConfigurationError('no Embedder / configuration.')

    if isinstance(storage_config, RedisConfig):
        storage = _create_redis_storage(
            config=storage_config,
            dimensions=embedder.dimensions,
            index_name=index_name,
        )
    else:
        storage = None

    if storage is None:
        raise ConfigurationError('no Storage / configuration.')

    ragstore = RAGStore(embedder=embedder, storage=storage)
    manager = RAGManager(
        config=manager_config,
        ragstore=ragstore,
    )

    return manager
