from ragl import RAGManager, HFEmbedder, RAGStore
from ragl.config import (
    EmbedderConfig,
    HFConfig,
    ManagerConfig,
    RedisConfig,
    StorageConfig,
)
from ragl.storage.redis import RedisStorage


def create_rag_manager(
        index_name: str,
        *,
        storage_config: RedisConfig = RedisConfig(),   # todo use base types?
        embedder_config: HFConfig = HFConfig(),
        manager_config: ManagerConfig = ManagerConfig(),
) -> RAGManager:
    """
    Convenience function to create a RAGManager with a default embedder.

    Args:
        index_name:
            Name of the vector storage index.
        storage_config:
            Redis configuration, represented by
            ragl.config.RedisConfig
        embedder_config:
            Configuration for the embedder, represented by
            ragl.config.EmbedderConfig
        manager_config:
            Configuration for the RAGManager, represented by
            ragl.config.ManagerConfig

    Returns:
        A new RAGManager instance.
    """
    embedder = HFEmbedder(config=embedder_config)
    storage = RedisStorage(
        redis_config=storage_config,
        dimensions= embedder.dimensions,
        index_name=index_name,
    )
    ragstore = RAGStore(embedder=embedder, storage=storage)
    manager = RAGManager(
        config=manager_config,
        ragstore=ragstore,
    )
    return manager
