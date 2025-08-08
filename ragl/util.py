from typing import Callable

from ragl.exceptions import ConfigurationError
from ragl.manager import RAGManager
from ragl.ragstore import RAGStore
from ragl.config import (
    EmbedderConfig,
    ManagerConfig,
    RedisConfig,
    SentenceTransformerConfig,
    StorageConfig,
)


__all__ = ('create_rag_manager',)


def _create_hf_embedder(*, config: SentenceTransformerConfig):
    try:
        # pylint: disable=import-outside-toplevel
        from ragl.embed.sentencetransformer import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(config=config)
    except ImportError as e:
        raise ConfigurationError('SentenceTransformer not available') from e


def _create_redis_storage(
        *,
        config: RedisConfig,
        dimensions: int,
        index_name: str,
):
    try:
        # pylint: disable=import-outside-toplevel
        from ragl.store.redis import RedisVectorStore
        return RedisVectorStore(
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
    Convenience function to create a RAGManager instance
    with Embedder and VectorStore implementations which are
    derived from the provided configurations.

    Args:
        index_name:
            Name of the vector store index.
        storage_config:
            VectorStore configuration, represented by StorageConfig.
        embedder_config:
            Embedder configuration, represented by EmbedderConfig.
        manager_config:
            RAGManager configuration, represented by ManagerConfig.

    Returns:
        A new RAGManager instance.
    """
    # todo turn this into a full blown factory with (auto?)-registration

    embedder_map: dict[type[EmbedderConfig], Callable] = {
        SentenceTransformerConfig: _create_hf_embedder,
    }
    storage_map: dict[type[StorageConfig], Callable] = {
        RedisConfig: _create_redis_storage,
    }

    embedder_config_type = type(embedder_config)
    try:
        embedder_init = embedder_map[embedder_config_type]
    except KeyError:
        raise ConfigurationError('no Embedder / configuration.')

    embedder = embedder_init(config=embedder_config)

    # if isinstance(embedder_config, SentencetransformerConfig):
    #     embedder = _create_hf_embedder(embedder_config)
    # else:
    #     embedder = None

    # if embedder is None:
    #     raise ConfigurationError('no Embedder / configuration.')

    storage_config_type = type(storage_config)
    try:
        storage_init = storage_map[storage_config_type]
    except KeyError:
        raise ConfigurationError('no VectorStore / configuration.')

    storage = storage_init(
        config=storage_config,
        dimensions=embedder.dimensions,
        index_name=index_name,
    )

    # if isinstance(storage_config, RedisConfig):
    #     storage = _create_redis_storage(
    #         config=storage_config,
    #         dimensions=embedder.dimensions,
    #         index_name=index_name,
    #     )
    # else:
    #     storage = None
    #
    # if storage is None:
    #     raise ConfigurationError('no Storage / configuration.')

    ragstore = RAGStore(embedder=embedder, storage=storage)
    manager = RAGManager(
        config=manager_config,
        ragstore=ragstore,
    )

    return manager
