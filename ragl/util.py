from contextlib import suppress
from typing import Any, Callable, Self

from ragl.exceptions import ConfigurationError
from ragl.manager import RAGManager
from ragl.ragstore import RAGStore
from ragl.config import (
    RaglConfig,
    EmbedderConfig,
    ManagerConfig,
    RedisConfig,
    SentenceTransformerConfig,
    StorageConfig,
)


__all__ = ('create_rag_manager',)


class AbstractFactory:

    _config_cls: type[RaglConfig] = RaglConfig
    _factory_map: dict[str, type[Self]] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        config_cls = kwargs.pop('config_cls', cls._config_cls)
        super().__init_subclass__(**kwargs)

        if cls._should_create_new_factory_map():
            print('Creating new factory map for', cls.__name__)  # todo log
            cls._factory_map = {}

        cls.register_cls(config_cls=config_cls, factory=cls)

    @classmethod
    def register_cls(cls, config_cls: type[RaglConfig], factory) -> None:
        if not issubclass(config_cls, RaglConfig):
            raise TypeError('config_cls must be a subclass of '
                            f'RaglConfig, got {config_cls.__name__}')

        if not issubclass(cls, AbstractFactory):
            raise TypeError('cls must be a subclass of '
                            f'EmbedderFactory, got {config_cls.__name__}')

        cls._factory_map[config_cls.__name__] = factory

    @classmethod
    def unregister_cls(cls, config_cls: type[RaglConfig]) -> None:
        with suppress(KeyError):
            del cls._factory_map[config_cls.__name__]

    def __call__(self, *args, **kwargs) -> Any:
        config = kwargs.get('config', RaglConfig())
        try:
            factory_cls = self._factory_map[config.__class__.__name__]
        except KeyError as e:
            raise ConfigurationError('No factory registered for this '
                                     f'configuration type: {e}') from e
        factory = factory_cls()
        return factory(*args, **kwargs)

    @classmethod
    def _should_create_new_factory_map(cls) -> bool:
        """Check if this class should get its own factory map."""
        # Return True only if this is a direct subclass of AbstractFactory
        return AbstractFactory in cls.__bases__


class EmbedderFactory(AbstractFactory):
    pass


class SentenceTransformerFactory(EmbedderFactory):

    _config_cls = SentenceTransformerConfig

    def __call__(self, *args, **kwargs) -> Any:
        config = kwargs.get('config', SentenceTransformerConfig())  # todo raise if config not set?
        try:
            # pylint: disable=import-outside-toplevel
            from ragl.embed.sentencetransformer import SentenceTransformerEmbedder  # noqa: E501
            return SentenceTransformerEmbedder(config=config)
        except ImportError as e:
            raise ConfigurationError('SentenceTransformerEmbedder '
                                     'not available') from e


class VectorStoreFactory(AbstractFactory):
    pass


class RedisVectorStoreFactory(VectorStoreFactory):

    _config_cls = RedisConfig

    def __call__(self, *args, **kwargs) -> Any:
        config = kwargs.get('config', RedisConfig())  # todo raise if config not set?
        dimensions = kwargs.get('dimensions', 0)  # todo raise if dimensions not set
        index_name = kwargs.get('index_name', 'default_index')  # todo raise if index_name not set
        try:
            # pylint: disable=import-outside-toplevel
            from ragl.store.redis import RedisVectorStore
            return RedisVectorStore(
                redis_config=config,
                dimensions=dimensions,
                index_name=index_name,
            )
        except ImportError as e:
            raise ConfigurationError('RedisVectorStore not available') from e


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

    # embedder = embedder_init(config=embedder_config)

    # todo experimental
    embedder_factory = EmbedderFactory()
    embedder = embedder_factory(config=embedder_config)

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

    # storage = storage_init(
    #     config=storage_config,
    #     dimensions=embedder.dimensions,
    #     index_name=index_name,
    # )

    # todo experimental
    storage_factory = VectorStoreFactory()
    storage = storage_factory(
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
