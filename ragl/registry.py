import logging
from contextlib import suppress
from typing import Any, Self

from ragl.config import (
    RaglConfig,
    EmbedderConfig,
    ManagerConfig,
    RedisConfig,
    SentenceTransformerConfig,
    StorageConfig,
)
from ragl.exceptions import ConfigurationError
from ragl.manager import RAGManager
from ragl.ragstore import RAGStore


__all__ = ('create_rag_manager',)


_LOG = logging.getLogger(__name__)


class AbstractFactory:

    _config_cls: type[RaglConfig] = RaglConfig
    _factory_map: dict[str, type[Self]] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        config_cls = kwargs.pop('config_cls', cls._config_cls)
        super().__init_subclass__(**kwargs)

        if cls._should_create_new_factory_map():
            _LOG.info('Creating new factory map for %s', cls.__name__)
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

        _LOG.info('Registering factory class %s', cls.__name__)
        cls._factory_map[config_cls.__name__] = factory

    @classmethod
    def unregister_cls(cls, config_cls: type[RaglConfig]) -> None:
        _LOG.info('Unregistering factory class %s', cls.__name__)
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
        try:
            config = kwargs['config']
        except KeyError as e:
            raise ConfigurationError('config must be provided') from e
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
        try:
            config = kwargs['config']
            dimensions = kwargs['dimensions']
            index_name = kwargs['index_name']
        except KeyError as e:
            raise ConfigurationError('config, dimensions, and index_name '
                                     'must be provided') from e

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
    embedder_factory = EmbedderFactory()
    embedder = embedder_factory(config=embedder_config)

    storage_factory = VectorStoreFactory()
    storage = storage_factory(
        config=storage_config,
        dimensions=embedder.dimensions,
        index_name=index_name,
    )

    ragstore = RAGStore(
        embedder=embedder,
        storage=storage,
    )

    manager = RAGManager(
        config=manager_config,
        ragstore=ragstore,
    )

    return manager
