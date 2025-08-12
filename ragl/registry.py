"""
Factory registry system for creating ragl components.

This module provides an abstract factory pattern implementation for
creating embedders, vector stores, and RAG managers based on
configuration objects. The registry system allows for dynamic
registration and instantiation of different implementation classes.

Classes:
- AbstractFactory:
    Base factory class with automatic registration mechanism
- EmbedderFactory:
    Factory for creating text embedding implementations
- VectorStoreFactory:
    Factory for creating vector storage implementations
- SentenceTransformerFactory:
    Factory for creating SentenceTransformer embedder instances
- RedisVectorStoreFactory:
    Factory for creating RedisVectorStore instances

Functions:
- create_rag_manager:
    Convenience function for complete ragl setup from configurations
"""

import logging
from contextlib import suppress
from typing import Any, Self

from ragl.config import (
    EmbedderConfig,
    ManagerConfig,
    RaglConfig,
    RedisConfig,
    SentenceTransformerConfig,
    VectorStoreConfig,
)
from ragl.exceptions import ConfigurationError
from ragl.manager import RAGManager
from ragl.ragstore import RAGStore


__all__ = (
    'AbstractFactory',
    'EmbedderFactory',
    'RedisVectorStoreFactory',
    'SentenceTransformerFactory',
    'VectorStoreFactory',
    'create_rag_manager',
)


_LOG = logging.getLogger(__name__)


class AbstractFactory:
    """
    Base factory class implementing registration and factory pattern.

    This abstract factory provides a registration mechanism where
    subclasses automatically register themselves with their associated
    configuration classes. Each direct subclass of AbstractFactory
    maintains its own factory map for type-safe instantiation of
    concrete implementations.

    Attributes:
        can_call_abstract__call__:
            List of factory class names that can call the abstract
            __call__ method. This is used to prevent subclasses from
            calling the abstract __call__ method directly.
        _config_cls:
            Configuration class type associated with this factory
        _factory_map:
            Dictionary mapping config class types to factory instances

    Class Methods:
        register_cls:
            Manually register a configuration class with a factory
        unregister_cls:
            Remove a configuration class from the factory registry

    The factory uses configuration class names as keys to determine
    which concrete implementation to instantiate. Subclasses should set
    _config_cls or pass config_cls in __init_subclass__ to enable
    automatic registration.

    Raises:
        ConfigurationError:
            When no factory is found for a configuration type
        TypeError:
            When invalid types are passed to registration methods
    """

    can_call_abstract__call__ = ['EmbedderFactory', 'VectorStoreFactory']

    _config_cls: type[RaglConfig] | None = None
    _factory_map: dict[type[RaglConfig], type[Self]] = {}

    def __init_subclass__(cls, **kwargs):
        """
        Initialize the subclass and set up factory map.

        Args:
            *args:
                Positional arguments (not used).
            **kwargs:
                Keyword arguments. Must include 'config_cls' if the
                'config_cls' attribute is not set on the subclass. Other
                keyword arguments are ignored and passed to the parent
                class's __init_subclass__ method.
        """
        config_cls = kwargs.pop('config_cls', cls._config_cls)
        super().__init_subclass__(**kwargs)

        if cls._is_direct_subclass():
            _LOG.debug('Creating new factory map for %s', cls.__name__)
            cls._factory_map = {}

        if config_cls is not None:
            cls.register_cls(config_cls=config_cls, factory=cls)

    @classmethod
    def register_cls(cls, config_cls: type[RaglConfig], factory) -> None:
        """
        Register a factory class with a configuration class.

        Args:
            config_cls:
                The configuration class type to register with the
                factory. Must be a subclass of RaglConfig.
            factory:
                The factory class to register. Must be a subclass of
                AbstractFactory.

        Raises:
            TypeError:
                If config_cls is not a subclass of RaglConfig or
                if factory is not a subclass of AbstractFactory.
        """
        _LOG.info('Registering factory class %s', cls.__name__)
        if not issubclass(config_cls, RaglConfig):
            msg = 'config_cls must be a subclass of RaglConfig'
            _LOG.critical(msg)
            raise TypeError(msg)

        if not issubclass(factory, AbstractFactory):
            msg = 'cls must be a subclass of AbstractFactory'
            _LOG.critical(msg)
            raise TypeError(msg)

        cls._factory_map[config_cls] = factory

    @classmethod
    def unregister_cls(cls, config_cls: type[RaglConfig]) -> None:
        """
        Unregister a factory class from the configuration class.

        Args:
            config_cls:
                The configuration class type to unregister from the
                factory.
        """
        _LOG.info('Unregistering factory class %s', cls.__name__)
        with suppress(KeyError):
            del cls._factory_map[config_cls]

    def __call__(self, *args, **kwargs) -> Any:
        """
        Create a concrete implementation based on configuration.

        Looks up the appropriate factory using the configuration class
        name and delegates instantiation to that factory.

        Subclasses should implement this method to instantiate
        and return specific classes based on the provided args and
        kwargs.

        Concrete implementations should not call this method directly;
        they should override it without calling `super().__call__` and
        implement the instantiation logic specific to their concrete
        type.

        Args:
            *args:
                Positional arguments for instantiation.
            **kwargs:
                Keyword arguments. Must include 'config' with a
                RaglConfig instance.

        Returns:
            Instance created by the appropriate concrete factory.

        Raises:
            ConfigurationError:
                When no factory found for configuration type.
        """
        _LOG.debug('Calling factory %s with args: %s, kwargs: %s',)
        if self.__class__.__name__ not in self.can_call_abstract__call__:
            msg = f'{self.__class__.__name__} cannot be called directly'
            _LOG.critical(msg)
            raise TypeError(msg)

        try:
            config = kwargs['config']
        except KeyError as e:
            msg = 'config must be provided'
            _LOG.critical(msg)
            raise ConfigurationError(msg) from e

        try:
            factory_cls = self._factory_map[type(config)]
        except KeyError as e:
            msg = f'No factory found for configuration type: {e}'
            _LOG.critical(msg)
            raise ConfigurationError(msg) from e

        factory = factory_cls()
        return factory(*args, **kwargs)

    @classmethod
    def _is_direct_subclass(cls) -> bool:
        """
        Check whether this class should get its own factory map.

        This method indicates whether the class is a direct subclass of
        AbstractFactory.

        Returns:
            bool:
                True if this class is a direct subclass of
                AbstractFactory, False otherwise.
        """
        _LOG.debug('Checking whether this class is a direct subclass of '
                   'AbstractFactory: %s', cls.__name__)
        # Return True only if this is a direct subclass of AbstractFactory
        return AbstractFactory in cls.__bases__

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the factory.

        Returns:
            A string showing the factory class name and registered
            configurations.
        """
        registered = list(self._factory_map.keys())
        return f'{self.__class__.__name__}(registered_configs={registered})'

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the factory.

        Returns:
            A string showing the factory class name, config class, and
            factory map.
        """
        if self._config_cls:
            config_cls_name = self._config_cls.__name__
        else:
            config_cls_name = None

        return (f'{self.__class__.__name__}('
                f'config_cls={config_cls_name}, '
                f'factory_map={self._factory_map})')


class EmbedderFactory(AbstractFactory):
    """
    Factory for creating text embedding implementations.

    This is the base factory for all text embedding
    classes which implement the EmbedderProtocol.

    This class provides a common namespace for all EmbedderFactory
    subclasses, allowing them to be registered and instantiated
    as part of the same registry mapping.

    Subclasses should implement the __call__ method to
    instantiate specific embedder classes based on the provided
    parameters.
    """


class SentenceTransformerFactory(EmbedderFactory):
    """Factory for creating SentenceTransformer embedder instances."""

    _config_cls = SentenceTransformerConfig

    def __call__(self, *args, **kwargs) -> Any:
        """
        Create a SentenceTransformerEmbedder instance.

        This method expects a 'config' keyword argument of type
        SentenceTransformerConfig. It raises a ConfigurationError if
        the config is not provided or if the SentenceTransformerEmbedder
        class is not available.

        Args:
            *args:
                Positional arguments (not used).
            **kwargs:
                Keyword arguments containing 'config'.
        """
        try:
            config = kwargs['config']
        except KeyError as e:
            msg = 'config parameter must be provided'
            _LOG.critical(msg)
            raise ConfigurationError(msg) from e
        try:
            # pylint: disable=import-outside-toplevel
            from ragl.embed.sentencetransformer import SentenceTransformerEmbedder  # noqa: E501
            return SentenceTransformerEmbedder(config=config)
        except ImportError as e:
            msg = 'SentenceTransformerEmbedder not available'
            _LOG.critical(msg)
            raise ConfigurationError(msg) from e


class VectorStoreFactory(AbstractFactory):
    """
    Factory for creating VectorStore implementations.

    This is the base factory for all vector storage
    classes which implement the VectorStoreProtocol.

    This class provides a common namespace for all VectorStoreFactory
    subclasses, allowing them to be registered and instantiated
    as part of the same registry mapping.

    Subclasses should implement the __call__ method to instantiate
    specific storage classes based on the provided parameters.
    """


class RedisVectorStoreFactory(VectorStoreFactory):
    """Factory for creating RedisVectorStore instances."""

    _config_cls = RedisConfig

    def __call__(self, *args, **kwargs) -> Any:
        """
        Create a RedisVectorStore instance.

        This method expects 'config', 'dimensions', and 'index_name'
        keyword arguments. It raises a ConfigurationError if any of
        these are not provided or if the RedisVectorStore class is
        not available.

        Args:
            *args:
                Positional arguments (not used).
            **kwargs:
                Keyword arguments containing 'config', 'dimensions',
                and 'index_name'.
        """
        _LOG.debug('Calling RedisVectorStore %s with args: %s, kwargs: %s',)
        try:
            config = kwargs['config']
            dimensions = kwargs['dimensions']
            index_name = kwargs['index_name']
        except KeyError as e:
            msg = 'config, dimensions, and index_name must be provided'
            _LOG.critical(msg)
            raise ConfigurationError(msg) from e

        try:
            # pylint: disable=import-outside-toplevel
            from ragl.store.redis import RedisVectorStore
            return RedisVectorStore(
                redis_config=config,
                dimensions=dimensions,
                index_name=index_name,
            )
        except ImportError as e:
            msg = 'RedisVectorStore not available'
            _LOG.critical(msg)
            raise ConfigurationError(msg) from e


def create_rag_manager(
        *,
        index_name: str,
        storage_config: VectorStoreConfig,
        embedder_config: EmbedderConfig,
        manager_config: ManagerConfig,
) -> RAGManager:
    """
    Create a RAGManager instance with injected components.

    Create a complete RAG system by instantiating Embedder and
    VectorStore components based on configurations, then assembling them
    and injecting into a RAGManager.

    Args:
        index_name:
            Name of the vector store index.
        storage_config:
            VectorStore configuration, represented by VectorStoreConfig.
        embedder_config:
            Embedder configuration, represented by EmbedderConfig.
        manager_config:
            RAGManager configuration, represented by ManagerConfig.

    Returns:
        A new RAGManager instance.

    Example:
        >>> create_rag_manager(
        ...     index_name="my_index",
        ...     storage_config=VectorStoreConfig(),
        ...     embedder_config=EmbedderConfig(),
        ...     manager_config=ManagerConfig(),
        ... )
    """
    _LOG.info('Creating RAG manager')
    embedder_factory = EmbedderFactory()
    embedder = embedder_factory(config=embedder_config)

    _LOG.info('Creating embedder')
    storage_factory = VectorStoreFactory()
    storage = storage_factory(
        config=storage_config,
        dimensions=embedder.dimensions,
        index_name=index_name,
    )

    _LOG.info('Creating vector store')
    ragstore = RAGStore(
        embedder=embedder,
        storage=storage,
    )

    _LOG.info('Creating RAG store')
    manager = RAGManager(
        config=manager_config,
        ragstore=ragstore,
    )

    _LOG.debug('%s created with embedder: %s, storage: %s',
               manager.__class__.__name__,
               embedder.__class__.__name__,
               storage.__class__.__name__,
               )

    return manager
