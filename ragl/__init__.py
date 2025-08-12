"""
ragl -- Text storage and retrieval for RAG use cases.

A Python library for building and managing Retrieval-Augmented
Generation (RAG) systems.
"""

import logging

from ragl.registry import create_rag_manager


__all__ = ('create_rag_manager',)


logging.getLogger(__name__).addHandler(logging.NullHandler())
