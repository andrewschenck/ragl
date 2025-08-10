"""
Metadata schema validation and sanitization utilities.

This module provides tools for defining and enforcing schemas on
metadata dictionaries, ensuring data consistency and type safety
for metadata fields used throughout the library.

Classes:
- SchemaField:
    TypedDict for defining field validation rules.

Functions:
- sanitize_metadata:
    Function to validate and clean metadata according to schema.
"""

import logging
from typing import (
    Any,
    Callable,
    Mapping,
    TypedDict,
)


__all__ = (
    'SchemaField',
    'sanitize_metadata',
)


_LOG = logging.getLogger(__name__)


class SchemaField(TypedDict, total=False):
    """
    Schema for metadata fields.

    Defines expected type, default, and optional conversion for
    fields.

    Attributes:
        type:
            Expected type of the field value (e.g., int, str, list).
        default:
            Default value to use if the field is missing or invalid.
        convert:
            Optional callable to convert the field value to the
            expected type. If not provided, the value must match
            the expected type directly.
    """

    type: type
    default: Any
    convert: Callable[[Any], Any]


def sanitize_metadata(
        metadata: Mapping[str, Any] | None,
        schema: Mapping[str, SchemaField] | None = None,
) -> dict[str, Any]:
    """
    Sanitize metadata for store.

    Process raw metadata according to an optional schema, ensuring
    fields match expected types, applying conversions if specified,
    and filling in defaults for missing or invalid values.

    Args:
        metadata:
            Raw metadata to sanitize (key-value Mapping).
        schema:
            Optional mapping of field names to type, default, and
            conversion rules.

    Returns:
        Sanitized metadata as a dictionary.
    """
    _LOG.debug('Sanitizing metadata')
    if not metadata:
        return {}

    default_schema: dict[str, SchemaField] = {
        'chunk_position': {
            'type':    int,
            'default': 0,
        },
        'timestamp': {
            'type':    int,
            'default': 0,
        },
        'confidence': {
            'type':    float,
            'default': 0.0,
        },
        'tags': {
            'type':    list,
            'default': [],
            'convert': lambda x: [str(x)] if not isinstance(x, list) else x,
        },
        'parent_id': {
            'type':    str,
            'default': '',
        },
        'source': {
            'type':    str,
            'default': '',
        },
        'language': {
            'type':    str,
            'default': '',
        },
        'section': {
            'type':    str,
            'default': '',
        },
        'author': {
            'type':    str,
            'default': '',
        },
    }
    schema = schema or default_schema

    sanitized = {}

    for key, value in metadata.items():
        if key in schema:
            spec = schema[key]
            field_type = spec['type']
            default_value = spec.get('default', '')
            convert_func = spec.get('convert', lambda x: x)

            if value is None:
                sanitized[key] = default_value
            else:
                converted_value = convert_func(value)
                if isinstance(converted_value, field_type):
                    sanitized[key] = converted_value
                else:
                    try:
                        sanitized[key] = field_type(converted_value)
                    except (ValueError, TypeError):
                        sanitized[key] = default_value
        else:
            sanitized[key] = value if value is not None else ''

    return sanitized
