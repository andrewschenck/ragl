"""
Text unit data structures and utilities.

This module defines the core TextUnit class for representing stored
text chunks with associated metadata.

Classes:
- TextUnit:
    Dataclass for representing text chunks with metadata.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Self


__all__ = ('TextUnit',)


@dataclass
class TextUnit:
    """
    Represent a stored text chunk.

    Various optional metadata fields are included to provide context
    and facilitate retrieval.


    Attributes:
        text:
            Text content.
        text_id:
            Unique identifier.
        parent_id:
            ID of parent document.
        chunk_position:
            Position in parent text.
        distance:
            Similarity distance.
        source:
            Source of the text.
        confidence:
            Confidence score.
        language:
            Language of the text.
        section:
            Section within source.
        author:
            Author of the text.
        tags:
            List of tags.
        timestamp:
            Storage timestamp.
    """

    # pylint: disable=too-many-instance-attributes
    text: str
    text_id: str | None = None
    parent_id: str | None = None
    chunk_position: int | None = None
    distance: float = 1.0
    source: str | None = None
    confidence: float | str | None = None
    language: str | None = None
    section: str | None = None
    author: str | None = None
    tags: list[str] | None = None
    timestamp: int = field(default_factory=lambda: int(time.time()))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """
        Create a TextUnit instance from a dictionary.

        Expects a mapping with optional fields like text_id, text,
        timestamp, tags, etc., using defaults for missing values.

        Args:
            data:
                Dictionary containing text unit data.

        Returns:
            New TextUnit instance populated with provided data.
        """
        tags = data.get('tags')
        if tags is not None:
            if not isinstance(tags, list):
                tags = [str(tags)]
            # Clean up tags by removing quotes and brackets
            tags = [str(tag).strip("[]'\" ") for tag in tags]
        return cls(
            text=data.get('text', ''),
            text_id=data.get('text_id'),
            parent_id=data.get('parent_id'),
            chunk_position=data.get('chunk_position'),
            distance=data.get('distance', 1.0),
            source=data.get('source'),
            confidence=data.get('confidence'),
            language=data.get('language'),
            section=data.get('section'),
            author=data.get('author'),
            tags=tags,
            timestamp=data.get('timestamp', int(time.time())),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the TextUnit instance to a dictionary.

        Unset optional fields are included with None values.

        Returns:
            A dict representation of the instance.
        """
        return {
            'text':                 self.text,
            'text_id':              self.text_id,
            'parent_id':            self.parent_id,
            'chunk_position':       self.chunk_position,
            'distance':             self.distance,
            'source':               self.source,
            'confidence':           self.confidence,
            'language':             self.language,
            'section':              self.section,
            'author':               self.author,
            'tags':                 self.tags,
            'timestamp':            self.timestamp,
        }

    def __eq__(self, other: Any) -> bool:
        """
        Check equality with another object.

        Compares text_id and text attributes for equality.

        Args:
            other:
                Object to compare against.

        Returns:
            True if both objects are TextUnit instances with the
            same values for key attributes, False otherwise.
        """
        if not isinstance(other, TextUnit):
            return NotImplemented
        return (self.text == other.text and
                self.text_id == other.text_id and
                self.parent_id == other.parent_id and
                self.chunk_position == other.chunk_position and
                self.distance == other.distance and
                self.source == other.source and
                self.confidence == other.confidence and
                self.language == other.language and
                self.section == other.section and
                self.author == other.author and
                self.tags == other.tags and
                self.timestamp == other.timestamp)

    def __len__(self) -> int:
        """
        Get the length of the text content.

        Returns:
            Length of the text string.
        """
        return len(self.text)

    def __str__(self) -> str:
        """
        Convert instance to string.

        Returns:
            Text content as a string.
        """
        return self.text

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the TextUnit instance.

        Returns:
            A string representation showing key attributes.
        """
        sep = '...' if len(self.text) > 50 else ''
        return (
            f'TextUnit(text_id={self.text_id!r}, '
            f'text="{self.text[:50]!r}{sep}", '
            f'distance={self.distance}, '
            f'chunk_position={self.chunk_position}, '
            f'parent_id={self.parent_id!r})'
        )
