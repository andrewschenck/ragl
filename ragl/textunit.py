"""
Text unit data structures and utilities.

This module defines the core TextUnit class for representing stored
text chunks with associated metadata.

Classes:
    TextUnit:
        Dataclass for representing text chunks with metadata.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Self

from ragl.exceptions import ValidationError


__all__ = ('TextUnit',)


_LOG = logging.getLogger(__name__)


@dataclass
class TextUnit:
    # pylint: disable=too-many-instance-attributes
    """
    Represent a stored text chunk.

    Various optional metadata fields are included to provide context
    and facilitate retrieval.

    Attributes:
        author:
            Author of the text.
        chunk_position:
            Position in parent text.
        confidence:
            Confidence score.
        distance:
            Similarity distance.
        language:
            Language of the text.
        parent_id:
            ID of parent document.
        section:
            Section within source.
        source:
            Source of the text.
        tags:
            List of tags.
        text:
            Text content.
        text_id:
            Unique identifier.
        timestamp:
            Storage timestamp.
    """

    _author: str | None = field(init=False)
    _chunk_position: int | None = field(init=False)
    _confidence: float | str | None = field(init=False)
    _distance: float = field(init=False)
    _language: str | None = field(init=False)
    _parent_id: str | None = field(init=False)
    _section: str | None = field(init=False)
    _source: str | None = field(init=False)
    _tags: list[str] | None = field(init=False)
    _text: str = field(init=False)
    _text_id: str | None = field(init=False)
    _timestamp: int = field(init=False)

    @property
    def author(self) -> str | None:
        """
        Get the author of the text.

        Returns:
            The value of the author attribute, or None if unset.
        """
        return self._author

    @author.setter
    def author(self, value: str | None) -> None:
        if value is not None:
            if not isinstance(value, str):
                msg = 'author must be a string'
                _LOG.error(msg)
                raise ValidationError(msg)
        self._author = value

    @property
    def chunk_position(self) -> int | None:
        """
        Get the chunk position.

        Returns:
            The value of the chunk_position attribute, or None if unset.
        """
        return self._chunk_position

    @chunk_position.setter
    def chunk_position(self, value: int | None) -> None:
        if value is not None:
            if not isinstance(value, int):
                msg = 'chunk_position must be an integer'
                _LOG.error(msg)
                raise ValidationError(msg)
            if value < 0:
                msg = 'chunk_position must be non-negative'
                _LOG.error(msg)
                raise ValidationError(msg)
        self._chunk_position = value

    @property
    def confidence(self) -> float | str | None:
        """
        Get the confidence score.

        Returns:
            The value of the confidence attribute, or None if unset.
        """
        return self._confidence

    @confidence.setter
    def confidence(self, value: float | str | None) -> None:
        if value is not None:
            if isinstance(value, str):
                if not value.strip():
                    msg = 'confidence cannot be empty or whitespace-only'
                    _LOG.error(msg)
                    raise ValidationError(msg)
            elif not isinstance(value, (int, float)):
                msg = 'confidence must be a string or number'
                _LOG.error(msg)
                raise ValidationError(msg)
        self._confidence = value

    @property
    def distance(self) -> float:
        """
        Get the similarity distance.

        Returns:
            The value of the distance attribute.
        """
        return self._distance

    @distance.setter
    def distance(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            msg = 'Distance must be a number'
            _LOG.error(msg)
            raise ValidationError(msg)
        value = float(value)
        if not 0.0 <= value <= 1.0:
            msg = 'distance must be between 0.0 and 1.0'
            _LOG.error(msg)
            raise ValidationError(msg)
        self._distance = value

    @property
    def language(self) -> str | None:
        """
        Get the language of the text.

        Returns:
            The value of the language attribute, or None if unset.
        """
        return self._language

    @language.setter
    def language(self, value: str | None) -> None:
        if value is not None:
            if not isinstance(value, str):
                msg = 'language must be a string'
                _LOG.error(msg)
                raise ValidationError(msg)
        self._language = value

    @property
    def parent_id(self) -> str | None:
        """
        Get the parent document ID.

        Returns:
            The value of the parent_id attribute, or None if unset.
        """
        return self._parent_id

    @parent_id.setter
    def parent_id(self, value: str | None) -> None:
        if value is not None:
            if not isinstance(value, str):
                msg = 'parent_id must be a string'
                _LOG.error(msg)
                raise ValidationError(msg)
            if not value.strip():
                msg = 'parent_id cannot be empty or whitespace-only'
                _LOG.error(msg)
                raise ValidationError(msg)
        self._parent_id = value

    @property
    def section(self) -> str | None:
        """
        Get the section within the source.

        Returns:
            The value of the section attribute, or None if unset.
        """
        return self._section

    @section.setter
    def section(self, value: str | None) -> None:
        if value is not None:
            if not isinstance(value, str):
                msg = 'section must be a string'
                _LOG.error(msg)
                raise ValidationError(msg)
        self._section = value

    @property
    def source(self) -> str | None:
        """
        Get the source of the text.

        Returns:
            The value of the source attribute, or None if unset.
        """
        return self._source

    @source.setter
    def source(self, value: str | None) -> None:
        if value is not None:
            if not isinstance(value, str):
                msg = 'source must be a string'
                _LOG.error(msg)
                raise ValidationError(msg)
        self._source = value

    @property
    def tags(self) -> list[str] | None:
        """
        Get the list of tags.

        Returns:
            The value of the tags attribute, or None if unset.
        """
        return self._tags

    @tags.setter
    def tags(self, value: list[str] | None) -> None:
        if value is not None:
            if not isinstance(value, list):
                msg = 'Tags must be a list of strings'
                _LOG.error(msg)
                raise ValidationError(msg)
            if not all(isinstance(tag, str) for tag in value):
                msg = 'tags must be strings'
                _LOG.error(msg)
                raise ValidationError(msg)
            if not all(tag.strip() for tag in value):
                msg = 'tags cannot be empty or whitespace-only'
                _LOG.error(msg)
                raise ValidationError(msg)
        self._tags = value

    @property
    def text(self) -> str:
        """
        Get the text content.

        Returns:
            The value of the text attribute.
        """
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        if not isinstance(value, str):
            msg = 'text must be a string'
            _LOG.error(msg)
            raise ValidationError(msg)
        if not value.strip():
            msg = 'text cannot be whitespace-only or zero-length'
            _LOG.error(msg)
            raise ValidationError(msg)
        self._text = value

    @property
    def text_id(self) -> str | None:
        """
        Get the text ID.

        Returns:
            The value of the text_id attribute, or None if unset.
        """
        return self._text_id

    @text_id.setter
    def text_id(self, value: str | None) -> None:
        if value is not None:
            if not isinstance(value, str):
                msg = 'text_id must be a string'
                _LOG.error(msg)
                raise ValidationError(msg)
            if not value.strip():
                msg = 'text_id cannot be empty or whitespace-only'
                _LOG.error(msg)
                raise ValidationError(msg)
        self._text_id = value

    @property
    def timestamp(self) -> int:
        """
        Get the storage timestamp.

        Returns:
            The value of the timestamp attribute.
        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: int) -> None:
        if not isinstance(value, int):
            msg = 'Timestamp must be an int'
            _LOG.error(msg)
            raise ValidationError(msg)
        if value < 0:
            msg = 'timestamp must be a positive integer'
            _LOG.error(msg)
            raise ValidationError(msg)
        self._timestamp = value

    def __init__(
            self,
            text: str,
            *,
            text_id: str | None = None,
            parent_id: str | None = None,
            chunk_position: int | None = None,
            distance: float = 1.0,
            source: str | None = None,
            confidence: float | str | None = None,
            language: str | None = None,
            section: str | None = None,
            author: str | None = None,
            tags: list[str] | None = None,
            timestamp: int | None = None,
    ):
        # pylint: disable=too-many-arguments
        """
        Initialize TextUnit and validate attributes.

        Args:
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
                Storage timestamp in microseconds since epoch.
                If None, current time is used.
        """
        self.text = text
        self.text_id = text_id
        self.parent_id = parent_id
        self.chunk_position = chunk_position
        self.distance = distance
        self.source = source
        self.confidence = confidence
        self.language = language
        self.section = section
        self.author = author
        self.tags = tags
        self.timestamp = (
            timestamp if timestamp is not None else
            int(time.time_ns() // 1000)
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """
        Create a TextUnit instance from a dictionary.

        Expects a mapping with optional fields like text_id, text,
        timestamp, tags, etc., using defaults for missing values.

        Args:
            data:
                Mapping containing text unit data.

        Returns:
            New TextUnit instance populated with provided data.
        """
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
            tags=data.get('tags'),
            timestamp=data.get('timestamp'),
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

    def __contains__(self, item) -> bool:
        """
        Check if a substring is in the text content.

        Args:
            item:
                Substring to check for.

        Returns:
            True if item is found in text, False otherwise.
        """
        if not isinstance(item, str):
            return False
        return item in self.text

    def __eq__(self, other: Any) -> bool:
        """
        Check equality with another object.

        Compares all attributes for equality.

        Args:
            other:
                Object to compare against.

        Returns:
            True if both objects are TextUnit instances with the
            same values for all attributes, False otherwise.
        """
        if not isinstance(other, TextUnit):
            return NotImplemented
        return (
            self.text == other.text and
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
            self.timestamp == other.timestamp
        )

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
            f'TextUnit(text="{self.text[:50]!r}{sep}", '
            f'text_id={self.text_id!r}, '
            f'parent_id={self.parent_id!r}, '
            f'distance={self.distance}, '
            f'chunk_position={self.chunk_position})'
        )
