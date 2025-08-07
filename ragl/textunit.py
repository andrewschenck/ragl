import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Self


__all__ = ('TextUnit',)


@dataclass
class TextUnit:
    """
    Represent a stored text chunk.

    Attributes:
        text_id:
            Unique identifier.
        text:
            Text content.
        chunk_position:
            Position in parent text.
        parent_id:
            ID of parent document.
        distance:
            Similarity distance.
        source:
            Source of the text.
        tags:
            List of tags.
        confidence:
            Confidence score.
        language:
            Language of the text.
        section:
            Section within source.
        author:
            Author of the text.
        timestamp:
            Storage timestamp.
    """

    # pylint: disable=too-many-instance-attributes
    text_id: str
    text: str
    distance: float
    chunk_position: int | None = None
    parent_id: str | None = None
    source: str | None = None
    tags: list[str] | None = None
    confidence: float | str | None = None
    language: str | None = None
    section: str | None = None
    author: str | None = None
    timestamp: int = field(default_factory=lambda: int(time.time()))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """
        Create a TextUnit instance from a dictionary.

        Expects a mapping with optional fields like text_id, text,
        timestamp, tags, etc., using defaults for missing values.

        Args:
            data: Dictionary containing text unit data.

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
            text_id=data.get('text_id', ''),
            text=data.get('text', ''),
            chunk_position=data.get('chunk_position'),
            parent_id=data.get('parent_id'),
            distance=data.get('distance', 0.0),
            source=data.get('source'),
            tags=tags,
            confidence=data.get('confidence'),
            language=data.get('language'),
            section=data.get('section'),
            author=data.get('author'),
            timestamp=data.get('timestamp', int(time.time())),
        )

    def as_dict(self) -> dict[str, Any]:
        """
        Convert the TextUnit instance to a dictionary.

        Unset optional fields are included with None values.

        Returns:
            A dict representation of the instance.
        """
        return {
            'text_id':              self.text_id,
            'text':                 self.text,
            'chunk_position':       self.chunk_position,
            'parent_id':            self.parent_id,
            'distance':             self.distance,
            'source':               self.source,
            'timestamp':            self.timestamp,
            'tags':                 self.tags,
            'confidence':           self.confidence,
            'language':             self.language,
            'section':              self.section,
            'author':               self.author,
        }

    def __str__(self) -> str:
        """
        Convert instance to string.

        Returns:
            Text content as a string.
        """
        return self.text
