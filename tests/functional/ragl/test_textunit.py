import unittest
from unittest.mock import patch
from ragl.textunit import TextUnit


class TestTextUnit(unittest.TestCase):
    """Test cases for the TextUnit class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = {
            'text_id':        'test_id',
            'text':           'Sample text content',
            'chunk_position': 1,
            'parent_id':      'parent_123',
            'distance':       0.5,
            'source':         'test_source',
            'tags':           ['tag1', 'tag2'],
            'confidence':     0.95,
            'language':       'en',
            'section':        'introduction',
            'author':         'test_author',
            'timestamp':      1234567890
        }

    def test_textunit_creation_with_all_fields(self):
        """Test creating TextUnit with all fields specified."""
        unit = TextUnit(
            text_id='test_id',
            text='Test content',
            distance=0.3,
            chunk_position=2,
            parent_id='parent_456',
            source='test_source',
            tags=['tag1'],
            confidence=0.8,
            language='en',
            section='body',
            author='author_name',
            timestamp=1000000000
        )

        self.assertEqual(unit.text_id, 'test_id')
        self.assertEqual(unit.text, 'Test content')
        self.assertEqual(unit.distance, 0.3)
        self.assertEqual(unit.chunk_position, 2)
        self.assertEqual(unit.parent_id, 'parent_456')
        self.assertEqual(unit.source, 'test_source')
        self.assertEqual(unit.tags, ['tag1'])
        self.assertEqual(unit.confidence, 0.8)
        self.assertEqual(unit.language, 'en')
        self.assertEqual(unit.section, 'body')
        self.assertEqual(unit.author, 'author_name')
        self.assertEqual(unit.timestamp, 1000000000)

    def test_textunit_creation_minimal_fields(self):
        """Test creating TextUnit with only required fields."""
        with patch('time.time', return_value=1500000000):
            unit = TextUnit(
                text_id='minimal_id',
                text='Minimal content',
                distance=0.1
            )

        self.assertEqual(unit.text_id, 'minimal_id')
        self.assertEqual(unit.text, 'Minimal content')
        self.assertEqual(unit.distance, 0.1)
        self.assertIsNone(unit.chunk_position)
        self.assertIsNone(unit.parent_id)
        self.assertIsNone(unit.source)
        self.assertIsNone(unit.tags)
        self.assertIsNone(unit.confidence)
        self.assertIsNone(unit.language)
        self.assertIsNone(unit.section)
        self.assertIsNone(unit.author)
        self.assertEqual(unit.timestamp, 1500000000)

    def test_from_dict_complete_data(self):
        """Test creating TextUnit from dictionary with complete data."""
        unit = TextUnit.from_dict(self.sample_data)

        self.assertEqual(unit.text_id, 'test_id')
        self.assertEqual(unit.text, 'Sample text content')
        self.assertEqual(unit.chunk_position, 1)
        self.assertEqual(unit.parent_id, 'parent_123')
        self.assertEqual(unit.distance, 0.5)
        self.assertEqual(unit.source, 'test_source')
        self.assertEqual(unit.tags, ['tag1', 'tag2'])
        self.assertEqual(unit.confidence, 0.95)
        self.assertEqual(unit.language, 'en')
        self.assertEqual(unit.section, 'introduction')
        self.assertEqual(unit.author, 'test_author')
        self.assertEqual(unit.timestamp, 1234567890)

    def test_from_dict_empty_data(self):
        """Test creating TextUnit from empty dictionary."""
        with patch('time.time', return_value=2000000000):
            unit = TextUnit.from_dict({})

        self.assertEqual(unit.text_id, '')
        self.assertEqual(unit.text, '')
        self.assertEqual(unit.distance, 0.0)
        self.assertIsNone(unit.chunk_position)
        self.assertIsNone(unit.parent_id)
        self.assertIsNone(unit.source)
        self.assertIsNone(unit.tags)
        self.assertIsNone(unit.confidence)
        self.assertIsNone(unit.language)
        self.assertIsNone(unit.section)
        self.assertIsNone(unit.author)
        self.assertEqual(unit.timestamp, 2000000000)

    def test_from_dict_partial_data(self):
        """Test creating TextUnit from dictionary with partial data."""
        partial_data = {
            'text_id':  'partial_id',
            'text':     'Partial content',
            'distance': 0.7,
            'tags':     ['single_tag']
        }

        with patch('time.time', return_value=1700000000):
            unit = TextUnit.from_dict(partial_data)

        self.assertEqual(unit.text_id, 'partial_id')
        self.assertEqual(unit.text, 'Partial content')
        self.assertEqual(unit.distance, 0.7)
        self.assertEqual(unit.tags, ['single_tag'])
        self.assertEqual(unit.timestamp, 1700000000)

    def test_from_dict_tags_not_list(self):
        """Test handling non-list tags in from_dict."""
        data = {'tags': 'single_tag'}
        unit = TextUnit.from_dict(data)
        self.assertEqual(unit.tags, ['single_tag'])

    def test_from_dict_tags_with_quotes_and_brackets(self):
        """Test cleaning tags with quotes and brackets."""
        data = {'tags': ["'tag1'", '"tag2"', '[tag3]', " tag4 "]}
        unit = TextUnit.from_dict(data)
        self.assertEqual(unit.tags, ['tag1', 'tag2', 'tag3', 'tag4'])

    def test_from_dict_tags_none(self):
        """Test handling None tags in from_dict."""
        data = {'tags': None}
        unit = TextUnit.from_dict(data)
        self.assertIsNone(unit.tags)

    def test_to_dict(self):
        """Test converting TextUnit to dictionary."""
        unit = TextUnit(
            text_id='dict_test',
            text='Dict test content',
            distance=0.4,
            chunk_position=3,
            parent_id='parent_789',
            source='dict_source',
            tags=['dict_tag'],
            confidence=0.9,
            language='fr',
            section='conclusion',
            author='dict_author',
            timestamp=1800000000
        )

        result = unit.to_dict()
        expected = {
            'text_id':        'dict_test',
            'text':           'Dict test content',
            'chunk_position': 3,
            'parent_id':      'parent_789',
            'distance':       0.4,
            'source':         'dict_source',
            'timestamp':      1800000000,
            'tags':           ['dict_tag'],
            'confidence':     0.9,
            'language':       'fr',
            'section':        'conclusion',
            'author':         'dict_author',
        }

        self.assertEqual(result, expected)

    def test_to_dict_with_none_values(self):
        """Test to_dict includes None values for optional fields."""
        unit = TextUnit(text_id='none_test', text='None test', distance=0.0)
        result = unit.to_dict()

        self.assertIn('chunk_position', result)
        self.assertIsNone(result['chunk_position'])
        self.assertIn('parent_id', result)
        self.assertIsNone(result['parent_id'])
        self.assertIn('source', result)
        self.assertIsNone(result['source'])
        self.assertIn('tags', result)
        self.assertIsNone(result['tags'])

    def test_str_method(self):
        """Test string representation returns text content."""
        unit = TextUnit(text_id='str_test', text='String test content',
                        distance=0.0)
        self.assertEqual(str(unit), 'String test content')

    def test_str_method_empty_text(self):
        """Test string representation with empty text."""
        unit = TextUnit(text_id='empty_str', text='', distance=0.0)
        self.assertEqual(str(unit), '')

    def test_repr_method_short_text(self):
        """Test repr method with text shorter than 50 characters."""
        unit = TextUnit(
            text_id='repr_test',
            text='Short text',
            distance=0.6,
            chunk_position=5,
            parent_id='parent_repr'
        )

        result = repr(unit)
        expected = (
            "TextUnit(text_id='repr_test', "
            "text=\"'Short text'\", "
            "distance=0.6, "
            "chunk_position=5, "
            "parent_id='parent_repr')"
        )
        # expected = (
        #     "TextUnit(text_id='repr_test', "
        #     "text='Short text', "
        #     "distance=0.6, "
        #     "chunk_position=5, "
        #     "parent_id='parent_repr')"
        # )
        self.assertEqual(result, expected)

    def test_repr_method_long_text(self):
        """Test repr method with text longer than 50 characters."""
        long_text = "This is a very long text that exceeds fifty characters in length"
        unit = TextUnit(
            text_id="long_repr",
            text=long_text,
            distance=0.8,
            chunk_position=None,
            parent_id=None
        )

        result = repr(unit)
        expected = (
            "TextUnit(text_id='long_repr', "
            "text=\"'This is a very long text that exceeds fifty charac'...\", "
            "distance=0.8, "
            "chunk_position=None, "
            "parent_id=None)"
        )
        print(result)
        self.assertEqual(result, expected)

    def test_timestamp_default_factory(self):
        """Test that timestamp uses current time when not specified."""
        with patch('time.time', return_value=1600000000):
            unit = TextUnit(text_id='time_test', text='Time test',
                            distance=0.0)
            self.assertEqual(unit.timestamp, 1600000000)

    def test_confidence_string_value(self):
        """Test confidence field accepts string values."""
        unit = TextUnit(
            text_id='conf_test',
            text='Confidence test',
            distance=0.0,
            confidence='high'
        )
        self.assertEqual(unit.confidence, 'high')

    def test_from_dict_round_trip(self):
        """Test that from_dict and to_dict are inverse operations."""
        unit1 = TextUnit.from_dict(self.sample_data)
        dict_representation = unit1.to_dict()
        unit2 = TextUnit.from_dict(dict_representation)

        self.assertEqual(unit1.text_id, unit2.text_id)
        self.assertEqual(unit1.text, unit2.text)
        self.assertEqual(unit1.distance, unit2.distance)
        self.assertEqual(unit1.chunk_position, unit2.chunk_position)
        self.assertEqual(unit1.parent_id, unit2.parent_id)
        self.assertEqual(unit1.source, unit2.source)
        self.assertEqual(unit1.tags, unit2.tags)
        self.assertEqual(unit1.confidence, unit2.confidence)
        self.assertEqual(unit1.language, unit2.language)
        self.assertEqual(unit1.section, unit2.section)
        self.assertEqual(unit1.author, unit2.author)
        self.assertEqual(unit1.timestamp, unit2.timestamp)


if __name__ == '__main__':
    unittest.main()
