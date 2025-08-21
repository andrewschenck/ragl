import unittest
from unittest.mock import patch

from ragl.exceptions import ValidationError
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
        with patch('time.time_ns', return_value=1500000000000):
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

    def test_from_dict_partial_data(self):
        """Test creating TextUnit from dictionary with partial data."""
        partial_data = {
            'text_id':  'partial_id',
            'text':     'Partial content',
            'distance': 0.7,
            'tags':     ['single_tag']
        }

        with patch('time.time_ns', return_value=1700000000000):
            unit = TextUnit.from_dict(partial_data)

        self.assertEqual(unit.text_id, 'partial_id')
        self.assertEqual(unit.text, 'Partial content')
        self.assertEqual(unit.distance, 0.7)
        self.assertEqual(unit.tags, ['single_tag'])
        self.assertEqual(unit.timestamp, 1700000000)

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
            "TextUnit(text=\"'Short text'\", "
            "text_id='repr_test', "
            "parent_id='parent_repr', "
            "distance=0.6, "
            "chunk_position=5)"
        )
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
            "TextUnit(text=\"'This is a very long text that exceeds fifty charac'...\", "
            "text_id='long_repr', "
            "parent_id=None, "
            "distance=0.8, "
            "chunk_position=None)"
        )
        self.assertEqual(result, expected)

    def test_timestamp_default_factory(self):
        """Test that timestamp uses current time when not specified."""
        with patch('time.time_ns', return_value=1600000000000):
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

    def test_eq_identical_units(self):
        """Test equality between identical TextUnits."""
        unit1 = TextUnit(
            text_id='test_id',
            text='Sample text',
            chunk_position=0,
            parent_id='parent_1',
            source='test_source',
            timestamp=1234567890,
            tags=['tag1', 'tag2'],
            confidence=0.95,
            language='en',
            section='intro',
            author='test_author',
            distance=0.5
        )

        unit2 = TextUnit(
            text_id='test_id',
            text='Sample text',
            chunk_position=0,
            parent_id='parent_1',
            source='test_source',
            timestamp=1234567890,
            tags=['tag1', 'tag2'],
            confidence=0.95,
            language='en',
            section='intro',
            author='test_author',
            distance=0.5
        )

        self.assertEqual(unit1, unit2)

    def test_eq_different_text_ids(self):
        """Test inequality when text_ids differ."""
        unit1 = TextUnit(text_id='id1', text='Same text')
        unit2 = TextUnit(text_id='id2', text='Same text')

        self.assertNotEqual(unit1, unit2)

    def test_eq_different_text_content(self):
        """Test inequality when text content differs."""
        unit1 = TextUnit(text_id='same_id', text='Text one')
        unit2 = TextUnit(text_id='same_id', text='Text two')

        self.assertNotEqual(unit1, unit2)

    def test_eq_different_metadata(self):
        """Test inequality when metadata differs."""
        unit1 = TextUnit(text_id='id', text='text', source='source1')
        unit2 = TextUnit(text_id='id', text='text', source='source2')

        self.assertNotEqual(unit1, unit2)

    def test_eq_with_non_textunit(self):
        """Test inequality when comparing with non-TextUnit object."""
        unit = TextUnit(text_id='id', text='text')

        self.assertNotEqual(unit, 'string')
        self.assertNotEqual(unit, {'text_id': 'id', 'text': 'text'})
        self.assertNotEqual(unit, None)

    def test_len_returns_text_length(self):
        """Test that len() returns the length of the text content."""
        unit = TextUnit(text_id='id', text='Hello world')
        self.assertEqual(len(unit), 11)

    def test_len_multiline_text(self):
        """Test len() with multiline text."""
        text = 'Line one\nLine two\nLine three'
        unit = TextUnit(text_id='id', text=text)
        self.assertEqual(len(unit), len(text))

    def test_len_unicode_text(self):
        """Test len() with unicode characters."""
        text = 'Hello 世界 🌍'
        unit = TextUnit(text_id='id', text=text)
        self.assertEqual(len(unit), len(text))

    def test_text_setter_validation_none(self):
        """Test text setter raises ValidationError for None."""
        unit = TextUnit(text_id='test', text='initial')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.text = None

            self.assertIn('text must be a string', str(cm.exception))
            mock_log.error.assert_called_with('text must be a string')

    def test_text_setter_validation_non_string(self):
        """Test text setter raises ValidationError for non-string."""
        unit = TextUnit(text_id='test', text='initial')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.text = 123

            self.assertIn('text must be a string', str(cm.exception))
            mock_log.error.assert_called_with('text must be a string')

    def test_text_setter_validation_empty_string(self):
        """Test text setter raises ValidationError for empty string."""
        unit = TextUnit(text_id='test', text='initial')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.text = ''

            self.assertIn('text cannot be whitespace-only or zero-length',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'text cannot be whitespace-only or zero-length')

    def test_text_setter_validation_whitespace_only(self):
        """Test text setter raises ValidationError for whitespace-only string."""
        unit = TextUnit(text_id='test', text='initial')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.text = '   \t\n  '

            self.assertIn('text cannot be whitespace-only or zero-length',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'text cannot be whitespace-only or zero-length')

    def test_text_id_setter_validation_non_string(self):
        """Test text_id setter raises ValidationError for non-string."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.text_id = 123

            self.assertIn('text_id must be a string', str(cm.exception))
            mock_log.error.assert_called_with('text_id must be a string')

    def test_text_id_setter_validation_empty_string(self):
        """Test text_id setter raises ValidationError for empty string."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.text_id = ''

            self.assertIn('text_id cannot be empty or whitespace-only',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'text_id cannot be empty or whitespace-only')

    def test_text_id_setter_validation_whitespace_only(self):
        """Test text_id setter raises ValidationError for whitespace-only string."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.text_id = '   '

            self.assertIn('text_id cannot be empty or whitespace-only',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'text_id cannot be empty or whitespace-only')

    def test_text_id_setter_none_allowed(self):
        """Test text_id setter allows None."""
        unit = TextUnit(text_id='test', text='content')
        unit.text_id = None
        self.assertIsNone(unit.text_id)

    def test_distance_setter_validation_non_numeric(self):
        """Test distance setter raises ValidationError for non-numeric."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.distance = 'invalid'

            self.assertIn('Distance must be a number', str(cm.exception))
            mock_log.error.assert_called_with('Distance must be a number')

    def test_distance_setter_validation_below_range(self):
        """Test distance setter raises ValidationError for value below 0.0."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.distance = -0.1

            self.assertIn('distance must be between 0.0 and 1.0',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'distance must be between 0.0 and 1.0')

    def test_distance_setter_validation_above_range(self):
        """Test distance setter raises ValidationError for value above 1.0."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.distance = 1.1

            self.assertIn('distance must be between 0.0 and 1.0',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'distance must be between 0.0 and 1.0')

    def test_distance_setter_accepts_int(self):
        """Test distance setter accepts integer values."""
        unit = TextUnit(text_id='test', text='content')
        unit.distance = 1
        self.assertEqual(unit.distance, 1.0)

    def test_parent_id_setter_validation_non_string(self):
        """Test parent_id setter raises ValidationError for non-string."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.parent_id = 123

            self.assertIn('parent_id must be a string', str(cm.exception))
            mock_log.error.assert_called_with('parent_id must be a string')

    def test_parent_id_setter_validation_empty_string(self):
        """Test parent_id setter raises ValidationError for empty string."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.parent_id = ''

            self.assertIn('parent_id cannot be empty or whitespace-only',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'parent_id cannot be empty or whitespace-only')

    def test_parent_id_setter_validation_whitespace_only(self):
        """Test parent_id setter raises ValidationError for whitespace-only string."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.parent_id = '   '

            self.assertIn('parent_id cannot be empty or whitespace-only',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'parent_id cannot be empty or whitespace-only')

    def test_chunk_position_setter_validation_non_int(self):
        """Test chunk_position setter raises ValidationError for non-integer."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.chunk_position = 'invalid'

            self.assertIn('chunk_position must be an integer',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'chunk_position must be an integer')

    def test_chunk_position_setter_validation_negative(self):
        """Test chunk_position setter raises ValidationError for negative value."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.chunk_position = -1

            self.assertIn('chunk_position must be non-negative',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'chunk_position must be non-negative')

    def test_author_setter_validation_non_string(self):
        """Test author setter raises ValidationError for non-string."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.author = 123

            self.assertIn('author must be a string', str(cm.exception))
            mock_log.error.assert_called_with('author must be a string')

    def test_source_setter_validation_non_string(self):
        """Test source setter raises ValidationError for non-string."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.source = 123

            self.assertIn('source must be a string', str(cm.exception))
            mock_log.error.assert_called_with('source must be a string')

    def test_language_setter_validation_non_string(self):
        """Test language setter raises ValidationError for non-string."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.language = 123

            self.assertIn('language must be a string', str(cm.exception))
            mock_log.error.assert_called_with('language must be a string')

    def test_section_setter_validation_non_string(self):
        """Test section setter raises ValidationError for non-string."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.section = 123

            self.assertIn('section must be a string', str(cm.exception))
            mock_log.error.assert_called_with('section must be a string')

    def test_confidence_setter_validation_non_string_non_numeric(self):
        """Test confidence setter raises ValidationError for invalid type."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.confidence = []

            self.assertIn('confidence must be a string or number',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'confidence must be a string or number')

    def test_confidence_setter_validation_empty_string(self):
        """Test confidence setter raises ValidationError for empty string."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.confidence = ''

            self.assertIn('confidence cannot be empty or whitespace-only',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'confidence cannot be empty or whitespace-only')

    def test_confidence_setter_validation_whitespace_only_string(self):
        """Test confidence setter raises ValidationError for whitespace-only string."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.confidence = '   '

            self.assertIn('confidence cannot be empty or whitespace-only',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'confidence cannot be empty or whitespace-only')

    def test_tags_setter_validation_non_list(self):
        """Test tags setter raises ValidationError for non-list."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.tags = 'not a list'

            self.assertIn('Tags must be a list of strings', str(cm.exception))
            mock_log.error.assert_called_with('Tags must be a list of strings')

    def test_tags_setter_validation_non_string_elements(self):
        """Test tags setter raises ValidationError for non-string list elements."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.tags = ['valid', 123, 'also_valid']

            self.assertIn('tags must be strings', str(cm.exception))
            mock_log.error.assert_called_with('tags must be strings')

    def test_tags_setter_validation_empty_string_elements(self):
        """Test tags setter raises ValidationError for empty string elements."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.tags = ['valid', '', 'also_valid']

            self.assertIn('tags cannot be empty or whitespace-only',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'tags cannot be empty or whitespace-only')

    def test_tags_setter_validation_whitespace_only_elements(self):
        """Test tags setter raises ValidationError for whitespace-only string elements."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.tags = ['valid', '   ', 'also_valid']

            self.assertIn('tags cannot be empty or whitespace-only',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'tags cannot be empty or whitespace-only')

    def test_timestamp_setter_validation_non_int(self):
        """Test timestamp setter raises ValidationError for non-integer."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.timestamp = 'invalid'

            self.assertIn('Timestamp must be an int', str(cm.exception))
            mock_log.error.assert_called_with('Timestamp must be an int')

    def test_timestamp_setter_validation_negative(self):
        """Test timestamp setter raises ValidationError for negative value."""
        unit = TextUnit(text_id='test', text='content')

        with patch('ragl.textunit._LOG') as mock_log:
            with self.assertRaises(ValidationError) as cm:
                unit.timestamp = -1

            self.assertIn('timestamp must be a positive integer',
                          str(cm.exception))
            mock_log.error.assert_called_with(
                'timestamp must be a positive integer')

    def test_all_optional_setters_allow_none(self):
        """Test that all optional fields accept None values."""
        unit = TextUnit(text_id='test', text='content')

        # These should not raise exceptions
        unit.text_id = None
        unit.parent_id = None
        unit.chunk_position = None
        unit.author = None
        unit.source = None
        unit.language = None
        unit.section = None
        unit.confidence = None
        unit.tags = None

        # Verify all are None
        self.assertIsNone(unit.text_id)
        self.assertIsNone(unit.parent_id)
        self.assertIsNone(unit.chunk_position)
        self.assertIsNone(unit.author)
        self.assertIsNone(unit.source)
        self.assertIsNone(unit.language)
        self.assertIsNone(unit.section)
        self.assertIsNone(unit.confidence)
        self.assertIsNone(unit.tags)


if __name__ == '__main__':
    unittest.main()
