import unittest
from unittest.mock import patch

from ragl.schema import SchemaField, sanitize_metadata


class TestSchemaField(unittest.TestCase):
    """Test cases for SchemaField TypedDict."""

    def test_schema_field_creation(self):
        """Test creating SchemaField instances."""
        # Test with all fields
        field = SchemaField(
            type=str,
            default='test',
            convert=lambda x: str(x)
        )
        self.assertEqual(field['type'], str)
        self.assertEqual(field['default'], 'test')
        self.assertTrue(callable(field['convert']))

        # Test with partial fields (total=False)
        field_partial = SchemaField(type=int)
        self.assertEqual(field_partial['type'], int)
        self.assertNotIn('default', field_partial)
        self.assertNotIn('convert', field_partial)


class TestSanitizeMetadata(unittest.TestCase):
    """Test cases for sanitize_metadata function."""

    @patch('ragl.schema._LOG')
    def test_sanitize_metadata_empty_input(self, mock_log):
        """Test sanitizing empty or None metadata."""
        # Test None metadata
        result = sanitize_metadata(None)
        self.assertEqual(result, {})
        mock_log.debug.assert_called_with('Sanitizing metadata')

        # Test empty dict
        result = sanitize_metadata({})
        self.assertEqual(result, {})

    @patch('ragl.schema._LOG')
    def test_sanitize_metadata_with_default_schema(self, mock_log):
        """Test sanitizing metadata using default schema."""
        metadata = {
            'chunk_position': 5,
            'timestamp':      1234567890,
            'confidence':     0.95,
            'tags':           ['tag1', 'tag2'],
            'parent_id':      'parent123',
            'source':         'test_source',
            'language':       'python',
            'section':        'main',
            'author':         'test_author'
        }

        result = sanitize_metadata(metadata)

        self.assertEqual(result['chunk_position'], 5)
        self.assertEqual(result['timestamp'], 1234567890)
        self.assertEqual(result['confidence'], 0.95)
        self.assertEqual(result['tags'], ['tag1', 'tag2'])
        self.assertEqual(result['parent_id'], 'parent123')
        self.assertEqual(result['source'], 'test_source')
        self.assertEqual(result['language'], 'python')
        self.assertEqual(result['section'], 'main')
        self.assertEqual(result['author'], 'test_author')

    def test_sanitize_metadata_with_none_values(self):
        """Test sanitizing metadata with None values."""
        metadata = {
            'chunk_position': None,
            'timestamp':      None,
            'confidence':     None,
            'tags':           None,
            'parent_id':      None
        }

        result = sanitize_metadata(metadata)

        self.assertEqual(result['chunk_position'], 0)
        self.assertEqual(result['timestamp'], 0)
        self.assertEqual(result['confidence'], 0.0)
        self.assertEqual(result['tags'], [])
        self.assertEqual(result['parent_id'], '')

    def test_sanitize_metadata_with_type_conversion(self):
        """Test sanitizing metadata with automatic type conversion."""
        metadata = {
            'chunk_position': '5',
            'timestamp':      '1234567890',
            'confidence':     '0.95',
            'parent_id':      123,
            'source':         456
        }

        result = sanitize_metadata(metadata)

        self.assertEqual(result['chunk_position'], 5)
        self.assertEqual(result['timestamp'], 1234567890)
        self.assertEqual(result['confidence'], 0.95)
        self.assertEqual(result['parent_id'], '123')
        self.assertEqual(result['source'], '456')

    def test_sanitize_metadata_with_conversion_errors(self):
        """Test sanitizing metadata when type conversion fails."""
        metadata = {
            'chunk_position': 'invalid_int',
            'timestamp':      'invalid_timestamp',
            'confidence':     'invalid_float'
        }

        result = sanitize_metadata(metadata)

        # Should use default values when conversion fails
        self.assertEqual(result['chunk_position'], 0)
        self.assertEqual(result['timestamp'], 0)
        self.assertEqual(result['confidence'], 0.0)

    def test_sanitize_metadata_tags_conversion(self):
        """Test special tags field conversion logic."""
        # Test non-list value gets converted to single-item list
        metadata = {'tags': 'single_tag'}
        result = sanitize_metadata(metadata)
        self.assertEqual(result['tags'], ['single_tag'])

        # Test list value stays as list
        metadata = {'tags': ['tag1', 'tag2']}
        result = sanitize_metadata(metadata)
        self.assertEqual(result['tags'], ['tag1', 'tag2'])

        # Test None value gets default
        metadata = {'tags': None}
        result = sanitize_metadata(metadata)
        self.assertEqual(result['tags'], [])

    def test_sanitize_metadata_unknown_fields(self):
        """Test sanitizing metadata with fields not in schema."""
        metadata = {
            'unknown_field':   'some_value',
            'another_unknown': 123,
            'null_unknown':    None
        }

        result = sanitize_metadata(metadata)

        # Unknown fields should be preserved as-is, except None becomes ''
        self.assertEqual(result['unknown_field'], 'some_value')
        self.assertEqual(result['another_unknown'], 123)
        self.assertEqual(result['null_unknown'], '')

    def test_sanitize_metadata_with_custom_schema(self):
        """Test sanitizing metadata with custom schema."""
        custom_schema = {
            'custom_field':  {
                'type':    int,
                'default': 42,
                'convert': lambda x: int(x) if isinstance(x, str) else x
            },
            'another_field': {
                'type':    str,
                'default': 'default_string'
            }
        }

        metadata = {
            'custom_field':  '100',
            'another_field': 'test_value',
            'unknown_field': 'preserved'
        }

        result = sanitize_metadata(metadata, custom_schema)

        self.assertEqual(result['custom_field'], 100)
        self.assertEqual(result['another_field'], 'test_value')
        self.assertEqual(result['unknown_field'], 'preserved')

    def test_sanitize_metadata_custom_schema_with_defaults(self):
        """Test custom schema using defaults for missing/invalid values."""
        custom_schema = {
            'test_field': {
                'type':    int,
                'default': 999
            }
        }

        # Test missing field and invalid conversion
        metadata = {'test_field': 'invalid_int'}
        result = sanitize_metadata(metadata, custom_schema)
        self.assertEqual(result['test_field'], 999)

        # Test None value
        metadata = {'test_field': None}
        result = sanitize_metadata(metadata, custom_schema)
        self.assertEqual(result['test_field'], 999)

    def test_sanitize_metadata_custom_convert_function(self):
        """Test custom conversion function in schema."""

        def custom_converter(value):
            if isinstance(value, str):
                return value.upper()
            return str(value).upper()

        custom_schema = {
            'uppercase_field': {
                'type':    str,
                'default': 'DEFAULT',
                'convert': custom_converter
            }
        }

        metadata = {'uppercase_field': 'hello'}
        result = sanitize_metadata(metadata, custom_schema)
        self.assertEqual(result['uppercase_field'], 'HELLO')

        # Test with non-string input
        metadata = {'uppercase_field': 123}
        result = sanitize_metadata(metadata, custom_schema)
        self.assertEqual(result['uppercase_field'], '123')

    def test_sanitize_metadata_schema_without_default(self):
        """Test schema field without default value."""
        custom_schema = {
            'no_default_field': {
                'type': str
            }
        }

        metadata = {'no_default_field': None}
        result = sanitize_metadata(metadata, custom_schema)
        # Should use empty string as fallback when no default specified
        self.assertEqual(result['no_default_field'], '')

    def test_sanitize_metadata_complex_conversion_scenario(self):
        """Test complex conversion scenarios."""
        custom_schema = {
            'list_field': {
                'type':    list,
                'default': [],
                'convert': lambda x: x if isinstance(x, list) else [x]
            }
        }

        # Test successful conversion
        metadata = {'list_field': 'single_item'}
        result = sanitize_metadata(metadata, custom_schema)
        self.assertEqual(result['list_field'], ['single_item'])

        # Test when converted value doesn't match expected type after conversion
        failing_schema = {
            'failing_field': {
                'type':    int,
                'default': 0,
                'convert': lambda x: 'not_an_int'  # Always returns string
            }
        }

        metadata = {'failing_field': 'test'}
        result = sanitize_metadata(metadata, failing_schema)
        # Should fall back to type conversion, which will fail, then use default
        self.assertEqual(result['failing_field'], 0)


if __name__ == '__main__':
    unittest.main()
