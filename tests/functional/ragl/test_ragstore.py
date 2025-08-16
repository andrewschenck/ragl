import unittest
from unittest.mock import Mock

from ragl.ragstore import RAGStore
from ragl.protocols import EmbedderProtocol, VectorStoreProtocol
from ragl.textunit import TextUnit


class TestRAGStore(unittest.TestCase):
    """Test suite for RAGStore class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock embedder
        self.mock_embedder = Mock(spec=EmbedderProtocol)
        self.mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

        # Create mock storage
        self.mock_storage = Mock(spec=VectorStoreProtocol)
        self.mock_storage.store_text.return_value = "test_id_123"
        self.mock_storage.store_texts.return_value = ["test_id_123"]
        self.mock_storage.get_relevant.return_value = [
            {"id": "doc1", "text": "sample text", "score": 0.95},
            {"id": "doc2", "text": "another text", "score": 0.85}
        ]
        self.mock_storage.list_texts.return_value = ["doc1", "doc2", "doc3"]
        self.mock_storage.delete_text.return_value = True

        # Create RAGStore instance
        self.rag_store = RAGStore(
            embedder=self.mock_embedder,
            storage=self.mock_storage
        )

    def test_init_valid_protocols(self):
        """Test initialization with valid protocol implementations."""
        rag_store = RAGStore(
            embedder=self.mock_embedder,
            storage=self.mock_storage
        )

        self.assertEqual(rag_store.embedder, self.mock_embedder)
        self.assertEqual(rag_store.storage, self.mock_storage)

    def test_init_invalid_embedder_protocol(self):
        """Test initialization with invalid embedder protocol."""
        invalid_embedder = "not_an_embedder"

        with self.assertRaises(TypeError) as context:
            RAGStore(embedder=invalid_embedder, storage=self.mock_storage)

        self.assertEqual(str(context.exception),
                         "embedder must implement EmbedderProtocol")

    def test_init_invalid_storage_protocol(self):
        """Test initialization with invalid storage protocol."""
        invalid_storage = "not_a_storage"

        with self.assertRaises(TypeError) as context:
            RAGStore(embedder=self.mock_embedder, storage=invalid_storage)

        self.assertEqual(str(context.exception),
                         "store must implement VectorStoreProtocol")

    def test_clear(self):
        """Test clearing all data from store."""
        self.rag_store.clear()
        self.mock_storage.clear.assert_called_once()

    def test_delete_text_success(self):
        """Test successful text deletion."""
        self.mock_storage.delete_text.return_value = True

        result = self.rag_store.delete_text("test_id")

        self.assertTrue(result)
        self.mock_storage.delete_text.assert_called_once_with("test_id")

    def test_delete_text_not_found(self):
        """Test text deletion when text doesn't exist."""
        self.mock_storage.delete_text.return_value = False

        result = self.rag_store.delete_text("nonexistent_id")

        self.assertFalse(result)
        self.mock_storage.delete_text.assert_called_once_with("nonexistent_id")

    def test_get_relevant_basic(self):
        """Test basic relevant text retrieval."""
        query = "test query"
        top_k = 5

        result = self.rag_store.get_relevant(query, top_k)

        self.mock_embedder.embed.assert_called_once_with(query)
        self.mock_storage.get_relevant.assert_called_once_with(
            embedding=[0.1, 0.2, 0.3],
            top_k=top_k,
            min_time=None,
            max_time=None
        )
        self.assertEqual(len(result), 2)

    def test_get_relevant_with_time_filters(self):
        """Test relevant text retrieval with time filters."""
        query = "test query"
        top_k = 3
        min_time = 1000
        max_time = 2000

        result = self.rag_store.get_relevant(
            query, top_k, min_time=min_time, max_time=max_time
        )

        self.mock_embedder.embed.assert_called_once_with(query)
        self.mock_storage.get_relevant.assert_called_once_with(
            embedding=[0.1, 0.2, 0.3],
            top_k=top_k,
            min_time=min_time,
            max_time=max_time
        )
        self.assertEqual(len(result), 2)

    def test_get_relevant_with_only_min_time(self):
        """Test relevant text retrieval with only min_time filter."""
        query = "test query"
        top_k = 3
        min_time = 1000

        result = self.rag_store.get_relevant(query, top_k, min_time=min_time)

        self.mock_storage.get_relevant.assert_called_once_with(
            embedding=[0.1, 0.2, 0.3],
            top_k=top_k,
            min_time=min_time,
            max_time=None
        )

    def test_get_relevant_with_only_max_time(self):
        """Test relevant text retrieval with only max_time filter."""
        query = "test query"
        top_k = 3
        max_time = 2000

        result = self.rag_store.get_relevant(query, top_k, max_time=max_time)

        self.mock_storage.get_relevant.assert_called_once_with(
            embedding=[0.1, 0.2, 0.3],
            top_k=top_k,
            min_time=None,
            max_time=max_time
        )

    def test_list_texts(self):
        """Test listing all text IDs."""
        result = self.rag_store.list_texts()

        self.mock_storage.list_texts.assert_called_once()
        self.assertEqual(result, ["doc1", "doc2", "doc3"])

    def test_store_text_basic(self):
        """Test basic text storage without ID or metadata."""
        text = "Hello, world!"
        original_text_id = "text-id-1"
        text_unit = TextUnit(
            text=text,
            text_id=original_text_id,
            distance=0.0,
        )

        result = self.rag_store.store_text(text_unit)

        self.assertIsInstance(result, TextUnit)
        self.assertEqual(result.text, text)
        # The text_id should be updated to what storage returned
        self.assertEqual(result.text_id, "text-id-1")
        self.mock_storage.store_text.assert_called_once()

        # Verify the storage was called with the correct arguments
        call_args = self.mock_storage.store_text.call_args
        # store_text now expects (text_unit, embedding) as separate arguments
        text_unit_arg = call_args.kwargs['text_unit']
        embedding_arg = call_args.kwargs['embedding']

        self.assertEqual(text_unit_arg.text, text)
        self.assertEqual(text_unit_arg.text_id, original_text_id)
        self.assertEqual(embedding_arg, [0.1, 0.2, 0.3])  # Mock embedder output

    def test_store_text_with_id(self):
        """Test text storage with provided ID."""
        text = "Hello, world!"
        text_id = "custom_id"
        text_unit = TextUnit(text=text, text_id=text_id, distance=0.0)

        result = self.rag_store.store_text(text_unit)

        self.assertIsInstance(result, TextUnit)
        self.assertEqual(result.text, text)
        self.mock_storage.store_text.assert_called_once()  # Changed from store_texts to store_text

        # Verify the call arguments
        call_args = self.mock_storage.store_text.call_args
        stored_text_unit = call_args[1]['text_unit']  # keyword argument
        embedding_arg = call_args[1]['embedding']

        self.assertEqual(stored_text_unit.text, text)
        self.assertEqual(stored_text_unit.text_id, text_id)
        self.assertEqual(embedding_arg,
                         [0.1, 0.2, 0.3])  # Mock embedder output

    def test_delete_texts_success(self):
        """Test successful deletion of multiple texts."""
        text_ids = ["test_id_1", "test_id_2", "test_id_3"]
        self.mock_storage.delete_texts.return_value = 3

        result = self.rag_store.delete_texts(text_ids)

        self.assertEqual(result, 3)
        self.mock_storage.delete_texts.assert_called_once_with(text_ids)

    def test_delete_texts_partial_success(self):
        """Test deletion when only some texts exist."""
        text_ids = ["test_id_1", "nonexistent_id", "test_id_3"]
        self.mock_storage.delete_texts.return_value = 2

        result = self.rag_store.delete_texts(text_ids)

        self.assertEqual(result, 2)
        self.mock_storage.delete_texts.assert_called_once_with(text_ids)

    def test_delete_texts_empty_list(self):
        """Test deletion with empty list."""
        text_ids = []
        self.mock_storage.delete_texts.return_value = 0

        result = self.rag_store.delete_texts(text_ids)

        self.assertEqual(result, 0)
        self.mock_storage.delete_texts.assert_called_once_with(text_ids)

    def test_delete_texts_none_found(self):
        """Test deletion when no texts exist."""
        text_ids = ["nonexistent_1", "nonexistent_2"]
        self.mock_storage.delete_texts.return_value = 0

        result = self.rag_store.delete_texts(text_ids)

        self.assertEqual(result, 0)
        self.mock_storage.delete_texts.assert_called_once_with(text_ids)

    def test_store_texts_multiple(self):
        """Test storing multiple TextUnits."""
        text_units = [
            TextUnit(text="First text", text_id="id_1", distance=0.0),
            TextUnit(text="Second text", text_id="id_2", distance=0.0),
            TextUnit(text="Third text", text_id="id_3", distance=0.0)
        ]
        self.mock_storage.store_texts.return_value = ["id_1", "id_2", "id_3"]

        result = self.rag_store.store_texts(text_units)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].text, "First text")
        self.assertEqual(result[1].text, "Second text")
        self.assertEqual(result[2].text, "Third text")

        # Verify embedder was called for each text
        self.assertEqual(self.mock_embedder.embed.call_count, 3)
        self.mock_embedder.embed.assert_any_call("First text")
        self.mock_embedder.embed.assert_any_call("Second text")
        self.mock_embedder.embed.assert_any_call("Third text")

        # Verify storage was called with correct structure
        self.mock_storage.store_texts.assert_called_once()
        call_args = self.mock_storage.store_texts.call_args[0][0]
        self.assertEqual(len(call_args), 3)

        # Check that each tuple contains TextUnit and embedding
        for i, (text_unit, embedding) in enumerate(call_args):
            self.assertIsInstance(text_unit, TextUnit)
            self.assertEqual(embedding, [0.1, 0.2, 0.3])

    def test_store_texts_empty_list(self):
        """Test storing empty list of TextUnits."""
        text_units = []

        result = self.rag_store.store_texts(text_units)

        self.assertEqual(result, [])
        self.mock_embedder.embed.assert_not_called()
        self.mock_storage.store_texts.assert_called_once_with([])

    def test_store_texts_single_text(self):
        """Test storing single TextUnit via store_texts method."""
        text_units = [
            TextUnit(text="Single text", text_id="single_id", distance=0.0)
        ]
        self.mock_storage.store_texts.return_value = ["single_id"]

        result = self.rag_store.store_texts(text_units)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Single text")
        self.assertEqual(result[0].text_id, "single_id")

        self.mock_embedder.embed.assert_called_once_with("Single text")
        self.mock_storage.store_texts.assert_called_once()

    def test_store_texts_preserves_metadata(self):
        """Test that store_texts preserves metadata from TextUnits."""
        text_units = [
            TextUnit(
                text="Text with metadata",
                text_id="meta_id",
                distance=0.0,
                timestamp=1234567890,
                tags=["tag1", "tag2"],
                source="test_source"
            )
        ]
        self.mock_storage.store_texts.return_value = ["meta_id"]

        result = self.rag_store.store_texts(text_units)

        self.assertEqual(len(result), 1)
        stored_unit = result[0]
        self.assertEqual(stored_unit.timestamp, 1234567890)
        self.assertEqual(stored_unit.tags, ["tag1", "tag2"])
        self.assertEqual(stored_unit.source, "test_source")

    def test_store_texts_without_text_ids(self):
        """Test storing TextUnits without predefined text IDs."""
        text_units = [
            TextUnit(text="Text without ID 1", text_id=None, distance=0.0),
            TextUnit(text="Text without ID 2", text_id=None, distance=0.0)
        ]
        self.mock_storage.store_texts.return_value = ["generated_1",
                                                      "generated_2"]

        result = self.rag_store.store_texts(text_units)

        self.assertEqual(len(result), 2)
        # The original TextUnits should be returned unchanged
        self.assertEqual(result[0].text, "Text without ID 1")
        self.assertEqual(result[1].text, "Text without ID 2")

    def test_repr(self):
        """Test __repr__ method."""
        result = repr(self.rag_store)
        expected = f"RAGStore(embedder={self.mock_embedder!r}, storage={self.mock_storage!r})"
        self.assertEqual(result, expected)

    def test_str(self):
        """Test __str__ method."""
        result = str(self.rag_store)
        embedder_name = type(self.mock_embedder).__name__
        storage_name = type(self.mock_storage).__name__
        expected = f"RAGStore with {embedder_name} embedder and {storage_name} storage"
        self.assertEqual(result, expected)


class TestRAGStoreLogging(unittest.TestCase):
    """Test logging functionality in RAGStore."""

    def setUp(self):
        """Set up test fixtures with logging capture."""
        self.mock_embedder = Mock(spec=EmbedderProtocol)
        self.mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

        self.mock_storage = Mock(spec=VectorStoreProtocol)
        self.mock_storage.store_text.return_value = "test_id"
        self.mock_storage.delete_text.return_value = True
        self.mock_storage.list_texts.return_value = ["doc1"]
        self.mock_storage.get_relevant.return_value = []

        self.rag_store = RAGStore(
            embedder=self.mock_embedder,
            storage=self.mock_storage
        )

    def test_logging_critical_invalid_embedder(self):
        """Test critical logging for invalid embedder."""
        with self.assertLogs('ragl.ragstore', level='CRITICAL') as log:
            with self.assertRaises(TypeError):
                RAGStore(embedder="invalid", storage=self.mock_storage)

        self.assertIn('embedder must implement EmbedderProtocol',
                      log.output[0])

    def test_logging_critical_invalid_storage(self):
        """Test critical logging for invalid storage."""
        with self.assertLogs('ragl.ragstore', level='CRITICAL') as log:
            with self.assertRaises(TypeError):
                RAGStore(embedder=self.mock_embedder, storage="invalid")

        self.assertIn('store must implement VectorStoreProtocol',
                      log.output[0])

    def test_logging_debug_delete_text(self):
        """Test debug logging for delete_text."""
        with self.assertLogs('ragl.ragstore', level='DEBUG') as log:
            self.rag_store.delete_text("test_id")

        self.assertIn('Deleting text test_id', log.output[0])

    def test_logging_debug_get_relevant(self):
        """Test debug logging for get_relevant."""
        with self.assertLogs('ragl.ragstore', level='DEBUG') as log:
            self.rag_store.get_relevant("test query", 5)

        self.assertIn('Retrieving relevant texts for query test query',
                      log.output[0])

    def test_logging_debug_list_texts(self):
        """Test debug logging for list_texts."""
        with self.assertLogs('ragl.ragstore', level='DEBUG') as log:
            self.rag_store.list_texts()

        self.assertIn('Listing all texts in store', log.output[0])

    def test_logging_debug_store_text(self):
        """Test debug logging for store_text."""
        text_unit = TextUnit(text="test text", text_id="test_id", distance=0.0)

        with self.assertLogs('ragl.ragstore', level='DEBUG') as log:
            self.rag_store.store_text(text_unit)

        self.assertIn('Storing TextUnit', log.output[0])

    def test_logging_debug_delete_texts(self):
        """Test debug logging for delete_texts."""
        text_ids = ["id1", "id2", "id3"]

        with self.assertLogs('ragl.ragstore', level='DEBUG') as log:
            self.rag_store.delete_texts(text_ids)

        self.assertIn('Deleting 3 texts', log.output[0])

    def test_logging_debug_store_texts(self):
        """Test debug logging for store_texts."""
        text_units = [
            TextUnit(text="test text 1", text_id="test_id_1", distance=0.0),
            TextUnit(text="test text 2", text_id="test_id_2", distance=0.0)
        ]

        with self.assertLogs('ragl.ragstore', level='DEBUG') as log:
            self.rag_store.store_texts(text_units)

        self.assertIn('Storing 2 TextUnits', log.output[0])


if __name__ == '__main__':
    unittest.main()
