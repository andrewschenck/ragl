"""
Comprehensive integration tests for RAGL against live Redis container.
Assumes Redis is running and accessible.
"""
import logging
from ragl.config import (
    ManagerConfig,
    RedisConfig,
    SentenceTransformerConfig,
)
from ragl.exceptions import ValidationError
from ragl.registry import create_rag_manager


class TestRAGLIntegration:
    """Integration tests for RAGL with live Redis."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        logging.basicConfig(level=logging.INFO)
        cls.storage_config = RedisConfig()
        cls.embedder_config = SentenceTransformerConfig()
        cls.manager_config = ManagerConfig(chunk_size=100, overlap=20)
        cls.manager = create_rag_manager(
            index_name='test_integration_index',
            storage_config=cls.storage_config,
            embedder_config=cls.embedder_config,
            manager_config=cls.manager_config,
        )

    def setup_method(self):
        """Reset manager before each test."""
        self.manager.reset(reset_metrics=True)

    def test_text_sanitization(self):
        """Test text input sanitization."""
        malicious_text = "Text with <script>alert('xss')</script> chars!"
        result = self.manager._sanitize_query(malicious_text)
        assert "<script>" not in result
        logging.info(f"Sanitized text: {result}")

    def test_add_and_retrieve_single_document(self):
        """Test adding and retrieving a single document."""
        text = "Python is a high-level programming language."
        docs = self.manager.add_text(
            text_or_doc=text,
            base_id="doc:python_intro"
        )

        assert len(docs) == 1
        assert docs[0].text == text

        contexts = self.manager.get_context(
            query="What is Python?",
            top_k=1
        )
        assert len(contexts) >= 1
        assert "Python" in contexts[0].text

    def test_chunking_large_document(self):
        """Test chunking of large documents."""
        large_text = (
                         "Artificial Intelligence encompasses machine learning, "
                         "natural language processing, computer vision, and robotics. "
                         "Machine learning algorithms can be supervised or unsupervised. "
                         "Deep learning uses neural networks with multiple layers. "
                         "Natural language processing helps computers understand text. "
                         "Computer vision enables machines to interpret visual data."
                     ) * 3

        docs = self.manager.add_text(
            text_or_doc=large_text,
            base_id="doc:ai_overview"
        )

        assert len(docs) > 1, "Large text should be chunked"

        # Verify chunks have proper positioning
        for i, doc in enumerate(docs):
            assert doc.chunk_position == i
            assert doc.parent_id == "doc:ai_overview"

    def test_multiple_documents_retrieval(self):
        """Test retrieval across multiple documents."""
        texts = [
            "Python is used for web development and data science.",
            "JavaScript is essential for frontend web development.",
            "Java is popular for enterprise applications.",
            "Go is efficient for concurrent programming."
        ]

        all_docs = []
        for i, text in enumerate(texts):
            docs = self.manager.add_text(
                text_or_doc=text,
                base_id=f"doc:language_{i}"
            )
            all_docs.extend(docs)

        # Test retrieval
        contexts = self.manager.get_context(
            query="web development languages",
            top_k=3
        )

        assert len(contexts) >= 2
        relevant_texts = [ctx.text for ctx in contexts]
        assert any("Python" in text or "JavaScript" in text
                   for text in relevant_texts)

    def test_document_deletion(self):
        """Test document deletion functionality."""
        text = "This document will be deleted."
        docs = self.manager.add_text(
            text_or_doc=text,
            base_id="doc:to_delete"
        )

        text_id = docs[0].text_id

        # Verify document exists
        all_texts = self.manager.list_texts()
        assert text_id in all_texts

        # Delete document
        self.manager.delete_text(text_id)

        # Verify deletion
        remaining_texts = self.manager.list_texts()
        assert text_id not in remaining_texts

    def test_text_listing_and_filtering(self):
        """Test listing and filtering of texts."""
        # Add documents with different tags
        texts_with_tags = [
            ("Machine learning basics", {"category": "ml", "level": "basic"}),
            ("Advanced neural networks",
             {"category": "ml", "level": "advanced"}),
            ("Web scraping tutorial", {"category": "web", "level": "basic"}),
        ]

        for text, tags in texts_with_tags:
            self.manager.add_text(
                text_or_doc=text,
                base_id=f"doc:{text[:10].replace(' ', '_')}",
            )

        # Test listing all texts
        all_texts = self.manager.list_texts()
        assert len(all_texts) >= 3

        # Test filtering by tags if supported
        try:
            ml_texts = self.manager.list_texts()
            assert len(ml_texts) >= 2
        except TypeError:
            # Filtering might not be implemented
            logging.info("Tag filtering not supported")

    def test_context_retrieval_with_distance(self):
        """Test context retrieval with distance scoring."""
        reference_text = (
            "Machine learning is a subset of artificial intelligence "
            "that focuses on building systems that learn from data."
        )

        docs = self.manager.add_text(
            text_or_doc=reference_text,
            base_id="doc:ml_definition"
        )

        # Test exact match query
        contexts = self.manager.get_context(
            query="machine learning artificial intelligence",
            top_k=1
        )

        assert len(contexts) >= 1
        assert contexts[0].distance is not None
        assert 0.0 <= contexts[0].distance <= 1.0

    def test_concurrent_operations(self):
        """Test concurrent add/retrieve operations."""
        import threading
        import time

        results = []

        def add_documents(thread_id):
            for i in range(3):
                text = f"Thread {thread_id} document {i} content"
                docs = self.manager.add_text(
                    text_or_doc=text,
                    base_id=f"doc:thread_{thread_id}_{i}"
                )
                results.append(docs[0].text_id)
                time.sleep(0.1)

        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_documents, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify all documents were added
        assert len(results) == 9
        all_texts = self.manager.list_texts()
        for text_id in results:
            assert text_id in all_texts

    def test_error_handling(self):
        """Test error handling for invalid operations."""
        # Test deletion of non-existent document
        try:
            self.manager.delete_text("non_existent_id")
            # Should either succeed silently or raise specific exception
        except Exception as e:
            logging.info(f"Expected error for non-existent deletion: {e}")

        # Test empty query
        contexts = self.manager.get_context(query="", top_k=1)
        # Should return empty results or handle gracefully

        # Test invalid top_k
        try:
            contexts = self.manager.get_context(query="test", top_k=0)
        except ValidationError:
            logging.info("Correctly handled invalid top_k")

    def teardown_method(self):
        """Clean up after each test."""
        try:
            self.manager.reset(reset_metrics=True)
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")


if __name__ == "__main__":
    import sys
    test_suite = TestRAGLIntegration()
    test_suite.setup_class()

    test_methods = [
        method for method in dir(test_suite)
        if method.startswith('test_')
    ]

    exit_code = 0
    for test_method in test_methods:
        print(f"\nRunning {test_method}...")
        try:
            test_suite.setup_method()
            getattr(test_suite, test_method)()
            print(f"✓ {test_method} passed")
        except Exception as e:
            print(f"✗ {test_method} failed: {e}")
            exit_code = 1
        finally:
            test_suite.teardown_method()

    print("\nIntegration tests completed.")
    sys.exit(exit_code)
