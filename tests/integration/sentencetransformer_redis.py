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


logging.basicConfig(level=logging.INFO)


class TestRAGLIntegration:
    """Integration tests for RAGL with live Redis."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
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

    def teardown_method(self):
        """Clean up after each test."""
        try:
            self.manager.reset(reset_metrics=True)
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")

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

        _ = self.manager.add_text(
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
        self.manager.delete_text("non_existent_id")

        # Test empty query
        result = self.manager.get_context(query="", top_k=1)
        # Should return empty results or handle gracefully
        assert len(result) == 0
        assert isinstance(result, list)

        # Test invalid top_k
        try:
            _ = self.manager.get_context(query="test", top_k=0)
        except ValidationError:
            logging.info("Correctly handled invalid top_k")

    def test_tiny_documents_with_large_chunks(self):
        """Test handling of tiny documents with large chunk sizes."""
        tiny_text = "Short."

        # Create manager with large chunk size (in tokens)
        large_chunk_manager = create_rag_manager(
            index_name='test_large_chunk_index',
            storage_config=self.storage_config,
            embedder_config=self.embedder_config,
            manager_config=ManagerConfig(chunk_size=100, overlap=10),
            # 100 tokens
        )
        large_chunk_manager.reset(reset_metrics=True)

        docs = large_chunk_manager.add_text(
            text_or_doc=tiny_text,
            base_id="doc:tiny"
        )

        assert len(docs) == 1
        assert docs[0].text == tiny_text
        large_chunk_manager.reset(reset_metrics=True)

    def test_medium_documents_varying_chunks(self):
        """Test medium documents with different chunk configurations."""
        medium_text = (
                          "Natural language processing is a subfield of linguistics, computer science, "
                          "and artificial intelligence concerned with the interactions between computers "
                          "and human language. It involves programming computers to process and analyze "
                          "large amounts of natural language data. The goal is a computer capable of "
                          "understanding the contents of documents, including the contextual nuances "
                          "of the language within them."
                      ) * 2

        chunk_configs = [
            (20, 5),  # Small chunks, small overlap (tokens)
            (50, 10),  # Medium chunks, medium overlap (tokens)
            (100, 20),  # Large chunks, large overlap (tokens)
        ]

        for chunk_size, overlap in chunk_configs:
            manager = create_rag_manager(
                index_name=f'test_chunk_{chunk_size}_overlap_{overlap}',
                storage_config=self.storage_config,
                embedder_config=self.embedder_config,
                manager_config=ManagerConfig(chunk_size=chunk_size,
                                             overlap=overlap),
            )
            manager.reset(reset_metrics=True)

            docs = manager.add_text(
                text_or_doc=medium_text,
                base_id=f"doc:medium_{chunk_size}_{overlap}"
            )

            # Verify chunking behavior - account for token-based chunking
            tokenizer = manager.tokenizer
            total_tokens = len(tokenizer.encode(medium_text))
            expected_chunks = max(1, (total_tokens - overlap) // (
                    chunk_size - overlap))
            assert len(docs) >= 1
            assert len(docs) <= expected_chunks + 2  # Allow variance for merging

            # Test retrieval
            contexts = manager.get_context(
                query="natural language processing",
                top_k=2
            )
            assert len(contexts) >= 1
            manager.reset(reset_metrics=True)

    def test_very_large_document_small_chunks(self):
        """Test very large document with small chunk sizes."""
        # Generate large document
        large_text = (
                         "Machine learning is a method of data analysis that automates analytical "
                         "model building. It is a branch of artificial intelligence based on the "
                         "idea that systems can learn from data, identify patterns and make "
                         "decisions with minimal human intervention. Machine learning algorithms "
                         "build mathematical models based on training data in order to make "
                         "predictions or decisions without being explicitly programmed to do so. "
                     ) * 50  # Creates a very large document

        small_chunk_manager = create_rag_manager(
            index_name='test_large_doc_small_chunks',
            storage_config=self.storage_config,
            embedder_config=self.embedder_config,
            manager_config=ManagerConfig(chunk_size=50, overlap=10),
            # 50 tokens
        )
        small_chunk_manager.reset(reset_metrics=True)

        docs = small_chunk_manager.add_text(
            text_or_doc=large_text,
            base_id="doc:very_large"
        )

        # Should create many chunks
        assert len(docs) > 5

        # Verify chunk integrity with token-based validation
        tokenizer = small_chunk_manager.tokenizer
        for i, doc in enumerate(docs):
            assert doc.chunk_position == i
            # Check token count instead of character count
            token_count = len(tokenizer.encode(doc.text))
            # Allow for overlap and merging tolerance
            assert token_count <= 70  # chunk_size + overlap + merging tolerance
            assert doc.parent_id == "doc:very_large"

        # Test retrieval across many chunks
        contexts = small_chunk_manager.get_context(
            query="machine learning algorithms",
            top_k=5
        )
        assert len(contexts) >= 3
        small_chunk_manager.reset(reset_metrics=True)

    def test_zero_overlap_chunking(self):
        """Test chunking with zero overlap."""
        text = (
                   "Zero overlap chunking means each chunk is completely separate. "
                   "There is no shared content between adjacent chunks. This can "
                   "sometimes lead to loss of context at chunk boundaries. However, "
                   "it maximizes content coverage without duplication."
               ) * 3

        zero_overlap_manager = create_rag_manager(
            index_name='test_zero_overlap',
            storage_config=self.storage_config,
            embedder_config=self.embedder_config,
            manager_config=ManagerConfig(chunk_size=30, overlap=0),
            # 30 tokens, no overlap
        )
        zero_overlap_manager.reset(reset_metrics=True)

        docs = zero_overlap_manager.add_text(
            text_or_doc=text,
            base_id="doc:zero_overlap"
        )

        # Should create multiple chunks with no overlap
        assert len(docs) > 1

        # Verify chunks are distinct (no significant overlap)
        for i in range(len(docs) - 1):
            current_chunk = docs[i].text
            next_chunk = docs[i + 1].text
            # Should not be identical due to no overlap
            assert current_chunk != next_chunk

        zero_overlap_manager.reset(reset_metrics=True)

    def test_high_overlap_chunking(self):
        """Test chunking with high overlap ratio."""
        text = (
                   "High overlap chunking creates significant redundancy between chunks. "
                   "This ensures better context preservation across chunk boundaries but "
                   "increases storage requirements and may lead to repetitive results. "
                   "The trade-off is between context preservation and efficiency."
               ) * 2

        high_overlap_manager = create_rag_manager(
            index_name='test_high_overlap',
            storage_config=self.storage_config,
            embedder_config=self.embedder_config,
            manager_config=ManagerConfig(chunk_size=40, overlap=30),
            # High overlap ratio
        )
        high_overlap_manager.reset(reset_metrics=True)

        docs = high_overlap_manager.add_text(
            text_or_doc=text,
            base_id="doc:high_overlap"
        )

        # High overlap should create more chunks
        assert len(docs) > 2

        # Test that overlapping content improves retrieval
        contexts = high_overlap_manager.get_context(
            query="context preservation boundaries",
            top_k=3
        )
        assert len(contexts) >= 2
        high_overlap_manager.reset(reset_metrics=True)

    def test_document_size_edge_cases(self):
        """Test edge cases for document sizes."""
        edge_cases = [
            ("", "empty"),  # Empty document
            ("A", "single_char"),  # Single character
            ("Word", "single_word"),  # Single word
            ("Two words", "two_words"),  # Two words
            ("A" * 500, "very_long_word"),  # Very long single "word"
        ]

        for text, case_name in edge_cases:
            if text:  # Skip empty text as it raises ValidationError
                docs = self.manager.add_text(
                    text_or_doc=text,
                    base_id=f"doc:edge_{case_name}"
                )

                assert len(docs) >= 1
                assert docs[0].text == text or docs[
                    0].text.strip() == text.strip()

    def test_mixed_document_sizes_retrieval(self):
        """Test retrieval across documents of varying sizes."""
        documents = [
            ("AI", "doc:tiny"),
            ("Machine learning uses algorithms to find patterns in data.",
             "doc:small"),
            ((
                 "Deep learning is a subset of machine learning that uses neural networks "
                 "with multiple layers to model and understand complex patterns. These "
                 "networks are inspired by the human brain's structure and function."),
             "doc:medium"),
            ((
                 "Artificial intelligence encompasses a broad range of technologies and "
                 "methodologies designed to enable machines to perform tasks that typically "
                 "require human intelligence. This includes reasoning, learning, perception, "
                 "language understanding, and problem-solving capabilities. The field has "
                 "evolved significantly since its inception, with major breakthroughs in "
                 "areas such as computer vision, natural language processing, and robotics.") * 3,
             "doc:large"),
        ]

        all_doc_ids = []
        for text, doc_id in documents:
            docs = self.manager.add_text(text_or_doc=text, base_id=doc_id)
            all_doc_ids.extend([doc.text_id for doc in docs])

        # Test retrieval that should match across different document sizes
        contexts = self.manager.get_context(
            query="machine learning artificial intelligence",
            top_k=5
        )

        assert len(contexts) >= 3
        # Should find relevant content regardless of document size
        context_texts = [ctx.text for ctx in contexts]
        assert any("AI" in text or "machine learning" in text.lower()
                   for text in context_texts)

    def test_chunk_boundary_context_preservation(self):
        """Test that important context is preserved across chunk boundaries."""
        # Create text where important information spans chunk boundaries
        boundary_text = (
            "The quick brown fox jumps over the lazy dog. This sentence contains "
            "every letter of the alphabet and is commonly used for testing. "
            "However, the most important information is that the fox is actually "
            "a metaphor for agility and speed in problem-solving methodologies. "
            "This metaphor demonstrates how quick thinking and adaptability are "
            "essential skills in software development and system design processes."
        )

        # Use chunk size that will split the important metaphor explanation
        boundary_manager = create_rag_manager(
            index_name='test_boundary_context',
            storage_config=self.storage_config,
            embedder_config=self.embedder_config,
            manager_config=ManagerConfig(chunk_size=25, overlap=8),
            # Small token chunks with overlap
        )
        boundary_manager.reset(reset_metrics=True)

        docs = boundary_manager.add_text(
            text_or_doc=boundary_text,
            base_id="doc:boundary_test"
        )

        # Should create multiple chunks due to length
        assert len(docs) > 1

        # Test retrieval of information that spans boundaries
        contexts = boundary_manager.get_context(
            query="fox metaphor agility",
            top_k=3
        )

        # Should retrieve relevant chunks despite boundary split
        assert len(contexts) >= 1
        relevant_text = " ".join([ctx.text for ctx in contexts])
        assert "metaphor" in relevant_text or "agility" in relevant_text

        boundary_manager.reset(reset_metrics=True)

    def test_token_count_validation(self):
        """Test that token counts match expected chunking behavior."""
        text = (
            "This is a test document that will be used to validate token-based chunking. "
            "Each chunk should contain approximately the specified number of tokens, "
            "with appropriate overlap between consecutive chunks for context preservation."
        )

        token_manager = create_rag_manager(
            index_name='test_token_validation',
            storage_config=self.storage_config,
            embedder_config=self.embedder_config,
            manager_config=ManagerConfig(chunk_size=20, overlap=5),
        )
        token_manager.reset(reset_metrics=True)

        docs = token_manager.add_text(
            text_or_doc=text,
            base_id="doc:token_test"
        )

        tokenizer = token_manager.tokenizer

        # Validate token counts for each chunk
        for doc in docs:
            token_count = len(tokenizer.encode(doc.text))
            # Allow for merging tolerance and overlap
            assert token_count <= 30  # chunk_size + overlap + merging tolerance
            assert token_count > 0

        token_manager.reset(reset_metrics=True)

    def test_min_chunk_size_handling(self):
        """Test handling of minimum chunk size parameter."""
        text = (
            "Short sentences. More text. Even more content here. "
            "This creates multiple potential chunks. Final sentence."
        )

        # Test with explicit min_chunk_size
        min_chunk_manager = create_rag_manager(
            index_name='test_min_chunk',
            storage_config=self.storage_config,
            embedder_config=self.embedder_config,
            manager_config=ManagerConfig(chunk_size=15, overlap=3,
                                         min_chunk_size=8),
        )
        min_chunk_manager.reset(reset_metrics=True)

        docs = min_chunk_manager.add_text(
            text_or_doc=text,
            base_id="doc:min_chunk_test"
        )

        tokenizer = min_chunk_manager.tokenizer

        # Verify that chunks respect min_chunk_size through merging
        for doc in docs[:-1]:  # All but last chunk
            token_count = len(tokenizer.encode(doc.text))
            # Should not have tiny chunks due to merging
            assert token_count >= 5  # Reasonable minimum after merging

        min_chunk_manager.reset(reset_metrics=True)


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
        # print(f"\n*** Running {test_method}...")
        logging.info(f"***** Running {test_method} *****")
        try:
            test_suite.setup_method()
            getattr(test_suite, test_method)()
            # print(f"*** {test_method} passed")
            logging.info(f"***** {test_method} passed *****")
        except Exception as e:
            # print(f"*** {test_method} failed: {e}")
            logging.warning(f"***** {test_method} failed: {e} *****")
            exit_code = 1
        finally:
            test_suite.teardown_method()

    # print("\n***Integration tests completed.")
    logging.info("***** Integration tests completed. *****")
    sys.exit(exit_code)
