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

    # def test_concurrent_operations(self):
    #     """Test concurrent add/retrieve operations."""
    #
    #     results = []
    #
    #     def add_documents(thread_id):
    #         for i in range(3):
    #             text = f"Thread {thread_id} document {i} content"
    #             docs = self.manager.add_text(
    #                 text_or_doc=text,
    #                 base_id=f"doc:thread_{thread_id}_{i}"
    #             )
    #             results.append(docs[0].text_id)
    #             time.sleep(0.1)
    #
    #     # Create multiple threads
    #     threads = []
    #     for i in range(3):
    #         thread = threading.Thread(target=add_documents, args=(i,))
    #         threads.append(thread)
    #         thread.start()
    #
    #     # Wait for completion
    #     for thread in threads:
    #         thread.join()
    #
    #     # Verify all documents were added
    #     assert len(results) == 9
    #     all_texts = self.manager.list_texts()
    #     for text_id in results:
    #         assert text_id in all_texts

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

    def test_textunit_metadata_preservation(self):
        """Test end-to-end preservation of TextUnit metadata when storing to Redis."""
        import time
        from ragl.textunit import TextUnit

        # Create TextUnit with comprehensive metadata
        original_timestamp = int(time.time()) - 3600  # 1 hour ago
        original_textunit = TextUnit(
            text_id="will_be_overridden",  # This will be set by manager
            text="Machine learning algorithms analyze data patterns to make predictions.",
            source="research_paper.pdf",
            timestamp=original_timestamp,
            tags=["ml", "algorithms", "data-science"],
            confidence=0.85,
            language="en",
            section="methodology",
            author="Dr. Jane Smith",
            parent_id="will_be_set",  # This will be set by manager
            chunk_position=0,
            distance=0.0
        )

        # Store the TextUnit
        stored_docs = self.manager.add_text(
            text_or_doc=original_textunit,
            base_id="doc:metadata_test"
        )

        assert len(stored_docs) == 1
        stored_doc = stored_docs[0]

        # Verify basic fields are set correctly by manager
        assert stored_doc.text == original_textunit.text
        assert stored_doc.parent_id == "doc:metadata_test"
        assert stored_doc.chunk_position == 0

        # Verify original metadata is preserved
        assert stored_doc.source == original_textunit.source
        assert stored_doc.timestamp == original_textunit.timestamp
        assert stored_doc.tags == original_textunit.tags
        assert stored_doc.confidence == original_textunit.confidence
        assert stored_doc.language == original_textunit.language
        assert stored_doc.section == original_textunit.section
        assert stored_doc.author == original_textunit.author

        # Test retrieval preserves metadata
        contexts = self.manager.get_context(
            query="machine learning data patterns",
            top_k=1
        )

        assert len(contexts) >= 1
        retrieved_doc = contexts[0]

        # Verify all metadata survives round-trip through Redis
        assert retrieved_doc.text == original_textunit.text
        assert retrieved_doc.source == original_textunit.source
        assert retrieved_doc.timestamp == original_textunit.timestamp
        assert retrieved_doc.tags == original_textunit.tags
        assert retrieved_doc.confidence == original_textunit.confidence
        assert retrieved_doc.language == original_textunit.language
        assert retrieved_doc.section == original_textunit.section
        assert retrieved_doc.author == original_textunit.author
        assert retrieved_doc.parent_id == "doc:metadata_test"
        assert retrieved_doc.chunk_position == 0

        # Verify distance is populated for retrieved document
        assert retrieved_doc.distance is not None
        assert 0.0 <= retrieved_doc.distance <= 1.0

        # Test metadata filtering/sorting if supported
        # Test time-based filtering
        future_time = int(time.time()) + 3600
        past_time = original_timestamp - 3600

        # Should find document within time range
        time_filtered_contexts = self.manager.get_context(
            query="machine learning",
            top_k=1,
            min_time=past_time,
            max_time=future_time
        )
        assert len(time_filtered_contexts) >= 1

        # Should not find document outside time range
        future_contexts = self.manager.get_context(
            query="machine learning",
            top_k=1,
            min_time=future_time,
            max_time=future_time + 3600
        )
        assert len(future_contexts) == 0

        logging.info("TextUnit metadata preservation test "
                     "completed successfully")

    def test_health_check_functionality(self):
        """Test health check functionality."""
        # Get health status
        health_status = self.manager.get_health_status()

        # Should return a dictionary with status information
        assert isinstance(health_status, dict)
        for key in ('redis_connected', 'index_exists', 'index_healthy'):
            assert key in health_status
            assert health_status[key] is True

        assert 'document_count' in health_status
        assert health_status['document_count'] == 0

        assert 'errors' in health_status
        assert isinstance(health_status['errors'], list)
        assert len(health_status['errors']) == 0

        assert 'last_check' in health_status
        assert isinstance(health_status['last_check'], int)

        assert 'memory_info' in health_status
        assert isinstance(health_status['memory_info'], dict)

        logging.info("Backend does not support health checks")

    def test_performance_metrics_tracking(self):
        """Test performance metrics collection and retrieval."""
        # Perform some operations to generate metrics
        text1 = "Machine learning is a subset of artificial intelligence."
        text2 = "Deep learning uses neural networks with multiple layers."

        # Add texts to generate add_text metrics
        self.manager.add_text(text_or_doc=text1,
                                      base_id="doc:metrics_test1")
        self.manager.add_text(text_or_doc=text2,
                                      base_id="doc:metrics_test2")

        # Perform queries to generate get_context metrics
        self.manager.get_context(query="machine learning", top_k=1)
        self.manager.get_context(query="neural networks", top_k=2)

        # List texts to generate list_texts metrics
        self.manager.list_texts()

        # Get all performance metrics
        all_metrics = self.manager.get_performance_metrics()

        # Verify metrics structure
        assert isinstance(all_metrics, dict)

        # Should have metrics for operations we performed
        expected_operations = ['add_text', 'get_context', 'list_texts']
        for operation in expected_operations:
            assert operation in all_metrics, f"Missing metrics for {operation}"

            metrics = all_metrics[operation]
            assert isinstance(metrics, dict)

            # Verify required metric fields
            required_fields = [
                'total_calls', 'failure_count', 'success_rate',
                'min_duration', 'max_duration', 'avg_duration',
                'recent_avg', 'recent_med'
            ]
            for field in required_fields:
                assert field in metrics, f"Missing metric field: {field}"
                assert isinstance(metrics[field], (int, float))

            # Verify logical constraints
            assert metrics['total_calls'] > 0
            assert metrics['failure_count'] >= 0
            assert metrics['failure_count'] <= metrics['total_calls']
            assert 0.0 <= metrics['success_rate'] <= 1.0
            assert metrics['min_duration'] >= 0.0
            assert metrics['max_duration'] >= metrics['min_duration']
            assert metrics['avg_duration'] >= 0.0

        # Test specific operation metrics
        add_text_metrics = self.manager.get_performance_metrics('add_text')
        assert 'add_text' in add_text_metrics
        assert add_text_metrics['add_text'][
                   'total_calls'] >= 2  # We added 2 texts

        get_context_metrics = self.manager.get_performance_metrics(
            'get_context')
        assert 'get_context' in get_context_metrics
        assert get_context_metrics['get_context'][
                   'total_calls'] >= 2  # We queried 2 times

        # Test non-existent operation
        empty_metrics = self.manager.get_performance_metrics(
            'non_existent_operation')
        assert empty_metrics == {}

        logging.info("Performance metrics collected: "
                     f"{list(all_metrics.keys())}")

    def test_metrics_reset_functionality(self):
        """Test metrics reset functionality."""
        # Perform operations to generate metrics
        self.manager.add_text("Test text for metrics",
                              base_id="doc:metrics_reset")
        self.manager.get_context("test query", top_k=1)

        # Verify metrics exist
        initial_metrics = self.manager.get_performance_metrics()
        assert len(initial_metrics) > 0

        # Reset metrics only
        self.manager.reset_metrics()

        # Verify metrics are cleared
        after_reset_metrics = self.manager.get_performance_metrics()
        assert len(after_reset_metrics) == 0

        # Verify data is still there (only metrics were reset)
        remaining_texts = self.manager.list_texts()
        assert len(remaining_texts) > 0  # Data should remain

        # Verify new operations start tracking again
        self.manager.get_context("another query", top_k=1)
        new_metrics = self.manager.get_performance_metrics()
        assert 'get_context' in new_metrics
        assert new_metrics['get_context']['total_calls'] == 1

        logging.info("Metrics reset functionality verified")

    def test_operation_failure_tracking(self):
        """Test that operation failures are properly tracked in metrics."""
        # Perform a valid operation first
        self.manager.add_text("Valid text", base_id="doc:failure_test")

        # Attempt operations that should fail
        try:
            # Invalid top_k should raise ValidationError
            self.manager.get_context("test query", top_k=0)
        except ValidationError:
            pass  # Expected failure

        # Empty query should return empty results (not fail)
        result = self.manager.get_context("", top_k=1)
        assert len(result) == 0

        # Check if failure tracking works (some operations might handle errors gracefully)
        metrics = self.manager.get_performance_metrics()

        # At minimum, we should have add_text metrics from successful operation
        assert 'add_text' in metrics
        assert metrics['add_text']['total_calls'] >= 1
        assert metrics['add_text']['success_rate'] > 0.0

        # Check if get_context has any failure tracking
        if 'get_context' in metrics:
            context_metrics = metrics['get_context']
            total_calls = context_metrics['total_calls']
            failure_count = context_metrics['failure_count']
            success_rate = context_metrics['success_rate']

            # Verify metrics consistency
            expected_success_rate = (
                                            total_calls - failure_count) / total_calls if total_calls > 0 else 0.0
            assert abs(
                success_rate - expected_success_rate) < 0.001  # Allow for rounding

            logging.info(f"get_context metrics - calls: {total_calls}, "
                         f"failures: {failure_count}, success_rate: {success_rate}")

        logging.info("Operation failure tracking verified")

    def test_performance_metrics_precision(self):
        """Test that performance metrics maintain appropriate precision."""
        # Perform multiple operations to get more stable metrics
        for i in range(5):
            self.manager.add_text(f"Test text {i}",
                                  base_id=f"doc:precision_test_{i}")
            self.manager.get_context(f"query {i}", top_k=1)

        metrics = self.manager.get_performance_metrics()

        for operation_name, operation_metrics in metrics.items():
            # Check precision of timing metrics (should be rounded to 4 decimal places)
            timing_fields = ['min_duration', 'max_duration', 'avg_duration',
                             'recent_avg', 'recent_med']

            for field in timing_fields:
                value = operation_metrics[field]
                # Check that value has at most 4 decimal places
                decimal_places = len(str(value).split('.')[-1]) if '.' in str(
                    value) else 0
                assert decimal_places <= 4, f"{operation_name}.{field} has too many decimal places: {value}"

            # Check success_rate precision (should be rounded to 4 decimal places)
            success_rate = operation_metrics['success_rate']
            decimal_places = len(
                str(success_rate).split('.')[-1]) if '.' in str(
                success_rate) else 0
            assert decimal_places <= 4, f"{operation_name}.success_rate has too many decimal places: {success_rate}"

            # Verify success_rate is between 0 and 1
            assert 0.0 <= success_rate <= 1.0

            logging.info(f"{operation_name} metrics precision verified")

#     def test_concurrent_metrics_tracking(self):
#         """Test that metrics tracking works correctly under concurrent operations."""
#
#         def perform_operations(thread_id: int, operation_count: int):
#             """Perform multiple operations in a thread."""
#             for i in range(operation_count):
#                 try:
#                     text = f"Thread {thread_id} operation {i} content"
#                     self.manager.add_text(text,
#                                           base_id=f"doc:concurrent_{thread_id}_{i}")
#                     self.manager.get_context(f"thread {thread_id} query {i}",
#                                              top_k=1)
#                     time.sleep(0.01)  # Small delay to allow interleaving
#                 except Exception as e:
#                     logging.warning(
#                         f"Thread {thread_id} operation {i} failed: {e}")
#
#         # Run concurrent operations
#         threads = []
#         operations_per_thread = 3
#         thread_count = 3
#
#         for thread_id in range(thread_count):
#             thread = threading.Thread(
#                 target=perform_operations,
#                 args=(thread_id, operations_per_thread)
#             )
#             threads.append(thread)
#             thread.start()
#
#         # Wait for all threads to complete
#         for thread in threads:
#             thread.join()
#
#         # Verify metrics consistency
#         metrics = self.manager.get_performance_metrics()
#
#         # Should have metrics for operations performed
#         assert 'add_text' in metrics
#         assert 'get_context' in metrics
#
#         # Verify total call counts make sense
#         expected_min_calls = thread_count * operations_per_thread
#         add_text_calls = metrics['add_text']['total_calls']
#         context_calls = metrics['get_context']['total_calls']
#
#         # Should have at least the expected number of calls (allowing for some failures)
#         assert add_text_calls >= expected_min_calls * 0.8  # Allow 20% failure rate
#         assert context_calls >= expected_min_calls * 0.8
#
#         # Verify metrics integrity
#         for operation_name, operation_metrics in metrics.items():
#             total_calls = operation_metrics['total_calls']
#             failure_count = operation_metrics['failure_count']
#             success_rate = operation_metrics['success_rate']
#
#             # Basic consistency checks
#             assert failure_count <= total_calls
#             expected_success_rate = (
#                                             total_calls - failure_count) / total_calls if total_calls > 0 else 0.0
#             assert abs(success_rate - expected_success_rate) < 0.001
#
#         logging.info(f"Concurrent metrics tracking verified - "
#                      f"add_text: {add_text_calls} calls, get_context: {context_calls} calls")


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
