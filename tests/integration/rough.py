# TODO:
#  - method ordering
#  - object names
#  - attribute names / ordering / docs
#  - str/repr
#  - More Logging / levels sanity check
#  - release package


from pprint import pprint
import logging

from ragl import RAGEngine
from ragl import HFEmbedder
from ragl import StandardRetriever
from ragl import TextUnit
from ragl.storage.redis import RedisStorage
from ragl.exceptions import ValidationError
# from ragl.tokenizer import TiktokenTokenizer

from ragl.config import EmbedderConfig, RAGConfig, RedisConfig


_LOG = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    redis_config = RedisConfig(host='localhost', port=6379, db=0)
    embedder_config = EmbedderConfig(cache_maxsize=20)
    rag_config = RAGConfig(chunk_size=100, overlap=20, index_name='rag_00')  # todo rename EngineConfig

    # embedder = HFEmbedder(model_name_or_path='all-MiniLM-L6-v2')
    embedder = HFEmbedder(config=embedder_config)
    # embedder = HFEmbedder(model_name_or_path='all-mpnet-base-v2')
    # embedder = HFEmbedder('sentence-transformers/all-MiniLM-L12-v2')

    # Setup for RedisStorage
    redis_storage = RedisStorage(  # todo combine dataclasses somehow so we only pass in one param? give it embedder and rag_config
        redis_config=redis_config,
        dimensions=embedder.dimensions,
        index_name=rag_config.index_name,
    )
    redis_retriever = StandardRetriever(embedder=embedder, storage=redis_storage)
    ragstore = RAGEngine(
        config=rag_config,
        retriever=redis_retriever,
        # tokenizer=TiktokenTokenizer(encoding_name='cl100k_base'),
    )

    def print_context(query: str, contexts: list[TextUnit]):
        print(f"\nContext for '{query}':")
        print("-" * 50)
        if not contexts:
            print("No results found.")
        for i, ctx in enumerate(contexts, 1):
            print(f"Result {i}:")
            print(f"  Text ID: {ctx.text_id}")
            print(f"  Text: {ctx.text[:50]}...")
            print(f"  Chunk Position: {ctx.chunk_position}")
            print(f"  Parent ID: {ctx.parent_id}")
            print(f"  Distance: {ctx.distance:.4f}")
            print(f"  Source: {ctx.source}")
            print(f"  Timestamp: {ctx.timestamp}")
            print(f"  Tags: {ctx.tags}")
            print()

    def print_texts(docs: list[TextUnit], message: str):
        print(f"\n{message}:")
        print("-" * 50)
        if not docs:
            print("No texts found.")
        for doc in docs:
            print(f"{doc.text_id}: {doc.text[:50]}... (Timestamp: {doc.timestamp})")

    # Test 1: Basic Functionality with String (Redis)
    _LOG.info("Starting Test 1: Basic Functionality with String (Redis)")
    ragstore.reset(reset_metrics=False)
    large_text = (
        "Artificial Intelligence (AI) is a broad field encompassing various techniques. "
        "Machine Learning (ML) is a subset of AI that focuses on training models with data. "
        "Deep Learning (DL), a further subset, uses neural networks with many layers."
    ) * 2
    docs = ragstore.add_text(text_or_doc=large_text, base_id="doc:large_redis")
    print_texts(docs, "Documents after adding large text (Redis)")
    contexts = ragstore.get_context(query="What is Deep Learning?", top_k=2)
    print_context("What is Deep Learning?", contexts)
    assert len(docs) > 1, "Should create multiple chunks"
    assert any("Deep Learning" in ctx.text for ctx in contexts), "Should retrieve relevant content"
    ragstore.delete_text(docs[0].text_id)
    remaining_texts = ragstore.list_texts()
    assert docs[0].text_id not in remaining_texts, "Deleted text should not be listed"

    # Test 4: No Split with Metadata (Redis)
    _LOG.info("Starting Test 4: No Split with Metadata (Redis)")
    ragstore.reset(reset_metrics=False)
    meta_doc = TextUnit(
        text_id="", text="AI is transformative", distance=0.0,
        tags=["tech", "AI"], confidence=0.9, timestamp=1698201600,
    )
    docs = ragstore.add_text(text_or_doc=meta_doc, base_id="doc:meta_redis", split=False)
    print_texts(docs, "Documents after adding with metadata (no split, Redis)")
    contexts = ragstore.get_context(query="AI transformative", top_k=1)
    print_context("AI transformative", contexts)
    assert len(docs) == 1, "Should store one chunk"
    assert contexts[0].tags == ["tech", "AI"], "Tags should be preserved"
    assert contexts[0].confidence == 0.9, "Confidence should be preserved"
    assert contexts[0].timestamp == 1698201600, "Timestamp should be preserved"

    # Test 6: Malformed Metadata (Redis)
    _LOG.info("Starting Test 6: Malformed Metadata (Redis)")
    ragstore.reset(reset_metrics=False)
    malformed_doc = TextUnit(
        text_id="", text="AI is cool", distance=0.0,
        timestamp="not_an_int", tags="not_a_list", confidence="invalid",
    )
    docs = ragstore.add_text(text_or_doc=malformed_doc, base_id="doc:malformed_redis", split=False)
    print_texts(docs, "Documents after adding malformed TextUnit (Redis)")
    contexts = ragstore.get_context(query="AI is cool", top_k=1)
    print_context("AI is cool", contexts)
    assert contexts[0].tags == ["not_a_list"], "Tags should be a list with the original string"
    assert isinstance(contexts[0].timestamp, int) and contexts[0].timestamp == 0, "Timestamp should default to 0"
    assert contexts[0].confidence == 0.0, "Confidence should default to 0.0"

    # Test 9: Timestamp Filtering (Redis)
    _LOG.info("Starting Test 9: Timestamp Filtering (Redis)")
    ragstore.reset(reset_metrics=False)
    timestamp_doc = TextUnit(text_id="", text="AI advancements", distance=0.0, timestamp=1700000000)
    docs = ragstore.add_text(text_or_doc=timestamp_doc, base_id="doc:time_redis")
    print_texts(docs, "Documents with fixed timestamp (Redis)")
    contexts = ragstore.get_context(query="AI advancements", top_k=1, min_time=1800000000)
    print_context("AI advancements (out of range)", contexts)
    assert not contexts, "No results should be returned due to timestamp filter"
    contexts = ragstore.get_context(query="AI advancements", top_k=1, min_time=1600000000)
    assert contexts, "Results should be returned within timestamp range"

    # Test 11: Empty Input and Reset (Redis)
    _LOG.info("Starting Test 11: Empty Input and Reset (Redis)")
    ragstore.reset(reset_metrics=False)
    try:
        ragstore.add_text("")
        assert False, "Empty string should raise ValueError"
    except ValidationError:
        pass
    docs = ragstore.add_text("Test reset", base_id="doc:reset_redis")
    ragstore.reset(reset_metrics=False)
    assert not ragstore.list_texts(), "Store should be empty after reset"

    # Test 14: Sort by Time (Redis)
    _LOG.info("Starting Test 14: Sort by Time (Redis)")
    ragstore.reset(reset_metrics=False)
    docs = [
        ragstore.add_text(
            TextUnit(text_id="", text=f"AI part {i}", distance=0.0,
                     timestamp=1698201600 + i * 1000),
            base_id=f"doc:time_sort_redis_{i}",
            split=False
        )[0]
        for i in range(3)
    ]
    print_texts(docs, "Documents with increasing timestamps (Redis)")
    contexts = ragstore.get_context(query="AI part", top_k=3,
                                    sort_by_time=True)
    print_context("AI part (sorted by time)", contexts)
    timestamps = [ctx.timestamp for ctx in contexts]
    assert timestamps == sorted(
        timestamps), "Results should be sorted by timestamp"
    assert len(contexts) == 3, "Should return all 3 results"

    # Test 18: Large top_k with Few Items (Redis)
    _LOG.info("Starting Test 18: Large top_k with Few Items (Redis)")
    ragstore.reset(reset_metrics=False)
    docs = ragstore.add_text("Small text", base_id="doc:small_redis")
    contexts = ragstore.get_context(query="Small", top_k=10)
    print_context("Small", contexts)
    assert len(contexts) == 1, "Should return only available items, not top_k"


    # Test 19: No Base ID (Redis)
    docs = ragstore.add_text("no base id")
    contexts = ragstore.get_context(query="no base id", top_k=1)
    print_context("no base id", contexts)



    #
    # # Test 21: Batch Store Multiple Texts (Redis)
    # _LOG.info("Starting Test 21: Batch Store Multiple Texts (Redis)")
    # ragstore.reset(reset_metrics=False)
    #
    # # Test the batch store functionality through retriever
    # batch_texts = [
    #     "neural networks Machine learning algorithms learn patterns from data",
    #     "neural networks are inspired by biological neurons",
    #     "neural networks Deep learning uses multiple layers for complex patterns",
    #     "neural networks are powerful machine learning models",
    #     "Deep learning uses neural networks with multiple layers",
    #     "Artificial neural networks mimic biological neurons",
    # ]
    #
    # batch_embeddings = [embedder.embed(text) for text in batch_texts]
    # batch_ids = ["txt:batch-1", "txt:batch-2", "txt:batch-3",
    #              "txt:batch-4", "txt:batch-5", "txt:batch-6"]
    # batch_metadata = [
    #     {"tags": ["ML", "algorithms"], "confidence": 0.8},
    #     {"tags": ["neural", "bio"], "confidence": 0.9},
    #     {"tags": ["deep", "layers"], "confidence": 0.95},
    #     {"tags": ["ML", "algorithms"], "confidence": 0.8},
    #     {"tags": ["neural", "bio"], "confidence": 0.9},
    #     {"tags": ["deep", "layers"], "confidence": 0.95},
    # ]
    #
    # # Store texts in batch using the storage layer directly
    # stored_ids = redis_storage.store_texts(
    #     texts=batch_texts,
    #     embeddings=batch_embeddings,
    #     text_ids=batch_ids,
    #     metadata_list=batch_metadata
    # )
    # print(f"Batch stored {len(stored_ids)} texts: {stored_ids}")
    #
    # # Verify all texts were stored
    # all_texts = ragstore.list_texts()
    # for text_id in batch_ids:
    #     assert text_id in all_texts, f"Batch stored text {text_id} should be listed"
    #
    # # Test retrieval works for batch stored texts
    # pprint(ragstore.list_texts())
    # contexts = ragstore.get_context("neural networks", top_k=3)
    # print_context("neural networks", contexts)
    # assert len(contexts) >= 1, "Should retrieve batch stored content"
    # assert "neural" in contexts[
    #     0].text.lower(), "Should find neural network content"
    #
    # # Test 22: Batch Delete Multiple Texts (Redis)
    # _LOG.info("Starting Test 22: Batch Delete Multiple Texts (Redis)")
    #
    # # Delete first two texts in batch
    # delete_ids = batch_ids[:2]
    # deleted_count = redis_storage.delete_texts(delete_ids)
    # print(f"Batch deleted {deleted_count}/{len(delete_ids)} texts")
    # assert deleted_count == 2, "Should delete exactly 2 texts"
    #
    # # Verify deletions worked
    # remaining_texts = ragstore.list_texts()
    # for text_id in delete_ids:
    #     assert text_id not in remaining_texts, f"Deleted text {text_id} should not be listed"
    # assert batch_ids[
    #            2] in remaining_texts, "Non-deleted text should still exist"
    #
    # # Test batch delete with non-existent IDs
    # fake_ids = ["txt:fake-1", "txt:fake-2"]
    # deleted_count = redis_storage.delete_texts(fake_ids)
    # print(f"Batch delete of non-existent texts deleted {deleted_count} items")
    # assert deleted_count == 0, "Should delete 0 non-existent texts"
    #
    #
    #
    # # Test 20: Get Metrics (Redis)




    embedding = embedder.embed('This is a test sentence')
    redis_storage.store_text('foo', embedding)
    _LOG.info("All tests completed successfully!")
    pprint(ragstore.get_performance_metrics())
    pprint(ragstore.get_health_status())
    pprint(embedder.get_memory_usage())

    # # Test cache hits
    # text = "This is a test sentence"
    # new_embedder = HFEmbedder(config=embedder_config)
    # new_embedder.embed(text)  # Cache miss
    # new_embedder.embed(text)  # Cache hit
    # new_embedder.embed(text)  # Cache hit
    # pprint(new_embedder.cache_info())
    # pprint(new_embedder.get_memory_info())
    #
    # # Create embedder once
    # text = "This is a test sentence"
    # new_embedder = HFEmbedder(config=embedder_config)
    #
    # # Clear any existing cache to start fresh
    # new_embedder.clear_cache()
    #
    # # First call - should be a cache miss
    # result1 = new_embedder.embed(text)
    # print("After first embed:")
    # pprint(new_embedder.cache_info())
    #
    # # Second call with same text - should be a cache hit
    # result2 = new_embedder.embed(text)
    # print("After second embed:")
    # pprint(new_embedder.cache_info())
    #
    # # Third call with same text - should be another cache hit
    # result3 = new_embedder.embed(text)
    # print("After third embed:")
    # pprint(new_embedder.cache_info())
    #
    # print("Final memory info:")
    # pprint(new_embedder.get_memory_info())
