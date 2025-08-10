from ragl.config import (
    ManagerConfig,
    RedisConfig,
    SentenceTransformerConfig,
)
from ragl.textunit import TextUnit
from ragl.registry import create_rag_manager


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    storage_config = RedisConfig()
    embedder_config = SentenceTransformerConfig()
    manager_config = ManagerConfig(chunk_size=50, overlap=20)
    manager = create_rag_manager(
        index_name='rag_index',
        storage_config=storage_config,
        embedder_config=embedder_config,
        manager_config=manager_config,
    )
    from ragl.registry import AbstractFactory, EmbedderFactory, VectorStoreFactory
    print(AbstractFactory._factory_map.items())
    print(EmbedderFactory._factory_map.items())
    print(VectorStoreFactory._factory_map.items())
    print('*******************************************')

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
    logging.log(logging.INFO, "Starting Test 1: Basic Functionality with String (Redis)")
    manager.reset(reset_metrics=False)
    large_text = (
                     "Artificial Intelligence (AI) is a broad field encompassing various techniques. "
                     "Machine Learning (ML) is a subset of AI that focuses on training models with data. "
                     "Deep Learning (DL), a further subset, uses neural networks with many layers."
                 ) * 2
    docs = manager.add_text(text_or_doc=large_text, base_id="doc:large_redis")
    print_texts(docs, "Documents after adding large text (Redis)")
    contexts = manager.get_context(query="What is Deep Learning?", top_k=2)
    print_context("What is Deep Learning?", contexts)
    assert len(docs) > 1, "Should create multiple chunks"
    assert any("Deep Learning" in ctx.text for ctx in contexts), "Should retrieve relevant content"
    manager.delete_text(docs[0].text_id)
    remaining_texts = manager.list_texts()
    assert docs[0].text_id not in remaining_texts, "Deleted text should not be listed"
