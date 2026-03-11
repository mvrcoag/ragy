from unittest.mock import MagicMock

from ragy.core import Chunk
from ragy.rag import RAG


def test_generate_builds_messages_from_system_chunks_and_query():
    embedding_model = MagicMock()
    embedding_model.create_embedding.return_value = [0.5, 0.7]

    vector_store = MagicMock()
    vector_store.retrieve_similar_chunks.return_value = [
        Chunk("c1", "alpha"),
        Chunk("c2", "beta"),
    ]

    ai_engine = MagicMock()
    ai_engine.generate_response.return_value = "final answer"

    rag = RAG(
        system_prompt="You are helpful",
        embedding_model=embedding_model,
        raw_document_retriever=MagicMock(),
        vector_store=vector_store,
        ai_engine=ai_engine,
    )

    result = rag.generate("what is this?", top_k=2)

    assert result == "final answer"
    embedding_model.create_embedding.assert_called_once_with("what is this?")
    vector_store.retrieve_similar_chunks.assert_called_once_with([0.5, 0.7], 2)

    messages = ai_engine.generate_response.call_args.args[0]
    assert messages == [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Retrieved Chunk ID: c1\nContent: alpha"},
        {"role": "user", "content": "Retrieved Chunk ID: c2\nContent: beta"},
        {"role": "user", "content": "what is this?"},
    ]


def test_generate_uses_top_k_and_returns_engine_response():
    embedding_model = MagicMock()
    embedding_model.create_embedding.return_value = [1.0]

    vector_store = MagicMock()
    vector_store.retrieve_similar_chunks.return_value = []

    ai_engine = MagicMock()
    ai_engine.generate_response.return_value = "ok"

    rag = RAG(
        system_prompt="system",
        embedding_model=embedding_model,
        raw_document_retriever=MagicMock(),
        vector_store=vector_store,
        ai_engine=ai_engine,
    )

    result = rag.generate("query", top_k=7)

    assert result == "ok"
    vector_store.retrieve_similar_chunks.assert_called_once_with([1.0], 7)


def test_generate_with_no_chunks_still_calls_engine():
    embedding_model = MagicMock()
    embedding_model.create_embedding.return_value = [0.1]

    vector_store = MagicMock()
    vector_store.retrieve_similar_chunks.return_value = []

    ai_engine = MagicMock()
    ai_engine.generate_response.return_value = "empty-context-answer"

    rag = RAG(
        system_prompt="system prompt",
        embedding_model=embedding_model,
        raw_document_retriever=MagicMock(),
        vector_store=vector_store,
        ai_engine=ai_engine,
    )

    result = rag.generate("question")

    assert result == "empty-context-answer"
    messages = ai_engine.generate_response.call_args.args[0]
    assert messages == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "question"},
    ]


def test_ingest_delegates_to_ingestor():
    rag = RAG(
        system_prompt="system",
        embedding_model=MagicMock(),
        raw_document_retriever=MagicMock(),
        vector_store=MagicMock(),
        ai_engine=MagicMock(),
    )
    rag.ingestor = MagicMock()

    rag.ingest(chunk_size=123, chunk_overlap=45)

    rag.ingestor.ingest.assert_called_once_with(123, 45)
