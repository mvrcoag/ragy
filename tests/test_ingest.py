from unittest.mock import MagicMock, call

from ragy.core import Document
from ragy.ingest import Ingestor


def test_split_document_into_chunks_with_overlap():
    ingestor = Ingestor(MagicMock(), MagicMock(), MagicMock())
    doc = Document("doc", "abcdefghij", {"source": "s"})

    chunks = ingestor.split_document_into_chunks(doc, chunk_size=4, chunk_overlap=1)

    assert [chunk.id for chunk in chunks] == [
        "doc_chunk_1",
        "doc_chunk_2",
        "doc_chunk_3",
        "doc_chunk_4",
    ]
    assert [chunk.content for chunk in chunks] == ["abcd", "defg", "ghij", "j"]
    assert [chunk.metadata for chunk in chunks] == [
        {"source": "s"},
        {"source": "s"},
        {"source": "s"},
        {"source": "s"},
    ]


def test_split_document_into_chunks_no_overlap():
    ingestor = Ingestor(MagicMock(), MagicMock(), MagicMock())
    doc = Document("doc", "abcdefghij")

    chunks = ingestor.split_document_into_chunks(doc, chunk_size=4, chunk_overlap=0)

    assert [chunk.id for chunk in chunks] == [
        "doc_chunk_1",
        "doc_chunk_2",
        "doc_chunk_3",
    ]
    assert [chunk.content for chunk in chunks] == ["abcd", "efgh", "ij"]


def test_split_document_into_chunks_empty_content():
    ingestor = Ingestor(MagicMock(), MagicMock(), MagicMock())
    doc = Document("doc", "")

    chunks = ingestor.split_document_into_chunks(doc, chunk_size=4, chunk_overlap=1)

    assert chunks == []


def test_ingest_calls_embedding_and_upsert_for_each_chunk():
    embedding_model = MagicMock()
    embedding_model.create_embedding.side_effect = [[0.1], [0.2]]

    raw_document_retriever = MagicMock()
    raw_document_retriever.retrieve_documents.return_value = [
        Document("doc1", "abcdef", {"source": "x"})
    ]

    vector_store = MagicMock()
    ingestor = Ingestor(embedding_model, raw_document_retriever, vector_store)

    ingestor.ingest(chunk_size=3, chunk_overlap=0)

    assert embedding_model.create_embedding.call_args_list == [call("abc"), call("def")]
    assert vector_store.upsert_chunk_embedding.call_count == 2

    first = vector_store.upsert_chunk_embedding.call_args_list[0].args
    second = vector_store.upsert_chunk_embedding.call_args_list[1].args

    assert first[0] == "doc1_chunk_1"
    assert first[1] == [0.1]
    assert first[2].id == "doc1_chunk_1"
    assert first[2].content == "abc"
    assert first[2].metadata == {"source": "x"}

    assert second[0] == "doc1_chunk_2"
    assert second[1] == [0.2]
    assert second[2].id == "doc1_chunk_2"
    assert second[2].content == "def"
    assert second[2].metadata == {"source": "x"}


def test_ingest_no_documents_does_nothing():
    embedding_model = MagicMock()

    raw_document_retriever = MagicMock()
    raw_document_retriever.retrieve_documents.return_value = []

    vector_store = MagicMock()
    ingestor = Ingestor(embedding_model, raw_document_retriever, vector_store)

    ingestor.ingest(chunk_size=3, chunk_overlap=0)

    embedding_model.create_embedding.assert_not_called()
    vector_store.upsert_chunk_embedding.assert_not_called()
