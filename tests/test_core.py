from ragy.core import Chunk, Document


def test_document_fields():
    doc = Document(id="doc-1", content="hello", metadata={"source": "unit"})

    assert doc.id == "doc-1"
    assert doc.content == "hello"
    assert doc.metadata == {"source": "unit"}


def test_chunk_fields():
    chunk = Chunk(id="doc-1_chunk_1", content="hello")

    assert chunk.id == "doc-1_chunk_1"
    assert chunk.content == "hello"
    assert chunk.metadata is None
