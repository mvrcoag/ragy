import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ragy.core import Chunk
from ragy.vector import ChromaVectorStore


def test_chroma_store_init_creates_collection(monkeypatch):
    client_instance = MagicMock()
    collection = MagicMock()
    client_instance.get_or_create_collection.return_value = collection
    persistent_client = MagicMock(return_value=client_instance)

    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        SimpleNamespace(PersistentClient=persistent_client),
    )

    store = ChromaVectorStore(path="/tmp/chroma", collection_name="my_collection")

    persistent_client.assert_called_once_with(path="/tmp/chroma")
    client_instance.get_or_create_collection.assert_called_once_with("my_collection")
    assert store.collection is collection


def test_upsert_chunk_embedding_passes_metadata_when_present():
    store = object.__new__(ChromaVectorStore)
    store.collection = MagicMock()
    chunk = Chunk("id1", "chunk text", {"source": "unit"})

    store.upsert_chunk_embedding("id1", [0.1, 0.2], chunk)

    store.collection.upsert.assert_called_once_with(
        ids=["id1"],
        embeddings=[[0.1, 0.2]],
        documents=["chunk text"],
        metadatas=[{"source": "unit"}],
    )


def test_upsert_chunk_embedding_passes_none_metadata_when_missing():
    store = object.__new__(ChromaVectorStore)
    store.collection = MagicMock()
    chunk = Chunk("id1", "chunk text")

    store.upsert_chunk_embedding("id1", [0.1, 0.2], chunk)

    store.collection.upsert.assert_called_once_with(
        ids=["id1"],
        embeddings=[[0.1, 0.2]],
        documents=["chunk text"],
        metadatas=None,
    )


def test_retrieve_similar_chunks_maps_results_to_chunk_objects():
    store = object.__new__(ChromaVectorStore)
    store.collection = MagicMock()
    store.collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "documents": [["doc one", "doc two"]],
    }

    chunks = store.retrieve_similar_chunks([0.9], top_k=2)

    store.collection.query.assert_called_once_with(
        query_embeddings=[[0.9]], n_results=2
    )
    assert [chunk.id for chunk in chunks] == ["id1", "id2"]
    assert [chunk.content for chunk in chunks] == ["doc one", "doc two"]
    assert [chunk.metadata for chunk in chunks] == [None, None]


@pytest.mark.parametrize(
    "results",
    [
        {"ids": None, "documents": [["doc one"]]},
        {"ids": [["id1"]], "documents": None},
    ],
)
def test_retrieve_similar_chunks_returns_empty_when_ids_or_docs_missing(results):
    store = object.__new__(ChromaVectorStore)
    store.collection = MagicMock()
    store.collection.query.return_value = results

    chunks = store.retrieve_similar_chunks([0.9], top_k=2)

    assert chunks == []
