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


def test_weaviate_store_init_creates_collection():
    from ragy.vector import WeaviateVectorStore
    client_instance = MagicMock()
    collection = MagicMock()
    client_instance.collections.get.return_value = collection

    store = WeaviateVectorStore(client_instance, "my_collection")

    client_instance.collections.get.assert_called_once_with("my_collection")
    assert store.collection is collection


def test_weaviate_upsert_chunk_embedding_with_metadata(monkeypatch):
    from ragy.vector import WeaviateVectorStore
    store = object.__new__(WeaviateVectorStore)
    store.collection = MagicMock()
    chunk = Chunk("id1", "chunk text", {"source": "unit"})
    
    mock_uuid = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "weaviate.util",
        SimpleNamespace(generate_uuid5=mock_uuid),
    )
    mock_uuid.return_value = "fake-uuid"

    store.upsert_chunk_embedding("id1", [0.1, 0.2], chunk)

    store.collection.data.insert.assert_called_once_with(
        properties={"chunk_id": "id1", "content": "chunk text", "metadata": '{"source": "unit"}'},
        uuid="fake-uuid",
        vector=[0.1, 0.2],
    )


def test_weaviate_retrieve_similar_chunks_maps_results():
    from ragy.vector import WeaviateVectorStore
    store = object.__new__(WeaviateVectorStore)
    store.collection = MagicMock()
    
    mock_obj1 = MagicMock()
    mock_obj1.uuid = "uuid1"
    mock_obj1.properties = {"chunk_id": "id1", "content": "doc one", "metadata": '{"source": "unit"}'}
    
    mock_obj2 = MagicMock()
    mock_obj2.uuid = "uuid2"
    mock_obj2.properties = {"chunk_id": "id2", "content": "doc two"}
    
    mock_response = MagicMock()
    mock_response.objects = [mock_obj1, mock_obj2]
    
    store.collection.query.near_vector.return_value = mock_response

    chunks = store.retrieve_similar_chunks([0.9], top_k=2)

    store.collection.query.near_vector.assert_called_once_with(
        near_vector=[0.9], limit=2, return_properties=["chunk_id", "content", "metadata"]
    )
    assert [chunk.id for chunk in chunks] == ["id1", "id2"]
    assert [chunk.content for chunk in chunks] == ["doc one", "doc two"]
    assert [chunk.metadata for chunk in chunks] == [{"source": "unit"}, None]
