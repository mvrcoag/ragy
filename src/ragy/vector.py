from abc import ABC, abstractmethod
from ragy.core import Chunk


class VectorStore(ABC):
    """The VectorStore is responsible for storing and retrieving chunk embeddings. It defines an interface for upserting chunk embeddings and retrieving similar chunks based on a query embedding, which can be implemented in various ways (e.g., using ChromaDB, using a local vector store, etc.)."""

    @abstractmethod
    def upsert_chunk_embedding(
        self,
        chunk_id: str,
        embedding: list[float],
        chunk: Chunk,
    ):
        pass

    @abstractmethod
    def retrieve_similar_chunks(
        self, query_embedding: list[float], top_k: int
    ) -> list[Chunk]:
        pass


class ChromaVectorStore(VectorStore):
    """The ChromaVectorStore stores and retrieves chunk embeddings using ChromaDB."""

    def __init__(self, collection_name: str):
        from chromadb import Client

        self.client = Client()
        self.collection = self.client.get_or_create_collection(collection_name)

    def upsert_chunk_embedding(
        self, chunk_id: str, embedding: list[float], chunk: Chunk
    ):
        """Upserts a chunk embedding into the ChromaDB collection.
        Args:
            chunk_id (str): The unique identifier for the chunk.
            embedding (list[float]): The embedding vector for the chunk.
            chunk (Chunk): The Chunk object containing the chunk's content and metadata.
        """
        self.collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk.content],
            metadatas=[chunk.metadata] if chunk.metadata is not None else None,
        )

    def retrieve_similar_chunks(self, query_embedding: list[float], top_k: int):
        """Retrieves similar chunks from the ChromaDB collection based on a query embedding.
        Args:
            query_embedding (list[float]): The embedding vector for the query.
            top_k (int): The number of similar chunks to retrieve.
        Returns:
            list[Chunk]: A list of similar chunks, where each chunk is represented as a Chunk object with an id and content.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        ids = results["ids"]
        documents = results["documents"]

        if ids is None or documents is None:
            return []

        chunks: list[Chunk] = []
        for id, document in zip(ids[0], documents[0]):
            chunks.append(Chunk(id, document))

        return chunks
