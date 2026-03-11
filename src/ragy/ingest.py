from ragy.core import Document, Chunk, EmbeddingModel, RawDocumentRetriever, VectorStore


class Ingestor:
    """The Ingestor is responsible for ingesting documents into the system. It retrieves raw documents, splits them into chunks, creates embeddings for each chunk, and stores the embeddings in a vector store."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        raw_document_retriever: RawDocumentRetriever,
        vector_store: VectorStore,
    ):
        self.embedding_model = embedding_model
        self.raw_document_retriever = raw_document_retriever
        self.vector_store = vector_store

    def split_document_into_chunks(
        self, document: Document, chunk_size: int, chunk_overlap: int
    ):
        """
        Splits a document into chunks of a specified size with a specified overlap.
            Args:
                document (dict[str, str]): The document to split, represented as a dictionary with "id" and "text" keys.
                chunk_size (int): The size of each chunk in characters.
                chunk_overlap (int): The number of characters to overlap between chunks.
            Returns:
                list[dict[str, str]]: A list of chunks, where each chunk is a dictionary with "id" and "text" keys.
        """
        chunks: list[Chunk] = []
        start = 0

        id = document.id
        text = document.content
        i = 1

        while start < len(text):
            end = start + chunk_size
            chunks.append(Chunk(f"{id}_chunk_{i}", text[start:end]))
            start = end - chunk_overlap
            i += 1

        return chunks

    def ingest(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Ingests documents into the system. Retrieves raw documents, splits them into chunks, creates embeddings for each chunk, and stores the embeddings in a vector store.
            Args:
                chunk_size (int): The size of each chunk in characters. Default is 1000.
                chunk_overlap (int): The number of characters to overlap between chunks. Default is 200.
        """
        chunked_docs = []

        docs = self.raw_document_retriever.retrieve_documents()

        for doc in docs:
            chunks = self.split_document_into_chunks(doc, chunk_size, chunk_overlap)
            chunked_docs.extend(chunks)

        for chunk in chunked_docs:
            embedding = self.embedding_model.create_embedding(chunk.content)

            self.vector_store.upsert_chunk_embedding(chunk.id, embedding, chunk, {})
