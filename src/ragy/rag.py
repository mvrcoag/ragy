from ragy.reasoning import EmbeddingModel, AIEngine
from ragy.rawdoc import RawDocumentRetriever
from ragy.vector import VectorStore
from ragy.ingest import Ingestor


class RAG:
    """The RAG class is the main entry point for the system. It is responsible for orchestrating the various components (embedding model, raw document retriever, vector store, and AI engine) to provide a complete retrieval-augmented generation pipeline."""

    def __init__(
        self,
        system_prompt: str,
        embedding_model: EmbeddingModel,
        raw_document_retriever: RawDocumentRetriever,
        vector_store: VectorStore,
        ai_engine: AIEngine,
    ):
        self.system_prompt = system_prompt
        self.embedding_model = embedding_model
        self.raw_document_retriever = raw_document_retriever
        self.vector_store = vector_store
        self.ai_engine = ai_engine
        self.ingestor = Ingestor(
            self.embedding_model, self.raw_document_retriever, self.vector_store
        )

    def generate(self, query: str, top_k: int = 5) -> str | None:
        """
        Generates a response to a user query. Retrieves similar chunks from the vector store based on the query embedding, constructs a list of messages for the AI engine, and generates a response.
        Args:
            query (str): The user query to process.
            top_k (int): The number of similar chunks to retrieve from the vector store. Default is 5.
        Returns:
            str | None: The generated response content, or None if no response was generated.
        """
        query_embedding = self.embedding_model.create_embedding(query)
        similar_chunks = self.vector_store.retrieve_similar_chunks(
            query_embedding, top_k
        )

        messages = [{"role": "system", "content": self.system_prompt}]
        for chunk in similar_chunks:
            messages.append({"role": "user", "content": chunk.content})
        messages.append({"role": "user", "content": query})

        response = self.ai_engine.generate_response(messages)
        return response

    def ingest(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Ingests documents into the system. Retrieves raw documents, splits them into chunks, creates embeddings for each chunk, and stores the embeddings in a vector store.
            Args:
                chunk_size (int): The size of each chunk in characters. Default is 1000.
                chunk_overlap (int): The number of characters to overlap between chunks. Default is 200.
        """
        self.ingestor.ingest(chunk_size, chunk_overlap)
