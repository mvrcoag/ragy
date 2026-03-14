import os
import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv

from ragy.rag import RAG
from ragy.reasoning import OpenAIEmbeddingModel, OpenAIGPTEngine
from ragy.rawdoc import DirectoryRawDocumentRetriever
from ragy.vector import WeaviateVectorStore

# Please add OPENAI_API_KEY or OPENROUTER_API_KEY to your .env file
load_dotenv()


# This example assumes you have a Weaviate instance running locally via Docker.
# Run command: docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semi/weaviate:latest
try:
    client = weaviate.connect_to_local()
except weaviate.exceptions.WeaviateStartUpError:
    print("Could not connect to Weaviate. Please ensure it is running in Docker.")
    exit(1)


try:
    collection_name = "RagyExampleCollection"

    # For this example, we delete the collection to ensure a clean start.
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    # Define the schema for our Weaviate collection.
    client.collections.create(
        name=collection_name,
        properties=[
            wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="metadata", data_type=wvc.config.DataType.TEXT),
        ],
    )

    vector_store = WeaviateVectorStore(client=client, collection_name=collection_name)
    
    embedding_model = OpenAIEmbeddingModel(model='text-embedding-3-small')
    raw_document_retriever = DirectoryRawDocumentRetriever(dir='./docs')
    ai_engine = OpenAIGPTEngine(model='gpt-4o')

    rag = RAG(
        system_prompt="You are a helpful assistant. Answer questions based on the retrieved documents.",
        embedding_model=embedding_model,
        raw_document_retriever=raw_document_retriever,
        vector_store=vector_store,
        ai_engine=ai_engine
    )

    print("Ingesting documents into Weaviate...")
    rag.ingest(chunk_size=512, chunk_overlap=128)

    print("\nGenerating response...")
    response = rag.generate('What is Ragy?')
    print(f"\n> Response: {response}")

finally:
    client.close()
    print("\nDone.")
