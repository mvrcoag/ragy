import os
from dotenv import load_dotenv

from ragy.rag import RAG
from ragy.reasoning import OpenAIEmbeddingModel, OpenAIGPTEngine
from ragy.rawdoc import DirectoryRawDocumentRetriever
from ragy.vector import ChromaVectorStore

# Please add OPENAI_API_KEY or OPENROUTER_API_KEY to your .env file
load_dotenv()


vector_store = ChromaVectorStore(path="./chroma_data", collection_name='RagyExampleCollection')

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

print("Ingesting documents into ChromaDB...")
rag.ingest(chunk_size=512, chunk_overlap=128)

print("\nGenerating response...")
response = rag.generate('What is Ragy?')
print(f"\n Response: {response}")

print("\nDone.")
