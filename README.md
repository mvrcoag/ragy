# RAGy

RAGy is a simple framework for building Retrieval-Augmented Generation (RAG) applications. It provides a set of tools and utilities to help you create RAG applications quickly and easily.

It ships with a simple interface for building RAG applications, as well as a set of pre-built components that you can use to get started quickly (OpenAI LLMs, Chroma vector stores, etc.).

## Installation

You can install RAGy using pip:

```bash
pip install ragy
```

## Usage

Here's a simple example of how to use RAGy to build a RAG application:

```python
from ragy.rag import RAG
from ragy.reasoning import OpenAIEmbeddingModel, OpenAIGPTEngine
from ragy.rawdoc import DirectoryRawDocumentRetriever
from ragy.vector import ChromaVectorStore

# Create a RAG interface with the necessary components
system_prompt = "You are a helpful assistant that provides accurate information."
embedding_model = OpenAIEmbeddingModel(model='text-embedding-3-small')
raw_document_retriever = DirectoryRawDocumentRetriever(dir='./docs')
vector_store = ChromaVectorStore(path="./chroma", collection_name='my_collection')
ai_engine = OpenAIGPTEngine(model='gpt-5.2')

rag = RAG(
    system_prompt=system_prompt,
    embedding_model=embedding_model,
    raw_document_retriever=raw_document_retriever,
    vector_store=vector_store,
    ai_engine=ai_engine
)

# Use the RAG interface to ingest documents into the vector store
rag.ingest(chunk_size=512, chunk_overlap=128)

# Use the RAG interface to generate a response to a query
response = rag.generate('What is the capital of France?')
print(response)
```

## Contributing
Contributions to RAGy are welcome! If you have an idea for a new feature or improvement, please open an issue or submit a pull request.

## License
RAGy is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

