from abc import ABC, abstractmethod
from ragy.core import Document


class RawDocumentRetriever(ABC):
    """The RawDocumentRetriever is responsible for retrieving raw documents from a source. It defines an interface for retrieving documents, which can be implemented in various ways (e.g., from a directory, from a database, from an API, etc.)."""

    @abstractmethod
    def retrieve_documents(self) -> list[Document]:
        pass


class DirectoryRawDocumentRetriever(RawDocumentRetriever):
    """The DirectoryRawDocumentRetriever retrieves raw documents from a directory. It supports .txt and .pdf files."""

    def __init__(self, dir: str):
        self.dir = dir

    def retrieve_documents(self):
        """
        Retrieves documents from a directory. Supports .txt and .pdf files.
            Args:
                dir (str): The directory to retrieve documents from.
            Returns:
                list[dict[str, str]]: A list of documents, where each document is a dictionary with "id" and "text" keys.
        """

        import os
        from pypdf import PdfReader

        docs: list[Document] = []

        for filename in os.listdir(self.dir):
            if filename.endswith(".txt"):
                with open(os.path.join(self.dir, filename)) as file:
                    docs.append(Document(filename, file.read()))

            if filename.endswith(".pdf"):
                reader = PdfReader(os.path.join(self.dir, filename))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                docs.append(Document(filename, text))

        return docs
