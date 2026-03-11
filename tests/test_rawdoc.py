import sys
from types import SimpleNamespace

from ragy.rawdoc import DirectoryRawDocumentRetriever


def test_directory_retriever_reads_txt_files(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")

    retriever = DirectoryRawDocumentRetriever(str(tmp_path))
    docs = retriever.retrieve_documents()

    docs_by_id = {doc.id: doc.content for doc in docs}
    assert docs_by_id == {"a.txt": "alpha", "b.txt": "beta"}


def test_directory_retriever_reads_pdf_files(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakePdfReader:
        def __init__(self, path):
            assert path == str(pdf_path)
            self.pages = [FakePage("one"), FakePage("two")]

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=FakePdfReader))

    retriever = DirectoryRawDocumentRetriever(str(tmp_path))
    docs = retriever.retrieve_documents()

    assert len(docs) == 1
    assert docs[0].id == "sample.pdf"
    assert docs[0].content == "one\ntwo\n"


def test_directory_retriever_ignores_unsupported_extensions(tmp_path):
    (tmp_path / "notes.md").write_text("ignore me")
    (tmp_path / "data.bin").write_bytes(b"\x00\x01")

    retriever = DirectoryRawDocumentRetriever(str(tmp_path))
    docs = retriever.retrieve_documents()

    assert docs == []
