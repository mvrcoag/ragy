class Document:
    """A Document represents a piece of text that can be ingested into the system. It has an id, content, and optional metadata."""

    def __init__(self, id: str, content: str, metadata: dict | None):
        self.id = id
        self.content = content
        self.metadata = metadata


class Chunk:
    """A Chunk represents a piece of a Document that has been split into smaller parts. It has an id, content, and optional metadata."""

    def __init__(self, id: str, content: str, metadata: dict | None = None):
        self.id = id
        self.content = content
        self.metadata = metadata
