class Document:
    """A Document represents a piece of text that can be ingested into the system. It has an id and content."""

    def __init__(self, id: str, content: str):
        self.id = id
        self.content = content


class Chunk:
    """A Chunk represents a piece of a Document that has been split into smaller parts. It has an id and content."""

    def __init__(self, id: str, content: str):
        self.id = id
        self.content = content
