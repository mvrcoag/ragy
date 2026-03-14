from abc import ABC, abstractmethod
from typing import Iterable, cast
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


class EmbeddingModel(ABC):
    """The EmbeddingModel is responsible for creating embeddings from text. It defines an interface for creating embeddings, which can be implemented in various ways (e.g., using OpenAI's embedding API, using a local embedding model, etc.)."""

    @abstractmethod
    def create_embedding(self, text: str) -> list[float]:
        pass


class OpenAIEmbeddingModel(EmbeddingModel):
    """The OpenAIEmbeddingModel creates embeddings using OpenAI's embedding API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        # Manually added api key here so we can use OPENROUTER api key (poor man's OpenAI lol)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def create_embedding(self, text: str) -> list[float]:
        """Creates an embedding from text using OpenAI's embedding API.
        Args:
            text (str): The text to create an embedding from.
        Returns:
            list[float]: The embedding vector for the input text.
        """
        res = self.client.embeddings.create(input=text, model=self.model)
        return res.data[0].embedding


class AIEngine(ABC):
    """The AIEngine is responsible for generating responses from a list of messages. It defines an interface for generating responses, which can be implemented in various ways (e.g., using OpenAI's chat completion API, using a local language model, etc.)."""

    @abstractmethod
    def generate_response(self, messages: list[dict[str, str]]) -> str | None:
        pass


class OpenAIGPTEngine(AIEngine):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        
        # Manually added api key here so we can use OPENROUTER api key (poor man's OpenAI lol)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    """The OpenAIGPTEngine generates responses using OpenAI's chat completion API."""

    def generate_response(self, messages: list[dict[str, str]]) -> str | None:
        """Generates a response from a list of messages using OpenAI's chat completion API.
        Args:
            messages (list[dict[str, str]]): A list of messages, where each message is a dictionary with "role" and "content" keys.
        Returns:
            str | None: The generated response content, or None if no response was generated.
        """
        res = self.client.chat.completions.create(
            model=self.model,
            messages=cast(Iterable[ChatCompletionMessageParam], messages),
        )
        return res.choices[0].message.content
