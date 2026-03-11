from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ragy.reasoning import OpenAIEmbeddingModel, OpenAIGPTEngine


def test_openai_embedding_model_returns_first_embedding():
    client = MagicMock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=[1.0, 2.0, 3.0])]
    )

    with patch("ragy.reasoning.OpenAI", return_value=client) as openai_cls:
        model = OpenAIEmbeddingModel(model="text-embedding-3-small")
        embedding = model.create_embedding("hello")

    openai_cls.assert_called_once_with()
    client.embeddings.create.assert_called_once_with(
        input="hello", model="text-embedding-3-small"
    )
    assert embedding == [1.0, 2.0, 3.0]


def test_openai_gpt_engine_calls_chat_completions_and_returns_content():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )
    messages = [{"role": "user", "content": "Hello"}]

    with patch("ragy.reasoning.OpenAI", return_value=client) as openai_cls:
        engine = OpenAIGPTEngine(model="gpt-5.2")
        response = engine.generate_response(messages)

    openai_cls.assert_called_once_with()
    client.chat.completions.create.assert_called_once_with(
        model="gpt-5.2", messages=messages
    )
    assert response == "answer"
