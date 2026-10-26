import httpx
import pytest

from app.rag.llm import LLMError, OpenAICompatibleClient

MESSAGES = [{"role": "user", "content": "x"}]
OK_BODY = {"choices": [{"message": {"content": " hi [1] "}}]}


def make_client(handler, api_key: str = "") -> OpenAICompatibleClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleClient("http://llm.test/v1", model="m", api_key=api_key, client=http)


def test_parses_response_and_strips_whitespace() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json=OK_BODY)

    assert make_client(handler).chat(MESSAGES) == "hi [1]"


def test_http_error_becomes_llm_error() -> None:
    client = make_client(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(LLMError, match="HTTP 500"):
        client.chat(MESSAGES)


def test_connection_error_becomes_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="Cannot reach"):
        make_client(handler).chat(MESSAGES)


def test_missing_model_is_rejected() -> None:
    with pytest.raises(LLMError, match="LLM_MODEL"):
        OpenAICompatibleClient("http://llm.test/v1", model="")


def test_api_key_sent_as_bearer_header() -> None:
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=OK_BODY)

    make_client(handler, api_key="secret").chat(MESSAGES)
    assert seen["auth"] == "Bearer secret"