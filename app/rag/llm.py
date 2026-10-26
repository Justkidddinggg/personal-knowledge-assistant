"""LLM client for any OpenAI-compatible /chat/completions endpoint."""
import logging
from functools import lru_cache
from typing import Protocol

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Anything that stops us from getting a completion."""


class ChatLLM(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str: ...


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.1,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not model:
            raise LLMError("LLM_MODEL is not set. Add it to your .env file.")
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._client = client or httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict[str, str]]) -> str:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        try:
            response = self._client.post(self._url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:300]
            raise LLMError(f"LLM returned HTTP {exc.response.status_code}: {body}") from exc
        except httpx.TimeoutException as exc:
            raise LLMError("LLM request timed out") from exc
        except httpx.RequestError as exc:
            raise LLMError(
                f"Cannot reach the LLM at {self._url}. Is the server running?"
            ) from exc

        try:
            return response.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("Unexpected response format from the LLM") from exc


@lru_cache
def get_llm() -> OpenAICompatibleClient:
    s = get_settings()
    return OpenAICompatibleClient(
        base_url=s.llm_base_url,
        model=s.llm_model,
        api_key=s.llm_api_key,
        temperature=s.llm_temperature,
    )