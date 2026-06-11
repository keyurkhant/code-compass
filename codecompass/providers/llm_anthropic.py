import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from codecompass.providers.base import LLMProvider


class AnthropicLLMProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: str) -> None:
        self._model = model_name
        self._client = anthropic.Anthropic(api_key=api_key)

    @retry(
        retry=retry_if_exception_type(anthropic.RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2048,
        system: str | None = None,
    ) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system or "",
            messages=messages,
        )
        return response.content[0].text
