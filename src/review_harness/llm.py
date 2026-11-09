"""A thin layer over each provider's official SDK.

There is deliberately no way to pass tools here. A model that reads an untrusted paper gets no web
search, code execution or file access, which also means it cannot look up the withdrawal notice.

Automatic fallback to another model is deliberately off, including Anthropic's server-side
`fallbacks`. Every result must belong to the model named in the results, so a refusal is recorded
as a refusal. Provider keys come from the environment and are never logged or written to a ledger.
"""

import os
import tomllib
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from importlib import resources
from typing import Any, Protocol

from pydantic import BaseModel

REQUEST_TIMEOUT_SECONDS = 900.0
KEY_VARIABLES = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
CLI_PROVIDERS = ("claude-code", "codex")


@dataclass(frozen=True)
class ModelInfo:
    name: str
    provider: str
    input_per_mtok: float
    output_per_mtok: float
    source: str
    knowledge_cutoff: date | None = None
    cached_per_mtok: float | None = None
    effort: str | None = None
    price_verified: bool = False
    # "subscription" models run through a signed-in CLI (see cli_llm.py): they cost nothing against
    # --max-usd, and their prices are kept only to report what the same run would cost on the API.
    billing: str = "api"
    cli_model: str | None = None

    @property
    def subscription(self) -> bool:
        return self.billing == "subscription"

    def is_unseen(self, event: date) -> bool:
        """True only if the event is known to fall after this model's training data."""
        return self.knowledge_cutoff is not None and event > self.knowledge_cutoff

    def cost_usd(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        cached_price = (
            self.cached_per_mtok if self.cached_per_mtok is not None else self.input_per_mtok
        )
        uncached = max(0, input_tokens - cached_tokens)
        return (
            uncached * self.input_per_mtok
            + cached_tokens * cached_price
            + output_tokens * self.output_per_mtok
        ) / 1_000_000


@lru_cache
def load_registry() -> dict[str, ModelInfo]:
    text = resources.files("review_harness").joinpath("models.toml").read_text(encoding="utf-8")
    registry = {}
    for name, fields in tomllib.loads(text)["models"].items():
        cutoff = fields.pop("knowledge_cutoff", None)
        registry[name] = ModelInfo(
            name=name,
            knowledge_cutoff=date.fromisoformat(cutoff) if cutoff else None,
            **fields,
        )
    return registry


def get_model(name: str) -> ModelInfo:
    registry = load_registry()
    if name in registry:
        return registry[name]
    if name.startswith("openrouter/"):
        # Unlisted route with no known cutoff or price: usable, but never reported as unseen.
        return ModelInfo(
            name=name.removeprefix("openrouter/"),
            provider="openrouter",
            input_per_mtok=0.0,
            output_per_mtok=0.0,
            source="https://openrouter.ai/models",
        )
    raise KeyError(f"Unknown model {name!r}. Add it to models.toml with its cutoff and prices.")


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    schema: dict[str, Any]
    schema_name: str
    max_output_tokens: int


class LLMResult(BaseModel):
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    stop: str = "complete"
    response_id: str | None = None


class ProviderError(Exception):
    """The call failed for an operational reason, so it may be retried on a later run."""


class ModelDeclined(Exception):
    """The model answered but gave no usable text: refused, or hit the output limit."""

    def __init__(self, outcome: str, result: LLMResult | None = None):
        super().__init__(outcome)
        self.outcome = outcome
        self.result = result


class LLM(Protocol):
    async def complete(self, model: ModelInfo, request: LLMRequest) -> LLMResult: ...


def _describe(error: Exception) -> str:
    """Class and status only: provider error messages can echo part of a key."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    return f"{type(error).__name__}" + (f" (status {status})" if status else "")


def _attr(obj: Any, *path: str, default: Any = 0) -> Any:
    for name in path:
        obj = getattr(obj, name, None)
        if obj is None:
            return default
    return obj


class ProviderLLM:
    """Real provider calls. Clients are created on first use, so only keys in use are needed."""

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}
        self._cli: Any = None

    def _key(self, provider: str) -> str:
        variable = KEY_VARIABLES[provider]
        value = os.environ.get(variable)
        if not value:
            raise ProviderError(f"{variable} is not set")
        return value

    def _client(self, provider: str) -> Any:
        if provider in self._clients:
            return self._clients[provider]
        key = self._key(provider)
        if provider == "openai":
            from openai import AsyncOpenAI

            client: Any = AsyncOpenAI(api_key=key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=3)
        elif provider == "openrouter":
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=key,
                base_url="https://openrouter.ai/api/v1",
                timeout=REQUEST_TIMEOUT_SECONDS,
                max_retries=3,
            )
        elif provider == "anthropic":
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=3)
        elif provider == "gemini":
            from google import genai

            client = genai.Client(api_key=key)
        else:
            raise ProviderError(f"Unknown provider {provider}")
        self._clients[provider] = client
        return client

    async def complete(self, model: ModelInfo, request: LLMRequest) -> LLMResult:
        if model.provider in CLI_PROVIDERS:
            if self._cli is None:
                from review_harness.cli_llm import CliLLM

                self._cli = CliLLM()
            return await self._cli.complete(model, request)
        call = {
            "openai": self._openai,
            "anthropic": self._anthropic,
            "gemini": self._gemini,
            "openrouter": self._openrouter,
        }[model.provider]
        client = self._client(model.provider)
        try:
            return await call(client, model, request)
        except (ModelDeclined, ProviderError):
            raise
        except Exception as error:
            raise ProviderError(f"{model.provider}: {_describe(error)}") from None

    async def _openai(self, client: Any, model: ModelInfo, request: LLMRequest) -> LLMResult:
        arguments: dict[str, Any] = {
            "model": model.name,
            "instructions": request.system,
            "input": request.user,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": request.schema_name,
                    "schema": request.schema,
                    "strict": True,
                }
            },
            "max_output_tokens": request.max_output_tokens,
            "store": False,
        }
        if model.effort:
            arguments["reasoning"] = {"effort": model.effort}
        response = await client.responses.create(**arguments)
        result = LLMResult(
            text=response.output_text or "",
            input_tokens=_attr(response, "usage", "input_tokens"),
            output_tokens=_attr(response, "usage", "output_tokens"),
            reasoning_tokens=_attr(response, "usage", "output_tokens_details", "reasoning_tokens"),
            cached_input_tokens=_attr(response, "usage", "input_tokens_details", "cached_tokens"),
            stop=response.status or "complete",
            response_id=response.id,
        )
        if response.status == "incomplete":
            reason = _attr(response, "incomplete_details", "reason", default="incomplete")
            raise ModelDeclined("refused" if reason == "content_filter" else "truncated", result)
        for item in response.output or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "refusal":
                    raise ModelDeclined("refused", result)
        return result

    async def _anthropic(self, client: Any, model: ModelInfo, request: LLMRequest) -> LLMResult:
        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": request.schema}
        }
        if model.effort:
            output_config["effort"] = model.effort
        # Streaming, since long inputs and large output limits can outlast a plain request.
        async with client.messages.stream(
            model=model.name,
            max_tokens=request.max_output_tokens,
            system=request.system,
            messages=[{"role": "user", "content": request.user}],
            output_config=output_config,
        ) as stream:
            message = await stream.get_final_message()
        text = "".join(block.text for block in message.content if block.type == "text")
        result = LLMResult(
            text=text,
            input_tokens=(
                _attr(message, "usage", "input_tokens")
                + _attr(message, "usage", "cache_read_input_tokens")
            ),
            output_tokens=_attr(message, "usage", "output_tokens"),
            cached_input_tokens=_attr(message, "usage", "cache_read_input_tokens"),
            stop=message.stop_reason or "complete",
            response_id=message.id,
        )
        if message.stop_reason == "refusal":
            raise ModelDeclined("refused", result)
        if message.stop_reason == "max_tokens":
            raise ModelDeclined("truncated", result)
        return result

    async def _gemini(self, client: Any, model: ModelInfo, request: LLMRequest) -> LLMResult:
        from google.genai import types

        response = await client.aio.models.generate_content(
            model=model.name,
            contents=request.user,
            config=types.GenerateContentConfig(
                system_instruction=request.system,
                response_mime_type="application/json",
                response_json_schema=request.schema,
                max_output_tokens=request.max_output_tokens,
            ),
        )
        candidates = response.candidates or []
        finish = str(_attr(candidates[0], "finish_reason", default="")) if candidates else "BLOCKED"
        result = LLMResult(
            text=response.text or "",
            input_tokens=_attr(response, "usage_metadata", "prompt_token_count"),
            output_tokens=(
                _attr(response, "usage_metadata", "candidates_token_count")
                + _attr(response, "usage_metadata", "thoughts_token_count")
            ),
            reasoning_tokens=_attr(response, "usage_metadata", "thoughts_token_count"),
            cached_input_tokens=_attr(response, "usage_metadata", "cached_content_token_count"),
            stop=finish,
            response_id=getattr(response, "response_id", None),
        )
        if "MAX_TOKENS" in finish:
            raise ModelDeclined("truncated", result)
        if not candidates or not result.text:
            raise ModelDeclined("refused", result)
        return result

    async def _openrouter(self, client: Any, model: ModelInfo, request: LLMRequest) -> LLMResult:
        response = await client.chat.completions.create(
            model=model.name,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "strict": True,
                    "schema": request.schema,
                },
            },
            max_tokens=request.max_output_tokens,
        )
        choice = response.choices[0]
        result = LLMResult(
            text=choice.message.content or "",
            input_tokens=_attr(response, "usage", "prompt_tokens"),
            output_tokens=_attr(response, "usage", "completion_tokens"),
            stop=choice.finish_reason or "complete",
            response_id=response.id,
        )
        if choice.finish_reason == "length":
            raise ModelDeclined("truncated", result)
        if choice.finish_reason == "content_filter" or getattr(choice.message, "refusal", None):
            raise ModelDeclined("refused", result)
        return result
