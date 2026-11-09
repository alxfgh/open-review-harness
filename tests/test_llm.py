"""Request shapes per provider. Only OpenAI has also been run for real; see the README."""

from types import SimpleNamespace as NS

import pytest

from review_harness.llm import (
    LLMRequest,
    ModelDeclined,
    ModelInfo,
    ProviderError,
    ProviderLLM,
    get_model,
    load_registry,
)
from review_harness.schemas import Findings, strict_schema

REQUEST = LLMRequest(
    system="sys",
    user="usr",
    schema=strict_schema(Findings),
    schema_name="Findings",
    max_output_tokens=99,
)


def model(provider="openai", effort=None):
    return ModelInfo("test-model", provider, 1.0, 2.0, "src", effort=effort)


class Recorder:
    """Stands in for a provider client and records exactly what was sent."""

    def __init__(self, reply):
        self.reply = reply
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.reply


def with_client(provider, client):
    llm = ProviderLLM()
    llm._clients[provider] = client
    return llm


def openai_reply(status="completed", text='{"findings": []}', refusal=False):
    parts = [NS(type="refusal")] if refusal else [NS(type="output_text")]
    return NS(
        output_text=text,
        status=status,
        id="resp_1",
        output=[NS(content=parts)],
        incomplete_details=NS(reason="max_output_tokens") if status == "incomplete" else None,
        usage=NS(
            input_tokens=100,
            output_tokens=50,
            output_tokens_details=NS(reasoning_tokens=30),
            input_tokens_details=NS(cached_tokens=20),
        ),
    )


class TestOpenAI:
    async def test_request_shape_and_usage(self):
        responses = Recorder(openai_reply())
        llm = with_client("openai", NS(responses=responses))
        result = await llm.complete(model(effort="medium"), REQUEST)
        sent = responses.kwargs
        assert sent["instructions"] == "sys" and sent["input"] == "usr"
        assert sent["text"]["format"]["type"] == "json_schema"
        assert sent["text"]["format"]["strict"] is True
        assert sent["reasoning"] == {"effort": "medium"}
        assert sent["store"] is False
        assert result.reasoning_tokens == 30 and result.cached_input_tokens == 20

    async def test_no_tools_are_ever_sent(self):
        responses = Recorder(openai_reply())
        await with_client("openai", NS(responses=responses)).complete(model(), REQUEST)
        assert "tools" not in responses.kwargs and "tool_choice" not in responses.kwargs

    async def test_older_models_get_no_reasoning_parameter(self):
        responses = Recorder(openai_reply())
        await with_client("openai", NS(responses=responses)).complete(model(), REQUEST)
        assert "reasoning" not in responses.kwargs

    async def test_truncation_is_not_treated_as_an_answer(self):
        llm = with_client("openai", NS(responses=Recorder(openai_reply(status="incomplete"))))
        with pytest.raises(ModelDeclined) as caught:
            await llm.complete(model(), REQUEST)
        assert caught.value.outcome == "truncated"

    async def test_refusal(self):
        llm = with_client("openai", NS(responses=Recorder(openai_reply(refusal=True))))
        with pytest.raises(ModelDeclined) as caught:
            await llm.complete(model(), REQUEST)
        assert caught.value.outcome == "refused"


class FakeStream:
    def __init__(self, message):
        self.message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def get_final_message(self):
        return self.message


class FakeMessages:
    def __init__(self, message):
        self.message = message
        self.kwargs = None

    def stream(self, **kwargs):
        self.kwargs = kwargs
        return FakeStream(self.message)


def anthropic_message(stop="end_turn"):
    return NS(
        id="msg_1",
        stop_reason=stop,
        content=[NS(type="thinking", thinking=""), NS(type="text", text='{"findings": []}')],
        usage=NS(input_tokens=100, output_tokens=40, cache_read_input_tokens=25),
    )


class TestAnthropic:
    async def test_request_shape_and_text(self):
        messages = FakeMessages(anthropic_message())
        llm = with_client("anthropic", NS(messages=messages))
        result = await llm.complete(model("anthropic", effort="medium"), REQUEST)
        sent = messages.kwargs
        assert sent["system"] == "sys"
        assert sent["messages"] == [{"role": "user", "content": "usr"}]
        assert sent["output_config"]["format"]["type"] == "json_schema"
        assert sent["output_config"]["effort"] == "medium"
        assert result.text == '{"findings": []}', "thinking blocks are not part of the answer"
        assert result.input_tokens == 125 and result.cached_input_tokens == 25

    async def test_no_tools_and_no_model_fallback(self):
        messages = FakeMessages(anthropic_message())
        await with_client("anthropic", NS(messages=messages)).complete(model("anthropic"), REQUEST)
        for forbidden in ("tools", "fallbacks", "betas", "tool_choice"):
            assert forbidden not in messages.kwargs

    @pytest.mark.parametrize(
        ("stop", "outcome"), [("refusal", "refused"), ("max_tokens", "truncated")]
    )
    async def test_stop_reasons(self, stop, outcome):
        llm = with_client("anthropic", NS(messages=FakeMessages(anthropic_message(stop))))
        with pytest.raises(ModelDeclined) as caught:
            await llm.complete(model("anthropic"), REQUEST)
        assert caught.value.outcome == outcome


class TestOpenRouter:
    def reply(self, finish="stop", content='{"findings": []}'):
        return NS(
            id="gen_1",
            choices=[NS(finish_reason=finish, message=NS(content=content, refusal=None))],
            usage=NS(prompt_tokens=10, completion_tokens=5),
        )

    async def test_request_shape(self):
        completions = Recorder(self.reply())
        llm = with_client("openrouter", NS(chat=NS(completions=completions)))
        result = await llm.complete(model("openrouter"), REQUEST)
        assert completions.kwargs["response_format"]["type"] == "json_schema"
        assert completions.kwargs["messages"][0] == {"role": "system", "content": "sys"}
        assert "tools" not in completions.kwargs
        assert result.input_tokens == 10

    async def test_length_is_truncation(self):
        llm = with_client("openrouter", NS(chat=NS(completions=Recorder(self.reply("length")))))
        with pytest.raises(ModelDeclined) as caught:
            await llm.complete(model("openrouter"), REQUEST)
        assert caught.value.outcome == "truncated"


class TestSafety:
    async def test_provider_errors_never_carry_the_message_text(self):
        class FailingResponses:
            async def create(self, **_):
                error = RuntimeError("Incorrect API key provided: sk-secret-1234")
                error.status_code = 401
                raise error

        llm = with_client("openai", NS(responses=FailingResponses()))
        with pytest.raises(ProviderError) as caught:
            await llm.complete(model(), REQUEST)
        assert "sk-secret" not in str(caught.value)
        assert caught.value.__cause__ is None
        assert "401" in str(caught.value)

    async def test_missing_key_is_a_clear_error(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY is not set"):
            await ProviderLLM().complete(model("anthropic"), REQUEST)


class TestRegistry:
    def test_every_entry_loads_and_names_its_source(self):
        registry = load_registry()
        assert len(registry) >= 10
        assert all(m.source.startswith("https://") for m in registry.values())

    def test_unseen_needs_a_known_cutoff(self):
        from datetime import date

        assert get_model("gpt-6.1-terra").is_unseen(date(2026, 7, 1)) is True
        assert get_model("gpt-6.1-terra").is_unseen(date(2026, 6, 1)) is False, (
            "on the day is not after"
        )
        assert get_model("gemini-4-argon").is_unseen(date(2030, 1, 1)) is False

    def test_unknown_model_is_refused(self):
        with pytest.raises(KeyError, match="models.toml"):
            get_model("gpt-unknown")

    def test_openrouter_route_is_never_assumed_unseen(self):
        from datetime import date

        route = get_model("openrouter/some/model")
        assert route.provider == "openrouter" and route.name == "some/model"
        assert route.is_unseen(date(2030, 1, 1)) is False

    def test_cost_uses_cached_price(self):
        cost = get_model("gpt-6.1-terra").cost_usd(1_000_000, 1_000_000, cached_tokens=1_000_000)
        assert cost == pytest.approx(0.05 + 5.00)
