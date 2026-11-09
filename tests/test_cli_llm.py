"""The CLI backends, tested on recorded output shapes. No CLI is started and nothing is spent."""

import json

import pytest

from review_harness import cli_llm
from review_harness.cli_llm import extract_json, parse_claude_output, parse_codex_events
from review_harness.ledger import Budget, Caller, Ledger
from review_harness.llm import ModelDeclined, ProviderError, get_model
from review_harness.schemas import Findings

from .fakes import FakeLLM, finding


def claude_reply(**fields):
    base = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 1,
        "result": "",
        "session_id": "s1",
        "usage": {
            "input_tokens": 100,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 10,
            "output_tokens": 40,
            "output_tokens_details": {"thinking_tokens": 30},
        },
        "permission_denials": [],
    }
    base.update(fields)
    return json.dumps(base)


def test_extract_json_tolerates_fences_and_prose():
    assert json.loads(extract_json('```json\n{"a": 1}\n```')) == {"a": 1}
    assert json.loads(extract_json('Here it is: {"a": {"b": 2}} done.')) == {"a": {"b": 2}}
    assert extract_json("no json here") == "no json here"


def test_claude_prefers_structured_output_and_counts_tokens():
    result = parse_claude_output(claude_reply(structured_output={"findings": []}, result="x"))
    assert json.loads(result.text) == {"findings": []}
    assert result.input_tokens == 160
    assert result.cached_input_tokens == 50
    assert result.reasoning_tokens == 30


def test_claude_falls_back_to_result_text():
    result = parse_claude_output(claude_reply(result='```json\n{"findings": []}\n```'))
    assert json.loads(result.text) == {"findings": []}


@pytest.mark.parametrize(
    "fields",
    [{"num_turns": 2}, {"permission_denials": [{"tool_name": "Read"}]}],
)
def test_claude_tool_use_voids_the_reply(fields):
    with pytest.raises(ProviderError, match="tool"):
        parse_claude_output(claude_reply(result='{"findings": []}', **fields))


def test_claude_sign_in_failure_is_an_outage_not_a_result():
    reply = claude_reply(is_error=True, result="Failed to authenticate: OAuth session expired")
    with pytest.raises(ProviderError, match="not signed in"):
        parse_claude_output(reply)


def test_claude_empty_answer_is_recorded_as_refused():
    with pytest.raises(ModelDeclined) as caught:
        parse_claude_output(claude_reply(result=""))
    assert caught.value.outcome == "refused"


def codex_events(*events):
    return "\n".join(json.dumps(e) for e in events)


def test_codex_reads_usage_and_last_message():
    out = codex_events(
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}},
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 1000,
                "cached_input_tokens": 400,
                "output_tokens": 50,
                "reasoning_output_tokens": 20,
            },
        },
    )
    result = parse_codex_events(out, '{"findings": []}')
    assert result.response_id == "t1"
    assert result.input_tokens == 1000 and result.cached_input_tokens == 400
    assert result.output_tokens == 70 and result.reasoning_tokens == 20


@pytest.mark.parametrize(
    "kind", ["command_execution", "file_change", "mcp_tool_call", "web_search"]
)
def test_codex_tool_use_voids_the_reply(kind):
    out = codex_events({"type": "item.completed", "item": {"type": kind}})
    with pytest.raises(ProviderError, match="tool"):
        parse_codex_events(out, '{"findings": []}')


def test_codex_web_search_through_the_apps_connector_voids_the_reply():
    # The shape seen on 9 November: a search for the paper's retraction through `codex_apps`.
    call = {
        "type": "mcp_tool_call",
        "server": "codex_apps",
        "tool": "search_service.web_run",
        "status": "in_progress",
    }
    out = codex_events({"type": "item.started", "item": call})
    with pytest.raises(ProviderError, match="tool"):
        parse_codex_events(out, '{"findings": []}')


def test_codex_command_switches_off_every_tool_feature(tmp_path):
    model = get_model("codex/gpt-6.1-terra")
    command = cli_llm.codex_command(
        ["codex"], model, tmp_path, tmp_path / "s.json", tmp_path / "last.txt"
    )
    disabled = {command[i + 1] for i, part in enumerate(command) if part == "--disable"}
    assert {"apps", "shell_tool", "unified_exec", "browser_use", "computer_use"} <= disabled
    assert set(cli_llm.CODEX_DISABLED_FEATURES) == disabled
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert 'web_search="disabled"' in command and "mcp_servers={}" in command
    assert command[-1] == "-"  # the prompt goes in on stdin


def test_codex_usage_limit_is_an_outage():
    out = codex_events({"type": "turn.failed", "error": {"message": "usage limit reached"}})
    with pytest.raises(ProviderError, match="usage limit"):
        parse_codex_events(out, "")


def test_child_env_drops_keys_and_host_session_variables(monkeypatch):
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "CLAUDECODE",
        "CLAUDE_CODE_SESSION_ID",
    ):
        monkeypatch.setenv(name, "secret")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "C:/cfg")
    env = cli_llm.child_env()
    assert not any(k.upper().startswith(("ANTHROPIC_", "CLAUDE_CODE")) for k in env)
    assert "OPENAI_API_KEY" not in env and "CLAUDECODE" not in env
    assert env["CLAUDE_CONFIG_DIR"] == "C:/cfg"
    assert "PATH" in env or "Path" in env


def test_launch_refuses_cmd_shims():
    with pytest.raises(cli_llm.CliNotFound, match="cmd.exe"):
        cli_llm._launch("C:/npm/claude.cmd")


async def test_subscription_calls_cost_nothing_but_record_api_price(tmp_path):
    model = get_model("cc/claude-haiku-5-5")
    assert model.subscription and model.provider == "claude-code"
    ledger = Ledger(tmp_path / "ledger.jsonl")
    budget = Budget(0.0)  # Any API spend at all would be refused.
    llm = FakeLLM(lambda request: {"findings": [finding()]})
    outcome, parsed = await Caller(llm, ledger, budget).call(
        "k", model, "system", "user", Findings, 1000
    )
    assert parsed is not None and outcome.cost_usd == 0.0
    entry = next(iter(ledger.entries.values()))
    assert entry["cost_usd"] == 0.0 and entry["api_equivalent_usd"] > 0
    assert ledger.total_cost() == 0.0 and ledger.api_equivalent_cost() > 0
