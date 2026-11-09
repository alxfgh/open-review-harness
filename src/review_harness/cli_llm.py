"""Models run through the Claude Code and Codex command-line tools, on the user's subscription.

This keeps evaluation runs off the paid APIs. The trade-off is that each CLI wraps the model in its own
agent runtime, so a CLI result is a result for "the model inside that tool", and is reported as such.

Two guarantees from `llm.py` still have to hold, and each is enforced here rather than trusted:

- **No tools.** The reviewer must not be able to look anything up, including the withdrawal notice
  or this repository's manifests. Claude Code runs with `--tools ""`, no MCP servers, no user
  settings, hooks or slash commands, and its system prompt replaced. Codex runs in a read-only
  sandbox with the user's config and rules ignored. Both run in a fresh empty directory. Any sign of
  a tool call in the CLI's output fails the call, so it is never recorded as a review.
- **No API billing by accident.** API keys are removed from the child's environment, so each CLI
  uses its own sign-in. A key left in the environment would otherwise switch it to paid API usage.
  Variables set by a host Claude Code session are removed too, so a run started from inside one
  behaves the same as a run from a plain terminal.

Claude Code 2.1 or later is needed (for `--tools`, `--json-schema` and `--effort`), and the Codex
build that ships with the Codex app, which has `--output-schema` and the Windows sandbox.

Paper text goes to the CLI on standard input, never on the command line, which Windows caps at about
32,000 characters.
"""

import asyncio
import glob
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from review_harness.llm import (
    REQUEST_TIMEOUT_SECONDS,
    LLMRequest,
    LLMResult,
    ModelDeclined,
    ModelInfo,
    ProviderError,
)

MIN_CLAUDE_VERSION = (2, 1)

# Removed from the child's environment. API keys would switch a CLI to paid API usage. Anything a
# host Claude Code session sets (its proxy URL, session ids, messaging socket) would tie the child
# to that session instead of the user's own sign-in. CLAUDE_CONFIG_DIR is kept: it says where the
# sign-in lives.
_STRIP_PREFIXES = ("ANTHROPIC_", "CLAUDE")
_KEEP = ("CLAUDE_CONFIG_DIR",)
_STRIP_EXACT = ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL")

# Codex event types that mean the agent did something other than answer.
_CODEX_TOOL_EVENTS = re.compile(
    r"exec_command|command_execution|patch_apply|file_change|mcp_tool_call|web_search|tool_call",
    re.I,
)
# Codex features that give the agent a tool. `web_search="disabled"` and `mcp_servers={}` do not
# reach the built-in `codex_apps` connector, whose `search_service.web_run` searches the web, so it
# is switched off here with the rest (Deviation 3). Names are from `codex features list` (0.167).
CODEX_DISABLED_FEATURES = (
    "apps",
    "plugins",
    "remote_plugin",
    "browser_use",
    "browser_use_external",
    "in_app_browser",
    "computer_use",
    "image_generation",
    "multi_agent",
    "tool_suggest",
    "skill_search",
    "shell_tool",
    "unified_exec",
)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


class CliNotFound(ProviderError):
    pass


def child_env() -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k in _KEEP or not (k.upper().startswith(_STRIP_PREFIXES) or k.upper() in _STRIP_EXACT)
    }
    env["NO_COLOR"] = "1"
    return env


def find_claude() -> list[str]:
    """ORH_CLAUDE_BIN, then the Claude desktop app's own build (newest first), then PATH, then npm."""
    override = os.environ.get("ORH_CLAUDE_BIN")
    candidates = [override] if override else []
    appdata = os.environ.get("APPDATA")
    if appdata:
        found = glob.glob(str(Path(appdata) / "Claude/claude-code/*/*/claude.exe"))
        candidates += sorted(found, key=os.path.getmtime, reverse=True)
    candidates.append(shutil.which("claude") or "")
    if appdata:
        candidates.append(str(Path(appdata) / "npm/node_modules/@anthropic-ai/claude-code/cli.js"))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return _launch(candidate)
    raise CliNotFound("Claude Code not found. Set ORH_CLAUDE_BIN to its executable or cli.js.")


def find_codex() -> list[str]:
    """ORH_CODEX_BIN, then the Codex app's own CLI (newest first), then `codex` on PATH.

    The app's build is preferred because the npm copy may predate `--output-schema` and the Windows
    sandbox, and this adapter relies on both.
    """
    override = os.environ.get("ORH_CODEX_BIN")
    candidates = [override] if override else []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        found = glob.glob(str(Path(local) / "OpenAI/Codex/bin/*/codex.exe"))
        candidates += sorted(found, key=os.path.getmtime, reverse=True)
    candidates.append(shutil.which("codex") or "")
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return _launch(candidate)
    raise CliNotFound("Codex not found. Set ORH_CODEX_BIN to its executable.")


def _launch(path: str) -> list[str]:
    # A .cmd shim would route arguments through cmd.exe, which mangles newlines and quotes.
    if path.endswith((".js", ".mjs")):
        node = shutil.which("node")
        if not node:
            raise CliNotFound("node is needed to run " + path)
        return [node, path]
    if path.lower().endswith((".cmd", ".bat")):
        raise CliNotFound(f"{path} is a cmd.exe shim. Point ORH_*_BIN at the real executable.")
    return [path]


def extract_json(text: str) -> str:
    """The JSON object in a reply, tolerating code fences or a sentence around it."""
    stripped = _FENCE.sub("", text.strip())
    try:
        json.loads(stripped)
        return stripped
    except ValueError:
        pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        candidate = stripped[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except ValueError:
            pass
    return stripped  # Left for the caller's validation to reject and retry.


def parse_claude_output(stdout: str) -> LLMResult:
    """Read `claude -p --output-format json`. Raises if the model used, or tried to use, a tool."""
    try:
        data = json.loads(stdout.strip().splitlines()[-1] if stdout.strip() else "")
    except (ValueError, IndexError):
        raise ProviderError("claude-code: output was not JSON") from None
    if not isinstance(data, dict):
        raise ProviderError("claude-code: unexpected output")
    usage = data.get("usage") or {}
    cached = int(usage.get("cache_read_input_tokens") or 0)
    structured = data.get("structured_output")
    text = (
        json.dumps(structured, ensure_ascii=False)
        if isinstance(structured, dict)
        else extract_json(str(data.get("result") or ""))
    )
    thinking = int((usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
    result = LLMResult(
        text=text,
        input_tokens=int(usage.get("input_tokens") or 0)
        + cached
        + int(usage.get("cache_creation_input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        reasoning_tokens=thinking,
        cached_input_tokens=cached,
        stop=str(data.get("subtype") or "complete"),
        response_id=data.get("session_id"),
    )
    if data.get("permission_denials") or int(data.get("num_turns") or 1) > 1:
        raise ProviderError("claude-code: the model tried to use a tool, so the reply is discarded")
    if data.get("is_error"):
        message = str(data.get("result") or "").lower()
        if "authenticat" in message or "login" in message or "oauth" in message:
            raise ProviderError(
                "claude-code: not signed in. Run `claude auth login` in a terminal."
            )
        if "limit" in message or "overloaded" in message or "rate" in message:
            raise ProviderError("claude-code: usage limit or overload, try again later")
        if "prompt is too long" in message or "context" in message:
            raise ModelDeclined("truncated", result)
        raise ProviderError(f"claude-code: {data.get('subtype') or 'error'}")
    if not result.text:
        raise ModelDeclined("refused", result)
    return result


def parse_codex_events(stdout: str, last_message: str) -> LLMResult:
    """Read `codex exec --json` events. Raises if the agent ran a command or any other tool."""
    tokens = {"input": 0, "cached": 0, "output": 0, "reasoning": 0}
    thread_id = None
    failure = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = str(event.get("type") or (event.get("msg") or {}).get("type") or "")
        item = event.get("item") or {}
        item_kind = str(item.get("type") or item.get("item_type") or "")
        if _CODEX_TOOL_EVENTS.search(kind) or _CODEX_TOOL_EVENTS.search(item_kind):
            raise ProviderError("codex: the agent used a tool, so the reply is discarded")
        if kind == "thread.started":
            thread_id = event.get("thread_id")
        if kind in ("turn.failed", "error"):
            failure = str((event.get("error") or {}).get("message") or event.get("message") or kind)
        usage = event.get("usage") or (event.get("msg") or {}).get("info", {}).get(
            "total_token_usage"
        )
        if isinstance(usage, dict):
            tokens["input"] = int(usage.get("input_tokens") or tokens["input"])
            tokens["cached"] = int(usage.get("cached_input_tokens") or tokens["cached"])
            tokens["output"] = int(usage.get("output_tokens") or tokens["output"])
            tokens["reasoning"] = (
                int(usage.get("reasoning_output_tokens") or usage.get("reasoning_tokens") or 0)
                or tokens["reasoning"]
            )
    result = LLMResult(
        text=extract_json(last_message),
        input_tokens=tokens["input"],
        cached_input_tokens=tokens["cached"],
        output_tokens=tokens["output"] + tokens["reasoning"],
        reasoning_tokens=tokens["reasoning"],
        response_id=thread_id,
    )
    if failure:
        lowered = failure.lower()
        if "context" in lowered or "too long" in lowered:
            raise ModelDeclined("truncated", result)
        raise ProviderError("codex: " + ("usage limit" if "limit" in lowered else "turn failed"))
    if not result.text:
        raise ModelDeclined("refused", result)
    return result


class CliLLM:
    """The `LLM` protocol for the `claude-code` and `codex` providers."""

    def __init__(self, timeout: float = REQUEST_TIMEOUT_SECONDS):
        self.timeout = timeout
        self._claude_checked = False

    async def complete(self, model: ModelInfo, request: LLMRequest) -> LLMResult:
        if model.provider == "claude-code":
            return await self._claude(model, request)
        if model.provider == "codex":
            return await self._codex(model, request)
        raise ProviderError(f"CliLLM cannot run provider {model.provider}")

    async def _run(
        self, command: list[str], stdin: str, cwd: str, extra_env: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env={**child_env(), **(extra_env or {})},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as error:
            raise ProviderError(
                f"could not start {Path(command[-1]).name}: {error.errno}"
            ) from None
        try:
            out, err = await asyncio.wait_for(
                process.communicate(stdin.encode("utf-8")), timeout=self.timeout
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise ProviderError("CLI timed out") from None
        return (
            process.returncode or 0,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
        )

    async def _claude(self, model: ModelInfo, request: LLMRequest) -> LLMResult:
        base = find_claude()
        await self._check_claude_version(base)
        command = [
            *base,
            "-p",
            "--output-format",
            "json",
            "--model",
            model.cli_model or model.name,
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "project",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--system-prompt",
            request.system,
            "--json-schema",
            json.dumps(request.schema),
        ]
        if model.effort:
            command += ["--effort", model.effort]
        with tempfile.TemporaryDirectory(prefix="orh-cc-") as empty:
            code, out, err = await self._run(command, request.user, empty)
        if code != 0 and not out.strip():
            raise ProviderError(f"claude-code exited with code {code}")
        return parse_claude_output(out)

    async def _check_claude_version(self, base: list[str]) -> None:
        if self._claude_checked:
            return
        with tempfile.TemporaryDirectory(prefix="orh-cc-") as empty:
            _, out, _ = await self._run([*base, "--version"], "", empty)
        match = re.match(r"\s*(\d+)\.(\d+)", out)
        if not match or (int(match[1]), int(match[2])) < MIN_CLAUDE_VERSION:
            raise CliNotFound(
                f"Claude Code {'.'.join(map(str, MIN_CLAUDE_VERSION))} or later is needed, found "
                f"{out.strip() or 'unknown'}. Set ORH_CLAUDE_BIN to a newer claude executable."
            )
        self._claude_checked = True

    async def _codex(self, model: ModelInfo, request: LLMRequest) -> LLMResult:
        with tempfile.TemporaryDirectory(prefix="orh-codex-") as empty:
            folder = Path(empty)
            work = folder / "work"
            work.mkdir()
            schema_path = folder / "schema.json"
            schema_path.write_text(json.dumps(request.schema), encoding="utf-8")
            last = folder / "last.txt"
            command = codex_command(find_codex(), model, work, schema_path, last)
            # Codex has no flag to replace its own instructions, so the system prompt leads the message.
            stdin = f"<instructions>\n{request.system}\n</instructions>\n\n{request.user}"
            code, out, err = await self._run(command, stdin, str(work))
            text = last.read_text(encoding="utf-8") if last.exists() else ""
        if code != 0 and not text:
            if "login" in err.lower() or "auth" in err.lower():
                raise ProviderError("codex: not signed in")
            raise ProviderError(f"codex exited with code {code}")
        return parse_codex_events(out, text)


def codex_command(
    base: list[str], model: ModelInfo, work: Path, schema_path: Path, last: Path
) -> list[str]:
    command = [
        *base,
        "exec",
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "-C",
        str(work),
        "-m",
        model.cli_model or model.name,
        "-c",
        'approval_policy="never"',
        "-c",
        "notify=[]",
        "-c",
        "mcp_servers={}",
        "-c",
        'web_search="disabled"',
    ]
    for feature in CODEX_DISABLED_FEATURES:
        command += ["--disable", feature]
    if model.effort:
        command += ["-c", f'model_reasoning_effort="{model.effort}"']
    return [*command, "--output-schema", str(schema_path), "-o", str(last), "-"]


def describe_runtime(model: ModelInfo) -> dict[str, Any]:
    """What a results table needs to say about how a CLI model was run."""
    if model.provider == "claude-code":
        return {
            "runtime": "Claude Code CLI, print mode",
            "tools": 'none (--tools ""), no MCP servers, no user settings, empty working directory',
            "system_prompt": "replaced with the harness prompt",
            "structured_output": "--json-schema",
            "effort": model.effort,
        }
    if model.provider == "codex":
        return {
            "runtime": "Codex CLI exec",
            "tools": (
                "read-only sandbox, user config ignored, no MCP, no web search, built-in apps "
                "connector, shell and other tool features disabled; any tool use voids the call"
            ),
            "system_prompt": "Codex's own instructions kept; harness prompt prepended to the message",
            "reasoning_effort": model.effort,
        }
    return {"runtime": "provider API"}
