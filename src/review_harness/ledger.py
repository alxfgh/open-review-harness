"""Every model call goes through here, so runs can resume, costs are capped and results are auditable."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from review_harness.llm import LLM, LLMRequest, LLMResult, ModelDeclined, ModelInfo, ProviderError
from review_harness.schemas import strict_schema

CHARS_PER_TOKEN_ESTIMATE = 3.0


def request_digest(request: LLMRequest) -> str:
    payload = json.dumps(
        [
            request.system,
            request.user,
            request.schema,
            request.schema_name,
            request.max_output_tokens,
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class BudgetExceeded(Exception):
    pass


class Budget:
    """Refuses a call unless the worst case still fits, so a run cannot overspend its limit."""

    def __init__(self, max_usd: float, allow_unpriced: bool = False):
        self.max_usd = max_usd
        self.spent = 0.0
        self.allow_unpriced = allow_unpriced

    def check(self, model: ModelInfo, input_chars: int, max_output_tokens: int) -> None:
        if model.subscription:
            return  # Billed to a CLI sign-in, so there is no API spend to cap.
        if model.input_per_mtok <= 0 and not self.allow_unpriced:
            raise BudgetExceeded(f"No price known for {model.name}, so the cap cannot protect you.")
        worst_case = model.cost_usd(int(input_chars / CHARS_PER_TOKEN_ESTIMATE), max_output_tokens)
        if self.spent + worst_case > self.max_usd:
            raise BudgetExceeded(
                f"Stopped: ${self.spent:.2f} spent, and the next call could cost "
                f"${worst_case:.2f} against a ${self.max_usd:.2f} limit."
            )

    def add(self, cost: float) -> None:
        self.spent += cost


class CallOutcome(BaseModel):
    status: str  # ok, refused, truncated, invalid_output
    key: str
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


class Ledger:
    """Append-only record of model calls. A call already recorded is never paid for twice."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.entries[entry["key"]] = entry

    def append(self, entry: dict[str, Any]) -> None:
        self.entries[entry["key"]] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def total_cost(self) -> float:
        return sum(entry.get("cost_usd", 0.0) for entry in self.entries.values())

    def api_equivalent_cost(self) -> float:
        """What every recorded call would have cost on the provider API, subscription runs included."""
        return sum(
            entry.get("api_equivalent_usd", entry.get("cost_usd", 0.0))
            for entry in self.entries.values()
        )


class Caller:
    def __init__(self, llm: LLM, ledger: Ledger, budget: Budget):
        self.llm = llm
        self.ledger = ledger
        self.budget = budget

    async def _complete(self, key: str, model: ModelInfo, request: LLMRequest) -> dict[str, Any]:
        recorded = self.ledger.entries.get(key)
        if recorded is not None:
            return recorded

        self.budget.check(model, len(request.system) + len(request.user), request.max_output_tokens)
        status = "ok"
        result: LLMResult | None
        try:
            result = await self.llm.complete(model, request)
        except ModelDeclined as declined:
            status, result = declined.outcome, declined.result
        # ProviderError propagates: an outage is not an outcome, so it must not be recorded as one.

        usage = result or LLMResult(text="")
        api_cost = model.cost_usd(
            usage.input_tokens, usage.output_tokens, usage.cached_input_tokens
        )
        cost = 0.0 if model.subscription else api_cost
        self.budget.add(cost)
        entry = {
            "key": key,
            "model": model.name,
            "status": status,
            "text": usage.text,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "cost_usd": cost,
            "api_equivalent_usd": api_cost,
            "response_id": usage.response_id,
            "recorded_at": time.time(),
        }
        self.ledger.append(entry)
        return entry

    async def call[T: BaseModel](
        self,
        key: str,
        model: ModelInfo,
        system: str,
        user: str,
        output_model: type[T],
        max_output_tokens: int,
    ) -> tuple[CallOutcome, T | None]:
        """Make or replay one call. A malformed reply gets one more try, then is recorded as such.

        The ledger key includes a hash of the exact request, so a changed prompt or schema is a new
        call and can never be answered by a stale recorded reply.
        """
        request = LLMRequest(
            system=system,
            user=user,
            schema=strict_schema(output_model),
            schema_name=output_model.__name__,
            max_output_tokens=max_output_tokens,
        )
        key = f"{key}|{request_digest(request)}"
        total_cost = 0.0
        for attempt_key in (key, f"{key}#retry"):
            entry = await self._complete(attempt_key, model, request)
            total_cost += entry["cost_usd"]
            outcome = CallOutcome(
                status=entry["status"],
                key=attempt_key,
                cost_usd=total_cost,
                input_tokens=entry["input_tokens"],
                output_tokens=entry["output_tokens"],
                reasoning_tokens=entry["reasoning_tokens"],
            )
            if entry["status"] != "ok":
                return outcome, None
            try:
                return outcome, output_model.model_validate_json(entry["text"])
            except ValidationError:
                continue
        outcome.status = "invalid_output"
        return outcome, None


__all__ = ["Budget", "BudgetExceeded", "CallOutcome", "Caller", "Ledger", "ProviderError"]
