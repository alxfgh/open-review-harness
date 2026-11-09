"""Runs the arms over a frozen manifest. Results are written even when a run stops early."""

import asyncio
import json
import tomllib
from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any

from review_harness import prompts
from review_harness.framing import FRAMING_ARMS, FRAMING_REVIEWERS
from review_harness.judge import JudgedPair, judge_one
from review_harness.ledger import Budget, BudgetExceeded, Caller, Ledger, ProviderError
from review_harness.llm import LLM, ModelInfo
from review_harness.papers import ArxivClient, PaperUnavailable, fetch_paper
from review_harness.review import (
    HARNESS,
    MEMORY,
    REVIEWERS,
    ZERO_SHOT,
    Paper,
    ReviewRecord,
    split_parts,
)
from review_harness.schemas import MatchKind, Severity

CHARS_PER_TOKEN = 3.5
VISIBLE_OUTPUT_TOKENS = {
    "zero_shot": 4000,
    "map": 2000,
    "part": 2500,
    "verify": 3000,
    "memory": 300,
}
JUDGE_TOKENS = (1200, 500)


ALL_REVIEWERS = {**REVIEWERS, **FRAMING_REVIEWERS}
ALL_ARMS = tuple(ALL_REVIEWERS)


class ManifestMismatch(Exception):
    pass


def check_prompts_frozen(manifest: dict[str, Any]) -> None:
    """Refuse to run if the prompts changed after the sample was frozen."""
    current = prompts.prompts_sha()
    if manifest["prompts_sha256"] != current:
        raise ManifestMismatch(
            "The prompts have changed since this manifest was frozen. Results would not match what "
            "was preregistered. Freeze a new manifest for the new prompts."
        )


async def load_papers(
    manifest: dict[str, Any], arxiv: ArxivClient, cache_dir: Path
) -> tuple[dict[str, Paper], list[str]]:
    """Load the exact frozen text of each paper, and report any that no longer match."""
    loaded: dict[str, Paper] = {}
    problems: list[str] = []
    for item in manifest["papers"]:
        try:
            text = await fetch_paper(arxiv, item["arxiv_id"], item["version"], cache_dir)
        except PaperUnavailable as error:
            problems.append(f"{item['key']}: {error}")
            continue
        if text.sha256 != item["text_sha256"]:
            problems.append(f"{item['key']}: text differs from the frozen version")
            continue
        loaded[item["key"]] = Paper(
            key=item["key"], title=item["title"], category=item["category"], text=text.text
        )
    return loaded, problems


def planned_reviews(
    manifest: dict[str, Any],
    models: list[ModelInfo],
    arms: list[str],
    include_seen: bool = False,
) -> list[tuple[ModelInfo, str, dict[str, Any]]]:
    """Which reviews to run. A paper a model may have seen is skipped, as it cannot count."""
    plan = []
    for model in models:
        for item in manifest["papers"]:
            for arm in arms:
                if item["role"] == "control" and arm == MEMORY:
                    continue  # There is no stated error for a title to recall.
                if (
                    item["role"] == "withdrawn"
                    and not include_seen
                    and not model.is_unseen(date.fromisoformat(item["withdrawn"]))
                ):
                    continue
                plan.append((model, arm, item))
    return plan


def estimate_cost(
    manifest: dict[str, Any],
    models: list[ModelInfo],
    arms: list[str],
    judges: list[ModelInfo],
    include_seen: bool = False,
) -> dict[str, Any]:
    """A range, because hidden reasoning tokens are billed and cannot be known in advance."""
    low = high = 0.0
    per_model: dict[str, list[float]] = {model.name: [0.0, 0.0] for model in models}
    calls = judged_reviews = 0
    plan = planned_reviews(manifest, models, arms, include_seen)
    for model, arm, item in plan:
        tokens = int(item["chars"] / CHARS_PER_TOKEN)
        steps: list[tuple[int, int]]
        if arm == ZERO_SHOT or arm in FRAMING_ARMS:
            steps = [(tokens, VISIBLE_OUTPUT_TOKENS["zero_shot"])]
        elif arm == MEMORY:
            steps = [(200, VISIBLE_OUTPUT_TOKENS["memory"])]
        else:
            parts = max(1, min(6, item["chars"] // 25_000 + 1))
            per_part = tokens // parts + 2000
            steps = [(tokens, VISIBLE_OUTPUT_TOKENS["map"])]
            steps += [(per_part, VISIBLE_OUTPUT_TOKENS["part"])] * parts
            steps.append((tokens + 1500, VISIBLE_OUTPUT_TOKENS["verify"]))
        reasoning_low, reasoning_high = (2, 4) if model.effort else (1, 1)
        for input_tokens, output_tokens in steps:
            calls += 1
            cost_low = model.cost_usd(input_tokens, output_tokens * reasoning_low)
            cost_high = model.cost_usd(input_tokens, output_tokens * reasoning_high)
            low += cost_low
            high += cost_high
            per_model[model.name][0] += cost_low
            per_model[model.name][1] += cost_high
        if item["role"] == "withdrawn":
            judged_reviews += 1
    judge_low = judge_high = 0.0
    for judge in judges:
        # Every review is judged twice: against its true reason and a shuffled one.
        input_tokens, output_tokens = JUDGE_TOKENS
        judge_low += judged_reviews * 2 * judge.cost_usd(input_tokens, output_tokens * 2)
        judge_high += judged_reviews * 2 * judge.cost_usd(input_tokens, output_tokens * 4)
    return {
        "planned_reviews": len(plan),
        "review_calls": calls,
        "judge_calls": judged_reviews * 2 * len(judges),
        "review_cost_usd": [round(low, 2), round(high, 2)],
        "judge_cost_usd": [round(judge_low, 2), round(judge_high, 2)],
        "total_usd": [round(low + judge_low, 2), round(high + judge_high, 2)],
        "per_model_usd": {name: [round(a, 2), round(b, 2)] for name, (a, b) in per_model.items()},
        "unverified_prices": sorted(m.name for m in [*models, *judges] if not m.price_verified),
    }


async def run_reviews(
    plan: list[tuple[ModelInfo, str, dict[str, Any]]],
    papers: dict[str, Paper],
    caller: Caller,
    concurrency: int = 4,
) -> tuple[list[ReviewRecord], list[str], str | None]:
    """Returns the finished reviews, failures, and a reason if the run stopped early."""
    gate = asyncio.Semaphore(concurrency)
    records: list[ReviewRecord] = []
    failures: list[str] = []
    stop_reason: str | None = None
    stop = asyncio.Event()

    async def one(model: ModelInfo, arm: str, item: dict[str, Any]) -> None:
        nonlocal stop_reason
        paper = papers.get(item["key"])
        if paper is None:
            return
        async with gate:
            if stop.is_set():
                return
            try:
                records.append(await ALL_REVIEWERS[arm](caller, model, paper))
            except BudgetExceeded as error:
                stop_reason = str(error)
                stop.set()
            except ProviderError as error:
                failures.append(f"{model.name} {arm} {item['key']}: {error}")

    await asyncio.gather(*(one(*entry) for entry in plan))
    return records, failures, stop_reason


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def read_reviews(path: Path) -> list[ReviewRecord]:
    return [
        ReviewRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_judged(path: Path) -> list[JudgedPair]:
    return [
        JudgedPair.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass(frozen=True)
class Canary:
    key: str
    title: str
    category: str
    text: str
    gold: str | None
    injection: bool


def load_canaries() -> list[Canary]:
    folder = resources.files("review_harness").joinpath("canaries")
    data = tomllib.loads(folder.joinpath("canaries.toml").read_text(encoding="utf-8"))
    return [
        Canary(
            key=f"canary:{item['file'].removesuffix('.md')}",
            title=item["title"],
            category=item["category"],
            text=folder.joinpath(item["file"]).read_text(encoding="utf-8"),
            gold=item.get("gold"),
            injection=bool(item.get("injection")),
        )
        for item in data["canary"]
    ]


async def run_canaries(
    models: list[ModelInfo],
    judge: ModelInfo,
    llm: LLM,
    ledger: Ledger,
    budget: Budget,
) -> dict[str, Any]:
    """Planted errors must be found, and the error-free paper must not get a serious finding."""
    caller = Caller(llm, ledger, budget)
    results: dict[str, Any] = {
        "judge": judge.name,
        "prompts_sha256": prompts.prompts_sha(),
        "models": {},
    }
    for model in models:
        rows = []
        for canary in load_canaries():
            paper = Paper(canary.key, canary.title, canary.category, canary.text)
            for arm in (ZERO_SHOT, HARNESS):
                record = await REVIEWERS[arm](caller, model, paper)
                row: dict[str, Any] = {
                    "canary": canary.key.removeprefix("canary:"),
                    "arm": arm,
                    "status": record.status,
                    "findings": len(record.findings),
                    "cost_usd": round(record.cost_usd, 4),
                }
                if canary.gold:
                    judged = await judge_one(caller, judge, record, canary.gold, canary.key, "true")
                    row["found_planted_error"] = MatchKind.exact in judged.matches
                    row["passed"] = row["found_planted_error"] and record.status == "ok"
                else:
                    serious = [
                        f for f in record.findings if f.severity in (Severity.fatal, Severity.major)
                    ]
                    row["serious_findings_on_clean_paper"] = len(serious)
                    row["passed"] = not serious and record.status == "ok"
                rows.append(row)
        results["models"][model.name] = {
            "passed": all(row["passed"] for row in rows),
            "rows": rows,
        }
    results["total_cost_usd"] = round(ledger.total_cost(), 4)
    return results


def render_canaries(results: dict[str, Any]) -> str:
    lines = [
        "# Canary results",
        "",
        "Synthetic papers with one planted error each, plus one with none. They test the harness: a "
        "failure here should be investigated before any result on real papers is trusted.",
        "",
        f"Judge: {results['judge']}. Prompts: `{results['prompts_sha256'][:12]}`. "
        f"Cost: ${results['total_cost_usd']:.2f}.",
    ]
    for name, model in results["models"].items():
        lines += [
            "",
            f"## {name}: {'all passed' if model['passed'] else 'NOT ALL PASSED'}",
            "",
            "| Canary | Arm | Result | Findings |",
            "|---|---|---|---|",
        ]
        for row in model["rows"]:
            outcome = "Passed" if row["passed"] else f"**Failed** ({row['status']})"
            lines.append(f"| {row['canary']} | {row['arm']} | {outcome} | {row['findings']} |")
    return "\n".join(lines) + "\n"


__all__ = ["json", "split_parts"]
