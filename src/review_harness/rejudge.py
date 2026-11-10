"""Deviation 1's check on the Codex-hosted judges: re-judge a random tenth of the judged pairs
through the provider API, with the same judge models, prompts and rule, and report agreement.

A "unit" is one review scored against one stated reason (its own, or the shuffled one). Units whose
review had no findings were never sent to a judge, so they are not sampled. Each sampled unit is
re-judged by the API version of every Codex judge, so the both-judges exact@5 rule can be applied to
each runtime and compared.
"""

import asyncio
import random
from collections import defaultdict
from typing import Any

from review_harness.judge import TOP_K, JudgedPair, judge_one
from review_harness.ledger import Caller
from review_harness.llm import ModelInfo
from review_harness.report import _hit
from review_harness.review import ReviewRecord
from review_harness.schemas import MatchKind
from review_harness.stats import cohen_kappa

FRACTION = 0.10
CLI_PREFIX = "codex/"

Unit = tuple[str, str, str, str, str]  # model, arm, paper, gold_paper, pair


def api_name(judge: str) -> str:
    """The API model that a Codex-hosted judge stands in for."""
    return judge.removeprefix(CLI_PREFIX)


def unit_of(item: JudgedPair) -> Unit:
    return (item.model, item.arm, item.paper, item.gold_paper, item.pair)


def sample_units(judged: list[JudgedPair], seed: int, fraction: float = FRACTION) -> list[Unit]:
    """A fixed random share of the units every judge scored, the same on every run."""
    judges = {item.judge for item in judged}
    scored: dict[Unit, set[str]] = defaultdict(set)
    for item in judged:
        if item.status == "ok" and item.matches:
            scored[unit_of(item)].add(item.judge)
    units = sorted(unit for unit, by in scored.items() if by == judges)
    count = round(len(units) * fraction)
    return sorted(random.Random(f"rejudge|{seed}").sample(units, count))


async def rejudge(
    caller: Caller,
    judges: list[ModelInfo],
    units: list[Unit],
    records: list[ReviewRecord],
    golds: dict[str, str],
    concurrency: int = 4,
) -> list[JudgedPair]:
    index = {(r.model, r.arm, r.paper): r for r in records}
    tasks = [
        (judge, index[(model, arm, paper)], golds[gold_paper], gold_paper, pair)
        for (model, arm, paper, gold_paper, pair) in units
        for judge in judges
    ]
    gate = asyncio.Semaphore(concurrency)

    async def run(task: tuple) -> JudgedPair:
        async with gate:
            return await judge_one(caller, *task)

    return list(await asyncio.gather(*(run(task) for task in tasks)))


def agreement(original: list[JudgedPair], repeat: list[JudgedPair]) -> dict[str, Any]:
    """How far the API judgments agree with the Codex ones on the same units."""
    first: dict[Unit, dict[str, list[MatchKind]]] = defaultdict(dict)
    for item in original:
        first[unit_of(item)][api_name(item.judge)] = item.matches
    second: dict[Unit, dict[str, list[MatchKind]]] = defaultdict(dict)
    for item in repeat:
        if item.status == "ok":
            second[unit_of(item)][api_name(item.judge)] = item.matches

    judges = sorted({api_name(item.judge) for item in repeat})
    per_judge: dict[str, Any] = {}
    for judge in judges:
        exact_pairs: list[tuple[bool, bool]] = []
        same_label = compared = 0
        for unit, by_judge in second.items():
            if judge not in by_judge or judge not in first.get(unit, {}):
                continue
            for a, b in zip(first[unit][judge], by_judge[judge], strict=False):
                compared += 1
                same_label += a == b
                exact_pairs.append((a is MatchKind.exact, b is MatchKind.exact))
        kappa = cohen_kappa(exact_pairs)
        per_judge[judge] = {
            "findings_compared": compared,
            "same_label": round(same_label / compared, 4) if compared else None,
            "exact_kappa": round(kappa, 3) if kappa is not None else None,
            "exact_codex": sum(a for a, _ in exact_pairs),
            "exact_api": sum(b for _, b in exact_pairs),
        }

    table = {"both": 0, "codex_only": 0, "api_only": 0, "neither": 0}
    shuffled_hits = {"codex": 0, "api": 0}
    hit_pairs: list[tuple[bool, bool]] = []
    units = 0
    for unit, by_judge in second.items():
        before = first.get(unit, {})
        if set(by_judge) != set(judges) or not set(judges) <= set(before):
            continue
        units += 1
        hit_codex = _hit([before[j] for j in judges], TOP_K, MatchKind.exact)
        hit_api = _hit([by_judge[j] for j in judges], TOP_K, MatchKind.exact)
        hit_pairs.append((hit_codex, hit_api))
        key = (
            "both"
            if hit_codex and hit_api
            else "codex_only"
            if hit_codex
            else "api_only"
            if hit_api
            else "neither"
        )
        table[key] += 1
        if unit[4] == "shuffled":
            shuffled_hits["codex"] += hit_codex
            shuffled_hits["api"] += hit_api
    hit_kappa = cohen_kappa(hit_pairs)
    return {
        "judges": judges,
        "units": units,
        "per_judge": per_judge,
        "exact_at_5": {
            **table,
            "agreement": round((table["both"] + table["neither"]) / units, 4) if units else None,
            "kappa": round(hit_kappa, 3) if hit_kappa is not None else None,
            "shuffled_hits": shuffled_hits,
        },
    }


def render(result: dict[str, Any], sampled: int, total: int) -> str:
    lines = [
        "## Judges: Codex against the API (Deviation 1)",
        "",
        f"A random {sampled} of the {total} judged units (one review scored against one stated",
        "reason) were judged again through the provider API by the same models, prompts and rule.",
        "",
        "| Judge | Findings compared | Same label | Kappa on exact | Exact (Codex) | Exact (API) |",
        "|---|---|---|---|---|---|",
    ]
    for judge, row in result["per_judge"].items():
        same = f"{row['same_label']:.0%}" if row["same_label"] is not None else "n/a"
        lines.append(
            f"| {judge} | {row['findings_compared']} | {same} | {row['exact_kappa']} | "
            f"{row['exact_codex']} | {row['exact_api']} |"
        )
    hits = result["exact_at_5"]
    agree = f"{hits['agreement']:.0%}" if hits["agreement"] is not None else "n/a"
    lines += [
        "",
        f"Exact@5 under the both-judges rule, over {result['units']} units: both runtimes hit "
        f"{hits['both']}, Codex only {hits['codex_only']}, API only {hits['api_only']}, neither "
        f"{hits['neither']}. Agreement {agree}, kappa {hits['kappa']}. Hits on shuffled reasons: "
        f"Codex {hits['shuffled_hits']['codex']}, API {hits['shuffled_hits']['api']}.",
        "",
    ]
    return "\n".join(lines)
