"""Turns reviews and judgments into the published result. Aggregates only.

Per-paper model output stays local: it is unverified machine text about identifiable people's work.
Every test here was fixed in PREREGISTRATION.md before running, and a failure is reported as a
failure. The judge rule is conservative: a hit needs every judge to agree it is an exact match.
"""

from collections import defaultdict
from datetime import date
from typing import Any

from review_harness.judge import SHUFFLED_PAIR, TOP_K, TRUE_PAIR, JudgedPair
from review_harness.llm import get_model
from review_harness.review import HARNESS, MEMORY, ZERO_SHOT, ReviewRecord
from review_harness.schemas import MatchKind, Severity
from review_harness.stats import cohen_kappa, fisher_exact, mcnemar_exact, wilson_interval

ALPHA_SIGNAL = 0.01
ALPHA = 0.05
MIN_QUOTE_VALIDITY = 0.95


def _hit(matches_by_judge: list[list[MatchKind]], k: int, kind: MatchKind) -> bool:
    """A rank counts only if every judge marked it at this level or stricter."""
    if not matches_by_judge:
        return False
    accepted = (
        {MatchKind.exact} if kind is MatchKind.exact else {MatchKind.exact, MatchKind.related}
    )
    ranks = min(len(m) for m in matches_by_judge)
    return any(
        all(matches[rank] in accepted for matches in matches_by_judge)
        for rank in range(min(k, ranks))
    )


def _rate(successes: int, total: int) -> dict[str, Any]:
    low, high = wilson_interval(successes, total)
    return {
        "successes": successes,
        "n": total,
        "rate": round(successes / total, 4) if total else None,
        "ci95": [round(low, 4), round(high, 4)],
    }


def _paired_p(a: dict[str, bool], b: dict[str, bool]) -> tuple[float, int, int]:
    papers = set(a) & set(b)
    only_a = sum(1 for p in papers if a[p] and not b[p])
    only_b = sum(1 for p in papers if b[p] and not a[p])
    return mcnemar_exact(only_a, only_b), only_a, only_b


def build_summary(
    manifest: dict[str, Any], records: list[ReviewRecord], judged: list[JudgedPair]
) -> dict[str, Any]:
    papers = {paper["key"]: paper for paper in manifest["papers"]}
    by_review: dict[tuple, list[JudgedPair]] = defaultdict(list)
    for item in judged:
        by_review[(item.model, item.arm, item.paper, item.pair)].append(item)

    def hits(model: str, arm: str, paper: str, pair: str, k: int, kind: MatchKind) -> bool:
        return _hit([j.matches for j in by_review.get((model, arm, paper, pair), [])], k, kind)

    record_index = {(r.model, r.arm, r.paper): r for r in records}
    models = sorted({r.model for r in records})
    summary: dict[str, Any] = {
        "manifest": manifest["name"],
        "prompts_sha256": manifest["prompts_sha256"],
        "judges": sorted({j.judge for j in judged}),
        "top_k": TOP_K,
        "models": {},
        "judge_agreement": _judge_agreement(judged),
        "total_cost_usd": round(
            sum(r.cost_usd for r in records) + sum(j.cost_usd for j in judged), 4
        ),
    }

    for model_name in models:
        model = get_model(model_name)
        unseen, seen_or_unknown = [], []
        for key, paper in papers.items():
            if paper["role"] != "withdrawn":
                continue
            is_unseen = model.is_unseen(date.fromisoformat(paper["withdrawn"]))
            (unseen if is_unseen else seen_or_unknown).append(key)
        primary = [k for k in unseen if papers[k]["discoverable"] == "text"]
        strict = [
            k
            for k in primary
            if date.fromisoformat(papers[k]["posted"]) > (model.knowledge_cutoff or date.max)
        ]
        external = [k for k in unseen if papers[k]["discoverable"] == "external"]
        controls = [k for k, p in papers.items() if p["role"] == "control"]

        result: dict[str, Any] = {
            "knowledge_cutoff": str(model.knowledge_cutoff) if model.knowledge_cutoff else None,
            "unseen_papers": len(unseen),
            "not_counted_possibly_seen": len(seen_or_unknown),
            "arms": {},
        }

        for arm in (ZERO_SHOT, HARNESS, MEMORY):
            reviewed = [k for k in primary if (model_name, arm, k) in record_index]
            if not reviewed:
                continue
            arm_records = [record_index[(model_name, arm, k)] for k in reviewed]
            all_findings = [f for r in arm_records for f in r.findings]
            quotes_ok = sum(1 for f in all_findings if f.quote_verified)
            result["arms"][arm] = {
                "exact_at_1": _rate(
                    sum(hits(model_name, arm, k, TRUE_PAIR, 1, MatchKind.exact) for k in reviewed),
                    len(reviewed),
                ),
                "exact_at_3": _rate(
                    sum(hits(model_name, arm, k, TRUE_PAIR, 3, MatchKind.exact) for k in reviewed),
                    len(reviewed),
                ),
                "exact_at_5": _rate(
                    sum(
                        hits(model_name, arm, k, TRUE_PAIR, TOP_K, MatchKind.exact)
                        for k in reviewed
                    ),
                    len(reviewed),
                ),
                "related_at_5": _rate(
                    sum(
                        hits(model_name, arm, k, TRUE_PAIR, TOP_K, MatchKind.related)
                        for k in reviewed
                    ),
                    len(reviewed),
                ),
                "shuffled_exact_at_5": _rate(
                    sum(
                        hits(model_name, arm, k, SHUFFLED_PAIR, TOP_K, MatchKind.exact)
                        for k in reviewed
                    ),
                    len(reviewed),
                ),
                "findings_per_paper": round(len(all_findings) / len(reviewed), 2),
                "quote_validity": round(quotes_ok / len(all_findings), 4) if all_findings else None,
                "not_completed": sum(1 for r in arm_records if r.status not in ("ok", "partial")),
                "cost_usd": round(sum(r.cost_usd for r in arm_records), 4),
            }
            if arm == MEMORY:
                result["arms"][arm]["recognised_rate"] = _rate(
                    sum(1 for r in arm_records if r.recognised), len(arm_records)
                )
            if strict and arm != MEMORY:
                strict_done = [k for k in strict if k in reviewed]
                result["arms"][arm]["exact_at_5_posted_after_cutoff"] = _rate(
                    sum(
                        hits(model_name, arm, k, TRUE_PAIR, TOP_K, MatchKind.exact)
                        for k in strict_done
                    ),
                    len(strict_done),
                )
            done_external = [k for k in external if (model_name, arm, k) in record_index]
            if done_external and arm != MEMORY:
                result["arms"][arm]["exact_at_5_may_need_code_or_data"] = _rate(
                    sum(
                        hits(model_name, arm, k, TRUE_PAIR, TOP_K, MatchKind.exact)
                        for k in done_external
                    ),
                    len(done_external),
                )

        result["tests"] = _tests(model_name, primary, controls, record_index, hits)
        summary["models"][model_name] = result
    return summary


def _judge_agreement(judged: list[JudgedPair]) -> dict[str, Any] | None:
    judges = sorted({j.judge for j in judged})
    if len(judges) != 2:
        return None
    by_pair: dict[tuple, dict[str, JudgedPair]] = defaultdict(dict)
    for item in judged:
        by_pair[(item.model, item.arm, item.paper, item.gold_paper)][item.judge] = item
    pairs: list[tuple[bool, bool]] = []
    for both in by_pair.values():
        if len(both) == 2:
            first, second = both[judges[0]], both[judges[1]]
            for a, b in zip(first.matches, second.matches, strict=False):
                pairs.append((a is MatchKind.exact, b is MatchKind.exact))
    kappa = cohen_kappa(pairs)
    return {
        "judges": judges,
        "findings_compared": len(pairs),
        "cohen_kappa": round(kappa, 3) if kappa is not None else None,
    }


def _tests(model, primary, controls, record_index, hits) -> dict[str, Any]:
    """The preregistered tests, in the same order as PREREGISTRATION.md."""
    tests: dict[str, Any] = {}

    def exact(arm: str, pair: str = TRUE_PAIR) -> dict[str, bool]:
        return {
            k: hits(model, arm, k, pair, TOP_K, MatchKind.exact)
            for k in primary
            if (model, arm, k) in record_index
        }

    harness_true = exact(HARNESS)
    if harness_true:
        p, only_true, only_shuffled = _paired_p(harness_true, exact(HARNESS, SHUFFLED_PAIR))
        tests["1_beats_shuffled_answers"] = {
            "question": "Do findings match a paper's real reason more than another paper's?",
            "p": round(p, 5),
            "only_real": only_true,
            "only_shuffled": only_shuffled,
            "passed": only_true > only_shuffled and p < ALPHA_SIGNAL,
        }
    memory = exact(MEMORY)
    if harness_true and memory:
        p, only_harness, only_memory = _paired_p(harness_true, memory)
        tests["2_beats_memory"] = {
            "question": "Does reading the paper beat knowing only its title?",
            "p": round(p, 5),
            "only_with_paper": only_harness,
            "only_from_title": only_memory,
            "passed": only_harness > only_memory and p < ALPHA,
        }
    zero_shot = exact(ZERO_SHOT)
    if harness_true and zero_shot:
        p, only_harness, only_zero = _paired_p(harness_true, zero_shot)
        tests["3_beats_single_prompt"] = {
            "question": "Does the pipeline beat one prompt with the same model?",
            "p": round(p, 5),
            "only_pipeline": only_harness,
            "only_single_prompt": only_zero,
            "passed": only_harness > only_zero and p < ALPHA,
        }

    def fatal_flagged(keys: list[str]) -> tuple[int, int]:
        done = [
            record_index[(model, HARNESS, k)] for k in keys if (model, HARNESS, k) in record_index
        ]
        flagged = sum(1 for r in done if any(f.severity is Severity.fatal for f in r.findings))
        return flagged, len(done)

    flagged_withdrawn, n_withdrawn = fatal_flagged(primary)
    flagged_control, n_control = fatal_flagged(controls)
    if n_withdrawn and n_control:
        p = fisher_exact(
            flagged_withdrawn,
            n_withdrawn - flagged_withdrawn,
            flagged_control,
            n_control - flagged_control,
        )
        tests["4_flags_withdrawn_more_than_controls"] = {
            "question": "Is a fatal finding more common on withdrawn papers than on matched controls?",
            "withdrawn": _rate(flagged_withdrawn, n_withdrawn),
            "controls": _rate(flagged_control, n_control),
            "p": round(p, 5),
            "passed": (flagged_withdrawn / n_withdrawn) > (flagged_control / n_control)
            and p < ALPHA,
        }

    harness_records = [
        record_index[(model, HARNESS, k)]
        for k in [*primary, *controls]
        if (model, HARNESS, k) in record_index
    ]
    findings = [f for r in harness_records for f in r.findings]
    if findings:
        validity = sum(1 for f in findings if f.quote_verified) / len(findings)
        tests["5_quotes_are_real"] = {
            "question": "Do quoted passages actually appear in the paper?",
            "validity": round(validity, 4),
            "findings": len(findings),
            "passed": validity >= MIN_QUOTE_VALIDITY,
        }
    return tests


def _pct(value: dict[str, Any] | None) -> str:
    if not value or value.get("rate") is None:
        return "not run"
    low, high = value["ci95"]
    return f"{value['rate'] * 100:.1f}% ({low * 100:.0f}-{high * 100:.0f}), {value['successes']}/{value['n']}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Results: {summary['manifest']}",
        "",
        "Generated by `orh report`. Aggregates only; per-paper model output is not published.",
        "A hit needs every judge to mark a finding as the exact flaw the authors stated. "
        "Ranges are 95% intervals.",
        "",
        f"Judges: {', '.join(summary['judges']) or 'none'}. Total cost: ${summary['total_cost_usd']:.2f}.",
    ]
    agreement = summary.get("judge_agreement")
    if agreement:
        lines.append(
            f"Judge agreement: Cohen's kappa {agreement['cohen_kappa']} over "
            f"{agreement['findings_compared']} findings."
        )
    else:
        lines.append(
            "Judge agreement: not measured, because this run did not use exactly two judges."
        )
    lines.append(
        "Human audit of the judge's decisions: **not done** unless stated in this repository's README."
    )
    for name, model in summary["models"].items():
        lines += [
            "",
            f"## {name}",
            "",
            f"Knowledge cutoff {model['knowledge_cutoff'] or 'unknown'}. "
            f"{model['unseen_papers']} papers withdrawn after it; "
            f"{model['not_counted_possibly_seen']} not counted because the model may have seen them.",
            "",
            "| Arm | Exact flaw in top 5 | Top 1 | Against shuffled answers | Findings per paper | Real quotes | Cost |",
            "|---|---|---|---|---|---|---|",
        ]
        labels = {
            HARNESS: "Harness pipeline",
            ZERO_SHOT: "Single prompt",
            MEMORY: "Title only (memory control)",
        }
        for arm in (HARNESS, ZERO_SHOT, MEMORY):
            data = model["arms"].get(arm)
            if not data:
                continue
            quotes = (
                "n/a"
                if arm == MEMORY or data["quote_validity"] is None
                else f"{data['quote_validity'] * 100:.0f}%"
            )
            lines.append(
                f"| {labels[arm]} | {_pct(data['exact_at_5'])} | {_pct(data['exact_at_1'])} | "
                f"{_pct(data['shuffled_exact_at_5'])} | {data['findings_per_paper']} | {quotes} | "
                f"${data['cost_usd']:.2f} |"
            )
        lines += ["", "| Preregistered test | Result | Detail |", "|---|---|---|"]
        for test in model["tests"].values():
            outcome = "Passed" if test["passed"] else "**Did not pass**"
            detail = ", ".join(
                f"{key} {value}"
                for key, value in test.items()
                if key not in ("question", "passed", "withdrawn", "controls")
            )
            if "withdrawn" in test:
                detail += (
                    f"; withdrawn {_pct(test['withdrawn'])}; controls {_pct(test['controls'])}"
                )
            lines.append(f"| {test['question']} | {outcome} | {detail} |")
    return "\n".join(lines) + "\n"
