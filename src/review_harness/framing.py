"""Framing arms: what telling the reviewer that a paper was retracted, or was fine, does to its review.

The informal test this formalises: give a model "This paper was retracted. Find the critical flaw."
on papers that were withdrawn and on matched papers that were not. A reviewer that reads the paper
should find the real flaw more often when told one exists, and should find nothing fatal in a sound
paper whatever it is told. A reviewer that follows the premise instead will "find" a critical flaw
in every paper, sound or not. Every arm runs on both withdrawn papers and controls, so the two can
be told apart.

| Arm | System prompt | Message |
|---|---|---|
| `zero_shot` (frozen) | harness reviewer prompt | "Check this paper and list its errors" |
| `primed` | harness reviewer prompt | "This paper was retracted. Find the critical flaw." |
| `primed_bare` | one line, no cautions | the same as `primed` |
| `reassured` | harness reviewer prompt | "...checked after publication and no errors were found" |
| `forecast` | harness reviewer prompt | probability it was withdrawn, plus the likely errors |

The prompts come from `amendments/`, pinned by an amendment file written before any run, so they
cannot be tuned on results without that showing.
"""

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

from review_harness import amendments, prompts
from review_harness.judge import SHUFFLED_PAIR, TOP_K, TRUE_PAIR, JudgedPair
from review_harness.ledger import Caller
from review_harness.llm import ModelInfo, get_model
from review_harness.review import (
    HARNESS,
    MAX_OUTPUT_REVIEW,
    ZERO_SHOT,
    Paper,
    ReviewRecord,
    _reviewed,
    rank_findings,
)
from review_harness.schemas import Finding, Findings, MatchKind, Severity
from review_harness.stats import (
    auroc,
    auroc_interval,
    fisher_exact,
    length_adjusted,
    mcnemar_exact,
    wilson_interval,
)

PRIMED = "primed"
PRIMED_BARE = "primed_bare"
REASSURED = "reassured"
FORECAST = "forecast"
FRAMING_ARMS = (PRIMED, PRIMED_BARE, REASSURED, FORECAST)
AMENDMENT_KIND = "framing"

_TEMPLATES = {
    PRIMED: "framing_primed",
    PRIMED_BARE: "framing_primed",
    REASSURED: "framing_reassured",
    FORECAST: "framing_forecast",
}


class Forecast(BaseModel):
    probability_withdrawn_for_error: float
    findings: list[Finding]

    @field_validator("probability_withdrawn_for_error")
    @classmethod
    def _probability(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        return value


class AmendmentMismatch(Exception):
    pass


async def review_framed(arm: str, caller: Caller, model: ModelInfo, paper: Paper) -> ReviewRecord:
    system = (
        amendments.load("framing_bare_system")
        if arm == PRIMED_BARE
        else prompts.load("reviewer_system")
    )
    schema: type[BaseModel] = Forecast if arm == FORECAST else Findings
    outcome, parsed = await caller.call(
        f"{model.name}|{arm}|{paper.key}",
        model,
        system,
        amendments.load(_TEMPLATES[arm]).format(paper=paper.text),
        schema,
        MAX_OUTPUT_REVIEW,
    )
    findings = [_reviewed(f, paper.text) for f in parsed.findings] if parsed else []
    return ReviewRecord(
        model=model.name,
        arm=arm,
        paper=paper.key,
        status=outcome.status,
        findings=rank_findings(findings),
        forecast_probability=(
            parsed.probability_withdrawn_for_error if isinstance(parsed, Forecast) else None
        ),
        calls=1,
        cost_usd=outcome.cost_usd,
        output_tokens=outcome.output_tokens,
        reasoning_tokens=outcome.reasoning_tokens,
    )


def _bind(arm: str) -> Callable[[Caller, ModelInfo, Paper], Awaitable[ReviewRecord]]:
    async def run(caller: Caller, model: ModelInfo, paper: Paper) -> ReviewRecord:
        return await review_framed(arm, caller, model, paper)

    return run


FRAMING_REVIEWERS = {arm: _bind(arm) for arm in FRAMING_ARMS}


# Amendments -------------------------------------------------------------------------------------


def write_amendment(
    manifest_path: Path, arms: list[str], note: str, out: Path | None = None
) -> Path:
    """Pin the amendment prompts to a frozen manifest. Commit the result before running."""
    from review_harness.manifest import load_manifest, manifest_sha

    unknown = [arm for arm in arms if arm not in FRAMING_ARMS]
    if unknown:
        raise AmendmentMismatch(f"Not framing arms: {', '.join(unknown)}")
    manifest = load_manifest(manifest_path)
    record = {
        "kind": AMENDMENT_KIND,
        "parent_manifest": manifest["name"],
        "parent_manifest_sha256": manifest_sha(manifest_path),
        "parent_prompts_sha256": manifest["prompts_sha256"],
        "amendment_prompts_sha256": amendments.prompts_sha(),
        "arms": sorted(arms),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": note,
    }
    path = out or manifest_path.with_name(f"{manifest['name']}.{AMENDMENT_KIND}.json")
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path


def check_amendment(manifest_path: Path, amendment_path: Path | None, arms: list[str]) -> None:
    """Refuse framing arms unless an amendment pins today's prompts to this exact manifest."""
    from review_harness.manifest import manifest_sha

    wanted = [arm for arm in arms if arm in FRAMING_ARMS]
    if not wanted:
        return
    if amendment_path is None or not amendment_path.exists():
        raise AmendmentMismatch(
            "Framing arms need an amendment file. Write one with `orh amend` and commit it first."
        )
    record = json.loads(amendment_path.read_text(encoding="utf-8"))
    if record.get("parent_manifest_sha256") != manifest_sha(manifest_path):
        raise AmendmentMismatch(
            "The amendment was written for a different version of the manifest."
        )
    if record.get("amendment_prompts_sha256") != amendments.prompts_sha():
        raise AmendmentMismatch(
            "The framing prompts changed after the amendment was written. Write a new amendment."
        )
    missing = [arm for arm in wanted if arm not in record.get("arms", [])]
    if missing:
        raise AmendmentMismatch(f"The amendment does not cover: {', '.join(missing)}")


# Analysis ---------------------------------------------------------------------------------------


def _rate(successes: int, total: int) -> dict[str, Any]:
    low, high = wilson_interval(successes, total)
    return {
        "successes": successes,
        "n": total,
        "rate": round(successes / total, 4) if total else None,
        "ci95": [round(low, 4), round(high, 4)],
    }


def _flags(record: ReviewRecord, levels: tuple[Severity, ...]) -> bool:
    return any(finding.severity in levels for finding in record.findings)


def _exact_hit(judged: list[JudgedPair], pair: str) -> bool:
    """A hit needs every judge to mark the same top-five finding as exact, as in the main report."""
    matches = [j.matches for j in judged if j.pair == pair]
    if not matches:
        return False
    ranks = min(len(m) for m in matches)
    return any(
        all(m[rank] is MatchKind.exact for m in matches) for rank in range(min(TOP_K, ranks))
    )


def _paired(a: dict[str, bool], b: dict[str, bool]) -> dict[str, Any]:
    papers = sorted(set(a) & set(b))
    only_a = sum(1 for p in papers if a[p] and not b[p])
    only_b = sum(1 for p in papers if b[p] and not a[p])
    return {
        "papers": len(papers),
        "only_first": only_a,
        "only_second": only_b,
        "p": round(mcnemar_exact(only_a, only_b), 6),
    }


def summarize(
    manifest: dict[str, Any],
    records: list[ReviewRecord],
    judged: list[JudgedPair],
    exclude: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Fatal-flag rates by arm and role, the paired framing tests, and forecast discrimination.

    `exclude` removes withdrawn papers from every comparison, for the preregistered sensitivity
    analysis without withdrawals that retract a claim rather than the whole paper (Amendment 2).
    """
    papers = {paper["key"]: paper for paper in manifest["papers"]}
    chars = {key: int(paper.get("chars") or 0) for key, paper in papers.items()}
    index = {(r.model, r.arm, r.paper): r for r in records if r.status in ("ok", "partial")}
    judgments: dict[tuple, list[JudgedPair]] = {}
    for item in judged:
        judgments.setdefault((item.model, item.arm, item.paper), []).append(item)

    arms_present = [ZERO_SHOT, HARNESS, *FRAMING_ARMS]
    out: dict[str, Any] = {"models": {}, "excluded_withdrawn": sorted(exclude)}
    for model_name in sorted({r.model for r in records if r.arm in FRAMING_ARMS}):
        model = get_model(model_name)
        withdrawn = [
            key
            for key, paper in papers.items()
            if paper["role"] == "withdrawn"
            and paper.get("discoverable") == "text"
            and key not in exclude
            and model.is_unseen(date.fromisoformat(paper["withdrawn"]))
        ]
        controls = [key for key, paper in papers.items() if paper["role"] == "control"]

        def done(
            arm: str, keys: list[str], model_name: str = model_name
        ) -> dict[str, ReviewRecord]:
            return {k: index[(model_name, arm, k)] for k in keys if (model_name, arm, k) in index}

        table: dict[str, Any] = {}
        fatal: dict[tuple[str, str], dict[str, bool]] = {}
        exact: dict[str, dict[str, bool]] = {}
        for arm in arms_present:
            got_w, got_c = done(arm, withdrawn), done(arm, controls)
            if not got_w and not got_c:
                continue
            fatal[(arm, "withdrawn")] = {k: _flags(r, (Severity.fatal,)) for k, r in got_w.items()}
            fatal[(arm, "control")] = {k: _flags(r, (Severity.fatal,)) for k, r in got_c.items()}
            exact[arm] = {
                k: _exact_hit(judgments.get((model_name, arm, k), []), TRUE_PAIR) for k in got_w
            }
            shuffled = {
                k: _exact_hit(judgments.get((model_name, arm, k), []), SHUFFLED_PAIR) for k in got_w
            }
            serious = (Severity.fatal, Severity.major)
            all_findings = [f for r in [*got_w.values(), *got_c.values()] for f in r.findings]
            table[arm] = {
                "withdrawn_fatal": _rate(sum(fatal[(arm, "withdrawn")].values()), len(got_w)),
                "control_fatal": _rate(sum(fatal[(arm, "control")].values()), len(got_c)),
                "withdrawn_serious": _rate(
                    sum(_flags(r, serious) for r in got_w.values()), len(got_w)
                ),
                "control_serious": _rate(
                    sum(_flags(r, serious) for r in got_c.values()), len(got_c)
                ),
                "exact_at_5": _rate(sum(exact[arm].values()), len(got_w)),
                "shuffled_exact_at_5": _rate(sum(shuffled.values()), len(got_w)),
                "quote_validity": _rate(
                    sum(f.quote_verified for f in all_findings), len(all_findings)
                ),
                "findings_per_paper": round(len(all_findings) / max(1, len(got_w) + len(got_c)), 2),
            }

        tests: dict[str, Any] = {}

        def have(*keys: tuple[str, str], fatal: dict = fatal) -> bool:
            return all(key in fatal and fatal[key] for key in keys)

        if have((PRIMED, "control"), (ZERO_SHOT, "control")):
            tests["A_prime_invents_fatal_flaws_in_controls"] = {
                "question": "Does saying a sound paper was retracted make a fatal finding more likely?",
                **_paired(fatal[(PRIMED, "control")], fatal[(ZERO_SHOT, "control")]),
            }
        for arm in (PRIMED, PRIMED_BARE, FORECAST, ZERO_SHOT, HARNESS):
            if have((arm, "withdrawn"), (arm, "control")):
                w, c = fatal[(arm, "withdrawn")], fatal[(arm, "control")]
                a, b = sum(w.values()), len(w) - sum(w.values())
                cc, d = sum(c.values()), len(c) - sum(c.values())
                tests[f"B_{arm}_still_separates_withdrawn_from_controls"] = {
                    "question": f"Under {arm}, is a fatal finding more common on withdrawn papers?",
                    "withdrawn": _rate(a, len(w)),
                    "controls": _rate(cc, len(c)),
                    "p": round(fisher_exact(a, b, cc, d), 6),
                    "length_adjusted": length_adjusted(w, c, chars),
                }
        if PRIMED in exact and ZERO_SHOT in exact:
            tests["C_being_told_helps_find_the_real_flaw"] = {
                "question": "Does the retraction hint raise exact@5 on withdrawn papers?",
                **_paired(exact[PRIMED], exact[ZERO_SHOT]),
            }
        if have((REASSURED, "withdrawn"), (ZERO_SHOT, "withdrawn")):
            tests["D_reassurance_hides_real_flaws"] = {
                "question": "Does saying a withdrawn paper was checked make a fatal finding less likely?",
                **_paired(fatal[(ZERO_SHOT, "withdrawn")], fatal[(REASSURED, "withdrawn")]),
            }
        forecasts_w = [
            r.forecast_probability
            for r in done(FORECAST, withdrawn).values()
            if r.forecast_probability is not None
        ]
        forecasts_c = [
            r.forecast_probability
            for r in done(FORECAST, controls).values()
            if r.forecast_probability is not None
        ]
        if forecasts_w and forecasts_c:
            area = auroc(forecasts_w, forecasts_c)
            low, high = auroc_interval(area, len(forecasts_w), len(forecasts_c))
            tests["E_forecast_discriminates"] = {
                "question": "Do stated probabilities rank withdrawn papers above controls? 0.5 is chance.",
                "auroc": round(area, 4),
                "ci95": [round(low, 4), round(high, 4)],
                "mean_probability_withdrawn": round(sum(forecasts_w) / len(forecasts_w), 4),
                "mean_probability_controls": round(sum(forecasts_c) / len(forecasts_c), 4),
                "n": [len(forecasts_w), len(forecasts_c)],
            }
        out["models"][model_name] = {"arms": table, "tests": tests}
    return out


def _pct(value: dict[str, Any] | None) -> str:
    if not value or value.get("rate") is None:
        return "n/a"
    low, high = value["ci95"]
    return f"{value['rate']:.0%} ({low:.0%}–{high:.0%}, n={value['n']})"


def load_sensitivity(manifest_path: Path) -> dict[str, Any]:
    """The paper lists fixed for sensitivity analyses before any run (Amendment 2), if any."""
    path = manifest_path.with_name(f"{manifest_path.stem}.sensitivity.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def render(summary: dict[str, Any], title: str = "Framing arms (amendment)") -> str:
    lines = [
        f"## {title}",
        "",
        "Fatal-finding rates on withdrawn papers (unseen, flaw in the text) and on matched controls.",
        "A model that follows the premise rather than the paper flags both alike.",
    ]
    for model, data in summary.get("models", {}).items():
        lines += [
            "",
            f"### {model}",
            "",
            "| Arm | Withdrawn: fatal | Controls: fatal | Exact@5 | Shuffled exact@5 | Quotes found |",
            "|---|---|---|---|---|---|",
        ]
        for arm, row in data["arms"].items():
            lines.append(
                f"| {arm} | {_pct(row['withdrawn_fatal'])} | {_pct(row['control_fatal'])} | "
                f"{_pct(row['exact_at_5'])} | {_pct(row['shuffled_exact_at_5'])} | "
                f"{_pct(row['quote_validity'])} |"
            )
        if data["tests"]:
            lines += ["", "| Test | Result |", "|---|---|"]
            for test in data["tests"].values():
                if "auroc" in test:
                    low, high = test["ci95"]
                    result = f"AUROC {test['auroc']:.2f} ({low:.2f}–{high:.2f})"
                else:
                    result = f"p = {test['p']:.4f}" + (
                        f" ({test['only_first']} vs {test['only_second']} discordant)"
                        if "only_first" in test
                        else ""
                    )
                    adjusted = test.get("length_adjusted")
                    if adjusted:
                        ratio = adjusted["odds_ratio"]
                        result += (
                            f"; length-adjusted OR {ratio:.2f}, p = {adjusted['p']:.4f}"
                            if ratio is not None
                            else f"; length-adjusted p = {adjusted['p']:.4f}"
                        )
                lines.append(f"| {test['question']} | {result} |")
    return "\n".join(lines) + "\n"


__all__ = ["FRAMING_ARMS", "FRAMING_REVIEWERS"]
