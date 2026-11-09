"""`orh`: freeze a sample, estimate its cost, run the arms, judge them and report."""

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

from review_harness import framing, report, runner
from review_harness import judge as judging
from review_harness import manifest as manifests
from review_harness.ledger import Budget, Caller, Ledger
from review_harness.llm import ProviderLLM, get_model, load_registry
from review_harness.papers import ArxivClient
from review_harness.review import ARMS

DEFAULT_CACHE = Path(".cache/papers")
DEFAULT_RUNS = Path("runs")


def _names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_dir(manifest: dict) -> Path:
    return DEFAULT_RUNS / manifest["name"]


def cmd_models(_: argparse.Namespace) -> int:
    print(f"{'model':<24}{'provider':<12}{'cutoff':<13}{'$/M in':>8}{'$/M out':>9}  price")
    for name, model in load_registry().items():
        print(
            f"{name:<24}{model.provider:<12}{str(model.knowledge_cutoff or 'unknown'):<13}"
            f"{model.input_per_mtok:>8.2f}{model.output_per_mtok:>9.2f}  "
            f"{'checked' if model.price_verified else 'NOT CHECKED'}"
        )
    return 0


async def _sample(args: argparse.Namespace) -> int:
    since, until = date.fromisoformat(args.since), date.fromisoformat(args.until)
    arxiv = ArxivClient()
    try:
        print(f"Finding withdrawals after {since}, up to {until} ({args.source})...", flush=True)
        if args.source == "withdrarxiv":
            entries = await manifests.harvest_withdrarxiv(arxiv, since, until)
            description = "alxfgh/withdrarxiv-factual-errors, dated through the arXiv API"
        else:
            entries = await manifests.harvest_recent(arxiv, since, until)
            description = "arXiv API, comments mentioning a withdrawal"
        print(f"{len(entries)} found. Filtering and downloading...", flush=True)
        manifest = await manifests.build_manifest(
            arxiv,
            name=args.name,
            entries=entries,
            source_description=description,
            since=since,
            until=until,
            target=args.n,
            target_external=args.n_external,
            seed=args.seed,
            cache_dir=args.cache,
            manual_exclusions=manifests.load_manual_exclusions(args.exclusions),
            with_controls=not args.no_controls,
        )
    finally:
        await arxiv.aclose()
    path = args.out or Path("manifests") / f"{args.name}.json"
    manifests.write_manifest(manifest, path)
    print(json.dumps(manifest["counts"], indent=1))
    print(f"Wrote {path}, sha256 {manifests.manifest_sha(path)}")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    manifest = manifests.load_manifest(args.manifest)
    estimate = runner.estimate_cost(
        manifest,
        [get_model(n) for n in _names(args.models)],
        _names(args.arms),
        [get_model(n) for n in _names(args.judges)],
        args.include_seen,
    )
    print(json.dumps(estimate, indent=1))
    return 0


def _merge_reviews(path: Path, new: list) -> list:
    """Keep earlier reviews of other arms or models, so separate runs add up instead of replacing."""
    merged = {}
    if path.exists():
        for record in runner.read_reviews(path):
            merged[(record.model, record.arm, record.paper)] = record
    for record in new:
        merged[(record.model, record.arm, record.paper)] = record
    return list(merged.values())


async def _run(args: argparse.Namespace) -> int:
    manifest = manifests.load_manifest(args.manifest)
    runner.check_prompts_frozen(manifest)
    models = [get_model(n) for n in _names(args.models)]
    arms = _names(args.arms)
    for arm in arms:
        if arm not in runner.ALL_ARMS:
            print(f"Unknown arm {arm}. Choose from {', '.join(runner.ALL_ARMS)}.", file=sys.stderr)
            return 2
    try:
        amendment = args.amendment or args.manifest.with_name(f"{args.manifest.stem}.framing.json")
        framing.check_amendment(args.manifest, amendment, arms)
    except framing.AmendmentMismatch as error:
        print(error, file=sys.stderr)
        return 2
    arxiv = ArxivClient()
    try:
        papers, problems = await runner.load_papers(manifest, arxiv, args.cache)
    finally:
        await arxiv.aclose()
    for problem in problems:
        print(f"Skipping {problem}", file=sys.stderr)

    folder = _run_dir(manifest)
    ledger = Ledger(folder / "ledger.jsonl")
    budget = Budget(args.max_usd)
    plan = runner.planned_reviews(manifest, models, arms, args.include_seen)
    print(f"{len(plan)} reviews planned, limit ${args.max_usd:.2f}.", flush=True)
    records, failures, stopped = await runner.run_reviews(
        plan, papers, Caller(ProviderLLM(), ledger, budget), args.concurrency
    )
    runner.write_jsonl(folder / "reviews.jsonl", _merge_reviews(folder / "reviews.jsonl", records))
    for failure in failures:
        print(f"Failed: {failure}", file=sys.stderr)
    print(
        f"{len(records)} reviews saved. New spend ${budget.spent:.2f}, total ${ledger.total_cost():.2f} "
        f"(${ledger.api_equivalent_cost():.2f} at API prices, subscription runs included)."
    )
    if stopped:
        print(stopped, file=sys.stderr)
    return 1 if failures or stopped else 0


async def _judge(args: argparse.Namespace) -> int:
    manifest = manifests.load_manifest(args.manifest)
    folder = _run_dir(manifest)
    records = runner.read_reviews(folder / "reviews.jsonl")
    golds = {p["key"]: p["gold"] for p in manifest["papers"] if p["role"] == "withdrawn"}
    categories = {p["key"]: p["category"] for p in manifest["papers"]}
    ledger = Ledger(folder / "ledger.jsonl")
    budget = Budget(args.max_usd)
    judged = await judging.judge_reviews(
        Caller(ProviderLLM(), ledger, budget),
        [get_model(n) for n in _names(args.judges)],
        records,
        golds,
        categories,
        manifest["seed"],
        args.concurrency,
    )
    runner.write_jsonl(folder / "judgments.jsonl", judged)
    print(f"{len(judged)} judgments saved. New spend ${budget.spent:.2f}.")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    manifest = manifests.load_manifest(args.manifest)
    folder = _run_dir(manifest)
    summary = report.build_summary(
        manifest,
        runner.read_reviews(folder / "reviews.jsonl"),
        runner.read_judged(folder / "judgments.jsonl"),
    )
    out = args.out or Path("results") / manifest["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    markdown = report.render_markdown(summary)
    framed = framing.summarize(manifest, *_report_inputs(folder))
    if framed["models"]:
        summary["framing"] = framed
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        markdown += "\n" + framing.render(framed)
    (out / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


def _report_inputs(folder: Path) -> tuple[list, list]:
    return runner.read_reviews(folder / "reviews.jsonl"), runner.read_judged(
        folder / "judgments.jsonl"
    )


def cmd_amend(args: argparse.Namespace) -> int:
    try:
        path = framing.write_amendment(args.manifest, _names(args.arms), args.note, args.out)
    except framing.AmendmentMismatch as error:
        print(error, file=sys.stderr)
        return 2
    print(f"Wrote {path}. Commit it before running these arms.")
    return 0


def _write_canary_results(results: dict, markdown: str) -> None:
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "canaries.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    (out / "canaries.md").write_text(markdown, encoding="utf-8")


async def _canary(args: argparse.Namespace) -> int:
    ledger = Ledger(DEFAULT_RUNS / "canaries" / "ledger.jsonl")
    budget = Budget(args.max_usd)
    results = await runner.run_canaries(
        [get_model(n) for n in _names(args.models)],
        get_model(args.judge),
        ProviderLLM(),
        ledger,
        budget,
    )
    markdown = runner.render_canaries(results)
    _write_canary_results(results, markdown)
    print(markdown)
    print(f"New spend ${budget.spent:.2f}.")
    return 0 if all(m["passed"] for m in results["models"].values()) else 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="orh", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("models", help="list models, cutoffs and prices").set_defaults(
        run=cmd_models
    )

    sample = commands.add_parser("sample", help="freeze a sample of papers")
    sample.add_argument("--name", required=True)
    sample.add_argument("--since", required=True, help="count withdrawals after this date")
    sample.add_argument("--until", default=date.today().isoformat())
    sample.add_argument("--n", type=int, default=60, help="papers whose flaw should be in the text")
    sample.add_argument(
        "--n-external", type=int, default=15, help="papers that may need code or data"
    )
    sample.add_argument("--seed", type=int, required=True)
    sample.add_argument("--source", choices=("arxiv", "withdrarxiv"), default="arxiv")
    sample.add_argument("--exclusions", type=Path, default=Path("manifests/exclusions.toml"))
    sample.add_argument("--no-controls", action="store_true")
    sample.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    sample.add_argument("--out", type=Path)
    sample.set_defaults(run_async=_sample)

    estimate = commands.add_parser("estimate", help="estimate cost without calling any model")
    estimate.add_argument("--manifest", type=Path, required=True)
    estimate.add_argument("--models", required=True)
    estimate.add_argument("--arms", default=",".join(ARMS), help="framing arms may be added")
    estimate.add_argument("--judges", default="gpt-6.1-sol,gpt-6.1-terra")
    estimate.add_argument("--include-seen", action="store_true")
    estimate.set_defaults(run=cmd_estimate)

    run = commands.add_parser("run", help="run reviews, resuming where a previous run stopped")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--models", required=True)
    run.add_argument("--arms", default=",".join(ARMS))
    run.add_argument("--max-usd", type=float, required=True)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--include-seen", action="store_true")
    run.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    run.add_argument(
        "--amendment", type=Path, help="amendment file, needed for framing arms (see orh amend)"
    )
    run.set_defaults(run_async=_run)

    amend = commands.add_parser(
        "amend", help="pin the framing-arm prompts to a frozen manifest before running them"
    )
    amend.add_argument("--manifest", type=Path, required=True)
    amend.add_argument("--arms", default=",".join(framing.FRAMING_ARMS))
    amend.add_argument("--note", required=True, help="why the amendment is made, for the record")
    amend.add_argument("--out", type=Path)
    amend.set_defaults(run=cmd_amend)

    judge = commands.add_parser("judge", help="score reviews against the authors' stated reasons")
    judge.add_argument("--manifest", type=Path, required=True)
    judge.add_argument("--judges", default="gpt-6.1-sol,gpt-6.1-terra")
    judge.add_argument("--max-usd", type=float, required=True)
    judge.add_argument("--concurrency", type=int, default=4)
    judge.set_defaults(run_async=_judge)

    report_parser = commands.add_parser("report", help="write the aggregate results")
    report_parser.add_argument("--manifest", type=Path, required=True)
    report_parser.add_argument("--out", type=Path)
    report_parser.set_defaults(run=cmd_report)

    canary = commands.add_parser("canary", help="check the harness on planted errors")
    canary.add_argument("--models", required=True)
    canary.add_argument("--judge", default="gpt-6.1-sol")
    canary.add_argument("--max-usd", type=float, required=True)
    canary.set_defaults(run_async=_canary)

    args = parser.parse_args()
    if hasattr(args, "run_async"):
        raise SystemExit(asyncio.run(args.run_async(args)))
    raise SystemExit(args.run(args))


if __name__ == "__main__":
    main()
