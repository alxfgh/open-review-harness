"""Keeps a rolling, append-only pool of arXiv withdrawals, so fresh samples can be drawn later.

A paper withdrawn after every model's training cutoff is unseen by construction. Running this
regularly records when each withdrawal was first seen here, which later allows a sample drawn only
from withdrawals that appeared after a given date: a prospective sample nobody could have chosen
with hindsight.

The pool holds arXiv metadata and the rule-based classification only. Paper text is never stored
here, and neither are author names: a later sample looks the authors up again when it needs the
per-author cap.

    uv run python -m review_harness.watch --since 2026-06-30
"""

import argparse
import asyncio
import json
import uuid
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from review_harness import filters
from review_harness.filters import Decision
from review_harness.manifest import filters_sha, harvest_recent
from review_harness.papers import ArxivClient, ArxivEntry

DEFAULT_POOL = Path("pool")
POOL_FILE = "withdrawals.jsonl"
RUNS_FILE = "runs.jsonl"
# arXiv's "updated" date can trail a withdrawal slightly, so a resumed run looks back this far.
OVERLAP_DAYS = 7


@dataclass(frozen=True)
class RunWindow:
    since: date
    until: date


def record_key(arxiv_id: str, version: int) -> str:
    return f"{arxiv_id}v{version}"


def pool_record(entry: ArxivEntry, run_id: str, seen_at: str, rules_sha: str) -> dict[str, Any]:
    """One withdrawal as the pool stores it. The version is the withdrawal version."""
    result = filters.classify(entry.comment, entry.abstract)
    keep = result.decision is Decision.keep
    return {
        "key": record_key(entry.arxiv_id, entry.version),
        "arxiv_id": entry.arxiv_id,
        "version": entry.version,
        "published": entry.published,
        "updated": entry.updated,
        "title": entry.title,
        "category": entry.category,
        "comment": entry.comment,
        "abstract": entry.abstract,
        "decision": result.decision.value,
        "reason": result.reason,
        "discoverable": result.discoverable.value if result.discoverable else None,
        "gold": filters.gold_statement(entry.comment, entry.abstract) if keep else None,
        "has_earlier_version": entry.version >= 2,
        "error_only_in_older_version": filters.misses_reviewed_version(
            entry.comment, entry.version
        ),
        "filters_sha256": rules_sha,
        "first_seen": seen_at,
        "run_id": run_id,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def new_records(existing: Iterable[dict[str, Any]], fresh: Iterable[dict[str, Any]]) -> list[dict]:
    """Records not yet in the pool. A record already there is never rewritten, so its first-seen
    time and the rules it was classified under stay as they were. A new version of a paper that is
    already in the pool is a new record."""
    known = {row["key"] for row in existing}
    out = []
    for row in fresh:
        if row["key"] not in known:
            known.add(row["key"])
            out.append(row)
    return out


def default_window(runs: list[dict[str, Any]], until: date) -> RunWindow | None:
    """Continue from the last run, looking back a few days in case of late updates."""
    if not runs:
        return None
    last = max(date.fromisoformat(run["until"]) for run in runs)
    return RunWindow(min(last - timedelta(days=OVERLAP_DAYS), until), until)


def summarise(rows: list[dict[str, Any]], cutoff: date | None = None) -> dict[str, Any]:
    decisions = Counter(row["decision"] for row in rows)
    kept = [row for row in rows if row["decision"] == Decision.keep.value]
    usable = [
        row for row in kept if row["has_earlier_version"] and not row["error_only_in_older_version"]
    ]
    summary: dict[str, Any] = {
        "withdrawals": len(rows),
        "by_decision": dict(sorted(decisions.items())),
        "specific_error": len(kept),
        "specific_error_with_earlier_version": len(usable),
        "usable_by_discoverability": dict(
            sorted(Counter(r["discoverable"] for r in usable).items())
        ),
    }
    if cutoff is not None:
        summary["usable_withdrawn_after_cutoff"] = sum(
            date.fromisoformat(r["updated"]) > cutoff for r in usable
        )
        summary["usable_posted_after_cutoff"] = sum(
            date.fromisoformat(r["published"]) > cutoff for r in usable
        )
    return summary


async def run_watch(
    arxiv: ArxivClient,
    window: RunWindow,
    pool_dir: Path,
    *,
    now: datetime | None = None,
    cutoff: date | None = None,
) -> dict[str, Any]:
    """Harvest one window, append what is new, and log the run. Returns the run's summary."""
    started = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    run_id = uuid.uuid4().hex[:12]
    rules_sha = filters_sha()
    entries = await harvest_recent(arxiv, window.since, window.until)
    fresh = [
        pool_record(entry, run_id, started, rules_sha)
        for entry in sorted(entries, key=lambda e: (e.arxiv_id, e.version))
    ]
    pool_path = pool_dir / POOL_FILE
    added = new_records(read_jsonl(pool_path), fresh)
    append_jsonl(pool_path, added)
    run = {
        "run_id": run_id,
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "since": window.since.isoformat(),
        "until": window.until.isoformat(),
        "filters_sha256": rules_sha,
        "harvested": len(fresh),
        "added": len(added),
        "already_in_pool": len(fresh) - len(added),
        "window_summary": summarise(fresh, cutoff),
        "added_summary": summarise(added, cutoff),
    }
    append_jsonl(pool_dir / RUNS_FILE, [run])
    return run


async def _main(args: argparse.Namespace) -> int:
    until = date.fromisoformat(args.until) if args.until else date.today()
    if args.since:
        window = RunWindow(date.fromisoformat(args.since), until)
    else:
        window = default_window(read_jsonl(args.pool / RUNS_FILE), until)
        if window is None:
            print("The pool has no runs yet, so --since is needed.")
            return 2
    cutoff = date.fromisoformat(args.cutoff) if args.cutoff else None
    print(
        f"Harvesting withdrawals updated after {window.since}, up to {window.until}...", flush=True
    )
    arxiv = ArxivClient()
    try:
        run = await run_watch(arxiv, window, args.pool, cutoff=cutoff)
    finally:
        await arxiv.aclose()
    print(json.dumps(run, indent=1))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m review_harness.watch", description=__doc__)
    parser.add_argument("--since", help="withdrawals updated after this date (default: resume)")
    parser.add_argument("--until", help="up to and including this date (default: today)")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument(
        "--cutoff",
        default="2026-06-30",
        help="count usable withdrawals after this date (the latest cutoff in models.toml)",
    )
    raise SystemExit(asyncio.run(_main(parser.parse_args())))


if __name__ == "__main__":
    main()
