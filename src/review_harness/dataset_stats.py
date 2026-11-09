"""Read-only checks of frozen manifests: what was sampled, what was excluded, and what to watch.

Nothing here changes a manifest. It reports the facts a reader needs to judge a sample, including
the ones that weaken it: how many papers were posted after the cutoff, gold statements with no
numbered location, statements that name people, and withdrawals that retract one claim in a
replacement rather than the whole paper.

    uv run python -m review_harness.dataset_stats
"""

import argparse
import asyncio
import csv
import hashlib
import json
import re
import statistics
from collections import Counter
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from review_harness import filters
from review_harness.papers import ArxivClient
from review_harness.rw_arxiv import field_group

# A claim withdrawn inside a replacement version: the paper itself stays up, corrected.
_CLAIM_IN_REPLACEMENT = re.compile(
    r"\b(?:v\d+|version \d+)\b[^.]{0,80}\b(?:correct|replac|revis|narrow|withdraws?)\w*"
    r"|\b(?:major|substantial|substantive) (?:correction|revision)"
    r"|\breplaces the withdrawn\b|\bcomplete rewrite\b|\bnew title\b",
    re.I,
)
_WHOLE_PAPER = re.compile(
    r"\b(?:paper|manuscript|article|preprint|note|submission|this version)\s+(?:is|has been|was|"
    r"is being|needs to be)\s+(?:hereby\s+)?withdrawn|\bwithdrawn by the authors?\b"
    r"|^\s*withdrawn\b|\bwe (?:withdraw|request)",
    re.I,
)
_PERSON = re.compile(
    r"\bDr\.?\s+[A-Z][a-z]+|\[REDACTED-NAME\]|\bby the authors?\s+[A-Z][a-z]+\s+[A-Z][a-z]+"
    r"|\bProf(?:essor)?\.?\s+[A-Z][a-z]+",
)
NEAR_DUPLICATE = 0.8


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def withdrawal_kind(statement: str) -> str:
    if _CLAIM_IN_REPLACEMENT.search(statement) and not _WHOLE_PAPER.search(statement):
        return "claim_in_replacement"
    return "whole_paper"


def _quartiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    values = sorted(values)
    if len(values) == 1:
        q1 = med = q3 = values[0]
    else:
        q1, med, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return {"min": values[0], "q1": q1, "median": med, "q3": q3, "max": values[-1]}


def paper_rows(
    manifest: dict[str, Any], version_dates: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """One row per paper, with the flags the notes discuss. Metadata only, never text.

    `posted` is the date of version 1. The version reviewed can be much newer, so when its own date
    is known (`version_dates`, from arXiv) it is reported too: it decides whether the reviewed text
    could have been in a model's training data."""
    start = date.fromisoformat(manifest["window"]["withdrawn_after"])
    version_dates = version_dates or {}
    rows = []
    for paper in manifest["papers"]:
        gold = paper.get("gold") or ""
        withdrawn = paper["role"] == "withdrawn"
        reviewed = version_dates.get(paper["key"], "")
        rows.append(
            {
                "manifest": manifest["name"],
                "key": paper["key"],
                "role": paper["role"],
                "category": paper["category"],
                "field": field_group(paper["category"]),
                "source": paper["source"],
                "chars": paper["chars"],
                "posted": paper["posted"],
                "withdrawn": paper.get("withdrawn") or "",
                "discoverable": paper.get("discoverable") or "",
                "matched_to": paper.get("matched_to") or "",
                "posted_after_window_start": date.fromisoformat(paper["posted"]) > start,
                "reviewed_version_posted": reviewed,
                "reviewed_version_after_window_start": (date.fromisoformat(reviewed) > start)
                if reviewed
                else "",
                "gold_chars": len(gold) if withdrawn else "",
                "gold_has_numbered_location": bool(filters._LOCATOR.search(gold))
                if withdrawn
                else "",
                "gold_specific_words": len(filters.specific_words(gold)) if withdrawn else "",
                "gold_names_a_person": bool(_PERSON.search(gold)) if withdrawn else "",
                "withdrawal_kind": withdrawal_kind(gold) if withdrawn else "",
                "leak_flags": len(paper.get("leak_flags") or []),
            }
        )
    return rows


def near_duplicates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    golds = [(p["key"], p["gold"]) for p in manifest["papers"] if p["role"] == "withdrawn"]
    pairs = []
    for i, (key_a, gold_a) in enumerate(golds):
        for key_b, gold_b in golds[i + 1 :]:
            ratio = SequenceMatcher(None, gold_a.lower(), gold_b.lower()).ratio()
            if ratio >= NEAR_DUPLICATE:
                pairs.append({"a": key_a, "b": key_b, "similarity": round(ratio, 3)})
    return pairs


def control_checks(manifest: dict[str, Any]) -> dict[str, Any]:
    papers = {p["key"]: p for p in manifest["papers"]}
    withdrawn = [p for p in manifest["papers"] if p["role"] == "withdrawn"]
    controls = [p for p in manifest["papers"] if p["role"] == "control"]
    matched = {c["matched_to"]: c for c in controls if c.get("matched_to")}
    same_category = same_month = 0
    ratios = []
    for control in controls:
        target = papers.get(control.get("matched_to") or "")
        if target is None:
            continue
        same_category += control["category"] == target["category"]
        same_month += control["posted"][:7] == target["posted"][:7]
        ratios.append(control["chars"] / target["chars"])
    return {
        "withdrawn_without_control": [p["key"] for p in withdrawn if p["key"] not in matched],
        "controls_same_category": same_category,
        "controls_same_month": same_month,
        "control_to_withdrawn_length_ratio": _quartiles([round(r, 2) for r in ratios])
        if ratios
        else {},
    }


def manifest_summary(path: Path, version_dates: dict[str, str] | None = None) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = paper_rows(manifest, version_dates)
    withdrawn = [r for r in rows if r["role"] == "withdrawn"]
    controls = [r for r in rows if r["role"] == "control"]
    text = [r for r in withdrawn if r["discoverable"] == "text"]
    exclusions = Counter(item["reason"].split(":")[0] for item in manifest["exclusions"])
    return {
        "manifest": manifest["name"],
        "file": path.as_posix(),
        "sha256": file_sha(path),
        "created_at": manifest["created_at"],
        "window": manifest["window"],
        "filters_sha256": manifest["filters_sha256"],
        "prompts_sha256": manifest["prompts_sha256"],
        "recorded_counts": manifest["counts"],
        "withdrawn": len(withdrawn),
        "controls": len(controls),
        "withdrawn_by_discoverability": dict(Counter(r["discoverable"] for r in withdrawn)),
        "text_source": dict(Counter(r["source"] for r in rows)),
        "withdrawn_by_field": dict(sorted(Counter(r["field"] for r in withdrawn).items())),
        "withdrawn_by_category": dict(sorted(Counter(r["category"] for r in withdrawn).items())),
        "withdrawn_by_month": dict(sorted(Counter(r["withdrawn"][:7] for r in withdrawn).items())),
        "withdrawn_posted_after_window_start": sum(
            r["posted_after_window_start"] for r in withdrawn
        ),
        "text_posted_after_window_start": sum(r["posted_after_window_start"] for r in text),
        "reviewed_version_dates_known": sum(bool(r["reviewed_version_posted"]) for r in withdrawn),
        "withdrawn_reviewed_version_after_window_start": sum(
            r["reviewed_version_after_window_start"] is True for r in withdrawn
        ),
        "text_reviewed_version_after_window_start": sum(
            r["reviewed_version_after_window_start"] is True for r in text
        ),
        "withdrawn_posted_by_year": dict(
            sorted(Counter(r["posted"][:4] for r in withdrawn).items())
        ),
        "chars_withdrawn": _quartiles([r["chars"] for r in withdrawn]),
        "chars_controls": _quartiles([r["chars"] for r in controls]),
        "gold_chars": _quartiles([r["gold_chars"] for r in withdrawn]),
        "gold_without_numbered_location": [
            r["key"] for r in withdrawn if not r["gold_has_numbered_location"]
        ],
        "gold_names_a_person": [r["key"] for r in withdrawn if r["gold_names_a_person"]],
        "claim_in_replacement": [
            r["key"] for r in withdrawn if r["withdrawal_kind"] == "claim_in_replacement"
        ],
        "near_duplicate_golds": near_duplicates(manifest),
        "papers_with_leak_flags": sum(r["leak_flags"] > 0 for r in rows),
        "controls_check": control_checks(manifest),
        "exclusions_by_kind": dict(sorted(exclusions.items())),
    }


async def fetch_version_dates(arxiv: ArxivClient, keys: list[str]) -> dict[str, str]:
    """The submission date of each exact version, from arXiv's API (one request per 100)."""
    found: dict[str, str] = {}
    for start in range(0, len(keys), 100):
        chunk = keys[start : start + 100]
        entries, _ = await arxiv.query(id_list=",".join(chunk), max_results=len(chunk))
        for entry in entries:
            found[f"{entry.arxiv_id}v{entry.version}"] = entry.updated
    return found


def load_version_dates(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["key"]: row["reviewed_version_posted"] for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


async def _fetch(keys: list[str]) -> dict[str, str]:
    arxiv = ArxivClient()
    try:
        return await fetch_version_dates(arxiv, keys)
    finally:
        await arxiv.aclose()


def frozen_manifests(folder: Path) -> list[Path]:
    """Sample manifests only: amendment files such as `*.framing.json` hold no papers."""
    return [
        path
        for path in sorted(folder.glob("*.json"))
        if "papers" in json.loads(path.read_text(encoding="utf-8"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m review_harness.dataset_stats", description=__doc__
    )
    parser.add_argument("--manifest", type=Path, action="append")
    parser.add_argument("--out", type=Path, default=Path("results/dataset"))
    parser.add_argument(
        "--version-dates",
        action="store_true",
        help="look up each reviewed version's own date on arXiv (cached in --out)",
    )
    args = parser.parse_args()
    paths = args.manifest or frozen_manifests(Path("manifests"))
    cache = args.out / "reviewed_version_dates.csv"
    dates = load_version_dates(cache)
    if args.version_dates:
        keys = [
            paper["key"]
            for path in paths
            for paper in json.loads(path.read_text(encoding="utf-8"))["papers"]
            if paper["key"] not in dates
        ]
        if keys:
            dates.update(asyncio.run(_fetch(keys)))
            write_csv(
                cache, [{"key": k, "reviewed_version_posted": v} for k, v in sorted(dates.items())]
            )
    summaries = [manifest_summary(path, dates) for path in paths]
    rows = [
        row
        for path in paths
        for row in paper_rows(json.loads(path.read_text(encoding="utf-8")), dates)
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest_stats.json").write_text(
        json.dumps(summaries, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    write_csv(args.out / "manifest_papers.csv", rows)
    for summary in summaries:
        print(
            f"{summary['manifest']}: {summary['withdrawn']} withdrawn "
            f"({summary['withdrawn_by_discoverability']}), {summary['controls']} controls, "
            f"{summary['withdrawn_posted_after_window_start']} posted after the window start"
        )


if __name__ == "__main__":
    main()
