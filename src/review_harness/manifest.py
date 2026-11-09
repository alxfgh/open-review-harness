"""Builds and freezes the sample of papers before any model sees them.

A manifest pins each paper's exact text by hash, and the filter code and prompts by hash too.
Committing it publicly before running is what makes the results credible: the git history shows the
sample could not have been chosen after seeing which papers the models did well on.
"""

import hashlib
import io
import json
import random
import re
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from review_harness import __version__, filters, prompts
from review_harness.filters import Decision, Discoverability
from review_harness.papers import (
    MAX_TEXT_CHARS,
    MIN_TEXT_CHARS,
    ArxivClient,
    ArxivEntry,
    PaperText,
    PaperUnavailable,
    fetch_paper,
    find_leaks,
)

HARVEST_QUERIES = ("co:withdrawn", "co:withdrawal", "co:retracted", 'abs:"has been withdrawn"')
PAGE_SIZE = 200
MAX_HARVEST = 6000
MAX_PER_FIRST_AUTHOR = 2
_KNOWN_PROBLEM = re.compile(r"withdraw|retract|erratum|corrigendum|error in", re.I)
WITHDRARXIV_URL = (
    "https://huggingface.co/datasets/alxfgh/withdrarxiv-factual-errors/resolve/main/"
    "data/train-00000-of-00001-d5a848e440c55d5a.parquet"
)


@dataclass
class Selected:
    key: str
    arxiv_id: str
    version: int
    role: str  # withdrawn or control
    title: str
    category: str
    posted: str
    source: str
    chars: int
    text_sha256: str
    withdrawal_version: int | None = None
    withdrawn: str | None = None
    discoverable: str | None = None
    gold: str | None = None
    matched_to: str | None = None
    leak_flags: list[str] = field(default_factory=list)


def filters_sha() -> str:
    text = resources.files("review_harness").joinpath("filters.py").read_text(encoding="utf-8")
    return hashlib.sha256(text.replace("\r\n", "\n").encode()).hexdigest()


async def harvest_recent(arxiv: ArxivClient, since: date, until: date) -> list[ArxivEntry]:
    """Collect papers updated in the window whose comments mention a withdrawal, newest first."""
    found: dict[str, ArxivEntry] = {}
    for query in HARVEST_QUERIES:
        start = 0
        while start < MAX_HARVEST:
            entries, _ = await arxiv.query(
                search_query=query,
                sortBy="lastUpdatedDate",
                sortOrder="descending",
                start=start,
                max_results=PAGE_SIZE,
            )
            if not entries:
                break
            for entry in entries:
                updated = date.fromisoformat(entry.updated)
                if since < updated <= until:
                    found.setdefault(entry.arxiv_id, entry)
            if date.fromisoformat(entries[-1].updated) <= since:
                break
            start += PAGE_SIZE
    return list(found.values())


async def harvest_withdrarxiv(
    arxiv: ArxivClient, since: date, until: date, url: str = WITHDRARXIV_URL
) -> list[ArxivEntry]:
    """Use the WithdrarXiv factual-errors dataset, looking up each withdrawal's date on arXiv."""
    import pyarrow.parquet as parquet

    response = await arxiv.client.get(url)
    if response.status_code != 200:
        raise PaperUnavailable(f"dataset download returned {response.status_code}")
    table = parquet.read_table(
        io.BytesIO(response.content), columns=["arxiv_id", "scrubbed_comments"]
    )
    comments = dict(zip(*(table.column(c).to_pylist() for c in table.column_names), strict=True))

    usable = [
        versioned
        for versioned, comment in comments.items()
        if filters.classify(comment).decision is Decision.keep
    ]
    found: list[ArxivEntry] = []
    for start in range(0, len(usable), 100):
        batch = usable[start : start + 100]
        entries, _ = await arxiv.query(id_list=",".join(batch), max_results=len(batch))
        for entry in entries:
            updated = date.fromisoformat(entry.updated)
            if since < updated <= until:
                # The dataset's cleaned comment, since it is what the dataset's categories rest on.
                versioned = f"{entry.arxiv_id}v{entry.version}"
                found.append(
                    ArxivEntry(
                        **{**asdict(entry), "comment": comments.get(versioned, entry.comment)}
                    )
                )
    return found


def load_manual_exclusions(path: Path | None) -> dict[str, str]:
    """Papers a person ruled out before any model ran, with the reason on public record."""
    if path is None or not path.exists():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: item["reason"] for item in data.get("exclude", [])}


def usable_text(paper: PaperText) -> tuple[bool, str]:
    if len(paper.text) < MIN_TEXT_CHARS:
        return False, "text_too_short"
    if len(paper.text) > MAX_TEXT_CHARS:
        return False, "text_too_long"  # Excluded rather than silently truncated.
    if find_leaks(paper.text):
        return False, "text_reveals_error"
    return True, ""


async def find_control(
    arxiv: ArxivClient,
    entry: ArxivEntry,
    used: set[str],
    rng: random.Random,
) -> ArxivEntry | None:
    """A paper in the same category and month, with no sign of a known problem.

    A random day keeps the choice from favouring the start of the month, which an API sorted by date
    would otherwise do. Absence of a notice is all this shows: a control can still contain errors.
    """
    month = entry.published[:7].replace("-", "")
    windows = [f"{month}{rng.randint(1, 28):02d}0000 TO {month}{rng.randint(1, 28):02d}2359"]
    windows[0] = " TO ".join(sorted(windows[0].split(" TO "), key=lambda value: value[:8]))
    windows.append(f"{month}010000 TO {month}312359")
    for window in windows:
        entries, _ = await arxiv.query(
            search_query=f"cat:{entry.category} AND submittedDate:[{window}]",
            start=0,
            max_results=50,
        )
        rng.shuffle(entries)
        for candidate in entries:
            if candidate.arxiv_id in used or candidate.arxiv_id == entry.arxiv_id:
                continue
            if _KNOWN_PROBLEM.search(candidate.comment):
                continue
            return candidate
    return None


async def build_manifest(
    arxiv: ArxivClient,
    *,
    name: str,
    entries: list[ArxivEntry],
    source_description: str,
    since: date,
    until: date,
    target: int,
    target_external: int,
    seed: int,
    cache_dir: Path,
    manual_exclusions: dict[str, str] | None = None,
    with_controls: bool = True,
) -> dict[str, Any]:
    manual_exclusions = manual_exclusions or {}
    rng = random.Random(seed)
    exclusions: list[dict[str, str]] = []
    counts: Counter[str] = Counter()

    def exclude(arxiv_id: str, reason: str) -> None:
        counts[reason] += 1
        exclusions.append({"id": arxiv_id, "reason": reason})

    candidates: list[tuple[ArxivEntry, filters.Classification]] = []
    seen_gold: set[str] = set()
    for entry in sorted(entries, key=lambda e: e.arxiv_id):
        result = filters.classify(entry.comment, entry.abstract)
        if result.decision is not Decision.keep:
            exclude(entry.arxiv_id, f"{result.decision.value}: {result.reason}")
            continue
        if entry.version < 2:
            exclude(entry.arxiv_id, "no_earlier_version")
            continue
        if filters.misses_reviewed_version(entry.comment, entry.version):
            exclude(entry.arxiv_id, "error_located_in_an_older_version")
            continue
        if entry.arxiv_id in manual_exclusions:
            exclude(entry.arxiv_id, f"manual: {manual_exclusions[entry.arxiv_id]}")
            continue
        gold = filters.gold_statement(entry.comment, entry.abstract)
        if gold in seen_gold:
            exclude(entry.arxiv_id, "same_statement_as_another_paper")
            continue
        seen_gold.add(gold)
        candidates.append((entry, result))

    rng.shuffle(candidates)
    selected: list[Selected] = []
    taken = Counter()
    per_author: Counter[str] = Counter()
    matched: list[ArxivEntry] = []
    limits = {Discoverability.text: target, Discoverability.external: target_external}

    for entry, result in candidates:
        kind = result.discoverable
        if taken[kind] >= limits[kind]:
            continue
        # Only used to stop one person's withdrawals dominating. Names are not stored.
        if per_author[entry.first_author] >= MAX_PER_FIRST_AUTHOR:
            exclude(entry.arxiv_id, "author_cap")
            continue
        review_version = entry.version - 1
        try:
            paper = await fetch_paper(arxiv, entry.arxiv_id, review_version, cache_dir)
        except PaperUnavailable as error:
            exclude(entry.arxiv_id, f"unavailable: {error}"[:80])
            continue
        ok, reason = usable_text(paper)
        if not ok:
            exclude(entry.arxiv_id, reason)
            continue
        per_author[entry.first_author] += 1
        taken[kind] += 1
        matched.append(entry)
        selected.append(
            Selected(
                key=f"{entry.arxiv_id}v{review_version}",
                arxiv_id=entry.arxiv_id,
                version=review_version,
                role="withdrawn",
                title=entry.title,
                category=entry.category,
                posted=entry.published,
                source=paper.source,
                chars=len(paper.text),
                text_sha256=paper.sha256,
                withdrawal_version=entry.version,
                withdrawn=entry.updated,
                discoverable=kind.value,
                gold=filters.gold_statement(entry.comment, entry.abstract),
            )
        )

    controls: list[Selected] = []
    if with_controls:
        used = {entry.arxiv_id for entry in matched}
        for entry in matched:
            control = await _select_control(arxiv, entry, used, rng, cache_dir)
            if control is None:
                counts["control_not_found"] += 1
                continue
            used.add(control.arxiv_id)
            controls.append(control)

    return {
        "name": name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "harness_version": __version__,
        "source": source_description,
        "window": {"withdrawn_after": since.isoformat(), "withdrawn_until": until.isoformat()},
        "seed": seed,
        "filters_sha256": filters_sha(),
        "prompts_sha256": prompts.prompts_sha(),
        "counts": {
            "harvested": len(entries),
            "usable_candidates": len(candidates),
            "selected_withdrawn": len(selected),
            "selected_controls": len(controls),
            "excluded": dict(sorted(counts.items())),
        },
        "papers": [asdict(item) for item in selected + controls],
        "exclusions": exclusions,
    }


async def _select_control(
    arxiv: ArxivClient,
    entry: ArxivEntry,
    used: set[str],
    rng: random.Random,
    cache_dir: Path,
) -> Selected | None:
    for _ in range(3):
        try:
            candidate = await find_control(arxiv, entry, used, rng)
        except PaperUnavailable:
            # Counted as control_not_found: a missing control must not abort the whole freeze.
            return None
        if candidate is None:
            continue
        try:
            paper = await fetch_paper(arxiv, candidate.arxiv_id, candidate.version, cache_dir)
        except PaperUnavailable:
            used.add(candidate.arxiv_id)
            continue
        ok, _ = usable_text(paper)
        if not ok:
            used.add(candidate.arxiv_id)
            continue
        return Selected(
            key=f"{candidate.arxiv_id}v{candidate.version}",
            arxiv_id=candidate.arxiv_id,
            version=candidate.version,
            role="control",
            title=candidate.title,
            category=candidate.category,
            posted=candidate.published,
            source=paper.source,
            chars=len(paper.text),
            text_sha256=paper.sha256,
            matched_to=f"{entry.arxiv_id}v{entry.version - 1}",
        )
    return None


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def manifest_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
