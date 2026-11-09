"""Finds journal retractions for error whose paper also has a version on arXiv.

Retraction Watch gives a second source of answer keys beside authors' own arXiv withdrawals: a
journal retraction with a coded reason, and often a preprint whose text can be reviewed. This
module only lists candidates. It freezes nothing, and a later sample must still check each one.

Input is Retraction Watch as the API service already holds it, exported by
`scripts/export_rw_retractions.sh`, so the dataset is not downloaded again. Matching, most reliable
first:

1. The record's own DOI is an arXiv DOI (10.48550/arXiv.*).
2. arXiv lists the journal DOI on the paper: an arXiv API `doi:` search, many DOIs per request.
3. OpenAlex lists an arXiv version under the journal DOI. Its grouping can be wrong, so the arXiv
   title must then resemble Retraction Watch's.
4. Optionally, for retractions after the cutoff: an arXiv title search, with a stricter title match.

A match says only that a preprint exists. It may predate the error or have been corrected after
it, so each candidate records whether the preprint was posted before the retraction.

    uv run python -m review_harness.rw_arxiv
"""

import argparse
import asyncio
import csv
import json
import re
import time
import xml.etree.ElementTree as ET  # noqa: S405
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx

from review_harness import filters
from review_harness.papers import (
    ARXIV_API,
    ATOM,
    USER_AGENT,
    ArxivClient,
    ArxivEntry,
    PaperUnavailable,
    parse_feed,
)

ERROR_REASONS = frozenset(
    {
        "Error in Analyses",
        "Error in Results and/or Conclusions",
        "Error in Data",
        "Error in Methods",
    }
)
# Reasons that point to misconduct or publishing problems rather than an honest error. A record
# with one of these alongside an error reason is kept, but flagged.
_MISCONDUCT = re.compile(
    r"fabricat|falsif|plagiar|paper mill|fake peer review|duplicat|euphemisms|misconduct"
    r"|manipulat|forged|computer-aided|computer-generated|randomly generated|rogue editor"
    r"|citation manipulation|sabotage|hoax",
    re.I,
)
ARXIV_NS_DOI = "{http://arxiv.org/schemas/atom}doi"
OPENALEX_WORKS = "https://api.openalex.org/works"
DOI_BATCH = 40
OPENALEX_BATCH = 50
ID_BATCH = 100
TITLE_MATCH = 0.6  # Same threshold the API service uses for title matches.
TITLE_SEARCH_MATCH = 0.9  # A title search finds lookalikes, so it must match almost exactly.
_ARXIV_URL = re.compile(
    r"arxiv\.org/(?:abs|pdf)/((?:[a-z-]+(?:\.[A-Z]{2})?/\d{7})|\d{4}\.\d{4,5})(?:v\d+)?", re.I
)
_ARXIV_DOI = re.compile(
    r"^10\.48550/arxiv\.((?:[a-z-]+(?:\.[a-z]{2})?/\d{7})|\d{4}\.\d{4,5})(?:v\d+)?$", re.I
)
_UNQUERYABLE_ARXIV = re.compile(r'["\s\\]')
_UNQUERYABLE_OPENALEX = re.compile(r"[|,\s\"]")


@dataclass
class RwRecord:
    record_id: str
    nature: str
    reasons: list[str]
    title: str
    journal: str
    publisher: str
    retraction_date: date | None
    retraction_doi: str | None
    original_date: date | None
    original_doi: str | None

    @property
    def error_reasons(self) -> list[str]:
        return sorted(ERROR_REASONS.intersection(self.reasons))

    @property
    def misconduct_signal(self) -> bool:
        return any(_MISCONDUCT.search(reason) for reason in self.reasons)


@dataclass
class Match:
    entry: ArxivEntry
    method: str
    title_similarity: float


@dataclass
class Progress:
    queried_arxiv_doi: int = 0
    queried_openalex: int = 0
    queried_title: int = 0
    unqueryable: int = 0
    failed_batches: list[str] = field(default_factory=list)
    rejected_title: int = 0


def _date(value: str | None) -> date | None:
    value = (value or "").strip()
    return date.fromisoformat(value[:10]) if value else None


def _doi(value: str | None) -> str | None:
    value = (value or "").strip().lower()
    return value or None


def read_records(path: Path) -> Iterator[RwRecord]:
    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield RwRecord(
                record_id=row["record_id"],
                nature=row["nature"],
                reasons=[r for r in (row.get("reasons") or "").split(";") if r],
                title=row.get("title") or "",
                journal=row.get("journal") or "",
                publisher=row.get("publisher") or "",
                retraction_date=_date(row.get("retraction_date")),
                retraction_doi=_doi(row.get("retraction_doi")),
                original_date=_date(row.get("original_date")),
                original_doi=_doi(row.get("original_doi")),
            )


def is_error_retraction(record: RwRecord) -> bool:
    return record.nature == "Retraction" and bool(record.error_reasons)


def arxiv_id_from_doi(doi: str | None) -> str | None:
    match = _ARXIV_DOI.match(doi or "")
    return match.group(1) if match else None


def arxiv_id_from_url(url: str | None) -> str | None:
    match = _ARXIV_URL.search(url or "")
    return match.group(1) if match else None


def normalise_title(title: str) -> str:
    title = re.sub(r"\$[^$]*\$|\\[a-zA-Z]+|<[^>]+>", " ", title or "")
    return " ".join(re.sub(r"[^0-9a-z]+", " ", title.lower()).split())


def title_similarity(a: str, b: str) -> float:
    a, b = normalise_title(a), normalise_title(b)
    if not a or not b:
        return 0.0
    return round(SequenceMatcher(None, a, b).ratio(), 3)


def parse_feed_with_dois(xml_text: str) -> list[tuple[ArxivEntry, list[str]]]:
    """parse_feed, plus the journal DOIs arXiv lists for each entry."""
    entries, _ = parse_feed(xml_text)
    root = ET.fromstring(xml_text)  # noqa: S314  (arXiv's own API, over HTTPS)
    dois: dict[str, list[str]] = {}
    for node in root.findall(f"{ATOM}entry"):
        versioned = (node.findtext(f"{ATOM}id") or "").split("/abs/")[-1].strip()
        listed = node.findtext(ARXIV_NS_DOI) or ""
        dois[versioned] = [d.lower() for d in listed.split()]
    return [(e, dois.get(f"{e.arxiv_id}v{e.version}", [])) for e in entries]


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def _doi_batch(
    arxiv: ArxivClient, dois: list[str], progress: Progress
) -> dict[str, ArxivEntry]:
    """One OR query for many DOIs. A batch that fails is split, so one odd DOI loses only itself."""
    query = " OR ".join(f'doi:"{doi}"' for doi in dois)
    progress.queried_arxiv_doi += 1
    try:
        response = await arxiv.get(
            ARXIV_API, params={"search_query": query, "max_results": len(dois) * 3}
        )
        if response.status_code != 200:
            raise PaperUnavailable(f"status {response.status_code}")
        parsed = parse_feed_with_dois(response.text)
    except (PaperUnavailable, ET.ParseError) as error:
        if len(dois) == 1:
            progress.failed_batches.append(f"arxiv doi {dois[0]}: {error}"[:120])
            return {}
        middle = len(dois) // 2
        found = await _doi_batch(arxiv, dois[:middle], progress)
        found.update(await _doi_batch(arxiv, dois[middle:], progress))
        return found
    wanted = set(dois)
    found: dict[str, ArxivEntry] = {}
    for entry, listed in parsed:
        for doi in wanted.intersection(listed):
            found.setdefault(doi, entry)
    return found


async def search_arxiv_dois(
    arxiv: ArxivClient, dois: Iterable[str], progress: Progress, batch: int = DOI_BATCH
) -> dict[str, ArxivEntry]:
    queryable = []
    for doi in dict.fromkeys(dois):
        if _UNQUERYABLE_ARXIV.search(doi):
            progress.unqueryable += 1
        else:
            queryable.append(doi)
    found: dict[str, ArxivEntry] = {}
    for chunk in _chunks(queryable, batch):
        found.update(await _doi_batch(arxiv, chunk, progress))
    return found


class OpenAlex:
    """OpenAlex's works endpoint, kept to one request a second."""

    MIN_INTERVAL_SECONDS = 1.0

    def __init__(self, client: httpx.AsyncClient | None = None, min_interval: float | None = None):
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(60.0), headers={"User-Agent": USER_AGENT}
        )
        self.min_interval = self.MIN_INTERVAL_SECONDS if min_interval is None else min_interval
        self._next = 0.0

    async def aclose(self) -> None:
        await self.client.aclose()

    async def works_by_doi(self, dois: list[str]) -> list[dict[str, Any]]:
        now = time.monotonic()
        if self._next > now:
            await asyncio.sleep(self._next - now)
        self._next = time.monotonic() + self.min_interval
        response = await self.client.get(
            OPENALEX_WORKS,
            params={
                "filter": "doi:" + "|".join(dois),
                "select": "doi,title,locations",
                "per-page": max(len(dois), 25),
            },
        )
        response.raise_for_status()
        return response.json().get("results", [])


def arxiv_ids_from_work(work: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for location in work.get("locations") or []:
        for value in (location.get("landing_page_url"), location.get("pdf_url")):
            found = arxiv_id_from_url(value)
            if not found and value and "10.48550" in value:
                found = arxiv_id_from_doi(value.split("doi.org/")[-1])
            if found and found not in ids:
                ids.append(found)
    return ids


@dataclass
class OpenAlexCache:
    """What OpenAlex was asked and what it listed, so a rerun asks only about new DOIs."""

    queried: set[str] = field(default_factory=set)
    found: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None) -> "OpenAlexCache":
        if path is None or not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(set(data["queried"]), data["found"])

    def save(self, path: Path | None) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"queried": sorted(self.queried), "found": dict(sorted(self.found.items()))}
        path.write_text(json.dumps(data, indent=0) + "\n", encoding="utf-8", newline="\n")


async def search_openalex(
    openalex: OpenAlex | None,
    dois: Iterable[str],
    progress: Progress,
    batch: int = OPENALEX_BATCH,
    cache: OpenAlexCache | None = None,
) -> dict[str, list[str]]:
    """Journal DOI to the arXiv ids OpenAlex lists for it. Cached DOIs are not asked again, and
    with no client only the cache is used."""
    cache = cache if cache is not None else OpenAlexCache()
    wanted = list(dict.fromkeys(dois))
    queryable = []
    for doi in wanted:
        if doi in cache.queried:
            continue
        if _UNQUERYABLE_OPENALEX.search(doi):
            progress.unqueryable += 1
        else:
            queryable.append(doi)
    for chunk in _chunks(queryable, batch) if openalex is not None else ():
        progress.queried_openalex += 1
        try:
            works = await openalex.works_by_doi(chunk)
        except (httpx.HTTPError, ValueError) as error:
            progress.failed_batches.append(f"openalex {chunk[0]}..: {type(error).__name__}")
            continue
        cache.queried.update(chunk)
        for work in works:
            doi = _doi((work.get("doi") or "").removeprefix("https://doi.org/"))
            ids = arxiv_ids_from_work(work)
            if doi and ids:
                cache.found[doi] = ids
    return {doi: cache.found[doi] for doi in wanted if doi in cache.found}


async def fetch_entries(
    arxiv: ArxivClient, arxiv_ids: Iterable[str], progress: Progress
) -> dict[str, ArxivEntry]:
    """Current arXiv metadata for each id, in batches."""
    unique = list(dict.fromkeys(arxiv_ids))
    found: dict[str, ArxivEntry] = {}
    for chunk in _chunks(unique, ID_BATCH):
        try:
            entries, _ = await arxiv.query(id_list=",".join(chunk), max_results=len(chunk))
        except (PaperUnavailable, ET.ParseError) as error:
            progress.failed_batches.append(f"arxiv id_list {chunk[0]}..: {error}"[:120])
            continue
        for entry in entries:
            found[entry.arxiv_id] = entry
    return found


async def search_title(
    arxiv: ArxivClient, title: str, progress: Progress
) -> tuple[ArxivEntry, float] | None:
    words = normalise_title(title).split()
    if len(words) < 4:
        return None
    progress.queried_title += 1
    try:
        entries, _ = await arxiv.query(search_query=f'ti:"{" ".join(words)}"', max_results=5)
    except (PaperUnavailable, ET.ParseError) as error:
        progress.failed_batches.append(f"arxiv title {title[:40]}: {error}"[:120])
        return None
    scored = [(entry, title_similarity(title, entry.title)) for entry in entries]
    best = max(scored, key=lambda pair: pair[1], default=None)
    return best if best and best[1] >= TITLE_SEARCH_MATCH else None


async def match_records(
    records: list[RwRecord],
    arxiv: ArxivClient,
    openalex: OpenAlex | None,
    *,
    cutoff: date,
    title_search: str = "unseen",
    openalex_cache: OpenAlexCache | None = None,
) -> tuple[dict[str, Match], Progress]:
    """Record id to its arXiv match, trying each method on what is still unmatched."""
    progress = Progress()
    matches: dict[str, Match] = {}
    by_doi: dict[str, list[RwRecord]] = {}
    for record in records:
        if record.original_doi:
            by_doi.setdefault(record.original_doi, []).append(record)

    direct = {r.record_id: arxiv_id_from_doi(r.original_doi) for r in records}
    direct = {key: value for key, value in direct.items() if value}
    direct_entries = await fetch_entries(arxiv, direct.values(), progress)
    for record in records:
        entry = direct_entries.get(direct.get(record.record_id, ""))
        if entry:
            matches[record.record_id] = Match(
                entry, "rw_arxiv_doi", title_similarity(record.title, entry.title)
            )

    pending = [doi for doi, rs in by_doi.items() if any(r.record_id not in matches for r in rs)]
    for doi, entry in (await search_arxiv_dois(arxiv, pending, progress)).items():
        for record in by_doi[doi]:
            matches.setdefault(
                record.record_id,
                Match(entry, "arxiv_doi", title_similarity(record.title, entry.title)),
            )

    if openalex is not None or openalex_cache is not None:
        pending = [doi for doi, rs in by_doi.items() if any(r.record_id not in matches for r in rs)]
        listed = await search_openalex(openalex, pending, progress, cache=openalex_cache)
        entries = await fetch_entries(arxiv, (i for ids in listed.values() for i in ids), progress)
        for doi, ids in listed.items():
            for record in by_doi[doi]:
                if record.record_id in matches:
                    continue
                best = max(
                    (
                        (entries[i], title_similarity(record.title, entries[i].title))
                        for i in ids
                        if i in entries
                    ),
                    key=lambda pair: pair[1],
                    default=None,
                )
                if best and best[1] >= TITLE_MATCH:
                    matches[record.record_id] = Match(best[0], "openalex", best[1])
                elif best:
                    progress.rejected_title += 1

    if title_search != "none":
        for record in records:
            if record.record_id in matches:
                continue
            unseen = record.retraction_date is not None and record.retraction_date > cutoff
            if title_search == "unseen" and not unseen:
                continue
            found = await search_title(arxiv, record.title, progress)
            if found:
                matches[record.record_id] = Match(found[0], "arxiv_title", found[1])
    return matches, progress


def arxiv_withdrawn(entry: ArxivEntry) -> bool:
    """Whether the authors also withdrew the preprint, by the same rule the sampler uses."""
    opening = " ".join(entry.abstract.split())[:400]
    return bool(filters._WITHDRAWAL.search(entry.comment) or filters._WITHDRAWAL.search(opening))


def candidate_row(record: RwRecord, match: Match, cutoff: date) -> dict[str, Any]:
    entry = match.entry
    posted = date.fromisoformat(entry.published)
    retracted = record.retraction_date
    withdrawn = arxiv_withdrawn(entry)
    arxiv_side = filters.classify(entry.comment, entry.abstract) if withdrawn else None
    return {
        "rw_record_id": record.record_id,
        "title": record.title,
        "journal": record.journal,
        "publisher": record.publisher,
        "retraction_date": retracted.isoformat() if retracted else None,
        "original_date": record.original_date.isoformat() if record.original_date else None,
        "original_doi": record.original_doi,
        "retraction_doi": record.retraction_doi,
        "reasons": record.reasons,
        "error_reasons": record.error_reasons,
        "misconduct_signal": record.misconduct_signal,
        "arxiv_id": entry.arxiv_id,
        "arxiv_latest_version": entry.version,
        "arxiv_published": entry.published,
        "arxiv_updated": entry.updated,
        "arxiv_category": entry.category,
        "arxiv_title": entry.title,
        "arxiv_comment": entry.comment,
        "arxiv_withdrawn": withdrawn,
        "arxiv_withdrawal_decision": arxiv_side.decision.value if arxiv_side else None,
        "match_method": match.method,
        "title_similarity": match.title_similarity,
        "unseen": (retracted > cutoff) if retracted else None,
        "preprint_before_retraction": (posted < retracted) if retracted else None,
    }


def field_group(category: str) -> str:
    head = category.split(".")[0]
    physics = {
        "astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th", "math-ph",
        "nlin", "nucl-ex", "nucl-th", "physics", "quant-ph",
    }  # fmt: skip
    return "physics" if head in physics else head or "unknown"


def summarise(
    error_records: list[RwRecord],
    rows: list[dict[str, Any]],
    progress: Progress,
    cutoff: date,
    dataset_version: str | None,
) -> dict[str, Any]:
    def split(selected: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "unseen": sum(r["unseen"] is True for r in selected),
            "seen": sum(r["unseen"] is False for r in selected),
            "undated": sum(r["unseen"] is None for r in selected),
        }

    honest = [r for r in rows if not r["misconduct_signal"]]
    usable = [r for r in honest if r["preprint_before_retraction"]]
    unseen_records = [r for r in error_records if r.retraction_date and r.retraction_date > cutoff]
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "retraction_watch_version": dataset_version,
        "cutoff": cutoff.isoformat(),
        "error_reasons": sorted(ERROR_REASONS),
        "error_retractions": len(error_records),
        "error_retractions_unseen": len(unseen_records),
        "with_doi": sum(r.original_doi is not None for r in error_records),
        "matched": split(rows),
        "matched_by_method": dict(sorted(Counter(r["match_method"] for r in rows).items())),
        "matched_no_misconduct_signal": split(honest),
        "matched_no_misconduct_preprint_before_retraction": split(usable),
        "matched_weak_title_match_below_0_8": sum(r["title_similarity"] < 0.8 for r in rows),
        "matched_by_field": dict(
            sorted(Counter(field_group(r["arxiv_category"]) for r in rows).items())
        ),
        "matched_also_withdrawn_on_arxiv": sum(r["arxiv_withdrawn"] for r in rows),
        "queries": {
            "arxiv_doi_batches": progress.queried_arxiv_doi,
            "openalex_batches": progress.queried_openalex,
            "arxiv_title_searches": progress.queried_title,
            "unqueryable_dois": progress.unqueryable,
            "openalex_rejected_on_title": progress.rejected_title,
            "failed": progress.failed_batches,
        },
    }


async def _openalex_only(
    records: list[RwRecord], cache: OpenAlexCache, args: argparse.Namespace, cutoff: date
) -> int:
    """Fill the OpenAlex cache without touching arXiv, for when arXiv must be left alone."""
    openalex = OpenAlex()
    progress = Progress()
    dois = [r.original_doi for r in records if r.original_doi]
    try:
        listed = await search_openalex(openalex, dois, progress, cache=cache)
    finally:
        await openalex.aclose()
        cache.save(args.openalex_cache)
    dated = {r.original_doi: r.retraction_date for r in records if r.original_doi}
    unseen = sum(bool(dated[d] and dated[d] > cutoff) for d in listed)
    report = {
        "dois": len(set(dois)),
        "with_arxiv_location": len(listed),
        "with_arxiv_location_unseen": unseen,
        "openalex_batches": progress.queried_openalex,
        "unqueryable": progress.unqueryable,
        "failed": progress.failed_batches,
    }
    print(json.dumps(report, indent=1))
    return 0


async def _main(args: argparse.Namespace) -> int:
    cutoff = date.fromisoformat(args.cutoff)
    records = [r for r in read_records(args.records) if is_error_retraction(r)]
    version_file = args.records.with_name(args.records.name + ".version")
    version = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else None
    cache = None if args.no_openalex else OpenAlexCache.load(args.openalex_cache)
    if args.openalex_only:
        return await _openalex_only(records, cache or OpenAlexCache(), args, cutoff)
    print(f"{len(records)} error retractions in {args.records}. Matching to arXiv...", flush=True)
    arxiv = ArxivClient()
    openalex = None if args.no_openalex else OpenAlex()
    try:
        matches, progress = await match_records(
            records,
            arxiv,
            openalex,
            cutoff=cutoff,
            title_search=args.title_search,
            openalex_cache=cache,
        )
    finally:
        await arxiv.aclose()
        if openalex:
            await openalex.aclose()
        if cache is not None:
            cache.save(args.openalex_cache)
    rows = [
        candidate_row(r, matches[r.record_id], cutoff) for r in records if r.record_id in matches
    ]
    rows.sort(key=lambda row: (row["retraction_date"] or "", row["rw_record_id"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarise(records, rows, progress, cutoff, version)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m review_harness.rw_arxiv", description=__doc__)
    parser.add_argument("--records", type=Path, default=Path(".cache/rw/rw_retractions.csv"))
    parser.add_argument("--out", type=Path, default=Path("pool/rw_arxiv_candidates.jsonl"))
    parser.add_argument(
        "--summary", type=Path, default=Path("results/dataset/rw_arxiv_summary.json")
    )
    parser.add_argument("--cutoff", default="2026-06-30")
    parser.add_argument("--no-openalex", action="store_true")
    parser.add_argument(
        "--openalex-cache", type=Path, default=Path(".cache/rw/openalex_arxiv.json")
    )
    parser.add_argument(
        "--openalex-only", action="store_true", help="only fill the OpenAlex cache; no arXiv calls"
    )
    parser.add_argument("--title-search", choices=("unseen", "all", "none"), default="unseen")
    raise SystemExit(asyncio.run(_main(parser.parse_args())))


if __name__ == "__main__":
    main()
