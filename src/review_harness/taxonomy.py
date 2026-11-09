"""Sorts the authors' stated errors into a few kinds, by fixed rules, so results can be broken down.

Kinds:
- proof: a proof, derivation or formal argument that fails (a gap, a false lemma, a counterexample);
- calculation: a wrong computed value, equation, sign, factor, normalisation or count;
- data: data, experiments, measurements or evaluation (leakage, confounds, wrong scoring);
- code: a bug in code, an implementation or a simulation;
- citation: a wrong citation or misattributed fact, or a flaw inherited from another paper's result;
- other: modelling, conceptual or interpretation errors, and statements too general to place.

Every kind whose rule matches is listed, and the first match in PRIORITY is the primary kind. The
order puts root causes first: an inherited lemma or a bug in code explains the failed proof or wrong
numbers that follow from it. These are keyword rules over one or two sentences, so they are coarse;
the agreement of an independent labelling with them is reported alongside the tables.

    uv run python -m review_harness.taxonomy
"""

import argparse
import csv
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from review_harness.rw_arxiv import field_group

PRIORITY = ("citation", "code", "proof", "calculation", "data", "other")

RULES: dict[str, re.Pattern[str]] = {
    "citation": re.compile(
        r"\b(?:relies|rely|relied|based)\s+(?:crucially\s+)?on\s+(?:this|that|the|a)\s+"
        r"(?:lemma|theorem|result|argument)"
        r"|\b(?:lemma|theorem|proposition)\s+[\w.]+\s+of\s+[\"“]"
        r"|based on the argument in arxiv"
        r"|\bcitations?\b|misquot|misattribut|\berrors? in the references\b|factual errors?",
        re.I,
    ),
    "code": re.compile(
        r"\b(?:code|codebase|bugs?|implementation|software|glitch|scripts?|prototype)\b"
        r"|signal generation|simulation code|audit of the simulation",
        re.I,
    ),
    "proof": re.compile(
        r"\bproofs?\b|\bproved?\b|\blemmas?\b|\btheorems?\b|\bpropositions?\b|\bprop\.|\bcorollar"
        r"|\bthm\b|\bgaps?\b|counter-?examples?|\barguments?\b|\bderivations?\b|\bderiving\b"
        r"|\bdo(?:es)? not hold\b|\bis false\b|not (?:established|justified|valid)\b"
        r"|\bestimates?\b|\bdefinition \d|implicit function|\bimplication\b|identifiability"
        r"|formal analysis|theoretical (?:arguments?|errors?)",
        re.I,
    ),
    "calculation": re.compile(
        r"miscalculat|\bcalculat|computational error|\balgebraic|\barithmetic|numerical error"
        r"|sign error|factor of|normali[sz]|\bunits\b|dimensional|\btypo|wrong axis"
        r"|computed incorrectly|incorrectly computed|\bcount\b|\beq(?:uation)?s?\.?\s*\(?\d"
        r"|\bformula\b",
        re.I,
    ),
    "data": re.compile(
        r"\bdata\b|datasets?|leakage|look-?ahead|\bexperiment(?:s|al)?\b|measurements?|calibrat"
        r"|evaluation|benchmarks?|\bsampl|stimul|confound|baselines?|train-?test|test set"
        r"|preprocessing|annotations?|reanalysis|re-?run|\breading\b|empirical|scoring"
        r"|\bcontrols\b|artifact",
        re.I,
    ),
}


@dataclass(frozen=True)
class Labels:
    primary: str
    all: tuple[str, ...]


def classify_error(statement: str | None) -> Labels:
    text = " ".join((statement or "").split())
    matched = tuple(kind for kind in PRIORITY[:-1] if RULES[kind].search(text))
    return Labels(matched[0] if matched else "other", matched or ("other",))


def manifest_rows(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for paper in manifest["papers"]:
        if paper["role"] != "withdrawn":
            continue
        rows.append(
            {
                "source": manifest["name"],
                "key": paper["key"],
                "arxiv_id": paper["arxiv_id"],
                "category": paper["category"],
                "discoverable": paper.get("discoverable") or "",
                "gold": paper["gold"],
            }
        )
    return rows


def pool_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["decision"] != "keep":
            continue
        rows.append(
            {
                "source": "pool",
                "key": record["key"],
                "arxiv_id": record["arxiv_id"],
                "category": record["category"],
                "discoverable": record.get("discoverable") or "",
                "gold": record["gold"],
            }
        )
    return rows


def label_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        labels = classify_error(row["gold"])
        out.append(
            {
                **row,
                "field": field_group(row["category"]),
                "primary": labels.primary,
                "labels": ";".join(labels.all),
            }
        )
    return out


DISPLAY_ORDER = ("proof", "calculation", "data", "code", "citation", "other")
PRIMARY_SAMPLE = "recent-2026h2"
SECONDARY_SAMPLE = "withdrarxiv-2023q4-2024q3"


def wide_summary(
    rows: list[dict[str, Any]],
    primary: str = PRIMARY_SAMPLE,
    secondary: str = SECONDARY_SAMPLE,
) -> list[dict[str, Any]]:
    """One row per kind: the primary and secondary samples first, for the paper's table, then the
    same split by discoverability and the whole pool. Ends with a total row."""
    columns = {
        primary: lambda r: r["source"] == primary,
        secondary: lambda r: r["source"] == secondary,
        f"{primary} text": lambda r: r["source"] == primary and r["discoverable"] == "text",
        f"{primary} external": lambda r: r["source"] == primary and r["discoverable"] == "external",
        f"{secondary} text": lambda r: r["source"] == secondary and r["discoverable"] == "text",
        "pool": lambda r: r["source"] == "pool",
    }
    table = []
    for kind in (*DISPLAY_ORDER, "total"):
        item: dict[str, Any] = {"error_kind": kind}
        for name, keep in columns.items():
            item[name] = sum(keep(r) and (kind == "total" or r["primary"] == kind) for r in rows)
        table.append(item)
    return table


def tables(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Counts by source and primary kind, and broken down by field and by discoverability."""
    totals = Counter(row["source"] for row in rows)

    def counted(*keys: str) -> list[dict[str, Any]]:
        counts = Counter(tuple(row[k] for k in ("source", *keys, "primary")) for row in rows)
        result = []
        for values, count in sorted(counts.items()):
            item = dict(zip(("source", *keys, "primary"), values, strict=True))
            item["papers"] = count
            item["share_of_source"] = round(count / totals[values[0]], 3)
            result.append(item)
        return result

    any_label = Counter((row["source"], kind) for row in rows for kind in row["labels"].split(";"))
    return {
        "taxonomy_summary": wide_summary(rows),
        "taxonomy_counts": counted(),
        "taxonomy_by_field": counted("field"),
        "taxonomy_by_discoverability": counted("discoverable"),
        "taxonomy_any_label": [
            {
                "source": source,
                "label": kind,
                "papers": count,
                "share_of_source": round(count / totals[source], 3),
            }
            for (source, kind), count in sorted(any_label.items())
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def frozen_manifests(folder: Path) -> list[Path]:
    """Sample manifests only: amendment files such as `*.framing.json` hold no papers."""
    return [
        path
        for path in sorted(folder.glob("*.json"))
        if "papers" in json.loads(path.read_text(encoding="utf-8"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m review_harness.taxonomy", description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        help="a frozen manifest (repeatable; default: every manifest in manifests/)",
    )
    parser.add_argument("--pool", type=Path, default=Path("pool/withdrawals.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/dataset"))
    args = parser.parse_args()

    manifests = args.manifest or frozen_manifests(Path("manifests"))
    rows: list[dict[str, Any]] = []
    for path in manifests:
        rows += manifest_rows(path)
    if args.pool.exists():
        rows += pool_rows(args.pool)
    labelled = label_rows(rows)
    columns = ("source", "key", "arxiv_id", "category", "field", "discoverable", "primary")
    write_csv(
        args.out / "taxonomy.csv",
        [{k: row[k] for k in (*columns, "labels", "gold")} for row in labelled],
    )
    for name, table in tables(labelled).items():
        write_csv(args.out / f"{name}.csv", table)
    summary = Counter((row["source"], row["primary"]) for row in labelled)
    for (source, kind), count in sorted(summary.items()):
        print(f"{source:<28}{kind:<13}{count:>4}")


if __name__ == "__main__":
    main()
