"""The review arms: a single prompt, the harness pipeline, and the memory control.

The pipeline follows what the published evidence supports. A read of the whole paper comes first,
because removing global context hurts cross-section errors. Sections are then reviewed in order with
shared notes. A verifier in a fresh context tries to refute each candidate, and refuted candidates
are kept on record but not put forward.
"""

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel

from review_harness import prompts
from review_harness.ledger import Caller
from review_harness.llm import ModelInfo
from review_harness.schemas import (
    Finding,
    Findings,
    MemoryAnswer,
    PaperMap,
    SectionReview,
    Severity,
    VerdictKind,
    Verification,
)

ZERO_SHOT = "zero_shot"
HARNESS = "harness"
MEMORY = "memory"
ARMS = (ZERO_SHOT, HARNESS, MEMORY)

MAX_OUTPUT_REVIEW = 24_000
MAX_OUTPUT_STEP = 16_000
MAX_PARTS = 6
MAX_PART_CHARS = 60_000
MIN_PART_CHARS = 6_000
MAX_CANDIDATES = 12

_SECTION_START = re.compile(
    r"^[ \t]*\\section\*?\s*[\[{]|^#{1,2}[ \t]+\S|^(?:\d{1,2}\.?|[IVX]{1,5}\.)[ \t]+[A-Z][^\n]{2,80}$",
    re.MULTILINE,
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SEVERITY_RANK = {Severity.fatal: 0, Severity.major: 1, Severity.minor: 2}
_VERDICT_RANK = {VerdictKind.upheld: 0, VerdictKind.uncertain: 1, VerdictKind.refuted: 2}


@dataclass(frozen=True)
class Paper:
    key: str
    title: str
    category: str
    text: str


class ReviewedFinding(BaseModel):
    title: str
    severity: Severity
    location: str
    evidence_quote: str
    explanation: str
    quote_verified: bool
    verdict: VerdictKind | None = None
    verdict_reason: str | None = None


class ReviewRecord(BaseModel):
    model: str
    arm: str
    paper: str
    status: str  # ok, partial, refused, truncated, invalid_output
    findings: list[ReviewedFinding]
    refuted: list[ReviewedFinding] = []
    recognised: bool | None = None
    known_withdrawn: bool | None = None
    forecast_probability: float | None = None  # Only the forecast framing arm sets this.
    calls: int = 0
    cost_usd: float = 0.0
    output_tokens: int = 0
    reasoning_tokens: int = 0


def _normalise(text: str) -> str:
    return _NON_ALNUM.sub("", text.lower())


def quote_found(quote: str, paper_text: str) -> bool:
    """Whether a quote really comes from the paper, ignoring spacing, case and punctuation.

    Models tidy up LaTeX when quoting, so an exact character match would reject honest quotes, while
    a fabricated quote still fails because its words are not in the paper in that order.
    """
    needle = _normalise(quote)
    return len(needle) >= 12 and needle in _normalise(paper_text)


def split_parts(text: str, max_parts: int = MAX_PARTS) -> list[str]:
    """Split a paper at its section boundaries into at most max_parts pieces, keeping all the text."""
    starts = sorted({0, *(m.start() for m in _SECTION_START.finditer(text))})
    pieces = [text[a:b] for a, b in zip(starts, [*starts[1:], len(text)], strict=True)]

    bounded: list[str] = []
    for piece in pieces:
        while len(piece) > MAX_PART_CHARS:
            cut = piece.rfind("\n\n", 0, MAX_PART_CHARS)
            cut = cut if cut > 0 else MAX_PART_CHARS
            bounded.append(piece[:cut])
            piece = piece[cut:]
        bounded.append(piece)

    target = max(MIN_PART_CHARS, len(text) // max_parts + 1)
    parts: list[str] = []
    current = ""
    for piece in bounded:
        if current and len(current) + len(piece) > target:
            parts.append(current)
            current = ""
        current += piece
    if current:
        parts.append(current)

    # Anything left over beyond the limit joins the last part, so nothing is dropped.
    if len(parts) > max_parts:
        parts = [*parts[: max_parts - 1], "".join(parts[max_parts - 1 :])]
    return [part for part in parts if part.strip()]


def _reviewed(finding: Finding, paper_text: str) -> ReviewedFinding:
    return ReviewedFinding(
        title=finding.title,
        severity=finding.severity,
        location=finding.location,
        evidence_quote=finding.evidence_quote,
        explanation=finding.explanation,
        quote_verified=quote_found(finding.evidence_quote, paper_text),
    )


def rank_findings(findings: list[ReviewedFinding]) -> list[ReviewedFinding]:
    """Upheld before uncertain, serious before minor, and checkable quotes before uncheckable."""

    def key(item: tuple[int, ReviewedFinding]) -> tuple:
        index, finding = item
        verdict = _VERDICT_RANK[finding.verdict] if finding.verdict else 0
        return (verdict, _SEVERITY_RANK[finding.severity], not finding.quote_verified, index)

    return [finding for _, finding in sorted(enumerate(findings), key=key)]


def _dedupe(findings: list[ReviewedFinding]) -> list[ReviewedFinding]:
    seen: set[str] = set()
    unique = []
    for finding in findings:
        identity = _normalise(finding.title)[:80] or _normalise(finding.evidence_quote)[:80]
        if identity not in seen:
            seen.add(identity)
            unique.append(finding)
    return unique


async def review_zero_shot(caller: Caller, model: ModelInfo, paper: Paper) -> ReviewRecord:
    outcome, parsed = await caller.call(
        f"{model.name}|{ZERO_SHOT}|{paper.key}",
        model,
        prompts.load("reviewer_system"),
        prompts.load("zero_shot").format(paper=paper.text),
        Findings,
        MAX_OUTPUT_REVIEW,
    )
    findings = [_reviewed(f, paper.text) for f in parsed.findings] if parsed else []
    return ReviewRecord(
        model=model.name,
        arm=ZERO_SHOT,
        paper=paper.key,
        status=outcome.status,
        findings=rank_findings(findings),
        calls=1,
        cost_usd=outcome.cost_usd,
        output_tokens=outcome.output_tokens,
        reasoning_tokens=outcome.reasoning_tokens,
    )


async def review_harness(caller: Caller, model: ModelInfo, paper: Paper) -> ReviewRecord:
    system = prompts.load("reviewer_system")
    base = f"{model.name}|{HARNESS}|{paper.key}"
    cost, calls, output, reasoning = 0.0, 0, 0, 0
    problems: list[str] = []

    def account(outcome) -> None:
        nonlocal cost, calls, output, reasoning
        cost += outcome.cost_usd
        calls += 1
        output += outcome.output_tokens
        reasoning += outcome.reasoning_tokens
        if outcome.status != "ok":
            problems.append(outcome.status)

    outcome, paper_map = await caller.call(
        f"{base}|map",
        model,
        system,
        prompts.load("map").format(paper=paper.text),
        PaperMap,
        MAX_OUTPUT_STEP,
    )
    account(outcome)
    map_text = paper_map.model_dump_json(indent=1) if paper_map else "(notes unavailable)"

    parts = split_parts(paper.text)
    candidates: list[ReviewedFinding] = []
    running_notes: list[str] = []
    section_template = prompts.load("section")
    for number, part in enumerate(parts, start=1):
        outcome, section = await caller.call(
            f"{base}|part{number}",
            model,
            system,
            section_template.format(
                paper_map=map_text,
                running_notes="\n".join(f"- {note}" for note in running_notes) or "(none yet)",
                part_number=number,
                part_count=len(parts),
                section=part,
            ),
            SectionReview,
            MAX_OUTPUT_STEP,
        )
        account(outcome)
        if section:
            candidates.extend(_reviewed(f, paper.text) for f in section.findings)
            running_notes.extend(section.notes_for_later_sections)

    candidates = rank_findings(_dedupe(candidates))[:MAX_CANDIDATES]
    final: list[ReviewedFinding] = []
    refuted: list[ReviewedFinding] = []
    if candidates:
        listing = json.dumps(
            [
                {
                    "id": index,
                    "title": c.title,
                    "location": c.location,
                    "quote": c.evidence_quote,
                    "explanation": c.explanation,
                }
                for index, c in enumerate(candidates, start=1)
            ],
            ensure_ascii=False,
            indent=1,
        )
        outcome, verification = await caller.call(
            f"{base}|verify",
            model,
            system,
            prompts.load("verify").format(paper=paper.text, candidates=listing),
            Verification,
            MAX_OUTPUT_STEP,
        )
        account(outcome)
        verdicts = {v.id: v for v in verification.verdicts} if verification else {}
        for index, candidate in enumerate(candidates, start=1):
            verdict = verdicts.get(index)
            # A candidate nobody could check stays, marked uncertain, rather than silently vanishing.
            candidate.verdict = verdict.verdict if verdict else VerdictKind.uncertain
            candidate.verdict_reason = verdict.reason if verdict else None
            if verdict:
                candidate.severity = verdict.severity
            (refuted if candidate.verdict is VerdictKind.refuted else final).append(candidate)

    status = "ok"
    if problems:
        status = "partial" if (final or parts and len(problems) < calls) else problems[0]
    return ReviewRecord(
        model=model.name,
        arm=HARNESS,
        paper=paper.key,
        status=status,
        findings=rank_findings(final),
        refuted=refuted,
        calls=calls,
        cost_usd=cost,
        output_tokens=output,
        reasoning_tokens=reasoning,
    )


async def review_memory(caller: Caller, model: ModelInfo, paper: Paper) -> ReviewRecord:
    """Title only, with no paper. Whatever scores here came from memory or a lucky guess."""
    outcome, parsed = await caller.call(
        f"{model.name}|{MEMORY}|{paper.key}",
        model,
        "You answer questions about scientific papers accurately. If you do not know, say so.",
        prompts.load("memory").format(title=paper.title, category=paper.category),
        MemoryAnswer,
        MAX_OUTPUT_STEP,
    )
    findings = [
        ReviewedFinding(**f.model_dump(), quote_verified=False)
        for f in (parsed.findings if parsed else [])
    ]
    return ReviewRecord(
        model=model.name,
        arm=MEMORY,
        paper=paper.key,
        status=outcome.status,
        findings=findings,
        recognised=parsed.recognised_this_paper if parsed else None,
        known_withdrawn=parsed.known_to_be_withdrawn_or_corrected if parsed else None,
        calls=1,
        cost_usd=outcome.cost_usd,
        output_tokens=outcome.output_tokens,
        reasoning_tokens=outcome.reasoning_tokens,
    )


REVIEWERS = {ZERO_SHOT: review_zero_shot, HARNESS: review_harness, MEMORY: review_memory}
