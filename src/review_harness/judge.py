"""Scores findings against the authors' stated reason, with a built-in test of the judge itself.

Each review is judged twice, in random order: against its own paper's reason, and against another
paper's reason from the same broad field. The judge sees only a statement and some findings, so it
cannot tell which is which. If true pairs score far above shuffled pairs, the judge is detecting
real agreement. If they score alike, the judge is just agreeing with plausible text.
"""

import asyncio
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel

from review_harness import prompts
from review_harness.ledger import Caller
from review_harness.llm import ModelInfo
from review_harness.review import ReviewedFinding, ReviewRecord
from review_harness.schemas import Judgment, MatchKind

TOP_K = 5
MAX_OUTPUT_JUDGE = 8_000
TRUE_PAIR = "true"
SHUFFLED_PAIR = "shuffled"


class JudgedPair(BaseModel):
    judge: str
    model: str
    arm: str
    paper: str
    gold_paper: str
    pair: str  # true or shuffled
    matches: list[MatchKind]  # By rank, for the top findings.
    status: str = "ok"
    cost_usd: float = 0.0


def archive_of(category: str) -> str:
    """The broad field of an arXiv category, for example math from math.AG."""
    return category.split(".", 1)[0].split("-", 1)[0]


def shuffled_assignment(paper_categories: dict[str, str], seed: int) -> dict[str, str]:
    """Give every paper another paper's reason, from the same field whenever possible.

    Keeping to one field makes this a harder test, because reasons from the same field share
    vocabulary with the findings, so the judge has more chance to be fooled.
    """
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for paper in sorted(paper_categories):
        groups[archive_of(paper_categories[paper])].append(paper)

    assignment: dict[str, str] = {}
    leftovers: list[str] = []
    for members in groups.values():
        if len(members) < 2:
            leftovers.extend(members)
            continue
        order = members[:]
        rng.shuffle(order)
        for index, paper in enumerate(order):
            assignment[paper] = order[(index + 1) % len(order)]

    if len(leftovers) >= 2:
        rng.shuffle(leftovers)
        for index, paper in enumerate(leftovers):
            assignment[paper] = leftovers[(index + 1) % len(leftovers)]
    elif leftovers:
        others = sorted(set(paper_categories) - set(leftovers))
        if others:
            assignment[leftovers[0]] = rng.choice(others)
    return assignment


def format_findings(findings: Sequence[ReviewedFinding]) -> str:
    return json.dumps(
        [
            {
                "id": index,
                "title": finding.title,
                "location": finding.location,
                "explanation": finding.explanation,
            }
            for index, finding in enumerate(findings, start=1)
        ],
        ensure_ascii=False,
        indent=1,
    )


async def judge_one(
    caller: Caller,
    judge: ModelInfo,
    record: ReviewRecord,
    gold: str,
    gold_paper: str,
    pair: str,
) -> JudgedPair:
    top = record.findings[:TOP_K]
    base = dict(
        judge=judge.name,
        model=record.model,
        arm=record.arm,
        paper=record.paper,
        gold_paper=gold_paper,
        pair=pair,
    )
    if not top:
        return JudgedPair(**base, matches=[])

    # The cache key uses a hash, so the true/shuffled label never appears in the request or key.
    identity = f"{record.model}|{record.arm}|{record.paper}|{gold_paper}"
    pair_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
    outcome, parsed = await caller.call(
        f"judge|{judge.name}|{pair_id}",
        judge,
        prompts.load("judge_system"),
        prompts.load("judge").format(gold=gold, findings=format_findings(top)),
        Judgment,
        MAX_OUTPUT_JUDGE,
    )
    if parsed is None:
        return JudgedPair(**base, matches=[], status=outcome.status, cost_usd=outcome.cost_usd)
    by_id = {match.id: match.match for match in parsed.matches}
    return JudgedPair(
        **base,
        matches=[by_id.get(rank, MatchKind.none) for rank in range(1, len(top) + 1)],
        cost_usd=outcome.cost_usd,
    )


async def judge_reviews(
    caller: Caller,
    judges: list[ModelInfo],
    records: list[ReviewRecord],
    golds: dict[str, str],
    categories: dict[str, str],
    seed: int,
    concurrency: int = 4,
) -> list[JudgedPair]:
    """Judge every review against its true reason and a shuffled one, in a random order."""
    assignment = shuffled_assignment({p: categories[p] for p in golds if p in categories}, seed)
    tasks = []
    for record in records:
        if record.paper not in golds:
            continue
        for judge in judges:
            tasks.append((judge, record, golds[record.paper], record.paper, TRUE_PAIR))
            other = assignment.get(record.paper)
            if other:
                tasks.append((judge, record, golds[other], other, SHUFFLED_PAIR))
    random.Random(seed).shuffle(tasks)

    gate = asyncio.Semaphore(concurrency)

    async def run(task: tuple) -> JudgedPair:
        async with gate:
            return await judge_one(caller, *task)

    return list(await asyncio.gather(*(run(task) for task in tasks)))
