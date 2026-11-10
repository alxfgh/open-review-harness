"""Deviation 1's re-judging check: sampling is fixed and fair, and agreement is computed correctly."""

from dataclasses import replace

import pytest

from review_harness.judge import SHUFFLED_PAIR, TRUE_PAIR, JudgedPair
from review_harness.ledger import Budget, Caller, Ledger
from review_harness.rejudge import (
    agreement,
    api_name,
    load_or_draw_units,
    merge,
    rejudge,
    render,
    render_all,
    sample_units,
    summarize,
    units_path,
)
from review_harness.review import HARNESS, ReviewedFinding, ReviewRecord
from review_harness.schemas import MatchKind

from .fakes import MODEL, FakeLLM

E, R, N = MatchKind.exact, MatchKind.related, MatchKind.none
CODEX = ("codex/gpt-6.1-sol", "codex/gpt-6.1-terra")


def pair(judge, paper, kind, matches, gold_paper=None, model="m"):
    return JudgedPair(
        judge=judge,
        model=model,
        arm=HARNESS,
        paper=paper,
        gold_paper=gold_paper or paper,
        pair=kind,
        matches=matches,
    )


def codex_judged(n=40, model="m"):
    items = []
    for i in range(n):
        for judge in CODEX:
            matches = [E, N] if i % 3 == 0 else [N, R]
            items.append(pair(judge, f"p{i}", TRUE_PAIR, matches, model=model))
            shuffled = f"p{(i + 1) % n}"
            items.append(pair(judge, f"p{i}", SHUFFLED_PAIR, [N, N], shuffled, model=model))
    # A review with no findings was never sent to a judge, so it cannot be sampled.
    for judge in CODEX:
        items.append(pair(judge, "empty", TRUE_PAIR, [], model=model))
    return items


def test_api_name_strips_only_the_codex_prefix():
    assert api_name("codex/gpt-6.1-sol") == "gpt-6.1-sol"
    assert api_name("gpt-6.1-sol") == "gpt-6.1-sol"


def test_sample_is_a_fixed_tenth_of_units_every_judge_scored():
    judged = codex_judged()
    units = sample_units(judged, seed=7)
    assert len(units) == 8  # 80 scored units, the empty review excluded
    assert units == sample_units(list(reversed(judged)), seed=7)
    assert all(unit[2] != "empty" for unit in units)
    assert units != sample_units(judged, seed=8)


def test_a_models_sample_is_unchanged_when_another_models_judgments_are_added():
    first = codex_judged(model="codex/a")
    alone = sample_units(first, seed=7)
    both = first + codex_judged(model="codex/b")
    assert sample_units(both, seed=7, models=["codex/a"]) == alone
    other = sample_units(both, seed=7, models=["codex/b"])
    assert len(other) == 8 and all(unit[0] == "codex/b" for unit in other)


def test_a_saved_sample_is_reused_rather_than_redrawn(tmp_path):
    judged_path = tmp_path / "judgments.jsonl"
    judged_path.write_text("x", encoding="utf-8")
    path = units_path(tmp_path, ["codex/a"])
    assert path.name == "rejudge-units-codex-a.json"
    first = codex_judged(model="codex/a")
    units, loaded = load_or_draw_units(path, judged_path, first, 7, models=["codex/a"])
    assert not loaded and path.exists() and len(units) == 8
    # Later judgments would change a fresh draw from the full frame; the saved sample stands.
    later = codex_judged(n=60, model="codex/a")
    again, loaded = load_or_draw_units(path, judged_path, later, 7, models=["codex/a"])
    assert loaded and again == units
    assert sample_units(later, 7, models=["codex/a"]) != units


def test_a_saved_sample_drawn_differently_is_refused(tmp_path):
    judged_path = tmp_path / "judgments.jsonl"
    judged_path.write_text("x", encoding="utf-8")
    path = units_path(tmp_path, ["codex/a"])
    judged = codex_judged(model="codex/a")
    load_or_draw_units(path, judged_path, judged, 7, models=["codex/a"])
    for changed in ({"seed": 8}, {"fraction": 0.2}):
        args = {"seed": 7, "fraction": 0.10, **changed}
        with pytest.raises(ValueError, match="Move it aside"):
            load_or_draw_units(
                path, judged_path, judged, args["seed"], args["fraction"], ["codex/a"]
            )


def test_a_models_frame_uses_only_the_judges_of_its_own_reviews():
    first = codex_judged(model="codex/a")
    # Model b was judged by a different pair; that must not empty model a's frame.
    other = [
        pair(judge, f"q{i}", TRUE_PAIR, [E], model="codex/b")
        for i in range(10)
        for judge in ("codex/x", "codex/y")
    ]
    assert sample_units(first + other, 7, models=["codex/a"]) == sample_units(first, 7)


def test_merge_keeps_other_units_and_replaces_redone_ones():
    old_a = pair("a", "p1", TRUE_PAIR, [N], model="codex/a")
    old_b = pair("a", "p1", TRUE_PAIR, [N], model="codex/b")
    new_b = pair("a", "p1", TRUE_PAIR, [E], model="codex/b")
    assert merge([old_a, old_b], [new_b]) == [old_a, new_b]


def test_summary_reports_each_reviewed_model_and_the_pool():
    original, repeat = [], []
    for model in ("codex/a", "codex/b"):
        for judge in ("a", "b"):
            original.append(pair(f"codex/{judge}", "p1", TRUE_PAIR, [E], model=model))
            repeat.append(pair(judge, "p1", TRUE_PAIR, [E], model=model))
    summary = summarize(original, repeat)
    assert set(summary["by_model"]) == {"codex/a", "codex/b"}
    assert summary["by_model"]["codex/a"]["units"] == 1
    assert summary["pooled"]["units"] == 2 and summary["pooled"]["exact_at_5"]["both"] == 2
    text = render_all(summary)
    assert "reviews by codex/a" in text and "reviews by codex/b" in text and "all models" in text


def test_agreement_counts_labels_and_both_judges_hits():
    original = [
        pair("codex/a", "p1", TRUE_PAIR, [E, N]),
        pair("codex/b", "p1", TRUE_PAIR, [E, R]),
        pair("codex/a", "p2", TRUE_PAIR, [E, N]),
        pair("codex/b", "p2", TRUE_PAIR, [N, N]),
        pair("codex/a", "p3", SHUFFLED_PAIR, [N, N], gold_paper="p1"),
        pair("codex/b", "p3", SHUFFLED_PAIR, [N, N], gold_paper="p1"),
    ]
    repeat = [
        pair("a", "p1", TRUE_PAIR, [E, N]),
        pair("b", "p1", TRUE_PAIR, [E, N]),
        pair("a", "p2", TRUE_PAIR, [E, N]),
        pair("b", "p2", TRUE_PAIR, [E, N]),
        pair("a", "p3", SHUFFLED_PAIR, [N, N], gold_paper="p1"),
        pair("b", "p3", SHUFFLED_PAIR, [N, N], gold_paper="p1"),
    ]
    result = agreement(original, repeat)
    assert result["judges"] == ["a", "b"] and result["units"] == 3
    assert result["per_judge"]["a"]["same_label"] == 1.0
    assert result["per_judge"]["b"]["findings_compared"] == 6
    assert result["per_judge"]["b"]["same_label"] == round(4 / 6, 4)
    hits = result["exact_at_5"]
    # p1 hit under both runtimes; p2 only once the API's b also marked rank 1 exact.
    assert (hits["both"], hits["codex_only"], hits["api_only"], hits["neither"]) == (1, 0, 1, 1)
    assert hits["shuffled_hits"] == {"codex": 0, "api": 0}
    assert "Codex only 0, API only 1" in render(result, 3, 30)


def test_failed_api_judgments_are_left_out_not_counted_as_misses():
    original = [pair("codex/a", "p1", TRUE_PAIR, [E])]
    repeat = [pair("a", "p1", TRUE_PAIR, []).model_copy(update={"status": "refused"})]
    result = agreement(original, repeat)
    assert result["units"] == 0 and result["exact_at_5"]["agreement"] is None


async def test_rejudge_sends_each_sampled_unit_to_every_judge(tmp_path):
    def respond(request):
        return {"matches": [{"id": 1, "match": "exact", "reason": "same"}]}

    finding = ReviewedFinding(
        title="t",
        severity="fatal",
        location="l",
        evidence_quote="q",
        explanation="e",
        quote_verified=True,
    )
    record = ReviewRecord(model="m", arm=HARNESS, paper="p1", status="ok", findings=[finding])
    second = replace(MODEL, name="fake-model-2")
    llm = FakeLLM(respond)
    out = await rejudge(
        Caller(llm, Ledger(tmp_path / "l.jsonl"), Budget(100)),
        [MODEL, second],
        [("m", HARNESS, "p1", "p2", SHUFFLED_PAIR)],
        [record],
        {"p1": "gold one", "p2": "gold two"},
    )
    assert len(llm.calls) == 2 and {j.judge for j in out} == {"fake-model", "fake-model-2"}
    assert all("gold two" in call.user for call in llm.calls)
    assert all(j.pair == SHUFFLED_PAIR and j.matches == [E] for j in out)
