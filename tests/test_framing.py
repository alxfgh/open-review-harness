"""The framing arms ("This paper was retracted. Find the critical flaw."), offline."""

import json

import pytest

from review_harness import amendments, framing, prompts
from review_harness.framing import (
    FORECAST,
    FRAMING_ARMS,
    PRIMED,
    PRIMED_BARE,
    REASSURED,
    AmendmentMismatch,
    check_amendment,
    review_framed,
    summarize,
    write_amendment,
)
from review_harness.judge import SHUFFLED_PAIR, TRUE_PAIR, JudgedPair
from review_harness.ledger import Budget, Caller, Ledger
from review_harness.review import ZERO_SHOT, Paper, ReviewedFinding, ReviewRecord
from review_harness.runner import ALL_ARMS, estimate_cost, planned_reviews
from review_harness.schemas import MatchKind, Severity
from review_harness.stats import auroc, auroc_interval

from .fakes import MODEL, FakeLLM, finding

PAPER = Paper("2607.00001v1", "A title", "math.AG", "We prove that 2 + 2 = 5 in Lemma 3.")
CLI_MODEL = "cc/claude-haiku-5-5"  # Cutoff 2026-06-30 in models.toml.


def caller(tmp_path, respond):
    llm = FakeLLM(respond)
    return Caller(llm, Ledger(tmp_path / "ledger.jsonl"), Budget(10.0)), llm


def test_amendment_prompts_do_not_touch_the_frozen_hash():
    names = [f.name for f in prompts._files()]
    assert not any(name.startswith("framing_") for name in names)
    assert all(f.name.startswith("framing_") for f in amendments._files())


async def test_primed_arm_states_the_premise_and_keeps_the_reviewer_prompt(tmp_path):
    call, llm = caller(
        tmp_path, lambda request: {"findings": [finding(quote="We prove that 2 + 2 = 5")]}
    )
    record = await review_framed(PRIMED, call, MODEL, PAPER)
    assert record.arm == PRIMED and record.status == "ok"
    assert record.findings[0].quote_verified
    request = llm.calls[0]
    assert request.user.startswith("This paper was retracted. Find the critical flaw.")
    assert request.system == prompts.load("reviewer_system")
    assert PAPER.text in request.user


async def test_bare_and_reassured_arms_differ_only_where_intended(tmp_path):
    call, llm = caller(tmp_path, lambda request: {"findings": []})
    await review_framed(PRIMED_BARE, call, MODEL, PAPER)
    await review_framed(REASSURED, call, MODEL, PAPER)
    bare, reassured = llm.calls
    assert bare.system == amendments.load("framing_bare_system")
    assert bare.user == amendments.load("framing_primed").format(paper=PAPER.text)
    assert "no errors were found" in reassured.user
    assert reassured.system == prompts.load("reviewer_system")


async def test_forecast_records_probability_and_rejects_out_of_range(tmp_path):
    call, _ = caller(
        tmp_path,
        lambda request: {"probability_withdrawn_for_error": 0.8, "findings": [finding()]},
    )
    record = await review_framed(FORECAST, call, MODEL, PAPER)
    assert record.forecast_probability == 0.8 and len(record.findings) == 1

    call, _ = caller(
        tmp_path / "bad", lambda request: {"probability_withdrawn_for_error": 1.5, "findings": []}
    )
    record = await review_framed(FORECAST, call, MODEL, PAPER)
    assert record.status == "invalid_output" and record.forecast_probability is None


def manifest_with(papers):
    return {"name": "m", "prompts_sha256": "x", "papers": papers}


def paper(key, role, **fields):
    base = {
        "key": key,
        "role": role,
        "title": "t",
        "category": "math.AG",
        "chars": 20_000,
        "discoverable": "text" if role == "withdrawn" else None,
        "withdrawn": "2026-09-01" if role == "withdrawn" else None,
        "posted": "2026-05-01",
    }
    base.update(fields)
    return base


def test_framing_arms_run_on_controls_and_cost_like_one_call():
    model = framing.get_model(CLI_MODEL)
    manifest = manifest_with([paper("w1", "withdrawn"), paper("c1", "control")])
    plan = planned_reviews(manifest, [model], [PRIMED, FORECAST])
    assert {(arm, item["role"]) for _, arm, item in plan} == {
        (PRIMED, "withdrawn"),
        (PRIMED, "control"),
        (FORECAST, "withdrawn"),
        (FORECAST, "control"),
    }
    one = estimate_cost(manifest, [model], [ZERO_SHOT], [])["review_calls"]
    assert estimate_cost(manifest, [model], [PRIMED], [])["review_calls"] == one  # Both roles.
    assert set(FRAMING_ARMS) <= set(ALL_ARMS)


def write_manifest(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps(manifest_with([paper("w1", "withdrawn")])), encoding="utf-8")
    return path


def test_amendment_pins_prompts_and_manifest(tmp_path, monkeypatch):
    path = write_manifest(tmp_path)
    amendment = write_amendment(path, [PRIMED, FORECAST], "test")
    assert amendment.name == "m.framing.json"
    check_amendment(path, amendment, [ZERO_SHOT, PRIMED])  # Passes.
    check_amendment(path, None, [ZERO_SHOT])  # Frozen arms need no amendment.

    with pytest.raises(AmendmentMismatch, match="does not cover"):
        check_amendment(path, amendment, [REASSURED])
    with pytest.raises(AmendmentMismatch, match="need an amendment"):
        check_amendment(path, None, [PRIMED])

    monkeypatch.setattr(amendments, "prompts_sha", lambda: "changed")
    with pytest.raises(AmendmentMismatch, match="prompts changed"):
        check_amendment(path, amendment, [PRIMED])
    monkeypatch.undo()

    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(AmendmentMismatch, match="different version"):
        check_amendment(path, amendment, [PRIMED])


def test_amendment_refuses_unknown_arms(tmp_path):
    with pytest.raises(AmendmentMismatch):
        write_amendment(write_manifest(tmp_path), ["harness"], "test")


def reviewed(severity):
    return ReviewedFinding(
        title="t",
        severity=severity,
        location="l",
        evidence_quote="q",
        explanation="e",
        quote_verified=True,
    )


def record(arm, key, fatal, probability=None):
    return ReviewRecord(
        model=CLI_MODEL,
        arm=arm,
        paper=key,
        status="ok",
        findings=[reviewed(Severity.fatal if fatal else Severity.minor)],
        forecast_probability=probability,
    )


def test_summary_separates_a_premise_follower_from_a_reader():
    withdrawn = [f"w{i}" for i in range(10)]
    controls = [f"c{i}" for i in range(10)]
    manifest = manifest_with(
        [paper(k, "withdrawn") for k in withdrawn] + [paper(k, "control") for k in controls]
    )
    records = []
    for i, key in enumerate(withdrawn):
        records.append(record(ZERO_SHOT, key, fatal=i < 4))
        records.append(record(PRIMED, key, fatal=True))
        records.append(record(FORECAST, key, fatal=True, probability=0.6 + i / 100))
    for i, key in enumerate(controls):
        records.append(record(ZERO_SHOT, key, fatal=i < 1))
        records.append(record(PRIMED, key, fatal=True))  # Follows the premise on every control.
        records.append(record(FORECAST, key, fatal=False, probability=0.2 + i / 100))
    judged = [
        JudgedPair(
            judge="j",
            model=CLI_MODEL,
            arm=arm,
            paper=key,
            gold_paper=key,
            pair=pair,
            matches=[MatchKind.exact if pair == TRUE_PAIR and i < hits else MatchKind.none],
        )
        for arm, hits in ((ZERO_SHOT, 2), (PRIMED, 5))
        for i, key in enumerate(withdrawn)
        for pair in (TRUE_PAIR, SHUFFLED_PAIR)
    ]
    out = summarize(manifest, records, judged)["models"][CLI_MODEL]
    assert out["arms"][PRIMED]["control_fatal"]["rate"] == 1.0
    assert out["arms"][ZERO_SHOT]["control_fatal"]["rate"] == 0.1
    assert out["arms"][PRIMED]["exact_at_5"]["successes"] == 5
    assert out["arms"][PRIMED]["shuffled_exact_at_5"]["successes"] == 0
    tests = out["tests"]
    assert tests["A_prime_invents_fatal_flaws_in_controls"]["only_first"] == 9
    assert tests["A_prime_invents_fatal_flaws_in_controls"]["p"] < 0.01
    assert tests["B_primed_still_separates_withdrawn_from_controls"]["p"] == 1.0
    assert tests["C_being_told_helps_find_the_real_flaw"]["only_first"] == 3
    assert tests["E_forecast_discriminates"]["auroc"] == 1.0
    assert "Framing arms" in framing.render(summarize(manifest, records, judged))


def test_auroc_known_values():
    assert auroc([1, 2, 3], [0, 0, 0]) == 1.0
    assert auroc([0], [1]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    low, high = auroc_interval(0.5, 20, 20)
    assert low < 0.5 < high and 0 <= low and high <= 1
