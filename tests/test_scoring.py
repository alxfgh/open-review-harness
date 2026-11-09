"""The numbers that will be published. Known answers, checked by hand or against textbook values."""

import pytest

from review_harness.judge import (
    SHUFFLED_PAIR,
    TRUE_PAIR,
    JudgedPair,
    judge_reviews,
    shuffled_assignment,
)
from review_harness.ledger import Budget, Caller, Ledger
from review_harness.report import _hit, build_summary, render_markdown
from review_harness.review import HARNESS, MEMORY, ZERO_SHOT, ReviewedFinding, ReviewRecord
from review_harness.schemas import MatchKind, Severity
from review_harness.stats import cohen_kappa, fisher_exact, mcnemar_exact, wilson_interval

from .fakes import MODEL, FakeLLM


class TestStats:
    def test_wilson_known_value(self):
        low, high = wilson_interval(10, 100)
        assert low == pytest.approx(0.0552, abs=0.001)
        assert high == pytest.approx(0.1744, abs=0.001)

    def test_wilson_at_zero_is_not_zero_width(self):
        low, high = wilson_interval(0, 20)
        assert low == 0.0 and high == pytest.approx(0.161, abs=0.002)

    def test_wilson_with_no_data_says_nothing(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_mcnemar_known_values(self):
        assert mcnemar_exact(0, 0) == 1.0
        assert mcnemar_exact(5, 5) == 1.0
        assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
        assert mcnemar_exact(8, 2) == pytest.approx(0.109375)

    def test_fisher_known_value(self):
        # Fisher's tea-tasting table.
        assert fisher_exact(3, 1, 1, 3) == pytest.approx(0.4857, abs=0.001)
        assert fisher_exact(8, 2, 1, 5) == pytest.approx(0.0350, abs=0.001)

    def test_kappa(self):
        assert cohen_kappa([(True, True), (False, False)]) == 1.0
        assert cohen_kappa([(True, False), (False, True)]) == pytest.approx(-1.0)
        assert cohen_kappa([]) is None
        assert cohen_kappa([(True, True)]) is None


class TestShuffle:
    def test_no_paper_gets_its_own_reason(self):
        categories = {f"p{i}": "math.AG" for i in range(10)}
        assignment = shuffled_assignment(categories, seed=1)
        assert set(assignment) == set(categories)
        assert all(paper != other for paper, other in assignment.items())

    def test_stays_within_the_field_when_possible(self):
        categories = {"a": "math.AG", "b": "math.NT", "c": "cs.CL", "d": "cs.CV", "e": "hep-ph"}
        assignment = shuffled_assignment(categories, seed=3)
        assert assignment["a"] == "b" and assignment["b"] == "a"
        assert assignment["c"] == "d" and assignment["d"] == "c"
        assert assignment["e"] != "e"

    def test_is_reproducible(self):
        categories = {f"p{i}": "cs.CL" for i in range(20)}
        assert shuffled_assignment(categories, 7) == shuffled_assignment(categories, 7)


class TestHitRule:
    def test_every_judge_must_agree(self):
        exact, none = MatchKind.exact, MatchKind.none
        assert _hit([[exact], [exact]], 5, MatchKind.exact) is True
        assert _hit([[exact], [none]], 5, MatchKind.exact) is False

    def test_judges_must_agree_on_the_same_finding(self):
        exact, none = MatchKind.exact, MatchKind.none
        assert _hit([[exact, none], [none, exact]], 5, MatchKind.exact) is False

    def test_only_the_top_k_count(self):
        none, exact = MatchKind.none, MatchKind.exact
        assert _hit([[none, none, exact]], 2, MatchKind.exact) is False
        assert _hit([[none, none, exact]], 3, MatchKind.exact) is True

    def test_related_is_not_exact(self):
        assert _hit([[MatchKind.related]], 5, MatchKind.exact) is False
        assert _hit([[MatchKind.related]], 5, MatchKind.related) is True

    def test_no_judgments_is_no_hit(self):
        assert _hit([], 5, MatchKind.exact) is False


def make_finding(severity=Severity.fatal, quote=True):
    return ReviewedFinding(
        title="t",
        severity=severity,
        location="l",
        evidence_quote="q",
        explanation="e",
        quote_verified=quote,
    )


class TestJudging:
    async def test_judge_sees_only_the_statement_and_findings(self, tmp_path):
        seen = []

        def respond(request):
            seen.append(request.user + request.system)
            return {"matches": [{"id": 1, "match": "exact", "reason": "same"}]}

        record = ReviewRecord(
            model="m", arm=HARNESS, paper="secret-paper-id", status="ok", findings=[make_finding()]
        )
        caller = Caller(FakeLLM(respond), Ledger(tmp_path / "l.jsonl"), Budget(100))
        judged = await judge_reviews(
            caller,
            [MODEL],
            [record, record.model_copy(update={"paper": "other-paper-id"})],
            {"secret-paper-id": "gold one", "other-paper-id": "gold two"},
            {"secret-paper-id": "math.AG", "other-paper-id": "math.NT"},
            seed=1,
        )
        assert {j.pair for j in judged} == {TRUE_PAIR, SHUFFLED_PAIR}
        for text in seen:
            assert "secret-paper-id" not in text and "other-paper-id" not in text
            assert "shuffled" not in text.lower() and TRUE_PAIR + " pair" not in text

    async def test_nothing_to_judge_costs_nothing(self, tmp_path):
        llm = FakeLLM(lambda _: {"matches": []})
        record = ReviewRecord(model="m", arm=HARNESS, paper="a", status="ok", findings=[])
        judged = await judge_reviews(
            Caller(llm, Ledger(tmp_path / "l.jsonl"), Budget(100)),
            [MODEL],
            [record],
            {"a": "g", "b": "h"},
            {"a": "x", "b": "x"},
            seed=1,
        )
        assert llm.calls == [] and judged[0].matches == []


def manifest_with(n_withdrawn=4, n_controls=4):
    papers = []
    for i in range(n_withdrawn):
        papers.append(
            {
                "key": f"w{i}",
                "role": "withdrawn",
                "withdrawn": "2026-09-01",
                "posted": "2026-08-01" if i == 0 else "2025-01-01",
                "discoverable": "text",
                "category": "math.AG",
                "gold": f"gold {i}",
            }
        )
    for i in range(n_controls):
        papers.append(
            {"key": f"c{i}", "role": "control", "posted": "2025-01-01", "category": "math.AG"}
        )
    return {"name": "test", "prompts_sha256": "abc", "papers": papers}


def judged_pair(model, arm, paper, pair, matches):
    return JudgedPair(
        judge="j", model=model, arm=arm, paper=paper, gold_paper=paper, pair=pair, matches=matches
    )


class TestSummary:
    MODEL_NAME = "gpt-6.1-terra"  # Cutoff 2026-06-01, so these papers count as unseen.

    def build(self, harness_hits=2, control_fatal=0):
        manifest = manifest_with()
        records, judged = [], []
        for i in range(4):
            key = f"w{i}"
            for arm in (HARNESS, ZERO_SHOT, MEMORY):
                records.append(
                    ReviewRecord(
                        model=self.MODEL_NAME,
                        arm=arm,
                        paper=key,
                        status="ok",
                        findings=[] if arm == MEMORY else [make_finding()],
                        recognised=False if arm == MEMORY else None,
                        cost_usd=1.0,
                    )
                )
            hit = [MatchKind.exact] if i < harness_hits else [MatchKind.none]
            judged.append(judged_pair(self.MODEL_NAME, HARNESS, key, TRUE_PAIR, hit))
            judged.append(
                judged_pair(self.MODEL_NAME, HARNESS, key, SHUFFLED_PAIR, [MatchKind.none])
            )
            judged.append(judged_pair(self.MODEL_NAME, ZERO_SHOT, key, TRUE_PAIR, [MatchKind.none]))
            judged.append(
                judged_pair(self.MODEL_NAME, ZERO_SHOT, key, SHUFFLED_PAIR, [MatchKind.none])
            )
        for i in range(4):
            findings = [make_finding()] if i < control_fatal else []
            records.append(
                ReviewRecord(
                    model=self.MODEL_NAME,
                    arm=HARNESS,
                    paper=f"c{i}",
                    status="ok",
                    findings=findings,
                )
            )
        return build_summary(manifest, records, judged)

    def test_rates_are_correct(self):
        arms = self.build()["models"][self.MODEL_NAME]["arms"]
        assert arms[HARNESS]["exact_at_5"]["successes"] == 2
        assert arms[HARNESS]["exact_at_5"]["n"] == 4
        assert arms[HARNESS]["shuffled_exact_at_5"]["successes"] == 0
        assert arms[ZERO_SHOT]["exact_at_5"]["successes"] == 0

    def test_posted_after_cutoff_subset(self):
        harness = self.build()["models"][self.MODEL_NAME]["arms"][HARNESS]
        assert harness["exact_at_5_posted_after_cutoff"]["n"] == 1

    def test_small_samples_do_not_claim_success(self):
        tests = self.build()["models"][self.MODEL_NAME]["tests"]
        assert tests["1_beats_shuffled_answers"]["passed"] is False, "2 of 4 cannot reach p < 0.01"

    def test_strong_signal_passes(self):
        manifest = manifest_with(n_withdrawn=12, n_controls=0)
        records, judged = [], []
        for i in range(12):
            key = f"w{i}"
            records.append(
                ReviewRecord(
                    model=self.MODEL_NAME,
                    arm=HARNESS,
                    paper=key,
                    status="ok",
                    findings=[make_finding()],
                )
            )
            judged.append(judged_pair(self.MODEL_NAME, HARNESS, key, TRUE_PAIR, [MatchKind.exact]))
            judged.append(
                judged_pair(self.MODEL_NAME, HARNESS, key, SHUFFLED_PAIR, [MatchKind.none])
            )
        tests = build_summary(manifest, records, judged)["models"][self.MODEL_NAME]["tests"]
        assert tests["1_beats_shuffled_answers"]["passed"] is True

    def test_flagging_everything_does_not_pass_the_controls_test(self):
        tests = self.build(control_fatal=4)["models"][self.MODEL_NAME]["tests"]
        assert tests["4_flags_withdrawn_more_than_controls"]["passed"] is False

    def test_invented_quotes_fail_the_quote_test(self):
        manifest = manifest_with(n_withdrawn=1, n_controls=0)
        records = [
            ReviewRecord(
                model=self.MODEL_NAME,
                arm=HARNESS,
                paper="w0",
                status="ok",
                findings=[make_finding(quote=False)],
            )
        ]
        tests = build_summary(manifest, records, [])["models"][self.MODEL_NAME]["tests"]
        assert tests["5_quotes_are_real"]["passed"] is False

    def test_papers_a_model_may_have_seen_are_not_counted(self):
        manifest = manifest_with()
        for paper in manifest["papers"]:
            if paper["role"] == "withdrawn":
                paper["withdrawn"] = "2026-01-01"  # Before this model's cutoff.
        records = [
            ReviewRecord(
                model=self.MODEL_NAME,
                arm=HARNESS,
                paper="w0",
                status="ok",
                findings=[make_finding()],
            )
        ]
        judged = [judged_pair(self.MODEL_NAME, HARNESS, "w0", TRUE_PAIR, [MatchKind.exact])]
        model = build_summary(manifest, records, judged)["models"][self.MODEL_NAME]
        assert model["unseen_papers"] == 0
        assert model["not_counted_possibly_seen"] == 4
        assert HARNESS not in model["arms"]

    def test_model_with_unknown_cutoff_never_counts_papers_as_unseen(self):
        manifest = manifest_with()
        records = [
            ReviewRecord(
                model="gemini-4-argon",
                arm=HARNESS,
                paper="w0",
                status="ok",
                findings=[make_finding()],
            )
        ]
        model = build_summary(manifest, records, [])["models"]["gemini-4-argon"]
        assert model["unseen_papers"] == 0

    def test_markdown_states_failures_plainly(self):
        text = render_markdown(self.build())
        assert "**Did not pass**" in text
        assert "not done" in text, "the lack of a human audit must be stated"
        assert "w0" not in text, "no per-paper output is published"

    def test_summary_contains_no_paper_level_detail(self):
        import json

        text = json.dumps(self.build())
        for key in ("w0", "w1", "c0"):
            assert f'"{key}"' not in text
