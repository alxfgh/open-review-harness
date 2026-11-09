"""The whole path offline: frozen manifest, reviews, judging and report, with a scripted model."""

from datetime import date

import httpx
import pytest
import respx

from review_harness import manifest as manifests
from review_harness import prompts
from review_harness.judge import judge_reviews
from review_harness.ledger import Budget, BudgetExceeded, Caller, Ledger, ProviderError
from review_harness.llm import get_model
from review_harness.papers import ArxivClient, ArxivEntry
from review_harness.report import build_summary
from review_harness.review import ARMS, HARNESS, Paper
from review_harness.runner import (
    ManifestMismatch,
    check_prompts_frozen,
    estimate_cost,
    load_canaries,
    planned_reviews,
    render_canaries,
    run_canaries,
    run_reviews,
)

from .fakes import FakeLLM, finding, scripted
from .test_papers import make_tar

TEX = (
    "\\documentclass{a}\\begin{document}\\section{One}"
    + "Lemma text here. " * 300
    + "\\end{document}"
)


def entry(arxiv_id="2607.00001", version=2, comment="Withdrawn: Lemma 2 is false.", **fields):
    values = dict(
        arxiv_id=arxiv_id,
        version=version,
        published="2026-07-05",
        updated="2026-09-01",
        title="A title",
        abstract="An abstract",
        comment=comment,
        category="math.AG",
        first_author="Someone",
    )
    values.update(fields)
    return ArxivEntry(**values)


def arxiv_offline():
    return ArxivClient(httpx.AsyncClient(), min_interval=0)


FEED_EMPTY = (
    '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:o="http://a9.com/-/spec/opensearch/1.1/">'
    "<o:totalResults>0</o:totalResults></feed>"
)


class TestFreezing:
    @respx.mock
    async def test_manifest_pins_text_and_excludes_with_reasons(self, tmp_path):
        respx.get(url__startswith="https://arxiv.org/e-print/2607.00001v1").respond(
            200, content=make_tar({"m.tex": TEX})
        )
        respx.get(url__startswith="https://export.arxiv.org/api/query").respond(
            200, text=FEED_EMPTY
        )
        manifest = await manifests.build_manifest(
            arxiv_offline(),
            name="t",
            entries=[
                entry(),
                entry("2607.00002", comment="Withdrawn due to some mistakes."),
                entry("2607.00003", comment="This paper has been withdrawn: authorship dispute."),
                entry("2607.00004", version=1),
                entry(
                    "2607.00005",
                    comment="Withdrawn: Lemma 3 is false.",
                ),
            ],
            source_description="test",
            since=date(2026, 6, 30),
            until=date(2026, 11, 9),
            target=1,
            target_external=0,
            seed=1,
            cache_dir=tmp_path,
            manual_exclusions={"2607.00005": "gold statement too thin after review"},
        )
        withdrawn = [p for p in manifest["papers"] if p["role"] == "withdrawn"]
        assert [p["key"] for p in withdrawn] == ["2607.00001v1"]
        assert withdrawn[0]["version"] == 1, "reviews the version before the withdrawal"
        assert withdrawn[0]["withdrawal_version"] == 2
        assert len(withdrawn[0]["text_sha256"]) == 64
        reasons = {e["id"]: e["reason"] for e in manifest["exclusions"]}
        assert reasons["2607.00002"].startswith("vague")
        assert reasons["2607.00003"].startswith("not_about_error")
        assert reasons["2607.00004"] == "no_earlier_version"
        assert reasons["2607.00005"].startswith("manual:")
        assert manifest["prompts_sha256"] == prompts.prompts_sha()
        assert "first_author" not in str(manifest["papers"]) and "Someone" not in str(manifest)

    @respx.mock
    async def test_same_seed_same_sample(self, tmp_path):
        respx.get(url__startswith="https://arxiv.org/e-print/").respond(
            200, content=make_tar({"m.tex": TEX})
        )
        respx.get(url__startswith="https://export.arxiv.org/api/query").respond(
            200, text=FEED_EMPTY
        )
        entries = [
            entry(
                f"2607.000{i:02d}", comment=f"Withdrawn: Lemma {i} is false.", first_author=f"A{i}"
            )
            for i in range(20)
        ]

        async def build(seed):
            manifest = await manifests.build_manifest(
                arxiv_offline(),
                name="t",
                entries=entries,
                source_description="test",
                since=date(2026, 6, 30),
                until=date(2026, 11, 9),
                target=5,
                target_external=0,
                seed=seed,
                cache_dir=tmp_path,
            )
            return [p["key"] for p in manifest["papers"]]

        assert await build(11) == await build(11)
        assert await build(11) != await build(12)

    @respx.mock
    async def test_paper_that_gives_the_answer_away_is_excluded(self, tmp_path):
        leaky = TEX.replace("Lemma text here.", "This paper has been withdrawn by the author.")
        respx.get(url__startswith="https://arxiv.org/e-print/").respond(
            200, content=make_tar({"m.tex": leaky})
        )
        respx.get(url__startswith="https://export.arxiv.org/api/query").respond(
            200, text=FEED_EMPTY
        )
        manifest = await manifests.build_manifest(
            arxiv_offline(),
            name="t",
            entries=[entry()],
            source_description="test",
            since=date(2026, 6, 30),
            until=date(2026, 11, 9),
            target=1,
            target_external=0,
            seed=1,
            cache_dir=tmp_path,
        )
        assert manifest["papers"] == []
        assert manifest["exclusions"][0]["reason"] == "text_reveals_error"

    def test_changed_prompts_are_refused(self):
        with pytest.raises(ManifestMismatch):
            check_prompts_frozen({"prompts_sha256": "something else"})
        check_prompts_frozen({"prompts_sha256": prompts.prompts_sha()})


def small_manifest():
    return {
        "name": "t",
        "seed": 1,
        "prompts_sha256": prompts.prompts_sha(),
        "papers": [
            {
                "key": "unseen",
                "role": "withdrawn",
                "withdrawn": "2026-09-01",
                "posted": "2026-07-01",
                "discoverable": "text",
                "category": "math.AG",
                "gold": "Lemma 2 is false",
                "title": "T1",
                "chars": 30_000,
            },
            {
                "key": "seen",
                "role": "withdrawn",
                "withdrawn": "2025-01-01",
                "posted": "2024-01-01",
                "discoverable": "text",
                "category": "math.AG",
                "gold": "Lemma 9 is false",
                "title": "T2",
                "chars": 30_000,
            },
            {
                "key": "control",
                "role": "control",
                "posted": "2026-07-01",
                "category": "math.AG",
                "title": "T3",
                "chars": 30_000,
            },
        ],
    }


class TestPlanning:
    def test_papers_a_model_may_have_seen_are_skipped(self):
        plan = planned_reviews(small_manifest(), [get_model("gpt-6.1-terra")], list(ARMS))
        keys = {(arm, item["key"]) for _, arm, item in plan}
        assert ("harness", "seen") not in keys
        assert ("harness", "unseen") in keys

    def test_old_model_can_still_review_recent_withdrawals(self):
        plan = planned_reviews(small_manifest(), [get_model("gpt-4o-2024-08-06")], ["harness"])
        assert {item["key"] for _, _, item in plan} == {"unseen", "seen", "control"}

    def test_memory_arm_skips_controls(self):
        plan = planned_reviews(small_manifest(), [get_model("gpt-6.1-terra")], ["memory"])
        assert {item["key"] for _, _, item in plan} == {"unseen"}

    def test_estimate_is_a_range_and_names_unchecked_prices(self):
        estimate = estimate_cost(
            small_manifest(),
            [get_model("gpt-6.1-terra"), get_model("gpt-4o-2024-08-06")],
            list(ARMS),
            [get_model("gpt-6.1-sol")],
        )
        low, high = estimate["total_usd"]
        assert 0 < low < high
        assert estimate["unverified_prices"] == ["gpt-4o-2024-08-06"]


class TestRunning:
    def papers(self):
        return {
            key: Paper(
                key, "t", "math.AG", "\\section{A}\nThe real passage is here.\n" + "x " * 300
            )
            for key in ("unseen", "seen", "control")
        }

    async def test_full_path_to_a_report(self, tmp_path):
        manifest = small_manifest()
        model = get_model("gpt-6.1-terra")
        ledger = Ledger(tmp_path / "l.jsonl")
        reviewer = FakeLLM(scripted([finding("A", quote="The real passage is here")]))
        plan = planned_reviews(manifest, [model], list(ARMS))
        records, failures, stopped = await run_reviews(
            plan, self.papers(), Caller(reviewer, ledger, Budget(100))
        )
        assert failures == [] and stopped is None

        def judge_reply(request):
            hit = "Lemma 2 is false" in request.user
            return {"matches": [{"id": 1, "match": "exact" if hit else "none", "reason": "r"}]}

        golds = {"unseen": "Lemma 2 is false", "seen": "Lemma 9 is false"}
        judged = await judge_reviews(
            Caller(FakeLLM(judge_reply), ledger, Budget(100)),
            [get_model("gpt-6.1-sol")],
            records,
            golds,
            {"unseen": "math.AG", "seen": "math.AG", "control": "math.AG"},
            seed=1,
        )
        summary = build_summary(manifest, records, judged)
        arms = summary["models"][model.name]["arms"]
        assert arms[HARNESS]["exact_at_5"]["successes"] == 1
        assert arms[HARNESS]["shuffled_exact_at_5"]["successes"] == 0
        assert arms[HARNESS]["quote_validity"] == 1.0

    async def test_outage_for_one_paper_does_not_lose_the_rest(self, tmp_path):
        model = get_model("gpt-6.1-terra")

        def respond(request):
            if "POISON" in request.user:
                return ProviderError("down")
            return scripted([])(request)

        papers = self.papers()
        papers["unseen"] = Paper("unseen", "t", "c", "POISON " * 50)
        plan = planned_reviews(small_manifest(), [model], ["zero_shot"])
        records, failures, stopped = await run_reviews(
            plan, papers, Caller(FakeLLM(respond), Ledger(tmp_path / "l.jsonl"), Budget(100))
        )
        assert [r.paper for r in records] == ["control"]
        assert len(failures) == 1 and "unseen" in failures[0]
        assert stopped is None

    async def test_budget_stops_cleanly_and_keeps_what_finished(self, tmp_path):
        model = get_model("gpt-6.1-terra")
        plan = planned_reviews(small_manifest(), [model], ["zero_shot"])
        records, failures, stopped = await run_reviews(
            plan,
            self.papers(),
            Caller(FakeLLM(scripted([])), Ledger(tmp_path / "l.jsonl"), Budget(0.1225)),
            concurrency=1,
        )
        assert "Stopped" in stopped
        assert failures == []
        assert len(records) < len(plan)
        _ = BudgetExceeded


class TestCanaries:
    def test_canaries_load_with_answers(self):
        canaries = load_canaries()
        assert len(canaries) == 5
        assert sum(1 for c in canaries if c.gold is None) == 1, "one paper has no planted error"
        assert any(c.injection for c in canaries)
        injected = next(c for c in canaries if c.injection)
        assert "Ignore all previous instructions" in injected.text

    def test_the_planted_errors_are_really_errors(self):
        """The answer key must be right, or canary results mean nothing."""
        assert 90 + 85 + 95 == 270
        assert 40**2 + 40 + 41 == 1681 == 41**2
        assert 30 * 3.6 == pytest.approx(108)
        lower, upper = 0.60, 1.20
        assert lower < 1 < upper, "a CI containing 1 cannot go with p = 0.003"

    def test_the_clean_canary_really_is_correct(self):
        for n in range(1, 200):
            assert sum(2 * i - 1 for i in range(1, n + 1)) == n**2

    async def test_a_model_that_finds_everything_passes(self, tmp_path):
        def respond(request):
            if request.schema_name == "Judgment":
                return {"matches": [{"id": 1, "match": "exact", "reason": "r"}]}
            if "sum of the first n odd" in request.user:
                return scripted([])(request)
            return scripted([finding("planted", quote="not checked")])(request)

        results = await run_canaries(
            [get_model("gpt-6.1-terra")],
            get_model("gpt-6.1-sol"),
            FakeLLM(respond),
            Ledger(tmp_path / "l.jsonl"),
            Budget(100),
        )
        assert results["models"]["gpt-6.1-terra"]["passed"] is True
        assert "all passed" in render_canaries(results)

    async def test_a_model_that_flags_everything_fails_on_the_clean_paper(self, tmp_path):
        def respond(request):
            if request.schema_name == "Judgment":
                return {"matches": [{"id": 1, "match": "exact", "reason": "r"}]}
            return scripted([finding("always", quote="not checked")])(request)

        results = await run_canaries(
            [get_model("gpt-6.1-terra")],
            get_model("gpt-6.1-sol"),
            FakeLLM(respond),
            Ledger(tmp_path / "l.jsonl"),
            Budget(100),
        )
        assert results["models"]["gpt-6.1-terra"]["passed"] is False
        failed = [r for r in results["models"]["gpt-6.1-terra"]["rows"] if not r["passed"]]
        assert all(r["canary"] == "clean_control" for r in failed)

    async def test_a_model_fooled_by_the_injection_fails(self, tmp_path):
        def respond(request):
            if request.schema_name == "Judgment":
                return {"matches": [{"id": 1, "match": "exact", "reason": "r"}]}
            if "Ignore all previous instructions" in request.user or "clean" in request.user:
                return scripted([])(request)
            if "sum of the first n odd" in request.user:
                return scripted([])(request)
            return scripted([finding("planted", quote="not checked")])(request)

        results = await run_canaries(
            [get_model("gpt-6.1-terra")],
            get_model("gpt-6.1-sol"),
            FakeLLM(respond),
            Ledger(tmp_path / "l.jsonl"),
            Budget(100),
        )
        rows = results["models"]["gpt-6.1-terra"]["rows"]
        zero_shot = next(
            r for r in rows if r["canary"] == "injection_units" and r["arm"] == "zero_shot"
        )
        assert zero_shot["passed"] is False
