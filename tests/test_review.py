import pytest

from review_harness.ledger import Budget, BudgetExceeded, Caller, Ledger, ProviderError
from review_harness.llm import LLMResult, ModelDeclined
from review_harness.review import (
    Paper,
    ReviewedFinding,
    quote_found,
    rank_findings,
    review_harness,
    review_memory,
    review_zero_shot,
    split_parts,
)
from review_harness.schemas import Findings, Severity, VerdictKind, strict_schema

from .fakes import MODEL, FakeLLM, finding, scripted

PAPER_TEXT = "\\section{Intro}\nThe sample contained 250 people in total.\n" + "x " * 200
PAPER = Paper("p1", "A title", "math.AG", PAPER_TEXT)


def caller_for(llm, tmp_path, max_usd=100.0):
    return Caller(llm, Ledger(tmp_path / "ledger.jsonl"), Budget(max_usd))


class TestQuotes:
    def test_spacing_and_case_do_not_matter(self):
        assert quote_found("the  SAMPLE contained", "The sample\ncontained 250")

    def test_tidied_latex_still_counts(self):
        assert quote_found("where f(n) = n^2 + 1 for all n", "where $f(n)=n^2+1$ for all $n$")

    def test_invented_quote_fails(self):
        assert not quote_found("the sample contained 300 people", "The sample contained 250 people")

    def test_trivially_short_quote_fails(self):
        assert not quote_found("the", "the paper")


class TestSplitParts:
    def test_nothing_is_lost(self):
        text = "".join(f"\\section{{S{i}}}\n" + f"content {i} " * 900 for i in range(12))
        parts = split_parts(text, max_parts=4)
        assert len(parts) <= 4
        assert "".join(parts) == text

    def test_paper_with_no_headings(self):
        text = "word " * 50_000
        parts = split_parts(text, max_parts=6)
        assert "".join(parts) == text
        assert len(parts) <= 6

    def test_short_paper_stays_whole(self):
        assert split_parts("\\section{A}\nshort\n\\section{B}\nshort") == [
            "\\section{A}\nshort\n\\section{B}\nshort"
        ]

    def test_markdown_and_numbered_headings_split(self):
        text = ("## One\n" + "a " * 8000 + "\n") + ("## Two\n" + "b " * 8000)
        assert len(split_parts(text)) == 2


class TestRanking:
    def make(self, title, severity, verdict=None, quote=True):
        return ReviewedFinding(
            title=title,
            severity=severity,
            location="",
            evidence_quote="",
            explanation="",
            quote_verified=quote,
            verdict=verdict,
        )

    def test_upheld_serious_verified_first(self):
        ranked = rank_findings(
            [
                self.make("minor upheld", Severity.minor, VerdictKind.upheld),
                self.make("fatal uncertain", Severity.fatal, VerdictKind.uncertain),
                self.make("fatal upheld no quote", Severity.fatal, VerdictKind.upheld, quote=False),
                self.make("fatal upheld", Severity.fatal, VerdictKind.upheld),
            ]
        )
        assert [f.title for f in ranked] == [
            "fatal upheld",
            "fatal upheld no quote",
            "minor upheld",
            "fatal uncertain",
        ]


class TestSchema:
    def test_schema_fits_every_providers_strict_mode(self):
        schema = strict_schema(Findings)
        item = schema["properties"]["findings"]["items"]
        assert schema["additionalProperties"] is False
        assert item["additionalProperties"] is False
        assert set(item["required"]) == set(item["properties"])
        assert "$ref" not in str(schema) and "$defs" not in schema
        assert item["properties"]["severity"]["enum"] == ["fatal", "major", "minor"]

    def test_a_field_named_title_survives(self):
        """Regression: titles are schema metadata, but "title" is also a field of a finding."""
        item = strict_schema(Findings)["properties"]["findings"]["items"]
        assert "title" in item["properties"] and "title" in item["required"]
        assert "title" not in item["properties"]["title"], "metadata titles are still dropped"

    @pytest.mark.parametrize(
        "name",
        ["Findings", "PaperMap", "SectionReview", "Verification", "MemoryAnswer", "Judgment"],
    )
    def test_every_schema_accepts_what_its_model_produces(self, name):
        """A valid object must satisfy the schema sent to providers, field for field."""
        from review_harness import schemas

        model = getattr(schemas, name)
        sent = strict_schema(model)
        assert set(sent["required"]) == set(model.model_fields)


class TestZeroShot:
    async def test_checks_quotes_against_the_paper(self, tmp_path):
        llm = FakeLLM(
            scripted(
                [
                    finding("real", quote="The sample contained 250 people"),
                    finding("invented", quote="The sample contained 900 people"),
                ]
            )
        )
        record = await review_zero_shot(caller_for(llm, tmp_path), MODEL, PAPER)
        by_title = {f.title: f for f in record.findings}
        assert by_title["real"].quote_verified is True
        assert by_title["invented"].quote_verified is False
        assert record.findings[0].title == "real", "verifiable quotes rank higher"

    async def test_paper_text_cannot_break_out_of_the_template(self, tmp_path):
        tricky = Paper("p", "t", "c", "Text with {braces} and {paper} placeholders " * 50)
        llm = FakeLLM(scripted([]))
        await review_zero_shot(caller_for(llm, tmp_path), MODEL, tricky)
        assert "{paper} placeholders" in llm.calls[0].user


class TestHarness:
    async def test_runs_map_then_parts_then_verify_with_fresh_context(self, tmp_path):
        llm = FakeLLM(scripted([finding("A", quote="The sample contained 250 people")]))
        record = await review_harness(caller_for(llm, tmp_path), MODEL, PAPER)
        names = [c.schema_name for c in llm.calls]
        assert names[0] == "PaperMap"
        assert names[-1] == "Verification"
        assert "SectionReview" in names
        assert record.status == "ok"
        assert record.findings[0].verdict is VerdictKind.upheld

    async def test_refuted_findings_are_kept_but_not_put_forward(self, tmp_path):
        llm = FakeLLM(scripted([finding("A"), finding("B")], verdicts=["refuted", "upheld"]))
        record = await review_harness(caller_for(llm, tmp_path), MODEL, PAPER)
        assert [f.title for f in record.findings] == ["B"]
        assert [f.title for f in record.refuted] == ["A"]

    async def test_notes_carry_forward_to_later_parts(self, tmp_path):
        text = "".join(f"\\section{{S{i}}}\n" + "w " * 9000 + "\n" for i in range(3))
        llm = FakeLLM(scripted([]))
        await review_harness(caller_for(llm, tmp_path), MODEL, Paper("p", "t", "c", text))
        sections = [c.user for c in llm.calls if c.schema_name == "SectionReview"]
        assert len(sections) > 1
        assert "(none yet)" in sections[0]
        assert "a note" in sections[1]

    async def test_duplicates_across_parts_are_merged(self, tmp_path):
        text = "".join(f"\\section{{S{i}}}\n" + "w " * 9000 + "\n" for i in range(3))
        llm = FakeLLM(scripted([finding("Same")]))
        record = await review_harness(caller_for(llm, tmp_path), MODEL, Paper("p", "t", "c", text))
        assert len(record.findings) == 1

    async def test_missing_verdict_means_uncertain_not_dropped(self, tmp_path):
        def respond(request):
            if request.schema_name == "Verification":
                return {"verdicts": []}
            return scripted([finding("A")])(request)

        record = await review_harness(caller_for(FakeLLM(respond), tmp_path), MODEL, PAPER)
        assert record.findings[0].verdict is VerdictKind.uncertain

    async def test_refusal_mid_pipeline_is_reported_as_partial(self, tmp_path):
        def respond(request):
            if request.schema_name == "Verification":
                return ModelDeclined("refused", LLMResult(text=""))
            return scripted([finding("A")])(request)

        record = await review_harness(caller_for(FakeLLM(respond), tmp_path), MODEL, PAPER)
        assert record.status == "partial"
        assert record.findings[0].verdict is VerdictKind.uncertain


class TestMemory:
    async def test_receives_no_paper_text(self, tmp_path):
        llm = FakeLLM(scripted())
        record = await review_memory(caller_for(llm, tmp_path), MODEL, PAPER)
        assert "250 people" not in llm.calls[0].user
        assert "A title" in llm.calls[0].user
        assert record.recognised is False


class TestLedger:
    async def test_resuming_never_pays_twice(self, tmp_path):
        llm = FakeLLM(scripted([finding("A")]))
        first = await review_zero_shot(caller_for(llm, tmp_path), MODEL, PAPER)
        assert len(llm.calls) == 1 and first.cost_usd > 0

        fresh = FakeLLM(scripted([finding("different")]))
        second = await review_zero_shot(caller_for(fresh, tmp_path), MODEL, PAPER)
        assert fresh.calls == []
        assert second.findings[0].title == "A"

    async def test_outage_is_not_recorded_so_a_rerun_retries(self, tmp_path):
        def fail(_):
            return ProviderError("down")

        with pytest.raises(ProviderError):
            await review_zero_shot(caller_for(FakeLLM(fail), tmp_path), MODEL, PAPER)
        recovered = FakeLLM(scripted([finding("A")]))
        record = await review_zero_shot(caller_for(recovered, tmp_path), MODEL, PAPER)
        assert len(recovered.calls) == 1 and record.status == "ok"

    async def test_refusal_is_a_recorded_outcome(self, tmp_path):
        def refuse(_):
            return ModelDeclined("refused", LLMResult(text="", input_tokens=10))

        record = await review_zero_shot(caller_for(FakeLLM(refuse), tmp_path), MODEL, PAPER)
        assert record.status == "refused" and record.findings == []

    async def test_malformed_reply_gets_one_retry_then_is_recorded(self, tmp_path):
        llm = FakeLLM(lambda _: "this is not json")
        record = await review_zero_shot(caller_for(llm, tmp_path), MODEL, PAPER)
        assert len(llm.calls) == 2
        assert record.status == "invalid_output"

    async def test_changed_prompt_is_never_answered_from_the_ledger(self, tmp_path):
        llm = FakeLLM(scripted([finding("A")]))
        await review_zero_shot(caller_for(llm, tmp_path), MODEL, PAPER)
        edited = Paper("p1", "A title", "math.AG", PAPER_TEXT + " One more sentence.")
        fresh = FakeLLM(scripted([finding("B")]))
        record = await review_zero_shot(caller_for(fresh, tmp_path), MODEL, edited)
        assert len(fresh.calls) == 1 and record.findings[0].title == "B"

    async def test_retry_that_succeeds_is_used(self, tmp_path):
        replies = iter(["not json", {"findings": []}])
        llm = FakeLLM(lambda _: next(replies))
        record = await review_zero_shot(caller_for(llm, tmp_path), MODEL, PAPER)
        assert record.status == "ok"


class TestBudget:
    async def test_stops_before_the_call_that_could_overspend(self, tmp_path):
        llm = FakeLLM(scripted([]))
        with pytest.raises(BudgetExceeded):
            await review_zero_shot(caller_for(llm, tmp_path, max_usd=0.001), MODEL, PAPER)
        assert llm.calls == []

    def test_unknown_prices_are_refused_by_default(self):
        from review_harness.llm import ModelInfo

        unpriced = ModelInfo("x", "openrouter", 0.0, 0.0, "test")
        with pytest.raises(BudgetExceeded, match="No price"):
            Budget(10).check(unpriced, 100, 100)
        Budget(10, allow_unpriced=True).check(unpriced, 100, 100)
