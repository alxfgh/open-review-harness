import csv
from datetime import date

import httpx
import respx

from review_harness import rw_arxiv
from review_harness.papers import ArxivClient

ENTRY = """
  <entry>
    <id>http://arxiv.org/abs/{id}v{v}</id>
    <updated>2024-02-01T10:00:00Z</updated>
    <published>{published}T10:00:00Z</published>
    <title>{title}</title>
    <summary>Abstract.</summary>
    <author><name>A Person</name></author>
    <arxiv:comment>{comment}</arxiv:comment>
    {doi}
    <arxiv:primary_category term="{category}"/>
  </entry>"""
FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>{n}</opensearch:totalResults>{entries}
</feed>"""


def arxiv_entry(
    arxiv_id,
    title,
    doi=None,
    published="2023-05-01",
    comment="",
    category="cond-mat.str-el",
    version=2,
):
    doi_tag = f"<arxiv:doi>{doi}</arxiv:doi>" if doi else ""
    return ENTRY.format(
        id=arxiv_id,
        v=version,
        published=published,
        title=title,
        comment=comment,
        doi=doi_tag,
        category=category,
    )


def feed(*entries):
    return FEED.format(n=len(entries), entries="".join(entries))


def record(record_id, doi, title, reasons="Error in Analyses", retracted="2024-03-01"):
    return rw_arxiv.RwRecord(
        record_id=record_id,
        nature="Retraction",
        reasons=reasons.split(";"),
        title=title,
        journal="J",
        publisher="P",
        retraction_date=date.fromisoformat(retracted) if retracted else None,
        retraction_doi=None,
        original_date=None,
        original_doi=doi,
    )


class TestIdentifiers:
    def test_arxiv_dois(self):
        assert rw_arxiv.arxiv_id_from_doi("10.48550/arxiv.2402.08303") == "2402.08303"
        assert rw_arxiv.arxiv_id_from_doi("10.48550/arxiv.math/9810162") == "math/9810162"
        assert rw_arxiv.arxiv_id_from_doi("10.48550/arxiv.2402.08303v2") == "2402.08303"
        assert rw_arxiv.arxiv_id_from_doi("10.1103/physrevb.1.1") is None

    def test_arxiv_urls(self):
        assert rw_arxiv.arxiv_id_from_url("http://arxiv.org/abs/1207.7214v2") == "1207.7214"
        assert rw_arxiv.arxiv_id_from_url("https://arxiv.org/pdf/1207.7214") == "1207.7214"
        assert (
            rw_arxiv.arxiv_id_from_url("https://arxiv.org/abs/hep-th/9901001") == "hep-th/9901001"
        )
        assert rw_arxiv.arxiv_id_from_url("https://example.org/abs/1207.7214") is None

    def test_title_similarity_ignores_case_markup_and_punctuation(self):
        a = "Spin liquids in the $J_1$-$J_2$ model: a DMRG study"
        b = "Spin Liquids in the J1-J2 Model. A DMRG Study"
        assert rw_arxiv.title_similarity(a, b) > 0.8
        assert rw_arxiv.title_similarity(a, "Protein folding in yeast") < 0.5


class TestReasons:
    def test_only_retractions_with_an_error_reason_count(self):
        assert rw_arxiv.is_error_retraction(record("1", None, "t"))
        assert not rw_arxiv.is_error_retraction(
            record("2", None, "t", reasons="Plagiarism of Article")
        )
        expression = record("3", None, "t")
        expression.nature = "Expression of concern"
        assert not rw_arxiv.is_error_retraction(expression)

    def test_misconduct_alongside_an_error_is_flagged(self):
        assert record(
            "1", None, "t", "Error in Data;Falsification/Fabrication of Data"
        ).misconduct_signal
        assert record("1", None, "t", "Error in Data;Paper Mill").misconduct_signal
        assert not record(
            "1", None, "t", "Error in Data;Results Not Reproducible"
        ).misconduct_signal


def test_reads_the_export_format(tmp_path):
    path = tmp_path / "rw.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "record_id", "nature", "reasons", "title", "journal", "publisher",
                "retraction_date", "retraction_doi", "original_date", "original_doi",
            ]
        )  # fmt: skip
        writer.writerow(
            ["7", "Retraction", "Error in Data;Error in Methods", "T", "J", "P", "2026-08-01",
             "", "", "10.1/ABC"]
        )  # fmt: skip
    (rec,) = rw_arxiv.read_records(path)
    assert rec.reasons == ["Error in Data", "Error in Methods"]
    assert rec.error_reasons == ["Error in Data", "Error in Methods"]
    assert rec.retraction_date == date(2026, 8, 1)
    assert rec.original_doi == "10.1/abc"
    assert rec.retraction_doi is None


def test_feed_dois_are_parsed_per_entry():
    xml = feed(
        arxiv_entry("2301.00001", "A", doi="10.1103/PhysRevB.1.1"),
        arxiv_entry("2301.00002", "B"),
    )
    parsed = rw_arxiv.parse_feed_with_dois(xml)
    assert [(e.arxiv_id, dois) for e, dois in parsed] == [
        ("2301.00001", ["10.1103/physrevb.1.1"]),
        ("2301.00002", []),
    ]


@respx.mock
async def test_a_failing_batch_is_split_so_one_bad_doi_loses_only_itself():
    good = arxiv_entry("2301.00001", "A", doi="10.1/good")

    def respond(request: httpx.Request) -> httpx.Response:
        query = request.url.params["search_query"]
        if "10.1/bad" in query:
            return httpx.Response(400, text="bad query")
        return httpx.Response(200, text=feed(good) if "10.1/good" in query else feed())

    respx.get(rw_arxiv.ARXIV_API).mock(side_effect=respond)
    arxiv = ArxivClient(httpx.AsyncClient(), min_interval=0)
    progress = rw_arxiv.Progress()
    try:
        found = await rw_arxiv.search_arxiv_dois(
            arxiv, ["10.1/good", "10.1/bad", "10.1/none"], progress, batch=3
        )
    finally:
        await arxiv.aclose()
    assert list(found) == ["10.1/good"]
    assert len(progress.failed_batches) == 1
    assert "10.1/bad" in progress.failed_batches[0]


@respx.mock
async def test_doi_search_only_accepts_an_exact_listed_doi():
    lookalike = arxiv_entry("2301.00003", "C", doi="10.1/good2")
    respx.get(rw_arxiv.ARXIV_API).respond(200, text=feed(lookalike))
    arxiv = ArxivClient(httpx.AsyncClient(), min_interval=0)
    try:
        found = await rw_arxiv.search_arxiv_dois(arxiv, ["10.1/good"], rw_arxiv.Progress())
    finally:
        await arxiv.aclose()
    assert found == {}


@respx.mock
async def test_matching_tries_each_method_in_turn():
    records = [
        record("1", "10.48550/arxiv.2301.00001", "Direct paper"),
        record("2", "10.1/listed", "Listed on arXiv"),
        record("3", "10.1/openalex", "Found through OpenAlex grouping", retracted="2026-08-01"),
        record("4", "10.1/wrong", "A completely different title here"),
        record("5", None, "Only findable by its exact long title", retracted="2026-09-01"),
        record("6", None, "Seen retraction not title searched", retracted="2024-01-01"),
    ]
    by_id = {
        "2301.00001": arxiv_entry("2301.00001", "Direct paper"),
        "2302.00003": arxiv_entry("2302.00003", "Found through OpenAlex grouping"),
        "2302.00004": arxiv_entry("2302.00004", "Unrelated preprint about cats"),
    }

    def arxiv_api(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "id_list" in params:
            ids = params["id_list"].split(",")
            return httpx.Response(200, text=feed(*(by_id[i] for i in ids if i in by_id)))
        query = params["search_query"]
        if query.startswith("ti:"):
            assert "seen retraction" not in query
            hit = arxiv_entry("2309.00005", "Only findable by its exact long title")
            return httpx.Response(200, text=feed(hit))
        if "10.1/listed" in query:
            return httpx.Response(
                200, text=feed(arxiv_entry("2302.00002", "Listed on arXiv", doi="10.1/listed"))
            )
        return httpx.Response(200, text=feed())

    def openalex_api(request: httpx.Request) -> httpx.Response:
        dois = request.url.params["filter"].removeprefix("doi:").split("|")
        assert "10.1/listed" not in dois  # Already matched on arXiv, so not asked again.
        results = []
        if "10.1/openalex" in dois:
            results.append(
                {
                    "doi": "https://doi.org/10.1/openalex",
                    "locations": [{"landing_page_url": "http://arxiv.org/abs/2302.00003"}],
                }
            )
        if "10.1/wrong" in dois:
            results.append(
                {
                    "doi": "https://doi.org/10.1/wrong",
                    "locations": [{"pdf_url": "https://arxiv.org/pdf/2302.00004v1"}],
                }
            )
        return httpx.Response(200, json={"results": results})

    respx.get(rw_arxiv.ARXIV_API).mock(side_effect=arxiv_api)
    respx.get(rw_arxiv.OPENALEX_WORKS).mock(side_effect=openalex_api)
    arxiv = ArxivClient(httpx.AsyncClient(), min_interval=0)
    openalex = rw_arxiv.OpenAlex(httpx.AsyncClient(), min_interval=0)
    try:
        matches, progress = await rw_arxiv.match_records(
            records, arxiv, openalex, cutoff=date(2026, 6, 30)
        )
    finally:
        await arxiv.aclose()
        await openalex.aclose()

    assert {key: m.method for key, m in matches.items()} == {
        "1": "rw_arxiv_doi",
        "2": "arxiv_doi",
        "3": "openalex",
        "5": "arxiv_title",
    }
    assert progress.rejected_title == 1  # Record 4's OpenAlex grouping had the wrong title.

    row = rw_arxiv.candidate_row(records[2], matches["3"], date(2026, 6, 30))
    assert row["unseen"] is True
    assert row["preprint_before_retraction"] is True
    assert row["arxiv_id"] == "2302.00003"
    summary = rw_arxiv.summarise(
        [r for r in records if r.original_doi],
        [
            rw_arxiv.candidate_row(records[int(k) - 1], m, date(2026, 6, 30))
            for k, m in matches.items()
        ],
        progress,
        date(2026, 6, 30),
        None,
    )
    assert summary["matched"] == {"unseen": 2, "seen": 2, "undated": 0}
    assert summary["matched_by_field"] == {"physics": 4}


def test_field_groups():
    assert rw_arxiv.field_group("cond-mat.str-el") == "physics"
    assert rw_arxiv.field_group("hep-th") == "physics"
    assert rw_arxiv.field_group("math.CO") == "math"
    assert rw_arxiv.field_group("cs.LG") == "cs"


def test_withdrawn_preprints_are_detected_by_the_sampler_rule():
    from review_harness.papers import ArxivEntry

    def make(comment, abstract="Abstract."):
        return ArxivEntry("1", 2, "2020-01-01", "2020-02-01", "t", abstract, comment, "hep-th", "a")

    assert rw_arxiv.arxiv_withdrawn(make("This paper has been withdrawn due to an error in Eq. 3"))
    assert rw_arxiv.arxiv_withdrawn(make("", "This paper has been withdrawn by the author."))
    assert not rw_arxiv.arxiv_withdrawn(make("5 pages, 3 figures"))


@respx.mock
async def test_openalex_cache_means_a_doi_is_asked_only_once(tmp_path):
    calls = []

    def openalex_api(request: httpx.Request) -> httpx.Response:
        dois = request.url.params["filter"].removeprefix("doi:").split("|")
        calls.append(dois)
        work = {
            "doi": "https://doi.org/10.1/a",
            "locations": [{"landing_page_url": "https://arxiv.org/abs/2301.00001v3"}],
        }
        return httpx.Response(200, json={"results": [work]})

    respx.get(rw_arxiv.OPENALEX_WORKS).mock(side_effect=openalex_api)
    openalex = rw_arxiv.OpenAlex(httpx.AsyncClient(), min_interval=0)
    cache = rw_arxiv.OpenAlexCache()
    try:
        first = await rw_arxiv.search_openalex(
            openalex, ["10.1/a", "10.1/b", "10.1/a,c"], rw_arxiv.Progress(), cache=cache
        )
        cache.save(tmp_path / "cache.json")
        reloaded = rw_arxiv.OpenAlexCache.load(tmp_path / "cache.json")
        again = await rw_arxiv.search_openalex(
            openalex, ["10.1/a", "10.1/b"], rw_arxiv.Progress(), cache=reloaded
        )
        offline = await rw_arxiv.search_openalex(
            None, ["10.1/a", "10.1/new"], rw_arxiv.Progress(), cache=reloaded
        )
    finally:
        await openalex.aclose()
    assert first == again == offline == {"10.1/a": ["2301.00001"]}
    assert calls == [["10.1/a", "10.1/b"]]  # The comma DOI cannot be put in a filter.
