import json
from datetime import UTC, date, datetime

import httpx
import respx

from review_harness import watch
from review_harness.papers import ArxivClient, ArxivEntry

ENTRY = """
  <entry>
    <id>http://arxiv.org/abs/{id}v{v}</id>
    <updated>{updated}T10:00:00Z</updated>
    <published>2026-03-18T10:00:00Z</published>
    <title>A paper</title>
    <summary>Abstract text.</summary>
    <author><name>First Author</name></author>
    <arxiv:comment>{comment}</arxiv:comment>
    <arxiv:primary_category term="math.SG"/>
  </entry>"""
FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>{total}</opensearch:totalResults>{entries}
</feed>"""


def feed(*entries: tuple[str, int, str, str]) -> str:
    body = "".join(ENTRY.format(id=i, v=v, updated=u, comment=c) for i, v, u, c in entries)
    return FEED.format(total=len(entries), entries=body)


def entry(arxiv_id="2603.00001", version=2, comment="Withdrawn because Lemma 2 is false"):
    return ArxivEntry(
        arxiv_id=arxiv_id,
        version=version,
        published="2026-03-18",
        updated="2026-08-01",
        title="A paper",
        abstract="Abstract text.",
        comment=comment,
        category="math.SG",
        first_author="First Author",
    )


class TestRecord:
    def test_specific_error_is_kept_with_its_gold_statement(self):
        row = watch.pool_record(entry(), "run1", "2026-11-09T12:00:00+00:00", "sha")
        assert row["key"] == "2603.00001v2"
        assert row["decision"] == "keep"
        assert row["discoverable"] == "text"
        assert row["gold"] == "Withdrawn because Lemma 2 is false"
        assert row["has_earlier_version"] is True
        assert row["first_seen"] == "2026-11-09T12:00:00+00:00"

    def test_vague_withdrawal_has_no_gold(self):
        row = watch.pool_record(entry(comment="Withdrawn due to an error"), "r", "t", "sha")
        assert row["decision"] == "vague"
        assert row["gold"] is None

    def test_no_author_names_are_stored(self):
        row = watch.pool_record(entry(), "r", "t", "sha")
        assert "First Author" not in json.dumps(row)


class TestAppendOnly:
    def test_existing_records_are_never_rewritten(self):
        old = [{"key": "a v1", "first_seen": "old"}]
        fresh = [{"key": "a v1", "first_seen": "new"}, {"key": "b v1", "first_seen": "new"}]
        assert watch.new_records(old, fresh) == [{"key": "b v1", "first_seen": "new"}]

    def test_a_new_version_of_a_known_paper_is_new(self):
        old = [watch.pool_record(entry(version=2), "r", "t", "s")]
        fresh = [watch.pool_record(entry(version=3), "r", "t", "s")]
        assert [r["key"] for r in watch.new_records(old, fresh)] == ["2603.00001v3"]

    def test_duplicates_within_one_batch_are_added_once(self):
        fresh = [{"key": "a"}, {"key": "a"}]
        assert watch.new_records([], fresh) == [{"key": "a"}]


def test_resumed_window_overlaps_the_last_run():
    runs = [{"until": "2026-11-01"}, {"until": "2026-11-09"}]
    window = watch.default_window(runs, date(2026, 11, 20))
    assert window == watch.RunWindow(date(2026, 11, 2), date(2026, 11, 20))
    assert watch.default_window([], date(2026, 11, 20)) is None


def test_summary_counts_only_reviewable_withdrawals_as_usable():
    rows = [
        watch.pool_record(entry("1", 2), "r", "t", "s"),
        watch.pool_record(entry("2", 1), "r", "t", "s"),  # no earlier version to review
        watch.pool_record(entry("3", 2, "Withdrawn due to an error"), "r", "t", "s"),
    ]
    summary = watch.summarise(rows, cutoff=date(2026, 6, 30))
    assert summary["withdrawals"] == 3
    assert summary["specific_error"] == 2
    assert summary["specific_error_with_earlier_version"] == 1
    assert summary["usable_withdrawn_after_cutoff"] == 1
    assert summary["usable_posted_after_cutoff"] == 0


@respx.mock
async def test_run_appends_new_withdrawals_and_logs_the_run(tmp_path):
    first_page = feed(
        ("2603.00001", 2, "2026-08-01", "Withdrawn because Lemma 2 is false"),
        ("2603.00002", 3, "2026-07-15", "Withdrawn due to a duplicate submission"),
        ("2603.00003", 2, "2026-01-01", "Withdrawn because Lemma 2 is false"),  # outside window
    )

    def respond(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params.get("start", "0"))
        return httpx.Response(200, text=first_page if start == 0 else feed())

    respx.get("https://export.arxiv.org/api/query").mock(side_effect=respond)
    arxiv = ArxivClient(httpx.AsyncClient(), min_interval=0)
    window = watch.RunWindow(date(2026, 6, 30), date(2026, 11, 9))
    now = datetime(2026, 11, 9, 12, tzinfo=UTC)
    try:
        run = await watch.run_watch(arxiv, window, tmp_path, now=now, cutoff=date(2026, 6, 30))
        again = await watch.run_watch(arxiv, window, tmp_path, now=now)
    finally:
        await arxiv.aclose()

    assert run["harvested"] == 2
    assert run["added"] == 2
    assert run["window_summary"]["by_decision"] == {"keep": 1, "not_about_error": 1}
    assert run["window_summary"]["usable_withdrawn_after_cutoff"] == 1
    assert again["added"] == 0
    assert again["already_in_pool"] == 2

    pool = watch.read_jsonl(tmp_path / watch.POOL_FILE)
    assert [row["key"] for row in pool] == ["2603.00001v2", "2603.00002v3"]
    assert all(row["first_seen"] == "2026-11-09T12:00:00+00:00" for row in pool)
    assert len(watch.read_jsonl(tmp_path / watch.RUNS_FILE)) == 2
