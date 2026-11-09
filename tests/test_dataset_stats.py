import json

import httpx
import pytest
import respx

from review_harness import dataset_stats
from review_harness.papers import ArxivClient


@pytest.mark.parametrize(
    ("statement", "kind"),
    [
        ("Withdrawn because Lemma 2 is false", "whole_paper"),
        ("This paper has been withdrawn due to a mistake in Theorem 3.1", "whole_paper"),
        (
            "Version 2 corrects the component count in Version 1. The previous claim of 38 "
            "components was incorrect. That claim is withdrawn.",
            "claim_in_replacement",
        ),
        (
            "Major correction. The previously claimed construction is withdrawn.",
            "claim_in_replacement",
        ),
        (
            "v4: scope narrowed to the theory. The benchmark study of v1-v3 is withdrawn",
            "claim_in_replacement",
        ),
    ],
)
def test_withdrawal_kind(statement, kind):
    assert dataset_stats.withdrawal_kind(statement) == kind


def make_manifest():
    def paper(key, role, gold=None, matched_to=None, posted="2026-03-01", chars=10_000):
        return {
            "key": key,
            "arxiv_id": key[:-2],
            "version": 1,
            "role": role,
            "title": "t",
            "category": "math.CO",
            "posted": posted,
            "source": "latex",
            "chars": chars,
            "text_sha256": "x",
            "withdrawal_version": 2 if role == "withdrawn" else None,
            "withdrawn": "2026-08-01" if role == "withdrawn" else None,
            "discoverable": "text" if role == "withdrawn" else None,
            "gold": gold,
            "matched_to": matched_to,
            "leak_flags": [],
        }

    return {
        "name": "m",
        "created_at": "2026-11-09T00:00:00+00:00",
        "window": {"withdrawn_after": "2026-06-30", "withdrawn_until": "2026-11-09"},
        "filters_sha256": "f",
        "prompts_sha256": "p",
        "counts": {},
        "papers": [
            paper("1111.00001v1", "withdrawn", "Withdrawn due to a gap in the proof of Lemma 4.2"),
            paper("1111.00002v1", "withdrawn", "Withdrawn due to a gap in the proof of Lemma 4.3"),
            paper(
                "1111.00003v1",
                "withdrawn",
                "Withdrawn by the author Ann Smith: a mistake in the main argument",
                posted="2026-07-15",
            ),
            paper("2222.00001v1", "control", matched_to="1111.00001v1", chars=20_000),
            paper("2222.00002v1", "control", matched_to="1111.00002v1", posted="2026-04-01"),
        ],
        "exclusions": [
            {"id": "a", "reason": "vague: states an error without saying what it is"},
            {"id": "b", "reason": "manual: says only that errors exist"},
            {"id": "c", "reason": "manual: wrong version"},
        ],
    }


def test_manifest_summary_reports_the_weak_points(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps(make_manifest()), encoding="utf-8")
    summary = dataset_stats.manifest_summary(path)
    assert summary["withdrawn"] == 3
    assert summary["controls"] == 2
    assert summary["withdrawn_posted_after_window_start"] == 1
    assert summary["gold_without_numbered_location"] == ["1111.00003v1"]
    assert summary["gold_names_a_person"] == ["1111.00003v1"]
    assert summary["near_duplicate_golds"] == [
        {"a": "1111.00001v1", "b": "1111.00002v1", "similarity": pytest.approx(0.979, abs=0.01)}
    ]
    checks = summary["controls_check"]
    assert checks["withdrawn_without_control"] == ["1111.00003v1"]
    assert checks["controls_same_category"] == 2
    assert checks["controls_same_month"] == 1
    assert summary["exclusions_by_kind"] == {"manual": 2, "vague": 1}


def test_reviewed_version_dates_decide_whether_the_text_is_new(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps(make_manifest()), encoding="utf-8")
    dates = {"1111.00001v1": "2026-07-20", "1111.00002v1": "2026-03-01"}
    summary = dataset_stats.manifest_summary(path, dates)
    assert summary["reviewed_version_dates_known"] == 2
    assert summary["withdrawn_reviewed_version_after_window_start"] == 1
    assert summary["text_reviewed_version_after_window_start"] == 1


@respx.mock
async def test_version_dates_come_from_the_versioned_arxiv_entry():
    feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2601.01855v7</id>
    <updated>2026-08-30T10:00:00Z</updated>
    <published>2026-01-05T10:00:00Z</published>
    <title>T</title><summary>S</summary>
    <arxiv:primary_category term="math.LO"/>
  </entry>
</feed>"""
    route = respx.get("https://export.arxiv.org/api/query").respond(200, text=feed)
    arxiv = ArxivClient(httpx.AsyncClient(), min_interval=0)
    try:
        dates = await dataset_stats.fetch_version_dates(arxiv, ["2601.01855v7"])
    finally:
        await arxiv.aclose()
    assert dates == {"2601.01855v7": "2026-08-30"}
    assert route.calls[0].request.url.params["id_list"] == "2601.01855v7"
