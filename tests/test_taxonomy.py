import json

import pytest

from review_harness import taxonomy


@pytest.mark.parametrize(
    ("statement", "primary"),
    [
        ("Withdrawn because implication (1) => (2) in Lemma 2 is false", "proof"),
        ("The paper is withdrawn due to a gap in the proof of Lemma 4.2", "proof"),
        ("withdrawn due to a computational error in section 4", "calculation"),
        (
            "an algebraic error which led to the wrong conclusion that the BH is regular",
            "calculation",
        ),
        ("There is an error in eq. (26) of v1 that invalidates the main result", "calculation"),
        ("an incorrectly normalised state in Eq. 8", "calculation"),
        ("an implementation error that resulted in look-ahead bias (data leakage)", "code"),
        ("these came from a sign error in the polarization of the accompanying code", "code"),
        ("critical data leakage identified in the temporal train-test splitting", "data"),
        ("a confound in stimulus assignment that invalidates the main results", "data"),
        ("our balance reading was impacted when light was incident at an angle", "data"),
        (
            'withdrawn due to a flaw in Lemma 3.3 of "Exotic Dehn twists" by others; the proof '
            "of our main result relies crucially on this lemma",
            "citation",
        ),
        (
            "an error in the proof of Lemma 3.10, which was based on the argument in arXiv:2202.03178v2",
            "citation",
        ),
        ("the first author wrongly inserted citations on background works", "citation"),
        (
            "Paper withdrawn due to a conceptual error regarding backward travelling perturbations",
            "other",
        ),
        ("", "other"),
    ],
)
def test_primary_kind(statement, primary):
    assert taxonomy.classify_error(statement).primary == primary


def test_every_matching_kind_is_listed_in_priority_order():
    labels = taxonomy.classify_error(
        "There is an error in the proof of Theorem 2.2, so formula (2.10) is wrong and the "
        "numerical experiments in Section 4 may be incorrect"
    )
    assert labels.primary == "proof"
    assert labels.all == ("proof", "calculation", "data")


def test_tables_count_by_source_field_and_discoverability(tmp_path):
    manifest = {
        "name": "m",
        "papers": [
            {"key": "1v1", "arxiv_id": "1", "role": "withdrawn", "category": "math.CO",
             "discoverable": "text", "gold": "a gap in the proof of Lemma 2"},
            {"key": "2v1", "arxiv_id": "2", "role": "withdrawn", "category": "cs.LG",
             "discoverable": "external", "gold": "a bug in our code"},
            {"key": "3v1", "arxiv_id": "3", "role": "control", "category": "cs.LG", "gold": None},
        ],
    }  # fmt: skip
    path = tmp_path / "m.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    rows = taxonomy.label_rows(taxonomy.manifest_rows(path))
    assert [(r["key"], r["field"], r["primary"]) for r in rows] == [
        ("1v1", "math", "proof"),
        ("2v1", "cs", "code"),
    ]
    tables = taxonomy.tables(rows)
    assert tables["taxonomy_summary"] == [
        {"source": "m", "primary": "code", "papers": 1, "share_of_source": 0.5},
        {"source": "m", "primary": "proof", "papers": 1, "share_of_source": 0.5},
    ]
    assert {r["discoverable"] for r in tables["taxonomy_by_discoverability"]} == {
        "text",
        "external",
    }
