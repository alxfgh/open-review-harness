"""Every comment here is a real arXiv withdrawal comment seen in the 2026 harvest."""

import pytest

from review_harness.filters import (
    Decision,
    Discoverability,
    classify,
    gold_statement,
    misses_reviewed_version,
)

KEEP_IN_TEXT = [
    "Withdrawn because implication (1) => (2) in Lemma 2 is false: hard Lefschetz in degree 1 does "
    "not imply that J preserves harmonic 1-forms. Consequently, the proofs of Theorems fail",
    "The paper is withdrawn due to a gap in the proof of Lemma 4.2, which affects the main theorem",
    "Withdrawn: Theorem 2 (the Unit Gap Theorem) is false. For 3-variable parity Khrapchenko forces "
    ">= 9 formula leaves (>= 8 gates), yet a 6-gate circuit computes it",
    "Withdrawn due to an error in the proof of Lemma 3.10 (Composition Lemma), which was based on "
    "the argument in arXiv:2202.03178v2",
    "Withdrawn by the authors after identifying errors in the equilibrium characterization and "
    "comparative-statics results. A substantially reconstructed model is in preparation",
    "Withdrawn by the author. The experiments lack length controls to isolate cross-lingual transfer "
    "from the effect of shortening on detection. The paper also lacks semantic controls",
]

KEEP_EXTERNAL = [
    "This paper has been withdrawn by the author due to an implementation error that resulted in "
    "look-ahead bias (data leakage)",
]

VAGUE = [
    "I have identified substantial errors in this manuscript and therefore request its withdrawal",
    "This was an early and insufficiently verified attempt. Errors affecting the main results were "
    "later identified, and the manuscript has been withdrawn",
    "This paper has been withdrawn by the author(s) due to some mistakes in this paper",
    "Withdrawn by the author after a careful re-examination revealed errors in the proofs/results. "
    "The paper is being withdrawn pending a thorough revision.",
    "The submitted version contains a significant error affecting the reported experimental results. "
    "The experiments must be repeated and the results re-evaluated. We therefore withdraw this paper",
]

NOT_ABOUT_ERROR = [
    "Withdrawn by the authors because the authorship attribution in version 1 does not accurately "
    "reflect the final agreed contributions to the work.",
    "This manuscript has been withdrawn pending resolution of an authorship attribution dispute.",
    "This submission has been withdrawn because it was created by mistake as an accidental "
    "duplicate submission. Please refer to arXiv:2502.11041",
    "This withdrawal is requested because the current preprint was submitted in error using an "
    "incorrect journal template, resulting in significant formatting inconsistencies.",
    "Withdrawn because the manuscript inadvertently used a publisher-specific journal template",
    "This article has been withdrawn by the author due to an intellectual property ownership issue",
    "The current article involves some product IP issues and needs to be withdrawn and re-approved",
    "The manuscript was submitted to arXiv without obtaining prior authorization from all listed "
    "co-authors.",
    "Withdrawn due to information disclosure concerns.",
    "Withdrawn pending internal review and alignment with external collaborators",
    "The paper is withdrawn because a more recent work proves more general results that include our "
    "theorem as a special case",
    "This preprint has been withdrawn as its contents have been fully revised and incorporated "
    "into arXiv:2506.20942",
    "Withdrawn for further revision. The authors have identified some points in the manuscript that "
    "require more rigorous treatment",
    "Withdrawn by the author. This Master's thesis is already publicly available through the "
    "University of Padua institutional repository.",
]


@pytest.mark.parametrize("comment", KEEP_IN_TEXT)
def test_specific_errors_are_kept(comment):
    result = classify(comment)
    assert result.decision is Decision.keep
    assert result.discoverable is Discoverability.text


@pytest.mark.parametrize("comment", KEEP_EXTERNAL)
def test_errors_needing_code_or_data_are_labelled(comment):
    result = classify(comment)
    assert result.decision is Decision.keep
    assert result.discoverable is Discoverability.external


@pytest.mark.parametrize("comment", VAGUE)
def test_vague_errors_are_excluded(comment):
    assert classify(comment).decision is Decision.vague


@pytest.mark.parametrize("comment", NOT_ABOUT_ERROR)
def test_withdrawals_not_about_errors_are_excluded(comment):
    assert classify(comment).decision is Decision.not_about_error


def test_admin_withdrawals_are_excluded():
    comment = (
        "arXiv admin note: This submission has been withdrawn because it does not meet arXiv's "
        "research content quality standards"
    )
    assert classify(comment).decision is Decision.admin_action


def test_corrections_that_are_not_withdrawals_are_separate():
    comment = (
        "Version 2 corrects the component count in Version 1. The previous claim of 38 irreducible "
        "boundary components was incorrect because S-equivalence among the maximal strata"
    )
    assert classify(comment).decision is Decision.partial_correction


class TestLocators:
    @pytest.mark.parametrize(
        "comment",
        [
            "This manuscript has been withdrawn due to a mistake in Theorem 3.1",
            "Withdrawn due to errors in Sections 3.3--3.4 and 4.2.",
            "The paper is withdrawn: Eq. (16) is wrong.",
            "Withdrawn: Sec. IV wrongly states the result.",
            "Withdrawn. The statement of Theorem B is false.",
            "Withdrawn as erroneous: the definition (Def. 4) fails.",
        ],
    )
    def test_numbered_or_named_locations_count(self, comment):
        assert classify(comment).decision is Decision.keep

    @pytest.mark.parametrize(
        "comment",
        [
            "The manuscript is withdrawn due to an error in the proof of a main theorem.",
            "The claim of this paper is wrong. Thus it is withdrawn",
            "The paper has been withdrawn due to a crucial mistake in the proof",
            "Withdrawn by the author due to unresolved gaps in the proofs of the main theorems.",
            "The methodology and analysis contain substantial errors and the reported results are "
            "incorrect and should not be relied upon. It is withdrawn.",
            "Withdrawn by the author due to a fundamental gap in the theoretical analysis that "
            "cannot be resolved within the current framework, and insufficient novelty relative to "
            "prior work",
        ],
    )
    def test_unnumbered_talk_of_errors_is_vague(self, comment):
        assert classify(comment).decision is Decision.vague


class TestVersionMentions:
    @pytest.mark.parametrize(
        ("comment", "version", "expected"),
        [
            ("Withdrawn: Lemma 2 is false.", 3, False),
            ("v2's central claims are withdrawn", 3, False),
            ("The benchmark study of v1-v3 is withdrawn", 4, False),
            ("Errors in v1 are corrected here", 2, False),
            ("An error in Section 8 of v2 was found", 5, True),
            ("The error was in v1", 4, True),
            ("v5: we withdraw this", 5, False),
        ],
    )
    def test_error_must_be_in_the_version_reviewed(self, comment, version, expected):
        assert misses_reviewed_version(comment, version) is expected


def test_empty_comment():
    assert classify("").decision is Decision.no_withdrawal
    assert classify(None).decision is Decision.no_withdrawal


def test_abstract_is_used_when_the_comment_is_empty():
    abstract = "This paper has been withdrawn due to an error in the proof of Lemma 3."
    assert classify("", abstract).decision is Decision.keep
    assert gold_statement("", abstract) == abstract


def test_gold_is_the_authors_words_with_whitespace_tidied():
    assert gold_statement("  Withdrawn:\n  Lemma 2   is false ") == "Withdrawn: Lemma 2 is false"
