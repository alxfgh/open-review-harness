"""Decides which withdrawal comments can serve as an answer key. Rules only, so anyone can rerun it.

Most withdrawals are unusable. Some are not about errors at all, and many say only "due to some
mistakes", which gives nothing to check a finding against. Keeping them would make scores depend on
how generously a judge reads a vague sentence, so only specific, stated errors are kept.
"""

import re
from dataclasses import dataclass
from enum import StrEnum


class Decision(StrEnum):
    keep = "keep"
    admin_action = "admin_action"
    not_about_error = "not_about_error"
    vague = "vague"
    no_withdrawal = "no_withdrawal"
    partial_correction = "partial_correction"


class Discoverability(StrEnum):
    text = "text"  # The flaw should be visible in the paper itself.
    external = "external"  # It may depend on code or data the paper's text does not contain.


@dataclass(frozen=True)
class Classification:
    decision: Decision
    reason: str
    discoverable: Discoverability | None = None


_ADMIN = re.compile(r"arxiv admin note|admin note:", re.I)

_NEVER_ABOUT_ERROR = {
    "duplicate": r"duplicat|accidental|by mistake|submitted (in|by) error|incorrect journal template"
    r"|journal template|wrong (file|version|template)",
    "authorship": r"authorship|co-?authors?\b|consent|authori[sz]ation|without permission"
    r"|differences among the authors|attribution",
    "legal_or_policy": r"intellectual property|copyright|\bIP\b|licen[cs]e|disclosure|confidential"
    r"|publishing-policy|embargo|patent|internal review|alignment with external",
}

_USUALLY_NOT_ABOUT_ERROR = {
    "superseded": r"supersed|merged|incorporated|subsumed|replaced by|replaces arxiv"
    r"|more recent work|more general results?|already (publicly )?available|fully revised",
    "plans": r"further revision|substantial revisions?|technical improvements|refinement"
    r"|further refinement|no longer|not be maintain|reorgani[sz]ed|architectural changes",
}

_WITHDRAWAL = re.compile(
    r"\b(has been|is|was|being|are|been)\s+(hereby\s+)?(withdrawn|retracted)\b"
    r"|^\s*(paper\s+)?(withdrawn|retracted)\b"
    r"|\bwithdrawn\s+(by|because|due|after|as|for|pending|since)\b"
    r"|\b(withdraw|retract)\s+(this|the|our|my)\b"
    r"|\b(its|this)\s+withdrawal\b"
    r"|\bwithdrawal\s+(is|of this)\b",
    re.I,
)

_ERROR_SIGNAL = re.compile(
    r"\b(errors?|erroneous|mistakes?|miscalculat\w*|flaws?|flawed|incorrect(ly)?|wrong(ly)?|false"
    r"|invalid(ates?)?|gaps?|bugs?|counter-?examples?|contradict\w*|unsound|inconsisten\w*"
    r"|not correct|not hold|fails?|failed|lacks?|look-?ahead|leakage)\b"
    r"|does(n't| not) hold|do not hold",
    re.I,
)

# A location only counts if it is numbered or named. "An error in the main theorem" points nowhere,
# while "an error in Theorem 2" tells a judge exactly what a finding must identify.
_LOCATOR = re.compile(
    r"(?i:\b(?:lemma|theorem|thm|proposition|prop|corollary|cor|conjecture|claim|equations?|eqn?s?"
    r"|formula|figure|fig|table|algorithm|definition|def|appendix|assumption|sections?|secs?|step)"
    r"\b\.?)\s*\(?(?:\d|[A-Z]{1,4}\b)"
)

_EXTERNAL = re.compile(
    r"\b(code|codebase|implementation|software|script|bugs?|debugging|dataset|data leakage"
    r"|look-?ahead|re-?run|hardware|simulations?|numerics?|benchmarks?|preprocessing|pipeline"
    r"|artifacts?|audit|evaluation error)\b",
    re.I,
)
_VERSION_MENTION = re.compile(r"\bv(\d{1,2})\b", re.I)
_VERSION_RANGE = re.compile(r"\bv(\d{1,2})\s*(?:-|–|to)\s*v?(\d{1,2})\b", re.I)

# Words that say nothing about what the flaw is. A comment needs more than these to be specific.
_GENERIC = frozenset(
    """
    this that these those with from have has been being were was are the and for not but its our
    their they them because due after before while when then than also may might must should would
    could can will shall therefore thus hence however which what who whom where further about into
    paper papers manuscript article submission version versions preprint work study
    withdrawn withdraw withdrawal withdrawing retract retracted retraction request requests
    requested author authors author's authors' error errors mistake mistakes erroneous flaw flaws
    flawed incorrect wrong false invalid problem problems issue issues some several many serious
    crucial critical significant substantial major fatal important main key central results result
    conclusion conclusions proof proofs argument arguments method methods analysis
    careful carefully thorough thoroughly examination reexamination re-examination revealed
    identified found discovered noticed later earlier early recently now longer affecting affects
    affected affect pending revision revised reconstructed insufficiently verified attempt attempts
    needs need require requires required apologies apologise apologize sorry pages figures tables
    reported experimental experiments experiment repeated evaluated re-evaluated reevaluated
    corrected correct correction corrections improved improvement improvements updated update
    new revised substantially entirely completely fully preparation appear appears
    methodology methodological limitations contain contains contained relied rely relying submitted
    cited form current currently stated established regarded validity valid reliability
    reproducibility fundamental fundamentally theoretical resolved unresolved framework insufficient
    novelty relative prior within cannot gap gaps theorem theorems lemma lemmas section sections
    comprehensive restructuring revising replace replaced different repaired original approach
    developed arxiv findings finding compromises compromise data calculations calculation discovery
    substantive core derivations derivation empirical setup setups mathematical formal formulation
    quantitative estimates statement statements reference references terminology reporting design
    correctable properly considered since superseded supersedes accepted publication until should
    consequently particular specifically certain various multiple additional overall general
    generally all any both each few more most other such very only even still yet there here
    author's authors' reanalysis analyses invalidates invalidate invalidating invalidated
    corrected inconsistencies inconsistency part parts point points whole entire
    """.split()
)
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
MIN_SPECIFIC_WORDS = 4
MAX_GOLD_CHARS = 1500


def specific_words(comment: str) -> list[str]:
    """Words in a comment beyond generic talk about papers and errors."""
    return [w for w in _WORD.findall(comment) if w.lower() not in _GENERIC]


def classify(comment: str | None, abstract: str | None = None) -> Classification:
    """Decide whether this comment states a specific error that can be scored against."""
    text = " ".join((comment or "").split())
    if not text:
        opening = " ".join((abstract or "").split())[:400]
        if _WITHDRAWAL.search(opening):
            text = opening
        else:
            return Classification(Decision.no_withdrawal, "no comment")

    if _ADMIN.search(text):
        return Classification(Decision.admin_action, "arXiv admin action, not an author's reason")

    for reason, pattern in _NEVER_ABOUT_ERROR.items():
        if re.search(pattern, text, re.I):
            return Classification(Decision.not_about_error, reason)

    has_error = bool(_ERROR_SIGNAL.search(text))
    if not has_error:
        for reason, pattern in _USUALLY_NOT_ABOUT_ERROR.items():
            if re.search(pattern, text, re.I):
                return Classification(Decision.not_about_error, reason)
        return Classification(Decision.not_about_error, "no stated error")

    if not _WITHDRAWAL.search(text):
        return Classification(Decision.partial_correction, "corrects a claim, not a withdrawal")

    has_locator = bool(_LOCATOR.search(text))
    if not has_locator and len(specific_words(text)) < MIN_SPECIFIC_WORDS:
        return Classification(Decision.vague, "states an error without saying what it is")

    discoverable = Discoverability.external if _EXTERNAL.search(text) else Discoverability.text
    return Classification(Decision.keep, "specific stated error", discoverable)


def misses_reviewed_version(comment: str | None, withdrawal_version: int) -> bool:
    """True if the comment names earlier versions, but not the one that would be reviewed.

    The version before the withdrawal is what gets reviewed. If the authors locate the error only
    in an older version, it may already have been fixed by then.
    """
    text = comment or ""
    mentioned = {int(v) for v in _VERSION_MENTION.findall(text)}
    for start, end in _VERSION_RANGE.findall(text):
        mentioned.update(range(int(start), int(end) + 1))
    mentioned.discard(withdrawal_version)
    return bool(mentioned) and (withdrawal_version - 1) not in mentioned


def gold_statement(comment: str | None, abstract: str | None = None) -> str:
    """The authors' words, used as the answer key."""
    text = " ".join((comment or "").split()) or " ".join((abstract or "").split())
    return text[:MAX_GOLD_CHARS]
