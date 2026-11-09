"""What models must return, and the JSON schemas that enforce it across providers."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class Severity(StrEnum):
    fatal = "fatal"  # A main claim of the paper does not stand.
    major = "major"  # A real error that changes a result, but the main claims may survive.
    minor = "minor"  # A real but contained error.


class Finding(BaseModel):
    title: str
    severity: Severity
    location: str
    evidence_quote: str
    explanation: str


class Findings(BaseModel):
    findings: list[Finding]


class PaperMap(BaseModel):
    main_claims: list[str]
    definitions_and_assumptions: list[str]
    dependencies_between_results: list[str]


class SectionReview(BaseModel):
    findings: list[Finding]
    notes_for_later_sections: list[str]


class VerdictKind(StrEnum):
    upheld = "upheld"
    refuted = "refuted"
    uncertain = "uncertain"


class Verdict(BaseModel):
    id: int
    verdict: VerdictKind
    severity: Severity
    reason: str


class Verification(BaseModel):
    verdicts: list[Verdict]


class MemoryAnswer(BaseModel):
    recognised_this_paper: bool
    known_to_be_withdrawn_or_corrected: bool
    findings: list[Finding]


class MatchKind(StrEnum):
    exact = "exact"  # The same specific flaw the authors stated.
    related = "related"  # The same area, but not that flaw.
    none = "none"


class FindingMatch(BaseModel):
    id: int
    match: MatchKind
    reason: str


class Judgment(BaseModel):
    matches: list[FindingMatch]


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Reduce a pydantic schema to what every provider's strict mode accepts.

    Inlines references, requires every field, forbids extras, and drops titles and defaults.
    Range and length limits are left out on purpose, since providers reject them inconsistently;
    pydantic checks the result afterwards.
    """
    raw = model.model_json_schema()
    definitions = raw.get("$defs", {})

    def simplify(node: Any) -> Any:
        if isinstance(node, list):
            return [simplify(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return simplify(definitions[node["$ref"].rsplit("/", 1)[-1]])
        cleaned = {}
        for key, value in node.items():
            if key == "properties":
                # Field names are data here, not schema keywords: a field may be called "title".
                cleaned[key] = {name: simplify(field) for name, field in value.items()}
            elif key not in ("title", "default", "$defs", "description"):
                cleaned[key] = simplify(value)
        if cleaned.get("type") == "object":
            cleaned["additionalProperties"] = False
            cleaned["required"] = sorted(cleaned.get("properties", {}))
        return cleaned

    return simplify(raw)
