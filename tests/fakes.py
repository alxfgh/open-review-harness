"""A scripted model, so the whole pipeline can be tested offline without spending anything."""

import json
from collections.abc import Callable

from review_harness.llm import LLMRequest, LLMResult, ModelDeclined, ModelInfo

MODEL = ModelInfo(
    name="fake-model",
    provider="openai",
    input_per_mtok=1.0,
    output_per_mtok=2.0,
    source="test",
    price_verified=True,
)


def finding(title="Error", severity="fatal", quote="not in paper at all", location="Section 1"):
    return {
        "title": title,
        "severity": severity,
        "location": location,
        "evidence_quote": quote,
        "explanation": f"Explanation of {title}.",
    }


class FakeLLM:
    def __init__(self, respond: Callable[[LLMRequest], dict | Exception]):
        self.respond = respond
        self.calls: list[LLMRequest] = []

    async def complete(self, model: ModelInfo, request: LLMRequest) -> LLMResult:
        self.calls.append(request)
        reply = self.respond(request)
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(
            text=reply if isinstance(reply, str) else json.dumps(reply),
            input_tokens=1000,
            output_tokens=500,
        )


def scripted(findings=(), verdicts=None, map_reply=None):
    """Respond by which kind of request this is, judged from its schema."""

    def respond(request: LLMRequest):
        name = request.schema_name
        if name == "Findings":
            return {"findings": list(findings)}
        if name == "PaperMap":
            return map_reply or {
                "main_claims": ["c"],
                "definitions_and_assumptions": ["d"],
                "dependencies_between_results": ["e"],
            }
        if name == "SectionReview":
            return {"findings": list(findings), "notes_for_later_sections": ["a note"]}
        if name == "Verification":
            count = request.user.count('"id":')
            items = verdicts or ["upheld"] * count
            return {
                "verdicts": [
                    {"id": i + 1, "verdict": v, "severity": "fatal", "reason": "because"}
                    for i, v in enumerate(items)
                ]
            }
        if name == "MemoryAnswer":
            return {
                "recognised_this_paper": False,
                "known_to_be_withdrawn_or_corrected": False,
                "findings": [],
            }
        raise AssertionError(f"unexpected schema {name}")

    return respond


__all__ = ["FakeLLM", "MODEL", "ModelDeclined", "finding", "scripted"]
