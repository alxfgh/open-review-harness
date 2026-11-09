"""`orh review` and `orh packet`: review one open paper, check the findings, and share them properly.

    orh review 2401.01246 --model cc/claude-opus-5-5
    orh packet mark packets/....json F1 checked --note "Confirmed in equation 12."
    orh packet set packets/....json --name "A. Reader" --licence CC-BY-4.0
    orh packet verify packets/....json
    orh packet render packets/....json --share

Packets go to packets/, which is not committed: until a person checks them they are unverified
machine text about identifiable people's work, the same reason runs/ stays local.
"""

import argparse
import asyncio
import json
import re
import sys
import tempfile
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from review_harness import prompts
from review_harness.ledger import Budget, BudgetExceeded, Caller, CallOutcome, Ledger
from review_harness.llm import CLI_PROVIDERS, ModelInfo, ProviderError, ProviderLLM, get_model
from review_harness.packet import (
    OAI_ENDPOINT,
    SHARE_LICENCES,
    SHAREABLE_ARMS,
    CallRecord,
    Method,
    OaiRecord,
    PacketError,
    PacketFinding,
    PaperInfo,
    PaperLicence,
    PaperTextInfo,
    RunModel,
    assess,
    build_packet,
    check_text,
    classify_licence,
    json_schema,
    load_packet,
    locate_quote,
    machine_digest,
    mark_finding,
    parse_arxiv_ref,
    parse_oai_record,
    render_markdown,
    set_sharing,
    timestamp,
    write_text,
)
from review_harness.papers import ArxivClient, ArxivEntry, PaperText, PaperUnavailable, fetch_paper
from review_harness.review import HARNESS, REVIEWERS, Paper, ReviewRecord

DEFAULT_CACHE = Path(".cache/papers")
DEFAULT_PACKETS = Path("packets")
PACKET_LEDGER = Path("runs/packets/ledger.jsonl")
HARNESS_NAME = "open-review-harness"


def harness_version() -> str:
    try:
        return package_version(HARNESS_NAME)
    except PackageNotFoundError:
        return "unknown"


class RecordingCaller(Caller):
    """A Caller that remembers which ledger entries this review used, for the packet's call list."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.used: list[CallOutcome] = []

    async def call(self, key, model, system, user, output_model, max_output_tokens):  # type: ignore[override]
        outcome, parsed = await super().call(
            key, model, system, user, output_model, max_output_tokens
        )
        self.used.append(outcome)
        return outcome, parsed


def call_records(used: list[CallOutcome], prefix: str) -> list[CallRecord]:
    records = []
    for outcome in used:
        rest = outcome.key.removeprefix(prefix)
        retry = rest.endswith("#retry")
        parts = rest.removesuffix("#retry").split("|")
        step, digest = (parts[0], parts[1]) if len(parts) == 2 else ("review", parts[0])
        records.append(
            CallRecord(step=step, request_digest=digest, retry=retry, status=outcome.status)
        )
    return records


async def tool_version(model: ModelInfo) -> str | None:
    """The CLI's own version line, for Claude Code and Codex runs. None if it cannot be read."""
    if model.provider not in CLI_PROVIDERS:
        return None
    from review_harness.cli_llm import child_env, find_claude, find_codex

    try:
        base = find_claude() if model.provider == "claude-code" else find_codex()
        with tempfile.TemporaryDirectory(prefix="orh-version-") as empty:
            process = await asyncio.create_subprocess_exec(
                *base,
                "--version",
                cwd=empty,
                env=child_env(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, _ = await asyncio.wait_for(process.communicate(), timeout=30)
            except TimeoutError:
                process.kill()
                await process.wait()
                return None
    except (OSError, ProviderError):
        return None
    lines = out.decode("utf-8", "replace").strip().splitlines()
    return lines[0].strip()[:80] if lines else None


async def fetch_oai(arxiv: ArxivClient, arxiv_id: str) -> OaiRecord | None:
    response = await arxiv.get(
        OAI_ENDPOINT,
        params={
            "verb": "GetRecord",
            "identifier": f"oai:arXiv.org:{arxiv_id}",
            "metadataPrefix": "arXiv",
        },
    )
    return parse_oai_record(response.text) if response.status_code == 200 else None


def cache_file(cache: Path, arxiv_id: str, version: int) -> Path:
    """Where papers.fetch_paper keeps a paper, so --offline can tell whether it is there."""
    safe_name = re.sub(r"[^A-Za-z0-9.\-]", "_", f"{arxiv_id}v{version}")
    return cache / f"{safe_name}.json"


def paper_info(
    entry: ArxivEntry, text: PaperText, oai: OaiRecord | None, read_on: date
) -> PaperInfo:
    if oai is None:
        licence = PaperLicence(
            url=None,
            name="Not known: arXiv's metadata service could not be read",
            reuse="unknown",
            source=f"arXiv metadata (OAI-PMH), unavailable on {read_on.isoformat()}",
        )
    else:
        licence = classify_licence(
            oai.licence_url,
            f"arXiv metadata (OAI-PMH), read {read_on.isoformat()}. arXiv records one current "
            "licence per paper.",
        )
    return PaperInfo(
        arxiv_id=text.arxiv_id,
        version=text.version,
        title=entry.title,
        category=entry.category,
        first_posted=entry.published or None,
        version_date=entry.updated or None,
        url=f"https://arxiv.org/abs/{text.arxiv_id}v{text.version}",
        arxiv_doi=f"10.48550/arXiv.{text.arxiv_id}",
        journal_doi=oai.journal_doi if oai else None,
        licence=licence,
        text=PaperTextInfo(
            source=text.source,
            sha256=text.sha256,
            chars=len(text.text),
            blinded=text.blinded,
            extractor=f"{HARNESS_NAME} {harness_version()}, papers.fetch_paper ({text.source})",
        ),
    )


def packet_findings(record: ReviewRecord, text: str) -> list[PacketFinding]:
    findings = []
    for index, item in enumerate([*record.findings, *record.refuted], start=1):
        anchor = locate_quote(item.evidence_quote, text)
        findings.append(
            PacketFinding(
                id=f"F{index}",
                title=item.title,
                severity=item.severity,
                location=item.location,
                quote=item.evidence_quote,
                quote_found=anchor is not None,
                anchor=anchor,
                explanation=item.explanation,
                verifier=item.verdict,
                verifier_reason=item.verdict_reason,
            )
        )
    return findings


def run_method(
    model: ModelInfo,
    arm: str,
    record: ReviewRecord,
    calls: list[CallRecord],
    first_posted: str | None,
    version: str | None,
) -> Method:
    from review_harness.cli_llm import describe_runtime

    posted_after = None
    if model.knowledge_cutoff is not None and first_posted:
        posted_after = model.is_unseen(date.fromisoformat(first_posted))
    runtime = {
        key: (str(value) if value is not None else None)
        for key, value in describe_runtime(model).items()
    }
    return Method(
        harness=HARNESS_NAME,
        harness_version=harness_version(),
        arm=arm,
        prompts_sha256=prompts.prompts_sha(),
        model=RunModel(
            name=model.name,
            provider=model.provider,
            model_id=model.cli_model or model.name,
            effort=model.effort,
            knowledge_cutoff=model.knowledge_cutoff.isoformat() if model.knowledge_cutoff else None,
            billing=model.billing,
            runtime=runtime,
            tool_version=version,
        ),
        status=record.status,
        calls=calls,
        posted_after_cutoff=posted_after,
    )


def packet_stem(arxiv_id: str, version: int, model: str, arm: str, packet_id: str) -> str:
    safe = arxiv_id.replace("/", "_")
    return f"{safe}v{version}--{model.replace('/', '_')}--{arm}--{packet_id.removeprefix('sha256:')[:10]}"


def read_raw(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PacketError(f"There is no file at {path}.") from None
    except json.JSONDecodeError as error:
        raise PacketError(f"{path} is not valid JSON (line {error.lineno}).") from None
    if not isinstance(raw, dict):
        raise PacketError(f"{path} is not a review packet.")
    return raw


def write_raw(path: Path, raw: dict[str, Any]) -> None:
    path.write_text(write_text(raw), encoding="utf-8", newline="\n")


def _draft_markdown(path: Path, raw: dict[str, Any]) -> None:
    packet = load_packet(raw)
    path.with_suffix(".md").write_text(
        render_markdown(packet, share=False, json_name=path.name), encoding="utf-8", newline="\n"
    )


# Commands


async def _review(args: argparse.Namespace) -> int:
    try:
        arxiv_id, wanted = parse_arxiv_ref(args.paper)
        model = get_model(args.model)
    except (PacketError, KeyError) as error:
        print(str(error).strip("'\""), file=sys.stderr)
        return 2

    arxiv = ArxivClient()
    try:
        entries, _ = await arxiv.query(id_list=f"{arxiv_id}v{wanted}" if wanted else arxiv_id)
        if not entries:
            print(f"arXiv has no paper {args.paper}.", file=sys.stderr)
            return 2
        entry = entries[0]
        version = wanted or entry.version
        # ascii(): a Windows console that is piped cannot always encode a title's characters.
        print(f"Fetching arXiv:{arxiv_id}v{version}, {ascii(entry.title)}...", flush=True)
        text = await fetch_paper(arxiv, arxiv_id, version, args.cache)
        try:
            oai = await fetch_oai(arxiv, arxiv_id)
        except PaperUnavailable:
            oai = None
    except PaperUnavailable as error:
        print(f"Could not fetch the paper: {error}", file=sys.stderr)
        return 1
    finally:
        await arxiv.aclose()

    info = paper_info(entry, text, oai, date.today())
    if info.licence.reuse not in ("open", "non_commercial"):
        print(
            f"Note: the paper's licence is {info.licence.name.lower()}. You can review it for "
            "yourself, but a packet is shared only for openly licensed papers.",
            file=sys.stderr,
        )

    paper = Paper(
        key=f"arxiv:{arxiv_id}v{version}",
        title=entry.title,
        category=entry.category,
        text=text.text,
    )
    started = datetime.now(UTC).timestamp()
    caller = RecordingCaller(ProviderLLM(), Ledger(PACKET_LEDGER), Budget(args.max_usd))
    print(f"Reviewing with {model.name} ({args.arm})...", flush=True)
    try:
        record = await REVIEWERS[args.arm](caller, model, paper)
    except BudgetExceeded as error:
        print(f"{error} Raise --max-usd, or use a cc/ or codex/ model.", file=sys.stderr)
        return 1
    except ProviderError as error:
        print(f"The model could not be reached: {error}", file=sys.stderr)
        return 1

    used = [caller.ledger.entries[outcome.key] for outcome in caller.used]
    fresh = any(item["recorded_at"] >= started for item in used)
    reviewed_at = max((item["recorded_at"] for item in used), default=started)
    prefix = f"{model.name}|{args.arm}|{paper.key}|"
    method = run_method(
        model,
        args.arm,
        record,
        call_records(caller.used, prefix),
        info.first_posted,
        # Only a fresh run can say which CLI version answered; a replay from the ledger cannot.
        await tool_version(model) if fresh else None,
    )
    _, raw = build_packet(
        paper=info,
        method=method,
        findings=packet_findings(record, text.text),
        created_at=timestamp(datetime.fromtimestamp(reviewed_at, UTC)),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{packet_stem(arxiv_id, version, model.name, args.arm, raw['id'])}.json"
    if path.exists():
        # The same run replayed from the ledger: keep the person's checks rather than erase them.
        existing = read_raw(path)
        if existing.get("id") == raw["id"]:
            raw["human"] = existing.get("human", raw["human"])
            print("This packet already existed. Your checks in it are kept.")
    write_raw(path, raw)
    _draft_markdown(path, raw)

    put_forward = len(record.findings)
    print(
        f"Run status {record.status}: {put_forward} findings put forward, {len(record.refuted)} "
        f"refuted by the verifier.\nWrote {path} and {path.with_suffix('.md').name}."
    )
    if not raw["findings"]:
        print(
            "\nThere is nothing to check or share. No findings is not evidence that the paper is "
            "free of errors, so a packet like this stays a private record."
        )
    else:
        print(
            "\nNext: read each finding against the paper, then\n"
            f"  orh packet mark {path} F1 checked   (or rejected, with --note to say why)\n"
            f"  orh packet set {path} --name NAME --licence CC-BY-4.0\n"
            f"  orh packet render {path} --share"
        )
    return 0 if record.status in ("ok", "partial") else 1


def _print_assessment(errors: list[str], blockers: list[str], warnings: list[str]) -> None:
    for title, items in (
        ("Problems with the packet itself", errors),
        ("Before it can be shared", blockers),
        ("Readers will be told", warnings),
    ):
        if items:
            print(f"\n{title}:")
            for item in items:
                print(f"  - {item}")


async def _verify(args: argparse.Namespace) -> int:
    try:
        raw = read_raw(args.file)
    except PacketError as error:
        print(error, file=sys.stderr)
        return 2
    packet, result = assess(raw)
    text_note = "not checked"
    if packet is not None:
        cache = args.cache
        arxiv = ArxivClient()
        try:
            if (
                args.offline
                and not cache_file(cache, packet.paper.arxiv_id, packet.paper.version).exists()
            ):
                raise PaperUnavailable("not in the local cache, and --offline was given")
            text = await fetch_paper(arxiv, packet.paper.arxiv_id, packet.paper.version, cache)
        except PaperUnavailable as error:
            text_note = f"not checked ({error})"
        else:
            problems = check_text(packet, text.text)
            result.errors.extend(problems)
            text_note = (
                "does not match"
                if problems
                else "matches, and every quote is where the packet says"
            )
        finally:
            await arxiv.aclose()

    if packet is not None:
        id_ok = machine_digest(raw) == packet.id
        print(
            f"{args.file.name}: arXiv:{packet.paper.arxiv_id}v{packet.paper.version}, "
            f"{len(packet.findings)} findings"
        )
        print(f"Id: {'matches the content' if id_ok else 'DOES NOT MATCH the content'}")
        print(f"Paper text: {text_note}")
    _print_assessment(result.errors, result.blockers, result.warnings)
    if result.errors:
        print("\nDo not share or rely on this packet until the problems above are explained.")
        return 1
    print(
        "\nReady to share: orh packet render FILE --share"
        if result.ready
        else "\nNot ready to share yet."
    )
    return 0


def _update(path: Path, change) -> int:  # type: ignore[no-untyped-def]
    try:
        raw = read_raw(path)
        updated = change(raw)
    except PacketError as error:
        print(error, file=sys.stderr)
        return 2
    write_raw(path, updated)
    _draft_markdown(path, updated)
    _, result = assess(updated)
    print(f"Saved {path}.")
    if result.ready:
        print("Ready to share: orh packet render FILE --share")
    elif result.blockers:
        print(f"Still needed before sharing: {result.blockers[0]}")
    return 0


def cmd_mark(args: argparse.Namespace) -> int:
    return _update(
        args.file,
        lambda raw: mark_finding(raw, args.finding, args.status, args.note, datetime.now(UTC)),
    )


def cmd_set(args: argparse.Namespace) -> int:
    if all(value is None for value in (args.name, args.orcid, args.licence, args.statement)):
        print("Give at least one of --name, --orcid, --licence or --statement.", file=sys.stderr)
        return 2
    return _update(
        args.file,
        lambda raw: set_sharing(
            raw, name=args.name, orcid=args.orcid, licence=args.licence, statement=args.statement
        ),
    )


def cmd_render(args: argparse.Namespace) -> int:
    try:
        raw = read_raw(args.file)
    except PacketError as error:
        print(error, file=sys.stderr)
        return 2
    packet, result = assess(raw)
    if packet is None or result.errors:
        _print_assessment(result.errors, [], [])
        return 1
    if args.share and not result.ready:
        print("Not ready to share yet:", file=sys.stderr)
        for blocker in result.blockers:
            print(f"  - {blocker}", file=sys.stderr)
        return 2
    out = args.out or args.file.with_name(f"{args.file.stem}{'.shared' if args.share else ''}.md")
    out.write_text(
        render_markdown(packet, share=args.share, json_name=args.file.name),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {out}.")
    if args.share:
        print("Share it together with the packet file, so readers can check the quotes.")
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    text = json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"Wrote {args.out}.")
    else:
        sys.stdout.write(text)
    return 0


def register(commands: Any) -> None:
    """Add `review` and `packet` to orh's command list."""
    review = commands.add_parser(
        "review", help="review one arXiv paper and write a review packet you can check and share"
    )
    review.add_argument("paper", help="an arXiv id, with or without version, e.g. 2401.01246v2")
    review.add_argument(
        "--model", required=True, help="e.g. cc/claude-opus-5-5 or codex/gpt-6.1-terra"
    )
    review.add_argument("--arm", choices=SHAREABLE_ARMS, default=HARNESS)
    review.add_argument(
        "--max-usd",
        type=float,
        default=0.0,
        help="cap on API spend (default 0); cc/ and codex/ models use your CLI sign-in instead",
    )
    review.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    review.add_argument("--out", type=Path, default=DEFAULT_PACKETS)
    review.set_defaults(run_async=_review)

    packet = commands.add_parser("packet", help="check, mark, verify and share review packets")
    actions = packet.add_subparsers(dest="packet_command", required=True)

    verify = actions.add_parser(
        "verify",
        help="check the id, re-check every quote against the paper, and say what is missing",
    )
    verify.add_argument("file", type=Path)
    verify.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    verify.add_argument("--offline", action="store_true", help="use only the local paper cache")
    verify.set_defaults(run_async=_verify)

    mark = actions.add_parser("mark", help="record that you checked or rejected a finding")
    mark.add_argument("file", type=Path)
    mark.add_argument("finding", help="a finding id, such as F1")
    mark.add_argument("status", choices=("checked", "rejected", "unchecked"))
    mark.add_argument("--note", help="what you found, in your own words")
    mark.set_defaults(run=cmd_mark)

    share = actions.add_parser("set", help="name who takes responsibility, and choose a licence")
    share.add_argument("file", type=Path)
    share.add_argument("--name", help="your name or a pseudonym")
    share.add_argument("--orcid", help="optional ORCID iD")
    share.add_argument("--licence", choices=tuple(SHARE_LICENCES))
    share.add_argument("--statement", help="replaces the default responsibility statement")
    share.set_defaults(run=cmd_set)

    render = actions.add_parser(
        "render", help="write the Markdown version; --share includes only checked findings"
    )
    render.add_argument("file", type=Path)
    render.add_argument("--share", action="store_true")
    render.add_argument("--out", type=Path)
    render.set_defaults(run=cmd_render)

    schema = actions.add_parser("schema", help="print or write the packet's JSON Schema")
    schema.add_argument("--out", type=Path)
    schema.set_defaults(run=cmd_schema)
