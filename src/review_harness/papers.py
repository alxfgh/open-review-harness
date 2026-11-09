"""Downloads and cleans the version of a paper from before its withdrawal.

Everything downloaded is untrusted. Archives are read in memory with size limits and are never
extracted to disk, LaTeX is never compiled, and text goes only to models that have no tools.
Downloads stay in a local cache: arXiv's terms allow analysis but not re-hosting.

Cleaning has two purposes beyond readability. LaTeX comments are removed because authors leave notes
there, including notes about errors, which would leak the answer. Author blocks are removed because
reviewers should judge the work, not the name, though blinding is known to be only partial.
"""

import asyncio
import gzip
import hashlib
import io
import json
import re
import tarfile
import time
import xml.etree.ElementTree as ET  # noqa: S405
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import BaseModel

USER_AGENT = (
    "open-review-harness/0.1 (research evaluation; https://github.com/alxfgh/open-review-harness)"
)
ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"

MAX_DOWNLOAD_BYTES = 80 * 1024 * 1024
MAX_UNPACKED_BYTES = 120 * 1024 * 1024
MAX_TEX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 240_000
MIN_TEXT_CHARS = 2_500
MAX_MACRO_CHARS = 4_000
MAX_RETRY_AFTER_SECONDS = 300.0

_AUTHOR_COMMANDS = (
    "author",
    "authors",
    "affiliation",
    "affil",
    "address",
    "email",
    "thanks",
    "institute",
    "institution",
    "orcid",
    "curraddr",
    "urladdr",
    "homepage",
    "icmlauthor",
    "icmlaffiliation",
    "correspondingauthor",
    "acknowledgments",
    "acknowledgements",
)
_MACRO_LINE = re.compile(
    r"^\s*\\(newcommand|renewcommand|providecommand|def|DeclareMathOperator|newtheorem)\b"
)
_LEAK_PATTERNS = re.compile(
    r"(this|the)\s+(paper|article|manuscript|preprint|submission|version)\s+(has\s+been|is|was)"
    r"\s+(withdrawn|retracted)"
    r"|withdrawn\s+by\s+the\s+authors?"
    r"|(note\s+added|erratum|warning|update)\s*[:.][^.\n]{0,160}"
    r"(error|mistake|gap|incorrect|wrong|flaw)",
    re.I,
)


class PaperText(BaseModel):
    arxiv_id: str
    version: int
    source: str  # "latex" or "pdf"
    text: str
    blinded: bool

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class PaperUnavailable(Exception):
    pass


@dataclass(frozen=True)
class ArxivEntry:
    arxiv_id: str
    version: int
    published: str
    updated: str
    title: str
    abstract: str
    comment: str
    category: str
    first_author: str


class ArxivClient:
    """Keeps to arXiv's limit of one request every three seconds, across all request types."""

    MIN_INTERVAL_SECONDS = 3.1

    def __init__(self, client: httpx.AsyncClient | None = None, min_interval: float | None = None):
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, read=120.0),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        self.min_interval = self.MIN_INTERVAL_SECONDS if min_interval is None else min_interval
        # After a 429 or 5xx, wait much longer than the normal pace: arXiv throttles for minutes.
        self.retry_base = 20.0 if min_interval is None else min_interval * 6
        self._next_time = 0.0
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next_time)
            self._next_time = start + self.min_interval
        if start > now:
            await asyncio.sleep(start - now)

    async def get(self, url: str, params: dict | None = None, attempts: int = 5) -> httpx.Response:
        last = "no attempt"
        for attempt in range(attempts):
            await self._wait()
            wait = self.retry_base * (attempt + 1)
            try:
                response = await self.client.get(url, params=params)
            except httpx.HTTPError as error:
                last = type(error).__name__
            else:
                if response.status_code not in (429, 500, 502, 503, 504):
                    return response
                last = f"HTTP {response.status_code}"
                retry_after = response.headers.get("retry-after", "")
                if retry_after.isdigit():
                    wait = max(wait, min(float(retry_after), MAX_RETRY_AFTER_SECONDS))
            if attempt < attempts - 1:
                # Hold back every request, not just this one, so a throttle is respected globally.
                async with self._lock:
                    self._next_time = max(self._next_time, time.monotonic() + wait)
        raise PaperUnavailable(f"{url}: {last} after {attempts} attempts")

    async def query(self, **params: object) -> tuple[list[ArxivEntry], int]:
        response = await self.get(ARXIV_API, params=params)
        if response.status_code != 200:
            raise PaperUnavailable(f"arXiv API returned {response.status_code}")
        return parse_feed(response.text)


def _text_of(entry: ET.Element, tag: str) -> str:
    node = entry.find(tag)
    return " ".join(node.text.split()) if node is not None and node.text else ""


def parse_feed(xml_text: str) -> tuple[list[ArxivEntry], int]:
    """Parse an arXiv Atom feed. Returns entries and the total number of results."""
    root = ET.fromstring(xml_text)  # noqa: S314  (arXiv's own API, over HTTPS)
    total_node = root.find(f"{OPENSEARCH}totalResults")
    total = int(total_node.text) if total_node is not None and total_node.text else 0
    entries = []
    for entry in root.findall(f"{ATOM}entry"):
        versioned = _text_of(entry, f"{ATOM}id").split("/abs/")[-1]
        match = re.match(r"(.+)v(\d+)$", versioned)
        if not match:
            continue
        category = entry.find(f"{ARXIV_NS}primary_category")
        author = entry.find(f"{ATOM}author/{ATOM}name")
        entries.append(
            ArxivEntry(
                arxiv_id=match.group(1),
                version=int(match.group(2)),
                published=_text_of(entry, f"{ATOM}published")[:10],
                updated=_text_of(entry, f"{ATOM}updated")[:10],
                title=_text_of(entry, f"{ATOM}title"),
                abstract=_text_of(entry, f"{ATOM}summary"),
                comment=_text_of(entry, f"{ARXIV_NS}comment"),
                category=category.attrib.get("term", "") if category is not None else "",
                first_author=(author.text or "") if author is not None else "",
            )
        )
    return entries, total


def strip_comments(tex: str) -> str:
    """Remove LaTeX comments, which can hold notes about errors or text aimed at reviewers."""
    tex = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", " ", tex, flags=re.S)
    tex = re.sub(r"\\iffalse\b.*?\\fi\b", " ", tex, flags=re.S)
    return re.sub(r"(?<!\\)(?:\\\\)*%.*", "", tex)


def remove_command(tex: str, name: str) -> str:
    """Remove \\name[..]{...} with balanced braces, which a regular expression cannot do."""
    pattern = re.compile(r"\\" + name + r"\*?\s*(\[[^\]]*\]\s*)?\{")
    out = []
    position = 0
    while True:
        match = pattern.search(tex, position)
        if not match:
            out.append(tex[position:])
            return "".join(out)
        out.append(tex[position : match.start()])
        depth = 1
        index = match.end()
        while index < len(tex) and depth:
            char = tex[index]
            if char == "\\":
                index += 2
                continue
            depth += {"{": 1, "}": -1}.get(char, 0)
            index += 1
        position = index


def _safe_gunzip(data: bytes) -> bytes:
    """Decompress with a limit, so a small archive cannot expand to fill memory."""
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
        out = stream.read(MAX_UNPACKED_BYTES + 1)
    if len(out) > MAX_UNPACKED_BYTES:
        raise PaperUnavailable("archive too large when unpacked")
    return out


def read_tex_members(data: bytes) -> dict[str, str]:
    """Read .tex files from an arXiv source download, in memory, whatever form it comes in."""
    if data[:2] == b"\x1f\x8b":
        try:
            data = _safe_gunzip(data)
        except (OSError, EOFError) as error:
            raise PaperUnavailable("corrupt archive") from error
    members: dict[str, str] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            total = 0
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith(".tex"):
                    continue
                if member.size > MAX_TEX_MEMBER_BYTES:
                    continue
                total += member.size
                if total > MAX_UNPACKED_BYTES:
                    raise PaperUnavailable("archive too large when unpacked")
                handle = archive.extractfile(member)
                if handle is not None:
                    members[member.name.replace("\\", "/")] = handle.read().decode(
                        "utf-8", errors="replace"
                    )
    except tarfile.TarError:
        text = data[:MAX_TEX_MEMBER_BYTES].decode("utf-8", errors="replace")
        if "\\begin{document}" in text or "\\documentclass" in text or "\\section" in text:
            members["main.tex"] = text
    return members


def _find_main(members: dict[str, str]) -> str | None:
    candidates = [name for name, text in members.items() if "\\begin{document}" in text]
    if candidates:
        # A real paper's main file has the most content; supplements are usually shorter.
        return max(candidates, key=lambda name: len(members[name]))
    return max(members, key=lambda name: len(members[name])) if members else None


def flatten_latex(members: dict[str, str]) -> str | None:
    """Join a multi-file paper into one text by following \\input and \\include."""
    main = _find_main(members)
    if main is None:
        return None
    by_stem = {name.rsplit(".", 1)[0].lower(): text for name, text in members.items()}
    by_base = {
        name.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower(): text for name, text in members.items()
    }

    def expand(text: str, depth: int) -> str:
        text = strip_comments(text)
        if depth >= 6:
            return text

        def replace(match: re.Match) -> str:
            target = match.group(1).strip().lower().removesuffix(".tex").removeprefix("./")
            found = by_stem.get(target) or by_base.get(target.rsplit("/", 1)[-1])
            return expand(found, depth + 1) if found is not None else ""

        return re.sub(r"\\(?:input|include)\s*\{([^}]*)\}", replace, text)

    return expand(members[main], 0)


def clean_latex(tex: str) -> str:
    """Keep the body and the macros needed to read it, without comments or author details."""
    tex = strip_comments(tex)
    preamble, separator, body = tex.partition("\\begin{document}")
    if not separator:
        preamble, body = "", tex
    body = body.split("\\end{document}", 1)[0]

    macros = [line.strip() for line in preamble.splitlines() if _MACRO_LINE.match(line)]
    macro_text = "\n".join(macros)[:MAX_MACRO_CHARS]

    title_match = re.search(r"\\title\s*(\[[^\]]*\]\s*)?\{", preamble)
    title_block = ""
    if title_match:
        remainder = preamble[title_match.start() :]
        without = remove_command(remainder, "title")
        title_block = remainder[: len(remainder) - len(without)]

    combined = f"{title_block}\n{body}"
    for command in _AUTHOR_COMMANDS:
        combined = remove_command(combined, command)
    # Bibliographies add length, and citing a later correction would leak the answer.
    combined = re.sub(
        r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ", combined, flags=re.S
    )
    combined = re.sub(r"\n{3,}", "\n\n", combined)
    if macro_text:
        combined = f"% Macros used below\n{macro_text}\n\n{combined}"
    return combined.strip()


def pdf_text(data: bytes) -> str:
    """Fallback for papers with no LaTeX source. Math comes through poorly, so it is flagged."""
    import pypdfium2

    document = pypdfium2.PdfDocument(data)
    try:
        pages = []
        for index in range(len(document)):
            page = document[index]
            textpage = page.get_textpage()
            pages.append(textpage.get_text_range())
            textpage.close()
            page.close()
    finally:
        document.close()
    return "\n\n".join(pages).strip()


def find_leaks(text: str) -> list[str]:
    """Signs that the text itself gives the answer away, such as a note about a known error."""
    return [match.group(0)[:120] for match in _LEAK_PATTERNS.finditer(text)]


async def fetch_paper(
    arxiv: ArxivClient, arxiv_id: str, version: int, cache_dir: Path
) -> PaperText:
    """Fetch one specific version. arXiv versions are immutable, so the cache never goes stale."""
    safe_name = re.sub(r"[^A-Za-z0-9.\-]", "_", f"{arxiv_id}v{version}")
    cache_file = cache_dir / f"{safe_name}.json"
    cached = await asyncio.to_thread(_read_cache, cache_file)
    if cached is not None:
        return cached

    versioned = f"{arxiv_id}v{version}"
    response = await arxiv.get(f"https://arxiv.org/e-print/{versioned}")
    if response.status_code != 200:
        raise PaperUnavailable(f"source returned {response.status_code}")
    if len(response.content) > MAX_DOWNLOAD_BYTES:
        raise PaperUnavailable("download too large")

    paper: PaperText | None = None
    if response.content[:4] != b"%PDF":
        flattened = flatten_latex(read_tex_members(response.content))
        if flattened:
            paper = PaperText(
                arxiv_id=arxiv_id,
                version=version,
                source="latex",
                text=clean_latex(flattened),
                blinded=True,
            )
    if paper is None:
        data = response.content
        if data[:4] != b"%PDF":
            pdf_response = await arxiv.get(f"https://arxiv.org/pdf/{versioned}")
            if pdf_response.status_code != 200:
                raise PaperUnavailable(f"pdf returned {pdf_response.status_code}")
            data = pdf_response.content
        try:
            text = pdf_text(data)
        except Exception as error:
            raise PaperUnavailable(f"pdf unreadable: {type(error).__name__}") from error
        paper = PaperText(
            arxiv_id=arxiv_id, version=version, source="pdf", text=text, blinded=False
        )

    await asyncio.to_thread(_write_cache, cache_file, paper)
    return paper


def _read_cache(path: Path) -> PaperText | None:
    if not path.exists():
        return None
    return PaperText.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _write_cache(path: Path, paper: PaperText) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(paper.model_dump_json(), encoding="utf-8")
