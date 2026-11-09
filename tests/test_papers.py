import gzip
import io
import tarfile

import httpx
import pytest
import respx

from review_harness.papers import (
    ArxivClient,
    PaperUnavailable,
    clean_latex,
    fetch_paper,
    find_leaks,
    flatten_latex,
    parse_feed,
    read_tex_members,
    remove_command,
    strip_comments,
)

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>7432</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2403.12304v3</id>
    <updated>2026-10-15T10:00:00Z</updated>
    <published>2024-03-18T10:00:00Z</published>
    <title>A paper
      over two lines</title>
    <summary>Abstract text.</summary>
    <author><name>First Author</name></author>
    <author><name>Second Author</name></author>
    <arxiv:comment>Withdrawn because Lemma 2 is false</arxiv:comment>
    <arxiv:primary_category term="math.SG"/>
  </entry>
</feed>"""


def make_tar(files: dict[str, str], gz: bool = True) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz" if gz else "w") as archive:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_parse_feed():
    entries, total = parse_feed(FEED)
    assert total == 7432
    entry = entries[0]
    assert entry.arxiv_id == "2403.12304"
    assert entry.version == 3
    assert entry.published == "2024-03-18"
    assert entry.updated == "2026-10-15"
    assert entry.title == "A paper over two lines"
    assert entry.comment == "Withdrawn because Lemma 2 is false"
    assert entry.category == "math.SG"


class TestComments:
    def test_comments_are_removed_because_they_can_hold_the_answer(self):
        tex = "Theorem holds. % TODO this proof has a gap, fix before submitting\nNext line."
        assert "gap" not in strip_comments(tex)
        assert "Next line." in strip_comments(tex)

    def test_escaped_percent_is_kept(self):
        assert "50\\% of cases" in strip_comments("in 50\\% of cases % hidden")

    def test_comment_environments_and_iffalse_are_removed(self):
        tex = "a \\begin{comment}secret\\end{comment} b \\iffalse hidden \\fi c"
        out = strip_comments(tex)
        assert "secret" not in out and "hidden" not in out
        assert "a" in out and "b" in out and "c" in out


class TestRemoveCommand:
    def test_balanced_braces(self):
        tex = r"before \author{Jane {Q} Doe \and Sam} after"
        assert remove_command(tex, "author") == "before  after"

    def test_optional_argument_and_star(self):
        assert remove_command(r"x\author*[1]{Name}y", "author") == "xy"

    def test_escaped_braces_do_not_end_it_early(self):
        assert remove_command(r"\thanks{a \} b} rest", "thanks") == " rest"

    def test_unterminated_command_does_not_hang(self):
        assert remove_command(r"\author{never closed", "author") == ""


class TestArchives:
    def test_tar_gz(self):
        members = read_tex_members(make_tar({"main.tex": "x", "fig.png": "y", "sub/a.tex": "z"}))
        assert set(members) == {"main.tex", "sub/a.tex"}

    def test_single_gzipped_file(self):
        members = read_tex_members(gzip.compress(b"\\documentclass{article}\\begin{document}x"))
        assert list(members) == ["main.tex"]

    def test_path_traversal_names_are_only_ever_dictionary_keys(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        members = read_tex_members(make_tar({"../../evil.tex": "boom"}))
        assert members == {"../../evil.tex": "boom"}
        assert list(tmp_path.iterdir()) == [], "nothing may be written to disk"

    def test_decompression_bomb_is_refused(self, monkeypatch):
        monkeypatch.setattr("review_harness.papers.MAX_UNPACKED_BYTES", 1000)
        with pytest.raises(PaperUnavailable, match="too large"):
            read_tex_members(gzip.compress(b"a" * 5000))

    def test_corrupt_gzip_is_refused(self):
        with pytest.raises(PaperUnavailable, match="corrupt"):
            read_tex_members(b"\x1f\x8b" + b"not really gzip")

    def test_non_tex_content_gives_nothing(self):
        assert read_tex_members(b"just some bytes") == {}


class TestFlatten:
    def test_follows_inputs_and_picks_the_main_file(self):
        members = {
            "supp.tex": "\\documentclass{a}\\begin{document}short\\end{document}",
            "main.tex": "\\documentclass{a}\\begin{document}\\input{sections/intro}"
            + " body " * 50
            + "\\end{document}",
            "sections/intro.tex": "INTRO % a comment\n",
        }
        flat = flatten_latex(members)
        assert "INTRO" in flat and "a comment" not in flat

    def test_input_cycle_does_not_hang(self):
        members = {"a.tex": "\\begin{document}\\input{b}", "b.tex": "\\input{a}"}
        assert flatten_latex(members) is not None

    def test_nothing_to_flatten(self):
        assert flatten_latex({}) is None


class TestCleanLatex:
    PAPER = r"""\documentclass{article}
\newcommand{\R}{\mathbb{R}}
\usepackage{amsmath}
\title{Great Result}
\author{Jane Doe}
\begin{document}
\maketitle
\begin{abstract}We prove things.\end{abstract}
\section{Intro}
Text in \R. \thanks{Grant 123}
\begin{thebibliography}{9}\bibitem{x} A later correction.\end{thebibliography}
\end{document}
trailing junk"""

    def test_keeps_the_science(self):
        out = clean_latex(self.PAPER)
        assert "We prove things." in out and "\\section{Intro}" in out
        assert "\\title{Great Result}" in out

    def test_keeps_macros_needed_to_read_it(self):
        assert "\\newcommand{\\R}{\\mathbb{R}}" in clean_latex(self.PAPER)

    def test_removes_names_grants_and_bibliography(self):
        out = clean_latex(self.PAPER)
        assert "Jane Doe" not in out
        assert "Grant 123" not in out
        assert "later correction" not in out
        assert "trailing junk" not in out


class TestLeaks:
    @pytest.mark.parametrize(
        "text",
        [
            "This paper has been withdrawn by the author.",
            "the manuscript was retracted",
            "Note added: there is an error in Section 3.",
            "Withdrawn by the authors",
        ],
    )
    def test_text_that_gives_the_answer_away(self, text):
        assert find_leaks(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Let r be a deformation retraction of X onto A.",
            "We measure the error in the estimate.",
            "The withdrawal of water from the reservoir.",
            "Earlier work was retracted by others [3].",
        ],
    )
    def test_ordinary_scientific_language_is_not_flagged(self, text):
        assert not find_leaks(text)


class TestFetch:
    @respx.mock
    async def test_downloads_cleans_and_caches(self, tmp_path):
        tex = "\\documentclass{a}\\begin{document}" + "Body text. " * 400 + "\\end{document}"
        route = respx.get("https://arxiv.org/e-print/1234.5678v1").respond(
            200, content=make_tar({"main.tex": tex})
        )
        arxiv = ArxivClient(httpx.AsyncClient(), min_interval=0)
        first = await fetch_paper(arxiv, "1234.5678", 1, tmp_path)
        second = await fetch_paper(arxiv, "1234.5678", 1, tmp_path)
        assert first.source == "latex" and first.blinded is True
        assert first.sha256 == second.sha256
        assert route.call_count == 1, "immutable versions are never downloaded twice"

    @respx.mock
    async def test_missing_version_is_unavailable(self, tmp_path):
        respx.get("https://arxiv.org/e-print/1234.5678v9").respond(404)
        arxiv = ArxivClient(httpx.AsyncClient(), min_interval=0)
        with pytest.raises(PaperUnavailable):
            await fetch_paper(arxiv, "1234.5678", 9, tmp_path)

    @respx.mock
    async def test_server_errors_are_retried(self, tmp_path):
        tex = "\\begin{document}" + "Body. " * 600
        route = respx.get("https://arxiv.org/e-print/1234.5678v1")
        route.side_effect = [
            httpx.Response(503),
            httpx.Response(200, content=make_tar({"m.tex": tex})),
        ]
        arxiv = ArxivClient(httpx.AsyncClient(), min_interval=0)
        paper = await fetch_paper(arxiv, "1234.5678", 1, tmp_path)
        assert paper.source == "latex"

    @respx.mock
    async def test_persistent_throttling_gives_up_with_the_reason(self, tmp_path):
        route = respx.get("https://arxiv.org/e-print/1234.5678v1").respond(429)
        arxiv = ArxivClient(httpx.AsyncClient(), min_interval=0)
        with pytest.raises(PaperUnavailable, match="HTTP 429 after 5 attempts"):
            await fetch_paper(arxiv, "1234.5678", 1, tmp_path)
        assert route.call_count == 5

    @respx.mock
    async def test_retry_after_holds_back_later_requests(self):
        respx.get("https://export.arxiv.org/api/query").mock(
            side_effect=[httpx.Response(429, headers={"retry-after": "0"}), httpx.Response(200)]
        )
        arxiv = ArxivClient(httpx.AsyncClient(), min_interval=0)
        response = await arxiv.get("https://export.arxiv.org/api/query")
        assert response.status_code == 200

    def test_real_client_backs_off_far_longer_than_its_normal_pace(self):
        arxiv = ArxivClient(httpx.AsyncClient())
        assert arxiv.retry_base >= 20 > arxiv.min_interval

    async def test_request_pacing_keeps_to_arxivs_limit(self):
        import asyncio
        import time

        arxiv = ArxivClient(httpx.AsyncClient(), min_interval=0.05)
        started = time.monotonic()
        await asyncio.gather(*(arxiv._wait() for _ in range(4)))
        assert time.monotonic() - started >= 0.14
