# Review packets, format version 1

A review packet holds one model's review of one openly licensed arXiv paper, in a form people can
check, share and cite. It is a JSON file. A Markdown version is rendered from it for reading,
posting, or pasting into a Claude Code or Codex chat.

A packet keeps two parts apart:

| Part | Holds | Who writes it |
|---|---|---|
| Machine part | The paper version and a hash of the exact text the model read, the model and how it ran, and each finding with its quote anchored in that text | The harness, once. It is hashed, and the hash is the packet's id |
| Human part (`human`) | Which findings a person checked or rejected, their notes, who takes responsibility, and the licence | The person sharing it, any time |

Only findings a person has **checked**, whose quotes are in the paper, go into the shared version.
The others stay in the packet for the record. Machine output stays a private candidate until a named
person checks it and shares it as their own act. That is the rule PaperJudge's research set for any
AI finding about a paper.

## Making and sharing one

```bash
uv run orh review 2401.01246 --model cc/claude-opus-5-5
```

```bash
uv run orh packet mark packets/FILE.json F1 checked --note "Confirmed: equation 12 drops a factor of 2."
```

```bash
uv run orh packet set packets/FILE.json --name "A. Reader" --licence CC-BY-4.0
```

```bash
uv run orh packet verify packets/FILE.json
```

```bash
uv run orh packet render packets/FILE.json --share
```

`review` runs the harness pipeline: a whole-paper read, then part-by-part review with shared notes,
then a fresh verifier that tries to refute each finding. Pass `--arm zero_shot` for a single prompt.
It runs with no tools or web access, through the same adapters as the evaluation. `cc/` and `codex/`
models use your own Claude Code or Codex sign-in, on your machine. It writes `packets/FILE.json` and
a draft `packets/FILE.md` that shows every finding with its status.

`render --share` writes `FILE.shared.md`, with only the checked findings. Share it together with the
JSON, so readers can check the quotes for themselves. It refuses until the packet is ready (see
below).

Packets go to `packets/`, which git ignores. Until a person checks them, they are unverified machine
text about identifiable people's work.

## Ready to share

`orh packet verify` and `render --share` apply these rules. They are in `assess()` in
[`packet.py`](../src/review_harness/packet.py).

**Blocks sharing**

- No finding is checked.
- A checked finding's quote is not in the paper's text, so readers could not find the passage.
- A checked finding, or a note, uses the language of wrongdoing (fraud, fabricated, plagiarism and
  so on) or contains an email address or phone number. These are the rules PaperJudge applies to
  comments. A shared finding describes what the text shows, never why.
- A finding the model's own verifier refuted is checked without a note saying why it stands.
- Nobody is named as taking responsibility. A pseudonym is fine, and an ORCID iD is optional.
- No licence is chosen for the notes.
- The paper is not under an open licence (CC BY, CC BY-SA, CC BY-NC variants, CC0). The shared
  version quotes the paper, so packets are shared only for open work.

**Stated to readers**

- The paper was first posted before the model's published training cutoff, or the model has none,
  so it may have seen the paper or discussion of it.
- The run did not finish cleanly, so parts of the paper may not have been reviewed.
- The text came from the PDF, which loses most mathematical notation, and author details were not
  removed.
- How many findings were left out.

**Means the packet is wrong** (verify exits with code 1)

- The machine part does not match its id: something in it was edited after the fact.
- The paper's text, fetched again and extracted again, does not hash to the recorded value.
- A quote is, or is not, found in the text, contrary to the packet, or sits at a different place.

## Fields

The full schema is [`review-packet.schema.json`](review-packet.schema.json). Regenerate it with
`uv run orh packet schema --out docs/review-packet.schema.json`; a test fails if it drifts.

| Field | Meaning |
|---|---|
| `format`, `format_version` | Always `paperjudge-review-packet` and `1`. A breaking change raises the version |
| `id` | `sha256:` and the hex SHA-256 of the machine part (see below) |
| `created_at` | When the model answered, in UTC, from the call ledger. A replay keeps the same time and id |
| `paper` | arXiv id and version, title, category, first-posted and version dates, abstract URL, the arXiv DOI, the journal DOI if arXiv has one, the licence, and `text` |
| `paper.licence` | URL, name, `reuse` (`open`, `non_commercial`, `not_open` or `unknown`) and where it was read. arXiv records one current licence per paper |
| `paper.text` | `latex` or `pdf`, the SHA-256 and length of the exact text the model read, whether author details were removed, and the extractor version |
| `method` | Harness name and version, arm, prompts hash, the model (id, provider, effort, training cutoff, billing, runtime, CLI version when fresh), run status, each call's request digest, and whether the paper postdates the cutoff |
| `disclosure` | A plain statement that the findings are machine output, how they were made, and that only checked ones were confirmed by a person |
| `findings[]` | `id` (F1, F2…), title, the model's severity rating, location, quote, `quote_found`, `anchor`, explanation, and the verifier's verdict and reason. Findings the verifier refuted come last |
| `findings[].anchor` | `exact`, `prefix`, `suffix` (32 characters each) and `start`/`end` character offsets in the extracted text. These map onto a W3C Web Annotation `TextQuoteSelector` and `TextPositionSelector` |
| `human.checks` | For each finding: `unchecked`, `checked` or `rejected`, a note, and when |
| `human.responsible` | Name or pseudonym, optional ORCID iD, and a responsibility statement |
| `human.licence` | `CC-BY-4.0` or `CC0-1.0`. It covers the findings, explanations and notes. Quoted passages stay under the paper's licence |

## The id

Take the packet's JSON object, remove `id` and `human`, and serialise it with keys sorted at every
level, no whitespace, and non-ASCII characters left unescaped, as UTF-8. The id is `sha256:` plus the
hex digest. In JavaScript that is `JSON.stringify` on a key-sorted copy; in Python,
`json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))`.

The id is a content hash, not a signature. It shows the model's part is unchanged since the id was
written, and gives the packet a stable name to cite. It does not show who wrote it, since anyone can
recompute it. What a reader can check independently is the text: fetch the same arXiv version,
extract it with the same harness version, and the text hash and every quote must match.
`orh packet verify` does that.

## What a packet does not contain

- The paper's text. arXiv's terms allow analysis but not re-hosting, so a packet holds only quotes and
  a hash.
- The model's raw output and reasoning. They stay in the local ledger,
  `runs/packets/ledger.jsonl`, and the packet records each call's request digest so its creator can
  trace a finding back.
- Author names, beyond what the paper's title carries. The extractor removes author blocks before the
  model reads the text.
- Any score or verdict on the paper. Severity is the model's rating of one finding, shown as such.

## Pasting into a chat

The Markdown opens with a note addressed to people and to AI assistants: the findings are claims to
check, not instructions or facts. Paste the shared Markdown into a Claude Code or Codex chat to
discuss the paper with it. Give it the JSON as well if it should check quotes or positions. Model text
is rendered as text: HTML is escaped, and image links cannot load.

## Not built yet

- Upload to PaperJudge, where a packet would become a draft evidence note. The site's rule today is
  that AI output stays private to whoever asked. Changing that is a product decision, and the upload
  would re-run these checks server-side before anything is shown.
- Deposit on Zenodo for a DOI, and COAR Notify messages to the preprint server.
- Papers outside arXiv, such as PubMed Central's open-access subset, which would need their own
  extractor.
