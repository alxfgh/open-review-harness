# Open review harness

Tests whether LLM paper review finds real errors. It uses papers that their own authors withdrew for a
stated error after each model's training cutoff, so the model cannot simply remember the answer.

**Status, 9 November 2026**

- **Harness checks:** all 20 canary checks pass on `gpt-6-luna` and `gpt-6.1-terra` through the API
  ([results/canaries.md](results/canaries.md)), and on `gpt-6.1-terra` through the Codex CLI
  ([results/canaries-cli.md](results/canaries-cli.md)).
- **Samples:** frozen and committed before any real paper was reviewed, with tests fixed in
  [PREREGISTRATION.md](PREREGISTRATION.md):
  - Primary: 53 papers withdrawn after 30 June 2026 for a stated error, 38 whose error should be
    visible in the text, plus 50 matched controls.
  - Secondary: 24 WithdrarXiv papers withdrawn between October 2023 and September 2024, plus 22
    controls.
- **Real-paper results:** not run yet. Nothing here yet shows how well any model reviews real papers.

Amendment 1 adds framing arms, which tell the model the paper was retracted or that it was checked and
found clean. It also adds runs through the Claude Code and Codex command-line tools on a subscription.
Both are reported separately from the main result.

## Why a careful test is needed

It is easy to make an AI reviewer look good. Three common shortcuts each inflate scores:

- **Memory.** A model trained after a retraction can "find" the error by recalling the notice.
- **Lenient judges.** An LLM judge asked whether a finding matches a stated error tends to say yes to
  anything plausible.
- **Flagging everything.** A reviewer that reports a fatal flaw in every paper catches every withdrawn
  one.

Each has a control here:

| Shortcut | Control |
|---|---|
| Memory | Only papers withdrawn after the model's published cutoff count, and a **title-only arm** measures memory directly. |
| Lenient judge | Every review is also judged against **another paper's reason** from the same field, without the judge knowing which is which. A real effect must beat this baseline. |
| Flagging everything | **Matched control papers** with no withdrawal, from the same category and month. Withdrawn papers must draw more fatal findings. |
| Invented findings | Every quoted passage is **checked against the paper's text** by machine. |
| Choosing papers after the fact | The sample is **frozen and hashed before any run**, with every exclusion and its reason listed. |
| Tuning prompts on the test | Prompts are hashed into the frozen sample, and the runner **refuses to run** if they change. |

The canaries test the harness itself. They are five short synthetic papers, four with one planted
error each (one inside a prompt-injection attempt) and one with no error. If a capable model misses
these, the fault is in the plumbing, not the science. The first canary run caught exactly that: a
schema bug was silently discarding every finding, and it was fixed before any real paper was used.

## How it works

1. **Sample** (`orh sample`): finds arXiv withdrawals in a date window, keeps those whose comment
   states a specific error, downloads the version before the withdrawal, removes LaTeX comments,
   author details and bibliography, and picks matched controls. It writes a manifest that pins every
   paper's text by hash.
2. **Review** (`orh run`): runs three arms with the same model and no tools:
   - **harness**: a whole-paper read, part-by-part review with shared notes, then a fresh verifier
     that tries to refute each candidate;
   - **single prompt**: the same instructions and paper in one call;
   - **title only**: no paper text, to measure memory.
3. **Judge** (`orh judge`): two judge models mark each top-five finding as an exact match, related
   or unrelated, against the true reason and against a shuffled one.
4. **Report** (`orh report`): exact@5 with 95% intervals, the five preregistered tests, quote
   validity and cost. Only aggregates are published.

## Quick start

Needs Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

```bash
uv run pytest
```

The tests run offline against scripted models, with no keys and no spend. To use real models, set
whichever keys you have: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` or
`OPENROUTER_API_KEY`. Keys are read from the environment and never logged or written to disk.

```bash
uv run orh models
```

```bash
uv run orh canary --models gpt-6.1-terra --judge gpt-6.1-sol --max-usd 1
```

```bash
uv run orh estimate --manifest manifests/recent-2026h2.json --models gpt-6.1-terra
```

```bash
uv run orh run --manifest manifests/recent-2026h2.json --models gpt-6.1-terra --max-usd 20
```

```bash
uv run orh judge --manifest manifests/recent-2026h2.json --max-usd 5
```

```bash
uv run orh report --manifest manifests/recent-2026h2.json
```

Every model call is recorded in `runs/<manifest>/ledger.jsonl`, keyed by a hash of the exact request.
An interrupted run resumes without paying twice, and a changed prompt is never answered from an old
record. `--max-usd` is a hard cap: the runner stops before any call whose worst case would exceed it.

## Review packets: sharing one review properly

`orh review` runs the same pipeline on a single openly licensed arXiv paper, through your own Claude
Code or Codex sign-in, and writes a review packet. You check each finding against the paper, name
yourself (a pseudonym is fine) and choose a licence. `orh packet render --share` then writes a
Markdown version with only the findings you checked. You can post it, or paste it into a Claude Code
or Codex chat. The JSON travels with it, so anyone can re-check every quote with `orh packet verify`.

```bash
uv run orh review 2401.01246 --model codex/gpt-6.1-terra
```

The format, the sharing rules and the id are described in
[docs/REVIEW_PACKET.md](docs/REVIEW_PACKET.md). These reviews sit outside the preregistered study
and never enter its results.

## Adding a model

Add it to [src/review_harness/models.toml](src/review_harness/models.toml) with its provider, prices
and `knowledge_cutoff`, and link the page they came from. A model without a known cutoff can run but
never has its papers counted as unseen. Any OpenRouter model can be named as `openrouter/<id>`, and
is treated the same way.

## What this does and does not show

- **It measures recall on stated errors.** A finding the authors did not mention scores as a miss even
  if it is correct, so this says little about false alarms beyond the control comparison.
- **Withdrawals are a skewed sample.** They lean heavily towards maths, physics and computer science,
  and rest on authors' own descriptions, which can be incomplete or wrong.
- **Judges are LLMs.** Agreement between two judges is reported, and the shuffled baseline limits how
  much leniency can inflate scores, but no person has yet checked their decisions.
- **Cutoffs are what providers publish.** A paper posted before a cutoff may be in training data even
  though its withdrawal is not, so a separate figure covers papers posted after the cutoff.
- **The reviewed text is what this harness extracts.** Figures are not shown to models, and PDF-only
  papers lose most mathematical notation.

## Data and licences

- Code: Apache-2.0.
- Paper texts are downloaded from arXiv into a local cache and are not redistributed, in line with
  [arXiv's terms](https://info.arxiv.org/help/api/tou.html). Manifests hold only arXiv metadata.
- The secondary sample uses
  [alxfgh/withdrarxiv-factual-errors](https://huggingface.co/datasets/alxfgh/withdrarxiv-factual-errors),
  a subset of [WithdrarXiv](https://huggingface.co/datasets/darpa-scify/withdrarxiv) (Rao et al.,
  [arXiv:2412.03775](https://arxiv.org/abs/2412.03775), Apache-2.0).
- Per-paper model output stays local in `runs/` and `packets/`. It is unverified machine text about
  identifiable people's work, so it is never committed. A single packet is shared only after a
  person checks it, under their name and a licence they choose.

## Background

This harness is the evaluation gate for the review features of PaperJudge (paperjudge.com), a platform for
checking what sources record about papers. No AI review feature ships there until it passes the tests
in [PREREGISTRATION.md](PREREGISTRATION.md).
