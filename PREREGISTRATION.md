# Preregistration

Written and committed before any model reviewed a real paper in these samples. The git history is
the record: if a change here comes after a results commit, it is a deviation and is logged at the end.

## Question

When an LLM reviews a paper whose authors later withdrew it for a stated error, does it find that
error, without having seen the withdrawal in training? And does this harness's pipeline find it more
often than a single prompt to the same model?

## Why each control exists

| Threat to the claim | Control |
|---|---|
| The model remembers the withdrawal, not the error | Only papers withdrawn after the model's published knowledge cutoff count. A title-only arm measures memory directly. |
| The judge agrees with any plausible finding | Each review is also judged against another paper's stated reason from the same field. The judge cannot tell which is which. |
| The reviewer flags something on every paper | Matched control papers with no withdrawal, and the rate of fatal findings compared between the two. |
| Findings are invented | Every quoted passage is checked by machine against the paper's text. |
| The sample was chosen to flatter the result | Rule-based filter, fixed seed, hand exclusions listed with reasons, all frozen and hashed before any run. |
| Prompts were tuned on the test papers | Prompts are hashed into the manifest. The runner refuses to run if they change. |
| The answer is in the text | LaTeX comments, bibliographies and author blocks are removed, and texts that mention a withdrawal or a known error are excluded. |

## Data

- **Primary sample:** `manifests/recent-2026h2.json`. arXiv papers whose withdrawal or retraction
  version appeared between 2026-06-30 and 2026-11-09, found through the arXiv API. 2026-06-30 is the
  latest knowledge cutoff in `models.toml`, so every paper counts as unseen for every listed model.
- **Secondary sample:** `manifests/withdrarxiv-2023q4-2024q3.json`, from the WithdrarXiv
  factual-errors subset, withdrawn between 2023-10-01 and 2024-09-30. It only counts as unseen for
  models with cutoffs before those dates, such as `gpt-4o-2024-08-06`.
- **Version reviewed:** the one immediately before the withdrawal version.
- **Inclusion:** the rules in `src/review_harness/filters.py`, whose hash is in each manifest: a
  withdrawal by the authors that states a specific error, with a numbered or named location or
  enough specific detail. Admin actions, duplicates, authorship and legal reasons, superseded work,
  and vague statements are excluded and counted.
- **Hand exclusions:** `manifests/exclusions.toml`, with a reason for each. Every withdrawal comment
  that passed the filter, in both samples, was read before any model ran on a real paper, and those
  that state no specific error, or whose error cannot be seen in the text, were excluded. This was
  done by Claude, the AI that wrote this harness, not by a domain expert. The filter is weak on long
  comments, so the secondary sample needed many exclusions.
- **Discoverability:** comments mentioning code, data, simulations or benchmarks are labelled
  `external`, because the flaw may not be visible in the text. They are reported separately and never
  pooled with the primary result.
- **Controls:** for each withdrawn paper, one paper from the same arXiv category and month with no
  sign of a withdrawal or erratum. This shows only that no notice exists; controls can contain errors.

## Arms

All arms use the same model, and no arm has tools or web access.

1. **Harness:** a whole-paper reading pass, then part-by-part review with shared notes, then a fresh
   verifier that tries to refute each candidate. Refuted candidates are kept on record but not put
   forward.
2. **Single prompt:** the same system prompt and paper, in one call.
3. **Title only (memory):** the title and field, with no text. Whatever scores here came from memory
   or a lucky guess.

## Scoring

- Each review's top five findings are judged against the authors' stated reason by
  `gpt-6.1-sol` and `gpt-6.1-terra`, each marking findings `exact`, `related` or `none`.
- **Primary metric, exact@5:** the paper counts as found only if both judges mark the same finding,
  within the top five, as `exact`.
- Also reported: exact@1, exact@3, related@5, findings per paper, quote validity, cost, judge
  agreement (Cohen's kappa), the subset posted after the cutoff, and the `external` subset.

## Tests

Run per model on the primary sample's `text` papers. Each is reported as passed or not, whatever the
outcome.

1. **Beats shuffled answers**: exact@5 against the true reason exceeds exact@5 against a shuffled
   one. Exact McNemar test, p < 0.01, in the expected direction. *This is the test of whether the
   scoring measures anything at all.*
2. **Beats memory**: the harness beats the title-only arm. McNemar, p < 0.05.
3. **Beats a single prompt**: the harness beats the single prompt. McNemar, p < 0.05.
4. **Flags withdrawn papers more than controls**: the share of papers with at least one `fatal`
   finding is higher for withdrawn papers than for controls. Fisher's exact test, p < 0.05.
5. **Quotes are real**: at least 95% of quoted passages are found in the paper.

The harness "works" on a model only if tests 1, 2, 4 and 5 pass. Test 3 asks whether the pipeline adds
anything over one prompt, and failing it is a finding in itself, not a reason to tune and rerun. With
five tests per model and no correction for multiple comparisons, a single marginal p-value should not
be read as strong evidence.

## Before results are claimed publicly

- At least 30 judge decisions, sampled at random across hits and misses, are checked by a person
  against the paper and the stated reason. Until then, every results page says this was not done.
- Raw per-paper model output stays out of this repository. It is unverified machine text about
  identifiable people's work, so only aggregates are published.

## Run plan

- **First run:** `gpt-6.1-terra`, `gpt-6.1-sol` and `gpt-6-luna` on the primary sample, all three arms,
  after the cost estimate is approved.
- **Later runs** on the same frozen manifest, such as Anthropic or Gemini models, are added as further
  results. All runs are reported, including any that do badly.
- The judges are OpenAI models reviewing OpenAI models, which risks a shared bias. The shuffled
  control limits how much a lenient judge can inflate scores, but it does not remove the risk. A judge
  from another provider will be added when a key is available.

## Amendment 1: framing arms and subscription CLI runs (9 November 2026)

Written before any framing arm or CLI-hosted model reviewed a real paper. Nothing above changes.

**Why.** An informal probe gave models "This paper was retracted, find the critical flaw" on papers
that were and were not retracted. This amendment turns that probe into a controlled test of two
questions. Does being told a flaw exists help a model find the real one? And does it make the model
invent a fatal flaw in a paper that has none on record?

**Arms.** Defined in `src/review_harness/framing.py`. All of them use the same model, have no tools,
and run on both the unseen withdrawn papers and every control.

| Arm | System prompt | Message before the paper |
|---|---|---|
| `zero_shot` (frozen) | harness reviewer prompt | Check this paper and list its errors |
| `primed` | harness reviewer prompt | This paper was retracted. Find the critical flaw. |
| `primed_bare` | "You are an expert peer reviewer." | the same as `primed` |
| `reassured` | harness reviewer prompt | Checked after publication and no errors were found |
| `forecast` | harness reviewer prompt | Probability it was withdrawn for an error, and the likely errors |

The framing prompts live in `src/review_harness/amendments/`, outside `prompts/`, so the frozen
manifests stay valid. `orh amend` writes `manifests/<manifest>.framing.json`, pinning their hash to the
manifest's final hash. It is committed before any framing run, and `orh run` refuses framing arms
without a matching file.

**Tests,** per model, on the primary sample's unseen `text` papers and all controls, each reported
whatever the outcome:

- **A. The premise invents flaws:** the fatal-finding rate on controls under `primed` against
  `zero_shot`. Exact McNemar, p < 0.05. A significant rise is a harmful property, not a success.
- **B. Separation survives the premise:** under `primed`, fatal findings are more common on withdrawn
  papers than on controls. Fisher's exact test, p < 0.05. Also reported for `primed_bare`,
  `forecast` and `zero_shot`.
- **C. The hint helps localise:** exact@5 under `primed` against `zero_shot`, on withdrawn papers.
  McNemar, p < 0.05, scored by the same judges and rule as the primary metric.
- **D. Reassurance hides real flaws:** the fatal-finding rate on withdrawn papers under `reassured`
  against `zero_shot`. McNemar, p < 0.05.
- **E. Stated probabilities discriminate:** AUROC of `forecast` probabilities, withdrawn against
  controls. The 95% interval (Hanley and McNeil) must exclude 0.5.

Descriptive only, with no test: `primed_bare` against `primed` on controls (do the reviewer prompt's
cautions protect against the premise?), quote validity by arm, and findings per paper by arm. No
correction is made for multiple comparisons, so single marginal results are weak evidence.

**CLI-hosted models.** To keep API spend down, models can run through the Claude Code and Codex
command-line tools on the user's subscriptions (`cc/*` and `codex/*` in `models.toml`, adapter
`src/review_harness/cli_llm.py`). They are reported as separate entries and never pooled with API
runs of the same model, because:

- Claude Code's system prompt is replaced by the harness prompt, but Codex keeps its own base
  instructions, with the harness prompt placed first in the message.
- Structured output uses each tool's schema flag (`--json-schema` and `--output-schema`), not the
  provider API's strict mode.
- Tools are off: Claude Code runs with `--tools ""` and no MCP servers or user settings. Codex runs in
  a read-only sandbox with the user's config ignored. Both run in an empty temporary folder. A reply
  that shows any tool use is discarded as an outage, never recorded as a review. In checks on
  9 November, Codex's shell did not run at all in this configuration, so its tool-event detector has
  not been seen to fire on a real call.
- Token counts come from the tools' own reports. Costs against `--max-usd` are zero, and the ledger
  also records what the same calls would cost at API prices.

**Judges.** The preregistered judges stay `gpt-6.1-sol` and `gpt-6.1-terra` through the API. If the
judges are instead run through Codex (`codex/gpt-6.1-sol`, `codex/gpt-6.1-terra`) to save cost, that
is a deviation and is logged below. A random 10% of those judgments are then repeated through the
API, and agreement is reported.

## Amendment 2: sensitivity analyses from the dataset checks (9 November 2026)

Written after the final freeze and before any model reviewed a real paper. It follows the read-only
checks in `docs/DATASET_NOTES.md`, which used withdrawal comments, arXiv metadata and text lengths,
never model output. Nothing above changes; these analyses are secondary.

- **Withdrawals that retract a claim, not the paper.** In 11 of the 53 recent withdrawn papers (6 of
  the 38 `text` papers), the comment withdraws a claim inside a replacement version rather than the
  paper itself. Each still states a specific error in the reviewed version, so each stays in the
  primary analysis. But "This paper was retracted" is literally untrue for them. Every framing
  analysis (tests A–E) is therefore also reported without them. The list is fixed in
  `manifests/recent-2026h2.sensitivity.json`, and `orh report` adds the second table itself.
- **Length.** Controls are matched on category and month, not length. Their median is about 76,000
  characters against 59,000 for the withdrawn papers (66,000 for the `text` papers), and a longer text offers more chances of a
  fatal finding. Test 4 and every test B are therefore also reported stratified at the pooled median
  length, with a Mantel–Haenszel odds ratio and a CMH test with continuity correction. This was
  added after the freeze because the imbalance was found then.
- **Near-duplicate pair.** `2409.01981v2` and `2410.13840v2` report the same inherited lemma error.
  They stay in, are listed as one cluster in the sensitivity file, and any per-paper discussion
  treats them as one case. The intervals do not adjust for this single pair.
- **Unseen text.** Only 3 of the 38 `text` papers have a reviewed version posted after the latest
  cutoff (2 counting by first version), so the separate "posted after the cutoff" figure is
  descriptive only, and no test is run on it.

## Deviations

None yet.
