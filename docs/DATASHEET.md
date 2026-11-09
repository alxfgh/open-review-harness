# Datasheet: withdrawn and retracted papers for testing LLM review

This describes the data the harness uses to test whether an LLM reviewer finds a paper's real error.
It follows the questions of Gebru et al., "Datasheets for Datasets" (2021). The counts below are as
of 9 November 2026; [DATASET_NOTES.md](DATASET_NOTES.md) has the checks behind them.

## Motivation

**Why it exists.** To test LLM reviewers on errors that are known to be real, without letting a model
score by remembering the correction. Each paper's answer key is what its own authors (or, for the
Retraction Watch candidates, its journal) said was wrong, and the paper's withdrawal falls after the
model's training cutoff.

**Who made it.** The open review harness in this repository, as the evaluation gate for the review
features of PaperJudge. The sampling rules, hand exclusions and checks were written by Claude (an AI)
and have not yet been reviewed by a domain expert.

## Composition

There are three parts. Only the first is used for scored results.

| Part | File | What one record is | Status |
|---|---|---|---|
| Frozen samples | `manifests/*.json` | a paper version to review, with its role (withdrawn or control) and, for withdrawn papers, the authors' stated error | frozen and hashed before any model ran |
| Withdrawal pool | `pool/withdrawals.jsonl` | an arXiv withdrawal or retraction comment, its rule-based classification and when this harness first saw it | append-only, grows each run |
| Journal retractions | `pool/rw_arxiv_candidates.jsonl` (local only) | a Retraction Watch retraction for error whose paper has an arXiv version | candidates only, nothing frozen |

**Frozen samples.** Each withdrawn paper is the version immediately before its withdrawal. Each
control is a paper from an arXiv listing in the same category and month with no withdrawal or erratum
in its comment. A control is not known to be free of errors; only the absence of a notice is known.
Withdrawn papers are split by whether the flaw should be visible in the text (`text`) or may depend on
code or data the text does not contain (`external`); the two are never pooled.

**Withdrawal pool.** Every withdrawal or retraction comment found on arXiv in a window, kept or not,
with the rule decision (`keep`, `vague`, `not_about_error`, `admin_action`, `partial_correction`),
the reason, the gold statement for kept ones, the hash of the rules used and a first-seen timestamp.

**Journal retractions.** Retraction Watch records whose reasons include Error in Analyses, Error in
Data, Error in Methods or Error in Results and/or Conclusions, matched to an arXiv version. Each
carries the coded reasons, a flag for any misconduct-related reason recorded alongside, the match
method and title similarity, whether the retraction falls after the cutoff, and whether the preprint
predates the retraction.

**Counts, 9 November 2026.**

| | Records | Detail |
|---|---|---|
| `recent-2026h2` (sha `ed9630cc…`) | 53 withdrawn, 50 controls | 38 `text`, 15 `external`; withdrawn 2026-06-30 to 2026-11-09 |
| `withdrarxiv-2023q4-2024q3` (sha `e6a0193d…`) | 24 withdrawn, 22 controls | 23 `text`, 1 `external`; withdrawn 2023-10-01 to 2024-09-30 |
| Withdrawal pool | 529 withdrawals | 76 state a specific error (57 `text`, 19 `external`) |
| Journal retractions for error, with an arXiv version | 49 of 3,878 | 1 retracted after 2026-06-30, 48 before; none with a misconduct reason |

**What is not included.** Paper text (downloaded to a local cache, never redistributed), author
names as data fields, figures, and any model output about a specific paper.

**Sensitive content.** Withdrawal comments are public arXiv metadata, but some name people: the
author, or a colleague who found the error. They are kept as written because they are the answer
key; `dataset_stats` lists which gold statements name a person. The WithdrarXiv statements were
already scrubbed by that dataset and contain `[REDACTED-NAME]`.

## Collection

- **arXiv withdrawals.** The arXiv API, searching comments for withdrawal and retraction wording, sorted
  by last update, keeping papers last updated inside the window. Requests are kept to one every three
  seconds across all request types. Recall is unknown: a withdrawal whose comment uses other words
  is missed.
- **WithdrarXiv.** The factual-errors subset of WithdrarXiv (Rao et al., arXiv:2412.03775,
  Apache-2.0), dated through the arXiv API, for a window early enough to be unseen by older models.
- **Retraction Watch.** The copy the PaperJudge API service already syncs from Crossref's GitLab
  repository, read from its local database by `scripts/export_rw_retractions.sh`, so nothing is
  downloaded again. Author, institution and country columns are never stored by that service.
- **Matching to arXiv.** In order: the record's own arXiv DOI; arXiv's `doi:` search for the journal
  DOI (many DOIs per request, accepted only when arXiv lists that exact DOI); OpenAlex's works
  endpoint for the journal DOI, accepted only if the arXiv title resembles Retraction Watch's (0.6
  similarity or more); and, for retractions after the cutoff, an arXiv title search accepted at 0.9.

## Preprocessing and labelling

- **Which withdrawals can be an answer key.** Fixed rules in `src/review_harness/filters.py` (hashed
  into each manifest): an author's withdrawal that names a specific error, with a numbered location
  or enough specific words. Admin actions, duplicates, authorship and legal reasons, superseded work
  and vague statements are excluded and counted.
- **Hand exclusions.** `manifests/exclusions.toml`, with a public reason for each, decided from the
  withdrawal comments alone before any model ran. Made by Claude, not a domain expert.
- **Text.** LaTeX source where available, otherwise PDF text. LaTeX comments, author blocks,
  acknowledgements and bibliographies are removed, because authors leave notes about errors in
  comments. Texts that mention a withdrawal or a known error are excluded.
- **Error kinds.** `src/review_harness/taxonomy.py` sorts each gold statement into proof,
  calculation, data, code, citation or other by keyword rules, with root causes ranked first. A
  second, independent labelling by a model run through the Codex CLI measures how far the rules can
  be trusted (`results/dataset/taxonomy_agreement.json`).

## Uses

**Intended.** Measuring whether an LLM reviewer finds an error its authors later reported, compared
with matched papers that have no notice, with memory controlled by the training cutoff.

**Not intended.**
- Judging authors. A withdrawal for an honest error is good practice, and nothing here says otherwise.
- Treating a control, or any paper without a notice, as error-free.
- Detecting misconduct. The arXiv sample is author withdrawals for stated errors, and journal
  retractions with misconduct reasons are flagged, not studied.
- Ranking people or institutions, or publishing per-paper model output.

## Distribution and licences

- Harness code: Apache-2.0.
- Manifests and the withdrawal pool hold arXiv metadata only (identifiers, titles, categories, dates,
  comments, abstracts), which arXiv makes available under CC0. Paper text is never redistributed,
  in line with arXiv's API terms of use; each user downloads it into their own `.cache/`.
- WithdrarXiv: Apache-2.0.
- Retraction Watch: reuse terms not yet confirmed in writing (see the root README). Files derived from
  it (`pool/rw_*`) are kept out of version control by `pool/.gitignore`; only counts are published.
- OpenAlex metadata: CC0.

## Maintenance

- `uv run python -m review_harness.watch` adds new withdrawals to the pool, resuming a week before the
  last run. Records are never rewritten; each keeps the hash of the rules that classified it.
- A frozen manifest is never edited. A change to the rules, prompts or exclusions means freezing a new
  manifest with a new name, and logging it under Deviations in `PREREGISTRATION.md`.
- `uv run python -m review_harness.dataset_stats --version-dates` recomputes the checks.

## Known limitations

See [DATASET_NOTES.md](DATASET_NOTES.md) for the numbers. In short: most "unseen" papers were posted
before the cutoff, so only the withdrawal is unseen, not necessarily the text; the samples are small;
controls are matched on category listing and month but not length; some withdrawals retract one
claim in a replacement version rather than the whole paper; and the answer key is the authors' own
account, which can be incomplete or wrong.
