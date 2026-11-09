# Dataset notes, 9 November 2026

These are read-only checks of the two frozen samples, the rolling withdrawal pool and the Retraction
Watch candidates. No manifest was edited. To reproduce, run these from `harness/`:

```bash
uv run python -m review_harness.dataset_stats --version-dates
```

```bash
uv run python -m review_harness.taxonomy
```

The figures come from `results/dataset/manifest_stats.json`, `manifest_papers.csv` and the taxonomy
tables. They were checked by Claude, not by a domain expert.

## The frozen samples at a glance

| | recent-2026h2 | withdrarxiv-2023q4-2024q3 |
|---|---|---|
| sha256 | `ed9630cc…8161917` | `e6a0193d…b6960c` |
| Withdrawn papers (text / external) | 53 (38 / 15) | 24 (23 / 1) |
| Controls | 50 | 22 |
| Withdrawn papers without a control | 3 | 2 |
| Target in the preregistration (text / external) | 60 / 15 | 60 / 15 |
| Reviewed version posted after the window start, all / text | 5 / 3 | 12 / 11 |
| Version 1 posted after the window start, all / text | 3 / 2 | 11 / 10 |
| PDF-only texts (withdrawn and controls) | 2 | 3 |
| Hand exclusions | 19 | 15 |
| Gold statements with no numbered location | 29 | 9 |
| Texts flagged for revealing the error | 0 | 0 |

The window start is 2026-06-30 for the recent sample, the latest knowledge cutoff in `models.toml`.
For WithdrarXiv it is 2023-10-01.

**Agreement with the pool.** The pool's run over the same window found the same 529 withdrawals, and
76 kept by the same rules (filters sha `eddceb09…`). Taking away the 19 hand exclusions leaves 57, which
is the manifest's `usable_candidates`. Four of those were then dropped (1 text too long, 1 text that
reveals the error, 2 unavailable), leaving 53. The harvester and the sampler agree.

## What to know before reading results

Most important first.

### 1. In the recent sample, almost no reviewed text postdates the cutoff

Only 5 of the 53 reviewed versions were posted after 2026-06-30, and only 3 of the 38 `text` papers:
`2607.19839v1`, `2606.27571v2` and `2607.18211v1`. Version 1 of 47 of the 53 papers was posted in
2024–2026, before the cutoff, and 3 date from 2013–2021. 18 papers are reviewed at a version newer
than version 1, which is why the version dates were looked up (`reviewed_version_dates.csv`; the
manifest stores only version 1's date).

So the withdrawal is unseen, but a model may well have read the paper. The title-only arm measures
whether a model remembers the withdrawal, not whether it remembers the text. The preregistration's
separate figure for papers posted after the cutoff rests on 3 papers, which supports a description
but no test. The rolling pool (`review_harness.watch`) is how this subset can grow: papers both
posted and withdrawn after the cutoff, recorded as they appear. The pool currently has 10 such
papers among 76 usable withdrawals, before hand review.

### 2. The samples are smaller than planned, and the window is exhausted

The recent sample has 38 `text` papers against a target of 60. Every usable candidate in the window
was used (57 usable, 53 selected), so redrawing the same window cannot fix this. WithdrarXiv has 23
`text` papers after its 15 hand exclusions.

What the recent sample can detect:

- **Test 4 (fatal findings, withdrawn vs controls).** Fisher's test on 38 `text` papers against 50
  controls has about 80% power for a 30-point difference (60% vs 30%), and about 46% for a 20-point
  difference (60% vs 40%). The normal approximation is used here.
- **Tests 1–3 (paired McNemar).** Significance needs a lopsided split of the discordant papers: 10:2
  of 12, 12:3 of 15, or 15:5 of 20, at p < 0.05.

A failed test 3 (harness vs single prompt) is therefore weak evidence that the pipeline adds nothing.
The preregistration already treats it as a finding rather than a reason to rerun.

### 3. Eleven recent "withdrawals" retract a claim, not the paper

In 11 of the 53 recent papers, the comment withdraws a claim inside a replacement version that stays
up corrected, rather than the paper itself: `2512.13406v1`, `2602.22673v2`, `2607.00728v1`,
`2605.06021v1`, `2601.01855v7`, `2605.01953v1`, `2605.01192v1`, `2502.20569v1`, `2605.23419v3`,
`2601.11799v2`, `2605.00309v1` (`withdrawal_kind` in `manifest_papers.csv`). Each still states a
specific error in the reviewed version, so each is a valid answer key.

This matters for the framing arms. Telling a model "this paper was retracted" is literally untrue for
these 11, so primed-arm results should also be reported without them. WithdrarXiv has none.

### 4. Controls match on listing and month, not on length

- 3 recent and 2 WithdrarXiv withdrawn papers got no control (`control_not_found`), so the comparisons
  are 53 vs 50 and 24 vs 22, unpaired.
- Every control shares its paper's month. Controls are found by an arXiv `cat:` search, which also
  returns cross-listed papers, so only 34 of 50 recent controls share the primary category. 45 of 50
  share the broad field.
- Lengths are not matched. In the recent sample the control-to-withdrawn length ratio has a median
  of 1.13, a quartile range of 0.58–2.05 and a full range of 0.18–6.71. Controls are longer on
  median: 76k characters against 59k. A longer text gives more chances for a fatal finding, so test
  4 should also be run with log length as a covariate, as a robustness check. That is a secondary
  analysis, since the preregistration does not name one.

### 5. Gold statements

- **Specificity.** 29 of the 53 recent statements and 9 of the 24 WithdrarXiv statements have no
  numbered location; they passed on specific wording. After the hand exclusions, every remaining
  one names a mechanism, such as "the weight v defined in (5.19) was incorrectly treated as
  self-adjoint" or "a confound in stimulus assignment". The vaguer ones that remain are `2606.09191v1`
  ("a counterexample… invalidates a key step in the risk-averse regret bounds") and `2601.07131v1`
  ("an implementation error that resulted in look-ahead bias").
- **One near-duplicate pair.** `2409.01981v2` and `2410.13840v2` (0.98 similar) report the same
  Composition Lemma error, whose argument came from arXiv:2202.03178. Their outcomes are not
  independent, so treat them as one cluster in any interval. They were kept, while `2607.15979` and
  `2607.24548` were excluded as "flaw is in a lemma of another paper". The line drawn is that the
  first two prove the flawed lemma in their own text, while the latter two cite it. That is
  defensible, but `exclusions.toml` does not say it.
- **People named.** One recent statement names its author, and three WithdrarXiv statements carry
  `[REDACTED-NAME]` from that dataset's scrubbing. Two recent `external` statements link to code
  repositories under personal usernames. Judges see these; reviewers never do.

### 6. What the errors are

These are rule-based primary kinds (`taxonomy.csv`). The independent check is in the next section.

| | proof | calculation | data | code | citation | other |
|---|---|---|---|---|---|---|
| recent, `text` (38) | 25 | 3 | 3 | 0 | 3 | 4 |
| recent, `external` (15) | 2 | 0 | 4 | 9 | 0 | 0 |
| withdrarxiv (24) | 16 | 4 | 0 | 0 | 1 | 3 |
| pool, all kept (76) | 37 | 4 | 10 | 11 | 6 | 8 |

Two-thirds of the primary `text` sample are proof errors, mostly in mathematics. A result on this
sample is mostly a result about finding gaps in proofs, and should be described that way. The
`external` papers are almost all code and data errors, which supports keeping them separate.

## How far the error kinds can be trusted

A second labelling was made independently by `codex/gpt-6.1-terra`, run through the Codex CLI on the
user's ChatGPT sign-in, with no tools and no API cost (`scripts/llm_label_taxonomy.py`). It covered
the 100 distinct statements in the two manifests and the pool, with the same six definitions and no
sight of the rules or their labels. The per-statement labels stay in `runs/`, and the counts are in
`results/dataset/taxonomy_agreement.*`.

| | Agreement | Cohen's κ |
|---|---|---|
| Six kinds | 74% | 0.62 |
| Three kinds: formal (proof, calculation), empirical (data, code), other | 78% | 0.61 |

The disagreements cluster at two boundaries:

- **Proof vs "too general to place"** (12 statements). For "a mistake in Theorem 3.1" or "an error in
  the derivation of the Langevin equation", the rules say proof because a theorem or derivation is
  named, while the model says other because no mechanism is given.
- **Code vs data** (4 statements), for evaluation and pipeline errors such as a leakage audit or
  scores read from the wrong axis.

Only 5 of the 100 statements cross between formal and empirical. So the kinds can be used to say
whether an error lies in reasoning or in empirical work. The finer kinds should be described as
approximate, and so should "other".

The model also put the two inherited Composition Lemma errors under proof rather than citation. That
is defensible, since the flawed proof is in each paper's own text.

## Journal retractions with an arXiv version

These come from Retraction Watch as synced locally (version `bcb455b6…`, generated 2026-11-06, with
retractions up to 2026-10-20). Run with `python -m review_harness.rw_arxiv`. The summary is in
`results/dataset/rw_arxiv_summary.json`, and the per-record list, `pool/rw_arxiv_candidates.jsonl`,
stays local.

| | Count |
|---|---|
| Retractions citing Error in Analyses, Data, Methods or Results and/or Conclusions | 3,878 |
| … with a DOI | 3,645 |
| … retracted after 2026-06-30 | 22 |
| Matched to an arXiv version | **49** |
| … retracted after the cutoff (unseen) | **1** |
| … retracted before it (seen) | **48** |
| … with a misconduct-related reason recorded as well | 0 |
| … preprint posted before the retraction | 49 |
| … also withdrawn on arXiv | 10 |
| … title match below 0.8, needing a person to check | 3 |

- **Matches by method.** 22 by arXiv's own DOI field, 26 by OpenAlex, and 1 whose Retraction Watch
  DOI is an arXiv DOI. 21 of the 22 arXiv-DOI matches are also in OpenAlex, so OpenAlex alone finds
  47 of the 49. 2 OpenAlex groupings were rejected because the titles differed. The 21 unseen
  retractions with no DOI match also got an arXiv title search, which found nothing.
- **By field.** Physics 29, mathematics 14, computer science 3, and one each in eess, q-bio and stat.
  Most come from APS, Springer Nature and AAAS journals, retracted from 2011 to 2026.
- **The 3 weak title matches** have similarities of 0.61–0.75. In one, the arXiv title was replaced
  by a retraction notice; the other two are renamed or shortened preprint titles. On reading the
  titles all three look like the same work, but nobody has checked further.
- **The one unseen match** has a preprint posted before 2026, so as in the arXiv sample only the
  retraction is unseen, not the text.
- Record-level matches (identifiers, titles, dates) stay in the git-ignored
  `pool/rw_arxiv_candidates.jsonl` until Retraction Watch's reuse terms are confirmed.

**What this means.** Retraction Watch adds almost nothing to the unseen sample: its error retractions
are mostly biomedical, few have arXiv versions, and the dataset lags by weeks. The 48 seen
retractions are useful for something else. They are journal-confirmed errors for testing memory: a
title-only probe on papers every model may know, or the "this paper was retracted, find the critical
flaw" framing on papers that really were retracted. Their reasons are coded categories rather than
the authors' words, so they need a stated error from the retraction notice before they can be an
answer key. A future manifest would add one by hand.

## Recommendations

None of these require changing a frozen manifest.

1. Report the post-cutoff subset (3 papers) as descriptive only, and run `watch.py` regularly so a
   later sample can be drawn from papers both posted and withdrawn after a cutoff.
2. Report the framing arms with and without the 11 claim-in-replacement papers.
3. Add log length as a covariate in a secondary analysis of test 4, and say it was added after the
   freeze.
4. Treat `2409.01981v2` and `2410.13840v2` as one cluster.
5. Add a line to `exclusions.toml` explaining why an inherited lemma proved in the paper's own text is
   kept. That can only affect a future manifest.
6. Describe results by the coarse error kinds (formal vs empirical), which two labellers agree on,
   rather than the six fine ones.
7. Use the 48 seen Retraction Watch matches for memory and framing probes, not as a scored sample,
   until each has a stated error taken from its retraction notice.
