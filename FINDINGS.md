# Findings and Results: Business Entity Resolution (Amazon ML Challenge 2026)

Status as of 2026-09-25 19:30 IST. Terms (Reference Record, Mention, Match, Distractor, Singleton,
Candidate, Blocking) are defined in `CONTEXT.md`.

## 1. Leaderboard vs validation

| Version | What changed | Validation F0.5 | Public leaderboard |
|---|---|---|---|
| v1 | TF-IDF Blocking + LightGBM | 0.9641 | not submitted |
| v2 | + e5 embedding kNN (Kaggle), house-number string features, leading-zero fix | 0.9731 | not submitted |
| v2s | v2 + stage-2 stacker, threshold 0.675 | 0.9769 | **0.943** |
| v3 | v2s + France (no-state) threshold 0.955 | 0.9769 (unchanged) | **0.948** |
| v4 | + sibling-word features (token rates), thresholds 0.65 / 0.96 | 0.9793 | **0.950** |
| v5 | + legal-form change + numeric house-number distance | not finished (stopped by user) | none |
| v6 | v5 + Ghost features: signed house-number difference, digit-edit relation, per-Reference number clusters (`ghost.py`) | **0.9887** (threshold 0.675, no-state 0.85) | **0.965** |

**Label-free leaderboard proxy (2026-09-26).** `runs/expected_f.py` sums each Reference Record's expected F0.5
from the model's probabilities. The test estimate runs a steady ~0.016 above the leaderboard and tracks its changes:
v4 0.9659 (LB 0.950), v6 0.9810 (LB 0.965). v6 by country: France 0.966, India 0.981, US 0.990.
Leaderboard top 3 on 2026-09-26: 0.9884 / 0.9881 / 0.9881.

**India Ghosts (v7).** Indian house numbers are compound (65/2688, 2-2-18/30/A); a Ghost changes one part
(65/2701, 2-2-18/31/A) or adds a number the Reference lacks ("No 90 C/O ..."), shared by its copies. v6 compared
only the first number, so v7 compares all address numbers and counts Candidates sharing the same added numbers.

**Ghost finding (2026-09-26).** Hard Distractors ("Ghosts", see `CONTEXT.md`) copy a Reference Record's core name,
usually change the legal form, and move the house number UP by 1-60 (520->529, 4440->4443, 8395->8404, France 39->43).
A Ghost often has 2-3 Mentions sharing the shifted number. True noisy copies keep the number or drop/alter digits
(8395->839, 2253->253). On v4 validation, Candidates with a number 1-40 higher gave 1,274 false positives at the
threshold versus 38 for 1-40 lower. v4 only saw the absolute distance, so it could not tell the two apart.

Leaderboard reference (2026-09-25): #1 0.9859, #2 0.9846, #3 0.9844, #4 0.971, #10 0.958, #30 0.919.

**Central problem: validation overestimates test by ~0.03 and does not track leaderboard gains.** A validation gain
of +0.0024 (v4) gave only +0.002 on the leaderboard. Validation on the geographic split is not representative of test.

## 2. Data facts (EDA)

- Train: 2.21M Reference Records, 5.03M S2 + 5.29M S3 Mentions. Test: 1.73M Reference Records, 4.89M + 5.08M Mentions.
- Every matched Mention belongs to exactly one Reference Record (7.64M matched IDs, all unique).
- Train: 26% of Mentions are Distractors; 5.6% of Reference Records are Singletons; 2–6 Matches per Reference Record.
- **Test has 5.75 Mentions per Reference Record; train has 4.67.** The number of "obvious" Matches per Reference Record
  is the same (2.21 vs 2.20), so the extra Mentions are mostly Distractors.
- True pairs always share the country label. France appears only in test (15% of Reference Records).
- About 23.5% of India Mention names are in Indic scripts (Devanagari, Tamil, Telugu, …).

## 3. Data-generator findings (most important)

### 3.1 House numbers are corrupted in noisy copies
Examples: 33427→63427, 3144→314, 2800→002800, 238→247, plus extra prefixes such as "#7-67".
Fixes in place: leading zeros stripped, number strings compared by fuzzy similarity (`num_tset`, `num_partial`, `num_ratio`).

### 3.2 Sibling Distractors use a fixed extra-word vocabulary
The Mention name is the Reference name plus one word, at a nearby but different number. Measured on training
pairs whose Mention name contains all Reference tokens plus exactly one extra token:

| Extra word | True-Match rate | Number differs |
|---|---|---|
| group, holdings, public, south/north/east/west, valley, downtown, midtown, harbor, metro, pc, clinic, care, overseas, exports, infratech | 0.000 | ~99% |
| partners, enterprises, ventures, industries, associates, trading | 0.00–0.09 | 95–99% |
| services, service, center, trust, foundation, association, society, authority | 0.34–0.46 | 63–73% |
| dr, mr, shri, smt, sri, lnc (noise) | 0.64–0.75 | 45–49% |

On test, the unlabelled number-change rate picks up the French sibling words: france 0.93, groupe 0.93,
international 0.97, holding 0.97, distribution 0.97, developpement 0.93, participations 0.97.
v4 implements this in `src/ber/sibling.py`.

### 3.3 A changed legal form marks a different entity (not yet in a submission)
Among training pairs with an identical normalized name:

| Legal form | Number | True-Match rate | Pairs |
|---|---|---|---|
| different (Inc→Corp) | nearby (≤30) | **0.026** | 27,660 |
| different | far | 0.048 | 47,354 |
| added (Reference had none) | nearby | **0.042** | 22,084 |
| same | equal | 0.998 | 137,885 |
| same | nearby | 0.492 | 10,015 |
| Mention has none | equal | 0.999 | 128,844 |

`normalize_name` drops legal forms, so up to v4 the model could not see this. v5 adds `legal_state`,
`num_absdiff_log` and `num_reldiff` (`legal_features` in `src/ber/sibling.py`). It is coded and unit-checked, but not
yet trained or submitted.

### 3.4 Test has many more uncertain Mentions than validation
Share of Mentions whose best score is between 0.05 and 0.9 (v4):

| Country | Test | Validation (Distractor-heavy copy) |
|---|---|---|
| US | 13.1% | 7.5% |
| India | 11.1% | 9.6% |
| France | 22.8% | n/a |

Typical uncertain US test pairs share the same core name with a changed legal form and a house number a few units away
(207→212, 1059→1068, 3118→3116), which is the pattern in 3.3.

### 3.5 Experiments that ruled things out
- **Distractor density.** Validation with 19.5% of Reference Records removed (5.75 Mentions per Reference Record, like
  test) scored 0.9752; its best threshold (0.70) was essentially unchanged. Plain density does not explain the gap.
- **Missing state (France condition).** Validation with state detection disabled drops to 0.936 at threshold 0.675
  and recovers to 0.9615 at 0.955 (v4: 0.9659 at 0.96). The model is overconfident without state features.
- **More true copies per entity in test.** Rejected: obvious Matches per Reference Record are the same on test and
  validation.

## 4. Pipeline (code in `code/business_entity_resolution/`)

1. **Normalize** (`normalize.py`): anyascii romanization, legal/filler word drop, consonant skeleton, address
   abbreviations for US, India and French forms, state names (Latin and Indic script) mapped to codes, leading zeros
   stripped.
2. **Block** (`block.py`, `embed_knn.py`): char/word TF-IDF (`sparse_dot_topn`) within states, `fullx` country-wide
   for Mentions with no state, and multilingual-e5-small exact cosine kNN on Kaggle 2× T4 (recall@10 0.990). Union
   recall 99.3–99.7%.
3. **Prune** to 6 Candidates per Mention (LightGBM on Blocking features; recall 99.29%).
4. **Features** (`features.py`, `sibling.py`): rapidfuzz name/address/number similarity, Candidate-set context
   (ranks and gaps on both sides), sibling-token rates, legal-form features (v5).
5. **Model**: LightGBM, plus a stage-2 stacker on out-of-fold scores (`stack.py`).
6. **Resolve**: each Mention goes to its best Reference Record if the score is at least the threshold. Separate
   threshold for countries without a state lexicon.

Runtime on the local machine (20 threads, 16 GB RAM): test Blocking plus features ~1.5 h (cached now), stacker scoring
~40 min. Every stage writes parquet checkpoints and resumes after a crash.

## 5. Operational lessons

- Kaggle: run a smoke step and a full-scale GPU-memory smoke step first, and save embeddings before the search step.
  The CLI output download resets on large files; list the file URL with the kagglesdk API and use `curl -C -`.
- Local 16 GB RAM: cap the process pool at 6 workers; free the Blocking index before scoring; write per-chunk side
  tables (the full 60M-row join crashed polars).
- Background wait loops pile up. Kill stale `bash`/`sleep` processes after each run.

## 6. Next steps, ranked

1. **Make validation represent test.** Build a validation set with the test's hard-case mix, i.e. more sibling and
   legal-form Distractors. Without that, validation gains do not predict leaderboard gains.
2. **Finish v5** (legal-form features). Caches are in `runs/v5/work`; `runs/v5/run_all.sh` resumes the chain.
3. **Train on more data.** The training sample is only 250k Reference Records out of 2.2M, and training loss was still
   improving at 2000 rounds.
4. **Reference-level reasoning.** Among a Reference Record's Candidates, tell the true copy apart from its siblings,
   e.g. features comparing each Candidate with the Reference's best Candidate (same legal form, same number).
5. **France**: French Blocking uses only 2 of 7 sources, so give France a partition key (city).
