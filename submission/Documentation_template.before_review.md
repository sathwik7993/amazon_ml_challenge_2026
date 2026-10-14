# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Slytherin
**Team Members:** Sathwik Chappala (CMR College of Engineering & Technology), Bojja Mahendra Sai (CMR College of Engineering & Technology)
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We solve entity resolution as multi-source Blocking, then a learned pair model with a stage-2 stacker, then a
one-to-one assignment with metric-aware decisions. Most of the gains came from measuring how the test data
differs from the training data, and then training and validating under test-like conditions:

- **Density:** we build all features on the *full* training set (2.2M Reference Records, 10.3M Mentions), so
  that the number of look-alike neighbours matches test.
- **Ghost Distractors:** look-alike businesses next door, detected by house-number, legal-form and
  name-uniqueness features.
- **France:** it gets the same state-partitioned Blocking as the other countries.
- **Unlabelled calibration:** per-country decision thresholds are set from the test score distribution,
  without labels.

Public leaderboard progression: 0.943 → 0.948 → 0.950 → 0.965 → 0.973 → 0.975 → 0.976 → 0.982 → 0.983.

---

## 2. Methodology

### 2.1 Problem Analysis (what the data generator does)

EDA on the 2.2M / 5.0M / 5.3M training records, and on the unlabelled test files:

- **Structure:** every matched Mention (Source 2/3 record) belongs to exactly one Reference Record (Source 1).
  About 26% of training Mentions match nothing (Distractors). 5.6% of Reference Records are Singletons.
  Each Reference Record has at most 5 Source-2 and 6 Source-3 Matches.
- **Name noise:** token shuffles, typos, legal-suffix rewrites (Pvt/Private, Ltd/Limited, Inc, SARL/SAS/EURL),
  junk prefixes, website/handle names, "formerly"/"doing business as" aliases, appended filler words, and
  Indic scripts (~23% of India names).
- **House numbers are corrupted in true copies** by digit drop or substitution (8395→839, 547→147) or
  leading zeros. They almost never shift upward.
- **Ghosts:** the hard Distractors copy a Reference Record's core name, change the legal form
  (Inc→Corp, SARL→SNC) or add one word from a fixed vocabulary (Group, Holdings, Groupe, International, ...),
  and move the house number **up** by 1–60 (520→529). One part of an Indian compound number changes
  (65/2688→65/2701). A Ghost often has 2–3 Mentions of its own that share the shifted number.
- **Test differs from training in measurable ways:** 5.76 vs 4.67 Mentions per Reference Record, with the extra
  ones being Distractors; ~18 vs ~3 Reference Records sharing a name per country; and France, which appears only
  in test and whose addresses carry regions and départements.

### 2.2 Solution Strategy

**Approach type:** Blocking, then gradient-boosted pair classifier, then out-of-fold stacker, then one-to-one
assignment with calibrated decisions.

**Validation:** a geographic split (whole states held out, `docs/adr/0001`). The key lesson was that
validation must reproduce test *density*. Our final validation scores the held-out states **inside a
full-training-set feature run**. A small-sample validation over-estimated the leaderboard by 0.02–0.03.

---

## 3. Candidate Generation (Blocking)

- **Normalization** (`normalize.py`): anyascii romanization of every script; legal forms and fillers dropped for
  the core name; a consonant skeleton; address abbreviations for US, Indian and French forms; state codes from
  Latin or Indic script. For France, the 3 regions and their départements (e.g. "Nord" → Hauts-de-France) are
  mapped to region codes. When an address carries no state, the state is filled from its city, using a
  city→state map learned from the Reference Records (a city must be seen ≥20 times, ≥95% of them in one state).
- **Sources** (`block.py`, `embed_knn.py`). For each Mention, Candidates from the same country come from:

  | Source | View | Scope | k |
  |---|---|---|---|
  | full / fullw | char 3-gram / word TF-IDF of name + address | same state | 10 |
  | name / squash / skel | char 3-grams of name, space-less name, consonant skeleton | same state | 5 each |
  | addr | char 3-grams of address | same state | 5 |
  | fullx | char 3-gram TF-IDF of name + address | whole country, Mentions with no state | 10 |
  | emb | multilingual-e5-small (MIT) cosine | whole country (exact, GPU) | 10 |
  | keyx | exact order-free name key keeping legal forms | whole country, key shared by ≤30 Reference Records | all |
  | namex | exact core name, for Mentions with no usable address | whole country, name shared by ≤50 Reference Records | all |

- **Pruning:** a LightGBM pruner keeps each Mention's 6 best Candidates. Exact-key and exact-name Candidates are always kept.
  This set is `candidate_pairs.tsv`.

---

## 4. Matching Model

**Features (~135):**
- Name, address and number similarities (rapidfuzz ratios on normalized and skeleton forms, and number-string
  similarity robust to corruption).
- Every Blocking similarity.
- Candidate-set context (rank and gap within the Mention's Candidates and within the Reference Record's).
- **Sibling words:** per added or dropped name token, the rate at which it comes with a changed house number,
  computed on each dataset without labels, plus a smoothed supervised rate from training.
- **Legal form:** same, different, added or dropped.
- **Ghost features:** signed house-number difference; digit-edit relation (truncation, highest changed digit);
  numbers the Mention adds or drops across the whole address; and per-Reference-Record counts of Candidates
  sharing this Mention's number, legal form or added numbers.
- **Name uniqueness:** how many Reference Records in the country share the exact key or the core name.
- No feature encodes the country label.

**Test-like training data:** test has 5.76 Mentions per Reference Record against 4.67 in training, i.e. about
19% of test entities have no Source 1 record and their Mentions act as perfect look-alike Distractors. We
reproduce this by removing a fixed 19% hash sample of training Reference Records before building features
(`src/final_stage/simulate_missing_references.py`); training and validation both use this orphaned, full-density data.

**Model:** LightGBM (binary, 127 leaves, lr 0.05, early stopping), trained on a 15% hash sample of Mentions
whose features were computed at full density. Two fold models give out-of-fold stage-1 scores for every row. A
stage-2 LightGBM stacker adds the stage-1 score's rank, gap and sums within the Mention and within the Reference
Record. The context is computed on 64-bit ID hashes, so 66M rows fit in 16 GB.

**Resolution and decisions:** each Mention goes to its highest-scoring Reference Record.
- The global threshold is tuned for macro F0.5 on the dense validation.
- **Unsupervised per-country calibration:** test has more look-alike Distractors than training. Assume the
  per-Reference-Record distribution of true copies over score bins is the same on test as on validation (the
  >0.99 bin matches exactly, 3.28 vs 3.29 per Reference Record). Then the test true-match rate of a bin is the
  validation's expected true count divided by the test count. Each country's threshold maximizes the expected
  macro F0.5 under these calibrated probabilities. France, which has no validation data, uses the pooled
  US+India profile. No test labels are involved.
- **Per-entity expected-F0.5 decision:** for each Reference Record, emit the prefix
  of its ranked Candidates that maximizes expected F0.5, including the empty set.

**Cross-encoder gate (final file):** two multilingual-e5 cross-encoders (MIT; e5-base 1 epoch, e5-small 2 epochs;
Kaggle 2× T4) are fine-tuned on "name | address" text pairs from training states: 500k easy pairs plus 406k
hard pairs (training-state Candidates whose stacker score is 0.02–0.98, i.e. the cases the gate decides).
They score only uncertain pairs (stacker score 0.02–0.98). The stacker still picks each Mention's Reference
Record; a logistic regression on the stacker logit, the averaged cross-encoder logit, their interaction, the
gap to the Mention's second-best score and the cross-encoder margin over its other Candidates re-scores that
best pair. It is fitted out-of-fold (5 folds grouped by Reference Record) on the held-out validation states.
Letting the cross-encoder re-rank References lowered validation F0.5, so it only gates keep/drop.

**Bagging:** two stacker runs trained on different 15% Mention samples are averaged.

**Final pipeline order:** Blocking and pruning → features → stage-1 + stacker (two bagged runs, averaged) →
cross-encoder gate on each Mention's best pair → per-country calibration → per-Reference expected-F0.5
decision → `matching_results.tsv`.

---

## 5. Results & Error Analysis

| Version | Main change | Validation F0.5 | Public LB |
|---|---|---|---|
| v2s | TF-IDF + embedding Blocking, stacker | 0.9769 (sparse) | 0.943 |
| v4 | sibling-word features, France threshold | 0.9793 (sparse) | 0.950 |
| v6 | Ghost house-number and cluster features | 0.9887 (sparse) | 0.965 |
| v9 | dense training, France states, exact-key Blocking, name uniqueness | 0.9869 (dense) | 0.973 |
| v9cal | unsupervised per-country thresholds | same model | 0.975 |
| v12ef | training/validation with 19% of Reference Records removed (test-like orphans), per-entity decision | 0.9855 (dense, orphaned) | 0.976 |
| v16gate | bagged stackers + cross-encoder gate (3 models, easy pairs) | 0.9891 (dense, orphaned) | 0.982 |
| final | gate cross-encoders trained on hard in-band pairs | 0.9899 (dense, orphaned) | 0.983 |

Error breakdown on dense validation (v9), as the F0.5 gain if the error type were fixed:
- True copies never retrieved: +0.0042. 92% of them have **no address**, and their name is shared by many
  Reference Records, so they are ambiguous from the name alone.
- True copies scored below threshold: +0.0039.
- False merges: +0.0029.
- True copies assigned to another Reference Record: +0.0024.

**Validation vs leaderboard (stability):** since the test-like validation (v12), every validation gain carried
over to the public leaderboard in the same direction: +0.0035 validation gave +0.006 (v12ef to v16gate), and
+0.00085 gave +0.001 (v16gate to final). Every model and decision rule was fitted on training states or out-of-fold
on the held-out validation states, so we expect the private leaderboard to track the public one.

**Leaderboard use, disclosed:** besides normal submissions, two diagnostic submissions (the v6 and v12ef files with all
France rows emptied) were used to estimate France's share of the error. No parameter was fitted to leaderboard scores.

---

## 6. Conclusion

The largest gains came from treating the gap between local and leaderboard scores as a diagnosis problem, then
making training and validation look like test: full density, Ghost-aware features, France handled like the
other countries, and decisions calibrated to test's Distractor rate without labels. The final step was a
cross-encoder gate trained on the hard, uncertain pairs, the largest single gain (0.976 to 0.983). No external
data or lookup was used.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` (see its `README.md` for exact run commands):
- `src/entity_resolution/`: core pipeline package (normalization, Blocking, features, Ghost and name-variant
  features, stacker, resolution, metric);
- `src/final_stage/`: final stage (`simulate_missing_references.py`, `export_cross_encoder_pairs.py`,
  `export_hard_training_pairs.py`, `average_model_scores.py`, `cross_encoder_gate.py`, `calibrate_and_decide.py`,
  `expected_f05_decision.py`);
- `src/gpu_jobs/`: Kaggle GPU notebooks (`embedding_knn`, `cross_encoder_e5_base`, `cross_encoder_e5_small`);
- `src/scripts/`: `make_split.py`, `train_dense.py`, `score_test.py`, `run_final.sh`, the validator;
- `src/tests/`: pytest, 58 tests;
- `requirements.txt`: pinned versions.

### B. Models and compute
- Models: intfloat/multilingual-e5-small (MIT, 118M parameters), intfloat/multilingual-e5-base (MIT, 278M) and LightGBM (MIT).
- Embeddings: exact top-10 per country on Kaggle 2× T4.
- Everything else runs on one 20-thread / 16 GB machine, with parquet checkpoints at every stage.
