# Business Entity Resolution (Amazon ML Challenge 2026)

Matches each Source 1 business (Reference Record) to its noisy copies in Sources 2 and 3 (Mentions).
Final public leaderboard score: **F0.5 = 0.983** (dense validation 0.98990).

Pipeline:

1. normalization;
2. multi-source Blocking (state-partitioned char/word TF-IDF, exact name keys, multilingual-e5 kNN);
3. learned pruning to 6 Candidates per Mention;
4. about 135 pair features (text, number, Ghost house-number and legal-form, name-uniqueness, Candidate
   context);
5. LightGBM stage-1 fold models plus a stage-2 stacker, trained at full density; two stacker runs averaged;
6. a cross-encoder gate (fine-tuned multilingual-e5 on hard pairs) that re-scores each Mention's best pair;
7. unsupervised per-country calibration and a per-Reference expected-F0.5 decision.

The methodology is in `Documentation_template.md` at the root of the submission.

## Layout

```
src/ber/                   core pipeline (library)
  normalize.py             name/address normalization, Indic romanization (anyascii), state codes, name keys
  block.py                 Blocking: state-partitioned TF-IDF sources, exact name key, city->state inference
  embed_knn.py             embedding Blocking source (intfloat/multilingual-e5-small, MIT)
  features.py              pair features (rapidfuzz, numbers) and Candidate-set context
  sibling.py, ghost.py     sibling-word, legal-form and Ghost (house-number shift, cluster) features
  stack.py                 stage-2 stacker context
  resolve.py, evaluate.py  one Reference per Mention with threshold; macro F0.5
  pipeline.py              Blocking/feature chunking with parquet checkpoints; sample-based training entry point
src/final/                 final stage
  orphan.py                test-like training: drop 19% of Reference Records
  ce_export.py             easy text pairs for cross-encoder training, plus validation/test uncertain pairs
  hard_export.py           hard text pairs: training-state Candidates with stacker score 0.02-0.98
  bag.py                   average the stacker scores of several dense runs
  ce_gate.py               cross-encoder gate, fitted out-of-fold on dense validation
  calibrate_test_ef.py     per-country calibration + per-Reference expected-F0.5 decision; writes output
  efdecide.py              the expected-F0.5 prefix decision
src/kaggle/                GPU jobs (Kaggle 2x T4)
  embed/                   embedding kNN for the full training set and test
  ce/                      cross-encoder kernel template (e5-small, easy pairs)
  ce_hard_base/            final cross-encoder 1: e5-base on easy + hard pairs
  ce_hard_small2/          final cross-encoder 2: e5-small, 2 epochs on easy + hard pairs
scripts/
  make_split.py            geographic validation split (whole states held out)
  train_dense.py           full-density features, stage-1 + stacker training, dense validation
  score_test.py            test Blocking/features/scores and a thresholded submission
  run_final.sh             steps 8-9 below in one command
  validate_submission.py   the official validator
tests/                     pytest suite
```

## Environment

- Python 3.13; install with `pip install -r requirements.txt`.
- CPU steps: 20 threads and 16 GB RAM.
- GPU steps: Kaggle notebooks with 2x T4 (`src/kaggle/*`, pushed with `kaggle kernels push -p <dir>`).
- Models used:
  - intfloat/multilingual-e5-small (MIT, 118M parameters);
  - intfloat/multilingual-e5-base (MIT, 278M parameters);
  - LightGBM (MIT).
- No external data.

## Reproduce end-to-end

Run from the repository root with `PYTHONPATH=code/business_entity_resolution/src`. The layout is:

- `student_resource/dataset/{train,test}`: the challenge data;
- `data/`: intermediate data;
- `runs/`: work dirs.

The `src/final` scripts find the root from their own location; override it with `BER_ROOT`. Every long
step writes parquet checkpoints, so rerunning the same command resumes.

```bash
S=code/business_entity_resolution/scripts; F=code/business_entity_resolution/src/final
DATA=student_resource/dataset

# 1. Geographic validation split: whole states held out (data/split/{fit,val})
python $S/make_split.py --train-dir $DATA/train --out data/split --frac 0.10 --seed 7
#    copy data/split/fit/* to data/fit_as_train as train_source{1,2,3}.tsv + train_ground_truth.tsv, then
#    draw the 15% sample-based training set used in step 2 (data/split_trn/val, copied to data/trn):
python $S/make_split.py --train-dir data/fit_as_train --out data/split_trn --frac 0.15 --seed 11

# 2. Pruner and token statistics from a 15% sample-based training run (writes <work>/pruner.txt,
#    <work>/token_te.parquet); see ber.pipeline --help
python -m ber.pipeline --train data/trn --val data/split/val --test $DATA/test --emb-dir data/emb --stack \
    --out runs/sample/output --work runs/sample/work

# 3. Embedding Blocking source on GPU (src/kaggle/embed): emb_full.parquet (whole training set)
#    and emb_test.parquet, downloaded to data/emb/

# 4. Full-density run, which builds Blocking chunks and features for the whole training set
python $S/train_dense.py --data $DATA/train --work runs/v9/work --emb data/emb/emb_full.parquet \
    --from-train runs/sample/work --val-refs data/split/val/source1.tsv --sample 0.15

# 5. Test-like density: drop 19% of Reference Records, then retrain at full density
python $F/orphan.py runs/v9/work runs/v12/work data/emb/emb_full.parquet data/emb/emb_full_orph.parquet \
    data/split/val/source1.tsv
python $S/train_dense.py --data $DATA/train --work runs/v12/work --emb data/emb/emb_full_orph.parquet \
    --from-train runs/v9/work/train --val-refs runs/v12/work/val_refs_orph.tsv --sample 0.15
python $S/score_test.py --test $DATA/test --work runs/v12/work/test --train-work runs/v12/work/train \
    --emb data/emb/emb_test.parquet --threshold 0.675 --out runs/v12/output

# 6. Second stacker (bagging): another 15% Mention sample, same features (link runs/v13b/work/full to
#    runs/v12/work/full, and copy runs/v12/work/test without scored.parquet into runs/v13b/work/test)
python $S/train_dense.py --data $DATA/train --work runs/v13b/work --emb data/emb/emb_full_orph.parquet \
    --from-train runs/v12/work/train --val-refs runs/v12/work/val_refs_orph.tsv --sample 0.15 --sample-seed 101
python $S/score_test.py --test $DATA/test --work runs/v13b/work/test --train-work runs/v13b/work/train \
    --emb data/emb/emb_test.parquet --threshold 0.675 --out runs/v13b/output

# 7. Cross-encoder data and training (GPU)
python $F/ce_export.py                         # data/ce/{train,val,test}.parquet   -> Kaggle dataset amlc26-ce-data
python $F/hard_export.py data/ce/hard.parquet 0,1,2,3,4   # 406k hard pairs    -> Kaggle dataset amlc26-ce-hard
#    push src/kaggle/ce_hard_base and src/kaggle/ce_hard_small2; download their ce_val.parquet and
#    ce_test.parquet to data/ce_hard_base/ and data/ce_hard_small2/

# 8-9. Bag, gate, calibrate, decide, validate -> output/
bash $S/run_final.sh output
```

`output/` receives:

- `matching_results.tsv`: the final matches;
- `candidate_pairs.tsv`: the pruned Candidate set the models scored.

## Tests

```bash
cd code/business_entity_resolution && python -m pytest -q tests
```
