"""Train the stage-1 fold models and the stacker on features built at full test-like density.

Features for the whole training set (every Reference Record and Mention, like test) come from
`pipeline.feature_parts`. Training uses a hash sample of fit-state Mentions; the geographic
validation states (data/split/val) are scored inside the same full-density run, so the local score
sees test-scale look-alike neighbourhoods. Also scores the dense validation with an older model dir
(--baseline) to measure how much the density gap cost it.

Usage (from code/business_entity_resolution, PYTHONPATH=src):
  python src/scripts/train_dense.py --data ../../student_resource/dataset/train --work ../../runs/v8/work \
      --emb ../../data/emb/emb_full.parquet --from-train ../../runs/v7/work/train --val-refs ../../data/split/val/source1.tsv \
      [--baseline ../../runs/v7/work/train] [--sample 0.15]
"""
import argparse
import json
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from entity_resolution.evaluate import evaluate, read_matches
from entity_resolution.pipeline import GRID, READ, _find_truth, feature_parts, fit_lgb, label, log
from entity_resolution.resolve import resolve
from entity_resolution.stacker import STACK_COLS, _load, _parts, _x, aligned_context


def weigh(df, matched, distractor_weight):
    """Sample weight: negatives whose Mention matches no Reference Record count `distractor_weight` times."""
    if distractor_weight == 1.0:
        return df
    return df.with_columns(pl.when((pl.col("y") == 0) & ~pl.col("mention_id").is_in(matched.implode()))
                           .then(distractor_weight).otherwise(1.0).alias("w"))


def score_all(train_work, parts, keep):
    """Stage-2 scores for every row of `parts` that passes `keep` (a polars filter over ref_id/mention_id)."""
    s1_models = [lgb.Booster(model_file=str(Path(train_work) / f"{n}.txt")) for n in ("s1_a", "s1_b")]
    stack = lgb.Booster(model_file=str(Path(train_work) / "stack.txt"))
    base = s1_models[0].feature_name()
    # `keep` selects whole Mentions, and every row of the kept Reference Records, so the stage-2 context
    # computed on the kept rows alone is exact for them
    kept = [_load(pair).filter(keep) for pair in parts]
    narrow = []
    for df in kept:
        s1 = np.mean([m.predict(_x(df, base), num_threads=0) for m in s1_models], axis=0).astype(np.float32)
        narrow.append(df.select("ref_id", "mention_id").with_columns(pl.Series("s1", s1)))
    ctx = aligned_context(narrow)
    del narrow
    out = []
    for i in range(len(parts)):
        df = kept[i].hstack(ctx[i])
        kept[i] = ctx[i] = None
        out.append(df.select("ref_id", "mention_id").with_columns(
            pl.Series("score", stack.predict(_x(df, stack.feature_name()), num_threads=0).astype(np.float32))))
    return pl.concat(out)


def tune(scored, ref_ids, truth, name):
    res = [(evaluate(resolve(scored, ref_ids, t), truth), t) for t in GRID]
    f, t = max(res)
    log(f"{name}: dense VAL F0.5={f:.5f} at threshold {t:.3f}; scan", [(round(t, 3), round(s, 4)) for s, t in res])
    return f, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--from-train", required=True, help="model dir providing pruner.txt and token_te.parquet")
    ap.add_argument("--val-refs", required=True)
    ap.add_argument("--baseline", help="older model dir to score on the dense validation")
    ap.add_argument("--sample", type=float, default=0.15)
    ap.add_argument("--sample-seed", type=int, default=23, help="hash seed of the training Mention sample (bagging)")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--distractor-weight", type=float, default=1.0,
                    help="weight of negative rows whose Mention is a Distractor (test/train Distractor prevalence ratio)")
    a = ap.parse_args()
    work, src = Path(a.work), Path(a.from_train)
    (work / "train").mkdir(parents=True, exist_ok=True)
    for f in ("pruner.txt", "token_te.parquet"):
        if not (work / "train" / f).exists():
            shutil.copy(src / f, work / "train" / f)
    pruner = lgb.Booster(model_file=str(work / "train" / "pruner.txt"))
    te = pl.read_parquet(work / "train" / "token_te.parquet")
    feature_parts(a.data, work / "full", pruner, 6, 1.0, 400_000, emb=pl.scan_parquet(a.emb), te=te)
    parts = _parts(work / "full")

    truth = read_matches(_find_truth(a.data))
    val_refs = pl.read_csv(a.val_refs, **READ)["entity_id"]
    val_set = set(val_refs.to_list())
    val_truth = {r: truth.get(r, []) for r in val_refs.to_list()}
    val_mentions = pl.Series("m", [m for ms in val_truth.values() for m in ms])
    matched = pl.Series("m", [m for ms in truth.values() for m in ms])
    # every row of a Mention that is a Candidate of some validation Reference Record: needed for its argmax
    val_cand_mentions = pl.concat([pl.read_parquet(p, columns=["ref_id", "mention_id"]).filter(pl.col("ref_id").is_in(val_refs.implode()))["mention_id"]
                                   for p, _ in parts]).unique()
    in_val = pl.col("mention_id").is_in(val_cand_mentions.implode())

    if a.baseline:
        scored = score_all(a.baseline, parts, in_val)
        tune(scored, val_refs.to_list(), val_truth, "baseline " + a.baseline)
        del scored

    tw = work / "train"
    if not (tw / "stack.txt").exists():
        train_row = (~pl.col("ref_id").is_in(val_refs.implode()) & ~pl.col("mention_id").is_in(val_mentions.implode())
                     & ~in_val & (pl.col("mention_id").hash(seed=a.sample_seed) % 1000 < int(a.sample * 1000)))
        df = weigh(label(pl.concat([_load(p).filter(train_row) for p in parts]), truth), matched, a.distractor_weight)
        cols = [c for c in df.columns if c not in ("ref_id", "mention_id", "y", "w", "p0")]
        log(f"dense training rows={df.height} pos={int(df['y'].sum())}")
        fold = (df["ref_id"].hash(seed=17) % 2).to_numpy()
        models = [fit_lgb(df.filter(pl.Series(fold == f)), cols, a.rounds, name, tw) for f, name in ((0, "s1_a"), (1, "s1_b"))]
        del df
        # out-of-fold stage-1 score for EVERY row (model of the other Reference fold), so the stage-2
        # context over each Mention and each Reference Record is complete, as it is on test
        narrow = []
        for pair in parts:
            d = _load(pair)
            f = (d["ref_id"].hash(seed=17) % 2).to_numpy()
            X = _x(d, cols)
            s1 = np.where(f == 0, models[1].predict(X, num_threads=0), models[0].predict(X, num_threads=0)).astype(np.float32)
            narrow.append(d.select("ref_id", "mention_id").with_columns(pl.Series("s1", s1)))
            del d, X
        ctx = aligned_context(narrow)
        del narrow
        rows = []
        for i, p in enumerate(parts):
            rows.append(_load(p).hstack(ctx[i]).filter(train_row))
            ctx[i] = None
        df = weigh(label(pl.concat(rows), truth), matched, a.distractor_weight)
        del rows, ctx
        fit_lgb(df, cols + STACK_COLS, a.rounds, "stack", tw)
        del df
    scored = score_all(tw, parts, in_val)
    scored.write_parquet(tw / "dense_val_scored.parquet")
    f, t = tune(scored, val_refs.to_list(), val_truth, "dense model")
    (tw / "dense_val.json").write_text(json.dumps({"val_f05": f, "threshold": float(t)}))


if __name__ == "__main__":
    main()
