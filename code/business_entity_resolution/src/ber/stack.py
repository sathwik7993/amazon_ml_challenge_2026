"""Stage-2 stacker: re-score Candidates using the stage-1 scores of their neighbours.

Stage 1 sees one pair at a time. Stage 2 also sees how the pair's stage-1 score compares with the
Mention's other Candidates and with the other Mentions of the same Reference Record. Stage-1 scores
for training come out-of-fold (two halves split by Reference Record) so stage 2 never trains on
scores its own rows produced.
"""
import glob
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

STACK_COLS = ["s1", "s1_rank_m", "s1_gap_m", "s1_second_m", "s1_sum_m", "s1_rank_r", "s1_gap_r", "s1_strong_r", "s1_sum_r"]


def stack_context(df):
    """Add stage-1 score context columns to (ref_id, mention_id, s1)."""
    return df.with_columns(
        pl.col("s1").rank("ordinal", descending=True).over("mention_id").cast(pl.Int16).alias("s1_rank_m"),
        (pl.col("s1") - pl.col("s1").max().over("mention_id")).alias("s1_gap_m"),
        pl.col("s1").sum().over("mention_id").alias("s1_sum_m"),
        pl.col("s1").rank("ordinal", descending=True).over("ref_id").cast(pl.Int16).alias("s1_rank_r"),
        (pl.col("s1") - pl.col("s1").max().over("ref_id")).alias("s1_gap_r"),
        (pl.col("s1") > 0.5).sum().over("ref_id").cast(pl.Int16).alias("s1_strong_r"),
        pl.col("s1").sum().over("ref_id").alias("s1_sum_r"),
    ).with_columns(
        pl.when(pl.col("s1_rank_m") == 2).then(pl.col("s1")).max().over("mention_id").fill_null(0.0).alias("s1_second_m"),
    )


def aligned_context(frames):
    """Stage-2 context for per-part (ref_id, mention_id, s1) frames, returned per part in the same row order.

    Groups on 64-bit hashes of the IDs instead of the strings, so tens of millions of rows fit in memory.
    """
    sizes = [f.height for f in frames]
    narrow = pl.concat([f.select(pl.col("ref_id").hash(seed=1), pl.col("mention_id").hash(seed=2), "s1") for f in frames])
    ctx = stack_context(narrow).select(STACK_COLS)
    del narrow
    out, offset = [], 0
    for n in sizes:
        out.append(ctx.slice(offset, n))
        offset += n
    return out


def _parts(work):
    """(feature part, Reference-side context part) path pairs written by pipeline.feature_parts."""
    paths = sorted(glob.glob(str(Path(work) / "feats_*.parquet")))
    sides = ("refctx_", "sib_", "lg_", "hn_", "hc_", "nk_", "te_")
    return [(p, [p.replace("feats_", s) for s in sides if Path(p.replace("feats_", s)).exists()]) for p in paths]


def _load(pair):
    feats, sides = pair
    df = pl.read_parquet(feats)
    for side in sides:
        df = df.join(pl.read_parquet(side), on=["ref_id", "mention_id"], how="left")
    return df


def _x(df, cols):
    return df.select(cols).to_numpy().astype(np.float32)


def train_stack(train_work, label_fn, fit_fn, rounds, log=print):
    """Fit two fold models (out-of-fold stage-1 scores) and the stage-2 model; saved in train_work."""
    train_work = Path(train_work)
    df = label_fn(pl.concat([_load(pair) for pair in _parts(train_work)]))
    cols = [c for c in df.columns if c not in ("ref_id", "mention_id", "y", "p0")]
    fold = (df["ref_id"].hash(seed=17) % 2).to_numpy()
    s1 = np.zeros(df.height, dtype=np.float32)
    for f, name in ((0, "s1_a"), (1, "s1_b")):
        model = fit_fn(df.filter(pl.Series(fold == f)), cols, rounds, name, train_work)
        s1[fold != f] = model.predict(_x(df.filter(pl.Series(fold != f)), cols), num_threads=0)
    df = stack_context(df.with_columns(pl.Series("s1", s1)))
    return fit_fn(df, cols + STACK_COLS, rounds, "stack", train_work)


def score_stack(train_work, data_work, log=print):
    """Stage-2 scores (ref_id, mention_id, score) for a data work dir whose feature parts already exist."""
    train_work = Path(train_work)
    s1_models = [lgb.Booster(model_file=str(train_work / f"{n}.txt")) for n in ("s1_a", "s1_b")]
    stack = lgb.Booster(model_file=str(train_work / "stack.txt"))
    base_cols = s1_models[0].feature_name()
    pairs = _parts(data_work)
    narrow = []
    for pair in pairs:
        df = _load(pair)
        X = _x(df, base_cols)
        s1 = np.mean([m.predict(X, num_threads=0) for m in s1_models], axis=0).astype(np.float32)
        narrow.append(df.select("ref_id", "mention_id").with_columns(pl.Series("s1", s1)))
        del df, X
    ctx = aligned_context(narrow)
    del narrow
    out = []
    for i, pair in enumerate(pairs):
        df = _load(pair).hstack(ctx[i])
        ctx[i] = None
        s = stack.predict(_x(df, stack.feature_name()), num_threads=0)
        out.append(df.select("ref_id", "mention_id").with_columns(pl.Series("score", s.astype(np.float32))))
    log(f"stacked scores for {sum(o.height for o in out)} Candidates")
    return pl.concat(out)
