"""Export text pairs for the cross-encoder reranker (Kaggle).

train.parquet: labelled pairs from fit-state Mentions (never validation), each Mention's 3 best Candidates by
               embedding rank, from the orphaned full-density training features (test-like).
val.parquet / test.parquet: every Candidate row whose stacker score is in (0.02, 0.98), with that score.
Text is "name | address" for both sides, raw (any script), so the model sees what a human would.
"""
import glob
import sys

import polars as pl

import os  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = os.environ.get("BER_ROOT", str(Path(__file__).resolve().parents[4]))  # repo root: student_resource/, data/, runs/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ber.evaluate import read_matches  # noqa: E402

R = ROOT + "/runs/v12/work"
OUT = ROOT + "/data/ce"
LO, HI = 0.02, 0.98


def texts(work, pairs):
    refs = pl.read_parquet(f"{work}/refs.parquet", columns=["entity_id", "business_name", "business_address"])
    mens = (pl.scan_parquet(f"{work}/mentions.parquet").select("entity_id", "business_name", "business_address")
            .filter(pl.col("entity_id").is_in(pairs["mention_id"].unique().implode())).collect())
    t = lambda df, idc, col: df.select(pl.col("entity_id").alias(idc),
                                       (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias(col))
    return pairs.join(t(refs, "ref_id", "a"), on="ref_id").join(t(mens, "mention_id", "b"), on="mention_id")


truth = read_matches(ROOT + "/student_resource/dataset/train/train_ground_truth.tsv")
tp = pl.DataFrame([(r, m) for r, ms in truth.items() for m in ms], schema=["ref_id", "mention_id"], orient="row").with_columns(pl.lit(1, dtype=pl.Int8).alias("y"))
val_refs = pl.read_csv(ROOT + "/data/split/val/source1.tsv", separator="\t", quote_char=None, infer_schema=False)["entity_id"]
val_mentions = tp.filter(pl.col("ref_id").is_in(val_refs.implode()))["mention_id"]

# train: fit-state Mentions only, 3 best Candidates by embedding rank, ~1.5M pairs
rows = []
for f in sorted(glob.glob(f"{R}/full/feats_*.parquet")):
    d = pl.read_parquet(f, columns=["ref_id", "mention_id", "sim_emb_rank_m", "sim_full_rank_m"])
    d = d.filter(~pl.col("ref_id").is_in(val_refs.implode()) & ~pl.col("mention_id").is_in(val_mentions.implode())
                 & (pl.col("mention_id").hash(seed=31) % 100 < 18)
                 & ((pl.col("sim_emb_rank_m") <= 3) | (pl.col("sim_full_rank_m") <= 2)))
    rows.append(d.select("ref_id", "mention_id"))
train = pl.concat(rows).unique().join(tp, on=["ref_id", "mention_id"], how="left").with_columns(pl.col("y").fill_null(0))
train = texts(f"{R}/full", train).sample(fraction=1.0, shuffle=True, seed=1)
print("train pairs", train.height, "positives", int(train["y"].sum()))
train.write_parquet(f"{OUT}/train.parquet")

val = pl.read_parquet(f"{R}/train/dense_val_scored.parquet").filter((pl.col("score") > LO) & (pl.col("score") < HI))
val = texts(f"{R}/full", val)
print("val pairs", val.height)
val.write_parquet(f"{OUT}/val.parquet")

test = pl.read_parquet(f"{R}/test/scored.parquet").filter((pl.col("score") > LO) & (pl.col("score") < HI))
test = texts(f"{R}/test", test)
print("test pairs", test.height)
test.write_parquet(f"{OUT}/test.parquet")
