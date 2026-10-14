"""Hard training pairs for the cross-encoder: uncertain Candidate rows of fit-state Mentions.

The gate works on each Mention's pairs whose stacker score is in (0.02, 0.98), so the cross-encoder should
learn from those pairs, not from easy top-3 embedding Candidates. Here fit-state Mentions that neither bagged
stacker trained on (samples: seed 23 and seed 101, 15% each) are scored by the v12 stacker in the full-density
run, and every row in the band is exported with its label and "name | address" texts.
Two hash buckets of 20% of Mentions each are scored separately to bound memory.

Usage: python export_hard_training_pairs.py OUT_PARQUET [BUCKETS, e.g. 0,1,2,3,4]
"""
import sys

import polars as pl

import os  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = os.environ.get("BER_ROOT", str(Path(__file__).resolve().parents[4]))  # repo root: student_resource/, data/, runs/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "scripts"))
from entity_resolution.evaluate import read_matches  # noqa: E402
from entity_resolution.pipeline import label, log  # noqa: E402
from entity_resolution.stacker import _parts  # noqa: E402
from train_dense import score_all  # noqa: E402

R = ROOT + "/runs/v12/work"
LO, HI = 0.02, 0.98
out = sys.argv[1]

truth = read_matches(ROOT + "/student_resource/dataset/train/train_ground_truth.tsv")
val_refs = pl.read_csv(f"{R}/val_refs_orph.tsv", separator="\t", quote_char=None, infer_schema=False)["entity_id"]
val_mentions = pl.Series([m for r in val_refs.to_list() for m in truth.get(r, [])])
parts = _parts(f"{R}/full")
# Mentions that are Candidates of a validation Reference Record stay out (their argmax involves validation)
val_cand = pl.concat([pl.read_parquet(p, columns=["ref_id", "mention_id"]).filter(pl.col("ref_id").is_in(val_refs.implode()))["mention_id"]
                      for p, _ in parts]).unique()
m = pl.col("mention_id")
fit = (~m.is_in(val_mentions.implode()) & ~m.is_in(val_cand.implode())
       & (m.hash(seed=23) % 1000 >= 150) & (m.hash(seed=101) % 1000 >= 150))

band = []
BUCKETS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1]
for bucket in BUCKETS:
    keep = fit & (m.hash(seed=77) % 5 == bucket)
    s = score_all(f"{R}/train", parts, keep)
    b = label(s.filter((pl.col("score") > LO) & (pl.col("score") < HI)), truth)
    log(f"bucket {bucket}: scored {s.height} rows, band {b.height}, positives {int(b['y'].sum())}")
    band.append(b)
    del s
band = pl.concat(band)

refs = pl.read_parquet(f"{R}/full/refs.parquet", columns=["entity_id", "business_name", "business_address"])
mens = (pl.scan_parquet(f"{R}/full/mentions.parquet").select("entity_id", "business_name", "business_address")
        .filter(pl.col("entity_id").is_in(band["mention_id"].unique().implode())).collect())
t = lambda df, idc, col: df.select(pl.col("entity_id").alias(idc),
                                   (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias(col))
band = (band.join(t(refs, "ref_id", "a"), on="ref_id").join(t(mens, "mention_id", "b"), on="mention_id")
        .select("ref_id", "mention_id", "a", "b", "y", "score"))
band.write_parquet(out)
log(f"wrote {out}: {band.height} hard pairs, positive rate {band['y'].mean():.3f}")
