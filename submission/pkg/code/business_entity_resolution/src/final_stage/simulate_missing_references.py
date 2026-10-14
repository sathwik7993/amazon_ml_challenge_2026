"""Make the full-density training run test-like by removing 19% of Reference Records ("orphans").

Test has 5.76 Mentions per Reference Record against 4.67 in training: about 19% of test entities have no
Source 1 record, so their Mentions are Distractors that look exactly like true copies. We reproduce that
by dropping a fixed 19% hash sample of training Reference Records (their Mentions stay, now unmatched).

Input is a completed (non-orphaned) dense work dir: its full/refs.parquet and the cached Blocking chunks
full/blk_*.parquet. Writes the orphaned work dir (refs, Blocking chunks filtered to kept References, the
Mentions file linked), the orphaned embedding file, and the validation Reference list restricted to kept
References (val_refs_orph.tsv), as used by train_dense.py --val-refs and the final-stage scripts.

Usage: python simulate_missing_references.py SRC_WORK DST_WORK EMB_IN EMB_OUT VAL_SOURCE1_TSV
  e.g. python simulate_missing_references.py runs/v9/work runs/v12/work data/emb/emb_full.parquet data/emb/emb_full_orph.parquet \
       data/split/val/source1.tsv
"""
import glob
import os
import shutil
import sys

import polars as pl

src, dst, emb_in, emb_out, val_tsv = sys.argv[1:6]
KEEP_FROM = 190  # keep a Reference Record when hash(entity_id, seed 4242) % 1000 >= 190 (81%)

os.makedirs(f"{dst}/full", exist_ok=True)
refs = pl.read_parquet(f"{src}/full/refs.parquet")
keep = refs.filter(pl.col("entity_id").hash(seed=4242) % 1000 >= KEEP_FROM)
print("refs", refs.height, "kept", keep.height, f"({keep.height / refs.height:.3f})")
keep.write_parquet(f"{dst}/full/refs.parquet")
kept = keep["entity_id"]
for f in sorted(glob.glob(f"{src}/full/blk_*.parquet")):
    pl.read_parquet(f).filter(pl.col("ref_id").is_in(kept.implode())).write_parquet(f"{dst}/full/{os.path.basename(f)}")
shutil.copy(f"{src}/full/mentions.parquet", f"{dst}/full/mentions.parquet")
pl.scan_parquet(emb_in).filter(pl.col("ref_id").is_in(kept.implode())).sink_parquet(emb_out)
v = pl.read_csv(val_tsv, separator="\t", quote_char=None, infer_schema=False)
v.filter(pl.col("entity_id").is_in(kept.implode())).write_csv(f"{dst}/val_refs_orph.tsv", separator="\t")
print("validation refs kept", v.filter(pl.col("entity_id").is_in(kept.implode())).height, "of", v.height)
