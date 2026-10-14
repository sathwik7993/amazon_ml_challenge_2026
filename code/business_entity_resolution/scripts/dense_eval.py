"""Score the dense validation (held-out states inside the full-train feature run) with any model dir.

Usage (PYTHONPATH=src): python scripts/dense_eval.py --work ../../runs/v9/work --model ../../runs/v7/work/train \
    --val-refs ../../data/split/val/source1.tsv --truth ../../student_resource/dataset/train [--save scored.parquet]
"""
import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from train_dense import score_all, tune  # noqa: E402

from ber.evaluate import read_matches
from ber.pipeline import READ, _find_truth
from ber.stack import _parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--val-refs", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--save")
    a = ap.parse_args()
    parts = _parts(Path(a.work) / "full")
    truth = read_matches(_find_truth(a.truth))
    val_refs = pl.read_csv(a.val_refs, **READ)["entity_id"]
    val_truth = {r: truth.get(r, []) for r in val_refs.to_list()}
    val_cand = pl.concat([pl.read_parquet(p, columns=["ref_id", "mention_id"]).filter(pl.col("ref_id").is_in(val_refs.implode()))["mention_id"]
                          for p, _ in parts]).unique()
    scored = score_all(a.model, parts, pl.col("mention_id").is_in(val_cand.implode()))
    if a.save:
        scored.write_parquet(a.save)
    tune(scored, val_refs.to_list(), val_truth, a.model)


if __name__ == "__main__":
    main()
