"""Score the test set with a trained model dir and write the submission files.

Usage (PYTHONPATH=src): python scripts/score_test.py --test DIR --work TEST_WORK --train-work TRAIN_WORK \
    --emb EMB_PARQUET --threshold T --out OUT_DIR
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import polars as pl

from ber.pipeline import feature_parts, log, write_outputs
from ber.resolve import resolve
from ber.stack import score_stack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--train-work", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tw, work = Path(a.train_work), Path(a.work)
    pruner = lgb.Booster(model_file=str(tw / "pruner.txt"))
    te = pl.read_parquet(tw / "token_te.parquet")
    refs, _, _ = feature_parts(a.test, work, pruner, 6, 1.0, 400_000, emb=pl.scan_parquet(a.emb), te=te)
    path = work / "scored.parquet"
    if path.exists():
        scored = pl.read_parquet(path)
    else:
        scored = score_stack(tw, work, log)
        scored.write_parquet(path)
    ids = refs["entity_id"].to_list()
    write_outputs(resolve(scored, ids, a.threshold), scored, ids, a.out)
    log(f"wrote {a.out} at threshold {a.threshold}")


if __name__ == "__main__":
    main()
