"""Average the stacker scores of several dense models (bagging), validated on the same held-out states.

Usage: python bag.py OUT_DIR MODEL_WORK [MODEL_WORK ...]
Each MODEL_WORK has train/dense_val_scored.parquet and test/scored.parquet over the same Candidate rows.
Writes OUT_DIR/val_scored.parquet, OUT_DIR/scored.parquet and a link to refs.parquet.
"""
import os
import sys

import numpy as np
import polars as pl

import os  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = os.environ.get("BER_ROOT", str(Path(__file__).resolve().parents[4]))  # repo root: student_resource/, data/, runs/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ber.evaluate import evaluate, read_matches  # noqa: E402
from ber.resolve import resolve  # noqa: E402

out, works = sys.argv[1], sys.argv[2:]
K = ["ref_id", "mention_id"]


def mean_of(paths):
    """Mean score over models. Rows are aligned by order when every file has the same key order (checked by
    hash), which avoids a 65M-row string join; otherwise falls back to a join."""
    base = pl.read_parquet(paths[0], columns=K + ["score"])
    h0 = base.select(pl.struct(K).hash(seed=5))
    total, aligned = base["score"].cast(pl.Float64), True
    for p in paths[1:]:
        o = pl.read_parquet(p, columns=K + ["score"])
        if o.height != base.height or not o.select(pl.struct(K).hash(seed=5)).to_series().equals(h0.to_series()):
            aligned = False
            break
        total = total + o["score"].cast(pl.Float64)
        del o
    if aligned:
        print(paths[0], "rows", base.height, "aligned by order")
        return base.select(K).with_columns((total / len(paths)).cast(pl.Float32).alias("score"))
    frames = [pl.read_parquet(p, columns=K + ["score"]).rename({"score": f"s{i}"}) for i, p in enumerate(paths)]
    d = frames[0]
    for f in frames[1:]:
        d = d.join(f, on=K, how="full", coalesce=True)
    cols = [f"s{i}" for i in range(len(paths))]
    print(paths[0], "rows", d.height, "joined; missing per model", [d[c].null_count() for c in cols])
    return d.select(*K, pl.mean_horizontal(cols).alias("score"))


truth_all = read_matches(ROOT + "/student_resource/dataset/train/train_ground_truth.tsv")
ids = pl.read_csv(f"{works[0]}/val_refs_orph.tsv" if os.path.exists(f"{works[0]}/val_refs_orph.tsv") else
                  ROOT + "/runs/v12/work/val_refs_orph.tsv", separator="\t", quote_char=None,
                  infer_schema=False)["entity_id"].to_list()
truth = {r: truth_all.get(r, []) for r in ids}
grid = np.round(np.arange(0.55, 0.85, 0.025), 3)
val_paths = [f"{w}/train/dense_val_scored.parquet" for w in works]
for p in val_paths:
    s = pl.read_parquet(p, columns=K + ["score"])
    print(p, max((round(evaluate(resolve(s, ids, t), truth), 5), t) for t in grid))
v = mean_of(val_paths)
print("bagged", max((round(evaluate(resolve(v, ids, t), truth), 5), t) for t in grid))
os.makedirs(out, exist_ok=True)
v.write_parquet(f"{out}/val_scored.parquet")
mean_of([f"{w}/test/scored.parquet" for w in works]).write_parquet(f"{out}/scored.parquet")
if not os.path.exists(f"{out}/refs.parquet"):
    os.link(f"{works[0]}/test/refs.parquet", f"{out}/refs.parquet")
print("wrote", out)
