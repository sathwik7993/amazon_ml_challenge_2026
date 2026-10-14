"""Per-entity expected-F0.5 decision vs a global threshold, on dense validation (labelled).

For each Reference Record, rank its best-assigned Mentions by calibrated probability p and emit the prefix
of length k (k = 0 allowed) that maximizes E[F0.5] ~ 1.25*sum(p_top_k) / (k + 0.25*sum(p_all)); k = 0 is
worth P(no true copy) ~ prod(1 - p). Probabilities are calibrated per score bin (validation true rate).
"""
import sys

import numpy as np
import polars as pl

import os  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = os.environ.get("BER_ROOT", str(Path(__file__).resolve().parents[4]))  # repo root: student_resource/, data/, runs/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from entity_resolution.evaluate import evaluate, read_matches  # noqa: E402
from entity_resolution.resolve import resolve  # noqa: E402

EDGES = np.r_[0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0001]


def ef_decide(best, p_col="p"):
    """best: (ref_id, mention_id, p) with one row per Mention. Returns {ref_id: [mention_id...]}."""
    b = best.filter(pl.col(p_col) > 0.02).sort(["ref_id", p_col], descending=[False, True])
    out = {}
    for ref, ms, ps in b.group_by("ref_id", maintain_order=True).agg("mention_id", p_col).iter_rows():
        ps = np.asarray(ps)
        tot = ps.sum()
        best_v, best_k = np.prod(1 - ps), 0
        c = np.cumsum(ps)
        for k in range(1, len(ps) + 1):
            v = 1.25 * c[k - 1] / (k + 0.25 * tot)
            if v > best_v:
                best_v, best_k = v, k
        if best_k:
            out[ref] = list(ms[:best_k])
    return out


if __name__ == "__main__":
    truth = read_matches(ROOT + "/student_resource/dataset/train/train_ground_truth.tsv")
    vref = pl.read_csv(ROOT + "/data/split/val/source1.tsv", separator="\t", quote_char=None,
                       infer_schema=False)["entity_id"]
    vt = {r: truth.get(r, []) for r in vref.to_list()}
    T = pl.DataFrame([(r, m) for r, ms in vt.items() for m in ms], schema=["ref_id", "mention_id"], orient="row").with_columns(pl.lit(1).alias("y"))
    scored = pl.read_parquet(sys.argv[1])
    best = scored.sort("score", descending=True).unique("mention_id", keep="first").filter(pl.col("ref_id").is_in(vref.implode()))
    best = best.with_columns(pl.Series("bin", np.digitize(best["score"].to_numpy(), EDGES) - 1))
    lab = best.join(T, on=["ref_id", "mention_id"], how="left").with_columns(pl.col("y").fill_null(0))
    cal = lab.group_by("bin").agg(pl.col("y").mean().alias("p"))
    best = best.join(cal, on="bin")
    ids = vref.to_list()
    print("global 0.675:", round(evaluate(resolve(scored, ids, 0.675), vt), 5))
    pred = ef_decide(best)
    print("expected-F per entity:", round(evaluate({r: pred.get(r, []) for r in ids}, vt), 5))
