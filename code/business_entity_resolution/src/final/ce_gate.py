"""Cross-encoder as a keep/drop gate on each Mention's best pair (the stacker still picks the Reference Record).

For every Mention, its best row by stacker score is kept as the Candidate assignment. If that row has a
cross-encoder score (stacker score in (0.02, 0.98)), its score is replaced by a logistic regression on
(logit(stacker), ce, logit(stacker) * ce), fitted on best rows only, out-of-fold with 5 folds grouped by
Reference Record. Non-best rows are left alone, so no Mention changes Reference Record; since only a
Mention's best row can be kept, lowering the other rows does not matter and they are capped below it.

Go/no-go on dense validation (each at its own best threshold); on GO writes TEST_OUT/{scored,val_scored,refs}.
Usage: python ce_gate.py CE_DIR VAL_SCORED TEST_SCORED TEST_OUT
"""
import os
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

import os  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = os.environ.get("BER_ROOT", str(Path(__file__).resolve().parents[4]))  # repo root: student_resource/, data/, runs/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ber.evaluate import evaluate, read_matches  # noqa: E402
from ber.resolve import resolve  # noqa: E402

W = ROOT + "/runs/v12/work"
K = ["ref_id", "mention_id"]
ce_dir, val_scored, test_scored, test_out = sys.argv[1:5]
GRID = np.round(np.arange(0.4, 0.96, 0.025), 3)


def design(df):
    s = df["score"].cast(pl.Float64).clip(1e-4, 1 - 1e-4).to_numpy()
    ls = np.log(s / (1 - s))
    ce, co = df["ce"].to_numpy(), df["ce_oth"].fill_null(-10.0).to_numpy()
    return np.c_[ls, ce, ls * ce, s - df["s2"].cast(pl.Float64).to_numpy(), ce - co, co > -10]


def ranked(scored, ce):
    """Rows with score >= 0.02 (lower rows are never kept) with Mention rank, second-best score and the best
    cross-encoder score among the Mention's other Candidates."""
    r = (scored.filter(pl.col("score") >= 0.02).join(ce, on=K, how="left")
         .with_columns(pl.col("score").rank("ordinal", descending=True).over("mention_id").alias("_r")))
    return r.with_columns(pl.col("score").filter(pl.col("_r") == 2).first().over("mention_id").fill_null(0).alias("s2"),
                          pl.col("ce").filter(pl.col("_r") > 1).max().over("mention_id").alias("ce_oth"))


def gated(scored, ce, new_score):
    """Replace the best row's score of Mentions whose best row has a cross-encoder score.
    Other rows of those Mentions are capped at 0.9 x the new best, so the argmax cannot change."""
    rank = ranked(scored, ce)
    best = rank.filter((pl.col("_r") == 1) & pl.col("ce").is_not_null())
    best = best.with_columns(pl.Series("_new", new_score(best)).cast(pl.Float64))
    out = rank.select(*K, "score").join(best.select("mention_id", "_new", pl.col("ref_id").alias("_bref")), on="mention_id", how="left")
    new = out.select(*K, pl.when(pl.col("_new").is_null()).then(pl.col("score").cast(pl.Float64))
                     .when(pl.col("ref_id") == pl.col("_bref")).then(pl.col("_new"))
                     .otherwise(pl.min_horizontal(pl.col("score").cast(pl.Float64), 0.9 * pl.col("_new")))
                     .cast(pl.Float32).alias("_g"))
    return (scored.join(new, on=K, how="left")
            .select(*K, pl.coalesce("_g", pl.col("score").cast(pl.Float32)).alias("score")))


def load_ce(dirs, name):
    """Cross-encoder logits from one or more comma-separated output dirs, averaged per pair."""
    frames = [pl.read_parquet(f"{d}/{name}.parquet").rename({"ce": f"ce{i}"}) for i, d in enumerate(dirs.split(","))]
    d = frames[0]
    for f in frames[1:]:
        d = d.join(f, on=K, how="inner")
    return d.select(*K, pl.mean_horizontal([f"ce{i}" for i in range(len(frames))]).alias("ce"))


def best_f(scored, ids, truth):
    return max((round(evaluate(resolve(scored, ids, t), truth), 5), t) for t in GRID)


truth_all = read_matches(ROOT + "/student_resource/dataset/train/train_ground_truth.tsv")
ids = pl.read_csv(f"{W}/val_refs_orph.tsv", separator="\t", quote_char=None, infer_schema=False)["entity_id"].to_list()
truth = {r: truth_all.get(r, []) for r in ids}
T = pl.DataFrame([(r, m) for r in ids for m in truth[r]], schema=K, orient="row").with_columns(pl.lit(1).alias("y"))

vs = pl.scan_parquet(val_scored).select(*K, "score").filter(pl.col("score") >= 0.02).collect()  # lower rows never kept
vce = load_ce(ce_dir, "ce_val")
vbest = (ranked(vs, vce).filter((pl.col("_r") == 1) & pl.col("ce").is_not_null())
         .filter(pl.col("ref_id").is_in(pl.Series(ids).implode()))  # labels exist only for validation References
         .join(T, on=K, how="left").with_columns(pl.col("y").fill_null(0)))
print("best rows with a cross-encoder score:", vbest.height, "true rate", round(vbest["y"].mean(), 4))
fold = (vbest["ref_id"].hash(seed=11) % 5).to_numpy()
X, y = design(vbest), vbest["y"].to_numpy()
oof = np.zeros(len(y))
for k in range(5):
    oof[fold == k] = LogisticRegression(max_iter=1000).fit(X[fold != k], y[fold != k]).predict_proba(X[fold == k])[:, 1]
final = LogisticRegression(max_iter=1000).fit(X, y)
print("coefficients:", final.coef_.round(3).tolist(), final.intercept_.round(3).tolist())
oof_df = vbest.select(K).with_columns(pl.Series("oof", oof))
val_g = gated(vs, vce, lambda b: b.join(oof_df, on=K, how="left")["oof"].to_numpy())

base, gate = best_f(vs, ids, truth), best_f(val_g, ids, truth)
print("stacker alone:", base, " gated:", gate)
if gate[0] <= base[0] + 0.0005:
    print("NO-GO: gain below 0.0005")
    sys.exit(1)
if test_scored == "NONE":
    print("GO (validation only)")
    sys.exit(0)
os.makedirs(test_out, exist_ok=True)
val_g.write_parquet(f"{test_out}/val_scored.parquet")
ts = pl.read_parquet(test_scored, columns=K + ["score"])
gated(ts, load_ce(ce_dir, "ce_test"), lambda b: final.predict_proba(design(b))[:, 1]).write_parquet(f"{test_out}/scored.parquet")
if not os.path.exists(f"{test_out}/refs.parquet"):
    os.link(f"{W}/test/refs.parquet", f"{test_out}/refs.parquet")
print("GO: wrote", test_out)
