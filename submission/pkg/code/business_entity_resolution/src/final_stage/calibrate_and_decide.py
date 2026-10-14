"""Label-free recalibration of test scores for test's extra look-alike Distractors, then per-country thresholds.

Assumption (checked on the >0.99 band, identical on test and dense validation): each Reference Record has the
same distribution of true copies over score bins on test as on validation. So the expected number of true
Mentions per Reference Record in a score bin is val_count_per_ref(bin) * val_true_rate(bin), and the test
true rate of that bin is that expectation divided by test_count_per_ref(bin). Countries without validation
data (France) use the pooled US+India profile. With the recalibrated probabilities we pick, per country, the
threshold maximizing the expected macro F0.5, and write the submission.

Usage: python calibrate_and_decide.py VAL_SCORED TEST_WORK OUT_DIR [VAL_REFS_TSV]
"""
import sys

import numpy as np
import polars as pl

import os  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = os.environ.get("BER_ROOT", str(Path(__file__).resolve().parents[4]))  # repo root: student_resource/, data/, runs/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from entity_resolution.evaluate import read_matches  # noqa: E402
from entity_resolution.pipeline import write_outputs  # noqa: E402
from entity_resolution.resolve import resolve  # noqa: E402

val_scored, test_work, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
EDGES = np.r_[0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0001]
GRID = [0.5, 0.6, 0.65, 0.675, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]

truth = read_matches(ROOT + "/student_resource/dataset/train/train_ground_truth.tsv")
VAL_REFS = sys.argv[4] if len(sys.argv) > 4 else ROOT + "/data/split/val/source1.tsv"
vref = (pl.read_csv(VAL_REFS, separator="\t", quote_char=None,
                    infer_schema=False).select(pl.col("entity_id").alias("ref_id"), "country"))
T = pl.DataFrame([(r, m) for r in vref["ref_id"].to_list() for m in truth.get(r, [])], schema=["ref_id", "mention_id"],
                 orient="row").with_columns(pl.lit(1).alias("y"))


def best_of(scored, refs):
    b = scored.sort("score", descending=True).unique("mention_id", keep="first").join(refs, on="ref_id")
    return b.with_columns(pl.Series("bin", np.digitize(b["score"].to_numpy(), EDGES) - 1))


bv = best_of(pl.read_parquet(val_scored), vref).join(T, on=["ref_id", "mention_id"], how="left").with_columns(pl.col("y").fill_null(0))
tref = pl.read_parquet(f"{test_work}/refs.parquet", columns=["entity_id", "country"]).rename({"entity_id": "ref_id"})
test_scored = pl.read_parquet(f"{test_work}/scored.parquet")
bt = best_of(test_scored, tref)


def true_per_ref(countries):
    n = vref.filter(pl.col("country").is_in(countries)).height
    return (bv.filter(pl.col("country").is_in(countries)).group_by("bin").agg((pl.col("y").sum() / n).alias("true_pr")))


thresholds, calibrated = {}, []
for country in tref["country"].unique().to_list():
    prof = true_per_ref([country] if country in vref["country"].unique().to_list() else ["US", "India"])
    nt = tref.filter(pl.col("country") == country).height
    cal = (bt.filter(pl.col("country") == country).group_by("bin").agg((pl.len() / nt).alias("test_pr"))
           .join(prof, on="bin", how="left").with_columns(pl.col("true_pr").fill_null(0.0))
           .with_columns((pl.col("true_pr") / pl.col("test_pr")).clip(0.0, 1.0).alias("p_cal")))
    b = bt.filter(pl.col("country") == country).join(cal.select("bin", "p_cal"), on="bin")
    calibrated.append(b)
    res = []
    for t in GRID:
        g = b.group_by("ref_id").agg(
            pl.col("p_cal").filter(pl.col("score") >= t).sum().alias("tp"),
            (1 - pl.col("p_cal")).filter(pl.col("score") >= t).sum().alias("fp"),
            pl.col("p_cal").filter(pl.col("score") < t).sum().alias("fn"),
            (pl.col("score") >= t).sum().alias("n"))
        d = tref.filter(pl.col("country") == country).join(g, on="ref_id", how="left").fill_null(0)
        f = pl.when(pl.col("n") == 0).then((-pl.col("fn")).exp()).otherwise(
            1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * pl.col("fn") + pl.col("fp")))
        res.append((round(d.select(f.mean()).item(), 5), t))
    thresholds[country] = max(res)[1]
    print(country, "calibrated p by bin:", [(round(float(EDGES[r[0]]), 3), round(r[1], 2)) for r in cal.sort("bin").select("bin", "p_cal").rows()])
    print(country, "expected F by threshold:", res, "-> chosen", thresholds[country])

ids = tref["ref_id"].to_list()
sys.path.insert(0, str(HERE))
from expected_f05_decision import ef_decide  # noqa: E402
best = pl.concat(calibrated).rename({"p_cal": "p"})
thr_pred = resolve(test_scored, ids, 0.675, dict(zip(tref["ref_id"].to_list(), [thresholds[c] for c in tref["country"].to_list()])))
ef_pred = ef_decide(best)
ef_pred = {r: ef_pred.get(r, []) for r in ids}


def expected_f(pred):
    sel = pl.DataFrame([(r, m) for r, ms in pred.items() for m in ms], schema=["ref_id", "mention_id"], orient="row").with_columns(pl.lit(True).alias("s"))
    b = best.join(sel, on=["ref_id", "mention_id"], how="left").with_columns(pl.col("s").fill_null(False))
    g = b.group_by("ref_id").agg(pl.col("p").filter("s").sum().alias("tp"), (1 - pl.col("p")).filter("s").sum().alias("fp"),
                                 pl.col("p").filter(~pl.col("s")).sum().alias("fn"), pl.col("s").sum().alias("n"))
    d = tref.join(g, on="ref_id", how="left").fill_null(0)
    f = pl.when(pl.col("n") == 0).then((-pl.col("fn")).exp()).otherwise(1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * pl.col("fn") + pl.col("fp")))
    return round(d.select(f.mean()).item(), 5)


print("expected F: per-country thresholds", expected_f(thr_pred), " per-entity expected-F", expected_f(ef_pred))
changed = sum(1 for r in ids if sorted(thr_pred[r]) != sorted(ef_pred[r]))
print("Reference Records changed:", changed, f"({changed / len(ids):.2%})")
write_outputs(ef_pred, test_scored, ids, out_dir)
print("written", out_dir)
