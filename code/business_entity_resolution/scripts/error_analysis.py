import sys
sys.path.insert(0, "src")
import glob
import lightgbm as lgb, polars as pl, numpy as np
from ber.pipeline import load_part, label
from ber.resolve import resolve
from ber.evaluate import read_matches, entity_score
work, val, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
model = lgb.Booster(model_file=f"{work}/train/model.txt")
ref_ctx = pl.read_parquet(f"{work}/val/ref_ctx.parquet")
parts = []
for p in sorted(glob.glob(f"{work}/val/feats_*.parquet")):
    df = load_part(p, ref_ctx)
    parts.append(df.select("ref_id", "mention_id").with_columns(pl.Series("score", model.predict(df.select(model.feature_name()).to_numpy().astype(np.float32)))))
scored = pl.concat(parts)
scored.write_parquet(f"{work}/val/scored.parquet")
refs = pl.read_parquet(f"{work}/val/refs.parquet")
truth = read_matches(f"{val}/ground_truth.tsv")
pred = resolve(scored, refs["entity_id"].to_list(), thr)
country = dict(zip(refs["entity_id"], refs["country"]))
rows = [(r, country[r], len(t), len(pred[r]), entity_score(pred[r], t)) for r, t in truth.items()]
df = pl.DataFrame(rows, schema=["ref", "country", "n_true", "n_pred", "f"], orient="row")
print(df.group_by("country").agg(pl.col("f").mean(), pl.len()))
print(df.with_columns((pl.col("n_true") == 0).alias("singleton")).group_by("singleton").agg(pl.col("f").mean(), pl.len()))
print(df.group_by("n_true").agg(pl.col("f").mean(), pl.len()).sort("n_true"))
# error types over pairs
m2r = {m: r for r, ms in truth.items() for m in ms}
cand = set(zip(scored["ref_id"], scored["mention_id"]))
fp_wrong_ref = fp_distractor = fn_not_cand = fn_low = fn_stolen = tp = 0
pred_m = {m: r for r, ms in pred.items() for m in ms}
for m, r in pred_m.items():
    if m2r.get(m) == r: tp += 1
    elif m in m2r: fp_wrong_ref += 1
    else: fp_distractor += 1
for m, r in m2r.items():
    if pred_m.get(m) == r: continue
    if (r, m) not in cand: fn_not_cand += 1
    elif m in pred_m: fn_stolen += 1
    else: fn_low += 1
print(dict(tp=tp, fp_wrong_ref=fp_wrong_ref, fp_distractor=fp_distractor, fn_not_candidate=fn_not_cand, fn_low_score=fn_low, fn_assigned_elsewhere=fn_stolen))
