import sys, time
import numpy as np, polars as pl
sys.path.insert(0, "src")
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from ber.block import add_text_views
from ber.evaluate import read_matches
d = "../../data/split/val"
R = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
refs = add_text_views(pl.read_csv(f"{d}/source1.tsv", **R).filter(pl.col("country") == "US"))
mens = add_text_views(pl.concat([pl.read_csv(f"{d}/source2.tsv", **R), pl.read_csv(f"{d}/source3.tsv", **R)]).filter(pl.col("country") == "US"))
truth = read_matches(f"{d}/ground_truth.tsv")
m2r = {m: r for r, ms in truth.items() for m in ms}
mens = mens.sample(50000, seed=1)
true_ref = [m2r.get(m) for m in mens["entity_id"]]
ref_idx = {r: i for i, r in enumerate(refs["entity_id"])}
tr = np.array([ref_idx.get(r, -1) for r in true_ref])
has = tr >= 0
print("US refs", refs.height, "sample mentions with true ref", has.sum())
def run(col, analyzer, ng, max_df, k):
    t = time.time()
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ng, sublinear_tf=True, dtype=np.float32, max_df=max_df)
    vec.fit(refs[col].to_numpy())
    A, B = vec.transform(mens[col].to_numpy()), vec.transform(refs[col].to_numpy())
    h = sp_matmul_topn(A, B.T.tocsr(), top_n=k, threshold=0.0, n_threads=20).tolil()
    hit = np.array([tr[i] in h.rows[i] for i in range(len(tr))])
    return hit, time.time() - t
hits = []
for cfg in [("full_n", "char_wb", (3, 3), 0.05, 10), ("full_n", "char_wb", (3, 3), 1.0, 10), ("full_n", "word", (1, 1), 1.0, 10),
            ("name_n", "word", (1, 1), 1.0, 10), ("addr_n", "word", (1, 1), 1.0, 10), ("name_sq", "char", (3, 3), 0.05, 10), ("addr_n", "char_wb", (3, 3), 0.05, 10)]:
    h, s = run(*cfg); hits.append(h)
    print(cfg, "recall", round(h[has].mean(), 4), "sec", round(s, 1), flush=True)
print("union all", np.any(hits, axis=0)[has].mean())
