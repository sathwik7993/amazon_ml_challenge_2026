"""Embedding Blocking source (runs on GPU, e.g. Kaggle 2x T4).

Encodes "name | address" of every Record with multilingual-e5-small (MIT license), then for each
Mention finds the top-k most similar Reference Records of the same country by exact cosine search.
Writes <out>/emb_<dataset>.parquet with (ref_id, mention_id, sim_emb).

Kaggle notebook (2x T4): run with `kaggle kernels push -p src/gpu_jobs/embedding_knn`; it reads the raw data
and splits from /kaggle/input and writes emb_*.parquet to /kaggle/working. For local use, see
src/entity_resolution/embed_knn.py (same code, command-line interface).
"""
import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "intfloat/multilingual-e5-small"
READ = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find(d, n):
    hits = [p for p in glob.glob(f"{d}/**/*source{n}.tsv", recursive=True)]
    assert len(hits) == 1, (d, n, hits)
    return hits[0]


class Encoder:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
        self.model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        log("encoder on", torch.cuda.device_count(), "GPUs")

    @torch.inference_mode()
    def __call__(self, texts, batch=1024):
        out = np.empty((len(texts), 384), dtype=np.float16)
        order = np.argsort([len(t) for t in texts])  # length-sorted batches waste less padding
        t0 = time.time()
        for i in range(0, len(texts), batch):
            idx = order[i:i + batch]
            enc = self.tok([texts[j] for j in idx], padding=True, truncation=True, max_length=64, return_tensors="pt")
            enc = {k: v.cuda() for k, v in enc.items()}
            h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
            out[idx] = e.cpu().numpy()
            if (i // batch) % 500 == 0:
                log(f"  encoded {i + len(idx)}/{len(texts)} ({(i + len(idx)) / (time.time() - t0 + 1e-9):.0f}/s)")
        return out


def texts(df):
    return ("query: " + df["business_name"] + " | " + df["business_address"]).to_list()


@torch.inference_mode()
def topk(ref_emb, men_emb, k, budget_bytes=1.5e9):
    """Exact cosine top-k in fp16; the query chunk is sized so the score block stays under budget_bytes."""
    R = torch.from_numpy(ref_emb).cuda()
    chunk = max(64, int(budget_bytes // (2 * R.shape[0])))
    sims, idx = [], []
    for i in range(0, len(men_emb), chunk):
        q = torch.from_numpy(men_emb[i:i + chunk]).cuda()
        scores = q @ R.T
        s, j = scores.topk(min(k, R.shape[0]), dim=1)
        sims.append(s.float().cpu().numpy())
        idx.append(j.cpu().numpy())
        del scores, q
    del R
    torch.cuda.empty_cache()
    return np.concatenate(sims), np.concatenate(idx)


def run(name, d, out, enc, k, smoke):
    path = Path(out) / f"emb_{name}.parquet"
    if path.exists():
        log("skip", path)
        return
    refs = pl.read_csv(find(d, 1), **READ).fill_null("")
    mens = pl.concat([pl.read_csv(find(d, 2), **READ), pl.read_csv(find(d, 3), **READ)]).fill_null("")
    if smoke:
        refs, mens = refs.head(3000), mens.head(6000)
    log(f"{name}: {refs.height} refs, {mens.height} mentions")
    re_, me_ = enc(texts(refs)), enc(texts(mens))
    if not smoke:  # keep the expensive part as an output even if the search step fails
        np.save(Path(out) / f"emb_{name}_refs.npy", re_)
        np.save(Path(out) / f"emb_{name}_mentions.npy", me_)
    parts = []
    for (country,), r in refs.with_row_index("i").group_by(["country"]):
        m = mens.with_row_index("i").filter(pl.col("country") == country)
        if m.height == 0:
            continue
        ri, mi = r["i"].to_numpy(), m["i"].to_numpy()
        s, j = topk(re_[ri], me_[mi], k)
        parts.append(pl.DataFrame({
            "ref_id": r["entity_id"].gather(j.ravel()),
            "mention_id": m["entity_id"].gather(np.repeat(np.arange(len(mi)), s.shape[1])),
            "sim_emb": s.ravel().astype(np.float32),
        }))
        log(f"  {country}: {m.height} mentions searched")
    df = pl.concat(parts)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)
    truth = glob.glob(f"{d}/**/*ground_truth.tsv", recursive=True)
    if truth and not smoke:
        gt = pl.read_csv(truth[0], **READ).fill_null("").with_columns(pl.col("matched_entity_ids").str.split(","))
        tp = gt.explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
        tp = tp.select(pl.col("source1_entity_id").alias("ref_id"), pl.col("matched_entity_ids").alias("mention_id"))
        log(f"{name}: embedding recall@{k} = {tp.join(df, on=['ref_id', 'mention_id']).height / tp.height:.4f}")
    log("wrote", path, df.height)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs=2, action="append", required=True, metavar=("NAME", "DIR"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    enc = Encoder()
    for name, d in a.dataset:
        run(name, d, a.out, enc, a.k, a.smoke)



if __name__ == "__main__":
    import sys
    RAW = glob.glob("/kaggle/input/**/amlc26-er-raw", recursive=True)[0]
    SPL = glob.glob("/kaggle/input/**/amlc26-er-splits", recursive=True)[0]
    os.system("nvidia-smi --query-gpu=name,memory.total --format=csv")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    # 1) smoke: whole code path on a few thousand rows
    sys.argv = ["x", "--dataset", "smoke", f"{SPL}/val", "--out", "/kaggle/working/smoke", "--smoke"]
    main()
    # 2) memory smoke: exact search at the largest full-train country size (India, ~883k Reference Records)
    rng = np.random.default_rng(0)
    R = rng.standard_normal((900_000, 384)).astype(np.float16)
    s, j = topk(R, R[:20_000], 10)
    log("memory smoke ok, peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
    del R, s, j
    # 3) full run: the whole training set, so training sees test-scale embedding neighbourhoods
    sys.argv = ["x", "--out", "/kaggle/working", "--dataset", "full", f"{RAW}/train"]
    main()
