"""Cross-encoder reranker for uncertain Candidate pairs (Kaggle, 2x T4).

Final cross-encoder 1. Fine-tunes intfloat/multilingual-e5-base (MIT, 278M) as a pair classifier on
"ref text" [SEP] "mention text", for 1 epoch over 500k easy pairs plus the 406k hard pairs shown twice,
then scores the validation and test uncertain pairs. Smoke step first, checkpoint and outputs written as it goes.
"""
import glob
import os
import time

import numpy as np
import polars as pl
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL = "intfloat/multilingual-e5-base"
MAX_LEN = 128
BATCH = 128
LR = 2e-5
OUT = "/kaggle/working"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find(name):
    return glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0]


tok = AutoTokenizer.from_pretrained(MODEL)


def batches(df, bs, with_y):
    a, b = df["a"].to_list(), df["b"].to_list()
    y = df["y"].to_numpy().astype(np.float32) if with_y else None
    for i in range(0, len(a), bs):
        enc = tok(a[i:i + bs], b[i:i + bs], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        yield enc, (torch.from_numpy(y[i:i + bs]) if with_y else None)


def train(df, steps_limit=None):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).cuda()
    model = nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n_steps = (df.height + BATCH - 1) // BATCH if steps_limit is None else steps_limit
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * n_steps), n_steps)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    t0, run = time.time(), 0.0
    for step, (enc, y) in enumerate(batches(df, BATCH, True)):
        if step >= n_steps:
            break
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits.squeeze(-1)
            loss = loss_fn(logits.float(), y.cuda())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 200 == 0:
            log(f"step {step}/{n_steps} loss {run:.4f} ({(time.time() - t0) / max(step, 1):.2f}s/step)")
    return model


@torch.no_grad()
def predict(model, df):
    model.eval()
    out = []
    for enc, _ in batches(df, 1024, False):
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    os.system("nvidia-smi --query-gpu=name,memory.total --format=csv")
    easy = pl.read_parquet(find("train.parquet")).sample(n=500_000, seed=5)
    hard = pl.read_parquet(find("hard.parquet")).select(easy.columns)
    # hard (uncertain-band) pairs twice, easy pairs once, shuffled
    tr = pl.concat([easy, hard, hard]).sample(fraction=1.0, shuffle=True, seed=9)
    va = pl.read_parquet(find("val.parquet"))
    te = pl.read_parquet(find("test.parquet"))
    log("train", tr.height, "val", va.height, "test", te.height)
    # 1) smoke: whole code path on a few thousand rows
    m = train(tr.head(4096), steps_limit=8)
    p = predict(m, va.head(2048))
    log("smoke ok", p[:5])
    del m
    torch.cuda.empty_cache()
    # 2) full fine-tune, one epoch
    model = train(tr)
    torch.save(model.module.state_dict(), f"{OUT}/ce_model.pt")
    # 3) score validation first (the go/no-go input), then test
    va.select("ref_id", "mention_id").with_columns(pl.Series("ce", predict(model, va))).write_parquet(f"{OUT}/ce_val.parquet")
    log("wrote ce_val")
    te.select("ref_id", "mention_id").with_columns(pl.Series("ce", predict(model, te))).write_parquet(f"{OUT}/ce_test.parquet")
    log("wrote ce_test; done")
