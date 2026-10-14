"""End-to-end pipeline: data dir -> Blocking -> pruning -> pair features -> LightGBM -> resolve -> output TSVs.

A data dir holds source1.tsv, source2.tsv, source3.tsv (and ground_truth.tsv for training),
or the challenge's own names (train_source1.tsv / test_source1.tsv ...).

Mentions are processed in hash chunks so memory stays bounded at test scale. Every stage writes a
parquet checkpoint into the work dir and is skipped when that file exists, so a crashed run resumes
from the last finished chunk.
"""
import argparse
import json
import math
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.block import Blocker, add_text_views, key_candidates, key_table, name_candidates, name_table, with_keys
from ber.normalize import STATE_CODES
from ber.evaluate import evaluate, read_matches
from ber.features import context_features, pair_features
from ber.ghost import cluster_features, keyed_pairs, pair_number_features, record_keys
from ber.resolve import resolve
from ber.sibling import legal_features, name_diff, sibling_features, target_features, token_counts, token_stats, token_target_rates
from ber.stack import score_stack, train_stack

READ = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
SIM_COLS = ["sim_full", "sim_fullx", "sim_name", "sim_squash", "sim_skel", "sim_addr", "sim_fullw"]
CONTEXT_COLS = ["sim_full", "sim_fullw", "name_tset", "addr_tset", "sq_jw"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _find(data_dir, n):
    d = Path(data_dir)
    for name in (f"source{n}.tsv", f"train_source{n}.tsv", f"test_source{n}.tsv"):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"no source{n} file in {d}")


def _find_truth(data_dir):
    for name in ("ground_truth.tsv", "train_ground_truth.tsv"):
        if (Path(data_dir) / name).exists():
            return Path(data_dir) / name
    return None


def _cached(path, build):
    path = Path(path)
    if path.exists():
        return pl.read_parquet(path)
    t = time.time()
    df = build()
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)  # atomic: a crash never leaves a half-written checkpoint
    log(f"built {path.parent.name}/{path.name} rows={df.height} in {time.time() - t:.0f}s")
    return df


def load_records(data_dir, work):
    refs = _cached(work / "refs.parquet", lambda: add_text_views(pl.read_csv(_find(data_dir, 1), **READ).fill_null("")))
    mens = _cached(work / "mentions.parquet", lambda: add_text_views(
        pl.concat([pl.read_csv(_find(data_dir, 2), **READ), pl.read_csv(_find(data_dir, 3), **READ)]).fill_null("")))
    return refs, mens


def mention_chunks(mens, chunk_size):
    n = max(1, math.ceil(mens.height / chunk_size))
    key = mens["entity_id"].hash(seed=11) % n
    return [mens.filter(key == i) for i in range(n)]


def blocked(blocker, mens, emb=None, raw=None, keys=None):
    """Raw Candidates for some Mentions, with Mention-side Blocking context (pruner input).

    `emb` is the optional embedding Blocking source (ref_id, mention_id, sim_emb) from embed_knn.py.
    `raw` is this chunk's cached text Blocking output, if `block_chunks` already built it.
    """
    cands = blocker.query(mens) if raw is None else raw
    if "sim_keyx" not in cands.columns:  # text Blocking cached before the exact-key source existed
        cands = cands.join(key_candidates(keys[0], mens), on=["ref_id", "mention_id"], how="full", coalesce=True).fill_null(0.0)
    if "sim_namex" not in cands.columns:  # ... or before the core-name source existed
        cands = cands.join(name_candidates(keys[1], mens), on=["ref_id", "mention_id"], how="full", coalesce=True).fill_null(0.0)
    sim_cols = SIM_COLS + ["sim_keyx", "sim_namex"]
    if emb is not None:
        e = emb.filter(pl.col("mention_id").is_in(mens["entity_id"].implode()))
        e = e.collect() if isinstance(e, pl.LazyFrame) else e
        cands = cands.join(e, on=["ref_id", "mention_id"], how="full", coalesce=True).fill_null(0.0)
        sim_cols = sim_cols + ["sim_emb"]
    return context_features(cands, sim_cols, "mention_id")


def load_emb(emb_dir, name):
    if emb_dir is None:
        return None
    path = Path(emb_dir) / f"emb_{name}.parquet"
    log("embedding source:", path)
    return pl.scan_parquet(path)  # lazy: each chunk reads only its own Mentions' rows


def prune(pruner, cands, top_n):
    """Keep each Mention's top_n Candidates by the pruner score: this is the final candidate set."""
    p = pruner.predict(cands.select(pruner.feature_name()).to_numpy().astype(np.float32), num_threads=0)
    keep = pl.col("p0").rank("ordinal", descending=True).over("mention_id") <= top_n
    for col in ("sim_keyx", "sim_namex"):  # exact-key and core-name Candidates always survive pruning
        if col in cands.columns:
            keep = keep | (pl.col(col) > 0)
    return cands.with_columns(pl.Series("p0", p.astype(np.float32))).filter(keep)


def context_cols(columns):
    return CONTEXT_COLS + (["sim_emb"] if "sim_emb" in columns else [])


def chunk_features(refs, mens, pruned):
    return context_features(pair_features(pruned, refs, mens), context_cols(pruned.columns), "mention_id")


def feature_parts(data_dir, work, pruner, top_n, k_scale, chunk_size, raw_cands=None, emb=None, te=None):
    """Per-chunk feature parquet files plus the Reference-side context table (all checkpointed)."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    manifest = work / "parts.json"
    if manifest.exists():  # every chunk already built: skip loading the Mentions (they are large)
        parts = [work / p for p in json.loads(manifest.read_text())]
        refs = pl.read_parquet(work / "refs.parquet")
    else:
        refs, mens = load_records(data_dir, work)
        chunks = mention_chunks(mens, chunk_size)
        parts, blocker, keys = [], None, None
        for i, m in enumerate(chunks):
            path = work / f"feats_{i:03d}.parquet"
            parts.append(path)
            if path.exists():
                continue
            log(f"chunk {i + 1}/{len(chunks)}: {m.height} mentions")
            if raw_cands is not None:
                cands = raw_cands.filter(pl.col("mention_id").is_in(m["entity_id"].implode()))
            else:
                blk = work / f"blk_{i:03d}.parquet"
                raw = pl.read_parquet(blk) if blk.exists() else None
                if raw is None:
                    blocker = blocker or Blocker(refs, k_scale=k_scale, log=log)
                elif keys is None:
                    keys = (key_table(refs), name_table(refs))
                cands = blocked(blocker, m, emb, raw, keys)
            _cached(path, lambda: chunk_features(refs, m, prune(pruner, cands, top_n)))
        del blocker, mens, chunks
        manifest.write_text(json.dumps([p.name for p in parts]))
    ctx_paths = [p.with_name(p.name.replace("feats_", "refctx_")) for p in parts]
    if not all(c.exists() for c in ctx_paths):
        # Reference-side context needs every chunk at once; it is then split back per chunk so that
        # scoring joins each chunk only with its own rows.
        cols = context_cols(list(pl.read_parquet_schema(parts[0])))
        narrow = pl.concat([pl.read_parquet(p, columns=["ref_id", "mention_id", *cols])
                            .with_columns(pl.lit(i, dtype=pl.Int16).alias("_part")) for i, p in enumerate(parts)])
        ctx = context_features(narrow, cols, "ref_id").select("_part", "ref_id", "mention_id", pl.selectors.ends_with("_r"))
        del narrow
        for i, c in enumerate(ctx_paths):
            ctx.filter(pl.col("_part") == i).drop("_part").write_parquet(c)
        log(f"built {len(ctx_paths)} Reference-side context files")
    sib_paths = [p.with_name(p.name.replace("feats_", "sib_")) for p in parts]
    if not all(p.exists() for p in sib_paths):
        # Sibling-word statistics need every chunk at once (per-token counts over the whole dataset).
        names = pl.read_parquet(work / "mentions.parquet", columns=["entity_id", "name_n"])
        diff_paths = [p.with_name(p.name.replace("feats_", "diff_")) for p in parts]
        partials = []
        for p, d in zip(parts, diff_paths):
            diff = name_diff(pl.read_parquet(p, columns=["ref_id", "mention_id", "num_tset"]), refs, names)
            diff.write_parquet(d)
            partials.append(token_counts(diff))
        del names
        stats = token_stats(partials)
        stats.write_parquet(work / "token_stats.parquet")
        for d, sp in zip(diff_paths, sib_paths):
            sibling_features(pl.read_parquet(d), stats).write_parquet(sp)
        log(f"built {len(sib_paths)} sibling-feature files from {stats.height} token statistics")
    lg_paths = [p.with_name(p.name.replace("feats_", "lg_")) for p in parts]
    if not all(p.exists() for p in lg_paths):
        cols = ["entity_id", "business_name", "addr_n"]
        rn, mn = pl.read_parquet(work / "refs.parquet", columns=cols), pl.read_parquet(work / "mentions.parquet", columns=cols)
        for p, lp in zip(parts, lg_paths):
            legal_features(pl.read_parquet(p, columns=["ref_id", "mention_id"]), rn, mn).write_parquet(lp)
        del rn, mn
        log(f"built {len(lg_paths)} legal-form feature files")
    hn_paths = [p.with_name(p.name.replace("feats_", "hn_")) for p in parts]
    hc_paths = [p.with_name(p.name.replace("feats_", "hc_")) for p in parts]
    if not all(p.exists() for p in hn_paths + hc_paths):
        # House-number pair features per chunk; ghost-cluster counts need every chunk at once.
        cols = ["entity_id", "business_name", "addr_n"]
        rk = record_keys(pl.read_parquet(work / "refs.parquet", columns=cols), "ref_id", "r")
        mk = record_keys(pl.read_parquet(work / "mentions.parquet", columns=cols), "mention_id", "m")
        key_paths = [p.with_name(p.name.replace("feats_", "hk_")) for p in parts]
        for i, (p, hp, kp) in enumerate(zip(parts, hn_paths, key_paths)):
            k = keyed_pairs(pl.read_parquet(p, columns=["ref_id", "mention_id"]), rk, mk)
            pair_number_features(k).write_parquet(hp)
            k.select("ref_id", "mention_id", "rh", "mh", "rlf", "mlf", "mx").write_parquet(kp)
        del rk, mk, k
        # Reference-level counts need all chunks, but only one Reference hash bucket at a time.
        n_buckets = max(1, len(parts) // 3)
        bucket_path = lambda i, b: key_paths[i].with_name(f"hb_{i:03d}_{b:03d}.parquet")
        for b in range(n_buckets):
            rows = pl.concat([pl.scan_parquet(kp).filter(pl.col("ref_id").hash(seed=5) % n_buckets == b)
                              .with_columns(pl.lit(i, dtype=pl.Int16).alias("_part")) for i, kp in enumerate(key_paths)]).collect()
            ctx = cluster_features(rows)
            del rows
            for i in range(len(parts)):
                ctx.filter(pl.col("_part") == i).drop("_part").write_parquet(bucket_path(i, b))
            del ctx
        for i, (c, kp) in enumerate(zip(hc_paths, key_paths)):
            files = [bucket_path(i, b) for b in range(n_buckets)]
            pl.concat([pl.read_parquet(f) for f in files]).write_parquet(c)
            for f in files + [kp]:
                f.unlink()
        log(f"built {len(hn_paths)} house-number and ghost-cluster feature files")
    nk_paths = [p.with_name(p.name.replace("feats_", "nk_")) for p in parts]
    if not all(p.exists() for p in nk_paths):
        # Name uniqueness: a Mention with no address is only safe to match when few Reference Records share its name.
        r = with_keys(pl.read_parquet(work / "refs.parquet", columns=["entity_id", "business_name", "country", "name_n"]))
        key_n = r.group_by("country", "nkey").len().rename({"len": "n"})
        r = r.with_columns(pl.len().over("country", "nkey").alias("nk_ref_key_n"), pl.len().over("country", "name_n").alias("nk_ref_name_n"))
        r = r.select(pl.col("entity_id").alias("ref_id"), pl.col("nkey").alias("rkey"), "nk_ref_key_n", "nk_ref_name_n")
        m = with_keys(pl.read_parquet(work / "mentions.parquet", columns=["entity_id", "business_name", "country"]))
        m = (m.join(key_n, on=["country", "nkey"], how="left")
             .select(pl.col("entity_id").alias("mention_id"), pl.col("nkey").alias("mkey"), pl.col("n").fill_null(0).alias("nk_men_key_n")))
        for p, np_ in zip(parts, nk_paths):
            (pl.read_parquet(p, columns=["ref_id", "mention_id"]).join(r, on="ref_id", how="left").join(m, on="mention_id", how="left")
             .select("ref_id", "mention_id", pl.col("nk_ref_key_n").cast(pl.Int32), pl.col("nk_ref_name_n").cast(pl.Int32),
                     pl.col("nk_men_key_n").cast(pl.Int32), (pl.col("rkey") == pl.col("mkey")).cast(pl.Int8).alias("nk_key_eq"))
             .write_parquet(np_))
        del r, m, key_n
        log(f"built {len(nk_paths)} name-uniqueness feature files")
    sides = [[c, sp, lp, hp, hc, nk] for c, sp, lp, hp, hc, nk in zip(ctx_paths, sib_paths, lg_paths, hn_paths, hc_paths, nk_paths)]
    if te is not None:
        te_paths = [p.with_name(p.name.replace("feats_", "te_")) for p in parts]
        for p, tp in zip(parts, te_paths):
            if not tp.exists():
                target_features(pl.read_parquet(p.with_name(p.name.replace("feats_", "diff_"))), te).write_parquet(tp)
        sides = [s + [tp] for s, tp in zip(sides, te_paths)]
    return refs, parts, sides


def block_chunks(data_dir, work, chunk_size=400_000, k_scale=1.0):
    """Cache each Mention chunk's text Blocking output (blk_NNN.parquet) so it can run before embeddings exist."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    refs, mens = load_records(data_dir, work)
    blocker = None
    for i, m in enumerate(mention_chunks(mens, chunk_size)):
        blk = work / f"blk_{i:03d}.parquet"
        if blk.exists():
            continue
        blocker = blocker or Blocker(refs, k_scale=k_scale, log=log)
        _cached(blk, lambda: blocker.query(m))


def load_part(path, side_paths):
    """A chunk's pair features joined with its side tables (Reference-side context, sibling features)."""
    df = pl.read_parquet(path)
    for side in side_paths:
        df = df.join(pl.read_parquet(side), on=["ref_id", "mention_id"], how="left")
    return df


def label(df, truth):
    m2r = pl.DataFrame([(m, r) for r, ms in truth.items() for m in ms], schema=["mention_id", "true_ref"], orient="row")
    return df.join(m2r, on="mention_id", how="left").with_columns(
        (pl.col("true_ref") == pl.col("ref_id")).fill_null(False).cast(pl.Int8).alias("y")).drop("true_ref")


def feature_cols(df):
    return [c for c in df.columns if c not in ("ref_id", "mention_id", "y", "w", "score", "p0")]


def fit_lgb(df, cols, rounds, name, work):
    hold = (df["ref_id"].hash(seed=3) % 10 == 0).to_numpy()
    X, y = df.select(cols).to_numpy().astype(np.float32), df["y"].to_numpy()
    w = df["w"].to_numpy() if "w" in df.columns else np.ones(len(y))  # optional prior-shift sample weights
    log(f"{name}: rows={len(y)} pos={y.sum()} features={len(cols)} weighted={'w' in df.columns}")
    dtr = lgb.Dataset(X[~hold], y[~hold], weight=w[~hold], feature_name=cols, free_raw_data=True)
    dva = lgb.Dataset(X[hold], y[hold], weight=w[hold], reference=dtr)
    booster = lgb.train(PARAMS, dtr, rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    booster.save_model(str(Path(work) / f"{name}.txt"))
    imp = sorted(zip(booster.feature_importance("gain"), cols), reverse=True)[:15]
    log(f"{name} best_iter={booster.best_iteration} top features:", [c for _, c in imp])
    return booster


def train(data_dir, work, rounds=2000, k_scale=1.0, top_n=6, chunk_size=400_000, emb=None):
    """Fit the pruner and the pair model on a labelled data dir."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    truth = read_matches(_find_truth(data_dir))
    n_true = sum(len(v) for v in truth.values())
    refs, mens = load_records(data_dir, work)
    cands = _cached(work / "cands.parquet", lambda: blocked(Blocker(refs, k_scale=k_scale, log=log), mens, emb))
    if (work / "pruner.txt").exists():
        pruner = lgb.Booster(model_file=str(work / "pruner.txt"))
    else:
        lab = label(cands, truth)
        log(f"train blocking recall: {lab['y'].sum() / n_true:.4f}")
        pruner = fit_lgb(lab, feature_cols(cands), min(rounds, 400), "pruner", work)
        del lab
    _, parts, _ = feature_parts(data_dir, work, pruner, top_n, k_scale, chunk_size, raw_cands=cands)
    del cands
    te = _cached(work / "token_te.parquet", lambda: token_target_rates(
        [pl.read_parquet(p.with_name(p.name.replace("feats_", "diff_"))) for p in parts], lambda d: label(d, truth)))
    _, parts, sides = feature_parts(data_dir, work, pruner, top_n, k_scale, chunk_size, te=te)
    df = label(pl.concat([load_part(p, c) for p, c in zip(parts, sides)]), truth)
    log(f"train recall after pruning: {df['y'].sum() / n_true:.4f}")
    return pruner, fit_lgb(df, feature_cols(df), rounds, "model", work)


def score(models, data_dir, work, k_scale=1.0, top_n=6, chunk_size=400_000, emb=None, te=None):
    pruner, booster = models
    refs, parts, ctx_paths = feature_parts(data_dir, work, pruner, top_n, k_scale, chunk_size, emb=emb, te=te)
    cols = booster.feature_name()
    scored = []
    for p, c in zip(parts, ctx_paths):
        df = load_part(p, c)
        s = booster.predict(df.select(cols).to_numpy().astype(np.float32), num_threads=0)
        scored.append(df.select("ref_id", "mention_id").with_columns(pl.Series("score", s.astype(np.float32))))
    return refs, pl.concat(scored)


GRID = np.concatenate([np.arange(0.2, 0.95, 0.025), np.arange(0.95, 0.996, 0.005)])


def tune_threshold(scored, reference_ids, truth, grid=GRID):
    results = [(evaluate(resolve(scored, reference_ids, t), truth), t) for t in grid]
    log("threshold scan:", [(round(t, 3), round(s, 4)) for s, t in results])
    return max(results)[1], max(results)[0]


def write_outputs(matches, scored, reference_ids, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cands = scored.group_by("ref_id").agg("mention_id")
    cand_map = dict(zip(cands["ref_id"].to_list(), cands["mention_id"].to_list()))
    with open(out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for r in reference_ids:
            f.write(f"{r}\t{','.join(matches.get(r, []))}\n")
    with open(out / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for r in reference_ids:
            f.write(f"{r}\t{','.join(cand_map.get(r, []))}\n")


def scored_for(data_dir, work, name, models, stack, k_scale, top_n, chunk_size, emb_dir):
    """Final Candidate scores for one data dir, cached as <work>/<name>/scored.parquet."""
    path = work / name / "scored.parquet"
    if path.exists():
        return pl.read_parquet(work / name / "refs.parquet"), pl.read_parquet(path)
    te_path = work / "train" / "token_te.parquet"
    te = pl.read_parquet(te_path) if te_path.exists() else None
    if stack:  # stage-2 scores only; the stage-1 pass would just hold memory
        refs, _, _ = feature_parts(data_dir, work / name, models[0], top_n, k_scale, chunk_size,
                                   emb=load_emb(emb_dir, name), te=te)
        scored = score_stack(work / "train", work / name, log)
    else:
        refs, scored = score(models, data_dir, work / name, k_scale, top_n, chunk_size, load_emb(emb_dir, name), te)
    scored.write_parquet(path)
    return refs, scored


def nostate_thresholds(refs, threshold):
    """Threshold override for Reference Records of countries with no state lexicon (e.g. France)."""
    if threshold is None:
        return None
    ids = refs.filter(~pl.col("country").is_in(list(STATE_CODES)))["entity_id"].to_list()
    log(f"no-state threshold {threshold} for {len(ids)} Reference Records")
    return dict.fromkeys(ids, threshold)


def run_pipeline(train_dir, test_dir, out_dir, work_dir, val_dir=None, threshold=None, k_scale=1.0, rounds=2000,
                 top_n=6, chunk_size=400_000, emb_dir=None, stack=False, threshold_nostate=None):
    """Train on train_dir, tune the threshold on val_dir (if given), predict test_dir into out_dir."""
    work = Path(work_dir)
    if (work / "train" / "model.txt").exists():
        log("reuse models")
        models = tuple(lgb.Booster(model_file=str(work / "train" / f"{n}.txt")) for n in ("pruner", "model"))
    else:
        models = train(train_dir, work / "train", rounds, k_scale, top_n, chunk_size, load_emb(emb_dir, "trn"))
    if stack and not (work / "train" / "stack.txt").exists():
        truth = read_matches(_find_truth(train_dir))
        train_stack(work / "train", lambda df: label(df, truth), fit_lgb, rounds, log)
    info = {"stack": stack}
    if val_dir is not None:
        refs, scored = scored_for(val_dir, work, "val", models, stack, k_scale, top_n, chunk_size, emb_dir)
        truth = read_matches(_find_truth(val_dir))
        threshold, f = tune_threshold(scored, refs["entity_id"].to_list(), truth)
        info.update(val_f05=f)
        log(f"VAL F0.5={f:.5f} at threshold {threshold:.3f}")
    threshold = 0.5 if threshold is None else threshold
    if test_dir is not None:
        refs, scored = scored_for(test_dir, work, "test", models, stack, k_scale, top_n, chunk_size, emb_dir)
        ids = refs["entity_id"].to_list()
        write_outputs(resolve(scored, ids, threshold, nostate_thresholds(refs, threshold_nostate)), scored, ids, out_dir)
    info["threshold"] = float(threshold)
    info["threshold_nostate"] = threshold_nostate
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "run_info.json").write_text(json.dumps(info, indent=2))
    log("done", info)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test")
    ap.add_argument("--val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--k-scale", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--top-n", type=int, default=6)
    ap.add_argument("--chunk-size", type=int, default=400_000)
    ap.add_argument("--emb-dir", help="folder with emb_trn/emb_val/emb_test.parquet from embed_knn.py")
    ap.add_argument("--stack", action="store_true", help="add the stage-2 stacker")
    ap.add_argument("--threshold-nostate", type=float,
                    help="threshold for countries with no state lexicon (tune it on a state-hidden validation copy)")
    a = ap.parse_args()
    run_pipeline(a.train, a.test, a.out, a.work, a.val, a.threshold, a.k_scale, a.rounds, a.top_n, a.chunk_size,
                 a.emb_dir, a.stack, a.threshold_nostate)


if __name__ == "__main__":
    main()
