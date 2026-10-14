"""Pair features for Candidates: string similarity, number agreement and Candidate-set context.

No feature uses the country label, so an unseen country (France) gets no special bucket.
"""
import re

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

VIEW_COLS = ["name_n", "name_sq", "skel", "addr_n", "business_name"]
_DIGITS = re.compile(r"\d+")


def _pair_sim(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw) / 100.0


def _numbers(col):
    return pl.col(col).str.extract_all(r"\d+").list.unique()


def pair_features(cands, references, mentions):
    """Join text views onto Candidates and add similarity features (one row per Candidate)."""
    r = references.select(pl.col("entity_id").alias("ref_id"), *[pl.col(c).alias(f"r_{c}") for c in VIEW_COLS])
    m = mentions.select(pl.col("entity_id").alias("mention_id"), *[pl.col(c).alias(f"m_{c}") for c in VIEW_COLS])
    df = cands.join(r, on="ref_id", how="left").join(m, on="mention_id", how="left")

    rn, mn = df["r_name_n"].to_list(), df["m_name_n"].to_list()
    rs, ms = df["r_name_sq"].to_list(), df["m_name_sq"].to_list()
    rk, mk = df["r_skel"].to_list(), df["m_skel"].to_list()
    ra, ma = df["r_addr_n"].to_list(), df["m_addr_n"].to_list()
    feats = {
        "name_ratio": _pair_sim(rn, mn, fuzz.ratio),
        "name_tset": _pair_sim(rn, mn, fuzz.token_set_ratio),
        "name_tsort": _pair_sim(rn, mn, fuzz.token_sort_ratio),
        "name_partial": _pair_sim(rn, mn, fuzz.partial_ratio),
        "sq_jw": cpdist(rs, ms, scorer=distance.JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32),
        "sq_partial": _pair_sim(rs, ms, fuzz.partial_ratio),
        "skel_ratio": _pair_sim(rk, mk, fuzz.ratio),
        "skel_tset": _pair_sim(rk, mk, fuzz.token_set_ratio),
        "addr_ratio": _pair_sim(ra, ma, fuzz.ratio),
        "addr_tset": _pair_sim(ra, ma, fuzz.token_set_ratio),
        "addr_tsort": _pair_sim(ra, ma, fuzz.token_sort_ratio),
        "addr_partial": _pair_sim(ra, ma, fuzz.partial_ratio),
    }
    df = df.with_columns(**{k: pl.Series(v) for k, v in feats.items()})

    # House numbers are often corrupted by a digit or two, so compare them as strings, and compare
    # the address with the numbers removed.
    nums = df.select(
        pl.col("r_addr_n").str.extract_all(r"\d+").list.join(" ").alias("rn"),
        pl.col("m_addr_n").str.extract_all(r"\d+").list.join(" ").alias("mn"),
        pl.col("r_addr_n").str.replace_all(r"\d+", "").str.replace_all(r"\s+", " ").str.strip_chars().alias("ra"),
        pl.col("m_addr_n").str.replace_all(r"\d+", "").str.replace_all(r"\s+", " ").str.strip_chars().alias("ma"),
    )
    rn, mn, rw, mw = (nums[c].to_list() for c in ("rn", "mn", "ra", "ma"))
    df = df.with_columns(
        num_tset=pl.Series(_pair_sim(rn, mn, fuzz.token_set_ratio)),
        num_partial=pl.Series(_pair_sim(rn, mn, fuzz.partial_ratio)),
        num_ratio=pl.Series(_pair_sim(rn, mn, fuzz.ratio)),
        addr_words_tset=pl.Series(_pair_sim(rw, mw, fuzz.token_set_ratio)),
        addr_words_ratio=pl.Series(_pair_sim(rw, mw, fuzz.ratio)),
    )

    df = df.with_columns(
        _numbers("r_addr_n").alias("_rnum"),
        _numbers("m_addr_n").alias("_mnum"),
        _numbers("r_name_n").alias("_rnn"),
        _numbers("m_name_n").alias("_mnn"),
        (pl.col("r_name_n").str.split(" ").list.set_intersection(pl.col("m_name_n").str.split(" ")).list.len()).alias("name_tok_common"),
    ).with_columns(
        pl.col("_rnum").list.set_intersection("_mnum").list.len().alias("num_common"),
        pl.col("_rnum").list.len().alias("r_num_n"),
        pl.col("_mnum").list.len().alias("m_num_n"),
        (pl.col("_rnum").list.first() == pl.col("_mnum").list.first()).cast(pl.Int8).fill_null(-1).alias("num_first_eq"),
        pl.col("_rnn").list.set_intersection("_mnn").list.len().alias("name_num_common"),
        (pl.col("_rnn").list.len() + pl.col("_mnn").list.len()).alias("name_num_n"),
        pl.col("r_name_n").str.len_chars().alias("r_name_len"),
        pl.col("m_name_n").str.len_chars().alias("m_name_len"),
        pl.col("r_name_n").str.count_matches(" ").alias("r_name_ntok"),
        pl.col("m_name_n").str.count_matches(" ").alias("m_name_ntok"),
        pl.col("m_addr_n").str.len_chars().alias("m_addr_len"),
        pl.col("r_addr_n").str.len_chars().alias("r_addr_len"),
        pl.col("mention_id").str.starts_with("S3").cast(pl.Int8).alias("m_is_s3"),
        pl.col("m_business_name").str.contains(r"[^\x00-ɏ]").cast(pl.Int8).alias("m_script"),
        pl.col("m_business_name").str.contains(r"\.(com|in|net|org|fr)$|^@|^www").cast(pl.Int8).alias("m_web"),
        pl.col("m_business_name").str.to_lowercase().str.contains("formerly").cast(pl.Int8).alias("m_formerly"),
    ).with_columns(
        (pl.col("num_common") / pl.max_horizontal(pl.min_horizontal("r_num_n", "m_num_n"), 1)).alias("num_frac"),
    )
    return df.drop([c for c in df.columns if c.startswith(("r_", "m_", "_")) and c not in KEEP_PREFIXED])


KEEP_PREFIXED = {"r_num_n", "m_num_n", "r_name_len", "m_name_len", "r_name_ntok", "m_name_ntok", "m_addr_len",
                 "r_addr_len", "m_is_s3", "m_script", "m_web", "m_formerly"}


def context_features(df, score_cols, side="mention_id"):
    """Candidate-set context on one side: rank, gap to best and Candidate count of each score within the
    Mention's Candidates (side="mention_id") or the Reference Record's Candidates (side="ref_id")."""
    tag = "m" if side == "mention_id" else "r"
    exprs = []
    for c in score_cols:
        exprs += [
            pl.col(c).rank("ordinal", descending=True).over(side).cast(pl.Int16).alias(f"{c}_rank_{tag}"),
            (pl.col(c) - pl.col(c).max().over(side)).alias(f"{c}_gap_{tag}"),
        ]
    exprs.append(pl.len().over(side).cast(pl.Int16).alias(f"n_cands_{tag}"))
    df = df.with_columns(exprs)
    if side == "mention_id":  # margin of the best score over the second best (large = unambiguous)
        main = score_cols[0]
        second = pl.when(pl.col(f"{main}_rank_m") == 2).then(pl.col(main)).max().over(side).fill_null(0.0)
        df = df.with_columns((pl.col(main).max().over(side) - second).alias(f"{main}_margin_m"))
    return df
