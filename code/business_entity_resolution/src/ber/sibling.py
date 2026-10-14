"""Sibling-word features.

The data contains "sibling" Distractors: a Reference Record's name plus one extra word from a fixed
vocabulary ("Group", "Holdings", "South", "Groupe", ...) at a nearby but different house number. Noise
copies of the same business also add words ("Services", "Dr", "Shri") but usually keep the number.

For every Candidate we find the name tokens the Mention adds or drops. For every such token we measure,
over the whole dataset and without labels, how often it co-occurs with a changed house number and how
common it is. Sibling words co-occur with a changed number ~99% of the time; noise words ~50%. The
statistic is computed per dataset, so an unseen language (France) gets its own sibling vocabulary.
"""
import math

import polars as pl

MIN_COUNT = 20  # below this a token's statistic is too noisy; leave it null


def name_diff(pairs, refs, mens):
    """(ref_id, mention_id, extra, missing, numdiff) for Candidate pairs with `num_tset` already computed."""
    r = refs.select(pl.col("entity_id").alias("ref_id"), pl.col("name_n").str.split(" ").alias("rt"))
    m = mens.select(pl.col("entity_id").alias("mention_id"), pl.col("name_n").str.split(" ").alias("mt"))
    df = pairs.select("ref_id", "mention_id", "num_tset").join(r, on="ref_id").join(m, on="mention_id")
    return df.select(
        "ref_id", "mention_id",
        pl.col("mt").list.set_difference("rt").list.eval(pl.element().filter(pl.element() != "")).alias("extra"),
        pl.col("rt").list.set_difference("mt").list.eval(pl.element().filter(pl.element() != "")).alias("missing"),
        (pl.col("num_tset") < 0.99).cast(pl.Int8).alias("numdiff"),
    )


def sibling_shaped(diff):
    """Pairs where one side's name is the other's plus exactly one word (where siblings and noise differ)."""
    e, m = pl.col("extra").list.len(), pl.col("missing").list.len()
    return diff.filter(((e == 1) & (m == 0)) | ((e == 0) & (m == 1)))


def token_counts(diff):
    """Partial per-token sums for one chunk (summed across chunks by `token_stats`)."""
    out, diff = [], sibling_shaped(diff)
    for kind in ("extra", "missing"):
        out.append(diff.select(kind, "numdiff").explode(kind).drop_nulls(kind)
                   .group_by(kind).agg(pl.len().alias("n"), pl.col("numdiff").sum().alias("nd"))
                   .rename({kind: "token"}).with_columns(pl.lit(kind).alias("kind")))
    return pl.concat(out)


def token_stats(partials):
    s = pl.concat(partials).group_by("kind", "token").agg(pl.col("n").sum(), pl.col("nd").sum())
    return s.filter(pl.col("n") >= MIN_COUNT).with_columns(
        (pl.col("nd") / pl.col("n")).cast(pl.Float32).alias("nd_rate"),
        pl.col("n").cast(pl.Float64).log10().cast(pl.Float32).alias("log_n"),
    ).select("kind", "token", "nd_rate", "log_n")


def token_target_rates(diffs, label_fn, prior_weight=20.0):
    """Smoothed true-Match rate per added/dropped token on sibling-shaped labelled pairs (training only)."""
    parts = [label_fn(sibling_shaped(d)).select("extra", "missing", "y") for d in diffs]
    df = pl.concat(parts)
    prior = df["y"].mean()
    out = []
    for kind in ("extra", "missing"):
        t = (df.select(kind, "y").explode(kind).drop_nulls(kind)
             .group_by(kind).agg(pl.len().alias("n"), pl.col("y").sum().alias("pos"))
             .filter(pl.col("n") >= MIN_COUNT)
             .select(pl.col(kind).alias("token"), pl.lit(kind).alias("kind"),
                     ((pl.col("pos") + prior_weight * prior) / (pl.col("n") + prior_weight)).cast(pl.Float32).alias("te")))
        out.append(t)
    return pl.concat(out)


def target_features(diff, te):
    """Per-pair min/max supervised token rate of added and dropped tokens (null for unseen tokens)."""
    feats = diff.select("ref_id", "mention_id")
    for kind in ("extra", "missing"):
        t = te.filter(pl.col("kind") == kind).drop("kind")
        agg = (diff.select("ref_id", "mention_id", kind).explode(kind).drop_nulls(kind)
               .join(t, left_on=kind, right_on="token", how="left")
               .group_by("ref_id", "mention_id")
               .agg(pl.col("te").min().alias(f"{kind}_te_min"), pl.col("te").max().alias(f"{kind}_te_max")))
        feats = feats.join(agg, on=["ref_id", "mention_id"], how="left")
    return feats


def sibling_features(diff, stats):
    """Per-pair aggregates of the token statistics of the added and dropped name tokens."""
    feats = diff.select("ref_id", "mention_id",
                        pl.col("extra").list.len().cast(pl.Int8).alias("n_extra_tok"),
                        pl.col("missing").list.len().cast(pl.Int8).alias("n_missing_tok"))
    for kind in ("extra", "missing"):
        st = stats.filter(pl.col("kind") == kind).drop("kind")
        agg = (diff.select("ref_id", "mention_id", kind).explode(kind).drop_nulls(kind)
               .join(st, left_on=kind, right_on="token", how="left")
               .group_by("ref_id", "mention_id")
               .agg(pl.col("nd_rate").max().alias(f"{kind}_nd_max"),
                    pl.col("nd_rate").min().alias(f"{kind}_nd_min"),
                    pl.col("log_n").max().alias(f"{kind}_logn_max"),
                    pl.col("nd_rate").is_null().sum().cast(pl.Int8).alias(f"{kind}_rare_n")))
        feats = feats.join(agg, on=["ref_id", "mention_id"], how="left")
    return feats


# Legal forms, canonicalized. Normalized names drop them (so noisy copies compare equal), but a changed
# legal form at a nearby number marks a different legal entity next door: in training such pairs are
# true Matches only 3-5% of the time, versus ~50% when the legal form is unchanged.
LEGAL_FORMS = {
    "inc": "inc", "incorporated": "inc", "llc": "llc", "corp": "corp", "corporation": "corp", "co": "co",
    "company": "co", "cie": "co", "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "llp": "llp",
    "lp": "lp", "pllc": "pllc", "pc": "pc", "plc": "plc", "sarl": "sarl", "sas": "sas", "sasu": "sasu",
    "eurl": "eurl", "sa": "sa", "sci": "sci", "snc": "snc", "gmbh": "gmbh",
}


def _legal(col):
    return (pl.col(col).str.to_lowercase().str.replace_all(r"\.", "").str.replace_all(r"[^a-z0-9]+", " ")
            .str.split(" ").list.eval(pl.element().replace_strict(LEGAL_FORMS, default=None).drop_nulls())
            .list.unique().list.sort().list.join(" "))


def legal_features(pairs, refs, mens):
    """Legal-form change and house-number distance for Candidate pairs."""
    r = refs.select(pl.col("entity_id").alias("ref_id"), _legal("business_name").alias("rl"),
                    pl.col("addr_n").str.extract(r"(\d+)").cast(pl.Int64, strict=False).alias("rn"))
    m = mens.select(pl.col("entity_id").alias("mention_id"), _legal("business_name").alias("ml"),
                    pl.col("addr_n").str.extract(r"(\d+)").cast(pl.Int64, strict=False).alias("mn"))
    df = pairs.select("ref_id", "mention_id").join(r, on="ref_id").join(m, on="mention_id")
    return df.select(
        "ref_id", "mention_id",
        pl.when((pl.col("rl") == "") & (pl.col("ml") == "")).then(0)
        .when(pl.col("rl") == pl.col("ml")).then(1)
        .when(pl.col("rl") == "").then(3)
        .when(pl.col("ml") == "").then(4)
        .otherwise(2).cast(pl.Int8).alias("legal_state"),
        ((pl.col("rn") - pl.col("mn")).abs() + 1).cast(pl.Float64).log10().cast(pl.Float32).alias("num_absdiff_log"),
        ((pl.col("rn") - pl.col("mn")).abs() / pl.max_horizontal("rn", "mn", pl.lit(1))).cast(pl.Float32).alias("num_reldiff"),
    )
