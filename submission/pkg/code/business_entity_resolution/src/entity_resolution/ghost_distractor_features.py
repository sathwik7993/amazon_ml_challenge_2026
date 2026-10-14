"""Ghost-Distractor features.

Hard Distractors in this data are "ghost" businesses next door to a Reference Record: the same core
name, often a different legal form or one extra sibling word, and a house number shifted UP by a small
amount (520 -> 529, 4440 -> 4443, 8395 -> 8404). A ghost often has two or three copies of its own that
share the shifted number. True noisy copies instead keep the number or lose/alter single digits
(8395 -> 839, 2253 -> 253, 547 -> 147).

Pair features describe the signed difference and digit-edit relation of the two house numbers.
Cluster features count, among all Candidates of the same Reference Record (across every Mention
chunk), how many share this Mention's number, legal form, or the Reference's own number.
"""
import polars as pl

from entity_resolution.normalize import HOUSE_NUMBER_RE, UNIT_NUMBER_RE
from entity_resolution.name_variant_features import _legal

MAX_DIGITS = 10


def record_keys(records, id_name, prefix):
    """(id, <prefix>hs house-number string, <prefix>lf legal-form hash) per Record."""
    street = pl.col("addr_n").str.replace_all(UNIT_NUMBER_RE, " ")
    return records.select(pl.col("entity_id").alias(id_name), street.str.extract(HOUSE_NUMBER_RE, 1).alias(f"{prefix}hs"),
                          street.str.extract_all(HOUSE_NUMBER_RE).alias(f"{prefix}nums"),
                          _legal("business_name").hash().alias(f"{prefix}lf"))


def keyed_pairs(pairs, ref_keys, men_keys):
    """Candidate pairs with both house numbers, and the address numbers the Mention adds or drops.

    Indian addresses carry compound numbers (65/2688, 2-2-18/30/A); a Ghost changes one part of them
    (65/2701, 2-2-18/31/A) or adds a number where the Reference has none. `mx` keys the added numbers
    so that a Ghost's own copies, which share them, can be counted per Reference Record.
    """
    extra = pl.col("mnums").list.set_difference("rnums")
    return pairs.join(ref_keys, on="ref_id", how="left").join(men_keys, on="mention_id", how="left").with_columns(
        pl.col("rhs").cast(pl.Int64, strict=False).alias("rh"), pl.col("mhs").cast(pl.Int64, strict=False).alias("mh"),
        extra.alias("extra_nums"), pl.col("rnums").list.set_difference("mnums").alias("missing_nums"),
    ).with_columns(
        pl.when(pl.col("extra_nums").list.len() > 0)
        .then(pl.col("extra_nums").list.sort().list.join(" ").hash()).alias("mx"),
    )


def pair_number_features(df):
    """Signed difference and digit-edit relation of the Reference and Mention house numbers."""
    rs, ms = pl.col("rhs"), pl.col("mhs")
    lr, lm = rs.str.len_chars().cast(pl.Int32), ms.str.len_chars().cast(pl.Int32)
    same = lr == lm
    diff_k = [(rs.str.slice(k, 1) != ms.str.slice(k, 1)) & (pl.lit(k) < lr) for k in range(MAX_DIGITS)]
    sd = pl.col("mh") - pl.col("rh")
    eq_ref = (pl.col("mh") == pl.col("rh")).fill_null(False)
    return df.select(
        "ref_id", "mention_id",
        (pl.when(rs.is_null()).then(1).otherwise(0) + pl.when(ms.is_null()).then(2).otherwise(0)).cast(pl.Int8).alias("hn_missing"),
        sd.clip(-1000, 1000).cast(pl.Float32).alias("hn_sd"),
        (sd.abs() + 1).cast(pl.Float64).log10().cast(pl.Float32).alias("hn_sd_log"),
        (sd.is_between(1, 60)).cast(pl.Int8).alias("hn_up_small"),
        (lm - lr).cast(pl.Int8).alias("hn_len_diff"),
        (rs.str.starts_with(ms) & (lm < lr)).cast(pl.Int8).alias("hn_trunc_right"),
        (rs.str.ends_with(ms) & (lm < lr)).cast(pl.Int8).alias("hn_trunc_left"),
        pl.when(same).then(pl.sum_horizontal([d.cast(pl.Int8) for d in diff_k])).cast(pl.Int8).alias("hn_ndiff"),
        pl.when(same).then(pl.max_horizontal([pl.when(d).then(lr - 1 - k) for k, d in enumerate(diff_k)]))
        .cast(pl.Int8).alias("hn_diff_pos"),  # position (from the right) of the highest changed digit
        (pl.col("mlf") == pl.col("rlf")).cast(pl.Int8).alias("hn_same_lf"),
        pl.col("extra_nums").list.len().cast(pl.Int8).alias("hn_n_extra"),
        pl.col("missing_nums").list.len().cast(pl.Int8).alias("hn_n_missing"),
        pl.col("mnums").list.set_intersection("rnums").list.len().cast(pl.Int8).alias("hn_n_common"),
        (pl.col("extra_nums").list.first().cast(pl.Int64, strict=False)
         - pl.col("missing_nums").list.first().cast(pl.Int64, strict=False)).clip(-1000, 1000).cast(pl.Float32).alias("hn_sd_changed"),
        # other Candidate References of this Mention whose number equals the Mention's (a chunk holds all of them)
        (eq_ref.sum().over("mention_id") - eq_ref.cast(pl.UInt32)).cast(pl.Int16).alias("hc_m_other_eq"),
    )


def cluster_features(df):
    """Per-Reference counts over all Candidates of each Reference (df may be one Reference hash bucket); df has ref_id, mention_id, rh, mh, rlf, mlf, mx."""
    has = pl.col("mh").is_not_null()
    eq_ref = (pl.col("mh") == pl.col("rh")).fill_null(False)
    up = (pl.col("mh") - pl.col("rh")).is_between(1, 60).fill_null(False)
    return df.with_columns(
        pl.len().over("ref_id").cast(pl.Int16).alias("hc_n_cands"),
        pl.when(has).then(pl.len().over(["ref_id", "mh"]) - 1).cast(pl.Int16).alias("hc_n_same_num"),
        pl.when(has).then(pl.len().over(["ref_id", "mh", "mlf"]) - 1).cast(pl.Int16).alias("hc_n_same_num_lf"),
        eq_ref.sum().over("ref_id").cast(pl.Int16).alias("hc_n_ref_num"),
        up.sum().over("ref_id").cast(pl.Int16).alias("hc_n_up_small"),
        (pl.col("mlf") == pl.col("rlf")).sum().over("ref_id").cast(pl.Int16).alias("hc_n_ref_lf"),
        pl.when(pl.col("mx").is_not_null()).then(pl.len().over(["ref_id", "mx"]) - 1).cast(pl.Int16).alias("hc_n_same_extra"),
        pl.col("mx").is_null().sum().over("ref_id").cast(pl.Int16).alias("hc_n_no_extra"),
    ).drop("rh", "mh", "rlf", "mlf", "mx")
