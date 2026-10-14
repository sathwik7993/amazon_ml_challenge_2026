"""Build the geographic validation split (see docs/adr/0001).

Writes <out>/fit and <out>/val, each with source1/2/3 TSVs in the test-file
format plus ground_truth.tsv, so the pipeline treats them exactly like test.
"""
import argparse
import random
from pathlib import Path

import polars as pl

INDIA_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Jammu and Kashmir", "Ladakh",
    "Chandigarh", "Puducherry", "Andaman and Nicobar Islands", "Lakshadweep",
    "Dadra and Nagar Haveli and Daman and Diu",
]
READ = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)


def read(p):
    return pl.read_csv(p, **READ).with_columns(pl.all().fill_null(""))


def last_token(col):
    return pl.col(col).str.split(", ").list.last().str.strip_chars().fill_null("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    d = Path(a.train_dir)
    s1 = read(d / "train_source1.tsv")
    s2 = read(d / "train_source2.tsv")
    s3 = read(d / "train_source3.tsv")
    gt = read(d / "train_ground_truth.tsv")

    # Region of a Reference Record: last address component equal to a US code or India state
    # (Source 1 is clean: US ends in a 2-letter code, India in the full state name).
    s1 = s1.with_columns(
        pl.when(pl.col("country") == "US")
        .then(pl.col("business_address").str.extract(r"(?:^|, )([A-Z]{2})(?:,|$)", 1))
        .otherwise(
            pl.col("business_address").str.split(", ").list.eval(
                pl.element().str.strip_chars().filter(pl.element().str.strip_chars().is_in(INDIA_STATES))
            ).list.last()
        )
        .fill_null("")
        .alias("region")
    )
    s1 = s1.with_columns((pl.col("country") + "|" + pl.col("region")).alias("region"))

    # Pick whole regions per country until ~frac of that country's Reference Records.
    rng = random.Random(a.seed)
    holdout = set()
    for country, grp in s1.group_by("country"):
        counts = grp["region"].value_counts().rows()
        rng.shuffle(counts)
        total, got = grp.height, 0
        for region, n in counts:
            if got >= a.frac * total:
                break
            if n > 0.3 * a.frac * total:  # avoid one giant region dominating
                continue
            holdout.add(region)
            got += n
    print("holdout regions:", sorted(holdout))

    pairs = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
    pairs = pairs.filter(pl.col("matched_entity_ids") != "").rename({"matched_entity_ids": "entity_id"})
    ref_region = s1.select(pl.col("entity_id").alias("source1_entity_id"), "region")
    mention_region = pairs.join(ref_region, on="source1_entity_id").select("entity_id", "region")

    # Distractors: learn "last address token -> region" from matched Mentions, majority vote.
    mentions = pl.concat([s2, s3])
    tok = mentions.select("entity_id", "country", last_token("business_address").alias("tok"))
    lex = (
        tok.join(mention_region, on="entity_id")
        .group_by("country", "tok", "region").len()
        .sort("len", descending=True)
        .unique(["country", "tok"], keep="first")
        .select("country", "tok", pl.col("region").alias("lex_region"))
    )
    tok = tok.join(mention_region, on="entity_id", how="left").join(lex, on=["country", "tok"], how="left")
    r = random.Random(a.seed + 1)
    unknown_draw = pl.Series([r.random() < a.frac for _ in range(tok.height)])
    tok = tok.with_columns(unknown_draw.alias("draw")).with_columns(
        pl.when(pl.col("region").is_not_null()).then(pl.col("region").is_in(list(holdout)))
        .when(pl.col("lex_region").is_not_null()).then(pl.col("lex_region").is_in(list(holdout)))
        .otherwise(pl.col("draw"))
        .alias("is_val")
    )
    val_mentions = set(tok.filter(pl.col("is_val"))["entity_id"].to_list())
    val_refs = set(s1.filter(pl.col("region").is_in(list(holdout)))["entity_id"].to_list())

    cols = ["entity_id", "business_name", "business_address", "country"]
    for part, keep_ref, keep_men in [("val", True, True), ("fit", False, False)]:
        out = Path(a.out) / part
        out.mkdir(parents=True, exist_ok=True)
        in_ref = pl.col("entity_id").is_in(list(val_refs))
        in_men = pl.col("entity_id").is_in(list(val_mentions))
        s1.filter(in_ref if keep_ref else ~in_ref).select(cols).write_csv(out / "source1.tsv", separator="\t", quote_style="never")
        s2.filter(in_men if keep_men else ~in_men).select(cols).write_csv(out / "source2.tsv", separator="\t", quote_style="never")
        s3.filter(in_men if keep_men else ~in_men).select(cols).write_csv(out / "source3.tsv", separator="\t", quote_style="never")
        g = gt.filter(pl.col("source1_entity_id").is_in(list(val_refs)) == keep_ref)
        g.write_csv(out / "ground_truth.tsv", separator="\t", quote_style="never")
        print(part, "refs", g.height)

    # Sanity: no true Match crosses the split.
    cross = pairs.filter(pl.col("source1_entity_id").is_in(list(val_refs)) != pl.col("entity_id").is_in(list(val_mentions)))
    print("true Matches crossing the split:", cross.height)


if __name__ == "__main__":
    main()
