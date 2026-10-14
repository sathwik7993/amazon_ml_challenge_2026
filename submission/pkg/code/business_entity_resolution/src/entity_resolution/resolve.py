"""Turn scored Candidates into Matches: one Reference Record per Mention, above a threshold."""
import polars as pl


def resolve(scored, reference_ids, threshold, ref_thresholds=None):
    """{ref_id: [mention_id, ...]} for every Reference Record; each Mention kept only for its best Reference.

    `ref_thresholds` optionally overrides `threshold` for some Reference Records ({ref_id: threshold}).
    """
    best = scored.sort("score", descending=True).unique("mention_id", keep="first")
    if ref_thresholds:
        t = pl.DataFrame({"ref_id": list(ref_thresholds), "_thr": list(ref_thresholds.values())},
                         schema={"ref_id": pl.String, "_thr": pl.Float64})
        best = best.join(t, on="ref_id", how="left").with_columns(pl.col("_thr").fill_null(threshold))
    else:
        best = best.with_columns(pl.lit(threshold, dtype=pl.Float64).alias("_thr"))
    best = (
        best.filter(pl.col("score") >= pl.col("_thr"))
        .sort(["ref_id", "score"], descending=[False, True])
        .group_by("ref_id", maintain_order=True)
        .agg("mention_id")
    )
    out = {r: [] for r in reference_ids}
    for ref, mentions in best.iter_rows():
        if ref in out:
            out[ref] = list(mentions)
    return out
