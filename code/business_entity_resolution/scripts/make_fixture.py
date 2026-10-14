"""Cut a small real-data fixture from the validation split for the end-to-end test."""
from pathlib import Path
import polars as pl
R = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
src = Path("../../data/split/val")
s1 = pl.read_csv(src / "source1.tsv", **R)
men = pl.concat([pl.read_csv(src / "source2.tsv", **R), pl.read_csv(src / "source3.tsv", **R)])
gt = pl.read_csv(src / "ground_truth.tsv", **R).fill_null("")
matched = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")["matched_entity_ids"]
distractors = men.filter(~pl.col("entity_id").is_in(matched.implode()))
for part, seed, names in [("train", 1, ("source1", "source2", "source3", "ground_truth")),
                          ("test", 2, ("test_source1", "test_source2", "test_source3", None))]:
    g = gt.sample(400, seed=seed)
    ids = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")["matched_entity_ids"]
    m = pl.concat([men.filter(pl.col("entity_id").is_in(ids.implode())), distractors.sample(300, seed=seed)])
    out = Path("tests/data") / part
    out.mkdir(parents=True, exist_ok=True)
    w = lambda df, n: df.write_csv(out / f"{n}.tsv", separator="\t", quote_style="never")
    w(s1.filter(pl.col("entity_id").is_in(g["source1_entity_id"].implode())), names[0])
    w(m.filter(pl.col("entity_id").str.starts_with("S2")), names[1])
    w(m.filter(pl.col("entity_id").str.starts_with("S3")), names[2])
    if names[3]:
        w(g, names[3])
    else:
        w(g, "hidden_ground_truth")
print("ok")
