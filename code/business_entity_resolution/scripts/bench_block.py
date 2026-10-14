import sys, time
import polars as pl
sys.path.insert(0, "src")
from ber.block import block, add_text_views
from ber.evaluate import read_matches
def main():
    d = sys.argv[1]; k = float(sys.argv[2])
    R = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
    t = time.time()
    refs = pl.read_csv(f"{d}/source1.tsv", **R)
    mens = pl.concat([pl.read_csv(f"{d}/source2.tsv", **R), pl.read_csv(f"{d}/source3.tsv", **R)])
    refs, mens = add_text_views(refs), add_text_views(mens)
    print("normalize s", round(time.time() - t), refs.height, mens.height); t = time.time()
    c = block(refs, mens, k_scale=k)
    print("block s", round(time.time() - t), "pairs", c.height, "per ref", c.height / refs.height)
    truth = read_matches(f"{d}/ground_truth.tsv")
    tp = pl.DataFrame([(r, m) for r, ms in truth.items() for m in ms], schema=["ref_id", "mention_id"], orient="row")
    j = tp.join(c, on=["ref_id", "mention_id"], how="left")
    print("recall union", j["sim_full"].is_not_null().mean())
    for s in [x for x in c.columns if x.startswith("sim_")]:
        print(" recall", s, (j[s].fill_null(0) > 0).mean())


if __name__ == "__main__":
    main()
