import sys
import polars as pl
sys.path.insert(0, "src")
from entity_resolution.normalize import normalize_address, US_STATES, INDIA_STATES
from entity_resolution.evaluate import read_matches
d = "../../data/split/val"
R = dict(separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
codes = {"US": set(US_STATES.values()) , "India": set(INDIA_STATES.values())}
def state(addr, country):
    toks = normalize_address(addr).split()
    c = codes.get(country, set())
    found = [t for t in toks if t in c]
    return found[-1] if found else ""
def load(f):
    x = pl.read_csv(f, **R)
    return {e: state(a, c) for e, a, c in zip(x["entity_id"], x["business_address"], x["country"])}
rs = load(f"{d}/source1.tsv"); ms = {**load(f"{d}/source2.tsv"), **load(f"{d}/source3.tsv")}
t = read_matches(f"{d}/ground_truth.tsv")
import collections
c = collections.Counter()
for r, m_ in t.items():
    for m in m_:
        c["ref_nostate" if not rs[r] else "m_nostate" if not ms[m] else "same" if rs[r] == ms[m] else "diff"] += 1
print(c, {k: v / sum(c.values()) for k, v in c.items()})
print("mentions with no state", sum(1 for v in ms.values() if not v) / len(ms))
