"""Blocking: pick Candidates for each Mention among the Reference Records of the same country.

Each source is a char n-gram TF-IDF view of the Records; for every Mention we keep the
top-k most similar Reference Records per source and union the results.
"""
import os
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from ber.normalize import address_state, name_key, name_skeleton, normalize_address, normalize_name

# source name -> (text column, analyzer, k, scope, max_df). Scope "state" searches the Reference Records of the
# Mention's state and skips Mentions with no known state; "nostate" searches the whole country but only for
# Mentions with no known state (e.g. every French Mention); "country" always searches the whole country.
# Country-wide char search is the expensive part, so it uses the strictest max_df.
SOURCES = {
    "full": ("full_n", "char_wb", 10, "state", 0.05),
    "fullx": ("full_n", "char_wb", 10, "nostate", 0.02),
    "name": ("name_n", "char_wb", 5, "state", 1.0),
    "squash": ("name_sq", "char", 5, "state", 1.0),
    "skel": ("skel", "char_wb", 5, "state", 1.0),
    "addr": ("addr_n", "char_wb", 5, "state", 0.05),
    "fullw": ("full_n", "word", 10, "state", 0.05),
}


def _views(rows):
    names, addrs, countries = rows
    addr_n = [normalize_address(a) for a in addrs]
    return ([normalize_name(n) for n in names], [name_skeleton(n) for n in names], addr_n,
            [address_state(a, c) for a, c in zip(addr_n, countries)])


def add_text_views(records, workers=None):
    """Add the normalized text columns every stage works on (in parallel for large inputs)."""
    cols = [records[c].fill_null("").to_list() for c in ("business_name", "business_address", "country")]
    n = records.height
    if n < 200_000:
        views = _views(cols)
    else:
        from concurrent.futures import ProcessPoolExecutor
        size = 100_000
        chunks = [tuple(c[i:i + size] for c in cols) for i in range(0, n, size)]
        with ProcessPoolExecutor(workers or min(6, os.cpu_count())) as ex:
            parts = list(ex.map(_views, chunks))
        views = tuple(sum((p[j] for p in parts), []) for j in range(4))
    name_n, skel, addr_n, state = views
    return records.with_columns(
        pl.Series("name_n", name_n, dtype=pl.String),
        pl.Series("skel", skel, dtype=pl.String),
        pl.Series("addr_n", addr_n, dtype=pl.String),
        pl.Series("state", state, dtype=pl.String),
    ).with_columns(
        pl.col("name_n").str.replace_all(" ", "").alias("name_sq"),
        (pl.col("name_n") + " | " + pl.col("addr_n")).alias("full_n"),
    )


# Exact name-key source: every Reference Record of the country whose `name_key` equals the Mention's, so a
# Mention without an address still reaches its Reference Record among many look-alikes. Keys shared by more
# than KEY_MAX_REFS Reference Records carry no identity and are skipped.
KEY_MAX_REFS = 30


def with_keys(records):
    return records.with_columns(pl.Series("nkey", [name_key(n) for n in records["business_name"].fill_null("").to_list()],
                                          dtype=pl.String))


def key_table(references):
    """(entity_id, country, nkey) of the Reference Records whose key is shared by at most KEY_MAX_REFS of them."""
    keys = with_keys(references).select("entity_id", "country", "nkey")
    n = keys.group_by("country", "nkey").len()
    return keys.join(n.filter(pl.col("len") <= KEY_MAX_REFS), on=["country", "nkey"]).drop("len")


def key_candidates(keys, mentions):
    """Exact name-key Candidates (ref_id, mention_id, sim_keyx=1) for some Mentions."""
    return (with_keys(mentions).select(pl.col("entity_id").alias("mention_id"), "country", "nkey")
            .join(keys, on=["country", "nkey"])
            .select(pl.col("entity_id").alias("ref_id"), "mention_id", pl.lit(1.0, dtype=pl.Float32).alias("sim_keyx")))


# Core-name source for Mentions without an address: every Reference Record of the country with the same
# normalized name (legal forms dropped). With no address, the name is all there is, and at test density a
# name is shared by more Reference Records than any top-k text source keeps.
NAME_MAX_REFS = 50
NO_ADDRESS_CHARS = 3


def name_table(references):
    t = references.select("entity_id", "country", "name_n").filter(pl.col("name_n") != "")
    n = t.group_by("country", "name_n").len()
    return t.join(n.filter(pl.col("len") <= NAME_MAX_REFS), on=["country", "name_n"]).drop("len")


def name_candidates(names, mentions):
    """Core-name Candidates (ref_id, mention_id, sim_namex=1) for the Mentions without an address."""
    m = mentions.filter(pl.col("addr_n").str.len_chars() < NO_ADDRESS_CHARS)
    return (m.select(pl.col("entity_id").alias("mention_id"), "country", "name_n").join(names, on=["country", "name_n"])
            .select(pl.col("entity_id").alias("ref_id"), "mention_id", pl.lit(1.0, dtype=pl.Float32).alias("sim_namex")))


# A place (one comma-separated address component, e.g. a city) implies a state when the Reference Records
# that mention it agree on the state: seen at least PLACE_MIN_COUNT times, PLACE_PURITY of them in one state.
PLACE_MIN_COUNT = 20
PLACE_PURITY = 0.95


def _places(records):
    return (records.select("entity_id", "country", "state",
                           pl.col("business_address").fill_null("").str.to_lowercase().str.split(",").alias("place"))
            .explode("place")
            .with_columns(pl.col("place").str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars())
            .filter(pl.col("place") != ""))


def place_states(refs):
    """(country, place, state) for places whose Reference Records almost always share one state."""
    p = _places(refs.filter(pl.col("state") != "")).group_by("country", "place", "state").len()
    tot = p.group_by("country", "place").agg(pl.col("len").sum().alias("tot"))
    return (p.join(tot, on=["country", "place"])
            .filter((pl.col("tot") >= PLACE_MIN_COUNT) & (pl.col("len") >= PLACE_PURITY * pl.col("tot")))
            .select("country", "place", "state"))


def with_states(records, places=None):
    """Recompute each Record's state from its address; fill a missing one from its places (city) if known."""
    st = [address_state(a, c) for a, c in zip(records["addr_n"].to_list(), records["country"].to_list())]
    records = records.with_columns(pl.Series("state", st, dtype=pl.String))
    if places is None or places.height == 0:
        return records
    guess = (_places(records.filter(pl.col("state") == "")).drop("state").join(places, on=["country", "place"])
             .group_by("entity_id").agg(pl.col("state").mode().first().alias("_guess")))
    return (records.join(guess, on="entity_id", how="left")
            .with_columns(pl.when(pl.col("state") == "").then(pl.col("_guess").fill_null("")).otherwise(pl.col("state")).alias("state"))
            .drop("_guess"))


def _vectorizer(analyzer, max_df, n_docs):
    """max_df drops terms shared by many Reference Records ("rue", " st", state codes): they carry little
    signal but dominate the cost of the sparse product. Only used on the costly address-bearing sources."""
    max_df = max_df if n_docs > 10_000 else 1.0
    if analyzer == "word":
        return TfidfVectorizer(analyzer="word", sublinear_tf=True, dtype=np.float32, max_df=max_df)
    return TfidfVectorizer(analyzer=analyzer, ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32, max_df=max_df)


class _CountryIndex:
    """TF-IDF matrices of one country's Reference Records for every source, fitted once and queried per chunk."""

    def __init__(self, refs, sources):
        self.ids = refs["entity_id"]
        state = refs["state"].to_list()
        self.by_state = {}
        for i, st in enumerate(state):
            if st:
                self.by_state.setdefault(st, []).append(i)
        self.by_state = {st: np.array(ix) for st, ix in self.by_state.items()}
        self.vec, self.RT = {}, {}
        for src, (col, analyzer, _, _, max_df) in sources.items():
            vec = _vectorizer(analyzer, max_df, refs.height)
            try:
                R = vec.fit_transform(refs[col].to_numpy())
            except ValueError:  # empty vocabulary (e.g. all-empty text in a tiny country)
                continue
            self.vec[src] = vec
            self.RT[src] = {None: R.T.tocsr()}
            self.RT[src].update({st: R[ix].T.tocsr() for st, ix in self.by_state.items()})

    def partitions(self, men_state, scope):
        """(reference row indices or None for all, reference key, mention row indices) to search within.

        Mentions whose state is unknown, or has no Reference Records, search the whole country only under
        scope "nostate", and are skipped under scope "state".
        """
        if scope == "country":
            yield None, np.arange(len(men_state))
            return
        groups, wide = {}, []
        for i, st in enumerate(men_state):
            (groups.setdefault(st, []) if st in self.by_state else wide).append(i)
        if scope == "nostate":
            if wide:
                yield None, np.array(wide)
            return
        for st, m in groups.items():
            yield st, np.array(m)


class Blocker:
    """Blocking index over the Reference Records; query it with any batch of Mentions."""

    def __init__(self, references, sources=SOURCES, k_scale=1.0, n_threads=-1, log=print):
        if "name_n" not in references.columns:
            references = add_text_views(references)
        self.sources, self.k_scale, self.log = sources, k_scale, log
        self.n_threads = n_threads if n_threads > 0 else os.cpu_count()
        t = time.time()
        references = with_states(references)
        self.places = place_states(references)
        references = with_states(references, self.places)
        self.keys = key_table(references)
        self.names = name_table(references)
        self.index = {country: _CountryIndex(refs, sources) for (country,), refs in references.group_by(["country"])}
        log(f"  blocker fitted on {references.height} Reference Records in {time.time() - t:.0f}s")

    def query(self, mentions):
        """Candidates as (ref_id, mention_id, sim_<source>...) with one row per pair."""
        if "name_n" not in mentions.columns:
            mentions = add_text_views(mentions)
        mentions = with_states(mentions, self.places)
        parts = []
        for country, idx in self.index.items():
            mens = mentions.filter(pl.col("country") == country)
            if mens.height == 0:
                continue
            men_state, per_source = mens["state"].to_list(), []
            for src, (col, _, k, scope, _) in self.sources.items():
                if src not in idx.vec:
                    continue
                t0, frames = time.time(), []
                M = idx.vec[src].transform(mens[col].to_numpy())
                kk = max(1, int(k * self.k_scale))
                for st, mi in idx.partitions(men_state, scope):
                    hits = sp_matmul_topn(M[mi], idx.RT[src][st], top_n=kk, threshold=0.05, n_threads=self.n_threads).tocoo()
                    ref_rows = hits.col if st is None else idx.by_state[st][hits.col]
                    frames.append(pl.DataFrame({"ref_id": idx.ids.gather(ref_rows),
                                                "mention_id": mens["entity_id"].gather(mi[hits.row]),
                                                f"sim_{src}": hits.data.astype(np.float32)}))
                if frames:
                    df = pl.concat(frames)
                    self.log(f"  block {country} {src}: {df.height} hits in {time.time() - t0:.0f}s")
                    per_source.append(df)
            if not per_source:
                continue
            merged = per_source[0]
            for df in per_source[1:]:
                merged = merged.join(df, on=["ref_id", "mention_id"], how="full", coalesce=True)
            parts.append(merged)
        for extra in (key_candidates(self.keys, mentions), name_candidates(self.names, mentions)):
            if extra.height:
                parts.append(extra)
        cols = ["ref_id", "mention_id"] + [f"sim_{s}" for s in self.sources] + ["sim_keyx", "sim_namex"]
        if not parts:
            return pl.DataFrame(schema={c: (pl.String if c.endswith("_id") else pl.Float32) for c in cols})
        out = pl.concat(parts, how="diagonal_relaxed").group_by("ref_id", "mention_id").agg(pl.all().max())
        for c in cols[2:]:
            if c not in out.columns:
                out = out.with_columns(pl.lit(0.0, dtype=pl.Float32).alias(c))
        return out.select(cols).fill_null(0.0)


def block(references, mentions, k_scale=1.0, n_threads=-1, sources=SOURCES, log=print):
    """Candidates as (ref_id, mention_id, sim_<source>...) with one row per pair."""
    return Blocker(references, sources, k_scale, n_threads, log).query(mentions)
