import polars as pl

from entity_resolution.resolve import resolve


def scored(rows):
    return pl.DataFrame(rows, schema=["ref_id", "mention_id", "score"], orient="row")


def test_mention_goes_only_to_its_highest_scoring_reference():
    s = scored([("S1-1", "S2-1", 0.9), ("S1-2", "S2-1", 0.95), ("S1-1", "S3-1", 0.8)])
    got = resolve(s, ["S1-1", "S1-2"], threshold=0.5)
    assert got == {"S1-1": ["S3-1"], "S1-2": ["S2-1"]}


def test_pairs_below_threshold_are_dropped_and_every_reference_is_listed():
    s = scored([("S1-1", "S2-1", 0.4), ("S1-2", "S2-2", 0.7)])
    got = resolve(s, ["S1-1", "S1-2", "S1-3"], threshold=0.5)
    assert got == {"S1-1": [], "S1-2": ["S2-2"], "S1-3": []}


def test_losing_reference_does_not_fall_back_to_a_mention_taken_elsewhere():
    s = scored([("S1-1", "S2-1", 0.6), ("S1-2", "S2-1", 0.9)])
    got = resolve(s, ["S1-1", "S1-2"], threshold=0.5)
    assert got == {"S1-1": [], "S1-2": ["S2-1"]}


def test_per_reference_thresholds_override_the_default():
    s = scored([("S1-1", "S2-1", 0.8), ("S1-2", "S2-2", 0.8)])
    got = resolve(s, ["S1-1", "S1-2"], threshold=0.5, ref_thresholds={"S1-2": 0.9})
    assert got == {"S1-1": ["S2-1"], "S1-2": []}
