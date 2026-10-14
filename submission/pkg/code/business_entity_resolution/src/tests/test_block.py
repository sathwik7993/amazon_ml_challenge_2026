from entity_resolution.block import block
from fixtures import MENTIONS, REFERENCES, TRUTH


def candidate_set(cands):
    return set(zip(cands["ref_id"].to_list(), cands["mention_id"].to_list()))


def test_every_true_match_becomes_a_candidate():
    got = candidate_set(block(REFERENCES, MENTIONS, k_scale=0.3))
    missing = {(r, m) for r, ms in TRUTH.items() for m in ms} - got
    assert missing == set()


def test_candidates_never_cross_country():
    got = candidate_set(block(REFERENCES, MENTIONS, k_scale=0.3))
    assert not any(m == "S3-9" and r == "S1-2" for r, m in got)
    country = dict(zip(REFERENCES["entity_id"], REFERENCES["country"])) | dict(zip(MENTIONS["entity_id"], MENTIONS["country"]))
    assert all(country[r] == country[m] for r, m in got)


def test_each_candidate_pair_appears_once():
    cands = block(REFERENCES, MENTIONS, k_scale=0.3)
    assert cands.height == len(candidate_set(cands))


def test_mention_with_only_a_city_gets_its_state_from_the_reference_records():
    import polars as pl

    words = ["Alpha", "Bravo", "Canal", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliet", "Kilo",
             "Lima", "Metro", "Nova", "Oscar", "Papa", "Quebec", "Romeo", "Sierra", "Tango", "Union", "Victor"]
    refs = pl.DataFrame({
        "entity_id": [f"S1-{i}" for i in range(2 * len(words))],
        "business_name": [f"{w} Club SARL" for w in words] + [f"{w} Sport SAS" for w in words],
        "business_address": [f"{i + 1} Rue Nationale, Lille, Hauts-de-France" for i in range(len(words))]
                            + [f"{i + 1} Cours Pasteur, Bordeaux, Nouvelle-Aquitaine" for i in range(len(words))],
        "country": ["France"] * (2 * len(words)),
    })
    mention = pl.DataFrame({"entity_id": ["S2-1"], "business_name": ["ALPHA CLUB SARL"],
                            "business_address": ["1 R NATIONALE, LILLE"], "country": ["France"]})
    cands = block(refs, mention)
    hit = cands.filter(pl.col("ref_id") == "S1-0")
    assert hit.height == 1 and hit["sim_name"][0] > 0  # "name" searches within a state only


def test_name_only_mention_reaches_its_exact_name_key_among_many_look_alikes():
    import polars as pl

    decoys = [f"Urgent Care Health {w}" for w in ("LLC", "Corp", "Group", "Partners", "Clinic", "West", "East",
                                                  "North", "South", "Center", "Services", "Holdings")]
    names = decoys * 2 + ["Urgent Care Health Inc"]
    refs = pl.DataFrame({
        "entity_id": [f"S1-{i}" for i in range(len(names))],
        "business_name": names,
        "business_address": [f"{100 + i} Main Street, Casper, WY" for i in range(len(names))],
        "country": ["US"] * len(names),
    })
    mention = pl.DataFrame({"entity_id": ["S2-1"], "business_name": ["URGENT CARE HEALTH-INC"],
                            "business_address": [""], "country": ["US"]})
    cands = block(refs, mention)
    hit = cands.filter(pl.col("ref_id") == f"S1-{len(names) - 1}")
    assert hit.height == 1 and hit["sim_keyx"][0] == 1.0


def test_mention_without_address_reaches_every_reference_sharing_its_core_name():
    import polars as pl

    n = 15  # more same-name branches than any top-k text source keeps
    refs = pl.DataFrame({
        "entity_id": [f"S1-{i}" for i in range(n)],
        "business_name": ["Bhopal Global Private Limited"] * n,
        "business_address": [f"{i + 1} MG Road, Bhopal, Madhya Pradesh" for i in range(n)],
        "country": ["India"] * n,
    })
    # a noisy copy: legal form reworded, so the exact name key differs from every Reference Record
    mention = pl.DataFrame({"entity_id": ["S3-1"], "business_name": ["Bhopal Global Pvt"],
                            "business_address": [""], "country": ["India"]})
    got = set(block(refs, mention)["ref_id"].to_list())
    assert got == set(refs["entity_id"].to_list())
