"""Leaderboard Score: macro F0.5 over Reference Records (see CONTEXT.md)."""
from pathlib import Path


def entity_score(predicted, truth):
    predicted, truth = set(predicted), set(truth)
    if not truth:
        return 1.0 if not predicted else 0.0
    hits = len(predicted & truth)
    if hits == 0:
        return 0.0
    p, r = hits / len(predicted), hits / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def evaluate(predicted, truth):
    """Mean Entity Score over every Reference Record in `truth`; missing predictions count as empty."""
    return sum(entity_score(predicted.get(k, ()), v) for k, v in truth.items()) / len(truth)


def read_matches(path):
    """Read a matching_results / ground_truth TSV into {reference_id: [mention_ids]}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            ref, _, ids = line.rstrip("\n").partition("\t")
            out[ref] = [i for i in ids.split(",") if i]
    return out


if __name__ == "__main__":
    import sys

    pred, gt = read_matches(Path(sys.argv[1])), read_matches(Path(sys.argv[2]))
    print(f"F0.5 = {evaluate(pred, gt):.5f}")
