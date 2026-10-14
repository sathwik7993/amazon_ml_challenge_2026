import subprocess
import sys
from pathlib import Path

from entity_resolution.evaluate import evaluate, read_matches
from entity_resolution.pipeline import run_pipeline

DATA = Path(__file__).parent / "data"
VALIDATOR = Path(__file__).resolve().parents[1] / "scripts" / "validate_submission.py"


def test_pipeline_output_passes_the_official_validator_and_beats_chance(tmp_path):
    out = tmp_path / "output"
    run_pipeline(DATA / "train", DATA / "test", out, tmp_path / "work", val_dir=DATA / "train", rounds=50, chunk_size=500, stack=True)

    check = subprocess.run(
        [sys.executable, str(VALIDATOR), "--matching", str(out / "matching_results.tsv"),
         "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(DATA / "test"), "--check-ids"],
        capture_output=True, text=True,
    )
    assert check.returncode == 0, check.stdout + check.stderr

    score = evaluate(read_matches(out / "matching_results.tsv"), read_matches(DATA / "test" / "hidden_ground_truth.tsv"))
    assert score > 0.8
