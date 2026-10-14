#!/bin/bash
# Final stage (steps 8-9 of README.md): average the two dense stacker runs, apply the cross-encoder gate,
# calibrate and decide per Reference Record, validate. Run from the repository root.
#   bash code/business_entity_resolution/src/scripts/run_final.sh [OUT_DIR]
set -euo pipefail
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
F=code/business_entity_resolution/src/final_stage
OUT=${1:-output}

python -u $F/average_model_scores.py runs/averaged_scores runs/v12/work runs/v13b/work
python -u $F/cross_encoder_gate.py data/cross_encoder_e5_base,data/cross_encoder_e5_small runs/averaged_scores/val_scored.parquet runs/averaged_scores/scored.parquet runs/gated_scores
python -u $F/calibrate_and_decide.py runs/gated_scores/val_scored.parquet runs/gated_scores "$OUT" runs/v12/work/val_refs_orph.tsv
python code/business_entity_resolution/src/scripts/validate_submission.py --matching "$OUT/matching_results.tsv" \
  --candidate "$OUT/candidate_pairs.tsv" --test-dir student_resource/dataset/test --check-ids
