#!/bin/bash
# Final stage (steps 8-9 of README.md): bag the two dense stackers, apply the cross-encoder gate,
# calibrate and decide per Reference Record, validate. Run from the repository root.
#   bash code/business_entity_resolution/scripts/run_final.sh [OUT_DIR]
set -euo pipefail
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
F=code/business_entity_resolution/src/final
OUT=${1:-output}

python -u $F/bag.py runs/v13bag runs/v12/work runs/v13b/work
python -u $F/ce_gate.py data/ce_hard_base,data/ce_hard_small2 runs/v13bag/val_scored.parquet runs/v13bag/scored.parquet runs/v18gate
python -u $F/calibrate_test_ef.py runs/v18gate/val_scored.parquet runs/v18gate "$OUT" runs/v12/work/val_refs_orph.tsv
python code/business_entity_resolution/scripts/validate_submission.py --matching "$OUT/matching_results.tsv" \
  --candidate "$OUT/candidate_pairs.tsv" --test-dir student_resource/dataset/test --check-ids
