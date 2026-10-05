#!/usr/bin/env bash
set -euo pipefail

# Factorial loss ablation: run listwise-only, pairwise odds, positive-title SFT
# regularization, and full SLA once per seed. The default is one seed; standard
# deviation is blank for a single run.
#
# Usage:
#   SEEDS="0" bash review_sla_loss_factorial.sh CATEGORY REVISE LLM
# Example:
#   SEEDS="0" bash review_sla_loss_factorial.sh All_Handmade 0 qwen17
# Minimal rebuttal command (one run per condition):
#   bash review_sla_loss_factorial.sh All_Handmade 0 qwen17

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
# Pin every Python process in this runner to physical GPU 1. CUDA exposes that
# device to PyTorch as cuda:0 after visibility filtering.
GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "Using physical GPU ${GPU_ID} (visible to Python as cuda:0)"
if [[ $# -lt 3 ]]; then
  echo "Usage: SEEDS='0' bash review_sla_loss_factorial.sh CATEGORY REVISE LLM" >&2
  exit 2
fi
CATEGORY="$1" REVISE="$2" LLM="$3"
read -r -a SEED_LIST <<< "${SEEDS:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/review_results/sla_loss_factorial/${CATEGORY}/${STAMP}"
mkdir -p "${OUT_DIR}"

for SEED in "${SEED_LIST[@]}"; do
  for CASE in listwise_only listwise_pairwise listwise_sft full_sla; do
    case "${CASE}" in
      listwise_only) ODDS=0.0; SFT=0.0 ;;
      listwise_pairwise) ODDS=0.1; SFT=0.0 ;;
      listwise_sft) ODDS=0.0; SFT=0.05 ;;
      full_sla) ODDS=0.1; SFT=0.05 ;;
    esac
    RUN_NAME="review_sla_factorial_${CATEGORY}_${CASE}_seed-${SEED}_${STAMP}"
    echo "== ${CASE}, seed=${SEED}, odds=${ODDS}, sft=${SFT} =="
    python -u train.py --category "${CATEGORY}" --revise "${REVISE}" --llm "${LLM}" \
      --run_name "${RUN_NAME}" --seed "${SEED}" --odds_weight "${ODDS}" \
      --sft_loss_weight "${SFT}" --disable_auto_skip_sla --force_rebuild_candidates \
      2>&1 | tee "${OUT_DIR}/${CASE}_seed-${SEED}_train.log"
    python -u evaluate.py --category "${CATEGORY}" --revise "${REVISE}" --llm "${LLM}" \
      --run_name "${RUN_NAME}" --seed "${SEED}" --force_rebuild_candidates \
      2>&1 | tee "${OUT_DIR}/${CASE}_seed-${SEED}_eval.log"
    python -u metrics.py --category "${CATEGORY}" --revise "${REVISE}" --llm "${LLM}" \
      --run_name "${RUN_NAME}" --topk 10
    cp "${ROOT}/metrics/${CATEGORY}/${CATEGORY}_${REVISE}_${RUN_NAME}.json" "${OUT_DIR}/${CASE}_seed-${SEED}_metrics.json"
    cp "${ROOT}/checkpoints/${CATEGORY}/${RUN_NAME}/run_summary.json" "${OUT_DIR}/${CASE}_seed-${SEED}_run_summary.json"
  done
done

python - "${OUT_DIR}" "${#SEED_LIST[@]}" "${SEED_LIST[@]}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

folder = Path(sys.argv[1])
nseed = int(sys.argv[2])
seeds = sys.argv[3:3+nseed]
cases = ("listwise_only", "listwise_pairwise", "listwise_sft", "full_sla")
metrics = ("HR1", "HR5", "HR10", "NDCG5", "NDCG10")
with (folder / "loss_mean_std.tsv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["case", "metric", "mean", "sample_std", "runs"])
    for case in cases:
        samples = {name: [] for name in metrics}
        for seed in seeds:
            with (folder / f"{case}_seed-{seed}_metrics.json").open(encoding="utf-8") as g:
                result = json.load(g)["results"]
            values = {"HR1": result["k1"]["hr"], "HR5": result["k5"]["hr"],
                      "HR10": result["k10"]["hr"], "NDCG5": result["k5"]["ndcg"],
                      "NDCG10": result["k10"]["ndcg"]}
            for key, value in values.items(): samples[key].append(float(value))
        for metric, values in samples.items():
            writer.writerow([case, metric, statistics.mean(values),
                             statistics.stdev(values) if len(values) > 1 else "", len(values)])
print(folder / "loss_mean_std.tsv")
PY

echo "SLA loss factorial complete: ${OUT_DIR}"
