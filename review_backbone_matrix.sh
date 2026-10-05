#!/usr/bin/env bash
set -euo pipefail

# Run the full system and its SFT-only controlled variant once for each selected
# backbone. The default is one seed; with one seed, sample standard deviation is blank.
#
# Usage:
#   SEEDS="0" bash review_backbone_matrix.sh REVISE CATEGORY [CATEGORY ...] -- LLM [LLM ...]
# Examples:
#   SEEDS="0" bash review_backbone_matrix.sh 0 All_Handmade -- gemma2b
#   SEEDS="0" bash review_backbone_matrix.sh 0 All_Handmade -- llama32_1b
#   SEEDS="0" bash review_backbone_matrix.sh 0 All_Handmade -- olmo2_1b
# Supported keys: qwen14 qwen8 qwen4 qwen17 qwen06 gemma2b llama32_1b olmo2_1b.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
if [[ $# -lt 4 ]]; then
  echo "Usage: SEEDS='0' bash review_backbone_matrix.sh REVISE CATEGORY [CATEGORY ...] -- LLM [LLM ...]" >&2
  exit 2
fi
REVISE="$1"; shift
CATEGORIES=(); LLMS=()
while (($#)); do
  if [[ "$1" == "--" ]]; then shift; LLMS=("$@"); break; fi
  CATEGORIES+=("$1"); shift
done
((${#CATEGORIES[@]} && ${#LLMS[@]})) || { echo "Supply categories before -- and model keys after it." >&2; exit 2; }
for llm in "${LLMS[@]}"; do
  case "$llm" in qwen14|qwen8|qwen4|qwen17|qwen06|gemma2b|llama32_1b|olmo2_1b) ;; *) echo "Unsupported LLM key: $llm" >&2; exit 2 ;; esac
done
read -r -a SEED_LIST <<< "${SEEDS:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/review_results/backbone_matrix/${STAMP}"
mkdir -p "${OUT_DIR}"

for CATEGORY in "${CATEGORIES[@]}"; do
  for LLM in "${LLMS[@]}"; do
    for SEED in "${SEED_LIST[@]}"; do
      for VARIANT in full sft_only; do
        RUN_NAME="review_backbone_${CATEGORY}_${LLM}_${VARIANT}_seed-${SEED}_${STAMP}"
        DEST="${OUT_DIR}/${CATEGORY}/${LLM}/${VARIANT}"
        mkdir -p "${DEST}"
        VARIANT_ARGS=()
        [[ "${VARIANT}" == "sft_only" ]] && VARIANT_ARGS=(--sla_epochs 0)
        echo "== ${CATEGORY} ${LLM} ${VARIANT} seed=${SEED} =="
        python -u "${ROOT}/review_backbone_run.py" "${CATEGORY}" "${REVISE}" "${LLM}" \
          --run_name "${RUN_NAME}" --seed "${SEED}" --force_rebuild_candidates \
          "${VARIANT_ARGS[@]}" 2>&1 | tee "${DEST}/seed-${SEED}.log"
        cp "${ROOT}/metrics/${CATEGORY}/${CATEGORY}_${REVISE}_${RUN_NAME}.json" "${DEST}/seed-${SEED}_metrics.json"
        cp "${ROOT}/checkpoints/${CATEGORY}/${RUN_NAME}/run_summary.json" "${DEST}/seed-${SEED}_run_summary.json"
      done
    done
  done
done

python - "${OUT_DIR}" "${#SEED_LIST[@]}" "${SEED_LIST[@]}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
nseed = int(sys.argv[2])
seeds = sys.argv[3:3 + nseed]
metrics = ("HR1", "HR5", "HR10", "NDCG5", "NDCG10")
rows = []
for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    for model_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
        for variant in ("full", "sft_only"):
            samples = {m: [] for m in metrics}
            elapsed = []
            for seed in seeds:
                with (model_dir / variant / f"seed-{seed}_metrics.json").open(encoding="utf-8") as f:
                    result = json.load(f)["results"]
                vals = {"HR1": result["k1"]["hr"], "HR5": result["k5"]["hr"],
                        "HR10": result["k10"]["hr"], "NDCG5": result["k5"]["ndcg"],
                        "NDCG10": result["k10"]["ndcg"]}
                for name, value in vals.items(): samples[name].append(float(value))
                with (model_dir / variant / f"seed-{seed}_run_summary.json").open(encoding="utf-8") as f:
                    elapsed.append(float(json.load(f).get("elapsed_seconds", 0.0)))
            for name, values in samples.items():
                rows.append([category_dir.name, model_dir.name, variant, name,
                             statistics.mean(values), statistics.stdev(values) if len(values) > 1 else "",
                             statistics.mean(elapsed), len(values)])
with (root / "backbone_mean_std.tsv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["category", "backbone", "variant", "metric", "mean", "sample_std", "mean_train_seconds", "runs"])
    w.writerows(rows)
print(root / "backbone_mean_std.tsv")
PY

echo "Backbone matrix complete: ${OUT_DIR}"
