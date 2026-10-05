#!/usr/bin/env bash
set -euo pipefail

# Evaluation-only controls for an existing checkpoint. Supported groups:
#   ranker: fusion / raw LLM / evidence-only, at the default slate size
#   budget: slate-size curves under global and stratified prefiltering
#   weights: one-at-a-time fusion-weight sensitivity
#
# Usage:
#   bash review_eval_controls.sh CATEGORY REVISE LLM RUN_NAME [ranker|budget|weights|all] [-- EVAL_ARGS...]
# Examples:
#   bash review_eval_controls.sh All_Handmade 0 qwen17 YOUR_RUN_NAME ranker -- --score_batch_size 8
#   SLATE_SIZES="10 20 30 40 60" bash review_eval_controls.sh All_Handmade 0 qwen17 YOUR_RUN_NAME budget
# Minimal rebuttal command (same-slate fusion / raw-LLM / evidence-only controls):
#   bash review_eval_controls.sh All_Handmade 0 qwen17 YOUR_RUN_NAME ranker
# Replace YOUR_RUN_NAME with a real directory name inside checkpoints/All_Handmade;
# do not run the placeholder literally. Use underscores without backslashes in CATEGORY.
# To use the category's latest recorded checkpoint on a Linux shell:
#   RUN_NAME="$(cat checkpoints/Health_and_Personal_Care/latest_run.txt)"
#   bash review_eval_controls.sh Health_and_Personal_Care 0 qwen17 "$RUN_NAME" ranker
# If latest_run.txt is absent, list checkpoint directories and choose an existing name:
#   ls checkpoints/Health_and_Personal_Care
#
# EVAL_ARGS should normally be batching/memory flags. Do not override ranker,
# slate_size, prefilter, or fusion weights; those are the experiment variables.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ $# -lt 4 ]]; then
  echo "Usage: bash review_eval_controls.sh CATEGORY REVISE LLM RUN_NAME [ranker|budget|weights|all] [-- EVAL_ARGS...]" >&2
  exit 2
fi

CATEGORY="$1"
REVISE="$2"
LLM="$3"
RUN_NAME="$4"
shift 4
MODE="ranker"
if (($#)) && [[ "$1" != "--" ]]; then
  MODE="$1"
  shift
fi
if (($#)); then
  [[ "$1" == "--" ]] || { echo "Use -- before optional evaluation arguments." >&2; exit 2; }
  shift
fi
EVAL_ARGS=("$@")

CHECKPOINT_DIR="${ROOT}/checkpoints/${CATEGORY}/${RUN_NAME}"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint run not found: ${CHECKPOINT_DIR}" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/review_results/eval_controls/${CATEGORY}/${RUN_NAME}/${STAMP}"
mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.tsv"
printf 'case\tslate_size\tprefilter\tranker\tllm_weight\tevidence_weight\tsource_weight\treservoir_recall\tdeploy_recall\tHR1\tHR5\tHR10\tNDCG5\tNDCG10\teval_seconds\n' > "${SUMMARY}"

run_case() {
  local TAG="$1" SLATE="$2" PREFILTER="$3" RANKER="$4" LW="$5" EW="$6" SW="$7" REBUILD="$8"
  local ARGS=(
    --category "${CATEGORY}" --revise "${REVISE}" --llm "${LLM}"
    --run_name "${RUN_NAME}" --wide_size 150 --slate_size "${SLATE}"
    --transition_k 80 --retrieval_k 40 --prefilter "${PREFILTER}"
    --ranker "${RANKER}" --llm_rank_weight "${LW}"
    --evidence_rank_weight "${EW}" --source_rank_weight "${SW}"
    --topk 10
  )
  if [[ "${REBUILD}" == "yes" ]]; then
    ARGS+=(--force_rebuild_candidates)
  fi
  ARGS+=("${EVAL_ARGS[@]}")

  echo "== ${TAG}: N=${SLATE}, prefilter=${PREFILTER}, ranker=${RANKER}, weights=${LW}/${EW}/${SW} =="
  python -u evaluate.py "${ARGS[@]}"
  python -u metrics.py --category "${CATEGORY}" --revise "${REVISE}" --llm "${LLM}" --run_name "${RUN_NAME}" --topk 10

  cp "${ROOT}/outputs/${CATEGORY}/${RUN_NAME}.json" "${OUT_DIR}/${TAG}_predictions.json"
  cp "${ROOT}/checkpoints/${CATEGORY}/${RUN_NAME}/eval_summary.json" "${OUT_DIR}/${TAG}_eval_summary.json"
  cp "${ROOT}/metrics/${CATEGORY}/${CATEGORY}_${REVISE}_${RUN_NAME}.json" "${OUT_DIR}/${TAG}_metrics.json"

  python - "${SUMMARY}" "${TAG}" "${SLATE}" "${PREFILTER}" "${RANKER}" "${LW}" "${EW}" "${SW}" "${OUT_DIR}/${TAG}_eval_summary.json" "${OUT_DIR}/${TAG}_metrics.json" <<'PY'
import csv
import json
import sys

summary, tag, slate, prefilter, ranker, lw, ew, sw, eval_path, metrics_path = sys.argv[1:]
with open(eval_path, "r", encoding="utf-8") as handle:
    evaluation = json.load(handle)
with open(metrics_path, "r", encoding="utf-8") as handle:
    metrics = json.load(handle)["results"]
reach = evaluation.get("reachability", {})
with open(summary, "a", encoding="utf-8", newline="") as handle:
    csv.writer(handle, delimiter="\t").writerow([
        tag, slate, prefilter, ranker, lw, ew, sw,
        reach.get("reservoir", ""), reach.get("deploy", ""),
        metrics["k1"]["hr"], metrics["k5"]["hr"], metrics["k10"]["hr"],
        metrics["k5"]["ndcg"], metrics["k10"]["ndcg"],
        evaluation.get("elapsed_seconds", ""),
    ])
PY
}

run_ranker_group() {
  local PREFILTER="$1"
  run_case "${PREFILTER}_fusion" 30 "${PREFILTER}" fusion 0.35 1.0 0.30 yes
  run_case "${PREFILTER}_llm" 30 "${PREFILTER}" llm 0.35 1.0 0.30 no
  run_case "${PREFILTER}_evidence" 30 "${PREFILTER}" evidence 0.35 1.0 0.30 no
}

run_budget_group() {
  local PREFILTER="$1"
  for SLATE in ${SLATE_SIZES:-10 20 30 40 60}; do
    run_case "${PREFILTER}_slate-${SLATE}" "${SLATE}" "${PREFILTER}" fusion 0.35 1.0 0.30 yes
  done
}

run_weight_group() {
  local PREFILTER=stratified
  run_case "weights_anchor" 30 "${PREFILTER}" fusion 0.35 1.0 0.30 yes
  for W in 0.0 0.7 1.0; do
    run_case "llm_weight-${W}" 30 "${PREFILTER}" fusion "${W}" 1.0 0.30 no
  done
  for W in 0.0 0.5 2.0; do
    run_case "evidence_weight-${W}" 30 "${PREFILTER}" fusion 0.35 "${W}" 0.30 no
  done
  for W in 0.0 0.15 0.6; do
    run_case "source_weight-${W}" 30 "${PREFILTER}" fusion 0.35 1.0 "${W}" no
  done
}

case "${MODE}" in
  ranker)
    for PREFILTER in ${PREFILTERS:-stratified global}; do run_ranker_group "${PREFILTER}"; done
    ;;
  budget)
    for PREFILTER in ${PREFILTERS:-stratified global}; do run_budget_group "${PREFILTER}"; done
    ;;
  weights) run_weight_group ;;
  all)
    for PREFILTER in ${PREFILTERS:-stratified global}; do run_ranker_group "${PREFILTER}"; done
    for PREFILTER in ${PREFILTERS:-stratified global}; do run_budget_group "${PREFILTER}"; done
    run_weight_group
    ;;
  *) echo "Unknown mode: ${MODE} (choose ranker, budget, weights, or all)" >&2; exit 2 ;;
esac

echo "Evaluation-control results: ${OUT_DIR}"
echo "Summary: ${SUMMARY}"
