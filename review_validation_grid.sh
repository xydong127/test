#!/usr/bin/env bash
set -euo pipefail

# Validation-only selection grid for source policies, slate-size/quota effects,
# and EOR weights. Original Python files are patched only in this process.
# The frozen-test mode evaluates exactly one configuration selected on validation.
#
# Usage:
#   bash review_validation_grid.sh CATEGORY REVISE LLM RUN_NAME [source|quota|weights|all|frozen-test]
# Examples:
#   bash review_validation_grid.sh All_Handmade 0 qwen17 YOUR_RUN_NAME quota
#   SLATE_SIZES="10 20 30 40 60" bash review_validation_grid.sh All_Handmade 0 qwen17 YOUR_RUN_NAME quota
# Compact rebuttal commands (validation split only):
#   SLATE_SIZES="20 40" QUOTA_MODES="current global" bash review_validation_grid.sh All_Handmade 0 qwen17 YOUR_RUN_NAME quota
#   bash review_validation_grid.sh All_Handmade 0 qwen17 YOUR_RUN_NAME source
#   LLM_WEIGHTS="0 0.35 0.7" EVIDENCE_WEIGHTS="0 1 2" SOURCE_WEIGHTS="0 0.3 0.6" bash review_validation_grid.sh All_Handmade 0 qwen17 YOUR_RUN_NAME weights
# After selecting a validation setting, replace the values below with that setting and run its frozen test once:
#   FROZEN_VALIDATION_SELECTION=selected SLATE_SIZE=30 QUOTA_MODE=current LLM_WEIGHT=0.35 EVIDENCE_WEIGHT=1 SOURCE_WEIGHT=0.3 bash review_validation_grid.sh All_Handmade 0 qwen17 YOUR_RUN_NAME frozen-test
# Replace YOUR_RUN_NAME with the trained checkpoint run name.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ $# -lt 4 ]]; then
  echo "Usage: bash review_validation_grid.sh CATEGORY REVISE LLM RUN_NAME [source|quota|weights|all|frozen-test]" >&2
  exit 2
fi
CATEGORY="$1" REVISE="$2" LLM="$3" RUN_NAME="$4" MODE="${5:-quota}"
if [[ -z "${RUN_NAME}" || "${RUN_NAME}" == 'YOUR_RUN_NAME' || "${RUN_NAME}" == '$RUN_NAME' ]]; then
  echo "RUN_NAME must be the actual checkpoint folder name under checkpoints/${CATEGORY}." >&2
  echo "For example, read checkpoints/${CATEGORY}/latest_run.txt and pass that value as argument 4." >&2
  exit 2
fi
CHECKPOINT_DIR="${ROOT}/checkpoints/${CATEGORY}/${RUN_NAME}"
[[ -d "${CHECKPOINT_DIR}" ]] || { echo "Checkpoint not found: ${CHECKPOINT_DIR}" >&2; exit 2; }

SLATE_SIZE="${SLATE_SIZE:-30}"
WIDE_SIZE="${WIDE_SIZE:-150}"
TRANSITION_K="${TRANSITION_K:-80}"
RETRIEVAL_K="${RETRIEVAL_K:-40}"
FILL_K="${FILL_K:-200}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-16}"
TOPK="${TOPK:-10}"
BASE_LW="${BASE_LW:-0.35}"
BASE_EW="${BASE_EW:-1.0}"
BASE_SW="${BASE_SW:-0.30}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/review_results/validation_grid/${CATEGORY}/${RUN_NAME}/${STAMP}"
mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/validation_summary.tsv"
printf 'case\tsplit\tslate_size\tquota\tomega\tkappa\tllm_weight\tevidence_weight\tsource_weight\treservoir_recall\tdeploy_recall\tconditional_HR5\tconditional_HR10\tHR5\tHR10\tNDCG10\teval_seconds\n' > "${SUMMARY}"

run_case() {
  local TAG="$1" N="$2" QUOTA="$3" OMEGA="$4" KAPPA="$5" LW="$6" EW="$7" SW="$8" SPLIT="${9:-valid}"
  echo "== ${SPLIT} case ${TAG}: N=${N}, quota=${QUOTA}, omega=${OMEGA}, kappa=${KAPPA}, weights=${LW}/${EW}/${SW} =="
  python - "${CATEGORY}" "${REVISE}" "${LLM}" "${RUN_NAME}" "${TAG}" "${OUT_DIR}" "${SUMMARY}" "${SPLIT}" \
    "${N}" "${QUOTA}" "${OMEGA}" "${KAPPA}" "${LW}" "${EW}" "${SW}" \
    "${WIDE_SIZE}" "${TRANSITION_K}" "${RETRIEVAL_K}" "${FILL_K}" "${MAX_LENGTH}" \
    "${TOPK}" "${EVAL_BATCH_SIZE}" "${SCORE_BATCH_SIZE}" <<'PY'
import csv
import json
import math
import sys
import time
from pathlib import Path

(category, revise, llm, run_name, tag, out_dir, summary, split,
 slate_size, quota_mode, omega_csv, kappa_csv, llm_weight, evidence_weight, source_weight,
 wide_size, transition_k, retrieval_k, fill_k, max_length, topk,
 eval_batch_size, score_batch_size) = sys.argv[1:24]
revise = int(revise)
out_dir = Path(out_dir)
omega_values = [float(v) for v in omega_csv.split(",")]
kappa_values = [float(v) for v in kappa_csv.split(",")]
omega = dict(zip(("beam", "transition", "retrieval", "fill"), omega_values))
kappa = dict(zip(("beam", "transition", "retrieval", "fill"), kappa_values))
base_omega = {"beam": 4.0, "transition": 3.0, "retrieval": 2.0, "fill": 0.25}

import candidate_builder as cb
import evaluate
import llm_candidate_scoring as scoring
from utils import build_paths, save_json, summarize_metrics

# Evaluate the validation CSV through evaluate.py's existing pipeline.
original_paths = evaluate.build_paths
def validation_paths(category_arg, revise_arg, run_name=None):
    paths = original_paths(category_arg, revise_arg, run_name=run_name)
    if split == "valid":
        paths["test_csv"] = paths["valid_csv"]
    elif split != "test":
        raise ValueError(f"Unknown split: {split}")
    return paths
evaluate.build_paths = validation_paths

original_add = cb._add_candidate
def scaled_add(candidate_map, title, source, evidence, rank, item_id_by_title):
    factor = omega[source] / base_omega[source] if base_omega[source] else 1.0
    return original_add(candidate_map, title, source, float(evidence) * factor, rank, item_id_by_title)
cb._add_candidate = scaled_add

original_selector = cb._select_deploy_candidates
def select_candidates(reservoir, slate_size, topk):
    limit = max(int(slate_size), int(topk))
    ordered = sorted(reservoir, key=lambda row: float(row.get("evidence_score", 0.0)), reverse=True)
    if quota_mode == "current":
        return original_selector(reservoir, slate_size, topk)
    if quota_mode == "global":
        return ordered[:limit]
    if quota_mode != "equal":
        raise ValueError(f"Unknown quota mode: {quota_mode}")
    sources = ("beam", "transition", "retrieval", "fill")
    source_lists = {
        source: sorted(
            [row for row in reservoir if row.get(source)],
            key=lambda row: (int((row.get("source_ranks", {}) or {}).get(source, 10**6)),
                             -float(row.get("evidence_score", 0.0))),
        ) for source in sources
    }
    selected, seen = [], set()
    cb._append_unique(selected, seen, ordered[:max(int(topk), limit // 4)], limit)
    quota = int(math.ceil(limit / len(sources)))
    for source in sources:
        cb._append_unique(selected, seen, source_lists[source][:quota], limit)
    cb._append_unique(selected, seen, ordered, limit)
    return selected[:limit]
cb._select_deploy_candidates = select_candidates

def source_prior(candidate):
    ranks = candidate.get("source_ranks", {}) or {}
    vals = [kappa[s] / max(float(ranks.get(s, 1)), 1.0)
            for s in ("beam", "transition", "retrieval", "fill") if candidate.get(s)]
    return max(vals) if vals else 0.0
scoring._candidate_source_prior = source_prior

slate_path = out_dir / f"{tag}_{split}_slates.json"
original_builder = evaluate.build_candidate_slates
def capture_slates(*args, **kwargs):
    slates = original_builder(*args, **kwargs)
    save_json(slate_path, slates)
    return slates
evaluate.build_candidate_slates = capture_slates

sys.argv = [
    "evaluate.py", "--category", category, "--revise", str(revise), "--llm", llm,
    "--run_name", run_name, "--wide_size", wide_size, "--slate_size", slate_size,
    "--transition_k", transition_k, "--retrieval_k", retrieval_k, "--fill_k", fill_k,
    "--max_length", max_length, "--topk", topk, "--eval_batch_size", eval_batch_size,
    "--score_batch_size", score_batch_size, "--prefilter", "stratified", "--ranker", "fusion",
    "--llm_rank_weight", llm_weight, "--evidence_rank_weight", evidence_weight,
    "--source_rank_weight", source_weight, "--force_rebuild_candidates",
]
evaluate.main()

paths = build_paths(category, revise, run_name=run_name)
predictions = json.loads(paths["output_json"].read_text(encoding="utf-8"))
metrics = summarize_metrics(predictions, topk_values=[1, 5, 10])
evaluation = json.loads(paths["eval_summary_json"].read_text(encoding="utf-8"))
save_json(out_dir / f"{tag}_predictions.json", predictions)
save_json(out_dir / f"{tag}_eval_summary.json", evaluation)

reachable = [row for row in predictions if row.get("deploy_reachable")]
def conditional_hit(k):
    if not reachable:
        return math.nan
    return sum(str(row.get("output", "")).casefold() in
               [str(x).casefold() for x in row.get("predict", [])[:k]] for row in reachable) / len(reachable)

slates = json.loads(slate_path.read_text(encoding="utf-8"))
source_names = ("beam", "transition", "retrieval", "fill")
composition = {s: 0 for s in source_names}
reached = {s: 0 for s in source_names}
retained = {s: 0 for s in source_names}
candidate_count = 0
for slate in slates:
    target = " ".join(str(slate.get("output_title", "")).casefold().split())
    target_row = next((row for row in slate.get("reservoir_candidates", [])
                       if " ".join(str(row.get("title", "")).casefold().split()) == target), None)
    deployed = {" ".join(str(row.get("title", "")).casefold().split())
                for row in slate.get("deploy_candidates", [])}
    for row in slate.get("deploy_candidates", []):
        candidate_count += 1
        for source in row.get("sources", []):
            if source in composition:
                composition[source] += 1
    if target_row:
        for source in target_row.get("sources", []):
            if source in reached:
                reached[source] += 1
                if target in deployed:
                    retained[source] += 1
source_rows = []
for source in source_names:
    source_rows.append({"source": source,
        "deploy_candidate_fraction": composition[source] / candidate_count if candidate_count else None,
        "targets_reached": reached[source], "targets_retained": retained[source],
        "retention_given_reach": retained[source] / reached[source] if reached[source] else None})
save_json(out_dir / f"{tag}_source_coverage.json", source_rows)

with open(summary, "a", encoding="utf-8", newline="") as handle:
    csv.writer(handle, delimiter="\t").writerow([
        tag, split, slate_size, quota_mode, omega_csv, kappa_csv,
        llm_weight, evidence_weight, source_weight,
        evaluation.get("reachability", {}).get("reservoir", ""),
        evaluation.get("reachability", {}).get("deploy", ""),
        conditional_hit(5), conditional_hit(10),
        metrics["k5"]["hr"], metrics["k10"]["hr"], metrics["k10"]["ndcg"],
        evaluation.get("elapsed_seconds", ""),
    ])
print(f"Saved validation case {tag} under {out_dir}")
PY
}

run_quota_grid() {
  for n in ${SLATE_SIZES:-10 20 30 40 60}; do
    for q in ${QUOTA_MODES:-current global equal}; do
      run_case "quota_N${n}_${q}" "$n" "$q" "4,3,2,0.25" "1,0.9,0.8,0.15" "$BASE_LW" "$BASE_EW" "$BASE_SW"
    done
  done
}
run_source_grid() {
  n="$SLATE_SIZE"
  run_case source_current "$n" current "4,3,2,0.25" "1,0.9,0.8,0.15" "$BASE_LW" "$BASE_EW" "$BASE_SW"
  run_case source_equal_omega "$n" current "1,1,1,1" "1,0.9,0.8,0.15" "$BASE_LW" "$BASE_EW" "$BASE_SW"
  run_case source_equal_prior "$n" current "4,3,2,0.25" "1,1,1,1" "$BASE_LW" "$BASE_EW" "$BASE_SW"
  run_case source_no_prior "$n" current "4,3,2,0.25" "0,0,0,0" "$BASE_LW" "$BASE_EW" "$BASE_SW"
  run_case source_no_transition_scale "$n" current "4,0,2,0.25" "1,0.9,0.8,0.15" "$BASE_LW" "$BASE_EW" "$BASE_SW"
  run_case source_no_retrieval_scale "$n" current "4,3,0,0.25" "1,0.9,0.8,0.15" "$BASE_LW" "$BASE_EW" "$BASE_SW"
}
run_weight_grid() {
  n="$SLATE_SIZE"
  run_case weight_anchor "$n" current "4,3,2,0.25" "1,0.9,0.8,0.15" "$BASE_LW" "$BASE_EW" "$BASE_SW"
  for w in ${LLM_WEIGHTS:-0 0.175 0.35 0.525 0.7}; do
    run_case "llm_weight_${w}" "$n" current "4,3,2,0.25" "1,0.9,0.8,0.15" "$w" "$BASE_EW" "$BASE_SW"
  done
  for w in ${EVIDENCE_WEIGHTS:-0 0.5 1 1.5 2}; do
    run_case "evidence_weight_${w}" "$n" current "4,3,2,0.25" "1,0.9,0.8,0.15" "$BASE_LW" "$w" "$BASE_SW"
  done
  for w in ${SOURCE_WEIGHTS:-0 0.15 0.3 0.45 0.6}; do
    run_case "source_weight_${w}" "$n" current "4,3,2,0.25" "1,0.9,0.8,0.15" "$BASE_LW" "$BASE_EW" "$w"
  done
}
run_frozen_test() {
  if [[ -z "${FROZEN_VALIDATION_SELECTION:-}" ]]; then
    echo "Set FROZEN_VALIDATION_SELECTION to the validation row/config you selected before running a test case." >&2
    exit 2
  fi
  run_case "frozen_test_${FROZEN_VALIDATION_SELECTION}" "${SLATE_SIZE}" "${QUOTA_MODE:-current}" \
    "${OMEGA_VALUES:-4,3,2,0.25}" "${KAPPA_VALUES:-1,0.9,0.8,0.15}" \
    "${LLM_WEIGHT:-$BASE_LW}" "${EVIDENCE_WEIGHT:-$BASE_EW}" "${SOURCE_WEIGHT:-$BASE_SW}" test
}

case "${MODE}" in
  quota) run_quota_grid ;;
  source) run_source_grid ;;
  weights) run_weight_grid ;;
  all) run_quota_grid; run_source_grid; run_weight_grid ;;
  frozen-test) run_frozen_test ;;
  *) echo "Unknown mode: ${MODE}; choose source, quota, weights, all, or frozen-test." >&2; exit 2 ;;
esac
echo "Validation grid complete: ${OUT_DIR}"
echo "Summary: ${SUMMARY}"
