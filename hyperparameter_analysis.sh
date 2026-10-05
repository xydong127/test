#!/usr/bin/env bash
set -euo pipefail

# Run the compact hyperparameter analysis for an existing Handmade checkpoint:
#   bash hyperparameter_analysis.sh All_Handmade 0 qwen17 YOUR_RUN_NAME quick
# Replace YOUR_RUN_NAME with the run name under checkpoints/All_Handmade.

cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
Usage:
  bash hyperparameter_analysis.sh CATEGORY REVISE LLM [RUN_NAME] [quick|full] [extra eval args...]

Examples:
  bash hyperparameter_analysis.sh All_Handmade 0 qwen17 latest quick

  bash hyperparameter_analysis.sh All_Handmade 0 qwen17 YOUR_RUN_NAME quick

  bash hyperparameter_analysis.sh All_Handmade 0 qwen17 YOUR_RUN_NAME \
    quick --eval_batch_size 1 --score_batch_size 4

Replace YOUR_RUN_NAME with a trained run under checkpoints/All_Handmade.

What this does:
  Evaluation-only hyperparameter analysis using an existing trained checkpoint.
  It does NOT retrain the LLM.

Swept hyperparameters in quick mode, designed for one 48GB GPU:
  1. wide_size:        30, 50, 100, 150
  2. slate_size:       10, 20, 30, 40
  3. transition_k:     0, 10, 40, 80
  4. llm_rank_weight:  0.00, 0.15, 0.35, 0.70

Full mode also uses four values per hyperparameter, but includes wider extremes:
  wide_size: 50, 100, 150, 250
  slate_size: 10, 20, 30, 50
  transition_k: 0, 40, 80, 120
  llm_rank_weight: 0.00, 0.35, 0.70, 1.00

Default anchor:
  wide_size=150, slate_size=30, transition_k=80, retrieval_k=40,
  ranker=fusion, llm_rank_weight=0.35, evidence_rank_weight=1.0,
  source_rank_weight=0.30.

Outputs:
  SLATE-Rec/hyperparams/<CATEGORY>/<RUN_NAME>/
    summary.tsv
    <case_tag>_metrics.json
    <case_tag>_eval_summary.json
    <case_tag>_predictions.json

Notes:
  - RUN_NAME may be "latest" to use checkpoints/<CATEGORY>/latest_run.txt.
  - MODE may be "quick" or "full"; quick is default and recommended for 1x48GB.
  - Because evaluate.py writes to the checkpoint run's normal output path,
    this script copies each case's files immediately after each run.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 3 ]]; then
  usage
  exit 1
fi

CATEGORY="$1"
REVISE="$2"
LLM="$3"
shift 3

RUN_NAME="latest"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  RUN_NAME="$1"
  shift
fi

MODE="quick"
if [[ $# -gt 0 && ( "${1:-}" == "quick" || "${1:-}" == "full" ) ]]; then
  MODE="$1"
  shift
fi

if [[ "${RUN_NAME}" == "latest" ]]; then
  LATEST_FILE="checkpoints/${CATEGORY}/latest_run.txt"
  if [[ ! -f "${LATEST_FILE}" ]]; then
    echo "Could not find ${LATEST_FILE}. Pass an explicit RUN_NAME." >&2
    exit 1
  fi
  RUN_NAME="$(tr -d '\r\n' < "${LATEST_FILE}")"
fi

if [[ -z "${RUN_NAME}" ]]; then
  echo "RUN_NAME is empty." >&2
  exit 1
fi

EXTRA_ARGS=("$@")

BASE_WIDE=150
BASE_SLATE=30
BASE_TRANSITION=80
BASE_RETRIEVAL=40
BASE_LLM_WEIGHT=0.35
BASE_EVIDENCE_WEIGHT=1.0
BASE_SOURCE_WEIGHT=0.30
BASE_PREFILTER="stratified"
BASE_RANKER="fusion"
BASE_TOPK=10

OUT_DIR="hyperparams/${CATEGORY}/${RUN_NAME}"
mkdir -p "${OUT_DIR}"
SUMMARY_TSV="${OUT_DIR}/summary.tsv"

echo -e "group\tcase\twide_size\tslate_size\ttransition_k\tretrieval_k\tllm_rank_weight\tevidence_rank_weight\tsource_rank_weight\tbeam_reach\tretrieval_reach\treservoir_reach\tdeploy_reach\tpreservation\tHR@1\tHR@5\tHR@10\tNDCG@5\tNDCG@10\tdeploy_jaccard_vs_default\tpred10_equal_vs_default\telapsed_seconds" > "${SUMMARY_TSV}"

echo "== SLATE-Rec hyperparameter analysis =="
echo "Category: ${CATEGORY}"
echo "Revise: ${REVISE}"
echo "LLM: ${LLM}"
echo "Checkpoint run: ${RUN_NAME}"
echo "Mode: ${MODE}"
echo "Output dir: ${OUT_DIR}"
echo "Tip for 1x48GB: pass --eval_batch_size 1 --score_batch_size 4 if scoring OOMs."
echo

python -u preprocess.py --category "${CATEGORY}" --revise "${REVISE}"

json_value() {
  local path="$1"
  local expr="$2"
  python - "$path" "$expr" <<'PY'
import json
import sys
path, expr = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
value = data
for key in expr.split("."):
    if key == "":
        continue
    value = value.get(key, None) if isinstance(value, dict) else None
    if value is None:
        break
if isinstance(value, float):
    print(f"{value:.8f}")
elif value is None:
    print("")
else:
    print(value)
PY
}

append_summary() {
  local group="$1"
  local tag="$2"
  local wide="$3"
  local slate="$4"
  local transition="$5"
  local retrieval="$6"
  local llm_weight="$7"
  local evidence_weight="$8"
  local source_weight="$9"

  local eval_path="${OUT_DIR}/${tag}_eval_summary.json"
  local metrics_path="${OUT_DIR}/${tag}_metrics.json"

  local beam retrieval_reach reservoir deploy preservation hr1 hr5 hr10 ndcg5 ndcg10 elapsed deploy_jaccard pred_equal
  beam="$(json_value "${eval_path}" "reachability.beam")"
  retrieval_reach="$(json_value "${eval_path}" "reachability.retrieval")"
  reservoir="$(json_value "${eval_path}" "reachability.reservoir")"
  deploy="$(json_value "${eval_path}" "reachability.deploy")"
  elapsed="$(json_value "${eval_path}" "elapsed_seconds")"
  preservation="$(python - "$reservoir" "$deploy" <<'PY'
import sys
reservoir = float(sys.argv[1] or 0.0)
deploy = float(sys.argv[2] or 0.0)
print(f"{(deploy / reservoir):.8f}" if reservoir > 0 else "")
PY
)"
  hr1="$(json_value "${metrics_path}" "results.k1.hr")"
  hr5="$(json_value "${metrics_path}" "results.k5.hr")"
  hr10="$(json_value "${metrics_path}" "results.k10.hr")"
  ndcg5="$(json_value "${metrics_path}" "results.k5.ndcg")"
  ndcg10="$(json_value "${metrics_path}" "results.k10.ndcg")"
  if [[ "${tag}" == "default" || ! -f "${OUT_DIR}/default_predictions.json" ]]; then
    deploy_jaccard="1.00000000"
    pred_equal="1.00000000"
  else
    read -r deploy_jaccard pred_equal < <(python - "${OUT_DIR}/default_predictions.json" "${OUT_DIR}/${tag}_predictions.json" <<'PY'
import json
import sys

def norm(text):
    return " ".join(str(text or "").strip().lower().split())

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    base = json.load(handle)
with open(sys.argv[2], "r", encoding="utf-8") as handle:
    cur = json.load(handle)

count = min(len(base), len(cur))
if count == 0:
    print("  ")
    raise SystemExit

jaccards = []
same_pred = 0
for left, right in zip(base[:count], cur[:count]):
    left_deploy = {norm(item) for item in left.get("deploy_candidates", [])}
    right_deploy = {norm(item) for item in right.get("deploy_candidates", [])}
    union = left_deploy | right_deploy
    if union:
        jaccards.append(len(left_deploy & right_deploy) / len(union))
    left_pred = [norm(item) for item in left.get("predict", [])[:10]]
    right_pred = [norm(item) for item in right.get("predict", [])[:10]]
    if left_pred == right_pred:
        same_pred += 1

mean_jaccard = sum(jaccards) / len(jaccards) if jaccards else 1.0
pred_rate = same_pred / count
print(f"{mean_jaccard:.8f} {pred_rate:.8f}")
PY
)"
  fi

  echo -e "${group}\t${tag}\t${wide}\t${slate}\t${transition}\t${retrieval}\t${llm_weight}\t${evidence_weight}\t${source_weight}\t${beam}\t${retrieval_reach}\t${reservoir}\t${deploy}\t${preservation}\t${hr1}\t${hr5}\t${hr10}\t${ndcg5}\t${ndcg10}\t${deploy_jaccard}\t${pred_equal}\t${elapsed}" >> "${SUMMARY_TSV}"
}

run_case() {
  local group="$1"
  local tag="$2"
  local wide="$3"
  local slate="$4"
  local transition="$5"
  local retrieval="$6"
  local llm_weight="$7"
  local evidence_weight="$8"
  local source_weight="$9"

  echo
  echo "== ${group}: ${tag} =="
  echo "wide=${wide}, slate=${slate}, transition=${transition}, retrieval=${retrieval}, llm_weight=${llm_weight}"

  python -u evaluate.py \
    --category "${CATEGORY}" \
    --revise "${REVISE}" \
    --llm "${LLM}" \
    --run_name "${RUN_NAME}" \
    --wide_size "${wide}" \
    --slate_size "${slate}" \
    --transition_k "${transition}" \
    --retrieval_k "${retrieval}" \
    --prefilter "${BASE_PREFILTER}" \
    --ranker "${BASE_RANKER}" \
    --llm_rank_weight "${llm_weight}" \
    --evidence_rank_weight "${evidence_weight}" \
    --source_rank_weight "${source_weight}" \
    --topk "${BASE_TOPK}" \
    "${EXTRA_ARGS[@]}"

  python -u metrics.py \
    --category "${CATEGORY}" \
    --revise "${REVISE}" \
    --llm "${LLM}" \
    --run_name "${RUN_NAME}" \
    --wide_size "${wide}" \
    --slate_size "${slate}" \
    --prefilter "${BASE_PREFILTER}" \
    --ranker "${BASE_RANKER}" \
    --llm_rank_weight "${llm_weight}" \
    --evidence_rank_weight "${evidence_weight}" \
    --source_rank_weight "${source_weight}" \
    --topk "${BASE_TOPK}" \
    "${EXTRA_ARGS[@]}"

  local safe_tag
  safe_tag="$(echo "${tag}" | tr '/ ' '__')"
  cp "outputs/${CATEGORY}/${RUN_NAME}.json" "${OUT_DIR}/${safe_tag}_predictions.json"
  cp "checkpoints/${CATEGORY}/${RUN_NAME}/eval_summary.json" "${OUT_DIR}/${safe_tag}_eval_summary.json"
  cp "metrics/${CATEGORY}/${CATEGORY}_${REVISE}_${RUN_NAME}.json" "${OUT_DIR}/${safe_tag}_metrics.json"
  append_summary "${group}" "${safe_tag}" "${wide}" "${slate}" "${transition}" "${retrieval}" "${llm_weight}" "${evidence_weight}" "${source_weight}"
}

# Anchor run.
run_case "anchor" "default" "${BASE_WIDE}" "${BASE_SLATE}" "${BASE_TRANSITION}" "${BASE_RETRIEVAL}" "${BASE_LLM_WEIGHT}" "${BASE_EVIDENCE_WEIGHT}" "${BASE_SOURCE_WEIGHT}"

if [[ "${MODE}" == "full" ]]; then
  WIDE_VALUES=(50 100 150 250)
  SLATE_VALUES=(10 20 30 50)
  TRANSITION_VALUES=(0 40 80 120)
  LLM_WEIGHT_VALUES=(0.00 0.35 0.70 1.00)
else
  WIDE_VALUES=(30 50 100 150)
  SLATE_VALUES=(10 20 30 40)
  TRANSITION_VALUES=(0 10 40 80)
  LLM_WEIGHT_VALUES=(0.00 0.15 0.35 0.70)
fi

# 1. Wide reservoir size.
for WIDE in "${WIDE_VALUES[@]}"; do
  [[ "${WIDE}" == "${BASE_WIDE}" ]] && continue
  run_case "wide_size" "wide-${WIDE}" "${WIDE}" "${BASE_SLATE}" "${BASE_TRANSITION}" "${BASE_RETRIEVAL}" "${BASE_LLM_WEIGHT}" "${BASE_EVIDENCE_WEIGHT}" "${BASE_SOURCE_WEIGHT}"
done

# 2. LLM-scored deployment slate size.
for SLATE in "${SLATE_VALUES[@]}"; do
  [[ "${SLATE}" == "${BASE_SLATE}" ]] && continue
  run_case "slate_size" "slate-${SLATE}" "${BASE_WIDE}" "${SLATE}" "${BASE_TRANSITION}" "${BASE_RETRIEVAL}" "${BASE_LLM_WEIGHT}" "${BASE_EVIDENCE_WEIGHT}" "${BASE_SOURCE_WEIGHT}"
done

# 3. Transition evidence budget.
for TRANSITION in "${TRANSITION_VALUES[@]}"; do
  [[ "${TRANSITION}" == "${BASE_TRANSITION}" ]] && continue
  run_case "transition_k" "transition-${TRANSITION}" "${BASE_WIDE}" "${BASE_SLATE}" "${TRANSITION}" "${BASE_RETRIEVAL}" "${BASE_LLM_WEIGHT}" "${BASE_EVIDENCE_WEIGHT}" "${BASE_SOURCE_WEIGHT}"
done

# 4. EOR fusion balance.
for LLM_WEIGHT in "${LLM_WEIGHT_VALUES[@]}"; do
  [[ "${LLM_WEIGHT}" == "${BASE_LLM_WEIGHT}" ]] && continue
  run_case "llm_rank_weight" "llm-weight-${LLM_WEIGHT}" "${BASE_WIDE}" "${BASE_SLATE}" "${BASE_TRANSITION}" "${BASE_RETRIEVAL}" "${LLM_WEIGHT}" "${BASE_EVIDENCE_WEIGHT}" "${BASE_SOURCE_WEIGHT}"
done

echo
echo "== Hyperparameter analysis complete =="
echo "Summary table: ${SUMMARY_TSV}"
echo "Per-case JSON files: ${OUT_DIR}/"
