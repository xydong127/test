#!/usr/bin/env bash
set -euo pipefail

# Run the full model and the three component ablations once each:
#   bash ablation.sh All_Handmade 0 qwen17
# Optional training/evaluation arguments can follow the three required values.

cd "$(dirname "$0")"

CATEGORY="${1:?category required}"
REVISE="${2:?revise required}"
LLM="${3:?llm required}"
shift 3
USER_ARGS=("$@")

python -u preprocess.py --category "${CATEGORY}" --revise "${REVISE}"

run_case() {
  local tag="$1"
  local description="$2"
  shift 2

  echo
  echo "== ${description} =="
  python -u train.py \
    --extra_tag "${tag}" \
    --category "${CATEGORY}" \
    --revise "${REVISE}" \
    --llm "${LLM}" \
    "${USER_ARGS[@]}" \
    "$@"
  python -u evaluate.py \
    --extra_tag "${tag}" \
    --category "${CATEGORY}" \
    --revise "${REVISE}" \
    --llm "${LLM}" \
    "${USER_ARGS[@]}" \
    "$@"
  python -u metrics.py \
    --extra_tag "${tag}" \
    --category "${CATEGORY}" \
    --revise "${REVISE}" \
    --llm "${LLM}" \
    "${USER_ARGS[@]}" \
    "$@"
}

run_case \
  "full-designed" \
  "Full design: TEF + SLA + EOR"

run_case \
  "no-tef" \
  "Remove TEF transition and lexical evidence; keep beam projection and popularity fill only" \
  --transition_k 0 \
  --retrieval_k 0 \
  --prefilter global

run_case \
  "no-eor" \
  "Remove EOR evidence calibration; rank by pure LLM likelihood" \
  --ranker llm

run_case \
  "no-sla" \
  "Skip SLA alignment after SFT" \
  --sla_epochs 0
