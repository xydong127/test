#!/usr/bin/env bash
set -euo pipefail

# Run preprocessing, training, evaluation, and metrics once:
#   bash run.sh All_Handmade 0 qwen17 --run_name handmade_full_s0 --seed 0

cd "$(dirname "$0")"

CATEGORY="${1:?category required}"
REVISE="${2:?revise required}"
LLM="${3:?llm required}"
shift 3

python -u preprocess.py --category "${CATEGORY}" --revise "${REVISE}"
python -u train.py --category "${CATEGORY}" --revise "${REVISE}" --llm "${LLM}" "$@"
python -u evaluate.py --category "${CATEGORY}" --revise "${REVISE}" --llm "${LLM}" "$@"
python -u metrics.py --category "${CATEGORY}" --revise "${REVISE}" --llm "${LLM}" "$@"
