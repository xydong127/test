#!/usr/bin/env bash
set -euo pipefail

# Evaluate an existing run and calculate its metrics:
#   bash evaluate.sh All_Handmade 0 qwen17 --run_name YOUR_RUN_NAME
# Replace YOUR_RUN_NAME with the run name under checkpoints/All_Handmade.

cd "$(dirname "$0")"

CATEGORY="${1:?category required}"
REVISE="${2:?revise required}"
LLM="${3:?llm required}"
shift 3

python -u evaluate.py --category "${CATEGORY}" --revise "${REVISE}" --llm "${LLM}" "$@"
python -u metrics.py --category "${CATEGORY}" --revise "${REVISE}" --llm "${LLM}" "$@"
