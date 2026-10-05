#!/usr/bin/env bash
set -euo pipefail

# Compare alternative SLA training slates with the current append-positive
# condition. By default, run only the two alternatives and reuse the matching
# full-SLA/append-positive result from review_sla_loss_factorial.sh as reference.
# Set MISMATCH_VARIANTS="append replace reachable_only" for a standalone run.
# candidate_builder.py is patched in memory for the training subprocess only.
#
# Usage:
#   bash review_sla_distribution_mismatch.sh CATEGORY REVISE LLM SEED [-- TRAIN_ARGS...]
# Example:
#   bash review_sla_distribution_mismatch.sh All_Handmade 0 qwen17 0 -- --micro_batch_size 1 --gradient_accumulation_steps 16
# Optional environment: MISMATCH_VARIANTS="replace reachable_only" (default)
# Minimal rebuttal command (seed 0; two alternatives, append-positive reference reused):
#   bash review_sla_distribution_mismatch.sh All_Handmade 0 qwen17 0

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
if [[ $# -lt 4 ]]; then
  echo "Usage: bash review_sla_distribution_mismatch.sh CATEGORY REVISE LLM SEED [-- TRAIN_ARGS...]" >&2
  exit 2
fi
CATEGORY="$1" REVISE="$2" LLM="$3" SEED="$4"
shift 4
if (($#)); then [[ "$1" == "--" ]] || { echo "Use -- before extra training arguments." >&2; exit 2; }; shift; fi
EXTRA_ARGS=("$@")

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/review_results/sla_distribution_mismatch/${CATEGORY}/seed-${SEED}/${STAMP}"
mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.tsv"
printf 'variant\trun_name\tall_train_examples\tselected_train_examples\tselected_fraction\tnaturally_reachable_fraction\tpositive_injected_fraction\tmean_train_slate_size\ttest_deploy_recall\ttest_conditional_HR5\ttest_conditional_HR10\ttest_HR5\ttest_HR10\ttest_NDCG10\ttrain_seconds\teval_seconds\n' > "${SUMMARY}"

run_case() {
  local VARIANT="$1"
  local RUN_NAME="review_mismatch_${CATEGORY}_seed-${SEED}_${VARIANT}_${STAMP}"
  local STATS_PATH="${OUT_DIR}/${VARIANT}_train_slate_stats.json"
  echo "== SLA candidate policy: ${VARIANT}; run=${RUN_NAME} =="

  python - "${VARIANT}" "${STATS_PATH}" "${CATEGORY}" "${REVISE}" "${LLM}" "${RUN_NAME}" "${SEED}" "${EXTRA_ARGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

variant, stats_path, category, revise, llm, run_name, seed, *extra = sys.argv[1:]
if variant not in ("append", "replace", "reachable_only"):
    raise SystemExit(f"Unknown training policy: {variant}")

import candidate_builder as cb
original_finish = cb._finish_slate
def modified_finish(*args, **kwargs):
    if variant == "append":
        row = original_finish(*args, **kwargs)
    elif variant == "replace":
        row = original_finish(*args, **kwargs)
        if kwargs.get("include_oracle_positive") and row.get("oracle_injected"):
            train_candidates = list(row["train_candidates"])
            positive = train_candidates.pop()
            if not train_candidates:
                raise RuntimeError("Cannot replace a negative in an empty deployment slate")
            lowest = min(range(len(train_candidates)),
                         key=lambda i: float(train_candidates[i].get("evidence_score", 0.0)))
            train_candidates[lowest] = positive
            row["train_candidates"] = train_candidates
            row["training_target_replaced"] = True
    else:
        call_kwargs = dict(kwargs)
        if call_kwargs.get("include_oracle_positive"):
            call_kwargs["include_oracle_positive"] = False
        row = original_finish(*args, **call_kwargs)
    return row
cb._finish_slate = modified_finish

original_builder = cb.build_candidate_slates
def observed_builder(*args, **kwargs):
    rows = original_builder(*args, **kwargs)
    if kwargs.get("split_name") == "train":
        natural = sum(bool(row.get("deploy_reachable")) for row in rows)
        injected = sum(bool(row.get("oracle_injected")) for row in rows)
        mean_size = (sum(len(row.get("train_candidates", [])) for row in rows) / len(rows)) if rows else 0.0
        payload = {
            "policy": variant,
            "all_train_examples": len(rows),
            "naturally_reachable_examples": natural,
            "naturally_reachable_fraction": natural / len(rows) if rows else 0.0,
            "positive_injected_examples": injected,
            "positive_injected_fraction": injected / len(rows) if rows else 0.0,
            "selected_train_examples": len(rows),
            "selected_fraction": 1.0 if rows else 0.0,
            "mean_train_slate_size": mean_size,
        }
        if variant == "reachable_only":
            rows = [row for row in rows if row.get("deploy_reachable")]
            payload["selected_train_examples"] = len(rows)
            payload["selected_fraction"] = len(rows) / payload["all_train_examples"] if payload["all_train_examples"] else 0.0
            if not rows:
                raise RuntimeError("No naturally reachable training positives; reachable_only condition is empty")
            payload["mean_train_slate_size"] = (
                sum(len(row.get("train_candidates", [])) for row in rows) / len(rows) if rows else 0.0
            )
        Path(stats_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return rows
cb.build_candidate_slates = observed_builder

import train
sys.argv = [
    "train.py", "--category", category, "--revise", revise, "--llm", llm,
    "--run_name", run_name, "--seed", seed, "--force_rebuild_candidates",
    "--disable_auto_skip_sla", *extra,
]
train.main()
PY

  python -u evaluate.py --category "${CATEGORY}" --revise "${REVISE}" --llm "${LLM}" \
    --run_name "${RUN_NAME}" --seed "${SEED}" --force_rebuild_candidates
  python -u metrics.py --category "${CATEGORY}" --revise "${REVISE}" --llm "${LLM}" --run_name "${RUN_NAME}" --topk 10

  cp "${ROOT}/checkpoints/${CATEGORY}/${RUN_NAME}/run_summary.json" "${OUT_DIR}/${VARIANT}_run_summary.json"
  cp "${ROOT}/checkpoints/${CATEGORY}/${RUN_NAME}/eval_summary.json" "${OUT_DIR}/${VARIANT}_eval_summary.json"
  cp "${ROOT}/outputs/${CATEGORY}/${RUN_NAME}.json" "${OUT_DIR}/${VARIANT}_predictions.json"
  cp "${ROOT}/metrics/${CATEGORY}/${CATEGORY}_${REVISE}_${RUN_NAME}.json" "${OUT_DIR}/${VARIANT}_metrics.json"

  python - "${SUMMARY}" "${VARIANT}" "${RUN_NAME}" "${STATS_PATH}" \
    "${OUT_DIR}/${VARIANT}_run_summary.json" "${OUT_DIR}/${VARIANT}_eval_summary.json" \
    "${OUT_DIR}/${VARIANT}_metrics.json" "${OUT_DIR}/${VARIANT}_predictions.json" <<'PY'
import csv
import json
import sys

summary, variant, run_name, stats_path, train_path, eval_path, metrics_path, pred_path = sys.argv[1:]
with open(stats_path, "r", encoding="utf-8") as f: train_stats = json.load(f)
with open(train_path, "r", encoding="utf-8") as f: train_summary = json.load(f)
with open(eval_path, "r", encoding="utf-8") as f: evaluation = json.load(f)
with open(metrics_path, "r", encoding="utf-8") as f: metrics = json.load(f)["results"]
with open(pred_path, "r", encoding="utf-8") as f: rows = json.load(f)

def norm(x): return " ".join(str(x or "").casefold().split())
exposed = [row for row in rows if row.get("deploy_reachable")]
def cond_hit(k):
    return sum(norm(row.get("output")) in {norm(x) for x in row.get("predict", [])[:k]}
               for row in exposed) / len(exposed) if exposed else ""

with open(summary, "a", encoding="utf-8", newline="") as f:
    csv.writer(f, delimiter="\t").writerow([
        variant, run_name, train_stats.get("all_train_examples", ""),
        train_stats.get("selected_train_examples", ""), train_stats.get("selected_fraction", ""),
        train_stats.get("naturally_reachable_fraction", ""),
        train_stats.get("positive_injected_fraction", ""),
        train_stats.get("mean_train_slate_size", ""),
        evaluation.get("reachability", {}).get("deploy", ""), cond_hit(5), cond_hit(10),
        metrics["k5"]["hr"], metrics["k10"]["hr"], metrics["k10"]["ndcg"],
        train_summary.get("elapsed_seconds", ""), evaluation.get("elapsed_seconds", ""),
    ])
PY
}

read -r -a VARIANT_LIST <<< "${MISMATCH_VARIANTS:-replace reachable_only}"
for VARIANT in "${VARIANT_LIST[@]}"; do
  case "${VARIANT}" in
    append|replace|reachable_only) run_case "${VARIANT}" ;;
    *) echo "Unknown mismatch variant: ${VARIANT}; choose append, replace, or reachable_only." >&2; exit 2 ;;
  esac
done
echo "Train/inference slate experiment complete: ${OUT_DIR}"
echo "Summary: ${SUMMARY}"
