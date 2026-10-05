#!/usr/bin/env bash
set -euo pipefail

# Evaluate one trained checkpoint with the raw LLM score and the configured
# fusion score, then report target reachability and exact ranks from the full
# candidate-score maps. The first evaluation rebuilds candidate slates; the
# second reuses them so both rankers see the same candidates.
#
# Usage:
#   bash review_stage_flow_audit.sh CATEGORY REVISE LLM RUN_NAME [EVAL_ARGS...]
# Example:
#   bash review_stage_flow_audit.sh All_Handmade 0 qwen17 YOUR_RUN_NAME --score_batch_size 8
# Run once per domain to report deployment-slate recall and conditional ranking:
#   bash review_stage_flow_audit.sh All_Handmade 0 qwen17 YOUR_RUN_NAME

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

if [[ $# -lt 4 ]]; then
  echo "Usage: bash review_stage_flow_audit.sh CATEGORY REVISE LLM RUN_NAME [EVAL_ARGS...]" >&2
  exit 2
fi

CATEGORY="$1"
REVISE="$2"
LLM="$3"
RUN_NAME="$4"
shift 4
EVAL_ARGS=("$@")

CHECKPOINT_DIR="${ROOT}/checkpoints/${CATEGORY}/${RUN_NAME}"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint run not found: ${CHECKPOINT_DIR}" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/review_results/stage_flow/${CATEGORY}/${RUN_NAME}/${STAMP}"
mkdir -p "${OUT_DIR}"

run_ranker() {
  local RANKER="$1"
  local REBUILD="$2"
  local CASE_ARGS=(
    --category "${CATEGORY}"
    --revise "${REVISE}"
    --llm "${LLM}"
    --run_name "${RUN_NAME}"
    --ranker "${RANKER}"
    --topk 10
  )
  if [[ "${REBUILD}" == "yes" ]]; then
    CASE_ARGS+=(--force_rebuild_candidates)
  fi
  CASE_ARGS+=("${EVAL_ARGS[@]}")

  echo "== Evaluate ranker=${RANKER}, rebuild_candidates=${REBUILD} =="
  python -u evaluate.py "${CASE_ARGS[@]}"
  python -u metrics.py --category "${CATEGORY}" --revise "${REVISE}" --llm "${LLM}" --run_name "${RUN_NAME}" --topk 10

  cp "${ROOT}/outputs/${CATEGORY}/${RUN_NAME}.json" "${OUT_DIR}/${RANKER}_predictions.json"
  cp "${ROOT}/checkpoints/${CATEGORY}/${RUN_NAME}/eval_summary.json" "${OUT_DIR}/${RANKER}_eval_summary.json"
  cp "${ROOT}/metrics/${CATEGORY}/${CATEGORY}_${REVISE}_${RUN_NAME}.json" "${OUT_DIR}/${RANKER}_metrics.json"
}

run_ranker fusion yes
run_ranker llm no

python - "${OUT_DIR}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])

def read_json(name):
    with (out_dir / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)

raw = read_json("llm_predictions.json")
final = read_json("fusion_predictions.json")
if len(raw) != len(final):
    raise SystemExit(f"Prediction count mismatch: raw={len(raw)}, fusion={len(final)}")

def norm(value):
    return " ".join(str(value or "").casefold().split())

def rank_of(row):
    target = norm(row.get("output"))
    scores = row.get("candidate_scores", {})
    matches = [(title, score) for title, score in scores.items() if norm(title) == target]
    if not matches:
        return None
    target_score = float(matches[0][1])
    # The score map contains all deployment candidates, including targets
    # below rank 10; it allows the audit to recover their full slate rank.
    return 1 + sum(float(score) > target_score for score in scores.values())

def rank_band(rank):
    if rank is None:
        return "absent"
    if rank == 1:
        return "1"
    if rank <= 5:
        return "2-5"
    if rank <= 10:
        return "6-10"
    return ">10"

detail_path = out_dir / "per_target_ranks.tsv"
with detail_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow([
        "user_id", "target", "reservoir_reachable", "deploy_reachable",
        "raw_llm_rank", "raw_llm_band", "final_score_rank", "final_score_band",
    ])
    for raw_row, final_row in zip(raw, final):
        if str(raw_row.get("user_id")) != str(final_row.get("user_id")) or norm(raw_row.get("output")) != norm(final_row.get("output")):
            raise SystemExit("Raw and fusion rows are not aligned by user/target order")
        raw_rank = rank_of(raw_row)
        final_rank = rank_of(final_row)
        writer.writerow([
            raw_row.get("user_id"), raw_row.get("output"),
            raw_row.get("reservoir_reachable"), raw_row.get("deploy_reachable"),
            raw_rank if raw_rank is not None else "", rank_band(raw_rank),
            final_rank if final_rank is not None else "", rank_band(final_rank),
        ])

summary_path = out_dir / "rank_flow_counts.tsv"
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow([
        "ranker", "targets", "reservoir_recall", "deploy_recall", "rank_1",
        "rank_2_5", "rank_6_10", "rank_gt_10", "rank_absent",
        "HR@5", "HR@10",
    ])
    for ranker, filename in (("llm", "llm_predictions.json"), ("fusion", "fusion_predictions.json")):
        rows = read_json(filename)
        total = len(rows)
        ranks = [rank_of(row) for row in rows]
        counts = {band: sum(rank_band(rank) == band for rank in ranks) for band in ("1", "2-5", "6-10", ">10", "absent")}
        reservoir_recall = sum(bool(row.get("reservoir_reachable")) for row in rows) / total if total else math.nan
        deploy_recall = sum(bool(row.get("deploy_reachable")) for row in rows) / total if total else math.nan
        hr5 = sum(rank is not None and rank <= 5 for rank in ranks) / total if total else math.nan
        hr10 = sum(rank is not None and rank <= 10 for rank in ranks) / total if total else math.nan
        writer.writerow([
            ranker, total, f"{reservoir_recall:.8f}", f"{deploy_recall:.8f}",
            counts["1"], counts["2-5"], counts["6-10"], counts[">10"], counts["absent"],
            f"{hr5:.8f}", f"{hr10:.8f}",
        ])

print(f"Saved per-target ranks: {detail_path}")
print(f"Saved rank-flow counts: {summary_path}")
PY

echo "Stage-flow audit files: ${OUT_DIR}"
