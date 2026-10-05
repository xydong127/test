#!/usr/bin/env bash
set -euo pipefail

# Profile one-user-at-a-time inference without changing evaluate.py. Candidate
# construction, beam generation, title scoring, fusion, total evaluation time,
# and per-user scoring p50/p95 are recorded. Batch size is fixed at one so each
# title-scoring call corresponds to one recommendation request.
#
# Usage:
#   bash review_latency_profile.sh CATEGORY REVISE LLM RUN_NAME
# Rebuttal command:
#   bash review_latency_profile.sh All_Handmade 0 qwen17 YOUR_RUN_NAME
# Replace YOUR_RUN_NAME with the trained checkpoint run name.
# Optional environment: SCORE_BATCH_SIZE=16 MAX_LENGTH=1024 SLATE_SIZE=30

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
if [[ $# -lt 4 ]]; then
  echo "Usage: bash review_latency_profile.sh CATEGORY REVISE LLM RUN_NAME" >&2
  exit 2
fi
CATEGORY="$1" REVISE="$2" LLM="$3" RUN_NAME="$4"
CHECKPOINT_DIR="${ROOT}/checkpoints/${CATEGORY}/${RUN_NAME}"
[[ -d "${CHECKPOINT_DIR}" ]] || { echo "Checkpoint not found: ${CHECKPOINT_DIR}" >&2; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/review_results/latency/${CATEGORY}/${RUN_NAME}/${STAMP}"
mkdir -p "${OUT_DIR}"

python - "${CATEGORY}" "${REVISE}" "${LLM}" "${RUN_NAME}" "${OUT_DIR}" \
  "${SCORE_BATCH_SIZE:-16}" "${MAX_LENGTH:-1024}" "${SLATE_SIZE:-30}" <<'PY'
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

category, revise, llm, run_name, out_dir, score_batch_size, max_length, slate_size = sys.argv[1:]
revise = int(revise)
out_dir = Path(out_dir)
timings = {"beam_generation_seconds": [], "candidate_build_seconds": [],
           "score_seconds": [], "fusion_seconds": []}

import torch
import candidate_builder as cb
import evaluate
from utils import build_paths, load_json, save_json, summarize_metrics

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

original_beams = cb._generate_projected_beams
def timed_beams(*args, **kwargs):
    synchronize(); start = time.perf_counter()
    result = original_beams(*args, **kwargs)
    synchronize(); timings["beam_generation_seconds"].append(time.perf_counter() - start)
    return result
cb._generate_projected_beams = timed_beams

original_builder = evaluate.build_candidate_slates
def timed_builder(*args, **kwargs):
    synchronize(); start = time.perf_counter()
    result = original_builder(*args, **kwargs)
    synchronize(); timings["candidate_build_seconds"].append(time.perf_counter() - start)
    return result
evaluate.build_candidate_slates = timed_builder

original_score = evaluate.score_slates
def timed_score(model, batch_slates, *args, **kwargs):
    synchronize(); start = time.perf_counter()
    result = original_score(model, batch_slates, *args, **kwargs)
    synchronize(); elapsed = time.perf_counter() - start
    user_ids = [str(row.get("user_id", "")) for row in batch_slates]
    for user_id in user_ids:
        timings["score_seconds"].append({"user_id": user_id, "seconds": elapsed / max(len(user_ids), 1)})
    return result
evaluate.score_slates = timed_score

original_fuse = evaluate.fuse_grouped_scores
def timed_fuse(grouped_scores, batch_slates, *args, **kwargs):
    synchronize(); start = time.perf_counter()
    result = original_fuse(grouped_scores, batch_slates, *args, **kwargs)
    synchronize(); elapsed = time.perf_counter() - start
    timings["fusion_seconds"].append(elapsed / max(len(batch_slates), 1))
    return result
evaluate.fuse_grouped_scores = timed_fuse

sys.argv = [
    "evaluate.py", "--category", category, "--revise", str(revise), "--llm", llm,
    "--run_name", run_name, "--eval_batch_size", "1", "--score_batch_size", score_batch_size,
    "--max_length", max_length, "--slate_size", slate_size, "--force_rebuild_candidates",
]
evaluate.main()

paths = build_paths(category, revise, run_name=run_name)
eval_summary = json.loads(paths["eval_summary_json"].read_text(encoding="utf-8"))
prediction_rows = load_json(paths["output_json"])
quality = summarize_metrics(prediction_rows, topk_values=[1, 5, 10])
try:
    train_summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    train_seconds = train_summary.get("elapsed_seconds")
except FileNotFoundError:
    train_seconds = None

def percentile(values, pct):
    if not values: return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position); upper = math.ceil(position)
    if lower == upper: return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)

score_rows = timings["score_seconds"]
score_values = [row["seconds"] for row in score_rows]
fusion_values = timings["fusion_seconds"]
beam_total = sum(timings["beam_generation_seconds"])
candidate_total = sum(timings["candidate_build_seconds"])
summary = {
    "category": category, "revise": revise, "llm": llm, "run_name": run_name,
    "num_users": eval_summary.get("num_predictions", len(score_rows)),
    "eval_batch_size": 1, "score_batch_size": int(score_batch_size),
    "training_wall_seconds": train_seconds,
    "total_evaluation_wall_seconds": eval_summary.get("elapsed_seconds"),
    "candidate_construction_seconds": candidate_total,
    "beam_generation_seconds": beam_total,
    "mean_title_scoring_ms_per_user": statistics.mean(score_values) * 1000 if score_values else None,
    "p50_title_scoring_ms_per_user": percentile(score_values, 50) * 1000 if score_values else None,
    "p95_title_scoring_ms_per_user": percentile(score_values, 95) * 1000 if score_values else None,
    "mean_fusion_ms_per_user": statistics.mean(fusion_values) * 1000 if fusion_values else None,
    "HR1": quality["k1"]["hr"], "HR5": quality["k5"]["hr"],
    "HR10": quality["k10"]["hr"], "NDCG5": quality["k5"]["ndcg"],
    "NDCG10": quality["k10"]["ndcg"],
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
}
save_json(out_dir / "latency_summary.json", summary)
with (out_dir / "per_user_latency.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["user_id", "title_scoring_ms"])
    writer.writerows((row["user_id"], f"{row['seconds'] * 1000:.6f}") for row in score_rows)
print(json.dumps(summary, indent=2))
PY

echo "Latency profile saved: ${OUT_DIR}"
