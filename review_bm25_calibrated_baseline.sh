#!/usr/bin/env bash
set -euo pipefail

# BM25 history-to-title retrieval followed by null-prompt-calibrated Qwen title
# scoring. The calibration alpha is selected on validation NDCG@10 and then a
# single frozen alpha is evaluated on test. Retrieval and slate construction
# are implemented in this added shell file's Python heredoc; project sources are
# not modified.
#
# Use an SFT-only checkpoint for SCORING_RUN_NAME so the baseline does not use
# SLA training. Train it first with run.sh ... --sla_epochs 0.
# Example preparation and evaluation commands:
#   bash run.sh All_Handmade 0 qwen17 --run_name handmade_sft_only_s0 --seed 0 --sla_epochs 0
#   bash review_bm25_calibrated_baseline.sh All_Handmade 0 qwen17 handmade_sft_only_s0
#
# Usage:
#   bash review_bm25_calibrated_baseline.sh CATEGORY REVISE LLM SCORING_RUN_NAME [alpha ...]
# Example:
#   bash review_bm25_calibrated_baseline.sh All_Handmade 0 qwen17 YOUR_SFT_RUN_NAME 0 0.5 1 1.5 2
# Minimal command using an existing SFT-only checkpoint:
#   bash review_bm25_calibrated_baseline.sh All_Handmade 0 qwen17 YOUR_SFT_RUN_NAME
# Replace YOUR_SFT_RUN_NAME with its directory name under checkpoints/All_Handmade.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
if [[ $# -lt 4 ]]; then
  echo "Usage: bash review_bm25_calibrated_baseline.sh CATEGORY REVISE LLM SCORING_RUN_NAME [alpha ...]" >&2
  exit 2
fi
CATEGORY="$1" REVISE="$2" LLM="$3" RUN_NAME="$4"
shift 4
if (($#)); then ALPHAS=("$@"); else read -r -a ALPHAS <<< "${ALPHA_GRID:-0 0.5 1 1.5 2}"; fi
CHECKPOINT_DIR="${ROOT}/checkpoints/${CATEGORY}/${RUN_NAME}"
[[ -d "${CHECKPOINT_DIR}" ]] || { echo "Checkpoint not found: ${CHECKPOINT_DIR}" >&2; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/review_results/bm25_calibrated/${CATEGORY}/${RUN_NAME}/${STAMP}"
mkdir -p "${OUT_DIR}"

python - "${CATEGORY}" "${REVISE}" "${LLM}" "${RUN_NAME}" "${OUT_DIR}" \
  "${SLATE_SIZE:-30}" "${WIDE_SIZE:-150}" "${MAX_LENGTH:-1024}" \
  "${EVAL_BATCH_SIZE:-2}" "${SCORE_BATCH_SIZE:-16}" "${ALPHAS[@]}" <<'PY'
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

category, revise, llm, run_name, out_dir, slate_size, wide_size, max_length, eval_batch, score_batch, *alphas = sys.argv[1:]
revise = int(revise)
slate_size, wide_size, max_length = int(slate_size), int(wide_size), int(max_length)
out_dir = Path(out_dir)
if not alphas:
    alphas = ["0", "0.5", "1", "1.5", "2"]
alphas = [float(x) for x in alphas]

import evaluate
from utils import build_paths, normalize_title, save_json, summarize_metrics, title_tokens

split_name = "valid"
original_paths = evaluate.build_paths
def split_paths(category_arg, revise_arg, run_name=None):
    paths = original_paths(category_arg, revise_arg, run_name=run_name)
    paths["test_csv"] = paths["valid_csv"] if split_name == "valid" else paths["test_csv"]
    return paths
evaluate.build_paths = split_paths

index_cache = {}
def bm25_slates(args, split_name, sources, train_sources, catalog_rows, model, tokenizer,
                device, include_oracle_positive, force_rebuild=False):
    if "index" not in index_cache:
        titles = [str(row["title"]) for row in catalog_rows]
        docs = [title_tokens(title) for title in titles]
        lengths = [len(doc) for doc in docs]
        avg_len = sum(lengths) / max(len(lengths), 1)
        postings = defaultdict(list)
        for doc_id, tokens in enumerate(docs):
            for token, tf in Counter(tokens).items():
                postings[token].append((doc_id, tf))
        idf = {token: math.log(1.0 + (len(docs) - len(rows) + 0.5) / (len(rows) + 0.5))
               for token, rows in postings.items()}
        train_popularity = Counter()
        for train_source in train_sources:
            train_popularity.update(int(item_id) for item_id in train_source.history_item_ids)
            train_popularity[int(train_source.output_item_id)] += 1
        popularity_order = sorted(
            range(len(catalog_rows)),
            key=lambda i: (train_popularity[int(catalog_rows[i]["item_id"])],
                           -int(catalog_rows[i]["item_id"])),
            reverse=True,
        )
        index_cache["index"] = (titles, docs, lengths, avg_len, postings, idf, popularity_order)
    titles, docs, lengths, avg_len, postings, idf, popularity_order = index_cache["index"]
    item_ids = [int(row["item_id"]) for row in catalog_rows]
    slates = []
    for source_index, source in enumerate(sources):
        query = Counter(title_tokens(" ".join(source.history_titles)))
        scores = defaultdict(float)
        for token, query_tf in query.items():
            entries = postings.get(token, ())
            token_idf = idf.get(token, 0.0)
            for doc_id, tf in entries:
                dl = lengths[doc_id]
                denom = tf + 1.5 * (1.0 - 0.75 + 0.75 * dl / max(avg_len, 1.0))
                scores[doc_id] += token_idf * (tf * 2.5 / max(denom, 1e-12)) * (1.0 + math.log(query_tf))
        history = {normalize_title(x) for x in source.history_titles}
        ranked = sorted(scores, key=lambda i: scores[i], reverse=True)
        ranked = [i for i in ranked if normalize_title(titles[i]) not in history]
        bm25_chosen = ranked[:wide_size]
        chosen = list(bm25_chosen)
        if len(chosen) < wide_size:
            present = set(chosen)
            for i in popularity_order:
                if i not in present and normalize_title(titles[i]) not in history:
                    chosen.append(i)
                    present.add(i)
                    if len(chosen) >= wide_size:
                        break
        deploy_ids = chosen[:max(slate_size, int(args.topk))]
        target = normalize_title(source.output_title)
        reservoir = []
        bm25_set = set(bm25_chosen)
        bm25_rank = {doc_id: rank for rank, doc_id in enumerate(bm25_chosen, 1)}
        fill_rank = 0
        for rank, i in enumerate(chosen, 1):
            score = float(scores.get(i, 0.0))
            source_name = "retrieval" if i in bm25_set else "fill"
            if source_name == "fill":
                fill_rank += 1
            source_rank = bm25_rank.get(i, fill_rank)
            reservoir.append({
                "title": titles[i], "item_id": item_ids[i], "evidence_score": score,
                "sources": [source_name], "source_ranks": {source_name: source_rank},
                "beam": False, "transition": False, "retrieval": source_name == "retrieval",
                "fill": source_name == "fill", "oracle": False,
                "gain": 1.0 if normalize_title(titles[i]) == target else 0.0,
            })
        deploy = [dict(row) for row in reservoir[:len(deploy_ids)]]
        deploy_titles = {normalize_title(row["title"]) for row in deploy}
        target_bm25 = target in {normalize_title(titles[i]) for i in bm25_chosen}
        target_reservoir = target in {normalize_title(row["title"]) for row in reservoir}
        target_deploy = target in deploy_titles
        slates.append({
            "source_index": source_index, "user_id": source.user_id,
            "prompt": source.prompt, "prompt_ids": list(source.prompt_ids),
            "history_titles": list(source.history_titles),
            "history_item_ids": list(source.history_item_ids),
            "output_title": source.output_title, "output_item_id": int(source.output_item_id),
            "reservoir_candidates": reservoir, "deploy_candidates": deploy,
            "train_candidates": [dict(row) for row in deploy],
            "beam_reachable": False, "retrieval_reachable": target_bm25,
            "reservoir_reachable": target_reservoir, "deploy_reachable": target_deploy,
            "oracle_injected": False,
        })
    return slates
tef_builder = evaluate.build_candidate_slates
evaluate.build_candidate_slates = bm25_slates

def run_eval(split, alpha, method="bm25"):
    global split_name
    split_name = split
    evaluate.build_candidate_slates = bm25_slates if method == "bm25" else tef_builder
    sys.argv = [
        "evaluate.py", "--category", category, "--revise", str(revise), "--llm", llm,
        "--run_name", run_name, "--ranker", "llm", "--null_alpha", str(alpha),
        "--slate_size", str(slate_size), "--wide_size", str(wide_size),
        "--max_length", str(max_length), "--eval_batch_size", eval_batch,
        "--score_batch_size", score_batch, "--prefilter", "stratified",
        "--force_rebuild_candidates",
    ]
    evaluate.main()
    paths = original_paths(category, revise, run_name=run_name)
    predictions = json.loads(paths["output_json"].read_text(encoding="utf-8"))
    evaluation = json.loads(paths["eval_summary_json"].read_text(encoding="utf-8"))
    metrics = summarize_metrics(predictions, topk_values=[1, 5, 10])
    exposed = [row for row in predictions if row.get("deploy_reachable")]
    def conditional_hit(k):
        return (sum(normalize_title(row.get("output")) in {normalize_title(x) for x in row.get("predict", [])[:k]}
                    for row in exposed) / len(exposed)) if exposed else None
    return predictions, evaluation, metrics, conditional_hit

summary_path = out_dir / "validation_alpha_grid.tsv"
val_rows = []
with summary_path.open("w", encoding="utf-8", newline="") as f:
    csv.writer(f, delimiter="\t").writerow([
        "method", "split", "alpha", "reservoir_recall", "deploy_recall", "conditional_HR5",
        "conditional_HR10", "HR5", "HR10", "NDCG10", "eval_seconds",
    ])
for alpha in alphas:
    preds, evaluation, metrics, conditional_hit = run_eval("valid", alpha, "bm25")
    case_dir = out_dir / f"validation_alpha_{alpha:g}"
    case_dir.mkdir(parents=True, exist_ok=True)
    save_json(case_dir / "predictions.json", preds)
    save_json(case_dir / "eval_summary.json", evaluation)
    row = ["BM25", "valid", alpha, evaluation.get("reachability", {}).get("reservoir", ""),
           evaluation.get("reachability", {}).get("deploy", ""), conditional_hit(5), conditional_hit(10),
           metrics["k5"]["hr"], metrics["k10"]["hr"], metrics["k10"]["ndcg"],
           evaluation.get("elapsed_seconds", "")]
    val_rows.append((alpha, float(metrics["k10"]["ndcg"])))
    with summary_path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(row)

best_alpha = max(val_rows, key=lambda x: (x[1], -x[0]))[0]
save_json(out_dir / "selected_alpha.json", {"criterion": "validation NDCG@10", "alpha": best_alpha,
                                             "validation_grid": val_rows})
for method in ("bm25", "tef"):
    preds, evaluation, metrics, conditional_hit = run_eval("test", best_alpha, method)
    test_dir = out_dir / f"test_{method}_frozen_alpha"
    test_dir.mkdir(parents=True, exist_ok=True)
    save_json(test_dir / "predictions.json", preds)
    save_json(test_dir / "eval_summary.json", evaluation)
    save_json(test_dir / "metrics.json", {
        "method": method, "alpha": best_alpha, "results": metrics,
        "conditional_HR5": conditional_hit(5), "conditional_HR10": conditional_hit(10),
        "reservoir_recall": evaluation.get("reachability", {}).get("reservoir"),
        "deploy_recall": evaluation.get("reachability", {}).get("deploy"),
    })
    with summary_path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow([
            method.upper(), "test_frozen", best_alpha,
            evaluation.get("reachability", {}).get("reservoir", ""),
            evaluation.get("reachability", {}).get("deploy", ""), conditional_hit(5), conditional_hit(10),
            metrics["k5"]["hr"], metrics["k10"]["hr"], metrics["k10"]["ndcg"],
            evaluation.get("elapsed_seconds", "")])
print(f"Selected validation alpha={best_alpha:g}; frozen test results saved for BM25 and TEF with the same SFT checkpoint.")
PY

echo "BM25 retrieval + calibrated scoring complete: ${OUT_DIR}"
