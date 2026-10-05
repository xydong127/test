import argparse
import time

from utils import build_paths, choose_run_name, format_seconds, load_json, log, read_latest_run_name, save_json, summarize_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Compute HR/NDCG metrics from SLATE-Rec outputs.")
    parser.add_argument("--category", required=True, type=str)
    parser.add_argument("--revise", required=True, type=int)
    parser.add_argument("--llm", required=True, type=str)
    parser.add_argument("--run_name", default="", type=str)
    parser.add_argument("--extra_tag", default="", type=str)
    parser.add_argument("--force_new_run", action="store_true")
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--sft_epochs", default=1, type=int)
    parser.add_argument("--sla_epochs", default=1, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--micro_batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--wide_size", default=150, type=int)
    parser.add_argument("--slate_size", default=30, type=int)
    parser.add_argument("--prefilter", choices=["stratified", "global"], default="stratified")
    parser.add_argument("--null_alpha", default=0.0, type=float)
    parser.add_argument("--ranker", choices=["fusion", "llm", "evidence"], default="fusion")
    parser.add_argument("--llm_rank_weight", default=0.35, type=float)
    parser.add_argument("--evidence_rank_weight", default=1.0, type=float)
    parser.add_argument("--source_rank_weight", default=0.30, type=float)
    parser.add_argument("--freeze_after_sft", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        log(f"Ignoring unknown metrics.py arguments: {' '.join(unknown)}")
    return args


def main():
    args = parse_args()
    started = time.time()
    run_name = choose_run_name(args)
    if args.force_new_run and not args.run_name:
        latest_run = read_latest_run_name(args.category)
        if latest_run:
            run_name = latest_run
    paths = build_paths(args.category, args.revise, run_name=run_name)
    predictions = load_json(paths["output_json"])
    metrics = summarize_metrics(predictions, topk_values=[1, 5, 10])
    result = {
        "framework": "SLATE-Rec",
        "category": args.category,
        "revise": args.revise,
        "run_name": run_name,
        "results": metrics,
        "num_queries": len(predictions),
        "topk": 10,
        "elapsed_seconds": time.time() - started,
    }
    save_json(paths["metrics_json"], result)
    log(
        "Metrics "
        f"HR@1={metrics['k1']['hr']:.4f}, HR@5={metrics['k5']['hr']:.4f}, HR@10={metrics['k10']['hr']:.4f}, "
        f"NDCG@1={metrics['k1']['ndcg']:.4f}, NDCG@5={metrics['k5']['ndcg']:.4f}, NDCG@10={metrics['k10']['ndcg']:.4f}"
    )
    log(f"Saved metrics to {paths['metrics_json']} in {format_seconds(result['elapsed_seconds'])}")


if __name__ == "__main__":
    main()
