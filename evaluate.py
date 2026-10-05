import argparse
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from candidate_builder import build_candidate_slates, reachability_summary
from data import build_sources
from dataset import CandidateSlateDataset, SlateCollator, build_null_prompt
from llm_candidate_scoring import fuse_grouped_scores, score_slates
from preprocess import preprocess_category
from utils import (
    autocast_context,
    build_paths,
    choose_run_name,
    filter_topk_unique,
    format_seconds,
    get_model_dtype,
    humanize_category,
    latest_existing_model_dir,
    load_json,
    log,
    read_latest_run_name,
    resolve_model_name,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SLATE-Rec with TEF slates and EOR reranking.")
    parser.add_argument("--category", required=True, type=str)
    parser.add_argument("--revise", required=True, type=int)
    parser.add_argument("--llm", required=True, type=str)
    parser.add_argument("--run_name", default="", type=str)
    parser.add_argument("--extra_tag", default="", type=str)
    parser.add_argument("--force_new_run", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--micro_batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=2, type=int)
    parser.add_argument("--score_batch_size", default=16, type=int)
    parser.add_argument("--sft_epochs", default=1, type=int)
    parser.add_argument("--sla_epochs", default=1, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--sla_learning_rate", default=1e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--freeze_after_sft", action="store_true")
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--beam_width", default=10, type=int)
    parser.add_argument("--beam_k", default=10, type=int)
    parser.add_argument("--max_new_tokens", default=32, type=int)
    parser.add_argument("--wide_size", default=150, type=int)
    parser.add_argument("--slate_size", default=30, type=int)
    parser.add_argument("--prefilter", choices=["stratified", "global"], default="stratified")
    parser.add_argument("--transition_k", default=80, type=int)
    parser.add_argument("--transition_recent_k", default=5, type=int)
    parser.add_argument("--retrieval_k", default=40, type=int)
    parser.add_argument("--fill_k", default=200, type=int)
    parser.add_argument("--length_gamma", default=1.0, type=float)
    parser.add_argument("--null_alpha", default=0.0, type=float)
    parser.add_argument("--ranker", choices=["fusion", "llm", "evidence"], default="fusion")
    parser.add_argument("--llm_rank_weight", default=0.35, type=float)
    parser.add_argument("--evidence_rank_weight", default=1.0, type=float)
    parser.add_argument("--source_rank_weight", default=0.30, type=float)
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--force_rebuild_candidates", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        log(f"Ignoring unknown evaluate.py arguments: {' '.join(unknown)}")
    return args


def main():
    args = parse_args()
    started = time.time()
    set_seed(args.seed)
    preprocess_category(args.category, args.revise)

    run_name = choose_run_name(args)
    if args.force_new_run and not args.run_name:
        latest_run = read_latest_run_name(args.category)
        if latest_run:
            run_name = latest_run
    args.run_name = run_name
    paths = build_paths(args.category, args.revise, run_name=run_name)

    checkpoint_dir = latest_existing_model_dir(paths)
    base_model_name = resolve_model_name(args.llm)
    model_name_or_path = str(checkpoint_dir) if checkpoint_dir else base_model_name
    if checkpoint_dir:
        log(f"Loading checkpoint from {checkpoint_dir}")
    else:
        log("No SLATE-Rec checkpoint found; using base model for both generation and scoring.")

    dtype = get_model_dtype(prefer_bf16=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    category_text = humanize_category(args.category)
    train_sources = build_sources(paths["train_csv"], tokenizer, category_text, args.max_length)
    test_sources = build_sources(paths["test_csv"], tokenizer, category_text, args.max_length)
    catalog_rows = load_json(paths["catalog_json"])["items"]

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    test_slates = build_candidate_slates(
        args,
        split_name="test",
        sources=test_sources,
        train_sources=train_sources,
        catalog_rows=catalog_rows,
        model=model,
        tokenizer=tokenizer,
        device=device,
        include_oracle_positive=False,
        force_rebuild=args.force_rebuild_candidates,
    )
    reachability = reachability_summary(test_slates)
    dataset = CandidateSlateDataset(test_slates, tokenizer, max_length=args.max_length, eos_token_id=tokenizer.eos_token_id)
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=SlateCollator())
    null_prompt_ids = None
    if args.null_alpha > 0:
        null_prompt_ids = tokenizer(build_null_prompt(category_text), add_special_tokens=True).input_ids

    predictions = []
    with torch.no_grad():
        for batch_slates in tqdm(loader, desc="SLATE-Rec eval", leave=False):
            with autocast_context(dtype):
                grouped_scores, titles_by_slate = score_slates(
                    model,
                    batch_slates,
                    max_length=args.max_length,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                    length_gamma=args.length_gamma,
                    max_score_batch_size=args.score_batch_size,
                    null_prompt_ids=null_prompt_ids,
                    null_alpha=args.null_alpha,
                )
                ranked_scores = fuse_grouped_scores(
                    grouped_scores,
                    batch_slates,
                    ranker=args.ranker,
                    llm_weight=args.llm_rank_weight,
                    evidence_weight=args.evidence_rank_weight,
                    source_weight=args.source_rank_weight,
                )
            for slate, scores, titles in zip(batch_slates, ranked_scores, titles_by_slate):
                order = torch.argsort(scores.detach().float().cpu(), descending=True).tolist()
                ranked = [titles[index] for index in order]
                ranked = filter_topk_unique(ranked, args.topk)
                if len(ranked) < args.topk:
                    fill_titles = [candidate["title"] for candidate in slate["candidate_meta"]]
                    ranked = filter_topk_unique(ranked + fill_titles, args.topk)
                score_map = {
                    titles[index]: float(scores[index].detach().float().cpu().item())
                    for index in range(len(titles))
                }
                predictions.append(
                    {
                        "user_id": slate["user_id"],
                        "prompt": slate["prompt"],
                        "history_titles": test_slates[slate["source_index"]]["history_titles"],
                        "output": slate["output_title"],
                        "predict": ranked[: args.topk],
                        "candidate_scores": score_map,
                        "ranker": args.ranker,
                        "deploy_candidates": titles,
                        "beam_reachable": bool(test_slates[slate["source_index"]].get("beam_reachable")),
                        "retrieval_reachable": bool(test_slates[slate["source_index"]].get("retrieval_reachable")),
                        "reservoir_reachable": bool(test_slates[slate["source_index"]].get("reservoir_reachable")),
                        "deploy_reachable": bool(test_slates[slate["source_index"]].get("deploy_reachable")),
                        "checkpoint": str(checkpoint_dir) if checkpoint_dir else base_model_name,
                    }
                )

    save_json(paths["output_json"], predictions)
    summary = {
        "framework": "SLATE-Rec",
        "category": args.category,
        "revise": args.revise,
        "run_name": run_name,
        "checkpoint": str(checkpoint_dir) if checkpoint_dir else base_model_name,
        "topk": args.topk,
        "num_predictions": len(predictions),
        "reachability": reachability,
        "elapsed_seconds": time.time() - started,
    }
    save_json(paths["eval_summary_json"], summary)
    log(
        f"Saved {len(predictions)} prediction rows to {paths['output_json']} "
        f"in {format_seconds(summary['elapsed_seconds'])}; reachability={reachability}"
    )


if __name__ == "__main__":
    main()
