import argparse
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from candidate_builder import build_candidate_slates, reachability_summary
from candidate_closed_loss import candidate_closed_loss
from data import build_sources
from dataset import CandidateSlateDataset, SFTDataset, SlateCollator, TrainCollator, TrainingEntry, build_null_prompt
from llm_candidate_scoring import fuse_grouped_scores, score_slates
from preprocess import preprocess_category
from utils import (
    autocast_context,
    build_paths,
    choose_run_name,
    ensure_dir,
    format_seconds,
    get_model_dtype,
    humanize_category,
    load_json,
    log,
    read_latest_run_name,
    resolve_model_name,
    save_json,
    set_seed,
    summarize_metrics,
    timestamp_suffix,
    write_latest_run_name,
)


def _load_local_torch_payload(path: Path, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _compute_sft_batch_loss(model, batch, device: str):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    shift_logits = outputs.logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    per_token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size())
    token_mask = shift_labels.ne(-100)
    token_counts = token_mask.sum(dim=1).clamp(min=1)
    sample_loss = per_token_loss.sum(dim=1) / token_counts
    weights = batch["loss_weight"].to(device=device, dtype=sample_loss.dtype)
    return (sample_loss * weights).mean()


def _build_sft_dataset(sources, tokenizer, max_length: int) -> SFTDataset:
    entries = []
    for source_idx, source in enumerate(sources):
        target_ids = tokenizer(source.output_title, add_special_tokens=False).input_ids
        entries.append(TrainingEntry(source_idx=source_idx, target_title=source.output_title, target_ids=target_ids))
    return SFTDataset(sources=sources, entries=entries, max_length=max_length, eos_token_id=tokenizer.eos_token_id)


def _run_sft_epoch(model, dataset, tokenizer, optimizer, scheduler, args, device: str, dtype: torch.dtype, epoch: int) -> float:
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=TrainCollator(tokenizer.pad_token_id),
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    progress = tqdm(loader, desc=f"SFT {epoch + 1}/{args.sft_epochs}", leave=False)
    for step, batch in enumerate(progress, start=1):
        with autocast_context(dtype):
            loss = _compute_sft_batch_loss(model, batch, device)
        (loss / max(args.gradient_accumulation_steps, 1)).backward()
        losses.append(float(loss.detach().cpu().item()))
        if step % args.gradient_accumulation_steps == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        progress.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return float(np.mean(losses)) if losses else 0.0


def _sft_validation_loss(model, dataset, tokenizer, args, device: str, dtype: torch.dtype) -> float:
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=TrainCollator(tokenizer.pad_token_id),
    )
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="SFT valid", leave=False):
            with autocast_context(dtype):
                losses.append(float(_compute_sft_batch_loss(model, batch, device).detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def _save_model(model, tokenizer, target_dir: Path) -> None:
    ensure_dir(target_dir)
    model_to_save = getattr(model, "_orig_mod", model)
    model_to_save.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)


def _train_sft(model, tokenizer, train_sources, valid_sources, paths, args, device: str, dtype: torch.dtype) -> Dict[str, float]:
    if args.sft_epochs <= 0:
        return {"sft_train_loss": 0.0, "sft_val_loss": 0.0}
    train_dataset = _build_sft_dataset(train_sources, tokenizer, args.max_length)
    valid_dataset = _build_sft_dataset(valid_sources, tokenizer, args.max_length)
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / max(args.micro_batch_size * args.gradient_accumulation_steps, 1)))
    total_steps = max(1, steps_per_epoch * args.sft_epochs)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )
    last_train_loss = 0.0
    last_val_loss = float("inf")
    for epoch in range(args.sft_epochs):
        last_train_loss = _run_sft_epoch(model, train_dataset, tokenizer, optimizer, scheduler, args, device, dtype, epoch)
        last_val_loss = _sft_validation_loss(model, valid_dataset, tokenizer, args, device, dtype)
        log(f"SFT epoch {epoch + 1}/{args.sft_epochs}: train_loss={last_train_loss:.4f}, val_loss={last_val_loss:.4f}")
    _save_model(model, tokenizer, paths["sft_model_dir"])
    return {"sft_train_loss": last_train_loss, "sft_val_loss": last_val_loss}


def _slate_sft_regularizer(grouped_scores: List[torch.Tensor], batch_slates: List[dict]) -> torch.Tensor:
    losses = []
    for scores, slate in zip(grouped_scores, batch_slates):
        gains = torch.tensor(slate["gains"], dtype=torch.float32, device=scores.device)
        positive = scores[gains > 0]
        if positive.numel() > 0:
            losses.append(-positive.mean())
    if not losses:
        return grouped_scores[0].new_tensor(0.0)
    return torch.stack(losses).mean()


def _evaluate_slates(model, dataset, tokenizer, args, device: str, dtype: torch.dtype, null_prompt_ids=None) -> Dict[str, object]:
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=SlateCollator())
    predictions = []
    losses = []
    model.eval()
    with torch.no_grad():
        for batch_slates in tqdm(loader, desc="Slate valid", leave=False):
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
                loss = candidate_closed_loss(grouped_scores, batch_slates, tau=args.tau, odds_weight=args.odds_weight)
            losses.append(float(loss.detach().cpu().item()))
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
                predictions.append({"output": slate["output_title"], "predict": ranked[: args.topk]})
    metrics = summarize_metrics(predictions, topk_values=[1, 5, 10])
    return {"loss": float(np.mean(losses)) if losses else 0.0, "metrics": metrics}


def _train_sla(model, tokenizer, train_slates, valid_slates, paths, args, device: str, dtype: torch.dtype, category_text: str) -> Dict[str, object]:
    if args.sla_epochs <= 0 or args.freeze_after_sft:
        _save_model(model, tokenizer, paths["last_model_dir"])
        _save_model(model, tokenizer, paths["best_model_dir"])
        return {"best_valid_ndcg10": 0.0, "best_valid_hr10": 0.0, "sla_train_loss": 0.0}

    train_dataset = CandidateSlateDataset(train_slates, tokenizer, max_length=args.max_length, eos_token_id=tokenizer.eos_token_id)
    valid_dataset = CandidateSlateDataset(valid_slates, tokenizer, max_length=args.max_length, eos_token_id=tokenizer.eos_token_id)
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / max(args.micro_batch_size * args.gradient_accumulation_steps, 1)))
    total_steps = max(1, steps_per_epoch * args.sla_epochs)
    optimizer = AdamW(model.parameters(), lr=args.sla_learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )
    null_prompt_ids = None
    if args.null_alpha > 0:
        null_prompt_ids = tokenizer(build_null_prompt(category_text), add_special_tokens=True).input_ids

    _save_model(model, tokenizer, paths["best_model_dir"])
    _save_model(model, tokenizer, paths["last_model_dir"])
    baseline_result = _evaluate_slates(model, valid_dataset, tokenizer, args, device, dtype, null_prompt_ids=null_prompt_ids)
    baseline_ndcg10 = float(baseline_result["metrics"]["k10"]["ndcg"])
    baseline_hr10 = float(baseline_result["metrics"]["k10"]["hr"])
    best_ndcg10 = baseline_ndcg10
    best_hr10 = baseline_hr10
    log(f"SLA gate baseline: HR@10={baseline_hr10:.4f}, NDCG@10={baseline_ndcg10:.4f}")
    train_deploy = sum(1 for row in train_slates if row.get("deploy_reachable")) / max(len(train_slates), 1)
    valid_deploy = sum(1 for row in valid_slates if row.get("deploy_reachable")) / max(len(valid_slates), 1)
    baseline_capture = baseline_hr10 / max(valid_deploy, 1e-12)
    if args.auto_skip_sla and (
        train_deploy - valid_deploy >= float(args.max_train_valid_reach_gap)
        or baseline_capture >= float(args.skip_sla_capture_ratio)
    ):
        log(
            "Skipping SLA: "
            f"train_deploy={train_deploy:.4f}, valid_deploy={valid_deploy:.4f}, "
            f"baseline_capture={baseline_capture:.4f}"
        )
        torch.save(
            {
                "completed": True,
                "skipped_sla": True,
                "best_valid_ndcg10": best_ndcg10,
                "best_valid_hr10": best_hr10,
                "train_deploy": train_deploy,
                "valid_deploy": valid_deploy,
                "baseline_capture": baseline_capture,
            },
            paths["state_pt"],
        )
        return {
            "best_valid_ndcg10": best_ndcg10,
            "best_valid_hr10": best_hr10,
            "sla_train_loss": 0.0,
            "skipped": True,
        }

    last_train_loss = 0.0
    loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=SlateCollator(),
    )
    for epoch in range(args.sla_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        progress = tqdm(loader, desc=f"SLA {epoch + 1}/{args.sla_epochs}", leave=False)
        for step, batch_slates in enumerate(progress, start=1):
            with autocast_context(dtype):
                grouped_scores, _ = score_slates(
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
                loss = candidate_closed_loss(grouped_scores, batch_slates, tau=args.tau, odds_weight=args.odds_weight)
                if args.sft_loss_weight > 0:
                    loss = loss + float(args.sft_loss_weight) * _slate_sft_regularizer(grouped_scores, batch_slates)
            (loss / max(args.gradient_accumulation_steps, 1)).backward()
            losses.append(float(loss.detach().cpu().item()))
            if step % args.gradient_accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        last_train_loss = float(np.mean(losses)) if losses else 0.0
        valid_result = _evaluate_slates(model, valid_dataset, tokenizer, args, device, dtype, null_prompt_ids=null_prompt_ids)
        ndcg10 = float(valid_result["metrics"]["k10"]["ndcg"])
        hr10 = float(valid_result["metrics"]["k10"]["hr"])
        passes_floor = hr10 + 1e-12 >= baseline_hr10 - float(args.accept_hr_drop)
        is_best = passes_floor and (ndcg10 > best_ndcg10 or (math.isclose(ndcg10, best_ndcg10) and hr10 >= best_hr10))
        if is_best:
            best_ndcg10 = ndcg10
            best_hr10 = hr10
            _save_model(model, tokenizer, paths["best_model_dir"])
        _save_model(model, tokenizer, paths["last_model_dir"])
        state_payload = {
            "epoch": epoch,
            "best_valid_ndcg10": best_ndcg10,
            "best_valid_hr10": best_hr10,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "completed": False,
        }
        torch.save(state_payload, paths["state_pt"])
        log(
            f"SLA epoch {epoch + 1}/{args.sla_epochs}: train_loss={last_train_loss:.4f}, "
            f"valid_loss={valid_result['loss']:.4f}, HR@10={hr10:.4f}, NDCG@10={ndcg10:.4f}"
        )

    final_state = _load_local_torch_payload(paths["state_pt"], map_location="cpu")
    final_state["completed"] = True
    torch.save(final_state, paths["state_pt"])
    return {"best_valid_ndcg10": best_ndcg10, "best_valid_hr10": best_hr10, "sla_train_loss": last_train_loss}


def parse_args():
    parser = argparse.ArgumentParser(description="Train SLATE-Rec with full-parameter SFT and SLA alignment.")
    parser.add_argument("--category", required=True, type=str)
    parser.add_argument("--revise", required=True, type=int)
    parser.add_argument("--llm", required=True, type=str)
    parser.add_argument("--run_name", default="", type=str)
    parser.add_argument("--extra_tag", default="", type=str)
    parser.add_argument("--force_new_run", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--micro_batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=2, type=int)
    parser.add_argument("--score_batch_size", default=16, type=int)
    parser.add_argument("--sft_epochs", default=1, type=int)
    parser.add_argument("--sla_epochs", default=1, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--sla_learning_rate", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--freeze_after_sft", action="store_true")
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
    parser.add_argument("--tau", default=1.0, type=float)
    parser.add_argument("--odds_weight", default=0.1, type=float)
    parser.add_argument("--sft_loss_weight", default=0.05, type=float)
    parser.add_argument("--accept_hr_drop", default=0.0, type=float)
    parser.add_argument("--disable_auto_skip_sla", dest="auto_skip_sla", action="store_false")
    parser.set_defaults(auto_skip_sla=True)
    parser.add_argument("--max_train_valid_reach_gap", default=0.35, type=float)
    parser.add_argument("--skip_sla_capture_ratio", default=0.88, type=float)
    parser.add_argument("--force_rebuild_candidates", action="store_true")
    parser.add_argument("--topk", default=10, type=int)
    args, unknown = parser.parse_known_args()
    if unknown:
        log(f"Ignoring unknown train.py arguments: {' '.join(unknown)}")
    return args


def main():
    args = parse_args()
    started = time.time()
    preprocess_category(args.category, args.revise)
    set_seed(args.seed)
    run_name = choose_run_name(args)
    if args.force_new_run:
        run_name = f"{run_name}_{timestamp_suffix()}"
    args.run_name = run_name
    paths = build_paths(args.category, args.revise, run_name=run_name)
    write_latest_run_name(args.category, run_name)

    dtype = get_model_dtype(prefer_bf16=True)
    base_model_name = resolve_model_name(args.llm)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    category_text = humanize_category(args.category)
    train_sources = build_sources(paths["train_csv"], tokenizer, category_text, args.max_length)
    valid_sources = build_sources(paths["valid_csv"], tokenizer, category_text, args.max_length)
    catalog_rows = load_json(paths["catalog_json"])["items"]

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.config.use_cache = False
    if not args.disable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    log(f"Training run name: {run_name}")
    log("LoRA/PEFT is not used; this is full-parameter training for a single GPU.")
    sft_summary = _train_sft(model, tokenizer, train_sources, valid_sources, paths, args, device, dtype)

    train_slates = build_candidate_slates(
        args,
        split_name="train",
        sources=train_sources,
        train_sources=train_sources,
        catalog_rows=catalog_rows,
        model=model,
        tokenizer=tokenizer,
        device=device,
        include_oracle_positive=True,
        force_rebuild=args.force_rebuild_candidates,
    )
    valid_slates = build_candidate_slates(
        args,
        split_name="valid",
        sources=valid_sources,
        train_sources=train_sources,
        catalog_rows=catalog_rows,
        model=model,
        tokenizer=tokenizer,
        device=device,
        include_oracle_positive=False,
        force_rebuild=args.force_rebuild_candidates,
    )
    train_reach = reachability_summary(train_slates)
    valid_reach = reachability_summary(valid_slates)
    log(f"Train reachability: {train_reach}")
    log(f"Valid reachability: {valid_reach}")

    sla_summary = _train_sla(model, tokenizer, train_slates, valid_slates, paths, args, device, dtype, category_text)
    summary = {
        "framework": "SLATE-Rec",
        "category": args.category,
        "revise": args.revise,
        "llm": args.llm,
        "run_name": run_name,
        "config": vars(args),
        "sft": sft_summary,
        "sla": sla_summary,
        "train_reachability": train_reach,
        "valid_reachability": valid_reach,
        "elapsed_seconds": time.time() - started,
    }
    save_json(paths["summary_json"], summary)
    latest = read_latest_run_name(args.category)
    log(f"Training completed in {format_seconds(summary['elapsed_seconds'])}; latest run pointer={latest}")


if __name__ == "__main__":
    main()
