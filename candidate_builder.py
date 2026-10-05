from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PromptCollator, PromptOnlyDataset
from utils import (
    build_catalog_indices,
    build_paths,
    build_transition_index,
    ensure_dir,
    filter_topk_unique,
    load_json,
    log,
    map_text_to_catalog,
    normalize_title,
    retrieve_catalog_titles,
    save_json,
    transition_retrieve_titles,
)


def _candidate_cache_path(args, split_name: str):
    paths = build_paths(args.category, args.revise, run_name=getattr(args, "run_name", ""))
    cache_dir = ensure_dir(paths["candidate_dir"])
    run_part = getattr(args, "run_name", "") or "base"
    cache_name = (
        f"{split_name}_{run_part}_{args.prefilter}-v4_histtarget_llm-{args.llm}_beam-{args.beam_width}_wide-{args.wide_size}_"
        f"slate-{args.slate_size}_trans-{args.transition_k}_ret-{args.retrieval_k}_"
        f"fill-{args.fill_k}_len-{args.max_length}.json"
    )
    return cache_dir / cache_name


def _add_candidate(candidate_map: Dict[str, dict], title: str, source: str, evidence: float, rank: int, item_id_by_title: Dict[str, int]):
    normalized = normalize_title(title)
    if not normalized:
        return
    current = candidate_map.get(normalized)
    if current is None:
        current = {
            "title": title,
            "item_id": int(item_id_by_title.get(normalized, -1)),
            "evidence_score": 0.0,
            "sources": [],
            "source_ranks": {},
            "beam": False,
            "transition": False,
            "retrieval": False,
            "fill": False,
            "oracle": False,
        }
        candidate_map[normalized] = current
    current["evidence_score"] += float(evidence)
    if source not in current["sources"]:
        current["sources"].append(source)
    current["source_ranks"][source] = min(int(rank), int(current["source_ranks"].get(source, rank)))
    if source in ("beam", "transition", "retrieval", "fill", "oracle"):
        current[source] = True


def _generate_projected_beams(
    model,
    tokenizer,
    sources,
    catalog_indices,
    args,
    device: str,
) -> List[List[str]]:
    dataset = PromptOnlyDataset(sources)
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=PromptCollator(tokenizer.pad_token_id, padding_side="left"),
    )
    beams = max(args.beam_width, args.beam_k)
    projected_by_source = [[] for _ in sources]
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Beam reservoir", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beams,
                num_return_sequences=args.beam_k,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            continuation_ids = generated[:, input_ids.shape[1] :]
            decoded = tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)
            for offset, source_index in enumerate(batch["source_index"]):
                texts = decoded[offset * args.beam_k : (offset + 1) * args.beam_k]
                mapped = []
                for text in texts:
                    suffix = text.strip().split("\n")[0].strip().strip('"')
                    title = map_text_to_catalog(
                        text=suffix,
                        titles=catalog_indices["titles"],
                        token_lists=catalog_indices["token_lists"],
                        normalized_to_indices=catalog_indices["normalized_to_indices"],
                        inverted_index=catalog_indices["inverted_index"],
                        popularity_norm=catalog_indices["popularity_norm"],
                    )
                    if title:
                        mapped.append(title)
                projected_by_source[int(source_index)] = filter_topk_unique(mapped, args.beam_k)
    return projected_by_source


def _source_rank(candidate: dict, source: str) -> int:
    return int(candidate.get("source_ranks", {}).get(source, 10**6))


def _append_unique(output: List[dict], seen: set, candidates: Sequence[dict], limit: int) -> None:
    for candidate in candidates:
        if len(output) >= limit:
            return
        normalized = normalize_title(candidate["title"])
        if normalized in seen:
            continue
        output.append(candidate)
        seen.add(normalized)


def _source_sorted(reservoir: Sequence[dict], source: str) -> List[dict]:
    return sorted(
        [candidate for candidate in reservoir if candidate.get(source)],
        key=lambda item: (_source_rank(item, source), -float(item.get("evidence_score", 0.0))),
    )


def _select_deploy_candidates(reservoir: Sequence[dict], slate_size: int, topk: int) -> List[dict]:
    """Preserve source diversity before LLM scoring.

    A single global evidence sort was dropping useful reservoir positives. This
    selector protects the highest ranked candidates from each evidence source,
    then fills the remaining slots by global evidence.
    """
    limit = max(int(slate_size), int(topk))
    global_sorted = sorted(reservoir, key=lambda item: float(item.get("evidence_score", 0.0)), reverse=True)
    source_lists = {
        "beam": _source_sorted(reservoir, "beam"),
        "transition": _source_sorted(reservoir, "transition"),
        "retrieval": _source_sorted(reservoir, "retrieval"),
        "fill": _source_sorted(reservoir, "fill"),
    }
    source_limits = {
        "beam": min(int(topk), limit),
        "transition": max(int(topk), limit // 3),
        "retrieval": max(int(topk), limit // 3),
        "fill": max(0, min(3, limit // 10)),
    }
    selected: List[dict] = []
    seen = set()
    _append_unique(selected, seen, global_sorted[: max(int(topk), limit // 4)], limit)
    for source in ("beam", "transition", "retrieval", "fill"):
        _append_unique(selected, seen, source_lists[source][: source_limits[source]], limit)
    _append_unique(selected, seen, global_sorted, limit)
    return selected[:limit]


def _finish_slate(
    source,
    source_index: int,
    candidate_map: Dict[str, dict],
    wide_size: int,
    slate_size: int,
    topk: int,
    prefilter: str,
    include_oracle_positive: bool,
) -> dict:
    target_norm = normalize_title(source.output_title)
    reservoir = sorted(candidate_map.values(), key=lambda item: item["evidence_score"], reverse=True)
    reservoir = reservoir[:wide_size]
    if prefilter == "global":
        deploy_candidates = reservoir[: max(int(slate_size), int(topk))]
    else:
        deploy_candidates = _select_deploy_candidates(reservoir, slate_size=slate_size, topk=topk)

    deploy_norms = {normalize_title(item["title"]) for item in deploy_candidates}
    reservoir_norms = {normalize_title(item["title"]) for item in reservoir}
    beam_norms = {normalize_title(item["title"]) for item in reservoir if item.get("beam")}
    retrieval_norms = {
        normalize_title(item["title"])
        for item in reservoir
        if item.get("beam") or item.get("transition") or item.get("retrieval")
    }

    train_candidates = [dict(item) for item in deploy_candidates]
    oracle_injected = False
    if include_oracle_positive and target_norm not in deploy_norms:
        oracle_injected = True
        train_candidates.append(
            {
                "title": source.output_title,
                "item_id": int(source.output_item_id),
                "evidence_score": max([item["evidence_score"] for item in train_candidates] + [0.0]) + 1.0,
                "sources": ["oracle"],
                "source_ranks": {"oracle": 1},
                "beam": False,
                "transition": False,
                "retrieval": False,
                "fill": False,
                "oracle": True,
            }
        )

    for candidate in train_candidates:
        candidate["gain"] = 1.0 if normalize_title(candidate["title"]) == target_norm else 0.0
    for candidate in deploy_candidates:
        candidate["gain"] = 1.0 if normalize_title(candidate["title"]) == target_norm else 0.0

    return {
        "source_index": int(source_index),
        "user_id": source.user_id,
        "prompt": source.prompt,
        "prompt_ids": list(source.prompt_ids),
        "history_titles": list(source.history_titles),
        "history_item_ids": list(source.history_item_ids),
        "output_title": source.output_title,
        "output_item_id": int(source.output_item_id),
        "reservoir_candidates": reservoir,
        "deploy_candidates": deploy_candidates,
        "train_candidates": train_candidates,
        "beam_reachable": target_norm in beam_norms,
        "retrieval_reachable": target_norm in retrieval_norms,
        "reservoir_reachable": target_norm in reservoir_norms,
        "deploy_reachable": target_norm in deploy_norms,
        "oracle_injected": oracle_injected,
    }


def build_candidate_slates(
    args,
    split_name: str,
    sources,
    train_sources,
    catalog_rows: Sequence[dict],
    model,
    tokenizer,
    device: str,
    include_oracle_positive: bool,
    force_rebuild: bool = False,
) -> List[dict]:
    cache_path = _candidate_cache_path(args, split_name)
    if cache_path.exists() and not force_rebuild:
        log(f"Loading cached {split_name} candidate slates from {cache_path}")
        return load_json(cache_path)

    log(f"Building {split_name} candidate slates with TEF")
    catalog_indices = build_catalog_indices(catalog_rows)
    item_id_by_title = {normalize_title(row["title"]): int(row["item_id"]) for row in catalog_rows}
    transition_index = build_transition_index(train_sources, catalog_rows, max_next_per_item=max(args.transition_k * 2, 20))
    projected_beams = _generate_projected_beams(model, tokenizer, sources, catalog_indices, args, device)
    popularity_ranked = sorted(catalog_rows, key=lambda row: row.get("popularity", 0), reverse=True)

    slates = []
    for source_index, source in enumerate(tqdm(sources, desc=f"{split_name} slates", leave=False)):
        candidate_map: Dict[str, dict] = {}
        exclude_titles = list(source.history_titles)

        for rank, title in enumerate(projected_beams[source_index], start=1):
            _add_candidate(candidate_map, title, "beam", evidence=4.0 / (rank + 1.0), rank=rank, item_id_by_title=item_id_by_title)

        transition_rows = transition_retrieve_titles(
            source.history_item_ids,
            transition_index,
            top_k=args.transition_k,
            recent_k=args.transition_recent_k,
            exclude_titles=exclude_titles,
        )
        for rank, row in enumerate(transition_rows, start=1):
            _add_candidate(
                candidate_map,
                row["title"],
                "transition",
                evidence=3.0 * float(row["score"]),
                rank=rank,
                item_id_by_title=item_id_by_title,
            )

        retrieved = retrieve_catalog_titles(
            history_titles=source.history_titles,
            catalog_rows=catalog_rows,
            catalog_indices=catalog_indices,
            top_k=args.retrieval_k,
            exclude_titles=exclude_titles,
        )
        for rank, title in enumerate(retrieved, start=1):
            _add_candidate(candidate_map, title, "retrieval", evidence=2.0 / (rank + 1.0), rank=rank, item_id_by_title=item_id_by_title)

        fill_rank = 0
        for row in popularity_ranked:
            if len(candidate_map) >= max(args.wide_size, args.slate_size, args.topk):
                break
            title = row["title"]
            if normalize_title(title) in {normalize_title(item) for item in exclude_titles}:
                continue
            fill_rank += 1
            _add_candidate(candidate_map, title, "fill", evidence=0.25 / (fill_rank + 1.0), rank=fill_rank, item_id_by_title=item_id_by_title)
            if fill_rank >= args.fill_k:
                break

        slates.append(
            _finish_slate(
                source=source,
                source_index=source_index,
                candidate_map=candidate_map,
                wide_size=args.wide_size,
                slate_size=args.slate_size,
                topk=args.topk,
                prefilter=args.prefilter,
                include_oracle_positive=include_oracle_positive,
            )
        )

    payload = {
        "framework": "SLATE-Rec",
        "split": split_name,
        "category": args.category,
        "revise": args.revise,
        "num_slates": len(slates),
        "slates": slates,
    }
    save_json(cache_path, payload["slates"])
    log(f"Saved {len(slates)} {split_name} slates to {cache_path}")
    return slates


def reachability_summary(slates: Sequence[dict]) -> Dict[str, float]:
    total = max(len(slates), 1)
    return {
        "beam": sum(1 for row in slates if row.get("beam_reachable")) / total,
        "retrieval": sum(1 for row in slates if row.get("retrieval_reachable")) / total,
        "reservoir": sum(1 for row in slates if row.get("reservoir_reachable")) / total,
        "deploy": sum(1 for row in slates if row.get("deploy_reachable")) / total,
        "oracle_injected": sum(1 for row in slates if row.get("oracle_injected")) / total,
    }
