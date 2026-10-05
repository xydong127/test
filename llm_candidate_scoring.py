from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def _pack_candidate_sequences(
    prompt_ids_list: Sequence[Sequence[int]],
    target_ids_list: Sequence[Sequence[int]],
    max_length: int,
    pad_token_id: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    packed_input_ids = []
    packed_labels = []
    for prompt_ids, target_ids in zip(prompt_ids_list, target_ids_list):
        target_ids = list(target_ids)
        combined = list(prompt_ids) + target_ids
        if len(combined) > max_length:
            combined = combined[-max_length:]
        kept_target = min(len(target_ids), len(combined))
        prompt_length = max(0, len(combined) - kept_target)
        labels = [-100] * prompt_length + combined[prompt_length:]
        packed_input_ids.append(combined)
        packed_labels.append(labels)

    max_seq_len = max(len(row) for row in packed_input_ids)
    input_ids = []
    labels = []
    attention_mask = []
    for ids, row_labels in zip(packed_input_ids, packed_labels):
        pad = max_seq_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        labels.append(row_labels + [-100] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
    }


def sequence_log_likelihoods(
    model,
    prompt_ids_list: Sequence[Sequence[int]],
    target_ids_list: Sequence[Sequence[int]],
    max_length: int,
    pad_token_id: int,
    device: str,
    length_gamma: float = 1.0,
) -> torch.Tensor:
    batch = _pack_candidate_sequences(prompt_ids_list, target_ids_list, max_length, pad_token_id, device)
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    shift_logits = outputs.logits[:, :-1, :].float()
    shift_labels = batch["labels"][:, 1:]
    per_token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size())
    token_mask = shift_labels.ne(-100)
    token_counts = token_mask.sum(dim=1).clamp(min=1).to(per_token_loss.dtype)
    logprob_sum = -(per_token_loss * token_mask.to(per_token_loss.dtype)).sum(dim=1)
    denom = torch.pow(token_counts, float(length_gamma)).clamp(min=1.0)
    return logprob_sum / denom


def score_slates(
    model,
    slates: Sequence[dict],
    max_length: int,
    pad_token_id: int,
    device: str,
    length_gamma: float = 1.0,
    max_score_batch_size: int = 32,
    null_prompt_ids: Sequence[int] = None,
    null_alpha: float = 0.0,
) -> Tuple[List[torch.Tensor], List[List[str]]]:
    prompt_ids_flat: List[List[int]] = []
    target_ids_flat: List[List[int]] = []
    slate_sizes = []
    titles_by_slate = []
    for slate in slates:
        slate_sizes.append(len(slate["candidates"]))
        titles_by_slate.append([candidate["title"] for candidate in slate["candidates"]])
        for candidate in slate["candidates"]:
            prompt_ids_flat.append(list(slate["prompt_ids"]))
            target_ids_flat.append(list(candidate["target_ids"]))

    conditional_scores = []
    for start in range(0, len(target_ids_flat), max_score_batch_size):
        stop = start + max_score_batch_size
        conditional_scores.append(
            sequence_log_likelihoods(
                model,
                prompt_ids_flat[start:stop],
                target_ids_flat[start:stop],
                max_length=max_length,
                pad_token_id=pad_token_id,
                device=device,
                length_gamma=length_gamma,
            )
        )
    scores_flat = torch.cat(conditional_scores, dim=0) if conditional_scores else torch.empty(0, device=device)

    if null_alpha > 0.0 and null_prompt_ids is not None:
        null_prompt_flat = [list(null_prompt_ids) for _ in target_ids_flat]
        null_scores = []
        with torch.no_grad():
            for start in range(0, len(target_ids_flat), max_score_batch_size):
                stop = start + max_score_batch_size
                null_scores.append(
                    sequence_log_likelihoods(
                        model,
                        null_prompt_flat[start:stop],
                        target_ids_flat[start:stop],
                        max_length=max_length,
                        pad_token_id=pad_token_id,
                        device=device,
                        length_gamma=length_gamma,
                    )
                )
        scores_flat = scores_flat - float(null_alpha) * torch.cat(null_scores, dim=0)

    grouped_scores = []
    offset = 0
    for size in slate_sizes:
        grouped_scores.append(scores_flat[offset : offset + size])
        offset += size
    return grouped_scores, titles_by_slate


def _standardize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    std = values.std(unbiased=False)
    if float(std.detach().cpu()) < 1e-6:
        return torch.zeros_like(values)
    return (values - values.mean()) / std


def _candidate_source_prior(candidate: dict) -> float:
    ranks = candidate.get("source_ranks", {}) or {}
    values = []
    if candidate.get("beam"):
        values.append(1.00 / max(float(ranks.get("beam", 1)), 1.0))
    if candidate.get("transition"):
        values.append(0.90 / max(float(ranks.get("transition", 1)), 1.0))
    if candidate.get("retrieval"):
        values.append(0.80 / max(float(ranks.get("retrieval", 1)), 1.0))
    if candidate.get("fill"):
        values.append(0.15 / max(float(ranks.get("fill", 1)), 1.0))
    return max(values) if values else 0.0


def fuse_grouped_scores(
    grouped_scores: Sequence[torch.Tensor],
    batch_slates: Sequence[dict],
    ranker: str = "fusion",
    llm_weight: float = 0.35,
    evidence_weight: float = 1.0,
    source_weight: float = 0.30,
) -> List[torch.Tensor]:
    if ranker == "llm":
        return list(grouped_scores)
    fused = []
    for scores, slate in zip(grouped_scores, batch_slates):
        meta = slate.get("candidate_meta", [])
        evidence = torch.tensor(
            [float(candidate.get("evidence_score", 0.0)) for candidate in meta],
            dtype=scores.dtype,
            device=scores.device,
        )
        source = torch.tensor(
            [_candidate_source_prior(candidate) for candidate in meta],
            dtype=scores.dtype,
            device=scores.device,
        )
        evidence = torch.log1p(torch.clamp(evidence, min=0.0))
        if ranker == "evidence":
            fused.append(_standardize(evidence) + float(source_weight) * _standardize(source))
        else:
            fused.append(
                float(llm_weight) * _standardize(scores)
                + float(evidence_weight) * _standardize(evidence)
                + float(source_weight) * _standardize(source)
            )
    return fused
