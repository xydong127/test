from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset


def build_prompt(category_text: str, history_titles: Sequence[str]) -> str:
    history_lines = "\n".join(f"{index + 1}. {title}" for index, title in enumerate(history_titles))
    return (
        "You are a recommendation model.\n"
        f"Category: {category_text}\n"
        "The user recently interacted with these item titles:\n"
        f"{history_lines}\n"
        "Predict exactly one next item title from the catalog.\n"
        "Return only the item title.\n"
        "Answer:"
    )


def build_null_prompt(category_text: str) -> str:
    return (
        "You are a recommendation model.\n"
        f"Category: {category_text}\n"
        "Score the item title as a catalog item.\n"
        "Answer:"
    )


@dataclass
class SourceExample:
    user_id: str
    prompt: str
    prompt_ids: List[int]
    history_titles: List[str]
    history_item_ids: List[int]
    output_title: str
    output_item_id: int


@dataclass
class TrainingEntry:
    source_idx: int
    target_title: str
    target_ids: List[int]
    loss_weight: float = 1.0


class SFTDataset(Dataset):
    def __init__(self, sources: Sequence[SourceExample], entries: Sequence[TrainingEntry], max_length: int, eos_token_id: int):
        self.sources = list(sources)
        self.entries = list(entries)
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, object]:
        entry = self.entries[index]
        source = self.sources[entry.source_idx]
        target_ids = list(entry.target_ids)
        if not target_ids or target_ids[-1] != self.eos_token_id:
            target_ids = target_ids + [self.eos_token_id]
        combined = source.prompt_ids + target_ids
        if len(combined) > self.max_length:
            combined = combined[-self.max_length :]
        kept_target = min(len(target_ids), len(combined))
        prompt_length = max(0, len(combined) - kept_target)
        labels = [-100] * prompt_length + combined[prompt_length:]
        return {
            "input_ids": combined,
            "labels": labels,
            "attention_mask": [1] * len(combined),
            "loss_weight": float(entry.loss_weight),
            "dataset_index": index,
        }


class PromptOnlyDataset(Dataset):
    def __init__(self, sources: Sequence[SourceExample]):
        self.sources = list(sources)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index: int) -> Dict[str, object]:
        source = self.sources[index]
        return {
            "input_ids": list(source.prompt_ids),
            "attention_mask": [1] * len(source.prompt_ids),
            "source_index": index,
            "history_titles": list(source.history_titles),
            "history_item_ids": list(source.history_item_ids),
            "output_title": source.output_title,
            "user_id": source.user_id,
            "prompt": source.prompt,
        }


class CandidateSlateDataset(Dataset):
    def __init__(self, slates: Sequence[dict], tokenizer, max_length: int, eos_token_id: int):
        self.slates = list(slates)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def __len__(self) -> int:
        return len(self.slates)

    def __getitem__(self, index: int) -> Dict[str, object]:
        slate = self.slates[index]
        candidates = []
        gains = []
        candidate_meta = []
        for candidate in slate["train_candidates"]:
            target_ids = self.tokenizer(candidate["title"], add_special_tokens=False).input_ids
            if not target_ids:
                continue
            if target_ids[-1] != self.eos_token_id:
                target_ids = target_ids + [self.eos_token_id]
            candidates.append({"title": candidate["title"], "target_ids": target_ids})
            gains.append(float(candidate.get("gain", 0.0)))
            candidate_meta.append(candidate)
        if not candidates:
            raise ValueError(f"Slate {index} has no tokenizable candidates")
        return {
            "source_index": int(slate["source_index"]),
            "user_id": slate["user_id"],
            "prompt": slate["prompt"],
            "prompt_ids": list(slate["prompt_ids"]),
            "output_title": slate["output_title"],
            "output_item_id": int(slate["output_item_id"]),
            "candidates": candidates,
            "gains": gains,
            "candidate_meta": candidate_meta,
            "deploy_reachable": bool(slate.get("deploy_reachable", False)),
            "oracle_injected": bool(slate.get("oracle_injected", False)),
        }


class TrainCollator:
    def __init__(self, pad_token_id: int, padding_side: str = "right"):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        labels = []
        attention_mask = []
        loss_weight = []
        dataset_indices = []
        for item in batch:
            length = len(item["input_ids"])
            pad = max_length - length
            if self.padding_side == "left":
                input_ids.append([self.pad_token_id] * pad + item["input_ids"])
                labels.append([-100] * pad + item["labels"])
                attention_mask.append([0] * pad + item["attention_mask"])
            else:
                input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
                labels.append(item["labels"] + [-100] * pad)
                attention_mask.append(item["attention_mask"] + [0] * pad)
            loss_weight.append(float(item["loss_weight"]))
            dataset_indices.append(int(item["dataset_index"]))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "loss_weight": torch.tensor(loss_weight, dtype=torch.float32),
            "dataset_index": torch.tensor(dataset_indices, dtype=torch.long),
        }


class PromptCollator:
    def __init__(self, pad_token_id: int, padding_side: str = "left"):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        max_length = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention_mask = []
        source_indices = []
        history_titles = []
        history_item_ids = []
        output_titles = []
        prompts = []
        user_ids = []
        for item in batch:
            length = len(item["input_ids"])
            pad = max_length - length
            if self.padding_side == "left":
                input_ids.append([self.pad_token_id] * pad + item["input_ids"])
                attention_mask.append([0] * pad + item["attention_mask"])
            else:
                input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
                attention_mask.append(item["attention_mask"] + [0] * pad)
            source_indices.append(int(item["source_index"]))
            history_titles.append(item["history_titles"])
            history_item_ids.append(item["history_item_ids"])
            output_titles.append(item["output_title"])
            prompts.append(item["prompt"])
            user_ids.append(item["user_id"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "source_index": source_indices,
            "history_titles": history_titles,
            "history_item_ids": history_item_ids,
            "output_title": output_titles,
            "prompt": prompts,
            "user_id": user_ids,
        }


class SlateCollator:
    def __call__(self, batch: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        return list(batch)
