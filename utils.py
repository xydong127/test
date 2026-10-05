import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


FRAMEWORK_NAME = "SLATE-Rec"
FRAMEWORK_ROOT = Path(__file__).resolve().parent

MODEL_REGISTRY = {
    "qwen14": "Qwen/Qwen3-14B",
    "qwen8": "Qwen/Qwen3-8B",
    "qwen4": "Qwen/Qwen3-4B",
    "qwen17": "Qwen/Qwen3-1.7B",
    "qwen06": "Qwen/Qwen3-0.6B",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{FRAMEWORK_NAME}][{now}] {message}", flush=True)


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - hours * 3600 - minutes * 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    if minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    return f"{secs:.1f}s"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def humanize_category(category: str) -> str:
    return category.replace("_", " ").strip()


def normalize_title(text: str) -> str:
    text = str(text or "").strip().strip('"').strip("'").strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def title_tokens(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(normalize_title(text))


def token_overlap_score(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    left_counter = Counter(left)
    right_counter = Counter(right)
    overlap = 0
    for token, count in left_counter.items():
        overlap += min(count, right_counter.get(token, 0))
    denom = len(left) + len(right)
    return 2.0 * overlap / denom if denom else 0.0


def set_overlap_score(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    union = len(left_set | right_set)
    return len(left_set & right_set) / union if union else 0.0


def minmax_normalize(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def resolve_model_name(llm: str) -> str:
    if llm not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported --llm '{llm}'. Supported values: {supported}")
    return MODEL_REGISTRY[llm]


def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def get_model_dtype(prefer_bf16: bool = True) -> torch.dtype:
    if prefer_bf16 and bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def autocast_context(dtype: torch.dtype):
    if not torch.cuda.is_available():
        return nullcontext()
    if dtype not in (torch.bfloat16, torch.float16):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def save_json(path: Path, data) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def dataset_root(source_root: Optional[str] = None) -> Path:
    if source_root:
        return Path(source_root).resolve()
    return (FRAMEWORK_ROOT.parent / "datasets").resolve()


def build_source_paths(category: str, revise: int, source_root: Optional[str] = None) -> Dict[str, Path]:
    root = dataset_root(source_root) / category
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    users_path = root / "users.json"
    split_path = root / "split.json"
    item_candidates = [
        root / f"item_{revise}.json",
        root / f"items_{revise}.json",
        root / f"item_{revise}.JSON",
        root / f"items_{revise}.JSON",
    ]
    item_path = next((candidate for candidate in item_candidates if candidate.exists()), None)
    if item_path is None:
        tried = ", ".join(str(candidate.name) for candidate in item_candidates)
        raise FileNotFoundError(f"Could not find an item file for revise={revise} under {root}. Tried: {tried}")
    for required_path in (users_path, split_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required source file not found: {required_path}")
    return {"root": root, "users": users_path, "items": item_path, "split": split_path}


def build_paths(category: str, revise: int, run_name: Optional[str] = None) -> Dict[str, Path]:
    stem = f"{category}_{revise}"
    paths = {
        "root": FRAMEWORK_ROOT,
        "train_csv": FRAMEWORK_ROOT / "train" / f"{stem}.csv",
        "valid_csv": FRAMEWORK_ROOT / "valid" / f"{stem}.csv",
        "test_csv": FRAMEWORK_ROOT / "test" / f"{stem}.csv",
        "info_txt": FRAMEWORK_ROOT / "info" / f"{stem}.txt",
        "catalog_json": FRAMEWORK_ROOT / "catalog" / f"{stem}.json",
        "stats_json": FRAMEWORK_ROOT / "stats" / f"{stem}.json",
        "cache_dir": FRAMEWORK_ROOT / "cache" / category,
        "candidate_dir": FRAMEWORK_ROOT / "candidates" / category,
        "outputs_dir": FRAMEWORK_ROOT / "outputs" / category,
        "metrics_dir": FRAMEWORK_ROOT / "metrics" / category,
        "checkpoints_dir": FRAMEWORK_ROOT / "checkpoints" / category,
    }
    if run_name:
        run_dir = FRAMEWORK_ROOT / "checkpoints" / category / run_name
        paths.update(
            {
                "run_dir": run_dir,
                "sft_model_dir": run_dir / "sft",
                "last_model_dir": run_dir / "last",
                "best_model_dir": run_dir / "best",
                "state_pt": run_dir / "training_state.pt",
                "summary_json": run_dir / "run_summary.json",
                "eval_summary_json": run_dir / "eval_summary.json",
                "output_json": FRAMEWORK_ROOT / "outputs" / category / f"{run_name}.json",
                "metrics_json": FRAMEWORK_ROOT / "metrics" / category / f"{category}_{revise}_{run_name}.json",
            }
        )
    return paths


def sanitize_tag(tag: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(tag))
    clean = re.sub(r"-{2,}", "-", clean).strip("-")
    return clean or "default"


def build_run_name(args) -> str:
    parts = [
        f"llm-{args.llm}",
        f"sft-{getattr(args, 'sft_epochs', 1)}",
        f"sla-{getattr(args, 'sla_epochs', 1)}",
        f"lr-{getattr(args, 'learning_rate', 2e-5):g}",
        f"mb-{getattr(args, 'micro_batch_size', 1)}",
        f"ga-{getattr(args, 'gradient_accumulation_steps', 8)}",
        f"len-{getattr(args, 'max_length', 1024)}",
        f"slate-{getattr(args, 'slate_size', 30)}",
        f"wide-{getattr(args, 'wide_size', 150)}",
        f"pref-{getattr(args, 'prefilter', 'stratified')}",
        f"alpha-{getattr(args, 'null_alpha', 0.0):g}",
        f"rank-{getattr(args, 'ranker', 'fusion')}",
        f"lw-{getattr(args, 'llm_rank_weight', 0.35):g}",
        f"ew-{getattr(args, 'evidence_rank_weight', 1.0):g}",
        f"sw-{getattr(args, 'source_rank_weight', 0.30):g}",
        f"k-{getattr(args, 'topk', 10)}",
    ]
    if getattr(args, "freeze_after_sft", False):
        parts.append("frozen")
    if getattr(args, "extra_tag", ""):
        parts.append(sanitize_tag(args.extra_tag))
    return "_".join(parts)


def choose_run_name(args) -> str:
    if getattr(args, "run_name", None):
        return sanitize_tag(args.run_name)
    return build_run_name(args)


def timestamp_suffix() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def write_info_file(path: Path, item_rows: List[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in item_rows:
            handle.write(f"{row['title']}\t{row['item_id']}\n")


def latest_run_pointer_path(category: str) -> Path:
    return FRAMEWORK_ROOT / "checkpoints" / category / "latest_run.txt"


def write_latest_run_name(category: str, run_name: str) -> None:
    path = latest_run_pointer_path(category)
    ensure_dir(path.parent)
    path.write_text(str(run_name).strip(), encoding="utf-8")


def read_latest_run_name(category: str) -> Optional[str]:
    path = latest_run_pointer_path(category)
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def build_catalog_indices(catalog_rows: Sequence[dict]) -> Dict[str, object]:
    titles = [row["title"] for row in catalog_rows]
    token_lists = [title_tokens(title) for title in titles]
    normalized_to_indices = defaultdict(list)
    inverted_index = defaultdict(set)
    popularity = []
    item_id_to_index = {}
    for index, row in enumerate(catalog_rows):
        normalized_to_indices[normalize_title(row["title"])].append(index)
        item_id_to_index[int(row["item_id"])] = index
        for token in set(token_lists[index]):
            inverted_index[token].add(index)
        popularity.append(int(row.get("popularity", 0)))
    popularity = np.asarray(popularity, dtype=np.float32)
    popularity_norm = minmax_normalize(popularity).tolist() if len(popularity) else []
    return {
        "titles": titles,
        "token_lists": token_lists,
        "normalized_to_indices": dict(normalized_to_indices),
        "inverted_index": {key: sorted(value) for key, value in inverted_index.items()},
        "popularity_norm": popularity_norm,
        "item_id_to_index": item_id_to_index,
    }


def get_candidate_indices_from_text(text: str, inverted_index: Dict[str, List[int]]) -> List[int]:
    candidates = set()
    for token in title_tokens(text):
        for index in inverted_index.get(token, []):
            candidates.add(index)
    return sorted(candidates)


def map_text_to_catalog(
    text: str,
    titles: Sequence[str],
    token_lists: Sequence[Sequence[str]],
    normalized_to_indices: Dict[str, List[int]],
    inverted_index: Dict[str, List[int]],
    popularity_norm: Optional[Sequence[float]] = None,
) -> Optional[str]:
    normalized = normalize_title(text)
    if not normalized:
        return None
    exact_indices = normalized_to_indices.get(normalized)
    if exact_indices:
        return titles[exact_indices[0]]
    candidate_ids = get_candidate_indices_from_text(normalized, inverted_index)
    if not candidate_ids:
        return None
    query_tokens = title_tokens(normalized)
    best_index = None
    best_score = -1.0
    for index in candidate_ids:
        score = token_overlap_score(query_tokens, token_lists[index])
        if popularity_norm is not None:
            score += 0.05 * float(popularity_norm[index])
        if score > best_score:
            best_score = score
            best_index = index
    if best_index is None or best_score <= 0.0:
        return None
    return titles[best_index]


def retrieve_catalog_titles(
    history_titles: Sequence[str],
    catalog_rows: Sequence[dict],
    catalog_indices: Dict[str, object],
    top_k: int,
    exclude_titles: Optional[Iterable[str]] = None,
) -> List[str]:
    exclude = {normalize_title(title) for title in (exclude_titles or [])}
    history_token_lists = [title_tokens(title) for title in history_titles]
    history_union = []
    for tokens in history_token_lists:
        history_union.extend(tokens)
    if not history_union:
        popular = sorted(catalog_rows, key=lambda row: row.get("popularity", 0), reverse=True)
        return [row["title"] for row in popular[:top_k]]
    candidate_ids = set()
    for token in set(history_union):
        for index in catalog_indices["inverted_index"].get(token, []):
            candidate_ids.add(index)
    if not candidate_ids:
        candidate_ids = set(range(len(catalog_rows)))
    scored = []
    for index in candidate_ids:
        title = catalog_rows[index]["title"]
        if normalize_title(title) in exclude:
            continue
        score = 0.65 * set_overlap_score(history_union, catalog_indices["token_lists"][index])
        local_scores = [token_overlap_score(tokens, catalog_indices["token_lists"][index]) for tokens in history_token_lists]
        if local_scores:
            score += 0.25 * float(sum(local_scores) / len(local_scores))
        score += 0.10 * float(catalog_indices["popularity_norm"][index]) if catalog_indices["popularity_norm"] else 0.0
        scored.append((score, int(catalog_rows[index].get("popularity", 0)), title))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [title for _, _, title in scored[:top_k]]


def build_transition_index(train_sources, catalog_rows: Sequence[dict], max_next_per_item: int = 100) -> Dict[int, List[dict]]:
    counts = defaultdict(Counter)
    for source in train_sources:
        history_ids = list(source.history_item_ids)
        target_id = int(source.output_item_id)
        # The deployed task predicts the held-out next item from the whole
        # history. Adjacent-only transitions are too sparse off train, so use
        # recency-weighted history-to-target evidence as the primary signal.
        for distance, history_id in enumerate(reversed(history_ids[-20:]), start=1):
            counts[int(history_id)][target_id] += 1.0 / math.sqrt(distance)
        # Keep a small adjacent-sequence signal for local session continuity.
        ids = history_ids + [target_id]
        for left, right in zip(ids[:-1], ids[1:]):
            counts[int(left)][int(right)] += 0.25
    item_id_to_row = {int(row["item_id"]): row for row in catalog_rows}
    transition_index = {}
    for left, counter in counts.items():
        total = float(sum(counter.values()))
        ranked = []
        for rank, (right, count) in enumerate(counter.most_common(max_next_per_item), start=1):
            row = item_id_to_row.get(int(right))
            if not row:
                continue
            ranked.append(
                {
                    "item_id": int(right),
                    "title": row["title"],
                    "count": float(count),
                    "prob": float(count / total) if total else 0.0,
                    "rank": rank,
                }
            )
        transition_index[int(left)] = ranked
    return transition_index


def transition_retrieve_titles(
    history_item_ids: Sequence[int],
    transition_index: Dict[int, List[dict]],
    top_k: int,
    recent_k: int = 5,
    exclude_titles: Optional[Iterable[str]] = None,
) -> List[dict]:
    exclude = {normalize_title(title) for title in (exclude_titles or [])}
    scored = {}
    recent_ids = list(history_item_ids)[-recent_k:]
    for recency_rank, item_id in enumerate(reversed(recent_ids), start=1):
        recency_weight = 1.0 / recency_rank
        for row in transition_index.get(int(item_id), []):
            title = row["title"]
            if normalize_title(title) in exclude:
                continue
            score = recency_weight * (float(row["prob"]) + 1.0 / (row["rank"] + 1.0))
            current = scored.get(title)
            if current is None or score > current["score"]:
                scored[title] = {
                    "title": title,
                    "score": float(score),
                    "rank": int(row["rank"]),
                    "from_item_id": int(item_id),
                    "transition_prob": float(row["prob"]),
                }
    ranked = sorted(scored.values(), key=lambda row: row["score"], reverse=True)
    return ranked[:top_k]


def filter_topk_unique(items: Sequence[str], top_k: int) -> List[str]:
    seen = set()
    output = []
    for item in items:
        normalized = normalize_title(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(item)
        if len(output) >= top_k:
            break
    return output


def latest_existing_model_dir(paths: Dict[str, Path]) -> Optional[Path]:
    for candidate in (paths.get("best_model_dir"), paths.get("last_model_dir"), paths.get("sft_model_dir")):
        if candidate and candidate.exists() and any(candidate.iterdir()):
            return candidate
    return None


def summarize_metrics(predictions: Sequence[dict], topk_values: Sequence[int]) -> Dict[str, Dict[str, float]]:
    topk_values = sorted(set(int(k) for k in topk_values))
    totals = {k: {"hr": 0.0, "ndcg": 0.0} for k in topk_values}
    count = len(predictions)
    for row in predictions:
        target = normalize_title(row["output"])
        ranked = [normalize_title(title) for title in row["predict"]]
        try:
            rank = ranked.index(target)
        except ValueError:
            rank = None
        for k in topk_values:
            if rank is not None and rank < k:
                totals[k]["hr"] += 1.0
                totals[k]["ndcg"] += 1.0 / math.log2(rank + 2.0)
    if count == 0:
        return {f"k{k}": {"hr": 0.0, "ndcg": 0.0} for k in topk_values}
    return {
        f"k{k}": {
            "hr": totals[k]["hr"] / count,
            "ndcg": totals[k]["ndcg"] / count,
        }
        for k in topk_values
    }
