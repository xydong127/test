import argparse
import csv
import time
from typing import Dict, List, Tuple

from utils import (
    build_paths,
    build_source_paths,
    ensure_dir,
    load_json,
    log,
    maybe_int,
    save_json,
    write_info_file,
)

PREPROCESS_VERSION = 2


def _parse_item_row(item_key: str, payload: dict) -> dict:
    title = str(payload.get("title", "")).replace("\n", " ").strip().strip('"')
    if not title:
        title = str(item_key)
    categories = payload.get("categories") or payload.get("category") or payload.get("cate") or []
    if isinstance(categories, str):
        categories = [categories]
    if categories and isinstance(categories[0], list):
        flattened = []
        for group in categories:
            flattened.extend(group)
        categories = flattened
    categories = [str(category).strip() for category in categories if str(category).strip()]
    primary_category = categories[-1] if categories else ""
    return {
        "item_key": str(item_key),
        "title": title,
        "categories": categories,
        "primary_category": primary_category,
    }


def _parse_interaction(entry) -> Tuple[str, float, int]:
    if isinstance(entry, dict):
        item_id = entry.get("item_id") or entry.get("item") or entry.get("asin") or entry.get("iid")
        rating = entry.get("rating", entry.get("score", entry.get("overall", 1.0)))
        timestamp = entry.get("timestamp", entry.get("time", entry.get("unixReviewTime", 0)))
    else:
        values = list(entry)
        item_id = values[0]
        rating = values[1] if len(values) > 1 else 1.0
        timestamp = values[2] if len(values) > 2 else 0
    return str(item_id), float(rating), int(timestamp)


def _build_rows(
    split_user_ids: List[str],
    users: Dict[str, list],
    item_lookup: Dict[str, dict],
    item2id: Dict[str, int],
) -> List[List[object]]:
    rows = []
    for user_id in split_user_ids:
        history = users[str(user_id)]
        parsed = [_parse_interaction(entry) for entry in history]
        parsed.sort(key=lambda item: item[2])
        if len(parsed) < 2:
            continue
        history_part = parsed[:-1]
        target_item, target_rating, target_time = parsed[-1]
        history_item_keys = [item_id for item_id, _, _ in history_part]
        history_titles = [item_lookup[item_id]["title"] for item_id in history_item_keys]
        history_ids = [item2id[item_id] for item_id in history_item_keys]
        history_ratings = [rating for _, rating, _ in history_part]
        history_times = [timestamp for _, _, timestamp in history_part]
        rows.append(
            [
                str(user_id),
                history_item_keys,
                target_item,
                history_ids,
                item2id[target_item],
                history_titles,
                item_lookup[target_item]["title"],
                history_ratings,
                target_rating,
                history_times,
                target_time,
            ]
        )
    return rows


def preprocess_category(category: str, revise: int, source_root: str = None, overwrite: bool = False) -> Dict[str, object]:
    started = time.time()
    source_paths = build_source_paths(category, revise, source_root=source_root)
    paths = build_paths(category, revise)

    existing_outputs_ready = (
        paths["train_csv"].exists()
        and paths["valid_csv"].exists()
        and paths["test_csv"].exists()
        and paths["catalog_json"].exists()
        and paths["info_txt"].exists()
        and paths["stats_json"].exists()
    )
    if not overwrite and existing_outputs_ready:
        try:
            stats = load_json(paths["stats_json"])
        except Exception:
            stats = {}
        if (
            int(stats.get("preprocess_version", 0)) >= PREPROCESS_VERSION
            and stats.get("popularity_scope") == "train_only"
        ):
            log(f"Preprocessed files already exist for {category}_{revise}; skipping.")
            return {"skipped": True, "elapsed_seconds": time.time() - started}
        log(
            f"Detected outdated preprocessing for {category}_{revise}; rebuilding "
            "to refresh train-only popularity statistics."
        )

    log(f"Loading source dataset from {source_paths['root']}")
    users = load_json(source_paths["users"])
    items = load_json(source_paths["items"])
    split = load_json(source_paths["split"])

    item_lookup = {}
    item_rows = []
    popularity = {}
    for item_key, payload in items.items():
        item_row = _parse_item_row(item_key, payload)
        item_lookup[item_row["item_key"]] = item_row
        popularity[item_row["item_key"]] = 0
        item_rows.append(item_row)

    for train_user_id in split["train"]:
        user_history = users[str(train_user_id)]
        for entry in user_history:
            item_id, _, _ = _parse_interaction(entry)
            if item_id in popularity:
                popularity[item_id] += 1

    item_rows.sort(key=lambda row: row["item_key"])
    item2id = {}
    for index, row in enumerate(item_rows):
        row["item_id"] = index
        row["popularity"] = int(popularity.get(row["item_key"], 0))
        item2id[row["item_key"]] = index

    train_rows = _build_rows(split["train"], users, item_lookup, item2id)
    valid_rows = _build_rows(split["val"], users, item_lookup, item2id)
    test_rows = _build_rows(split["test"], users, item_lookup, item2id)

    header = [
        "user_id",
        "item_asins",
        "item_asin",
        "history_item_id",
        "item_id",
        "history_item_title",
        "item_title",
        "history_rating",
        "rating",
        "history_timestamp",
        "timestamp",
    ]

    for key in ("train_csv", "valid_csv", "test_csv", "catalog_json", "stats_json", "info_txt"):
        ensure_dir(paths[key].parent)

    for csv_path, rows in (
        (paths["train_csv"], train_rows),
        (paths["valid_csv"], valid_rows),
        (paths["test_csv"], test_rows),
    ):
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    write_info_file(paths["info_txt"], item_rows)
    save_json(paths["catalog_json"], {"category": category, "revise": revise, "items": item_rows})
    save_json(
        paths["stats_json"],
        {
            "category": category,
            "revise": revise,
            "num_items": len(item_rows),
            "num_train_users": len(train_rows),
            "num_valid_users": len(valid_rows),
            "num_test_users": len(test_rows),
            "source_root": str(source_paths["root"]),
            "item_file": source_paths["items"].name,
            "preprocess_version": PREPROCESS_VERSION,
            "popularity_scope": "train_only",
            "elapsed_seconds": time.time() - started,
        },
    )

    elapsed = time.time() - started
    log(
        f"Finished preprocessing {category}_{revise}: "
        f"{len(item_rows)} items, {len(train_rows)} train, {len(valid_rows)} valid, {len(test_rows)} test "
        f"in {elapsed:.1f}s"
    )
    return {"skipped": False, "elapsed_seconds": elapsed}


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess datasets for SLATE-Rec.")
    parser.add_argument("--category", required=True, type=str)
    parser.add_argument("--revise", required=True, type=int)
    parser.add_argument("--source_root", default=None, type=str)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    preprocess_category(
        category=args.category,
        revise=maybe_int(args.revise),
        source_root=args.source_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
