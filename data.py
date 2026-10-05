import ast
from pathlib import Path
from typing import List

import pandas as pd

from dataset import SourceExample, build_prompt


def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return list(ast.literal_eval(str(value)))


def build_sources(csv_path: Path, tokenizer, category_text: str, max_length: int) -> List[SourceExample]:
    frame = pd.read_csv(csv_path)
    sources = []
    for row in frame.itertuples(index=False):
        history_titles = [str(title) for title in parse_list(row.history_item_title)]
        history_item_ids = [int(item_id) for item_id in parse_list(row.history_item_id)]
        prompt = build_prompt(category_text, history_titles)
        prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        prompt_ids = prompt_ids[-max(64, max_length - 48) :]
        sources.append(
            SourceExample(
                user_id=str(row.user_id),
                prompt=prompt,
                prompt_ids=prompt_ids,
                history_titles=history_titles,
                history_item_ids=history_item_ids,
                output_title=str(row.item_title),
                output_item_id=int(row.item_id),
            )
        )
    return sources
