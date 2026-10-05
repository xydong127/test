#!/usr/bin/env python3
"""Run the existing SLATE-Rec stages with additional backbone aliases.

This keeps model registration local to this new runner and does not edit utils.py.
"""

import gc
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

EXTRA_MODELS = {
    "gemma2b": "unsloth/gemma-2-2b-it",
    "llama32_1b": "unsloth/Llama-3.2-1B-Instruct",
    "olmo2_1b": "unsloth/OLMo-2-0425-1B-Instruct",
}


def run_stage(script_name: str, args: list[str]) -> None:
    script_path = ROOT / script_name
    sys.argv = [str(script_path), *args]
    runpy.run_path(str(script_path), run_name="__main__")
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: review_backbone_run.py CATEGORY REVISE LLM [training/evaluation arguments...]"
        )

    category, revise, llm = sys.argv[1:4]
    extra_args = sys.argv[4:]

    import utils

    if llm in EXTRA_MODELS:
        utils.MODEL_REGISTRY[llm] = EXTRA_MODELS[llm]
        print(f"Backbone {llm} -> {EXTRA_MODELS[llm]}", flush=True)

    base_args = ["--category", category, "--revise", revise]
    run_stage("preprocess.py", base_args)

    model_args = [*base_args, "--llm", llm, *extra_args]
    run_stage("train.py", model_args)
    run_stage("evaluate.py", model_args)
    run_stage("metrics.py", model_args)


if __name__ == "__main__":
    main()
