# SLATE-Rec

`SLATE-Rec` implements the paper's candidate-closed framework for LLM-based recommendation: Set-Normalized LLM Alignment with Traceable Evidence for Recommendation.

The code keeps the shared dataset conventions of the original project, but changes the recommendation logic. Instead of training only with next-title SFT and then repairing open-ended generations, SLATE-Rec constructs catalog-valid candidate slates, aligns the LLM within those slates, and reranks with traceable evidence.

## Pipeline

`raw dataset -> preprocess -> SFT warm start -> TEF candidate slates -> SLA alignment -> EOR reranking -> metrics`

The three paper components are:

- `TEF`: Traceable Evidence Filling, which builds a wide evidence reservoir from projected LLM beams, train-only history-to-target transitions, lexical retrieval, and popularity filling, then compresses it into a deployable slate.
- `SLA`: Set-Normalized Likelihood Alignment, which trains the LLM to compare candidates within retrieval-filled closed slates instead of optimizing only isolated next-title likelihood.
- `EOR`: Evidence-Odds Reranking, which combines aligned LLM title likelihood with deterministic source and evidence metadata for final ranking.

## Input

The source dataset layout is:

```text
datasets/<category>/users.json
datasets/<category>/split.json
datasets/<category>/item_<revise>.json
```

`items_<revise>.json` is also accepted.

## Quick Start

For one 48G GPU, start with Qwen3-1.7B or Qwen3-4B and micro-batch size 1:

```bash
cd SLATE-Rec
bash run.sh All_Beauty 0 qwen17 --sft_epochs 1 --sla_epochs 1 --micro_batch_size 1 --gradient_accumulation_steps 16
```

If SLA alignment is too expensive, keep the LLM frozen after SFT and still use EOR reranking:

```bash
cd SLATE-Rec
bash run.sh All_Beauty 0 qwen17 --freeze_after_sft --sft_epochs 1
```

Evaluation only:

```bash
cd SLATE-Rec
bash evaluate.sh All_Beauty 0 qwen17
```

## Important Flags

- `--wide_size`: wide reservoir budget \(B_W\), default `150`.
- `--slate_size`: deployment slate budget \(N\), default `30`.
- `--transition_k`: transition retrieval budget \(B_T\), default `80`.
- `--retrieval_k`: lexical retrieval budget \(B_R\), default `40`.
- `--null_alpha`: optional null-prompt likelihood-ratio correction weight, default `0`.
- `--ranker`: final ranking mode, one of `fusion`, `llm`, or `evidence`; default `fusion`.
- `--llm_rank_weight`, `--evidence_rank_weight`, `--source_rank_weight`: EOR fusion weights for LLM likelihood, evidence strength, and source prior.
- `--freeze_after_sft`: skip SLA alignment and use the SFT checkpoint for candidate scoring.
- `--disable_auto_skip_sla`: force SLA alignment even when validation diagnostics predict limited benefit.
- `--disable_gradient_checkpointing`: turn off gradient checkpointing. It is enabled by default for single-GPU memory.
- `--force_rebuild_candidates`: regenerate cached TEF slates.

## What Gets Written

All generated files stay inside `SLATE-Rec/`:

- `train/`, `valid/`, `test/`: processed CSV splits.
- `catalog/`, `info/`, `stats/`: normalized item metadata.
- `candidates/`: cached wide reservoirs and deploy/train slates.
- `checkpoints/`: SFT, last, and best checkpoints.
- `outputs/`: ranked top-10 prediction JSON files.
- `metrics/`: HR/NDCG metric JSON files.

## Notes

- LoRA/PEFT is intentionally not used.
- Training is full-parameter and single-process.
- `bf16` is used when supported, otherwise `fp16` on CUDA.
- The final top-k list is ranked by aligned LLM token likelihood fused with cached source/evidence priors, not by a separate neural ranker.
- SLA is validation gated and auto-skipped by default when train/valid candidate reachability diverges too much or the SFT ranker already captures most reachable validation positives.
