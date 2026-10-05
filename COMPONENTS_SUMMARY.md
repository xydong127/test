# SLATE-Rec Components Summary

## Overview

`SLATE-Rec` is a self-contained implementation of Set-Normalized LLM Alignment with Traceable Evidence for Recommendation. It follows the paper's exposure-to-ranking view: build a catalog-valid decision space, align the LLM on closed slates, and use trace evidence to calibrate final ranking.

The method flow is:

`preprocess -> SFT warm start -> TEF candidate slates -> optional SLA alignment -> EOR evaluation -> metrics`

## Files

| File | Role |
| --- | --- |
| `preprocess.py` | Converts shared raw datasets into local CSV/catalog artifacts. |
| `data.py` | Loads processed CSV rows into prompt-based `SourceExample` records. |
| `dataset.py` | Defines prompt, SFT, prompt-only, and candidate-slate datasets/collators. |
| `candidate_builder.py` | Implements TEF by building wide reservoirs, deploy slates, train slates, and reachability diagnostics. |
| `llm_candidate_scoring.py` | Computes length-normalized LLM candidate-title likelihood and EOR fusion scores. |
| `candidate_closed_loss.py` | Implements SLA listwise and pairwise odds losses. |
| `train.py` | Runs full-parameter SFT warm start, candidate caching, and optional SLA alignment. |
| `evaluate.py` | Builds deployment slates and ranks candidates with EOR. |
| `metrics.py` | Computes HR and NDCG from prediction JSON. |
| `utils.py` | Path, model, catalog, retrieval, transition, JSON, and metric helpers. |
| `run.sh` | Full preprocess/train/evaluate/metrics pipeline. |
| `evaluate.sh` | Evaluation and metrics only. |
| `ablation.sh` | SLATE-Rec component ablations. |
| `hyperparameter_analysis.sh` | Sensitivity analysis for \(B_W\), \(N\), \(B_T\), and \(w_L\). |

## Algorithmic Components

### TEF: Traceable Evidence Filling

TEF constructs a wide reservoir from projected LLM beams, train-only history-to-target transition evidence, lexical catalog retrieval, and popularity filling. Each candidate keeps source identity, source rank, and evidence strength, so the system can trace how the item reached the reservoir and deployable slate.

The reservoir is compressed into a smaller `slate_size` before LLM scoring with a source-preserving selector. This prevents one evidence source from dominating the compact slate and reduces compressed-evidence failures.

Main file:

- `candidate_builder.py`

### SLA: Set-Normalized Likelihood Alignment

Training examples are closed slates rather than single target strings. The gold next item is added to the training slate when the deployment slate misses it, and the code records this as `oracle_injected` so evaluation remains honest.

SLA optimizes listwise and pairwise losses over the TEF-induced slate, while retaining the SFT title-likelihood signal. In implementation, SLA is validation-gated and can be auto-skipped when the train/valid reachability gap is too large or the SFT ranker already captures most reachable validation positives.

Main files:

- `candidate_closed_loss.py`
- `llm_candidate_scoring.py`
- `train.py`

### EOR: Evidence-Odds Reranking

At evaluation, candidates are ranked by length-normalized LLM title likelihood fused with trace metadata from TEF. The fusion uses the aligned LLM score, accumulated evidence strength, and a source-rank prior. Optional null-prompt correction is controlled by `--null_alpha`, and `--ranker llm` restores pure LLM likelihood ranking for ablation.

Main files:

- `llm_candidate_scoring.py`
- `evaluate.py`

## Hardware Defaults

The defaults are chosen for one 48GB GPU:

- full-parameter training, no LoRA;
- micro-batch size `1`;
- gradient accumulation `16`;
- gradient checkpointing enabled by default;
- Qwen3-1.7B (`qwen17`) or Qwen3-4B (`qwen4`) recommended.
