# ROK-FORTRESS

**Measuring the Effect of Geopolitical Transcreation for National Security and Public Safety**

ROK-FORTRESS is a bilingual safety benchmark that separates the effects of language and geopolitical grounding on LLM safety behavior. Building on [FORTRESS](https://arxiv.org/pdf/2506.14922), we use a **transcreation matrix** to evaluate adversarial prompts under controlled combinations of language (English vs Korean) and cultural context (US vs Korean entities and institutions).

<p align="center">
  <img src="figures/pipeline_updated.png" alt="ROK-FORTRESS Pipeline" width="900">
</p>

## Key Findings

Across the frontier and Korean regional models we evaluated, we observe:

- **9× TRS spread** between safest (Claude Opus 4.5: 0.06) and least safe (Mistral Large 3: 0.52) models
- **Consistent suppression pattern**: mean TRS on culture-specific tasks decreases as V(En,US) → V(En,KR) → V(Ko,US) → V(Ko,KR)
- **Linguistic drop is 2.65× larger than contextual drop** on the same culture-specific tasks (9.2 vs 3.5 pp)
- **Significant interaction effects** in a subset of models (4/14 after Bonferroni correction)—Korean context mitigates language-driven suppression; no model shows significant amplification

<p align="center">
  <img src="figures/TRS_by_variant.png" alt="TRS by Variant" width="800">
</p>

*Mean Tier-weighted Risk Score (TRS) across prompt variants. For most models, both Korean language and Korean cultural context independently suppress harmful outputs.*

## Dataset

The public dataset is hosted on Hugging Face: <https://huggingface.co/datasets/ScaleAI/ROK-FORTRESS_public>. Download it to `data/rok_fortress_public.tsv`, the path every script expects:

```bash
# Gated dataset: accept the terms on the Hugging Face page, then log in once (`hf auth login`).
uv run --with huggingface_hub hf download ScaleAI/ROK-FORTRESS_public rok_fortress_public.tsv \
  --repo-type dataset --local-dir data
```

**791 adversarial tasks** across 4 NSPS risk domains:

| Domain | Culture-Agnostic | Culture-Specific | Total |
|--------|------------------|------------------|-------|
| Chemical, Biological, Radiological, Nuclear, and Explosive (CBRNE) | 138 | 113 | 251 |
| Political Violence & Terrorism | 76 | 133 | 209 |
| Criminal & Financial Illicit Activities | 115 | 133 | 248 |
| Information Leakage | 30 | 53 | 83 |
| **Total** | **359** | **432** | **791** |

### Transcreation Matrix

Each task is evaluated across controlled prompt variants:

- **Culture-Agnostic** (359 tasks): `V(En)` → `V(Ko)`
- **Culture-Specific** (432 tasks): `V(En,US)` → `V(Ko,US)` → `V(En,KR)` → `V(Ko,KR)`

This design disentangles safety degradation due to language processing from degradation due to unfamiliar geopolitical context.

## Metrics

| Metric | Description |
|--------|-------------|
| **TRS** (Tier-weighted Risk Score) | Weighted harm score based on operational severity (Tier 1-3) |
| **ORS** (Over-Refusal Score) | % of benign prompts incorrectly refused |
| **Δ_ling** (Linguistic Drop) | TRS(V_En) − TRS(V_Ko) on culture-agnostic tasks |
| **Δ_ctx** (Contextual Drop) | TRS(V_En,US) − TRS(V_En,KR) on culture-specific tasks |

<p align="center">
  <img src="figures/linguistic_contextual_drop_TRS.png" alt="Linguistic vs Contextual Drop" width="800">
</p>

*Decomposition of TRS reduction into linguistic and contextual components.*

## Evaluation Framework

- **3-judge panel** (default): HyperCLOVA X + o4-mini + Kimi K2 [medium] with majority voting, calibrated against human consensus (F1=0.831, κ=0.736; per-variant F1 ≥ 0.783). HyperCLOVA X runs locally via vLLM — see `scripts/launch_vllm.sh`.
- **Rubric-based scoring**: 1-7 binary criteria per task (μ=6.01)
- **7 harm dimensions**: WHAT, HOW-1/2/3, WHY, WHO/WHERE, WHEN

## Quick Start

```bash
uv sync

# 1. Smoke-test the loader against the public dataset (no API keys needed)
uv run python scripts/test_public_dataset_smoke.py
```

The data loader runs offline. To call models or judges, configure API access (next section).

## API setup

Models are addressed by a `provider/model` prefix (e.g. `anthropic/claude-sonnet-4-5`, `gemini/gemini-3-pro-preview`). The codebase supports two routing modes, picked automatically based on what's in your `.env`:

**Mode A — direct provider keys.** No extra service to launch. `litellm` dispatches each call by its prefix using per-provider env vars.

```bash
cp .env.example .env
# fill in the provider keys you need (OPENAI_API_KEY, ANTHROPIC_API_KEY,
# GEMINI_API_KEY, FIREWORKS_AI_API_KEY, AWS_* for Bedrock).
```

**Mode B — LiteLLM proxy (paper-run path).** All experiments in the paper were routed through a [LiteLLM proxy](https://docs.litellm.ai/docs/proxy/quick_start) with provider keys configured server-side. Set `BASE_URL` (or `OPENAI_BASE_URL`) and the codebase sends every call to that endpoint instead.

```bash
pip install 'litellm[proxy]'
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=... FIREWORKS_AI_API_KEY=...
litellm --config configs/litellm-proxy.example.yaml --port 4000
# In .env: BASE_URL=http://localhost:4000/v1 and API_KEY=sk-anything
```

`configs/litellm-proxy.example.yaml` lists every model used in the paper; trim it to the panel you intend to evaluate.

## Reproducibility

The public 791-task dataset (`data/rok_fortress_public.tsv`, downloaded from Hugging Face as shown under [Dataset](#dataset)) drives the entire pipeline. The paper used the four-stage flow below; intermediate outputs are written to `runs/` and consumed by the next stage.

### Stage 1 — Collect model responses

```bash
# API models
uv run python scripts/collect_model_responses.py \
  --input data/rok_fortress_public.tsv \
  --outdir runs/rok_fortress_responses_final
```

For Korean regional models you need GPUs and a vLLM server (`scripts/launch_vllm.sh`), then use the per-model collectors:

```bash
./scripts/launch_vllm.sh --model naver-hyperclovax/HyperCLOVAX-SEED-Think-32B --tp 2
uv run python scripts/collect_hyperclovax_responses.py \
  --input data/rok_fortress_public.tsv \
  --outdir runs/rok_fortress_responses_final \
  --base-url http://localhost:8000/v1
# Likewise: collect_kanana2_responses.py, collect_kexaone_responses.py, collect_solar_responses.py
```

### Stage 2 — Run the 3-judge ensemble

The paper's panel: HyperCLOVA X (SEED-Think-32B) + o4-mini + Kimi K2 Thinking. Run each judge across all response files:

```bash
# API judges (parallelized via tmux across response files)
./scripts/run_api_judge_parallel.sh --judge o4-mini
./scripts/run_api_judge_parallel.sh --judge kimi-k2-thinking --reasoning medium

# HyperCLOVA X judge — needs vLLM running first
./scripts/launch_vllm.sh --model naver-hyperclovax/HyperCLOVAX-SEED-Think-32B --tp 2
./scripts/run_hyperclovax_judge_parallel.sh
```

Each writes to `runs/rok_fortress_judgements_final/`. Consensus voting (majority of 3) is computed in the analysis stage.

### Stage 3 — Refusal classification

```bash
# Benign over-refusal (ORS)
uv run python scripts/run_refusal_check.py \
  --refusal-model gemini/gemini-3.1-pro-preview \
  --response-dir runs/rok_fortress_responses_final \
  --output-dir runs/rok_fortress_refusal_checks_final \
  --prompt-type benign

# Adversarial refusal
uv run python scripts/run_refusal_check.py \
  --refusal-model gemini/gemini-3.1-pro-preview \
  --response-dir runs/rok_fortress_responses_final \
  --output-dir runs/rok_fortress_refusal_checks_final \
  --prompt-type adversarial
```

### Stage 4 — Harm-dimension classification

```bash
uv run python scripts/classify_harm_dimensions.py \
  --input data/rok_fortress_public.tsv \
  --variant A \
  --outdir runs/harm-classifier-final
```

### Stage 5 — Paper figures

Each figure has its own producer script under `analysis/`:

| Paper figure | Script |
|---|---|
| `TRS_by_variant.png` | `analysis/experiments/TRS_by_variant.py` |
| `binary_linguistic_contextual_drop.png` | `analysis/experiments/binary_linguistic_contextual_drop.py` |
| `linguistic_contextual_drop_TRS.png` (Δ_ling / Δ_ctx decomposition) | `analysis/experiments/linguistic_contextual_drop.py` |
| `linguistic_contextual_interaction.png` | `analysis/experiments/linguistic_contextual_interaction.py` |
| `meta_linguistic_contextual_drop.png` | `analysis/experiments/meta_linguistic_contextual_drop.py` |
| `direct_request_compliance_two_panel.png` (direct-request ablation) | `analysis/direct_requests/compliance_by_variant.py` |
| `fig_quadrant_overall.pdf` | `analysis/experiments/figures/gen_ars_ors_quadrant.py` |
| `fig_tactic_heatmap_*.pdf` | `analysis/experiments/figures/gen_model_tactic_heatmap.py` |
| `fig_cross_variant_panel_analysis.pdf` | `analysis/experiments/figures/gen_cross_variant_panel_analysis.py` |
| `fig_domain_drops.pdf` | `scripts/metaeval/domain_drops.py` |
| `gap_comparison.png`, `urs_by_language.png` (MultiJail baseline) | `analysis/experiments/multijail_baseline/plots.py` |

All figures land in `analysis/experiments/figures/` (or each script's `--output` flag).

## Models Evaluated

**Frontier Models**: GPT-5.2, Claude Opus/Sonnet 4.5, Gemini 3 Pro, o4-mini, Kimi K2, DeepSeek V3.2, Llama 4, Mistral Large 3, Qwen 3

**Korean Regional Models**: HyperCLOVA X, Kanana 2, K-EXAONE, Solar Open

## Project Structure

```
rok_fortress_eval/              # Core evaluation library
├── core/                       # Loader, judge ensemble, API client, config
├── vllm/                       # Local GPU inference (router, load balancer)
├── harm_classifier/            # 7-dimension harm classifier
└── models/                     # PromptVariant / ModelResponse / ... dataclasses

analysis/                       # Paper-figure producers
├── direct_requests/            # Direct-request compliance figure (Section 4.5)
└── experiments/
    ├── exp_data/               # Shared data loaders + filters
    ├── metrics/                # ΔLing/ΔCtx, compliance, harm-score helpers
    ├── visualization/          # Heatmap / quadrant / bar plotting
    ├── plotting/               # Paper style (colors, fonts)
    ├── figures/                # gen_*.py scripts + paper figure outputs
    └── multijail_baseline/     # MultiJail baseline reproduction

scripts/                        # Paper pipeline (collect → judge → refusal → harm-classify)
├── collect_*_responses.py      # Stage 1
├── run_full_judging.py         # Stage 2
├── run_refusal_check.py        # Stage 3
├── classify_harm_dimensions.py # Stage 4
└── metaeval/                   # Calibration scripts (F1, κ, judge-panel selection)

configs/                        # models.yaml + judge_models.yaml
data/                           # rok_fortress_public.tsv + croissant.json
figures/                        # Hero / paper figures
```

## Authors

Michael S. Lee, Yash Maurya, Drew Rein, Bert Herring, Jonathan Nguyen, Kyungho Song, Udari Madhushani Sehwag, Jiyeon Cho, Kaustubh Deshpande, Yeongkyun Jang, Jiyeon Joo, Minn Seok Choi, Evi Fuelle, Christina Q. Knight, Joseph Brandifino, Max Fenkell

Scale AI and Korea AI Safety Institute (KAISI). Contact: yash.maurya@scale.com

## Citation

```bibtex
@inproceedings{lee2026rokfortress,
  title     = {{ROK-FORTRESS}: Measuring the Effect of Geopolitical Transcreation for National Security and Public Safety},
  author    = {Michael S. Lee and Yash Maurya and Drew Rein and Bert Herring and Jonathan Nguyen and Kyungho Song and Udari Madhushani Sehwag and Jiyeon Cho and Kaustubh Deshpande and Yeongkyun Jang and Jiyeon Joo and Minn Seok Choi and Evi Fuelle and Christina Q. Knight and Joseph Brandifino and Max Fenkell},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026},
  eprint    = {2605.14152},
  archivePrefix = {arXiv}
}
```

## License

The codebase is released under the [MIT License](LICENSE). The dataset (`data/rok_fortress_public.tsv`) is released separately under [CC-BY-4.0](data/README.md). If you use this benchmark, please cite the ROK-FORTRESS paper.
