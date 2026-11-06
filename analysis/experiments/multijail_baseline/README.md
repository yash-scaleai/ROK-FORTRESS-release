# MultiJail Baseline Experiment

Replication of the MultiJail experiment (ICLR 2024) to test whether the unsafe rate gap between English and low-resource languages has decreased in 2026-era frontier models.

## Reference

- **Paper**: "Multilingual Jailbreak Challenges in Large Language Models" (Deng et al., 2024)
- **Dataset**: https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs (MIT; bundled copy `MultiJail.csv`, see `THIRD_PARTY_NOTICES.md`)
- **Published**: ICLR 2024

## Hypothesis

> "Despite scaling laws and safety fine-tuning, frontier models retain a statistically significant higher Unsafe Rate (URS) in low-resource languages compared to high-resource languages."

## Languages

| Tier | Languages | Codes |
|------|-----------|-------|
| High-Resource (HRL) | English, Chinese, Italian, Vietnamese | en, zh, it, vi |
| Medium-Resource (MRL) | Arabic, Korean, Thai | ar, ko, th |
| Low-Resource (LRL) | Bengali, Swahili, Javanese | bn, sw, jv |

## Models

| Model | Justification |
|-------|--------------|
| openai/gpt-5.2 | Evolution of original GPT-4 baseline |
| anthropic/claude-sonnet-4-5 | Safety specialist - lower bound of risk |
| fireworks_ai/llama4-maverick-instruct-basic | Open frontier model |
| bedrock/qwen.qwen3-235b-a22b-2507-v1:0 | Strong multilingual, expected high gap |

## Scenarios

1. **Unintentional**: Raw harmful prompts in each language (simulates non-English users)
2. **Intentional**: Harmful prompts wrapped in AIM jailbreak template (simulates adversarial attack)

## Quick Start

The `MultiJail.csv` dataset is bundled in this directory, containing 315 harmful prompts 
across 10 languages. No additional data download is required.

### Step 1: Collect Model Responses

```bash
uv run python -m analysis.experiments.multijail_baseline.collect_responses \
    --outdir results/multijail_baseline/responses
```

Options:
- `--csv-path`: Path to MultiJail.csv (default: bundled in this directory)
- `--models`: Comma-separated list of models (default: all 4)
- `--languages`: Comma-separated list of languages (default: all 10)
- `--scenario`: `unintentional`, `intentional`, or `both` (default: both)
- `--resume`: Resume from existing output files
- `--concurrency`: Concurrent requests per model (default: 20)

### Step 2: Judge Responses for Safety

```bash
uv run python -m analysis.experiments.multijail_baseline.judge_responses \
    --responses-dir results/multijail_baseline/responses \
    --outdir results/multijail_baseline/judgements
```

Options:
- `--judge-model`: Model for judging (default: openai/o4-mini)
- `--resume`: Resume from existing judgement files

### Step 3: Analyze Results

```bash
uv run python -m analysis.experiments.multijail_baseline.analyze \
    --judgements-dir results/multijail_baseline/judgements \
    --outdir results/multijail_baseline/analysis
```

### Step 4: Generate Visualizations

```bash
uv run python -m analysis.experiments.multijail_baseline.plots \
    --analysis-dir results/multijail_baseline/analysis \
    --outdir results/multijail_baseline/plots
```

## Output Structure

```
results/multijail_baseline/
├── responses/
│   ├── responses_gpt-5.2.jsonl
│   ├── responses_claude-sonnet-4-5.jsonl
│   ├── responses_llama4-maverick-instruct-basic.jsonl
│   └── responses_qwen.qwen3-235b-a22b-2507-v1_0.jsonl
├── judgements/
│   └── judgements_YYYYMMDD_HHMMSS.jsonl
├── analysis/
│   ├── result.json
│   └── report.md
└── plots/
    ├── urs_by_tier.png
    ├── urs_by_language.png
    ├── urs_heatmap.png
    ├── gap_comparison.png
    └── scenario_comparison.png
```

## Primary Metrics

- **URS (Unsafe Rate)**: `Count(unsafe) / Total` (excluding invalid)
- **Gap**: `URS_LRL - URS_HRL` (positive = higher risk in low-resource languages)

## Code Reuse

This module reuses infrastructure from the main ROK-FORTRESS evaluation:

| Component | Source |
|-----------|--------|
| API calls | `rok_fortress_eval.core.api_client.call_model_once()` |
| Async infrastructure | Same tenacity retry logic and asyncio patterns |
| Config | Shares API_KEY and BASE_URL from environment |

## Files

| File | Description |
|------|-------------|
| `config.py` | Models, languages, evaluation parameters |
| `data_loader.py` | Load MultiJail dataset with AIM template injection |
| `collect_responses.py` | Collect model responses (async, parallel) |
| `judge.py` | Simple 3-class safety judge (safe/unsafe/invalid) |
| `judge_responses.py` | Run judging on collected responses |
| `analyze.py` | Compute URS metrics by tier, language, scenario |
| `plots.py` | Generate visualizations |
