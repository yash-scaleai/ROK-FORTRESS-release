# ROK-FORTRESS Evaluation Framework

Modular building blocks the paper-flow scripts (under `scripts/`) compose to produce paper results. This package is a library — there is no top-level `run_eval` entry point. See the repository root README for the four-stage paper pipeline.

## Structure

```
rok_fortress_eval/
├── core/
│   ├── config.py           # Default judge panel, parsing model, timeouts
│   ├── data_loader.py      # TSV loading and variant expansion
│   ├── api_client.py       # Async API client (LiteLLM) with retry
│   └── judge.py            # 3-judge ensemble + consensus voting + refusal check
├── vllm/                   # Local GPU inference (router, load balancer, model configs)
├── harm_classifier/        # Harm-dimension classification utilities
└── models/
    └── data_models.py      # Pydantic/dataclass models
```

## Default judge panel

Defined in `core/config.py:DEFAULT_JUDGE_MODELS`:

```
vllm/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B
openai/o4-mini
fireworks_ai/kimi-k2-thinking
```

Consumed by `scripts/run_full_judging.py`. HyperCLOVA X runs locally via vLLM (`scripts/launch_vllm.sh`).

## Data models

- **PromptVariant** — single prompt variant (original_en, translated_kr, cultural_adapted_en, transcreated_kr)
- **ModelResponse** — response from the evaluated model
- **JudgeEvaluation** — raw judge verdicts + reasoning per judge
- **ResponseEvaluation** — aggregated evaluation (final score, refusal flag, per-judge votes)
