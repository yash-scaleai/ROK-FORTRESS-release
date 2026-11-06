# Analysis Code

Scripts that produce the figures and tables in the ROK-FORTRESS paper. Each top-level script is self-contained; shared infrastructure (data loading, filtering, plotting primitives) lives in subpackages.

## Layout

```
analysis/
├── direct_requests/
│   └── compliance_by_variant.py            # → direct_request_compliance_two_panel.png
└── experiments/
    ├── TRS_by_variant.py                   # → TRS_by_variant.png
    ├── binary_linguistic_contextual_drop.py # → binary_linguistic_contextual_drop.png
    ├── linguistic_contextual_drop.py       # → linguistic_contextual_drop.png (Δ_ling / Δ_ctx)
    ├── linguistic_contextual_interaction.py # → linguistic_contextual_interaction.png
    ├── meta_linguistic_contextual_drop.py  # → meta_linguistic_contextual_drop.png
    ├── figures/
    │   ├── gen_ars_ors_quadrant.py         # → fig_quadrant_overall.pdf
    │   ├── gen_model_tactic_heatmap.py     # → fig_tactic_heatmap_*.pdf
    │   └── gen_cross_variant_panel_analysis.py # → fig_cross_variant_panel_analysis.pdf
    ├── multijail_baseline/
    │   └── plots.py                        # → gap_comparison.png, urs_by_language.png
    ├── exp_data/                           # data loaders + filters + variant constants
    ├── metrics/                            # alignment metrics, compliance, harm-score helpers
    ├── visualization/                      # heatmap / quadrant / bar plotting primitives
    ├── plotting/                           # paper style (colors, sizes)
    ├── base.py                             # ExperimentConfig
    └── entity_mappings.py                  # Korean entity → display name maps
```

`scripts/metaeval/domain_drops.py` (under `scripts/`, not `analysis/`) produces `fig_domain_drops.pdf` (Table 18). The judge / harm-classifier calibration scripts under `scripts/metaeval/` produce the calibration tables in the appendix.

## Running

Each figure script accepts `--help` for its own flags. Most read consolidated judgement output from `runs/rok_fortress_judgements_final/` (produced by Stage 2 of the paper pipeline — see the repo-root README).

```bash
uv run python -m analysis.experiments.TRS_by_variant
uv run python -m analysis.experiments.linguistic_contextual_drop
uv run python -m analysis.experiments.figures.gen_ars_ors_quadrant
uv run python -m analysis.direct_requests.compliance_by_variant
```

## Data loading

`analysis.experiments.exp_data` provides the consolidated-result loader the figure scripts share:

```python
from analysis.experiments.exp_data import (
    load_all_judgements_with_responses,
    filter_culture_specific,
)
results = load_all_judgements_with_responses()
cs = filter_culture_specific(results)
```

The default judge panel (HyperCLOVA X + o4-mini + Kimi K2) is defined in `rok_fortress_eval/core/config.py:DEFAULT_JUDGE_MODELS`.
