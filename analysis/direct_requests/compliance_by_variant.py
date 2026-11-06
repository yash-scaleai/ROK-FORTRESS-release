#!/usr/bin/env python3
"""
Compliance Rate by Model and Variant (Direct Requests).

Analogous to TRS_by_variant.py but using binary compliance/refusal
instead of tier-weighted harm scores from rubric judging.

Generates two outputs:
  1. Single-panel bar chart (pooled CA+CS, original behavior)
  2. Two-panel bar chart (CA top, CS bottom) matching the TRS layout

Usage:
    python -m analysis.direct_requests.compliance_by_variant

    python -m analysis.direct_requests.compliance_by_variant \
        --compliance-dir runs/direct-request-compliance \
        --output results/direct_requests/compliance_by_variant.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from analysis.experiments.exp_data.direct_request_loaders import (
    load_direct_request_results,
)
from analysis.experiments.exp_data.filters import (
    filter_culture_agnostic,
    filter_culture_specific,
)
from analysis.experiments.exp_data.models import (
    get_model_display_name,
    GLOBAL_MODELS,
    REGIONAL_MODELS,
    normalize_model_name,
)
from analysis.experiments.exp_data.variants import (
    ORIGINAL_EN,
    TRANSLATED_KR,
    CULTURAL_ADAPTED_EN,
    TRANSCREATED_KR,
)
from analysis.experiments.metrics.compliance import (
    compute_compliance_by_variant,
)
from analysis.experiments.visualization.plots import (
    plot_weighted_ars_by_variant,
    plot_weighted_ars_by_variant_two_panel,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compliance Rate by Variant bar plot (Direct Requests)"
    )
    parser.add_argument(
        "--compliance-dir", type=str,
        default="runs/direct-request-compliance",
        help="Directory containing refusal check JSONL files",
    )
    parser.add_argument(
        "--output", type=str,
        default="results/direct_requests/compliance_by_variant.png",
        help="Output path for the single-panel plot",
    )
    parser.add_argument(
        "--global-only", action="store_true",
        help="Only include global models",
    )
    parser.add_argument(
        "--regional-only", action="store_true",
        help="Only include regional models",
    )
    args = parser.parse_args()

    compliance_dir = Path(args.compliance_dir)
    output_path = Path(args.output)

    # Print banner
    print("=" * 70)
    print("COMPLIANCE RATE BY VARIANT (DIRECT REQUESTS)")
    print("=" * 70)
    print(f"  Compliance dir: {compliance_dir}")
    print(f"  Output: {output_path}")
    print("=" * 70)

    # Load data
    print("\nLoading compliance data...")
    data = load_direct_request_results(compliance_dir, prompt_type="direct")

    if not data:
        print("ERROR: No data loaded!")
        return 1

    print(f"  Loaded {len(data)} models")
    for model, results in sorted(data.items()):
        print(f"    {model}: {len(results)} results")

    # Compute compliance rate per variant per model (pooled, CA, CS)
    print("\nComputing compliance rates by variant...")
    all_variant_scores: Dict[str, Dict[str, float]] = {}
    ca_variant_scores: Dict[str, Dict[str, float]] = {}
    cs_variant_scores: Dict[str, Dict[str, float]] = {}
    all_variant_metrics = {}

    for model, results in data.items():
        normalized = normalize_model_name(model)

        # Pooled (all tasks)
        by_variant = compute_compliance_by_variant(results)
        variant_scores = {
            variant: cr.compliance_rate
            for variant, cr in by_variant.items()
        }
        all_variant_scores[normalized] = variant_scores
        all_variant_metrics[normalized] = {
            variant: {
                "compliance_rate": cr.compliance_rate,
                "refusal_rate": cr.refusal_rate,
                "complied_count": cr.complied_count,
                "refused_count": cr.refused_count,
                "total_count": cr.total_count,
            }
            for variant, cr in by_variant.items()
        }

        # Culture-agnostic
        ca_results = filter_culture_agnostic(results)
        ca_by_variant = compute_compliance_by_variant(ca_results)
        ca_variant_scores[normalized] = {
            variant: cr.compliance_rate
            for variant, cr in ca_by_variant.items()
        }

        # Culture-specific
        cs_results = filter_culture_specific(results)
        cs_by_variant = compute_compliance_by_variant(cs_results)
        cs_variant_scores[normalized] = {
            variant: cr.compliance_rate
            for variant, cr in cs_by_variant.items()
        }

        display_name = get_model_display_name(normalized)
        print(f"\n  {display_name}:")
        print("    Culture-Agnostic:")
        for variant in [ORIGINAL_EN, TRANSLATED_KR]:
            cr = ca_by_variant.get(variant)
            if cr:
                print(f"      {variant}: {cr.compliance_rate:.4f} ({cr.complied_count}/{cr.total_count})")
        print("    Culture-Specific:")
        for variant in [ORIGINAL_EN, TRANSLATED_KR, CULTURAL_ADAPTED_EN, TRANSCREATED_KR]:
            cr = cs_by_variant.get(variant)
            if cr:
                print(f"      {variant}: {cr.compliance_rate:.4f} ({cr.complied_count}/{cr.total_count})")

    # Filter models
    if args.global_only:
        models_to_plot = [m for m in all_variant_scores if any(
            m.endswith(g) or g in m for g in GLOBAL_MODELS
        )]
        title_suffix = " (Global Models)"
    elif args.regional_only:
        models_to_plot = [m for m in all_variant_scores if any(
            m.endswith(r) or r in m for r in REGIONAL_MODELS
        )]
        title_suffix = " (Regional Models)"
    else:
        models_to_plot = list(all_variant_scores.keys())
        title_suffix = ""

    # Sort: Global first, then Regional
    def model_sort_key(m):
        is_regional = any(r in m for r in REGIONAL_MODELS)
        return (is_regional, get_model_display_name(m))

    models_to_plot = sorted(models_to_plot, key=model_sort_key)
    display_names = [get_model_display_name(m) for m in models_to_plot]

    scores_for_plot = {
        get_model_display_name(m): all_variant_scores[m]
        for m in models_to_plot
    }
    ca_scores_for_plot = {
        get_model_display_name(m): ca_variant_scores[m]
        for m in models_to_plot
    }
    cs_scores_for_plot = {
        get_model_display_name(m): cs_variant_scores[m]
        for m in models_to_plot
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Single-panel plot (original behavior)
    print(f"\nGenerating single-panel plot with {len(models_to_plot)} models...")
    plot_weighted_ars_by_variant(
        models=display_names,
        variant_scores=scores_for_plot,
        title=f"Direct Request Compliance Rate by Model and Variant{title_suffix}",
        output_path=output_path,
        show_mean_line=True,
        ylabel="Compliance Rate",
    )
    print(f"  Saved: {output_path}")

    # 2. Two-panel plot (CA top, CS bottom)
    two_panel_path = output_path.with_name("compliance_by_variant_two_panel.png")
    print(f"\nGenerating two-panel plot...")
    plot_weighted_ars_by_variant_two_panel(
        models=display_names,
        ca_variant_scores=ca_scores_for_plot,
        cs_variant_scores=cs_scores_for_plot,
        title=f"Direct Request Compliance Rate by Model and Prompt Variants{title_suffix}",
        output_path=two_panel_path,
        ylabel="Compliance Rate",
    )
    print(f"  Saved: {two_panel_path}")

    metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_variant_metrics, f, indent=2)

    print(f"  Metrics saved: {metrics_path}")
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
