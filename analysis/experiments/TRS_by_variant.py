#!/usr/bin/env python3
"""
Generate bar plot showing Weighted ARS by Model and Rubric/Variant Type.

Creates a vertically stacked two-panel bar chart with:
- Top panel: culture-agnostic tasks (`V(En)`, `V(Ko)`)
- Bottom panel: culture-specific tasks (`V(En,US)`, `V(Ko,US)`, `V(En,KR)`, `V(Ko,KR)`)
- X-axis: Models (both Global and Regional)
- Y-axis: Weighted ARS / TRS

Usage:
    python -m analysis_scripts.experiments.TRS_by_variant
    
    # With custom output path
    python -m analysis_scripts.experiments.TRS_by_variant \
        --output results/TRS_by_variant/TRS_by_variant.png
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .base import ExperimentConfig
from .exp_data.loaders import (
    load_all_judgements_with_responses,
    load_harm_classifications,
    merge_harm_dimensions,
    ConsolidatedResult,
)
from .exp_data.filters import (
    filter_culture_agnostic,
    filter_culture_specific,
)
from .exp_data.models import (
    get_model_display_name,
    GLOBAL_MODELS,
    REGIONAL_MODELS,
)
from .exp_data.variants import (
    ORIGINAL_EN,
    TRANSLATED_KR,
    CULTURAL_ADAPTED_EN,
    TRANSCREATED_KR,
)
from .metrics.harm_scores import compute_harm_score_for_result
from .visualization.plots import plot_weighted_ars_by_variant_two_panel


def compute_variant_scores(
    results: List[ConsolidatedResult],
    use_weights: bool = True,
) -> Dict[str, float]:
    """
    Compute mean weighted ARS per variant for a model's results.
    
    Args:
        results: List of ConsolidatedResult for a single model
        use_weights: Whether to use tier-weighted scoring
    
    Returns:
        Dict mapping variant_name -> mean weighted ARS
    """
    variant_scores: Dict[str, List[float]] = defaultdict(list)
    
    for result in results:
        if not result.is_valid or result.is_excluded:
            continue
        
        score_result = compute_harm_score_for_result(result, use_weights=use_weights)
        variant_scores[result.variant_name].append(score_result.harm_score)
    
    # Compute means
    means = {}
    for variant, scores in variant_scores.items():
        if scores:
            means[variant] = sum(scores) / len(scores)
    
    return means


def main():
    parser = argparse.ArgumentParser(
        description="Generate Weighted ARS by Variant bar plot"
    )
    parser.add_argument(
        "--judgements-dir",
        type=str,
        default="runs/rok_fortress_judgements_final",
        help="Directory containing judgement files",
    )
    parser.add_argument(
        "--responses-dir",
        type=str,
        default="runs/rok_fortress_responses_final",
        help="Directory containing response files",
    )
    parser.add_argument(
        "--harm-classifier",
        type=str,
        default="runs/harm-classifier-final/results_gemini_gemini-3-pro-preview.jsonl",
        help="Path to harm classifier results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/TRS_by_variant/TRS_by_variant.png",
        help="Output path for the plot",
    )
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="Only include global models",
    )
    parser.add_argument(
        "--regional-only",
        action="store_true",
        help="Only include regional models",
    )
    
    args = parser.parse_args()
    
    judgements_dir = Path(args.judgements_dir)
    responses_dir = Path(args.responses_dir)
    harm_classifier_path = Path(args.harm_classifier)
    output_path = Path(args.output)
    
    # Print banner
    print("=" * 70)
    print("WEIGHTED ARS BY VARIANT - BAR PLOT GENERATOR")
    print("=" * 70)
    print(f"  Judgements dir: {judgements_dir}")
    print(f"  Responses dir: {responses_dir}")
    print(f"  Output: {output_path}")
    print("=" * 70)
    
    # Load data
    print("\nLoading judgements with response files...")
    data = load_all_judgements_with_responses(
        judgements_dir,
        responses_dir,
        target_models=None,  # Load all
    )
    
    if not data:
        print("ERROR: No data loaded!")
        return 1
    
    print(f"  Loaded {len(data)} models")
    
    # Load and merge harm dimensions
    if harm_classifier_path.exists():
        print("Loading harm classifications...")
        harm_cache = load_harm_classifications(harm_classifier_path)
        for model, results in data.items():
            merge_harm_dimensions(results, harm_cache)
    
    # Compute scores per model
    print("\nComputing weighted ARS by variant and task type...")
    ca_variant_scores: Dict[str, Dict[str, float]] = {}
    cs_variant_scores: Dict[str, Dict[str, float]] = {}
    
    for model, results in data.items():
        ca_results = filter_culture_agnostic(results)
        cs_results = filter_culture_specific(results)
        ca_scores = compute_variant_scores(ca_results, use_weights=True)
        cs_scores = compute_variant_scores(cs_results, use_weights=True)
        ca_variant_scores[model] = ca_scores
        cs_variant_scores[model] = cs_scores
        
        # Print summary
        display_name = get_model_display_name(model)
        print(f"\n  {display_name}:")
        print("    Culture-Agnostic:")
        for variant in [ORIGINAL_EN, TRANSLATED_KR]:
            score = ca_scores.get(variant, 0)
            print(f"      {variant}: {score:.4f}")
        print("    Culture-Specific:")
        for variant in [ORIGINAL_EN, TRANSLATED_KR, CULTURAL_ADAPTED_EN, TRANSCREATED_KR]:
            score = cs_scores.get(variant, 0)
            print(f"      {variant}: {score:.4f}")
    
    # Filter models based on flags
    if args.global_only:
        models_to_plot = [m for m in data.keys() if any(
            m.endswith(g) or g in m for g in GLOBAL_MODELS
        )]
        title_suffix = " (Global Models)"
    elif args.regional_only:
        models_to_plot = [m for m in data.keys() if any(
            m.endswith(r) or r in m for r in REGIONAL_MODELS
        )]
        title_suffix = " (Regional Models)"
    else:
        models_to_plot = list(data.keys())
        title_suffix = ""
    
    # Sort models: Global first, then Regional
    def model_sort_key(m):
        is_regional = any(r in m for r in REGIONAL_MODELS)
        return (is_regional, get_model_display_name(m))
    
    models_to_plot = sorted(models_to_plot, key=model_sort_key)
    
    # Get display names
    display_names = [get_model_display_name(m) for m in models_to_plot]
    
    # Build scores dicts with display names as keys
    ca_scores_for_plot = {
        get_model_display_name(m): ca_variant_scores[m]
        for m in models_to_plot
    }
    cs_scores_for_plot = {
        get_model_display_name(m): cs_variant_scores[m]
        for m in models_to_plot
    }
    
    # Generate plot
    print(f"\nGenerating plot with {len(models_to_plot)} models...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_weighted_ars_by_variant_two_panel(
        models=display_names,
        ca_variant_scores=ca_scores_for_plot,
        cs_variant_scores=cs_scores_for_plot,
        title="",
        output_path=output_path,
    )
    
    print(f"\n  Saved: {output_path}")
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
