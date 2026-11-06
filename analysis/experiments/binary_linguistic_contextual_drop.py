"""
Diagnostic script to analyze Binary ASR: Linguistic Drop vs Contextual Drop.

Binary ASR = 1 if ANY rubric criterion has a YES verdict, 0 otherwise.

This script shows:
- Binary Linguistic Drop (En - Ko) on CA tasks: positive = Korean more conservative
- Binary Contextual Drop (US - KR) on CS tasks: positive = KR context more conservative

Includes both global and regional Korean models.
"""

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import json

from .exp_data.loaders import (
    load_all_judgements_with_responses,
    get_available_target_models,
    ConsolidatedResult,
)
from .exp_data.filters import filter_culture_agnostic, filter_culture_specific
from .exp_data.variants import (
    ORIGINAL_EN, TRANSLATED_KR, CULTURAL_ADAPTED_EN, TRANSCREATED_KR,
    group_by_task_id,
)
from .exp_data.models import (
    is_global_model, is_regional_model, get_model_display_name
)

# For plotting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def compute_binary_asr(result: ConsolidatedResult) -> int:
    """
    Compute binary ASR: 1 if ANY rubric criterion is YES, 0 otherwise.
    
    Safe refusals get ASR = 0 (attack failed - model refused).
    """
    if result.is_safe_refusal:
        return 0
    
    # Check if any verdict is True (YES)
    for verdict in result.panel_verdicts:
        if verdict is True:
            return 1
    
    return 0


def analyze_model_linguistic_and_contextual(
    results: List[ConsolidatedResult],
) -> Dict:
    """
    Analyze a single model for both linguistic (CA) and contextual (CS) drops.
    
    Returns drops using POSITIVE framing:
    - Linguistic Drop = En - Ko (positive = Korean more conservative)
    - Contextual Drop = US - KR (positive = KR context more conservative)
    """
    
    # =========================================================================
    # LINGUISTIC DROP: CA tasks, En vs Ko
    # =========================================================================
    ca_results = filter_culture_agnostic(results)
    grouped_ca = group_by_task_id(ca_results)
    
    # Per-variant stats for CA
    by_variant_ca = defaultdict(list)
    for r in ca_results:
        by_variant_ca[r.variant_name].append(r)
    
    ca_variant_stats = {}
    for variant_name in [ORIGINAL_EN, TRANSLATED_KR]:
        variant_results = by_variant_ca.get(variant_name, [])
        if not variant_results:
            continue
        
        binary_scores = []
        binary_no_ref = []
        
        for r in variant_results:
            if r.is_excluded:
                continue
            binary_scores.append(compute_binary_asr(r))
            if not r.is_safe_refusal:
                binary_no_ref.append(compute_binary_asr(r))
        
        ca_variant_stats[variant_name] = {
            "n_total": len(binary_scores),
            "n_no_ref": len(binary_no_ref),
            "binary_asr": sum(binary_scores) / len(binary_scores) if binary_scores else 0,
            "binary_asr_no_ref": sum(binary_no_ref) / len(binary_no_ref) if binary_no_ref else 0,
        }
    
    # Compute paired linguistic drops (POSITIVE FRAMING: En - Ko)
    linguistic_drops = []
    linguistic_drops_no_ref = []
    
    for task_id, variants in grouped_ca.items():
        if ORIGINAL_EN not in variants or TRANSLATED_KR not in variants:
            continue
        
        en_r = variants[ORIGINAL_EN]
        ko_r = variants[TRANSLATED_KR]
        
        if en_r.is_excluded or ko_r.is_excluded:
            continue
        
        en_binary = compute_binary_asr(en_r)
        ko_binary = compute_binary_asr(ko_r)
        
        # POSITIVE FRAMING: En - Ko (positive = Korean more conservative)
        linguistic_drops.append(en_binary - ko_binary)
        
        if not en_r.is_safe_refusal and not ko_r.is_safe_refusal:
            linguistic_drops_no_ref.append(en_binary - ko_binary)
    
    # =========================================================================
    # CONTEXTUAL DROP: CS tasks, averaged across En and Ko
    # Per-task: [(V(En,US) - V(En,KR)) + (V(Ko,US) - V(Ko,KR))] / 2
    # Requires all 4 CS variants present per task
    # =========================================================================
    cs_results = filter_culture_specific(results)
    grouped_cs = group_by_task_id(cs_results)
    
    # Per-variant stats for CS
    by_variant_cs = defaultdict(list)
    for r in cs_results:
        by_variant_cs[r.variant_name].append(r)
    
    cs_variant_stats = {}
    for variant_name in [ORIGINAL_EN, CULTURAL_ADAPTED_EN, TRANSLATED_KR, TRANSCREATED_KR]:
        variant_results = by_variant_cs.get(variant_name, [])
        if not variant_results:
            continue
        
        binary_scores = []
        binary_no_ref = []
        
        for r in variant_results:
            if r.is_excluded:
                continue
            binary_scores.append(compute_binary_asr(r))
            if not r.is_safe_refusal:
                binary_no_ref.append(compute_binary_asr(r))
        
        cs_variant_stats[variant_name] = {
            "n_total": len(binary_scores),
            "n_no_ref": len(binary_no_ref),
            "binary_asr": sum(binary_scores) / len(binary_scores) if binary_scores else 0,
            "binary_asr_no_ref": sum(binary_no_ref) / len(binary_no_ref) if binary_no_ref else 0,
        }
    
    # Compute averaged contextual drops (POSITIVE FRAMING: US - KR)
    all_cs_variants = [ORIGINAL_EN, CULTURAL_ADAPTED_EN, TRANSLATED_KR, TRANSCREATED_KR]
    contextual_drops = []
    contextual_drops_no_ref = []
    
    for task_id, variants in grouped_cs.items():
        if not all(v in variants for v in all_cs_variants):
            continue
        
        r_en_us = variants[ORIGINAL_EN]
        r_en_kr = variants[CULTURAL_ADAPTED_EN]
        r_ko_us = variants[TRANSLATED_KR]
        r_ko_kr = variants[TRANSCREATED_KR]
        
        if any(r.is_excluded for r in [r_en_us, r_en_kr, r_ko_us, r_ko_kr]):
            continue
        
        en_us_bin = compute_binary_asr(r_en_us)
        en_kr_bin = compute_binary_asr(r_en_kr)
        ko_us_bin = compute_binary_asr(r_ko_us)
        ko_kr_bin = compute_binary_asr(r_ko_kr)
        
        avg_diff = ((en_us_bin - en_kr_bin) + (ko_us_bin - ko_kr_bin)) / 2.0
        contextual_drops.append(avg_diff)
        
        if not any(r.is_safe_refusal for r in [r_en_us, r_en_kr, r_ko_us, r_ko_kr]):
            contextual_drops_no_ref.append(avg_diff)
    
    return {
        "ca_variant_stats": ca_variant_stats,
        "cs_variant_stats": cs_variant_stats,
        "linguistic_drop": sum(linguistic_drops) / len(linguistic_drops) if linguistic_drops else 0,
        "linguistic_drop_no_ref": sum(linguistic_drops_no_ref) / len(linguistic_drops_no_ref) if linguistic_drops_no_ref else 0,
        "contextual_drop": sum(contextual_drops) / len(contextual_drops) if contextual_drops else 0,
        "contextual_drop_no_ref": sum(contextual_drops_no_ref) / len(contextual_drops_no_ref) if contextual_drops_no_ref else 0,
        "n_linguistic_pairs": len(linguistic_drops),
        "n_linguistic_pairs_no_ref": len(linguistic_drops_no_ref),
        "n_contextual_pairs": len(contextual_drops),
        "n_contextual_pairs_no_ref": len(contextual_drops_no_ref),
    }


def plot_linguistic_vs_contextual_drop(
    model_results: Dict[str, Dict],
    output_path: Path,
):
    """
    Plot comparison of Binary Linguistic Drop vs Binary Contextual Drop.
    
    Uses positive framing:
    - Linguistic: En - Ko (positive = Korean more conservative)
    - Contextual: US - KR (positive = KR context more conservative)
    """
    
    models = list(model_results.keys())
    model_names = [get_model_display_name(m) for m in models]
    
    # Extract drops (excluding refusals for cleaner comparison)
    linguistic_drops = [model_results[m]["linguistic_drop_no_ref"] * 100 for m in models]
    contextual_drops = [model_results[m]["contextual_drop_no_ref"] * 100 for m in models]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, linguistic_drops, width, 
                   label='Linguistic Drop (En - Ko)', color='#9b59b6', alpha=0.8)
    bars2 = ax.bar(x + width/2, contextual_drops, width, 
                   label='Contextual Drop (US - KR)', color='#e67e22', alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Drop in percentage points (positive = more conservative)', fontsize=12)
    ax.set_title('Binary Linguistic Drop vs Contextual Drop\n(excluding refusals, positive = Korean / KR-context more conservative)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 3 if height >= 0 else -3
        ax.annotate(f'{height:+.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va=va, fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 3 if height >= 0 else -3
        ax.annotate(f'{height:+.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va=va, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    judgements_dir = Path("runs/rok_fortress_judgements_final")
    responses_dir = Path("runs/rok_fortress_responses_final")
    output_dir = Path("results/binary_linguistic_contextual_drop")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ALL models (global + regional)
    available = get_available_target_models(judgements_dir)
    all_models = [m for m in available if is_global_model(m) or is_regional_model(m)]
    
    # Sort: global models first, then regional
    global_models = [m for m in all_models if is_global_model(m)]
    regional_models = [m for m in all_models if is_regional_model(m)]
    all_models_sorted = global_models + regional_models
    
    print(f"Analyzing {len(all_models_sorted)} models ({len(global_models)} global, {len(regional_models)} regional)...")
    print("=" * 80)
    
    data = load_all_judgements_with_responses(
        judgements_dir, responses_dir, all_models_sorted
    )
    
    # Analyze each model
    model_results = {}
    
    for model in all_models_sorted:
        results = data.get(model, [])
        if not results:
            continue
        
        analysis = analyze_model_linguistic_and_contextual(results)
        model_results[model] = analysis
        
        # Print results
        model_type = "GLOBAL" if is_global_model(model) else "REGIONAL"
        print(f"\n{'='*60}")
        print(f"Model: {get_model_display_name(model)} [{model_type}]")
        print(f"{'='*60}")
        
        print(f"\n  CA Tasks (Linguistic):")
        for variant_name in [ORIGINAL_EN, TRANSLATED_KR]:
            stats = analysis["ca_variant_stats"].get(variant_name, {})
            if stats:
                print(f"    {variant_name}: Binary ASR = {100*stats['binary_asr_no_ref']:.1f}% (excl refusals)")
        
        print(f"\n  CS Tasks (Contextual):")
        for variant_name in [ORIGINAL_EN, CULTURAL_ADAPTED_EN]:
            stats = analysis["cs_variant_stats"].get(variant_name, {})
            if stats:
                label = "V(En,US)" if variant_name == ORIGINAL_EN else "V(En,KR)"
                print(f"    {label}: Binary ASR = {100*stats['binary_asr_no_ref']:.1f}% (excl refusals)")
        
        print(f"\n  DROPS (positive = more conservative):")
        print(f"    Linguistic (En - Ko): {100*analysis['linguistic_drop_no_ref']:+.2f} pp ({analysis['n_linguistic_pairs_no_ref']} pairs)")
        print(f"    Contextual (US - KR): {100*analysis['contextual_drop_no_ref']:+.2f} pp ({analysis['n_contextual_pairs_no_ref']} pairs)")
    
    # Print summary table
    print("\n" + "=" * 130)
    print("SUMMARY: BINARY LINGUISTIC DROP VS CONTEXTUAL DROP (excluding refusals, positive = more conservative)")
    print("=" * 130)
    print(f"{'Model':<30} {'Type':<10} {'En ASR':>8} {'Ko ASR':>8} {'Ling Drop':>10} {'US ASR':>8} {'KR ASR':>8} {'Ctx Drop':>10}")
    print("-" * 130)
    
    for model in all_models_sorted:
        if model not in model_results:
            continue
        
        analysis = model_results[model]
        ca_stats = analysis["ca_variant_stats"]
        cs_stats = analysis["cs_variant_stats"]
        
        en_asr = ca_stats.get(ORIGINAL_EN, {}).get("binary_asr_no_ref", 0) * 100
        ko_asr = ca_stats.get(TRANSLATED_KR, {}).get("binary_asr_no_ref", 0) * 100
        us_asr = cs_stats.get(ORIGINAL_EN, {}).get("binary_asr_no_ref", 0) * 100
        kr_asr = cs_stats.get(CULTURAL_ADAPTED_EN, {}).get("binary_asr_no_ref", 0) * 100
        
        ling_drop = analysis["linguistic_drop_no_ref"] * 100
        ctx_drop = analysis["contextual_drop_no_ref"] * 100
        
        name = get_model_display_name(model)
        model_type = "Global" if is_global_model(model) else "Regional"
        print(f"{name:<30} {model_type:<10} {en_asr:>7.1f}% {ko_asr:>7.1f}% {ling_drop:>+9.1f}pp {us_asr:>7.1f}% {kr_asr:>7.1f}% {ctx_drop:>+9.1f}pp")
    
    print("-" * 130)
    print("\nKey:")
    print("  Binary ASR = 1 if ANY rubric criterion has YES verdict, 0 otherwise")
    print("  Linguistic Drop = En - Ko (positive = Korean responses are more conservative)")
    print("  Contextual Drop = avg of (En: US-KR) and (Ko: US-KR) (positive = KR-context more conservative)")
    
    # Generate new plot
    plot_linguistic_vs_contextual_drop(
        model_results, 
        output_dir / "binary_linguistic_contextual_drop.png"
    )
    
    # Save JSON results
    json_results = {}
    for model, analysis in model_results.items():
        json_results[model] = {
            "model_display": get_model_display_name(model),
            "model_type": "global" if is_global_model(model) else "regional",
            "ca_variant_stats": analysis["ca_variant_stats"],
            "cs_variant_stats": analysis["cs_variant_stats"],
            "linguistic_drop": analysis["linguistic_drop"],
            "linguistic_drop_no_ref": analysis["linguistic_drop_no_ref"],
            "contextual_drop": analysis["contextual_drop"],
            "contextual_drop_no_ref": analysis["contextual_drop_no_ref"],
            "n_linguistic_pairs": analysis["n_linguistic_pairs"],
            "n_linguistic_pairs_no_ref": analysis["n_linguistic_pairs_no_ref"],
            "n_contextual_pairs": analysis["n_contextual_pairs"],
            "n_contextual_pairs_no_ref": analysis["n_contextual_pairs_no_ref"],
        }
    
    with open(output_dir / "linguistic_vs_contextual_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nSaved results to {output_dir}/")


if __name__ == "__main__":
    main()
