#!/usr/bin/env python3
"""
Generate TRS vs ORS Quadrant Plots.

Creates publication-quality scatter plots showing the trade-off between:
- TRS (Tiered Risk Score): Higher = more harmful responses (weighted by harm dimension severity)
- ORS (Over-Refusal Score): Higher = more refusals on benign prompts

Generates:
- Overall quadrant (all variants combined)
- Per-variant quadrants (original_en, translated_kr, cultural_adapted_en, transcreated_kr)

Output: figures/fig_quadrant_*.pdf
"""

import sys
from pathlib import Path
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from analysis.experiments.plotting import (
    setup_paper_style,
    save_figure,
    FIGURE_DIR,
    FAMILY_COLORS,
    MODEL_FAMILY_MAP,
    get_model_color,
    get_model_family,
)
from analysis.experiments.metrics.harm_scores import (
    DOMAIN_WEIGHTS,
    DOMAIN_NAME_MAP,
    get_dimension_key,
)
from analysis.experiments.figures import _canonical

# =============================================================================
# Configuration
# =============================================================================

JUDGEMENTS_DIR = Path("runs/final-rok-fortress-judgements")
BENIGN_REFUSAL_DIR = Path("runs/final-benign-refusal-checks")
ADVERSARIAL_REFUSAL_DIR = Path("runs/refusal-checks")
HARM_CLASSIFIER_FILE = Path("runs/harm-classifier-full/results_gemini_gemini-3-pro-preview.jsonl")

VARIANTS = ["original_en", "translated_kr", "cultural_adapted_en", "transcreated_kr"]
VARIANT_DISPLAY = {
    "original_en": "Original English",
    "translated_kr": "Translated Korean",
    "cultural_adapted_en": "Cultural Adapted (EN)",
    "transcreated_kr": "Transcreated Korean",
}

# Model display names (with version/size info)
MODEL_DISPLAY = {
    "claude-opus-4-5": "Claude Opus 4.5",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "gpt-5.2": "GPT-5.2",
    "o4-mini": "o4-mini",
    "gemini-3-pro-preview": "Gemini 3 Pro Preview",
    "deepseek-v3p2": "DeepSeek V3.2",
    "kimi-k2-thinking": "Kimi K2 Thinking",
    "llama4-maverick-instruct-basic": "Llama 4 Maverick 17B",
    "mistral.mistral-large-3-675b-instruct": "Mistral Large 3 (675B)",
    "qwen.qwen3-235b-a22b-2507-v1:0": "Qwen3 235B-A22B",
    "hyperclovax": "HyperCLOVA X Seed",
    "track_a_model": "HyperCLOVA X Seed",
    "kanana2": "Kanana 2 (30B-A3B)",
    "kanana-2-30b-a3b-thinking": "Kanana 2 (30B-A3B)",
    "kexaone": "K-EXAONE (236B-A23B)",
    "K-EXAONE-236B-A23B": "K-EXAONE (236B-A23B)",
    "solar_open": "Solar Open (100B)",
    "Solar-Open-100B": "Solar Open (100B)",
}

JUDGE_NAMES = ["kimi_k2", "hyperclovax", "o4_mini"]


def load_harm_classifications() -> Dict[str, Dict]:
    """Deprecated: harm dimensions are merged inside the canonical loader. Kept for main() compat."""
    return {}


def load_judgements() -> Dict[str, Dict[str, List]]:
    """{model_token: {variant: [ConsolidatedResult]}} via the canonical 3-judge loader."""
    return _canonical.load_results_by_model()


def load_benign_refusals() -> Dict[str, Dict[str, Tuple[int, int]]]:
    """{model_token: {variant: (refusals, total)}} via the canonical benign refusal loader."""
    return _canonical.load_benign_ors_counts()


def compute_trs(judgements, harm_classifications=None,
                variant: Optional[str] = None) -> float:
    """Mean tier-weighted TRS (%) over the given consolidated results (canonical scoring)."""
    if variant:
        judgements = [r for r in judgements if r.variant_name == variant]
    return _canonical.trs_percent(judgements)


def compute_ors(benign_refusals: Dict[str, Tuple[int, int]],
                variant: Optional[str] = None) -> float:
    """
    Compute ORS (Over-Refusal Score) from benign refusal data.

    ORS = (benign refusals) / (total benign prompts) * 100
    """
    if variant:
        if variant in benign_refusals:
            refusals, total = benign_refusals[variant]
            return (refusals / total * 100) if total > 0 else 0.0
        return 0.0

    # Overall: sum across all variants
    total_refusals = sum(counts[0] for counts in benign_refusals.values())
    total_prompts = sum(counts[1] for counts in benign_refusals.values())

    return (total_refusals / total_prompts * 100) if total_prompts > 0 else 0.0


def plot_quadrant(
    models: List[str],
    trs_scores: Dict[str, float],
    ors_scores: Dict[str, float],
    title: str,
    output_name: str,
    label_pos: Dict[str, Tuple[float, float, str]] = None,
) -> List[Path]:
    """Create a single quadrant plot. label_pos: model -> (trs, ors, ha) text position, drawn with a leader line."""
    setup_paper_style()

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Collect data points
    points = []
    for model in models:
        if model in trs_scores and model in ors_scores:
            trs = trs_scores[model]
            ors = ors_scores[model]
            points.append({
                "model": model,
                "trs": trs,
                "ors": ors,
                "color": get_model_color(model),
                "family": get_model_family(model),
                "display": MODEL_DISPLAY.get(model, model),
            })

    if not points:
        print(f"No data for {title}")
        plt.close(fig)
        return []

    # Calculate medians
    all_trs = [p["trs"] for p in points]
    all_ors = [p["ors"] for p in points]
    median_trs = np.median(all_trs)
    median_ors = np.median(all_ors)

    # Set axis limits with padding
    max_trs = max(all_trs) + 5
    max_ors = max(max(all_ors) + 3, 15)
    ax.set_xlim(0, max_trs)
    ax.set_ylim(0, max_ors)

    # Draw quadrant backgrounds
    # Top-right (high TRS, high ORS) = worst = red
    # Bottom-left (low TRS, low ORS) = best = green
    ax.axhspan(median_ors, max_ors, xmin=median_trs/max_trs, xmax=1,
               color="#ffcccc", alpha=0.4, zorder=0)
    ax.axhspan(0, median_ors, xmin=0, xmax=median_trs/max_trs,
               color="#ccffcc", alpha=0.4, zorder=0)

    # Draw median lines
    ax.axhline(y=median_ors, color="#888888", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(x=median_trs, color="#888888", linestyle="--", linewidth=1.5, alpha=0.7)

    # Plot points
    for p in points:
        ax.scatter(p["trs"], p["ors"], s=250, c=p["color"],
                  edgecolor="white", linewidth=2, zorder=5, alpha=0.9)

    # Add labels with smart positioning
    for p in points:
        if label_pos and p["model"] in label_pos:
            tx, ty, tha = label_pos[p["model"]]
            txt = ax.annotate(
                p["display"], (p["trs"], p["ors"]), xytext=(tx, ty), textcoords="data",
                ha=tha, va="center", fontsize=9, fontweight="medium", zorder=10,
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6, shrinkA=2, shrinkB=7),
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=3, foreground="white")])
            continue
        # Offset based on position relative to median
        x_off = 8 if p["trs"] < median_trs else -8
        y_off = 8 if p["ors"] < median_ors else -8
        ha = "left" if x_off > 0 else "right"
        va = "bottom" if y_off > 0 else "top"

        txt = ax.annotate(
            p["display"], (p["trs"], p["ors"]),
            xytext=(x_off, y_off), textcoords="offset points",
            ha=ha, va=va, fontsize=9, fontweight="medium",
            zorder=10
        )
        txt.set_path_effects([
            path_effects.withStroke(linewidth=3, foreground="white")
        ])

    # Quadrant labels
    ax.text(median_trs/2, max_ors - 1, "Low Risk\nHigh Over-Refusal",
           ha="center", va="top", fontsize=10, color="#666666", style="italic")
    ax.text(max_trs - 2, max_ors - 1, "High Risk\nHigh Over-Refusal",
           ha="right", va="top", fontsize=10, color="#aa0000", fontweight="bold")
    ax.text(median_trs/2, 1, "Low Risk\nLow Over-Refusal",
           ha="center", va="bottom", fontsize=10, color="#007700", fontweight="bold")
    ax.text(max_trs - 2, 1, "High Risk\nLow Over-Refusal",
           ha="right", va="bottom", fontsize=10, color="#666666", style="italic")

    # Add median annotations
    ax.annotate(f"Median TRS = {median_trs:.2f}%",
               xy=(median_trs, 0.5), fontsize=9, ha="left",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.annotate(f"Median ORS = {median_ors:.2f}%",
               xy=(1, median_ors), fontsize=9, ha="left",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Labels
    ax.set_xlabel("Tiered Risk Score (TRS) %\n← Safer | More Harmful →", fontsize=12, fontweight="bold")
    ax.set_ylabel("Over-Refusal Score (ORS) %\n← Better | Over-Refusing →", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Legend for model families
    families_in_plot = list(set(p["family"] for p in points))
    handles = [mpatches.Patch(color=FAMILY_COLORS.get(f, "#999999"), label=f)
               for f in sorted(families_in_plot)]
    ax.legend(handles=handles, loc="upper left", fontsize=9,
             framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()

    return save_figure(fig, output_name, formats=("pdf",))


# Hand-placed labels for the crowded lower-right cluster of the overall plot (data coords).
# Re-tune if the underlying TRS/ORS values move.
OVERALL_LABEL_POS = {
    "qwen.qwen3-235b-a22b-2507-v1:0": (45.0, 7.85, "center"),
    "mistral.mistral-large-3-675b-instruct": (56.0, 7.05, "right"),
    "kimi-k2-thinking": (45.8, 3.6, "right"),
    "kexaone": (41.5, 2.85, "right"),
    "gemini-3-pro-preview": (41.5, 2.1, "right"),
    "hyperclovax": (49.4, 2.85, "left"),
    "kanana2": (47.3, 1.45, "right"),
}


def main():
    """Generate all quadrant plots."""
    print("Loading judgements...")
    judgements = load_judgements()
    print(f"  Found {len(judgements)} models")

    print("Loading harm classifications...")
    harm_classifications = load_harm_classifications()
    print(f"  Found {len(harm_classifications)} task classifications")

    print("Loading benign refusals...")
    benign_refusals = load_benign_refusals()
    print(f"  Found {len(benign_refusals)} models")

    # Get common models
    models = sorted(set(judgements.keys()) & set(benign_refusals.keys()))
    print(f"\nCommon models: {len(models)}")

    # Compute overall scores
    print("\nComputing overall scores (weighted TRS)...")
    overall_trs = {}
    overall_ors = {}

    for model in models:
        all_judgements = []
        for variant_judgements in judgements[model].values():
            all_judgements.extend(variant_judgements)

        overall_trs[model] = compute_trs(all_judgements, harm_classifications)
        overall_ors[model] = compute_ors(benign_refusals.get(model, {}))

        print(f"  {model}: TRS={overall_trs[model]:.1f}%, ORS={overall_ors[model]:.1f}%")

    # Plot overall
    print("\nGenerating overall quadrant...")
    plot_quadrant(
        models, overall_trs, overall_ors,
        "Model Safety-Utility Trade-off: TRS vs ORS (Overall)",
        "fig_quadrant_overall",
        label_pos=OVERALL_LABEL_POS,
    )

    # Plot per variant
    for variant in VARIANTS:
        print(f"\nGenerating quadrant for {variant}...")
        variant_trs = {}
        variant_ors = {}

        for model in models:
            model_judgements = judgements[model].get(variant, [])
            if model_judgements:
                variant_trs[model] = compute_trs(model_judgements,
                                                  harm_classifications,
                                                  variant)

            model_refusals = benign_refusals.get(model, {})
            if variant in model_refusals:
                variant_ors[model] = compute_ors(model_refusals, variant)

        if variant_trs and variant_ors:
            plot_quadrant(
                [m for m in models if m in variant_trs and m in variant_ors],
                variant_trs, variant_ors,
                f"Model Safety-Utility Trade-off: TRS vs ORS ({VARIANT_DISPLAY[variant]})",
                f"fig_quadrant_{variant}"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
