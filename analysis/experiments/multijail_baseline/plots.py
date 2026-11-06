#!/usr/bin/env python3
"""
Generate visualizations for MultiJail baseline experiment.

Creates plots comparing URS across languages, tiers, and models.

Usage:
    uv run python -m analysis.experiments.multijail_baseline.plots \
        --analysis-dir results/multijail_baseline/analysis \
        --outdir results/multijail_baseline/plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .config import (
    LANGUAGES,
    LANGUAGE_TIERS,
    LANGUAGE_NAMES,
    MULTIJAIL_MODELS,
    RESULTS_DIR,
)


# Color scheme
TIER_COLORS = {
    "HRL": "#2ecc71",  # Green
    "MRL": "#f1c40f",  # Yellow
    "LRL": "#e74c3c",  # Red
}

MODEL_COLORS = {
    "openai/gpt-5.2": "#10a37f",
    "anthropic/claude-sonnet-4-5": "#cc785c",
    "fireworks_ai/llama4-maverick-instruct-basic": "#6366f1",
    "bedrock/qwen.qwen3-235b-a22b-2507-v1:0": "#0ea5e9",
}


def load_analysis_result(analysis_dir: Path) -> Dict:
    """Load analysis result from JSON."""
    result_path = analysis_dir / "result.json"
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_urs_by_tier(
    result: Dict,
    output_path: Path,
    scenario: Optional[str] = None,
):
    """
    Bar chart: URS by language tier for each model.
    
    This is the primary visualization for the "Gap" hypothesis.
    Args:
        scenario: If provided, use scenario-specific data. None = combined.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(result["by_model_tier"].keys())
    tiers = ["HRL", "MRL", "LRL"]
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, tier in enumerate(tiers):
        urs_values = []
        for model in models:
            if scenario and "by_model_tier_scenario" in result:
                # Scenario-specific
                if model in result["by_model_tier_scenario"]:
                    tier_data = result["by_model_tier_scenario"][model].get(tier, {})
                    urs_values.append(tier_data.get(scenario, {}).get("urs", 0))
                else:
                    urs_values.append(0)
            else:
                # Combined
                if tier in result["by_model_tier"][model]:
                    urs_values.append(result["by_model_tier"][model][tier]["urs"])
                else:
                    urs_values.append(0)
        
        bars = ax.bar(
            x + i * width,
            urs_values,
            width,
            label=f"{tier} ({['High', 'Medium', 'Low'][i]}-Resource)",
            color=TIER_COLORS[tier],
            edgecolor="black",
            linewidth=0.5,
        )
        
        # Add value labels
        for bar, val in zip(bars, urs_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Unsafe Rate (%)", fontsize=12)
    
    title = "Unsafe Rate by Language Resource Tier"
    if scenario:
        title += f" ({scenario.title()} Scenario)"
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.split("/")[-1] for m in models], rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(100, ax.get_ylim()[1] * 1.1))
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_urs_by_language(
    result: Dict,
    output_path: Path,
    scenario: Optional[str] = None,
):
    """
    Grouped bar chart: URS by individual language for each model.
    Args:
        scenario: If provided, use scenario-specific data. None = combined.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    models = list(result["by_model_language"].keys())
    
    x = np.arange(len(LANGUAGES))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        urs_values = []
        for lang in LANGUAGES:
            if scenario and "by_model_language_scenario" in result:
                # Scenario-specific
                if model in result["by_model_language_scenario"]:
                    lang_data = result["by_model_language_scenario"][model].get(lang, {})
                    urs_values.append(lang_data.get(scenario, {}).get("urs", 0))
                else:
                    urs_values.append(0)
            else:
                # Combined
                if lang in result["by_model_language"][model]:
                    urs_values.append(result["by_model_language"][model][lang]["urs"])
                else:
                    urs_values.append(0)
        
        color = MODEL_COLORS.get(model, f"C{i}")
        bars = ax.bar(
            x + i * width - (len(models) - 1) * width / 2,
            urs_values,
            width,
            label=model.split("/")[-1],
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
    
    # Add tier labels at bottom
    for i, lang in enumerate(LANGUAGES):
        tier = LANGUAGE_TIERS[lang]
        ax.annotate(
            tier,
            xy=(i, -5),
            ha="center",
            fontsize=8,
            color=TIER_COLORS[tier],
            fontweight="bold",
        )
    
    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Unsafe Rate (%)", fontsize=12)
    
    title = "Unsafe Rate by Language"
    if scenario:
        title += f" ({scenario.title()} Scenario)"
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels([LANGUAGE_NAMES[l] for l in LANGUAGES], rotation=45, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_ylim(-10, max(100, ax.get_ylim()[1] * 1.1))
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_urs_heatmap(
    result: Dict,
    output_path: Path,
    scenario: Optional[str] = None,
):
    """
    Heatmap: Model × Language URS matrix.
    Args:
        scenario: If provided, use scenario-specific data. None = combined.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    models = sorted(result["by_model_language"].keys())
    
    # Build matrix
    matrix = []
    for model in models:
        row = []
        for lang in LANGUAGES:
            if scenario and "by_model_language_scenario" in result:
                # Scenario-specific
                if model in result["by_model_language_scenario"]:
                    lang_data = result["by_model_language_scenario"][model].get(lang, {})
                    row.append(lang_data.get(scenario, {}).get("urs", 0))
                else:
                    row.append(0)
            else:
                # Combined
                if lang in result["by_model_language"][model]:
                    row.append(result["by_model_language"][model][lang]["urs"])
                else:
                    row.append(0)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Create heatmap
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Unsafe Rate (%)", rotation=-90, va="bottom")
    
    # Labels
    ax.set_xticks(np.arange(len(LANGUAGES)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([f"{LANGUAGE_NAMES[l]}\n({LANGUAGE_TIERS[l]})" for l in LANGUAGES])
    ax.set_yticklabels([m.split("/")[-1] for m in models])
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(LANGUAGES)):
            value = matrix[i, j]
            color = "white" if value > 50 else "black"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=color, fontsize=9)
    
    title = "Unsafe Rate Heatmap (Model × Language)"
    if scenario:
        title += f" - {scenario.title()}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_gap_comparison(
    result: Dict,
    output_path: Path,
    show_scenarios: bool = False,
):
    """
    Bar chart: Gap (LRL URS - HRL URS) by model.
    
    Shows which models have the largest/smallest multilingual safety gap.
    Args:
        show_scenarios: If True, show unintentional and intentional side by side.
    """
    gaps = result.get("gap_by_model", {})
    if not gaps:
        print("  No gap data available")
        return
    
    models = list(gaps.keys())
    
    if show_scenarios and "gap_by_model_scenario" in result:
        # Side-by-side comparison of scenarios with combined average
        fig, ax = plt.subplots(figsize=(12, 7))
        
        gap_scenario = result.get("gap_by_model_scenario", {})
        
        # Get short names
        short_names = [m.split("/")[-1] for m in models]
        
        y = np.arange(len(models))
        height = 0.25  # Smaller height for 3 bars
        
        # Unintentional gaps
        unint_gaps = [gap_scenario.get(m, {}).get("unintentional", 0) for m in models]
        # Intentional gaps  
        int_gaps = [gap_scenario.get(m, {}).get("intentional", 0) for m in models]
        # Combined/average gaps
        combined_gaps = [gaps[m] for m in models]
        
        # Sort by combined gap
        sorted_idx = sorted(range(len(models)), key=lambda i: combined_gaps[i], reverse=True)
        
        short_names = [short_names[i] for i in sorted_idx]
        unint_gaps = [unint_gaps[i] for i in sorted_idx]
        int_gaps = [int_gaps[i] for i in sorted_idx]
        combined_gaps = [combined_gaps[i] for i in sorted_idx]
        
        # Draw bars
        bars1 = ax.barh(y - height, unint_gaps, height, label="Unintentional", 
                        color="#3498db", edgecolor="black", linewidth=0.5, alpha=0.8)
        bars2 = ax.barh(y, int_gaps, height, label="Intentional (AIM)", 
                        color="#9b59b6", edgecolor="black", linewidth=0.5, alpha=0.8)
        bars3 = ax.barh(y + height, combined_gaps, height, label="Combined", 
                        color="#95a5a6", edgecolor="black", linewidth=0.5, alpha=0.9)
        
        # Add value labels
        for bars, gaps_list in [(bars1, unint_gaps), (bars2, int_gaps), (bars3, combined_gaps)]:
            for bar, val in zip(bars, gaps_list):
                x_pos = bar.get_width() + (0.3 if val >= 0 else -0.3)
                ax.text(
                    x_pos,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.1f}",
                    ha="left" if val >= 0 else "right",
                    va="center",
                    fontsize=8,
                )
        
        ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(short_names, fontsize=11)
        ax.set_xlabel("Gap: LRL URS - HRL URS (%)", fontsize=12)
        ax.set_title("Multilingual Safety Gap by Model and Scenario", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        
    else:
        # Original single-bar version
        fig, ax = plt.subplots(figsize=(10, 6))
        
        values = [gaps[m] for m in models]
        
        # Sort by gap value
        sorted_pairs = sorted(zip(models, values), key=lambda x: x[1], reverse=True)
        models, values = zip(*sorted_pairs)
        
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]
        
        bars = ax.barh(
            range(len(models)),
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        
        # Add value labels
        for bar, val in zip(bars, values):
            x_pos = bar.get_width() + (0.3 if val >= 0 else -0.3)
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%",
                ha="left" if val >= 0 else "right",
                va="center",
                fontsize=10,
            )
        
        ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m.split("/")[-1] for m in models], fontsize=11)
        ax.set_xlabel("Gap: LRL URS - HRL URS (%)", fontsize=12)
        ax.set_title("Multilingual Safety Gap by Model", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_scenario_comparison(
    result: Dict,
    output_path: Path,
):
    """
    Grouped bar chart: URS by scenario (unintentional vs intentional) for each model.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(result["by_model_scenario"].keys())
    scenarios = ["unintentional", "intentional"]
    
    x = np.arange(len(models))
    width = 0.35
    
    scenario_colors = {
        "unintentional": "#3498db",
        "intentional": "#9b59b6",
    }
    
    for i, scenario in enumerate(scenarios):
        urs_values = []
        for model in models:
            if scenario in result["by_model_scenario"][model]:
                urs_values.append(result["by_model_scenario"][model][scenario]["urs"])
            else:
                urs_values.append(0)
        
        bars = ax.bar(
            x + (i - 0.5) * width,
            urs_values,
            width,
            label=scenario.title(),
            color=scenario_colors[scenario],
            edgecolor="black",
            linewidth=0.5,
        )
        
        # Add value labels
        for bar, val in zip(bars, urs_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Unsafe Rate (%)", fontsize=12)
    ax.set_title("Unsafe Rate: Unintentional vs Intentional (AIM) Scenarios", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in models], rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(100, ax.get_ylim()[1] * 1.1))
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for MultiJail baseline experiment",
    )
    
    parser.add_argument(
        "--analysis-dir", type=str, default=None,
        help="Directory containing analysis result.json",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory for plots",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup directories
    if args.analysis_dir:
        analysis_dir = Path(args.analysis_dir)
    else:
        analysis_dir = RESULTS_DIR / "analysis"
    
    if args.outdir:
        output_dir = Path(args.outdir)
    else:
        output_dir = RESULTS_DIR / "plots"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print banner
    print("=" * 70)
    print("MULTIJAIL BASELINE - VISUALIZATION")
    print("=" * 70)
    print(f"  Analysis dir: {analysis_dir}")
    print(f"  Output dir: {output_dir}")
    print("=" * 70)
    
    # Load result
    print("\nLoading analysis result...")
    try:
        result = load_analysis_result(analysis_dir)
    except FileNotFoundError:
        print(f"\n❌ No result.json found in {analysis_dir}")
        print("  Run analyze.py first")
        return 1
    
    print("  Loaded successfully")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Combined (average of both scenarios)
    plot_urs_by_tier(result, output_dir / "urs_by_tier.png")
    plot_urs_by_language(result, output_dir / "urs_by_language.png")
    plot_urs_heatmap(result, output_dir / "urs_heatmap.png")
    plot_gap_comparison(result, output_dir / "gap_comparison.png")
    plot_gap_comparison(result, output_dir / "gap_comparison_by_scenario.png", show_scenarios=True)
    plot_scenario_comparison(result, output_dir / "scenario_comparison.png")
    
    # Scenario-specific plots
    print("\n  Generating scenario-specific plots...")
    for scenario in ["unintentional", "intentional"]:
        plot_urs_by_tier(result, output_dir / f"urs_by_tier_{scenario}.png", scenario=scenario)
        plot_urs_by_language(result, output_dir / f"urs_by_language_{scenario}.png", scenario=scenario)
        plot_urs_heatmap(result, output_dir / f"urs_heatmap_{scenario}.png", scenario=scenario)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"  Output directory: {output_dir}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
