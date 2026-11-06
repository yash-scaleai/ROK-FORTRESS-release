#!/usr/bin/env python3
"""
Generate Model vs Tactic Heatmap.

Creates publication-quality heatmaps showing TRS (Tiered Risk Score) for each model
across different attack tactics.

Generates:
- Overall heatmap (all variants)
- Per-variant heatmaps

Output: figures/fig_tactic_heatmap_*.pdf
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from analysis.experiments.plotting import (
    setup_paper_style,
    save_figure,
    FIGURE_DIR,
    get_model_family,
)
from analysis.experiments.figures import _canonical
from analysis.experiments.metrics.harm_scores import (
    DOMAIN_WEIGHTS,
    DOMAIN_NAME_MAP,
    get_dimension_key,
)

# =============================================================================
# Configuration
# =============================================================================

JUDGEMENTS_DIR = Path("runs/final-rok-fortress-judgements")
MAIN_DATASET = Path("data/Copy of fortress private classification - 2026-01-14-1126 - vFinal MAIN PIPELINE MASTER (DO NOT EDIT -with harm dimensions.tsv")
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

# High-level tactic categories
TACTIC_CATEGORIES = {
    "Roleplay / Persona": ["Roleplay", "Persona", "Fictionalization", "role play", "Mirror World"],
    "Direct Request": ["Direct Request", "Explicit Ask", "Command"],
    "Output Manipulation": ["Output Formatting", "Output Dictation", "Requesting Specific Output", "Stylized"],
    "Obfuscation": ["Obfuscation", "Encoding", "Encoded", "Synonyms", "Foreign Language"],
    "Prompt Injection": ["Prompt Injection", "Instruction Injection", "Dialog Injection", "Indirect Prompt"],
    "Emotional/Social": ["Emotional Appeal", "Appeal to Authority", "Urgency", "Acting Innocent", "Foot-in-the-door"],
    "Hypothetical/False": ["Hypothetical", "False Premise", "Misdirection"],
    "Gamification": ["Gamification", "Trick", "Game"],
    "Context Manipulation": ["Crowding", "Context Overload", "Priming", "Bootstrapping", "Context Drift", "Splitting"],
}

JUDGE_NAMES = ["kimi_k2", "hyperclovax", "o4_mini"]


def categorize_tactic(tactic: str) -> str:
    """Map a tactic string to a high-level category."""
    if pd.isna(tactic) or not tactic:
        return "Other"

    tactic_lower = tactic.lower()

    for category, keywords in TACTIC_CATEGORIES.items():
        for keyword in keywords:
            if keyword.lower() in tactic_lower:
                return category

    return "Other"


def load_task_tactics() -> Dict[str, str]:
    """Load task_id -> tactic mapping from main dataset."""
    df = pd.read_csv(MAIN_DATASET, sep='\t')

    task_tactics = {}
    for _, row in df.iterrows():
        # Use sheetRepid (mapping) which matches judgement task_ids
        task_id = row.get('sheetRepid (mapping)') or row.get('sheetRepid (mapping).1')
        tactic = row.get('Tactic', '')
        if task_id and pd.notna(task_id):
            task_tactics[str(task_id)] = categorize_tactic(tactic)

    return task_tactics


def load_judgements():
    """{model_token: {variant: [ConsolidatedResult]}} via the canonical 3-judge loader."""
    return _canonical.load_results_by_model()


def load_harm_classifications():
    """Deprecated: harm dimensions merged inside the canonical loader. Kept for compat."""
    return {}


def compute_trs_by_tactic(results, task_tactics, harm_classifications=None):
    """{tactic_category: mean TRS %} over the given consolidated results (canonical scoring)."""
    by_tactic = defaultdict(list)
    for r in results:
        if not r.is_valid or r.is_excluded:
            continue
        tactic = task_tactics.get(r.task_id, "Other")
        by_tactic[tactic].append(_canonical.response_trs(r))
    return {t: sum(s) / len(s) * 100 for t, s in by_tactic.items() if s}


def plot_tactic_heatmap(
    data: Dict[str, Dict[str, float]],
    title: str,
    output_name: str,
) -> List[Path]:
    """Create a heatmap of models vs tactics."""
    setup_paper_style()

    # Get all models and tactics
    models = sorted(data.keys(), key=lambda m: MODEL_DISPLAY.get(m, m))

    # Get all tactics that appear
    all_tactics = set()
    for model_data in data.values():
        all_tactics.update(model_data.keys())

    # Order tactics by category order, then "Other" last
    tactic_order = list(TACTIC_CATEGORIES.keys()) + ["Other"]
    tactics = [t for t in tactic_order if t in all_tactics]

    if not models or not tactics:
        print(f"No data for {title}")
        return []

    # Build matrix
    matrix = np.zeros((len(models), len(tactics)))
    for i, model in enumerate(models):
        for j, tactic in enumerate(tactics):
            matrix[i, j] = data.get(model, {}).get(tactic, np.nan)

    # Create figure (larger to accommodate bigger fonts)
    fig_width = max(16, len(tactics) * 1.4 + 6)
    fig_height = max(11, len(models) * 0.75 + 4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Disable grid
    ax.grid(False)

    # Custom colormap: green (safe) -> yellow -> red (harmful)
    cmap = LinearSegmentedColormap.from_list(
        "safety_cmap",
        [(0.2, 0.7, 0.3), (0.98, 0.92, 0.5), (0.85, 0.25, 0.25)],
        N=256
    )

    # Plot heatmap using pcolormesh for cleaner rendering
    x = np.arange(len(tactics) + 1) - 0.5
    y = np.arange(len(models) + 1) - 0.5
    im = ax.pcolormesh(x, y, matrix, cmap=cmap, vmin=0, vmax=80,
                       edgecolors='white', linewidth=0.5)
    ax.set_xlim(-0.5, len(tactics) - 0.5)
    ax.set_ylim(len(models) - 0.5, -0.5)

    # Add cell values (with 1 decimal place)
    for i in range(len(models)):
        for j in range(len(tactics)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                       fontsize=14, color=color, fontweight="bold")

    # Labels
    model_labels = [MODEL_DISPLAY.get(m, m) for m in models]
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_labels, fontsize=14, fontweight="medium")

    ax.set_xticks(range(len(tactics)))
    ax.set_xticklabels(tactics, fontsize=14, fontweight="medium", rotation=45, ha="right")

    ax.set_xlabel("Attack Tactic Category", fontsize=16, fontweight="bold")
    ax.set_ylabel("Model", fontsize=16, fontweight="bold")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=15)

    # Colorbar - match height of heatmap
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("TRS (%)", fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)

    # Add note
    ax.text(0.01, -0.15, "Lower (green) = safer, Higher (red) = more harmful",
           transform=ax.transAxes, fontsize=12, style="italic", color="#666666")

    plt.tight_layout()

    return save_figure(fig, output_name, formats=("pdf",))


def main():
    """Generate all tactic heatmaps."""
    print("Loading task tactics from main dataset...")
    task_tactics = load_task_tactics()
    print(f"  Loaded tactics for {len(task_tactics)} tasks")

    # Count tactics
    from collections import Counter
    tactic_counts = Counter(task_tactics.values())
    print("\nTactic category distribution:")
    for t, c in tactic_counts.most_common():
        print(f"  {t}: {c}")

    print("\nLoading harm classifications...")
    harm_classifications = load_harm_classifications()
    print(f"  Found {len(harm_classifications)} task classifications")

    print("\nLoading judgements...")
    judgements = load_judgements()
    print(f"  Found {len(judgements)} models")

    # Compute overall (all variants)
    print("\nComputing overall TRS by tactic (weighted)...")
    overall_data = {}
    for model, variants in judgements.items():
        all_judgements = []
        for v_judgements in variants.values():
            all_judgements.extend(v_judgements)
        overall_data[model] = compute_trs_by_tactic(all_judgements, task_tactics, harm_classifications)

    print("\nGenerating overall heatmap...")
    plot_tactic_heatmap(
        overall_data,
        "Model Vulnerability by Attack Tactic (Overall)",
        "fig_tactic_heatmap_overall"
    )

    # Per variant
    for variant in VARIANTS:
        print(f"\nGenerating heatmap for {variant}...")
        variant_data = {}
        for model, model_variants in judgements.items():
            if variant in model_variants:
                variant_data[model] = compute_trs_by_tactic(model_variants[variant], task_tactics, harm_classifications)

        if variant_data:
            plot_tactic_heatmap(
                variant_data,
                f"Model Vulnerability by Attack Tactic: {VARIANT_DISPLAY[variant]}",
                f"fig_tactic_heatmap_{variant}"
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
