#!/usr/bin/env python3
"""
Analyze judge panel Cohen's Kappa for rebuttal.

Computes per-judge and per-panel Kappa against human consensus,
broken down by variant. Generates markdown report + publication figure.

Output:
  runs/rebuttal-calibration-analysis/judge_panel_kappa_report.md
  analysis/experiments/figures/fig_cross_variant_panel_kappa.{pdf,png}
"""

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.metaeval.analyze_judge_calibration import (
    JudgeResult,
    align_labels,
    cohen_kappa,
    compute_consensus_labels,
    compute_metrics,
    load_judge_results,
    load_uuid_metadata,
)
from scripts.metaeval.dedup_models import deduplicate_judge_files, extract_judge_name

# --- Panel definitions (from calibration_report_by_variant.md) ---

VARIANTS = ["original_en", "translated_kr", "cultural_adapted_en", "transcreated_kr"]
VARIANT_DISPLAY = {
    "original_en": "Original English",
    "translated_kr": "Translated Korean",
    "cultural_adapted_en": "Cultural Adapted EN",
    "transcreated_kr": "Transcreated Korean",
}

# Best panels per variant (judge names as they appear after dedup: reasoning-preferred,
# filename stems with _ replaced by /)
BEST_PANELS_BY_VARIANT = {
    "original_en": {
        "label": "claude-opus-4.5 [r] + deepseek-v3p2 + o4-mini",
        "judges": [
            "anthropic/claude/opus/4/5/reasoning/medium",
            "fireworks/ai/deepseek/v3p2/reasoning/medium",
            "openai/o4/mini/reasoning/medium",
        ],
    },
    "translated_kr": {
        "label": "k-exaone + kanana2 + o4-mini",
        "judges": [
            "kexaone",
            "kanana2",
            "openai/o4/mini/reasoning/medium",
        ],
    },
    "cultural_adapted_en": {
        "label": "hyperclovax + kimi-k2-thinking + gemini-3-pro [r]",
        "judges": [
            "hyperclovax",
            "fireworks/ai/kimi/k2/thinking/reasoning/medium",
            "gemini/gemini/3/pro/preview/reasoning/medium",
        ],
    },
    "transcreated_kr": {
        "label": "gpt-5.2 + hyperclovax + solar-open",
        "judges": [
            "openai/gpt/5/2/reasoning/medium",
            "hyperclovax",
            "solar/open",
        ],
    },
}

CANDIDATE_UNIVERSAL_PANEL = {
    "label": "hyperclovax + o4-mini + kimi-k2-thinking [r]",
    "judges": [
        "hyperclovax",
        "openai/o4/mini/reasoning/medium",
        "fireworks/ai/kimi/k2/thinking/reasoning/medium",
    ],
}


def load_deduplicated_judge_results(
    results_dir: Path,
    dataset_path: Path,
) -> Dict[str, List[JudgeResult]]:
    """Load judge results, keeping only reasoning-preferred deduplicated models."""
    uuid_metadata = load_uuid_metadata(dataset_path)
    all_results = load_judge_results(results_dir, uuid_metadata=uuid_metadata)

    # Get deduplicated file list
    dedup_files = deduplicate_judge_files(results_dir)
    dedup_names = set()
    for f in dedup_files:
        name = f.stem.replace("results_", "").replace("_", "/")
        dedup_names.add(name)

    # Filter to only deduplicated judges
    filtered = {k: v for k, v in all_results.items() if k in dedup_names}
    return filtered


def compute_panel_kappa_by_variant(
    panel_judges: List[str],
    judge_results: Dict[str, List[JudgeResult]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute panel metrics (Kappa, F1, agreement) per variant via majority voting.

    Returns:
        Dict mapping variant_name -> {kappa, f1, agreement, n_compared}
    """
    # Collect results by (uuid, variant)
    uuid_variant_to_judge_results: Dict[Tuple[str, str], Dict[str, JudgeResult]] = defaultdict(dict)
    for judge_name in panel_judges:
        if judge_name not in judge_results:
            continue
        for result in judge_results[judge_name]:
            key = (result.uuid, result.variant_name)
            uuid_variant_to_judge_results[key][judge_name] = result

    # Compute majority vote per variant
    variant_human: Dict[str, List[str]] = defaultdict(list)
    variant_panel: Dict[str, List[str]] = defaultdict(list)

    for (uuid, variant), judge_map in uuid_variant_to_judge_results.items():
        if len(judge_map) < 2:
            continue  # Need at least 2 judges for meaningful panel

        # Get human consensus from first judge's result
        first_result = next(iter(judge_map.values()))
        human_consensus = compute_consensus_labels(first_result.human_annotations)

        # Majority vote per rubric item
        for i in range(len(human_consensus)):
            votes = []
            for judge_name, result in judge_map.items():
                if i < len(result.judge_labels):
                    votes.append(result.judge_labels[i])

            if len(votes) < 2:
                continue

            yes_count = votes.count("yes")
            panel_label = "yes" if yes_count > len(votes) / 2 else "no"

            variant_human[variant].append(human_consensus[i])
            variant_panel[variant].append(panel_label)

    # Compute metrics per variant
    results = {}
    for variant in VARIANTS:
        human = variant_human.get(variant, [])
        panel = variant_panel.get(variant, [])
        if human:
            metrics = compute_metrics(human, panel)
            results[variant] = metrics
        else:
            results[variant] = {"kappa": 0.0, "f1": 0.0, "agreement": 0.0, "n_compared": 0}

    return results


def compute_pairwise_judge_kappa(
    panel_judges: List[str],
    judge_results: Dict[str, List[JudgeResult]],
) -> Dict[Tuple[str, str], float]:
    """Compute pairwise Cohen's Kappa between judges in a panel."""
    pairwise = {}
    for j1, j2 in combinations(panel_judges, 2):
        if j1 not in judge_results or j2 not in judge_results:
            pairwise[(j1, j2)] = 0.0
            continue

        # Align by UUID
        j1_by_uuid = {r.uuid: r for r in judge_results[j1]}
        j2_by_uuid = {r.uuid: r for r in judge_results[j2]}
        common_uuids = set(j1_by_uuid.keys()) & set(j2_by_uuid.keys())

        all_j1_labels = []
        all_j2_labels = []
        for uuid in common_uuids:
            r1 = j1_by_uuid[uuid]
            r2 = j2_by_uuid[uuid]
            min_len = min(len(r1.judge_labels), len(r2.judge_labels))
            all_j1_labels.extend(r1.judge_labels[:min_len])
            all_j2_labels.extend(r2.judge_labels[:min_len])

        if all_j1_labels:
            pairwise[(j1, j2)] = cohen_kappa(all_j1_labels, all_j2_labels)
        else:
            pairwise[(j1, j2)] = 0.0

    return pairwise


def compute_individual_judge_kappa_by_variant(
    judge_results: Dict[str, List[JudgeResult]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute per-judge Kappa and F1 broken down by variant.

    Returns:
        Dict mapping judge_name -> variant_name -> {kappa, f1, agreement, n_compared}
    """
    results = {}
    for judge_name, judge_data in judge_results.items():
        # Group by variant
        by_variant: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"human": [], "judge": []})
        for result in judge_data:
            consensus = compute_consensus_labels(result.human_annotations)
            aligned_h, aligned_j = align_labels(consensus, result.judge_labels)
            if aligned_h:
                by_variant[result.variant_name]["human"].extend(aligned_h)
                by_variant[result.variant_name]["judge"].extend(aligned_j)

        variant_metrics = {}
        for variant in VARIANTS:
            data = by_variant.get(variant, {"human": [], "judge": []})
            if data["human"]:
                variant_metrics[variant] = compute_metrics(data["human"], data["judge"])
            else:
                variant_metrics[variant] = {"kappa": 0.0, "f1": 0.0, "agreement": 0.0, "n_compared": 0}

        results[judge_name] = variant_metrics

    return results


def generate_report(
    individual_metrics: Dict[str, Dict[str, Dict[str, float]]],
    panel_metrics: Dict[str, Dict[str, Dict[str, float]]],
    pairwise_kappas: Dict[str, Dict[Tuple[str, str], float]],
    panel_definitions: Dict[str, Dict[str, Any]],
    output_path: Path,
):
    """Generate the markdown report."""
    lines = []
    lines.append("# Judge Panel Kappa Analysis (Rebuttal)")
    lines.append("")
    lines.append("Cohen's Kappa values for individual judges and judge panels across all 4 variants,")
    lines.append("computed on 97 calibration tasks (266 records) against human consensus labels.")
    lines.append("")
    lines.append("**Model selection:** Reasoning variants preferred; non-reasoning used when no reasoning available.")
    lines.append("")

    # Section 1: Individual judge Kappa by variant
    lines.append("## Individual Judge Kappa by Variant")
    lines.append("")
    lines.append("| Judge | original_en |  | translated_kr |  | cultural_adapted_en |  | transcreated_kr |  |")
    lines.append("|-------|:-----------:|:-:|:-------------:|:-:|:-------------------:|:-:|:---------------:|:-:|")
    lines.append("|       | Kappa | F1 | Kappa | F1 | Kappa | F1 | Kappa | F1 |")

    # Sort judges by average kappa across variants
    judge_avg_kappa = {}
    for judge_name, variant_data in individual_metrics.items():
        kappas = [variant_data[v]["kappa"] for v in VARIANTS if variant_data[v]["n_compared"] > 0]
        judge_avg_kappa[judge_name] = np.mean(kappas) if kappas else 0.0

    sorted_judges = sorted(judge_avg_kappa.keys(), key=lambda j: judge_avg_kappa[j], reverse=True)

    for judge_name in sorted_judges:
        variant_data = individual_metrics[judge_name]
        cells = [f"`{judge_name}`"]
        for v in VARIANTS:
            m = variant_data[v]
            if m["n_compared"] > 0:
                cells.append(f"{m['kappa']:.3f}")
                cells.append(f"{m['f1']:.3f}")
            else:
                cells.append("-")
                cells.append("-")
        lines.append(f"| {' | '.join(cells)} |")

    lines.append("")

    # Section 2: Panel Kappa by variant
    lines.append("## Panel Kappa by Variant")
    lines.append("")
    lines.append("Majority voting across 3 judges per rubric item.")
    lines.append("")

    panel_names = list(VARIANTS) + ["universal"]
    lines.append("| Panel | Source | original_en |  | translated_kr |  | cultural_adapted_en |  | transcreated_kr |  |")
    lines.append("|-------|--------|:-----------:|:-:|:-------------:|:-:|:-------------------:|:-:|:---------------:|:-:|")
    lines.append("|       |        | Kappa | F1 | Kappa | F1 | Kappa | F1 | Kappa | F1 |")

    for panel_key in panel_names:
        if panel_key == "universal":
            label = CANDIDATE_UNIVERSAL_PANEL["label"]
            source = "Universal"
        else:
            label = panel_definitions[panel_key]["label"]
            source = VARIANT_DISPLAY[panel_key]

        p_metrics = panel_metrics[panel_key]
        cells = [f"**{label}**", source]
        for v in VARIANTS:
            m = p_metrics[v]
            if m["n_compared"] > 0:
                cells.append(f"{m['kappa']:.3f}")
                cells.append(f"{m['f1']:.3f}")
            else:
                cells.append("-")
                cells.append("-")
        lines.append(f"| {' | '.join(cells)} |")

    lines.append("")

    # Section 3: Pairwise inter-judge Kappa within panels
    lines.append("## Pairwise Inter-Judge Kappa Within Panels")
    lines.append("")

    for panel_key in panel_names:
        if panel_key == "universal":
            label = "Candidate Universal"
            judges = CANDIDATE_UNIVERSAL_PANEL["judges"]
        else:
            label = VARIANT_DISPLAY[panel_key]
            judges = panel_definitions[panel_key]["judges"]

        lines.append(f"### {label}")
        lines.append("")

        pw = pairwise_kappas.get(panel_key, {})
        if pw:
            lines.append("| Judge 1 | Judge 2 | Kappa |")
            lines.append("|---------|---------|-------|")
            for (j1, j2), kappa_val in pw.items():
                lines.append(f"| `{j1}` | `{j2}` | {kappa_val:.3f} |")
            lines.append("")
        else:
            lines.append("No pairwise data available.")
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to {output_path}")


def generate_kappa_figure(
    panel_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_name: str = "fig_cross_variant_panel_kappa",
):
    """
    Generate a 5x4 heatmap (same layout as fig_cross_variant_panel_analysis.pdf)
    but showing Kappa values.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap

    from analysis.experiments.plotting import setup_paper_style, save_figure, FIGURE_DIR

    setup_paper_style()

    n_rows = 5  # 4 variant panels + 1 universal
    n_cols = 4  # 4 variants (no delta column for kappa)

    matrix = np.zeros((n_rows, n_cols))
    native_cells = []

    panel_keys = list(VARIANTS) + ["universal"]
    row_labels = []

    for i, panel_key in enumerate(panel_keys):
        if panel_key == "universal":
            row_labels.append("Candidate Universal")
        else:
            row_labels.append(VARIANT_DISPLAY[panel_key])

        for j, variant in enumerate(VARIANTS):
            kappa = panel_metrics[panel_key][variant]["kappa"]
            matrix[i, j] = kappa

            if panel_key == variant:
                native_cells.append((i, j))

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.grid(False)

    # Colormap: red -> yellow -> green (same as F1 figure)
    cmap = LinearSegmentedColormap.from_list(
        "rg_cmap",
        [(0.85, 0.25, 0.25), (0.98, 0.92, 0.55), (0.25, 0.72, 0.35)],
        N=256,
    )

    # Plot heatmap
    x = np.arange(n_cols + 1) - 0.5
    y = np.arange(n_rows + 1) - 0.5
    X, Y = np.meshgrid(x, y)

    im = ax.pcolormesh(X, Y, matrix, cmap=cmap, vmin=0.55, vmax=0.85,
                       edgecolors="white", linewidth=1)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    # Add cell values
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            mark = "*" if (i, j) in native_cells else ""
            color = "white" if val > 0.72 else "black"
            ax.text(j, i, f"{val:.3f}{mark}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    # Separator for universal row
    line_style = dict(color="#1976D2", linestyle="--", linewidth=2.5, alpha=0.9)
    ax.axhline(y=3.5, **line_style)
    ax.axhline(y=4.5, **line_style)
    ax.plot([-0.5, -0.5], [3.5, 4.5], **line_style)
    ax.plot([n_cols - 0.5, n_cols - 0.5], [3.5, 4.5], **line_style)

    # Labels
    variant_display_short = ["Original\nEnglish", "Translated\nKorean",
                             "Cultural\nAdapted EN", "Transcreated\nKorean"]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(variant_display_short, fontsize=8, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", pad=8)

    ax.set_xlabel("Target Variant", fontsize=9, fontweight="bold", labelpad=8)
    ax.set_ylabel("Panel Source", fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Cohen's Kappa", fontsize=8, fontweight="medium")
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Cross-Variant Panel Analysis: Cohen's Kappa",
                 fontsize=10, fontweight="bold", pad=12)

    legend_text = "* = native variant | Kappa: 0.61-0.80 = substantial, 0.81+ = near-perfect"
    props = dict(boxstyle="round,pad=0.3", facecolor="#FFFDE7", edgecolor="#FBC02D", alpha=0.95)
    ax.text(0.5, -0.08, legend_text, transform=ax.transAxes, fontsize=7,
            ha="center", va="top", style="italic", bbox=props)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, left=0.22)

    paths = save_figure(fig, output_name, formats=("pdf", "png"))
    for p in paths:
        print(f"  Saved: {p}")


def main():
    base_dir = PROJECT_ROOT
    results_dir = base_dir / "runs" / "metaeval-judge-calibration"
    dataset_path = results_dir / "calibration_dataset.jsonl"
    output_dir = base_dir / "runs" / "rebuttal-calibration-analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("JUDGE PANEL KAPPA ANALYSIS (REBUTTAL)")
    print("=" * 70)
    print()

    # Load deduplicated judge results
    print("Loading deduplicated judge results...")
    judge_results = load_deduplicated_judge_results(results_dir, dataset_path)
    print(f"Loaded {len(judge_results)} judges (reasoning-preferred)")
    for name in sorted(judge_results.keys()):
        print(f"  {name} ({len(judge_results[name])} results)")
    print()

    # Validate panel judge names exist in loaded results
    all_panel_judges = set()
    for panel_def in BEST_PANELS_BY_VARIANT.values():
        all_panel_judges.update(panel_def["judges"])
    all_panel_judges.update(CANDIDATE_UNIVERSAL_PANEL["judges"])

    missing = all_panel_judges - set(judge_results.keys())
    if missing:
        print("WARNING: The following panel judges are not in the loaded results:")
        for m in sorted(missing):
            print(f"  MISSING: '{m}'")
        print()
        print("Available judge names:")
        for name in sorted(judge_results.keys()):
            print(f"  '{name}'")
        print()

    # 1. Individual judge Kappa by variant
    print("Computing individual judge Kappa by variant...")
    individual_metrics = compute_individual_judge_kappa_by_variant(judge_results)
    print()

    # 2. Panel Kappa by variant
    print("Computing panel Kappa by variant...")
    panel_metrics = {}
    for variant_key, panel_def in BEST_PANELS_BY_VARIANT.items():
        metrics = compute_panel_kappa_by_variant(panel_def["judges"], judge_results)
        panel_metrics[variant_key] = metrics
        native_kappa = metrics[variant_key]["kappa"]
        print(f"  {variant_key} panel: native Kappa={native_kappa:.3f}")

    # Universal panel
    universal_metrics = compute_panel_kappa_by_variant(
        CANDIDATE_UNIVERSAL_PANEL["judges"], judge_results
    )
    panel_metrics["universal"] = universal_metrics
    avg_kappa = np.mean([universal_metrics[v]["kappa"] for v in VARIANTS])
    print(f"  Universal panel: avg Kappa={avg_kappa:.3f}")
    print()

    # 3. Pairwise inter-judge Kappa within panels
    print("Computing pairwise inter-judge Kappa...")
    pairwise_kappas = {}
    for variant_key, panel_def in BEST_PANELS_BY_VARIANT.items():
        pairwise_kappas[variant_key] = compute_pairwise_judge_kappa(
            panel_def["judges"], judge_results
        )
    pairwise_kappas["universal"] = compute_pairwise_judge_kappa(
        CANDIDATE_UNIVERSAL_PANEL["judges"], judge_results
    )
    print()

    # Generate report
    print("Generating report...")
    report_path = output_dir / "judge_panel_kappa_report.md"
    generate_report(
        individual_metrics,
        panel_metrics,
        pairwise_kappas,
        BEST_PANELS_BY_VARIANT,
        report_path,
    )

    # Generate figure
    print("Generating figure...")
    generate_kappa_figure(panel_metrics, output_name="fig_cross_variant_panel_kappa")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
