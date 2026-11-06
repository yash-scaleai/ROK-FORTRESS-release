#!/usr/bin/env python3
"""
Generate Cross-Variant Panel Analysis: F1 Score Matrix.

Creates a publication-quality heatmap showing:
- Best judge panel per variant (4 rows)
- Candidate universal panel (1 row, below dashed line)
- F1 scores across all variants + Cumulative Delta

Data source: runs/metaeval-judge-calibration/calibration_report_by_variant.md

Output: figures/fig_cross_variant_panel_analysis.pdf
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from analysis.experiments.plotting import setup_paper_style, save_figure, FIGURE_DIR

# =============================================================================
# Data from calibration_report_by_variant.md
# =============================================================================

VARIANTS = ["original_en", "translated_kr", "cultural_adapted_en", "transcreated_kr"]
VARIANT_DISPLAY = ["Original\nEnglish", "Translated\nKorean", "Cultural\nAdapted EN", "Transcreated\nKorean"]

# Best panels per variant (from report)
BEST_PANELS = {
    "original_en": {
        "panel": "claude-opus-4.5 [r] + deepseek-v3p2 + o4-mini",
        "f1_scores": {"original_en": 0.882, "translated_kr": 0.780, "cultural_adapted_en": 0.800, "transcreated_kr": 0.754},
    },
    "translated_kr": {
        "panel": "k-exaone-236b + kanana2-30b + o4-mini",
        "f1_scores": {"original_en": 0.853, "translated_kr": 0.810, "cultural_adapted_en": 0.806, "transcreated_kr": 0.806},
    },
    "cultural_adapted_en": {
        "panel": "hyperclovax-seed + kimi-k2-thinking + gemini-3-pro [r]",
        "f1_scores": {"original_en": 0.865, "translated_kr": 0.791, "cultural_adapted_en": 0.852, "transcreated_kr": 0.780},
    },
    "transcreated_kr": {
        "panel": "gpt-5.2 + hyperclovax-seed + solar-open-100b",
        "f1_scores": {"original_en": 0.841, "translated_kr": 0.764, "cultural_adapted_en": 0.806, "transcreated_kr": 0.855},
    },
}

# Candidate universal panel
CANDIDATE_UNIVERSAL = {
    "name": "Candidate Universal",
    "panel": "hyperclovax + o4-mini + kimi-k2-thinking [r]",
    "f1_scores": {"original_en": 0.864, "translated_kr": 0.783, "cultural_adapted_en": 0.823, "transcreated_kr": 0.844},
}

# Best F1 per variant (diagonal values)
BEST_F1_PER_VARIANT = {
    "original_en": 0.882,
    "translated_kr": 0.810,
    "cultural_adapted_en": 0.852,
    "transcreated_kr": 0.855,
}


def plot_cross_variant_panel_analysis(
    output_name: str = "fig_cross_variant_panel_analysis",
) -> list[Path]:
    """Create the cross-variant panel analysis heatmap."""
    setup_paper_style()

    # Build data
    n_rows = 5  # 4 variant panels + 1 candidate
    n_cols = 5  # 4 variants + cumulative delta

    row_labels_variant = []  # Bold variant names
    row_labels_judges = []   # Regular judge names
    matrix = np.zeros((n_rows, n_cols))
    native_cells = []  # (row, col) for cells to mark with *

    # Add best panel per variant (rows 0-3)
    for i, variant in enumerate(VARIANTS):
        panel_data = BEST_PANELS[variant]
        panel_str = panel_data["panel"]
        # Split judges onto separate lines
        judges = panel_str.split(" + ")
        judges_multiline = "\n".join(judges)
        row_labels_variant.append(variant)
        row_labels_judges.append(f"({judges_multiline})")

        cumulative_delta = 0
        for j, v in enumerate(VARIANTS):
            f1 = panel_data["f1_scores"][v]
            matrix[i, j] = f1
            cumulative_delta += f1 - BEST_F1_PER_VARIANT[v]

            # Mark native variant
            if v == variant:
                native_cells.append((i, j))

        matrix[i, 4] = cumulative_delta

    # Add candidate universal panel (row 4)
    universal_judges = CANDIDATE_UNIVERSAL['panel'].split(" + ")
    universal_judges_multiline = "\n".join(universal_judges)
    row_labels_variant.append(CANDIDATE_UNIVERSAL['name'])
    row_labels_judges.append(f"({universal_judges_multiline})")
    cumulative_delta = 0
    for j, v in enumerate(VARIANTS):
        f1 = CANDIDATE_UNIVERSAL["f1_scores"][v]
        matrix[4, j] = f1
        cumulative_delta += f1 - BEST_F1_PER_VARIANT[v]
    matrix[4, 4] = cumulative_delta

    # Create figure (single column width ~3.5in, but taller for readability)
    fig, ax = plt.subplots(figsize=(7, 6))

    # Disable grid
    ax.grid(False)

    # Custom colormap: red -> yellow -> green
    cmap = LinearSegmentedColormap.from_list(
        "rg_cmap",
        [(0.85, 0.25, 0.25), (0.98, 0.92, 0.55), (0.25, 0.72, 0.35)],
        N=256
    )

    # Plot heatmap using pcolormesh for cleaner rendering
    f1_matrix = matrix[:, :4]

    # Create meshgrid for pcolormesh
    x = np.arange(5) - 0.5  # 5 edges for 4 columns
    y = np.arange(n_rows + 1) - 0.5  # n_rows+1 edges for n_rows rows
    X, Y = np.meshgrid(x, y)

    im = ax.pcolormesh(X, Y, f1_matrix, cmap=cmap, vmin=0.70, vmax=0.90,
                       edgecolors='none', linewidth=0)

    # Extend plot area for delta column
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(n_rows - 0.5, -0.5)  # Flip y-axis

    # Add delta column with blue gradient (distinct from F1 heatmap)
    # Darker blue = worse (more negative), lighter = better (closer to 0)
    delta_values = matrix[:, 4]
    delta_min = min(delta_values)  # Most negative
    delta_max = max(delta_values)  # Closest to 0

    for i in range(n_rows):
        # Normalize delta to 0-1 range (0 = most negative, 1 = closest to 0)
        if delta_max != delta_min:
            norm_val = (delta_values[i] - delta_min) / (delta_max - delta_min)
        else:
            norm_val = 0.5

        # Blue gradient: dark blue (worst) -> light blue/white (best)
        # RGB interpolation
        dark_blue = (0.15, 0.25, 0.55)  # Dark navy
        light_blue = (0.85, 0.92, 0.98)  # Very light blue
        color = tuple(dark_blue[c] + norm_val * (light_blue[c] - dark_blue[c]) for c in range(3))

        rect = mpatches.Rectangle(
            (3.5, i - 0.5), 1, 1,
            facecolor=color, edgecolor="white", linewidth=1
        )
        ax.add_patch(rect)

    # Add cell values with deltas
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]

            if j == 4:  # Delta column
                text = f"{val:.3f}"
                # Use white text on darker blue backgrounds
                if delta_max != delta_min:
                    norm_val = (val - delta_min) / (delta_max - delta_min)
                else:
                    norm_val = 0.5
                color = "white" if norm_val < 0.6 else "#1a1a4e"
                fontsize = 8
            else:
                # Compute delta from best for this variant
                delta = val - BEST_F1_PER_VARIANT[VARIANTS[j]]

                # Mark native cells with *
                if (i, j) in native_cells:
                    text = f"{val:.3f}*"
                    delta_text = ""
                else:
                    text = f"{val:.3f}"
                    delta_text = f"\n({delta:+.3f})"

                # Text color based on value
                color = "white" if val > 0.83 else "black"
                fontsize = 8

                ax.text(j, i, text, ha="center", va="center",
                       fontsize=fontsize, color=color, fontweight="bold")

                # Add delta in smaller font below
                if delta_text and (i, j) not in native_cells:
                    delta_color = "#8B0000" if delta < -0.05 else "#666666"
                    ax.text(j, i + 0.25, delta_text.strip(), ha="center", va="top",
                           fontsize=7, color=delta_color, style="italic")
                continue

            ax.text(j, i, text, ha="center", va="center",
                   fontsize=fontsize, color=color, fontweight="medium")

    # Draw separator box around candidate universal row
    line_style = dict(color="#1976D2", linestyle="--", linewidth=2.5, alpha=0.9)
    ax.axhline(y=3.5, **line_style)  # Top
    ax.axhline(y=4.5, **line_style)  # Bottom
    ax.plot([-0.5, -0.5], [3.5, 4.5], **line_style)  # Left
    ax.plot([4.5, 4.5], [3.5, 4.5], **line_style)  # Right

    # Axis labels
    col_labels = VARIANT_DISPLAY + ["Cumulative\nDelta (Σ)"]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([""] * n_rows)  # Hide default labels

    # Manually add y-axis labels with different font weights
    for i in range(n_rows):
        # Variant name (bold)
        ax.text(-0.07, i - 0.15, row_labels_variant[i], ha="right", va="bottom",
                fontsize=8, fontweight="bold", transform=ax.get_yaxis_transform())
        # Judge names (regular) - immediately below variant name
        ax.text(-0.07, i - 0.12, row_labels_judges[i], ha="right", va="top",
                fontsize=7, fontweight="normal", transform=ax.get_yaxis_transform())

    # Move x-axis to top
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis='x', pad=8)

    # Labels
    ax.set_xlabel("Target Variant", fontsize=9, fontweight="bold", labelpad=8)
    ax.set_ylabel("Panel Source", fontsize=9, fontweight="bold")

    # Colorbar - match height of heatmap
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("F1 Score", fontsize=8, fontweight="medium")
    cbar.ax.tick_params(labelsize=7)

    # Title
    ax.set_title(
        "Cross-Variant Panel Analysis: F1 Score Matrix",
        fontsize=10, fontweight="bold", pad=12
    )

    # Legend box
    legend_text = (
        "* = native variant | Cumulative Δ = sum of F1 diff from best\n"
        "Lower Δ (closer to 0) = better universal panel"
    )
    props = dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7', edgecolor='#FBC02D', alpha=0.95)
    ax.text(0.5, -0.12, legend_text, transform=ax.transAxes, fontsize=7,
            ha='center', va='top', style='italic', bbox=props)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.28)

    return save_figure(fig, output_name, formats=("pdf", "png"))


def main():
    """Generate the cross-variant panel analysis figure."""
    print("Generating Cross-Variant Panel Analysis figure...")
    print(f"\nBest panels per variant:")
    for v in VARIANTS:
        print(f"  {v}: F1={BEST_PANELS[v]['f1_scores'][v]:.3f}")
        print(f"    {BEST_PANELS[v]['panel']}")

    print(f"\nCandidate Universal Panel:")
    print(f"  {CANDIDATE_UNIVERSAL['panel']}")
    avg_f1 = np.mean(list(CANDIDATE_UNIVERSAL["f1_scores"].values()))
    print(f"  Avg F1: {avg_f1:.3f}")

    paths = plot_cross_variant_panel_analysis()

    print(f"\nSaved figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
