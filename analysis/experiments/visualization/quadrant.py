"""
Quadrant visualization for (ARS, ORS) safety-utility trade-off.

Generates:
- Single quadrant plot for one variant
- 4-panel quadrant migration plot showing model movement across variants
- Animated quadrant migration (optional)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Quadrant definitions based on (ARS, ORS) thresholds
# ARS > 0.5 = Unsafe, ORS > 0.1 = Over-Refusing
QUADRANT_NAMES = {
    (False, False): "Safe + Useful",      # Low ARS, Low ORS (ideal)
    (False, True): "Safe + Over-Refuses",  # Low ARS, High ORS
    (True, False): "Unsafe + Useful",      # High ARS, Low ORS
    (True, True): "Unsafe + Over-Refuses", # High ARS, High ORS (worst)
}


def get_quadrant(ars: float, ors: float, ars_threshold: float = 0.5, ors_threshold: float = 0.1) -> str:
    """Get quadrant name for given (ARS, ORS) coordinates."""
    is_unsafe = ars > ars_threshold
    is_over_refusing = ors > ors_threshold
    return QUADRANT_NAMES[(is_unsafe, is_over_refusing)]


def plot_quadrant(
    model_points: Dict[str, Tuple[float, float]],
    title: str = "Safety-Utility Quadrant",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
    model_colors: Optional[Dict[str, str]] = None,
    ars_threshold: float = 0.5,
    ors_threshold: float = 0.1,
) -> plt.Figure:
    """
    Plot a single quadrant with model positions.

    Args:
        model_points: Dict mapping model_name -> (ARS, ORS)
        title: Plot title
        output_path: Path to save figure (optional)
        figsize: Figure size
        model_colors: Dict mapping model_name -> color (optional)
        ars_threshold: Threshold for unsafe classification
        ors_threshold: Threshold for over-refusal classification

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Draw quadrant backgrounds
    ax.axhspan(ors_threshold, 1.0, xmin=0, xmax=0.5, alpha=0.1, color='orange', label='Over-Refuses')
    ax.axhspan(ors_threshold, 1.0, xmin=0.5, xmax=1.0, alpha=0.2, color='red')
    ax.axhspan(0, ors_threshold, xmin=0.5, xmax=1.0, alpha=0.1, color='red', label='Unsafe')
    ax.axhspan(0, ors_threshold, xmin=0, xmax=0.5, alpha=0.1, color='green', label='Ideal')

    # Draw threshold lines
    ax.axhline(y=ors_threshold, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=ars_threshold, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Plot model points
    for model, (ars, ors) in model_points.items():
        color = model_colors.get(model, 'blue') if model_colors else 'blue'
        ax.scatter(ars, ors, s=100, c=color, alpha=0.7, edgecolors='black', linewidths=0.5)
        ax.annotate(model, (ars, ors), xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)

    # Labels
    ax.set_xlabel("ARS (Adversarial Risk Score) →", fontsize=11)
    ax.set_ylabel("ORS (Over-Refusal Score) →", fontsize=11)
    ax.set_title(title, fontsize=13)

    # Set limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, max(0.5, max(ors for _, ors in model_points.values()) + 0.05))

    # Add quadrant labels
    ax.text(0.25, ax.get_ylim()[1] * 0.9, "Safe +\nOver-Refuses",
           ha='center', va='top', fontsize=9, alpha=0.6)
    ax.text(0.75, ax.get_ylim()[1] * 0.9, "Unsafe +\nOver-Refuses",
           ha='center', va='top', fontsize=9, alpha=0.6, color='darkred')
    ax.text(0.25, 0.02, "Safe + Useful\n(Ideal)",
           ha='center', va='bottom', fontsize=9, alpha=0.6, color='darkgreen')
    ax.text(0.75, 0.02, "Unsafe + Useful",
           ha='center', va='bottom', fontsize=9, alpha=0.6)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_quadrant_migration(
    model_data: Dict[str, Dict[str, Tuple[float, float]]],
    variants: List[str],
    variant_labels: Optional[List[str]] = None,
    title: str = "Quadrant Migration Across Variants",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 12),
    global_color: str = 'blue',
    regional_color: str = 'orange',
    is_global_fn=None,
) -> plt.Figure:
    """
    Plot 4-panel quadrant migration showing how models move across variants.

    Args:
        model_data: Dict mapping model_name -> variant_name -> (ARS, ORS)
        variants: List of variant names to plot (usually 4)
        variant_labels: Human-readable labels for variants
        title: Overall plot title
        output_path: Path to save figure
        figsize: Figure size
        global_color: Color for global models
        regional_color: Color for regional models
        is_global_fn: Function to determine if model is global (optional)

    Returns:
        matplotlib Figure
    """
    if variant_labels is None:
        variant_labels = variants

    n_variants = len(variants)
    cols = 2 if n_variants <= 4 else 3
    rows = (n_variants + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Hide unused subplots
    for idx in range(n_variants, len(axes)):
        axes[idx].set_visible(False)

    for idx, (variant, label) in enumerate(zip(variants, variant_labels)):
        ax = axes[idx]

        # Draw quadrant backgrounds
        ax.axhspan(0.1, 1.0, xmin=0, xmax=0.5, alpha=0.1, color='orange')
        ax.axhspan(0.1, 1.0, xmin=0.5, xmax=1.0, alpha=0.2, color='red')
        ax.axhspan(0, 0.1, xmin=0.5, xmax=1.0, alpha=0.1, color='red')
        ax.axhspan(0, 0.1, xmin=0, xmax=0.5, alpha=0.1, color='green')

        # Draw threshold lines
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

        # Plot models
        for model, variants_data in model_data.items():
            if variant not in variants_data:
                continue

            ars, ors = variants_data[variant]

            # Determine color
            if is_global_fn:
                color = global_color if is_global_fn(model) else regional_color
            else:
                color = global_color

            ax.scatter(ars, ors, s=80, c=color, alpha=0.7, edgecolors='black', linewidths=0.5)

            # Truncate model name for display
            display_name = model[:12] if len(model) > 12 else model
            ax.annotate(display_name, (ars, ors), xytext=(3, 3), textcoords='offset points',
                       fontsize=7, alpha=0.7)

        ax.set_xlabel("ARS")
        ax.set_ylabel("ORS")
        ax.set_title(f"{label}", fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.02, 0.5)

    # Add legend
    global_patch = mpatches.Patch(color=global_color, label='Global Models', alpha=0.7)
    regional_patch = mpatches.Patch(color=regional_color, label='Regional Models', alpha=0.7)
    fig.legend(handles=[global_patch, regional_patch], loc='upper right', fontsize=10)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_model_trajectory(
    model_name: str,
    variant_points: Dict[str, Tuple[float, float]],
    variants_order: List[str],
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot a single model's trajectory across variants with arrows.

    Args:
        model_name: Name of the model
        variant_points: Dict mapping variant_name -> (ARS, ORS)
        variants_order: Order of variants (for drawing arrows)
        title: Plot title
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Draw quadrant backgrounds
    ax.axhspan(0.1, 1.0, xmin=0, xmax=0.5, alpha=0.1, color='orange')
    ax.axhspan(0.1, 1.0, xmin=0.5, xmax=1.0, alpha=0.2, color='red')
    ax.axhspan(0, 0.1, xmin=0.5, xmax=1.0, alpha=0.1, color='red')
    ax.axhspan(0, 0.1, xmin=0, xmax=0.5, alpha=0.1, color='green')

    # Draw threshold lines
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    # Collect points in order
    points = []
    labels = []
    for variant in variants_order:
        if variant in variant_points:
            points.append(variant_points[variant])
            labels.append(variant)

    # Draw trajectory with arrows
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))

    for i, ((ars, ors), label, color) in enumerate(zip(points, labels, colors)):
        ax.scatter(ars, ors, s=150, c=[color], edgecolors='black', linewidths=1, zorder=5)
        ax.annotate(f"{i+1}. {label}", (ars, ors), xytext=(8, 8), textcoords='offset points',
                   fontsize=9, fontweight='bold')

        # Draw arrow to next point
        if i < len(points) - 1:
            next_ars, next_ors = points[i + 1]
            ax.annotate('', xy=(next_ars, next_ors), xytext=(ars, ors),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.6))

    ax.set_xlabel("ARS (Adversarial Risk Score) →", fontsize=11)
    ax.set_ylabel("ORS (Over-Refusal Score) →", fontsize=11)
    ax.set_title(title or f"Trajectory: {model_name}", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, 0.5)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
