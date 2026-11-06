"""
Heatmap visualization for confusion matrices.

Generates:
- 2x2 Y/N confusion matrix heatmaps
- 7x7 dimension confusion heatmaps
- Human inter-annotator agreement heatmaps
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..metrics.confusion import ConfusionMatrix, DimensionConfusionMatrix


def plot_yn_confusion_matrix(
    cm: ConfusionMatrix,
    title: str = "Y/N Confusion Matrix",
    output_path: Optional[Path] = None,
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """
    Plot a 2x2 Y/N confusion matrix as a heatmap.
    
    Args:
        cm: ConfusionMatrix object
        title: Plot title
        output_path: Path to save the figure (optional)
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create matrix: [[TN, FP], [FN, TP]]
    matrix = np.array([[cm.tn, cm.fp], [cm.fn, cm.tp]])
    
    # Create heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NO", "YES"],
        yticklabels=["NO", "YES"],
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    
    ax.set_xlabel("Judge Prediction")
    ax.set_ylabel("Human Label")
    ax.set_title(title)
    
    # Add metrics annotation
    metrics_text = (
        f"Accuracy: {cm.accuracy:.1%}\n"
        f"Precision: {cm.precision:.1%}\n"
        f"Recall: {cm.recall:.1%}\n"
        f"F1: {cm.f1:.1%}"
    )
    ax.text(
        1.35, 0.5, metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_dimension_heatmap(
    dcm: DimensionConfusionMatrix,
    title: str = "Dimension Confusion Matrix",
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
    normalize: bool = False,
) -> plt.Figure:
    """
    Plot a 7x7 dimension confusion matrix as a heatmap.
    
    Args:
        dcm: DimensionConfusionMatrix object
        title: Plot title
        output_path: Path to save the figure (optional)
        figsize: Figure size
        normalize: Whether to normalize rows (show as percentages)
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get matrix
    matrix = dcm.to_array().astype(float)
    
    # Normalize if requested
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / row_sums
    
    # Short labels for dimensions
    dim_labels = ["WHAT", "HOW-1", "HOW-2", "HOW-3", "WHY", "WHO/WHERE", "WHEN"]
    
    # Create heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0%" if normalize else ".0f",
        cmap="YlOrRd",
        xticklabels=dim_labels,
        yticklabels=dim_labels,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    
    ax.set_xlabel("Predicted Dimension (Judge)")
    ax.set_ylabel("True Dimension (Human)")
    ax.set_title(title)
    
    # Rotate x labels for readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    # Add accuracy annotation
    accuracy_text = f"Overall Accuracy: {dcm.accuracy:.1%}"
    ax.text(
        0.5, -0.15, accuracy_text,
        transform=ax.transAxes,
        fontsize=11,
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_multi_judge_comparison(
    confusion_matrices: Dict[str, ConfusionMatrix],
    title: str = "Judge Comparison",
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Plot multiple judge confusion matrices side by side.
    
    Args:
        confusion_matrices: Dict mapping judge_name -> ConfusionMatrix
        title: Overall plot title
        output_path: Path to save the figure (optional)
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    n_judges = len(confusion_matrices)
    fig, axes = plt.subplots(1, n_judges, figsize=figsize)
    
    if n_judges == 1:
        axes = [axes]
    
    for ax, (judge_name, cm) in zip(axes, confusion_matrices.items()):
        matrix = np.array([[cm.tn, cm.fp], [cm.fn, cm.tp]])
        
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["NO", "YES"],
            yticklabels=["NO", "YES"],
            ax=ax,
            cbar=False,
        )
        
        ax.set_xlabel("Prediction")
        ax.set_ylabel("True")
        ax.set_title(f"{judge_name}\n(F1: {cm.f1:.1%})")
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_variant_comparison_heatmap(
    matrices_by_variant: Dict[str, ConfusionMatrix],
    metric: str = "f1",
    title: str = "Judge Performance by Variant",
    output_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Plot a heatmap comparing judge performance across variants.
    
    Args:
        matrices_by_variant: Dict mapping variant_name -> ConfusionMatrix
        metric: Metric to display ("accuracy", "precision", "recall", "f1")
        title: Plot title
        output_path: Path to save the figure (optional)
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    variants = list(matrices_by_variant.keys())
    values = []
    
    for variant in variants:
        cm = matrices_by_variant[variant]
        if metric == "accuracy":
            values.append(cm.accuracy)
        elif metric == "precision":
            values.append(cm.precision)
        elif metric == "recall":
            values.append(cm.recall)
        else:  # f1
            values.append(cm.f1)
    
    # Create bar chart
    colors = sns.color_palette("husl", len(variants))
    bars = ax.barh(variants, values, color=colors)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            fontsize=10,
        )
    
    ax.set_xlabel(metric.capitalize())
    ax.set_title(title)
    ax.set_xlim(0, 1.15)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_human_agreement_heatmap(
    human1_dims: List[str],
    human2_dims: List[str],
    title: str = "Human Inter-Annotator Agreement",
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Plot a heatmap of Human 1 vs Human 2 dimension labels.
    
    Helps identify commonly confused dimensions (subjective cases).
    
    Args:
        human1_dims: Dimension labels from annotator 1
        human2_dims: Dimension labels from annotator 2
        title: Plot title
        output_path: Path to save the figure (optional)
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    from ..metrics.confusion import build_dimension_confusion_matrix
    
    dcm = build_dimension_confusion_matrix(human1_dims, human2_dims)
    
    return plot_dimension_heatmap(
        dcm,
        title=title,
        output_path=output_path,
        figsize=figsize,
        normalize=True,
    )
