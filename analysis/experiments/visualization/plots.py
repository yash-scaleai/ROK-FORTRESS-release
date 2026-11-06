"""
Comparison plots for experiment results.

Generates:
- Drop comparison bar charts (Linguistic vs Contextual)
- Harm score distributions by variant
- Model comparison charts
- Modality and geopolitical gap visualizations
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..metrics.alignment_metrics import DropResult
from ..metrics.harm_scores import HarmScoreResult


def plot_drop_comparison(
    models: List[str],
    linguistic_drops: List[DropResult],
    contextual_drops: List[DropResult],
    title: str = r'Linguistic $\Delta_{\text{ling}}$ and Contextual $\Delta_{\text{ctx}}$ Drop',
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
    score_type: str = "weighted",
    experiment: Optional[str] = None,
) -> plt.Figure:
    """
    Plot bar chart comparing Linguistic Drop vs Contextual Drop across models.
    
    This is the key visualization for Experiment 1 (Global Alignment Overfitting).
    
    Args:
        models: List of model names
        linguistic_drops: DropResult for each model
        contextual_drops: DropResult for each model
        title: Plot title
        output_path: Path to save the figure (optional)
        figsize: Figure size
        score_type: "weighted" or "ars" for subtitle display
        experiment: "exp1" or "exp2" to show experiment-specific formulas on y-axis
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(models))
    width = 0.35
    
    # Scale values to 100
    ling_values = [d.drop_value * 100 for d in linguistic_drops]
    cont_values = [d.drop_value * 100 for d in contextual_drops]
    
    # Error bars from 95% CI (preferred) or fallback to approximate SE, scaled to 100
    def get_ci_95(d: DropResult) -> float:
        if d.ci_95 is not None:
            return d.ci_95 * 100
        # Fallback: approximate using SE of mean difference
        if d.source_n > 1:
            return 1.96 * d.target_std / np.sqrt(d.source_n) * 100
        return 0.0
    
    ling_err = [get_ci_95(d) for d in linguistic_drops]
    cont_err = [get_ci_95(d) for d in contextual_drops]
    
    bars1 = ax.bar(
        x - width / 2, ling_values, width,
        label=r'$\Delta_{\text{ling}}$',
        color="steelblue",
        yerr=ling_err,
        capsize=3,
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2, cont_values, width,
        label=r'$\Delta_{\text{ctx}}$',
        color="coral",
        yerr=cont_err,
        capsize=3,
        alpha=0.8,
    )
    
    # Add significance markers for both drops
    for i, (ld, cd) in enumerate(zip(linguistic_drops, contextual_drops)):
        # Mark linguistic drop significance
        if ld.is_significant:
            y_pos = ling_values[i] - ling_err[i] - 1.5 if ling_values[i] < 0 else ling_values[i] + ling_err[i] + 3.5
            ax.annotate(
                "*",
                xy=(x[i] - width / 2, y_pos),
                ha="center",
                fontsize=14,
                fontweight="bold",
                color="steelblue",
            )
        # Mark contextual drop significance
        if cd.is_significant:
            y_pos = cont_values[i] - cont_err[i] - 1.5 if cont_values[i] < 0 else cont_values[i] + cont_err[i] + 3.5
            ax.annotate(
                "*",
                xy=(x[i] + width / 2, y_pos),
                ha="center",
                fontsize=14,
                fontweight="bold",
                color="coral",
            )
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Drop value")
    
    ax.set_title(title)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    
    # Add interpretation note
    ax.text(
        0.02, 0.98,
        "Error bars = 95% CI\n* = p < 0.05",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
    )
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_harm_scores_by_variant(
    variant_scores: Dict[str, List[float]],
    title: str = "Harm Score Distribution by Variant",
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot box plot of harm scores by variant.
    
    Args:
        variant_scores: Dict mapping variant_name -> list of harm scores
        title: Plot title
        output_path: Path to save the figure (optional)
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for boxplot
    data = []
    labels = []
    for variant, scores in variant_scores.items():
        data.append(scores)
        labels.append(variant)
    
    # Create boxplot
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # Color by language/context
    colors = {
        "original_en": "lightblue",
        "translated_kr": "lightcoral",
        "cultural_adapted_en": "lightskyblue",
        "transcreated_kr": "salmon",
    }
    
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(colors.get(label, "lightgray"))
    
    ax.set_xlabel("Variant")
    ax.set_ylabel("Harm Score")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    
    # Add mean markers
    for i, scores in enumerate(data):
        mean_val = np.mean(scores)
        ax.scatter(i + 1, mean_val, color="red", marker="D", s=50, zorder=3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_model_comparison(
    models: List[str],
    scores: List[float],
    title: str = "Model Comparison",
    metric_name: str = "Harm Score",
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    highlight_threshold: Optional[float] = None,
) -> plt.Figure:
    """
    Plot horizontal bar chart comparing models on a single metric.
    
    Args:
        models: List of model names
        scores: Score for each model
        title: Plot title
        metric_name: Name of the metric being plotted
        output_path: Path to save the figure (optional)
        figsize: Figure size
        highlight_threshold: Optional threshold line to draw
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by score
    sorted_pairs = sorted(zip(models, scores), key=lambda x: x[1], reverse=True)
    sorted_models, sorted_scores = zip(*sorted_pairs)
    
    # Color by category (global vs regional)
    from ..exp_data.models import is_regional_model
    colors = ["coral" if is_regional_model(m) else "steelblue" for m in sorted_models]
    
    y_pos = np.arange(len(sorted_models))
    bars = ax.barh(y_pos, sorted_scores, color=colors, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_models)
    ax.set_xlabel(metric_name)
    ax.set_title(title)
    
    # Add value labels
    for bar, score in zip(bars, sorted_scores):
        ax.text(
            score + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=9,
        )
    
    # Add threshold line if specified
    if highlight_threshold is not None:
        ax.axvline(x=highlight_threshold, color="red", linestyle="--", alpha=0.7)
    
    # Add legend for model categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="Global Models"),
        Patch(facecolor="coral", label="Regional Models"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_modality_gap(
    models: List[str],
    generative_scores: List[float],
    retrieval_scores: List[float],
    title: str = "Modality Gap: Generative vs Retrieval Dimensions",
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot grouped bar chart for Experiment 4.1 (Modality Gap).
    
    Compares harm scores on Generative (HOW-2, HOW-3) vs Retrieval (WHAT, WHO/WHERE) dimensions.
    
    Args:
        models: List of model names
        generative_scores: Harm scores for generative dimensions
        retrieval_scores: Harm scores for retrieval dimensions
        title: Plot title
        output_path: Path to save the figure (optional)
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(
        x - width / 2, generative_scores, width,
        label="Generative (HOW-2, HOW-3)",
        color="forestgreen",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2, retrieval_scores, width,
        label="Retrieval (WHAT, WHO/WHERE)",
        color="darkorange",
        alpha=0.8,
    )
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Harm Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    
    # Add gap annotations
    for i, (gen, ret) in enumerate(zip(generative_scores, retrieval_scores)):
        gap = ret - gen
        if abs(gap) > 0.05:
            ax.annotate(
                f"Δ={gap:+.2f}",
                xy=(x[i], max(gen, ret) + 0.02),
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_geopolitical_gap(
    models: List[str],
    us_scores: List[float],
    kr_scores: List[float],
    title: str = "Geopolitical Gap: US vs ROK Secret Protection",
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot grouped bar chart for Experiment 4.2 (Geopolitical Gap).
    
    Compares harm scores on US-context vs KR-context prompts in Information Leakage domain.
    
    Args:
        models: List of model names
        us_scores: Harm scores for US-context (CA) prompts
        kr_scores: Harm scores for KR-context (CS) prompts
        title: Plot title
        output_path: Path to save the figure (optional)
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(
        x - width / 2, us_scores, width,
        label="US Context (Foreign Secrets)",
        color="navy",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2, kr_scores, width,
        label="KR Context (Domestic Secrets)",
        color="crimson",
        alpha=0.8,
    )
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Harm Score (Higher = More Leakage)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    
    # Add bias annotations
    for i, (us, kr) in enumerate(zip(us_scores, kr_scores)):
        bias = us - kr
        if abs(bias) > 0.05:
            color = "red" if bias > 0 else "green"
            ax.annotate(
                f"Bias={bias:+.2f}",
                xy=(x[i], max(us, kr) + 0.02),
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=color,
            )
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_overfitting_summary(
    models: List[str],
    overfitting_scores: List[float],
    title: str = "Alignment Overfitting Score by Model",
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot alignment overfitting scores (Contextual Drop - Linguistic Drop).
    
    Higher score indicates more alignment overfitting.
    
    Args:
        models: List of model names
        overfitting_scores: Overfitting score for each model
        title: Plot title
        output_path: Path to save the figure (optional)
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by overfitting score
    sorted_pairs = sorted(zip(models, overfitting_scores), key=lambda x: x[1], reverse=True)
    sorted_models, sorted_scores = zip(*sorted_pairs)
    
    # Color by severity
    colors = ["darkred" if s > 0.1 else "orange" if s > 0.05 else "green" for s in sorted_scores]
    
    y_pos = np.arange(len(sorted_models))
    bars = ax.barh(y_pos, sorted_scores, color=colors, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_models)
    ax.set_xlabel("Overfitting Score (Contextual Drop - Linguistic Drop)")
    ax.set_title(title)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    
    # Add interpretation
    ax.text(
        0.02, 0.98,
        "← Better generalization | Overfitting →",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
    )
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig


def _plot_variant_panel(
    ax: plt.Axes,
    models: List[str],
    variant_scores: Dict[str, Dict[str, float]],
    variants: List[str],
    variant_labels: List[str],
    colors: List[str],
    panel_title: str,
    ylabel: str,
    show_mean_line: bool = True,
    annotate_values: bool = False,
    show_xticklabels: bool = True,
    ylim: Tuple[float, float] = (0, 80),
) -> None:
    """Render a single CA or CS variant panel."""
    x = np.arange(len(models))
    width = min(0.35, 0.78 / max(len(variants), 1))
    offsets = [(i - (len(variants) - 1) / 2) * width for i in range(len(variants))]

    all_scores = []
    bar_groups = []

    for variant, label, color, offset in zip(variants, variant_labels, colors, offsets):
        scores = []
        for model in models:
            model_scores = variant_scores.get(model, {})
            score = model_scores.get(variant, 0.0) * 100
            scores.append(score)
            if score > 0:
                all_scores.append(score)

        bars = ax.bar(
            x + offset,
            scores,
            width,
            label=label,
            color=color,
        )
        bar_groups.append((bars, scores))

    if annotate_values:
        for bars, scores in bar_groups:
            for bar, score in zip(bars, scores):
                if score > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{score:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

    if show_mean_line and all_scores:
        mean_ars = np.mean(all_scores)
        ax.axhline(
            y=mean_ars,
            color="black",
            linestyle="--",
            linewidth=1.25,
            alpha=0.7,
            label=f"Mean = {mean_ars:.1f}",
        )

    ax.set_title(panel_title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_ylim(*ylim)

    if show_xticklabels:
        ax.set_xticklabels(models, rotation=45, ha="right")
    else:
        ax.tick_params(axis="x", labelbottom=False)

    legend_cols = min(len(variants) + (1 if show_mean_line and all_scores else 0), 5)
    ax.legend(loc="upper center", ncol=legend_cols, frameon=False, fontsize=9)


def plot_weighted_ars_by_variant_two_panel(
    models: List[str],
    ca_variant_scores: Dict[str, Dict[str, float]],
    cs_variant_scores: Dict[str, Dict[str, float]],
    title: str = "Tier-weighted Risk Score by Model and Prompt Variants",
    output_path: Optional[Path] = None,
    figsize: tuple = (16, 10.5),
    show_mean_line: bool = True,
    ylabel: str = "Tier-weighted Risk Score",
    annotate_values: bool = False,
) -> plt.Figure:
    """
    Plot vertically stacked CA and CS TRS panels.

    The top panel shows culture-agnostic task variants V(En) and V(Ko).
    The bottom panel shows culture-specific task variants V(En,US),
    V(Ko,US), V(En,KR), and V(Ko,KR).
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)

    colors = {
        "original_en": "#7EB6D9",
        "translated_kr": "#E8A898",
        "cultural_adapted_en": "#2B5797",
        "transcreated_kr": "#C45B5B",
    }

    _plot_variant_panel(
        ax=axes[0],
        models=models,
        variant_scores=ca_variant_scores,
        variants=["original_en", "translated_kr"],
        variant_labels=[r"$V_{En}$", r"$V_{Ko}$"],
        colors=[colors["original_en"], colors["translated_kr"]],
        panel_title="Culture-Agnostic",
        ylabel=ylabel,
        show_mean_line=show_mean_line,
        annotate_values=annotate_values,
        show_xticklabels=False,
    )

    _plot_variant_panel(
        ax=axes[1],
        models=models,
        variant_scores=cs_variant_scores,
        variants=["original_en", "translated_kr", "cultural_adapted_en", "transcreated_kr"],
        variant_labels=[
            r"$V_{En,\,US}$",
            r"$V_{Ko,\,US}$",
            r"$V_{En,\,KR}$",
            r"$V_{Ko,\,KR}$",
        ],
        colors=[
            colors["original_en"],
            colors["translated_kr"],
            colors["cultural_adapted_en"],
            colors["transcreated_kr"],
        ],
        panel_title="Culture-Specific",
        ylabel=ylabel,
        show_mean_line=show_mean_line,
        annotate_values=annotate_values,
        show_xticklabels=True,
    )

    axes[1].set_xlabel("Model")
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_weighted_ars_by_variant(
    models: List[str],
    variant_scores: Dict[str, Dict[str, float]],
    title: str = "Tier-weighted Risk Score by Model and Prompt Variants",
    output_path: Optional[Path] = None,
    figsize: tuple = (16, 8),
    show_mean_line: bool = True,
    ylabel: str = "Tier-weighted Risk Score",
) -> plt.Figure:
    """
    Plot grouped bar chart showing Weighted ARS by Model and Rubric/Variant Type.
    
    Creates a grouped bar chart with 4 bars per model:
    - original_en: Original (EN) - light blue
    - translated_kr: Translated (KR) - light red
    - cultural_adapted_en: Culturally Adapted (EN) - dark blue
    - transcreated_kr: Transcreated (KR) - dark red
    
    Args:
        models: List of model display names (in order to plot)
        variant_scores: Dict mapping model_name -> {variant_name -> score}
        title: Plot title
        output_path: Path to save the figure (optional)
        figsize: Figure size
        show_mean_line: Whether to show horizontal dashed line at mean weighted ARS
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Variant definitions matching reference image
    variants = ["original_en", "translated_kr", "cultural_adapted_en", "transcreated_kr"]
    # LaTeX-style legend labels (using matplotlib mathtext)
    variant_labels = [
        r'$V_{En}$ & $V_{En,\,US}$',
        r'$V_{Ko}$ & $V_{Ko,\,US}$',
        r'$V_{En,\,KR}$',
        r'$V_{Ko,\,KR}$',
    ]
    # Colors: blue tones for EN, red tones for KR (light for original/translated, dark for adapted)
    colors = ["#7EB6D9", "#E8A898", "#2B5797", "#C45B5B"]
    
    x = np.arange(len(models))
    width = 0.2
    
    # Collect all scores for computing the mean (scaled to 100)
    all_scores = []
    bar_groups = []
    
    for i, (variant, label, color) in enumerate(zip(variants, variant_labels, colors)):
        scores = []
        for model in models:
            model_scores = variant_scores.get(model, {})
            score = model_scores.get(variant, 0.0) * 100  # Scale to 100
            scores.append(score)
            if score > 0:
                all_scores.append(score)
        
        bars = ax.bar(
            x + (i - 1.5) * width, scores, width,
            label=label,
            color=color,
        )
        bar_groups.append((bars, scores))
    
    # Add value labels on top of each bar
    for bars, scores in bar_groups:
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{score:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )
    
    # Add horizontal dashed line at mean weighted ARS
    if show_mean_line and all_scores:
        mean_ars = np.mean(all_scores)
        ax.axhline(
            y=mean_ars,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Mean = {mean_ars:.1f}",
        )
    
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 80)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    return fig
