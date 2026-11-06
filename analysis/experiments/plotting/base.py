"""
Base plotting utilities for ROK-FORTRESS research figures.

This module provides:
- Paper-quality matplotlib configuration
- Consistent color schemes for models, variants, and domains
- Utility functions for saving figures in publication formats
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib as mpl

# =============================================================================
# Output Directory
# =============================================================================

FIGURE_DIR = Path(__file__).parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


# =============================================================================
# Color Schemes
# =============================================================================

# Model family colors (colorblind-friendly palette)
FAMILY_COLORS = {
    "OpenAI": "#E64B35",       # Vermilion
    "Anthropic": "#7E6148",    # Brown
    "Google": "#4DBBD5",       # Cyan
    "Meta": "#00A087",         # Teal
    "Mistral": "#8491B4",      # Purple-gray
    "DeepSeek": "#3C5488",     # Navy blue
    "Kimi": "#F39B7F",         # Light coral
    "Alibaba": "#91D1C2",      # Light teal
    # Regional models
    "Naver": "#00C73C",        # Naver green
    "Kakao": "#B09C85",        # Kakao tan (adjusted for visibility)
    "LG": "#DC0000",           # LG red
    "Upstage": "#7570B3",      # Purple
    "Other": "#999999",        # Gray
}

# Model to family mapping
MODEL_FAMILY_MAP = {
    # Global models
    "gpt-5.2": "OpenAI",
    "gpt-5.1": "OpenAI",
    "o4-mini": "OpenAI",
    "claude-opus-4-5": "Anthropic",
    "claude-sonnet-4-5": "Anthropic",
    "gemini-3-pro-preview": "Google",
    "llama4-maverick-instruct-basic": "Meta",
    "mistral.mistral-large-3-675b-instruct": "Mistral",
    "deepseek-v3p2": "DeepSeek",
    "kimi-k2-thinking": "Kimi",
    "qwen.qwen3-235b-a22b-2507-v1:0": "Alibaba",
    # Regional models
    "hyperclovax": "Naver",
    "kanana2": "Kakao",
    "kexaone": "LG",
    "solar_open": "Upstage",
}

# Variant colors (sequential for language/culture progression)
VARIANT_COLORS = {
    "original_en": "#2166AC",           # Dark blue
    "translated_kr": "#67A9CF",         # Light blue
    "cultural_adapted_en": "#EF8A62",   # Light red
    "transcreated_kr": "#B2182B",       # Dark red
}

# Variant display labels
VARIANT_LABELS = {
    "original_en": "Original (EN)",
    "translated_kr": "Translated (KO)",
    "cultural_adapted_en": "Adapted (EN)",
    "transcreated_kr": "Transcreated (KO)",
}

# Short variant labels for space-constrained plots
VARIANT_LABELS_SHORT = {
    "original_en": "V(En)",
    "translated_kr": "V(Ko)",
    "cultural_adapted_en": "V(En,US)",
    "transcreated_kr": "V(Ko,KR)",
}

# Domain colors
DOMAIN_COLORS = {
    "CBRNE": "#D73027",           # Red
    "Political": "#FC8D59",       # Orange
    "Criminal": "#FEE090",        # Yellow
    "Information": "#91BFDB",     # Blue
}

# Domain display labels
DOMAIN_LABELS = {
    "CBRNE": "CBRNE",
    "Political": "Political Violence",
    "Criminal": "Criminal Activity",
    "Information": "Information Leakage",
}


# =============================================================================
# Matplotlib Configuration
# =============================================================================

def setup_paper_style(
    font_family: str = "serif",
    font_size: int = 10,
    use_latex: bool = False,
) -> None:
    """
    Configure matplotlib for publication-quality figures.

    Args:
        font_family: Font family to use ("serif", "sans-serif", "monospace")
        font_size: Base font size in points
        use_latex: Whether to use LaTeX for text rendering (requires LaTeX installation)
    """
    # Use a clean style as base
    plt.style.use("seaborn-v0_8-whitegrid")

    # Paper-quality settings
    params = {
        # Font settings
        "font.family": font_family,
        "font.size": font_size,
        "axes.labelsize": font_size + 2,
        "axes.titlesize": font_size + 4,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size + 6,

        # Figure settings
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "figure.autolayout": False,

        # Save settings
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",

        # Axes settings
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Grid settings
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,

        # Legend settings
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.facecolor": "white",
        "legend.edgecolor": "#CCCCCC",

        # Line settings
        "lines.linewidth": 1.5,
        "lines.markersize": 6,

        # Patch settings (for bars, etc.)
        "patch.linewidth": 0.5,
        "patch.edgecolor": "#333333",

        # PDF/PS settings for publication
        "pdf.fonttype": 42,  # TrueType fonts
        "ps.fonttype": 42,
    }

    # LaTeX settings (optional)
    if use_latex:
        params.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        })

    plt.rcParams.update(params)


def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: Optional[Path] = None,
    formats: tuple = ("pdf",),
    close: bool = True,
) -> list[Path]:
    """
    Save figure in publication formats.

    Args:
        fig: Matplotlib figure to save
        name: Base filename (without extension)
        output_dir: Output directory (defaults to FIGURE_DIR)
        formats: Tuple of formats to save ("pdf", "png", "svg")
        close: Whether to close the figure after saving

    Returns:
        List of saved file paths
    """
    if output_dir is None:
        output_dir = FIGURE_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(
            path,
            format=fmt,
            bbox_inches="tight",
            pad_inches=0.1,
            facecolor="white",
            edgecolor="none",
        )
        saved_paths.append(path)
        print(f"Saved: {path}")

    if close:
        plt.close(fig)

    return saved_paths


# =============================================================================
# Helper Functions
# =============================================================================

def get_model_color(model_name: str) -> str:
    """Get color for a model based on its family."""
    family = MODEL_FAMILY_MAP.get(model_name, "Other")
    return FAMILY_COLORS.get(family, FAMILY_COLORS["Other"])


def get_model_family(model_name: str) -> str:
    """Get family name for a model."""
    return MODEL_FAMILY_MAP.get(model_name, "Other")


def get_variant_color(variant_name: str) -> str:
    """Get color for a variant."""
    return VARIANT_COLORS.get(variant_name, "#999999")


def get_variant_label(variant_name: str, short: bool = False) -> str:
    """Get display label for a variant."""
    if short:
        return VARIANT_LABELS_SHORT.get(variant_name, variant_name)
    return VARIANT_LABELS.get(variant_name, variant_name)


def create_model_legend(
    ax: plt.Axes,
    models: list[str],
    loc: str = "best",
    **kwargs,
) -> None:
    """
    Create a legend showing model families.

    Args:
        ax: Matplotlib axes
        models: List of model names to include
        loc: Legend location
        **kwargs: Additional arguments passed to ax.legend()
    """
    from matplotlib.patches import Patch

    # Get unique families for models
    families = {}
    for model in models:
        family = get_model_family(model)
        if family not in families:
            families[family] = get_model_color(model)

    handles = [
        Patch(facecolor=color, edgecolor="white", label=family)
        for family, color in families.items()
    ]

    ax.legend(handles=handles, loc=loc, **kwargs)


def create_variant_legend(
    ax: plt.Axes,
    variants: list[str],
    short_labels: bool = False,
    loc: str = "best",
    **kwargs,
) -> None:
    """
    Create a legend showing variants.

    Args:
        ax: Matplotlib axes
        variants: List of variant names to include
        short_labels: Whether to use short labels
        loc: Legend location
        **kwargs: Additional arguments passed to ax.legend()
    """
    from matplotlib.patches import Patch

    handles = [
        Patch(
            facecolor=get_variant_color(v),
            edgecolor="white",
            label=get_variant_label(v, short=short_labels),
        )
        for v in variants
    ]

    ax.legend(handles=handles, loc=loc, **kwargs)
