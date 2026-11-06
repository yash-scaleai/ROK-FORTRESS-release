"""
ROK-FORTRESS Paper Figure Generation Module.

This module provides research-grade plotting utilities for generating
publication-quality figures (PDF format preferred).

Usage:
    from analysis.experiments.plotting import setup_paper_style, save_figure

    setup_paper_style()
    fig, ax = plt.subplots()
    # ... plotting code ...
    save_figure(fig, "my_figure")  # Saves as PDF in figures/
"""

from .base import (
    setup_paper_style,
    save_figure,
    FIGURE_DIR,
    # Colors
    FAMILY_COLORS,
    MODEL_FAMILY_MAP,
    VARIANT_COLORS,
    VARIANT_LABELS,
    VARIANT_LABELS_SHORT,
    DOMAIN_COLORS,
    DOMAIN_LABELS,
    # Helper functions
    get_model_color,
    get_model_family,
    get_variant_color,
    get_variant_label,
)

__all__ = [
    "setup_paper_style",
    "save_figure",
    "FIGURE_DIR",
    "FAMILY_COLORS",
    "MODEL_FAMILY_MAP",
    "VARIANT_COLORS",
    "VARIANT_LABELS",
    "VARIANT_LABELS_SHORT",
    "DOMAIN_COLORS",
    "DOMAIN_LABELS",
    "get_model_color",
    "get_model_family",
    "get_variant_color",
    "get_variant_label",
]
