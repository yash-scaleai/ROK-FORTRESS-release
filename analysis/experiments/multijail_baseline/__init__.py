"""
MultiJail Baseline Experiment

Replication of the MultiJail experiment (ICLR 2024) to test whether the unsafe rate gap
between English and low-resource languages has decreased in 2026-era frontier models.

Reference: https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs
Paper: "Multilingual Jailbreak Challenges in Large Language Models" (Deng et al., 2024)
"""

from .config import (
    MULTIJAIL_MODELS,
    JUDGE_MODEL,
    LANGUAGES,
    LANGUAGE_TIERS,
    AIM_PREFIX,
)
from .data_loader import (
    MultiJailPrompt,
    load_multijail_dataset,
    apply_aim_template,
)

__all__ = [
    "MULTIJAIL_MODELS",
    "JUDGE_MODEL", 
    "LANGUAGES",
    "LANGUAGE_TIERS",
    "AIM_PREFIX",
    "MultiJailPrompt",
    "load_multijail_dataset",
    "apply_aim_template",
]
