"""
Harm dimension classifier module for ROK-FORTRESS.

Assigns harm dimensions to rubric criteria using LLM judge panels.
"""

from .data_models import (
    HarmDimension,
    RubricClassification,
    TaskClassificationInput,
    TaskClassificationOutput,
    JudgeResult,
    ConsensusResult,
)
from .classifier import classify_task
from .judge_panel import JudgePanel, classify_task_with_panel
from .meta_eval import run_meta_eval, MetaEvalResult

__all__ = [
    "HarmDimension",
    "RubricClassification",
    "TaskClassificationInput",
    "TaskClassificationOutput",
    "JudgeResult",
    "ConsensusResult",
    "classify_task",
    "JudgePanel",
    "classify_task_with_panel",
    "run_meta_eval",
    "MetaEvalResult",
]
