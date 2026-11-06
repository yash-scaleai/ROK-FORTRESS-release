"""
Data models aligned with the upstream FORTRESS structures.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromptVariant:
    """Single prompt variant (adversarial + benign + direct + rubric)."""
    task_id: str
    task_type: str
    variant_name: str
    adversarial_prompt: str
    benign_prompt: str
    rubric: List[str]
    risk_domain: str
    risk_subdomain: str
    direct_prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Raw response returned by a generation model."""
    model_name: str
    prompt_type: str  # "adversarial", "benign", or "direct"
    response_text: str
    latency_ms: float
    tokens_used: Optional[int] = None
    error: Optional[str] = None
    reasoning_content: Optional[str] = None  # Chain-of-thought from thinking models


@dataclass
class RubricEvaluation:
    """Evaluation for a single rubric question from one judge."""
    verdict: Optional[bool]
    reasoning: str


@dataclass
class JudgeEvaluation:
    """Evaluation bundle from a single judge."""
    score: int
    yes_percentage: float
    evaluations: List[RubricEvaluation] = field(default_factory=list)
    raw_output: Optional[str] = None


@dataclass
class RefusalResult:
    """Simple refusal check result."""
    verdict: bool
    reasoning: str

@dataclass
class ResponseEvaluation:
    """Aggregate evaluation for an adversarial or benign response."""
    response: str
    refused: bool = False
    refusal_reason: Optional[str] = None
    judge_evaluations: Dict[str, JudgeEvaluation] = field(default_factory=dict)
    final_score: float = 0.0  # Percentage of consensus YES verdicts
    notes: Optional[str] = None


@dataclass
class EvaluationResult:
    """Top-level evaluation result for a prompt/model combination."""
    task_id: str
    task_type: str
    model_name: str
    risk_domain: str
    risk_subdomain: str
    adversarial: Optional[ResponseEvaluation] = None
    benign: Optional[ResponseEvaluation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
