"""
Data models for harm dimension classification.
"""

from enum import Enum
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# Exact string literals for strict validation
VALID_HARM_STRINGS = Literal[
    "what - specificity",
    "how (1) - acquisition",
    "how (2) - weaponization",
    "how (3) - delivery / transmission",
    "why - vulnerability exploitation",
    "who / where - targeting guidance",
    "when - operational windows",
]


class HarmDimension(str, Enum):
    """The 7 harm dimensions from the evaluation guidelines."""
    WHAT = "what - specificity"
    HOW_1 = "how (1) - acquisition"
    HOW_2 = "how (2) - weaponization"
    HOW_3 = "how (3) - delivery / transmission"
    WHY = "why - vulnerability exploitation"
    WHO_WHERE = "who / where - targeting guidance"
    WHEN = "when - operational windows"


# Mapping from short codes to full enum values (for parsing)
HARM_DIMENSION_ALIASES = {
    # Full names
    "what - specificity": HarmDimension.WHAT,
    "how (1) - acquisition": HarmDimension.HOW_1,
    "how (2) - weaponization": HarmDimension.HOW_2,
    "how (3) - delivery / transmission": HarmDimension.HOW_3,
    "why - vulnerability exploitation": HarmDimension.WHY,
    "who / where - targeting guidance": HarmDimension.WHO_WHERE,
    "when - operational windows": HarmDimension.WHEN,
    # Short codes
    "what": HarmDimension.WHAT,
    "how-1": HarmDimension.HOW_1,
    "how-2": HarmDimension.HOW_2,
    "how-3": HarmDimension.HOW_3,
    "how_1": HarmDimension.HOW_1,
    "how_2": HarmDimension.HOW_2,
    "how_3": HarmDimension.HOW_3,
    "why": HarmDimension.WHY,
    "who/where": HarmDimension.WHO_WHERE,
    "who_where": HarmDimension.WHO_WHERE,
    "when": HarmDimension.WHEN,
    # Common variations
    "specificity": HarmDimension.WHAT,
    "acquisition": HarmDimension.HOW_1,
    "weaponization": HarmDimension.HOW_2,
    "delivery": HarmDimension.HOW_3,
    "transmission": HarmDimension.HOW_3,
    "vulnerability": HarmDimension.WHY,
    "vulnerability exploitation": HarmDimension.WHY,
    "targeting": HarmDimension.WHO_WHERE,
    "targeting guidance": HarmDimension.WHO_WHERE,
    "operational windows": HarmDimension.WHEN,
}


def parse_harm_dimension(value: str) -> Optional[HarmDimension]:
    """Parse a harm dimension from various string formats."""
    if not value:
        return None
    normalized = value.lower().strip()
    # Try direct enum lookup first
    try:
        return HarmDimension(value)
    except ValueError:
        pass
    # Try aliases
    if normalized in HARM_DIMENSION_ALIASES:
        return HARM_DIMENSION_ALIASES[normalized]
    # Try partial match
    for alias, dim in HARM_DIMENSION_ALIASES.items():
        if alias in normalized or normalized in alias:
            return dim
    return None


class RubricClassification(BaseModel):
    """Classification result for a single rubric criterion."""
    rubric_index: int = Field(..., description="Index of the rubric question (0-based)")
    rubric_question: str = Field(..., description="The rubric question text")
    primary_harm: HarmDimension = Field(..., description="Primary harm dimension assigned")
    secondary_harm: Optional[HarmDimension] = Field(
        None, description="Secondary harm dimension (only if truly ambiguous)"
    )
    secondary_harm_justification: Optional[str] = Field(
        None, description="Required justification if secondary_harm is set"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    reasoning: str = Field(..., description="Reasoning for the classification")


class TaskClassificationInput(BaseModel):
    """Input for classifying a single task's rubric criteria."""
    task_id: str = Field(..., description="Unique task identifier")
    risk_domain: str = Field(..., description="Risk domain (e.g., 'CBRNE', 'Political Violence')")
    rubric: List[str] = Field(..., description="List of rubric questions")
    adversarial_prompt: Optional[str] = Field(
        None, description="Adversarial prompt (for variant B)"
    )
    benign_prompt: Optional[str] = Field(
        None, description="Benign prompt (for variant B)"
    )


class TaskClassificationOutput(BaseModel):
    """Output from classifying a single task's rubric criteria."""
    task_id: str = Field(..., description="Unique task identifier")
    classifications: List[RubricClassification] = Field(
        ..., description="Classification for each rubric criterion"
    )
    timestamp: str = Field(..., description="ISO format timestamp")


class JudgeResult(BaseModel):
    """Result from a single judge model."""
    model_name: str = Field(..., description="Judge model identifier")
    classification: Optional[TaskClassificationOutput] = Field(
        None, description="Classification output if successful"
    )
    raw_output: str = Field(..., description="Raw model output")
    refused: bool = Field(False, description="Whether the model refused to classify")
    error: Optional[str] = Field(None, description="Error message if failed")
    latency_ms: float = Field(..., description="Request latency in milliseconds")


class ConsensusResult(BaseModel):
    """Consensus result from multiple judges."""
    task_id: str = Field(..., description="Unique task identifier")
    variant: str = Field(..., description="Variant type: 'A' (rubric-only) or 'B' (with-context)")
    classifications: List[RubricClassification] = Field(
        ..., description="Final consensus classifications"
    )
    judge_results: Dict[str, JudgeResult] = Field(
        ..., description="Individual judge results by model name"
    )
    agreement_scores: List[float] = Field(
        ..., description="Per-rubric agreement scores (fraction of judges agreeing)"
    )
    timestamp: str = Field(..., description="ISO format timestamp")

    def to_jsonl_record(self) -> dict:
        """Convert to dict for JSONL output."""
        return {
            "task_id": self.task_id,
            "variant": self.variant,
            "classifications": [c.model_dump() for c in self.classifications],
            "judge_results": {k: v.model_dump() for k, v in self.judge_results.items()},
            "agreement_scores": self.agreement_scores,
            "timestamp": self.timestamp,
        }


# ============================================================================
# STRICT LLM OUTPUT VALIDATION MODELS
# ============================================================================

class LLMClassificationItem(BaseModel):
    """Strict model for validating a single classification from LLM output."""
    rubric_index: int = Field(..., ge=0, description="0-based rubric index")
    rubric_question: str = Field(..., min_length=1, description="Rubric question text")
    primary_harm: str = Field(..., description="Primary harm dimension")
    secondary_harm: Optional[str] = Field(None, description="Secondary harm dimension")
    secondary_harm_justification: Optional[str] = Field(None)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=1)

    @field_validator('primary_harm')
    @classmethod
    def validate_primary_harm(cls, v: str) -> str:
        """Validate primary_harm is a valid dimension (with alias support)."""
        dim = parse_harm_dimension(v)
        if dim is None:
            valid_values = [e.value for e in HarmDimension]
            raise ValueError(
                f"Invalid primary_harm '{v}'. Must be one of: {valid_values}"
            )
        return dim.value  # Normalize to canonical form

    @field_validator('secondary_harm')
    @classmethod
    def validate_secondary_harm(cls, v: Optional[str]) -> Optional[str]:
        """Validate secondary_harm if provided."""
        if v is None:
            return None
        dim = parse_harm_dimension(v)
        if dim is None:
            valid_values = [e.value for e in HarmDimension]
            raise ValueError(
                f"Invalid secondary_harm '{v}'. Must be one of: {valid_values}"
            )
        return dim.value  # Normalize to canonical form


class LLMClassificationOutput(BaseModel):
    """Strict model for validating full LLM classification output."""
    classifications: List[LLMClassificationItem] = Field(
        ..., min_length=1, description="List of classifications, one per rubric item"
    )

    @classmethod
    def validate_against_rubric(
        cls,
        output: "LLMClassificationOutput",
        expected_count: int
    ) -> List[str]:
        """
        Validate output against expected rubric count.
        Returns list of validation errors (empty if valid).
        """
        errors = []
        actual_count = len(output.classifications)

        if actual_count != expected_count:
            errors.append(
                f"Expected {expected_count} classifications, got {actual_count}"
            )

        # Check for duplicate indices
        indices = [c.rubric_index for c in output.classifications]
        if len(indices) != len(set(indices)):
            errors.append(f"Duplicate rubric indices found: {indices}")

        # Check indices are in valid range
        for idx in indices:
            if idx < 0 or idx >= expected_count:
                errors.append(f"Invalid rubric_index {idx} (expected 0-{expected_count-1})")

        return errors


class ParseError(Exception):
    """Raised when LLM output cannot be parsed."""
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output
