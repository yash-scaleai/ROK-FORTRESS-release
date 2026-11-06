"""
Multi-model judge panel for harm dimension classification.

Runs multiple judge models in parallel and computes consensus.
"""

import asyncio
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

from .data_models import (
    HarmDimension,
    RubricClassification,
    TaskClassificationInput,
    ConsensusResult,
    JudgeResult,
)
from .classifier import classify_task


# Default judge panel: 1 frontier + 1 opensource + 1 either
DEFAULT_JUDGES = [
    "openai/gpt-5.2",                        # frontier
    "fireworks_ai/llama4-maverick-instruct-basic",  # opensource
    "anthropic/claude-sonnet-4-5",           # either/frontier
]

# Alternative/fallback judges
FALLBACK_JUDGES = [
    "gemini/gemini-3-pro-preview",           # frontier
    "fireworks_ai/deepseek-v3p2",            # opensource
    "bedrock/qwen.qwen3-235b-a22b-2507-v1:0",  # opensource
]


class JudgePanel:
    """
    Multi-model judge panel for harm dimension classification.

    Runs 3 judges in parallel, computes majority vote consensus.
    """

    def __init__(
        self,
        judges: Optional[List[str]] = None,
        fallback_judges: Optional[List[str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        timeout: float = 120.0,
        reasoning_effort: Optional[str] = None,
    ):
        """
        Initialize judge panel.

        Args:
            judges: List of judge model names (default: DEFAULT_JUDGES)
            fallback_judges: Fallback judges if primary judges refuse
            temperature: Sampling temperature for all judges
            max_tokens: Max tokens for each judge
            timeout: Timeout per judge request in seconds
            reasoning_effort: Reasoning mode - "low", "medium", "high", or None (disabled)
        """
        self.judges = judges or DEFAULT_JUDGES
        self.fallback_judges = fallback_judges or FALLBACK_JUDGES
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort

    async def classify_with_consensus(
        self,
        task_input: TaskClassificationInput,
        variant: str = "A",
    ) -> ConsensusResult:
        """
        Classify a task using all judges and compute consensus.

        Args:
            task_input: Input for classification
            variant: "A" (rubric-only) or "B" (with-context)

        Returns:
            ConsensusResult with final classifications and agreement scores
        """
        # Run all judges in parallel
        judge_tasks = [
            classify_task(
                task_input=task_input,
                model_name=judge,
                variant=variant,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                reasoning_effort=self.reasoning_effort,
            )
            for judge in self.judges
        ]

        results = await asyncio.gather(*judge_tasks, return_exceptions=True)

        # Process results
        judge_results: Dict[str, JudgeResult] = {}
        successful_results: List[JudgeResult] = []

        for judge, result in zip(self.judges, results):
            if isinstance(result, Exception):
                judge_results[judge] = JudgeResult(
                    model_name=judge,
                    classification=None,
                    raw_output="",
                    refused=False,
                    error=str(result),
                    latency_ms=0.0,
                )
            else:
                judge_results[judge] = result
                if result.classification and not result.refused and not result.error:
                    successful_results.append(result)

        # If not enough successful results, try fallback judges
        if len(successful_results) < 2 and self.fallback_judges:
            needed = 2 - len(successful_results)
            fallback_to_try = self.fallback_judges[:needed]

            fallback_tasks = [
                classify_task(
                    task_input=task_input,
                    model_name=judge,
                    variant=variant,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    reasoning_effort=self.reasoning_effort,
                )
                for judge in fallback_to_try
            ]

            fallback_results = await asyncio.gather(*fallback_tasks, return_exceptions=True)

            for judge, result in zip(fallback_to_try, fallback_results):
                if isinstance(result, Exception):
                    judge_results[judge] = JudgeResult(
                        model_name=judge,
                        classification=None,
                        raw_output="",
                        refused=False,
                        error=str(result),
                        latency_ms=0.0,
                    )
                else:
                    judge_results[judge] = result
                    if result.classification and not result.refused and not result.error:
                        successful_results.append(result)

        # Compute consensus
        consensus_classifications, agreement_scores = self._compute_consensus(
            successful_results, task_input
        )

        return ConsensusResult(
            task_id=task_input.task_id,
            variant=variant,
            classifications=consensus_classifications,
            judge_results=judge_results,
            agreement_scores=agreement_scores,
            timestamp=datetime.utcnow().isoformat(),
        )

    def _compute_consensus(
        self,
        successful_results: List[JudgeResult],
        task_input: TaskClassificationInput,
    ) -> tuple[List[RubricClassification], List[float]]:
        """
        Compute consensus classifications from multiple judge results.

        Uses majority vote for each rubric criterion.
        """
        if not successful_results:
            # No successful results - return empty
            return [], []

        if len(successful_results) == 1:
            # Only one result - use it directly
            return successful_results[0].classification.classifications, [1.0] * len(
                successful_results[0].classification.classifications
            )

        # Get number of rubric questions
        num_rubrics = len(task_input.rubric)

        consensus_classifications = []
        agreement_scores = []

        for rubric_idx in range(num_rubrics):
            # Collect votes from each judge for this rubric
            votes: List[tuple[HarmDimension, JudgeResult]] = []

            for result in successful_results:
                if result.classification:
                    # Find classification for this rubric index
                    for cls in result.classification.classifications:
                        if cls.rubric_index == rubric_idx:
                            votes.append((cls.primary_harm, result))
                            break

            if not votes:
                # No votes for this rubric - skip
                agreement_scores.append(0.0)
                continue

            # Count votes
            vote_counts = Counter(v[0] for v in votes)
            most_common = vote_counts.most_common(1)[0]
            winning_harm = most_common[0]
            winning_count = most_common[1]

            # Calculate agreement score
            agreement = winning_count / len(votes)
            agreement_scores.append(agreement)

            # Find the best classification with the winning harm dimension
            # Prefer higher confidence
            winning_classification = None
            best_confidence = -1.0

            for harm, result in votes:
                if harm == winning_harm:
                    for cls in result.classification.classifications:
                        if cls.rubric_index == rubric_idx and cls.confidence > best_confidence:
                            winning_classification = cls
                            best_confidence = cls.confidence

            if winning_classification:
                # Update confidence based on agreement
                consensus_cls = RubricClassification(
                    rubric_index=rubric_idx,
                    rubric_question=(
                        task_input.rubric[rubric_idx]
                        if rubric_idx < len(task_input.rubric)
                        else winning_classification.rubric_question
                    ),
                    primary_harm=winning_harm,
                    secondary_harm=winning_classification.secondary_harm,
                    secondary_harm_justification=winning_classification.secondary_harm_justification,
                    confidence=winning_classification.confidence * agreement,
                    reasoning=f"Consensus ({winning_count}/{len(votes)} judges): {winning_classification.reasoning}",
                )
                consensus_classifications.append(consensus_cls)

        return consensus_classifications, agreement_scores


async def classify_task_with_panel(
    task_input: TaskClassificationInput,
    variant: str = "A",
    judges: Optional[List[str]] = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    timeout: float = 120.0,
    reasoning_effort: Optional[str] = None,
) -> ConsensusResult:
    """
    Convenience function to classify a task with the default judge panel.

    Args:
        task_input: Input for classification
        variant: "A" (rubric-only) or "B" (with-context)
        judges: Optional list of judge models
        temperature: Sampling temperature
        max_tokens: Max tokens per judge
        timeout: Timeout per judge request
        reasoning_effort: Reasoning mode - "low", "medium", "high", or None (disabled)

    Returns:
        ConsensusResult with final classifications
    """
    panel = JudgePanel(
        judges=judges,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )
    return await panel.classify_with_consensus(task_input, variant)
