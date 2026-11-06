"""
Meta-evaluation module for comparing LLM classifications with human labels.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .data_models import HarmDimension, ConsensusResult, parse_harm_dimension


@dataclass
class MetaEvalResult:
    """Results from comparing LLM classifications to human labels."""
    # Overall metrics
    exact_match_rate: float  # Per-rubric exact match
    task_level_accuracy: float  # All rubrics correct
    total_rubrics: int
    total_tasks: int

    # Per-dimension metrics
    per_dimension_precision: Dict[str, float] = field(default_factory=dict)
    per_dimension_recall: Dict[str, float] = field(default_factory=dict)
    per_dimension_f1: Dict[str, float] = field(default_factory=dict)

    # Confusion matrix: predicted -> actual -> count
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Low-confidence mismatches for analysis
    low_confidence_mismatches: List[dict] = field(default_factory=list)

    # Agreement breakdown
    agreement_by_confidence: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "exact_match_rate": self.exact_match_rate,
            "task_level_accuracy": self.task_level_accuracy,
            "total_rubrics": self.total_rubrics,
            "total_tasks": self.total_tasks,
            "per_dimension_precision": self.per_dimension_precision,
            "per_dimension_recall": self.per_dimension_recall,
            "per_dimension_f1": self.per_dimension_f1,
            "confusion_matrix": self.confusion_matrix,
            "low_confidence_mismatches": self.low_confidence_mismatches[:50],  # Limit size
            "agreement_by_confidence": self.agreement_by_confidence,
        }


def load_human_labels(
    meta_eval_path: str,
    task_id_column: str = "TASK_ID",
    harm_column: str = "ORIGINAL_RUBRIC_HARM",
) -> Dict[str, List[HarmDimension]]:
    """
    Load human harm dimension labels from meta-eval file.

    Args:
        meta_eval_path: Path to TSV/CSV file with human labels
        task_id_column: Column name for task IDs
        harm_column: Column name for harm dimension labels

    Returns:
        Dict mapping task_id to list of HarmDimension (one per rubric)
    """
    # Load file
    path = Path(meta_eval_path)
    if path.suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    human_labels = {}

    for _, row in df.iterrows():
        task_id = str(row.get(task_id_column, "")).strip()
        if not task_id:
            continue

        harm_str = row.get(harm_column, "")
        if pd.isna(harm_str) or not harm_str:
            continue

        # Parse harm labels (JSON array format)
        try:
            if isinstance(harm_str, str):
                harm_list = json.loads(harm_str)
            else:
                harm_list = harm_str

            dimensions = []
            for harm in harm_list:
                dim = parse_harm_dimension(harm)
                if dim:
                    dimensions.append(dim)

            if dimensions:
                human_labels[task_id] = dimensions

        except (json.JSONDecodeError, TypeError):
            continue

    return human_labels


def load_llm_classifications(
    classifications_path: str,
) -> Dict[str, ConsensusResult]:
    """
    Load LLM classifications from JSONL output file.

    Args:
        classifications_path: Path to JSONL file with ConsensusResult records

    Returns:
        Dict mapping task_id to ConsensusResult
    """
    llm_results = {}

    with open(classifications_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                result = ConsensusResult(
                    task_id=data["task_id"],
                    variant=data["variant"],
                    classifications=[],  # Will parse below
                    judge_results={},  # Not needed for comparison
                    agreement_scores=data.get("agreement_scores", []),
                    timestamp=data.get("timestamp") or "",
                )

                # Parse classifications
                for cls_data in data.get("classifications", []):
                    from .data_models import RubricClassification

                    primary = parse_harm_dimension(cls_data.get("primary_harm", ""))
                    if not primary:
                        continue

                    secondary = None
                    if cls_data.get("secondary_harm"):
                        secondary = parse_harm_dimension(cls_data["secondary_harm"])

                    cls = RubricClassification(
                        rubric_index=cls_data.get("rubric_index", 0),
                        rubric_question=cls_data.get("rubric_question", ""),
                        primary_harm=primary,
                        secondary_harm=secondary,
                        secondary_harm_justification=cls_data.get("secondary_harm_justification"),
                        confidence=cls_data.get("confidence", 0.5),
                        reasoning=cls_data.get("reasoning", ""),
                    )
                    result.classifications.append(cls)

                llm_results[data["task_id"]] = result

            except (json.JSONDecodeError, KeyError):
                continue

    return llm_results


def compare_classifications(
    human_labels: Dict[str, List[HarmDimension]],
    llm_results: Dict[str, ConsensusResult],
    confidence_threshold: float = 0.5,
) -> MetaEvalResult:
    """
    Compare LLM classifications to human labels.

    Args:
        human_labels: Dict of task_id -> list of human-assigned HarmDimensions
        llm_results: Dict of task_id -> ConsensusResult from LLM
        confidence_threshold: Threshold below which to flag as low-confidence

    Returns:
        MetaEvalResult with comparison metrics
    """
    # Track matches
    total_rubrics = 0
    exact_matches = 0
    task_matches = 0
    total_tasks = 0

    # Per-dimension tracking
    true_positives = Counter()
    false_positives = Counter()
    false_negatives = Counter()

    # Confusion matrix
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Low confidence mismatches
    low_confidence_mismatches = []

    # Confidence buckets
    confidence_buckets = {
        "0.0-0.5": {"correct": 0, "total": 0},
        "0.5-0.7": {"correct": 0, "total": 0},
        "0.7-0.9": {"correct": 0, "total": 0},
        "0.9-1.0": {"correct": 0, "total": 0},
    }

    # Find overlapping tasks
    overlapping_tasks = set(human_labels.keys()) & set(llm_results.keys())

    for task_id in overlapping_tasks:
        human = human_labels[task_id]
        llm = llm_results[task_id]

        total_tasks += 1
        task_all_correct = True

        # Compare each rubric
        for i, human_dim in enumerate(human):
            total_rubrics += 1

            # Find LLM classification for this rubric index
            llm_cls = None
            for cls in llm.classifications:
                if cls.rubric_index == i:
                    llm_cls = cls
                    break

            if not llm_cls:
                # LLM didn't classify this rubric
                task_all_correct = False
                false_negatives[human_dim.value] += 1
                continue

            predicted = llm_cls.primary_harm
            actual = human_dim

            # Update confusion matrix
            confusion[predicted.value][actual.value] += 1

            # Track confidence bucket
            conf = llm_cls.confidence
            if conf < 0.5:
                bucket = "0.0-0.5"
            elif conf < 0.7:
                bucket = "0.5-0.7"
            elif conf < 0.9:
                bucket = "0.7-0.9"
            else:
                bucket = "0.9-1.0"
            confidence_buckets[bucket]["total"] += 1

            # Check match
            if predicted == actual:
                exact_matches += 1
                true_positives[actual.value] += 1
                confidence_buckets[bucket]["correct"] += 1
            else:
                task_all_correct = False
                false_positives[predicted.value] += 1
                false_negatives[actual.value] += 1

                # Track low confidence mismatch
                if conf < confidence_threshold:
                    low_confidence_mismatches.append({
                        "task_id": task_id,
                        "rubric_index": i,
                        "rubric_question": llm_cls.rubric_question,
                        "predicted": predicted.value,
                        "actual": actual.value,
                        "confidence": conf,
                        "reasoning": llm_cls.reasoning,
                    })

        if task_all_correct:
            task_matches += 1

    # Calculate metrics
    exact_match_rate = exact_matches / total_rubrics if total_rubrics > 0 else 0.0
    task_level_accuracy = task_matches / total_tasks if total_tasks > 0 else 0.0

    # Per-dimension precision/recall/F1
    per_dim_precision = {}
    per_dim_recall = {}
    per_dim_f1 = {}

    for dim in HarmDimension:
        tp = true_positives[dim.value]
        fp = false_positives[dim.value]
        fn = false_negatives[dim.value]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_dim_precision[dim.value] = round(precision, 4)
        per_dim_recall[dim.value] = round(recall, 4)
        per_dim_f1[dim.value] = round(f1, 4)

    # Agreement by confidence
    agreement_by_conf = {}
    for bucket, stats in confidence_buckets.items():
        if stats["total"] > 0:
            agreement_by_conf[bucket] = round(stats["correct"] / stats["total"], 4)
        else:
            agreement_by_conf[bucket] = 0.0

    return MetaEvalResult(
        exact_match_rate=round(exact_match_rate, 4),
        task_level_accuracy=round(task_level_accuracy, 4),
        total_rubrics=total_rubrics,
        total_tasks=total_tasks,
        per_dimension_precision=per_dim_precision,
        per_dimension_recall=per_dim_recall,
        per_dimension_f1=per_dim_f1,
        confusion_matrix=dict(confusion),
        low_confidence_mismatches=low_confidence_mismatches,
        agreement_by_confidence=agreement_by_conf,
    )


def run_meta_eval(
    llm_classifications_path: str,
    human_labels_path: str,
    task_id_column: str = "TASK_ID",
    harm_column: str = "ORIGINAL_RUBRIC_HARM",
) -> MetaEvalResult:
    """
    Run meta-evaluation comparing LLM classifications to human labels.

    Args:
        llm_classifications_path: Path to LLM classifications JSONL
        human_labels_path: Path to human labels TSV/CSV
        task_id_column: Column name for task IDs in human labels file
        harm_column: Column name for harm dimensions in human labels file

    Returns:
        MetaEvalResult with comparison metrics
    """
    human_labels = load_human_labels(human_labels_path, task_id_column, harm_column)
    llm_results = load_llm_classifications(llm_classifications_path)

    return compare_classifications(human_labels, llm_results)


def print_meta_eval_summary(result: MetaEvalResult) -> str:
    """Generate a human-readable summary of meta-eval results."""
    lines = [
        "=" * 60,
        "META-EVALUATION SUMMARY",
        "=" * 60,
        "",
        f"Total Tasks Evaluated: {result.total_tasks}",
        f"Total Rubric Items: {result.total_rubrics}",
        "",
        "OVERALL METRICS",
        "-" * 40,
        f"Exact Match Rate (per-rubric): {result.exact_match_rate:.2%}",
        f"Task-Level Accuracy (all rubrics correct): {result.task_level_accuracy:.2%}",
        "",
        "PER-DIMENSION F1 SCORES",
        "-" * 40,
    ]

    for dim in HarmDimension:
        f1 = result.per_dimension_f1.get(dim.value, 0.0)
        lines.append(f"  {dim.value}: {f1:.2%}")

    lines.extend([
        "",
        "AGREEMENT BY CONFIDENCE",
        "-" * 40,
    ])

    for bucket, rate in sorted(result.agreement_by_confidence.items()):
        lines.append(f"  {bucket}: {rate:.2%}")

    lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(lines)
