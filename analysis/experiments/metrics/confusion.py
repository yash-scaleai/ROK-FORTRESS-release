"""
Confusion matrix utilities for judge calibration (Experiment 0).

Computes:
- Y/N confusion matrices (2x2) for comparing judge vs human verdicts
- Dimension confusion matrices (7x7) for harm dimension classification
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

from rok_fortress_eval.harm_classifier.data_models import HarmDimension, parse_harm_dimension


@dataclass
class ConfusionMatrix:
    """2x2 confusion matrix for binary classification."""
    
    tp: int = 0  # True Positive (both YES)
    fp: int = 0  # False Positive (judge YES, human NO)
    fn: int = 0  # False Negative (judge NO, human YES)
    tn: int = 0  # True Negative (both NO)
    
    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn
    
    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.tp + self.tn) / self.total
    
    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0
    
    @property
    def f1(self) -> float:
        """F1 = 2 * precision * recall / (precision + recall)"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def specificity(self) -> float:
        """Specificity = TN / (TN + FP)"""
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to 2x2 numpy array [[TN, FP], [FN, TP]]."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])


@dataclass
class DimensionConfusionMatrix:
    """7x7 confusion matrix for harm dimension classification."""
    
    # Matrix[true_dim][pred_dim] = count
    matrix: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    
    # All dimension names
    dimensions: List[str] = field(default_factory=lambda: [
        "WHAT", "HOW_1", "HOW_2", "HOW_3", "WHY", "WHO_WHERE", "WHEN"
    ])
    
    def add(self, true_dim: str, pred_dim: str) -> None:
        """Add a single prediction to the matrix."""
        true_key = self._normalize_dim(true_dim)
        pred_key = self._normalize_dim(pred_dim)
        self.matrix[true_key][pred_key] += 1
    
    def _normalize_dim(self, dim: str) -> str:
        """Normalize dimension string to key."""
        parsed = parse_harm_dimension(dim)
        if parsed is None:
            return "UNKNOWN"
        
        dim_map = {
            HarmDimension.WHAT: "WHAT",
            HarmDimension.HOW_1: "HOW_1",
            HarmDimension.HOW_2: "HOW_2",
            HarmDimension.HOW_3: "HOW_3",
            HarmDimension.WHY: "WHY",
            HarmDimension.WHO_WHERE: "WHO_WHERE",
            HarmDimension.WHEN: "WHEN",
        }
        return dim_map.get(parsed, "UNKNOWN")
    
    def to_array(self) -> np.ndarray:
        """Convert to 7x7 numpy array."""
        arr = np.zeros((len(self.dimensions), len(self.dimensions)), dtype=int)
        for i, true_dim in enumerate(self.dimensions):
            for j, pred_dim in enumerate(self.dimensions):
                arr[i, j] = self.matrix[true_dim][pred_dim]
        return arr
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy (diagonal sum / total)."""
        arr = self.to_array()
        total = arr.sum()
        if total == 0:
            return 0.0
        return np.trace(arr) / total
    
    def per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute per-class precision, recall, F1."""
        arr = self.to_array()
        metrics = {}
        
        for i, dim in enumerate(self.dimensions):
            tp = arr[i, i]
            fp = arr[:, i].sum() - tp  # Column sum minus diagonal
            fn = arr[i, :].sum() - tp  # Row sum minus diagonal
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[dim] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        
        return metrics


def build_yn_confusion_matrix(
    judge_verdicts: List[bool],
    human_verdicts: List[bool],
) -> ConfusionMatrix:
    """
    Build a 2x2 Y/N confusion matrix.
    
    Args:
        judge_verdicts: List of judge verdicts (True=YES, False=NO)
        human_verdicts: List of human verdicts (ground truth)
    
    Returns:
        ConfusionMatrix
    """
    cm = ConfusionMatrix()
    
    for judge, human in zip(judge_verdicts, human_verdicts):
        if judge and human:
            cm.tp += 1
        elif judge and not human:
            cm.fp += 1
        elif not judge and human:
            cm.fn += 1
        else:
            cm.tn += 1
    
    return cm


def build_dimension_confusion_matrix(
    predicted_dims: List[str],
    actual_dims: List[str],
) -> DimensionConfusionMatrix:
    """
    Build a 7x7 dimension confusion matrix.
    
    Args:
        predicted_dims: List of predicted harm dimensions
        actual_dims: List of actual (human-labeled) dimensions
    
    Returns:
        DimensionConfusionMatrix
    """
    dcm = DimensionConfusionMatrix()
    
    for pred, actual in zip(predicted_dims, actual_dims):
        dcm.add(actual, pred)
    
    return dcm


def compute_cohens_kappa(cm: ConfusionMatrix) -> float:
    """
    Compute Cohen's Kappa for inter-rater reliability.
    
    Kappa = (Po - Pe) / (1 - Pe)
    where Po = observed agreement, Pe = expected agreement by chance
    
    Args:
        cm: ConfusionMatrix
    
    Returns:
        Cohen's Kappa (-1 to 1, where 1 is perfect agreement)
    """
    if cm.total == 0:
        return 0.0
    
    # Observed agreement
    po = (cm.tp + cm.tn) / cm.total
    
    # Expected agreement by chance
    p_judge_yes = (cm.tp + cm.fp) / cm.total
    p_human_yes = (cm.tp + cm.fn) / cm.total
    p_judge_no = (cm.tn + cm.fn) / cm.total
    p_human_no = (cm.tn + cm.fp) / cm.total
    
    pe = p_judge_yes * p_human_yes + p_judge_no * p_human_no
    
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    
    return (po - pe) / (1 - pe)


def compute_harm_severity_delta(
    predicted_dims: List[str],
    actual_dims: List[str],
    domain: str,
) -> float:
    """
    Compute Harm Severity Delta.
    
    Measures the difference in harm weights between predicted and actual dimensions.
    
    Delta = mean(weight(predicted) - weight(actual))
    
    Positive delta means judge overestimates severity.
    Negative delta means judge underestimates severity.
    
    Args:
        predicted_dims: List of predicted harm dimensions
        actual_dims: List of actual (human-labeled) dimensions
        domain: Risk domain for weight lookup
    
    Returns:
        Harm severity delta
    """
    from .harm_scores import get_weight
    
    deltas = []
    for pred, actual in zip(predicted_dims, actual_dims):
        pred_weight = get_weight(domain, pred)
        actual_weight = get_weight(domain, actual)
        deltas.append(pred_weight - actual_weight)
    
    return float(np.mean(deltas)) if deltas else 0.0


def aggregate_confusion_matrices(matrices: List[ConfusionMatrix]) -> ConfusionMatrix:
    """
    Aggregate multiple confusion matrices by summing counts.
    
    Args:
        matrices: List of ConfusionMatrix
    
    Returns:
        Aggregated ConfusionMatrix
    """
    result = ConfusionMatrix()
    for cm in matrices:
        result.tp += cm.tp
        result.fp += cm.fp
        result.fn += cm.fn
        result.tn += cm.tn
    return result


def compute_confusion_by_variant(
    judge_results: Dict[str, List[bool]],
    human_results: Dict[str, List[bool]],
) -> Dict[str, ConfusionMatrix]:
    """
    Compute confusion matrices per variant.
    
    Args:
        judge_results: Dict mapping variant_name -> list of judge verdicts
        human_results: Dict mapping variant_name -> list of human verdicts
    
    Returns:
        Dict mapping variant_name -> ConfusionMatrix
    """
    matrices = {}
    for variant in judge_results:
        if variant in human_results:
            matrices[variant] = build_yn_confusion_matrix(
                judge_results[variant],
                human_results[variant],
            )
    return matrices
