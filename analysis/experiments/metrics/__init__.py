"""Metrics calculation utilities."""

from .harm_scores import (
    compute_harm_score,
    compute_ars,
    DOMAIN_WEIGHTS,
    HarmScoreResult,
)
from .alignment_metrics import (
    compute_linguistic_drop,
    compute_contextual_drop,
    compute_averaged_contextual_drop,
    DropResult,
)
from .compliance import (
    compute_compliance_rate,
    compute_compliance_by_variant,
    compute_compliance_by_domain,
    ComplianceResult,
)
# Note: For linguistic/contextual drop on compliance data, use compute_drop()
# from alignment_metrics with per-task compliance scores (1.0/0.0).

__all__ = [
    # harm_scores
    "compute_harm_score",
    "compute_ars",
    "DOMAIN_WEIGHTS",
    "HarmScoreResult",
    # alignment_metrics
    "compute_linguistic_drop",
    "compute_contextual_drop",
    "compute_averaged_contextual_drop",
    "DropResult",
    # compliance
    "compute_compliance_rate",
    "compute_compliance_by_variant",
    "compute_compliance_by_domain",
    "ComplianceResult",
]
