"""
Filtering utilities for CA/CS tasks and domain stratification.

Culture-Agnostic (CA) tasks: Have 2 variants (original_en, translated_kr)
Culture-Specific (CS) tasks: Have 4 variants (all four)

Task type is inferred from the variants present for each task_id.
"""

from typing import Any, Dict, List, Literal, Optional, Set
from collections import defaultdict

from .variants import (
    CA_VARIANTS,
    CS_VARIANTS,
    CULTURAL_ADAPTED_EN,
    TRANSCREATED_KR,
)


def _get_attr(obj: Any, name: str, default: str = "") -> str:
    """Get attribute from object or dict, with default value."""
    if hasattr(obj, name):
        return getattr(obj, name)
    elif isinstance(obj, dict):
        return obj.get(name, default)
    return default


# Risk domain constants (matching the data)
DOMAIN_CBRNE = "Chemical, Biological, Radiological, Nuclear, and Explosive (CBRNE)"
DOMAIN_POLITICAL = "Political Violence & Terrorism"
DOMAIN_CRIMINAL = "Criminal & Financial Illicit Activities"
DOMAIN_INFORMATION = "Information Leakage"

# Short names for convenience
DOMAINS = {
    "CBRNE": DOMAIN_CBRNE,
    "Political": DOMAIN_POLITICAL,
    "Criminal": DOMAIN_CRIMINAL,
    "Information": DOMAIN_INFORMATION,
}

# Reverse mapping
DOMAIN_SHORT_NAMES = {v: k for k, v in DOMAINS.items()}


TaskType = Literal["CA", "CS", "UNKNOWN"]


def get_task_type_from_variants(variant_names: Set[str]) -> TaskType:
    """
    Determine task type based on which variants are present.
    
    CA tasks: Have only original_en and translated_kr (2 variants)
    CS tasks: Have all 4 variants including cultural_adapted_en or transcreated_kr
    
    Args:
        variant_names: Set of variant names present for a task
    
    Returns:
        "CA", "CS", or "UNKNOWN"
    """
    # If any CS-only variants are present, it's a CS task
    if CULTURAL_ADAPTED_EN in variant_names or TRANSCREATED_KR in variant_names:
        return "CS"
    
    # If only CA variants are present (and we have at least one), it's CA
    if variant_names and variant_names.issubset(CA_VARIANTS):
        return "CA"
    
    return "UNKNOWN"


def get_task_type(results: List, task_id: str) -> TaskType:
    """
    Determine task type for a specific task based on available variants.
    
    Args:
        results: List of results (ConsolidatedResult or dicts)
        task_id: The task ID to check
    
    Returns:
        "CA", "CS", or "UNKNOWN"
    """
    variant_names = set()
    for r in results:
        r_task_id = _get_attr(r, "task_id")
        if r_task_id == task_id:
            variant_name = _get_attr(r, "variant_name")
            variant_names.add(variant_name)
    
    return get_task_type_from_variants(variant_names)


def classify_all_tasks(results: List) -> Dict[str, TaskType]:
    """
    Classify all tasks as CA or CS based on their variants.
    
    Args:
        results: List of results (ConsolidatedResult or dicts)
    
    Returns:
        Dict mapping task_id -> TaskType
    """
    # Group variants by task_id
    task_variants: Dict[str, Set[str]] = defaultdict(set)
    for r in results:
        # Handle both dataclass objects and dicts
        task_id = _get_attr(r, "task_id")
        variant_name = _get_attr(r, "variant_name")
        task_variants[task_id].add(variant_name)
    
    # Classify each task
    return {
        task_id: get_task_type_from_variants(variants)
        for task_id, variants in task_variants.items()
    }


def filter_culture_agnostic(results: List) -> List:
    """
    Filter for Culture-Agnostic (CA) tasks only.
    
    Args:
        results: List of results
    
    Returns:
        Filtered list containing only results from CA tasks
    """
    task_types = classify_all_tasks(results)
    ca_task_ids = {tid for tid, ttype in task_types.items() if ttype == "CA"}
    
    return [
        r for r in results
        if (_get_attr(r, "task_id") in ca_task_ids)
    ]


def filter_culture_specific(results: List) -> List:
    """
    Filter for Culture-Specific (CS) tasks only.
    
    Args:
        results: List of results
    
    Returns:
        Filtered list containing only results from CS tasks
    """
    task_types = classify_all_tasks(results)
    cs_task_ids = {tid for tid, ttype in task_types.items() if ttype == "CS"}
    
    return [
        r for r in results
        if (_get_attr(r, "task_id") in cs_task_ids)
    ]


def filter_by_variant(results: List, variant_name: str) -> List:
    """
    Filter for a specific variant.
    
    Args:
        results: List of results
        variant_name: Variant to filter for
    
    Returns:
        Filtered list
    """
    return [
        r for r in results
        if (_get_attr(r, "variant_name") == variant_name)
    ]


def filter_by_domain(results: List, domain: str) -> List:
    """
    Filter by risk domain.
    
    Args:
        results: List of results
        domain: Domain name (short or full name)
    
    Returns:
        Filtered list
    """
    # Resolve short name to full name if needed
    full_domain = DOMAINS.get(domain, domain)
    
    return [
        r for r in results
        if (_get_attr(r, "risk_domain") == full_domain)
    ]


def filter_by_domains(results: List, domains: List[str]) -> List:
    """
    Filter by multiple risk domains.
    
    Args:
        results: List of results
        domains: List of domain names (short or full)
    
    Returns:
        Filtered list
    """
    # Resolve all domain names
    full_domains = {DOMAINS.get(d, d) for d in domains}
    
    return [
        r for r in results
        if (_get_attr(r, "risk_domain") in full_domains)
    ]


def get_domain_distribution(results: List) -> Dict[str, int]:
    """
    Count results per domain.
    
    Args:
        results: List of results
    
    Returns:
        Dict mapping domain short name -> count
    """
    counts: Dict[str, int] = defaultdict(int)
    for r in results:
        domain = _get_attr(r, "risk_domain")
        short_name = DOMAIN_SHORT_NAMES.get(domain, domain)
        counts[short_name] += 1
    return dict(counts)


def get_task_type_distribution(results: List) -> Dict[str, int]:
    """
    Count unique tasks per task type.
    
    Args:
        results: List of results
    
    Returns:
        Dict with keys "CA", "CS", "UNKNOWN" -> task count
    """
    task_types = classify_all_tasks(results)
    counts: Dict[str, int] = defaultdict(int)
    for task_type in task_types.values():
        counts[task_type] += 1
    return dict(counts)
