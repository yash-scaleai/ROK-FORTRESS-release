"""
Variant mapping for the Transcreation Matrix.

Maps variant names to experimental codes based on task type (CA vs CS).

Culture-Agnostic (CA) tasks have 2 variants:
- V(En): original_en - English baseline
- V(Ko): translated_kr - Korean translation

Culture-Specific (CS) tasks have 4 variants:
- V(En,US): original_en - US context / English
- V(En,KR): cultural_adapted_en - KR context / English
- V(Ko,US): translated_kr - US context / Korean
- V(Ko,KR): transcreated_kr - KR context / Korean
"""

from typing import Any, Dict, List, Literal, Optional, Tuple
from collections import defaultdict


def _get_attr(obj: Any, name: str, default: str = "") -> str:
    """Get attribute from object or dict, with default value."""
    if hasattr(obj, name):
        return getattr(obj, name)
    elif isinstance(obj, dict):
        return obj.get(name, default)
    return default


# Variant name constants
ORIGINAL_EN = "original_en"
TRANSLATED_KR = "translated_kr"
CULTURAL_ADAPTED_EN = "cultural_adapted_en"
TRANSCREATED_KR = "transcreated_kr"

# All variant names
ALL_VARIANTS = [ORIGINAL_EN, TRANSLATED_KR, CULTURAL_ADAPTED_EN, TRANSCREATED_KR]

# CA variants (2)
CA_VARIANTS = {ORIGINAL_EN, TRANSLATED_KR}

# CS variants (4)
CS_VARIANTS = {ORIGINAL_EN, TRANSLATED_KR, CULTURAL_ADAPTED_EN, TRANSCREATED_KR}

# Experimental variant codes
VariantCode = Literal["V_En", "V_Ko", "V_En_US", "V_En_KR", "V_Ko_US", "V_Ko_KR"]

# Mapping from (task_type, variant_name) to experimental code
VARIANT_CODES: Dict[Tuple[str, str], str] = {
    # Culture-Agnostic tasks
    ("CA", ORIGINAL_EN): "V_En",
    ("CA", TRANSLATED_KR): "V_Ko",
    # Culture-Specific tasks
    ("CS", ORIGINAL_EN): "V_En_US",
    ("CS", TRANSLATED_KR): "V_Ko_US",
    ("CS", CULTURAL_ADAPTED_EN): "V_En_KR",
    ("CS", TRANSCREATED_KR): "V_Ko_KR",
}

# Reverse mapping: code -> (task_type, variant_name)
CODE_TO_VARIANT: Dict[str, Tuple[str, str]] = {v: k for k, v in VARIANT_CODES.items()}


def get_variant_code(task_type: str, variant_name: str) -> Optional[str]:
    """
    Get the experimental variant code for a task type and variant name.
    
    Args:
        task_type: "CA" for Culture-Agnostic, "CS" for Culture-Specific
        variant_name: One of original_en, translated_kr, cultural_adapted_en, transcreated_kr
    
    Returns:
        Variant code (e.g., "V_En", "V_Ko_KR") or None if invalid
    """
    return VARIANT_CODES.get((task_type, variant_name))


def get_variant_name(code: str) -> Optional[Tuple[str, str]]:
    """
    Get the task type and variant name for an experimental code.
    
    Args:
        code: Variant code (e.g., "V_En", "V_Ko_KR")
    
    Returns:
        Tuple of (task_type, variant_name) or None if invalid
    """
    return CODE_TO_VARIANT.get(code)


def get_language(variant_name: str) -> str:
    """Get the language of a variant (EN or KO)."""
    if variant_name in {ORIGINAL_EN, CULTURAL_ADAPTED_EN}:
        return "EN"
    elif variant_name in {TRANSLATED_KR, TRANSCREATED_KR}:
        return "KO"
    return "UNKNOWN"


def get_context(variant_name: str, task_type: str) -> str:
    """
    Get the cultural context of a variant.
    
    For CA tasks: Returns "NEUTRAL" (no specific context)
    For CS tasks: Returns "US" or "KR"
    """
    if task_type == "CA":
        return "NEUTRAL"
    
    if variant_name in {ORIGINAL_EN, TRANSLATED_KR}:
        return "US"
    elif variant_name in {CULTURAL_ADAPTED_EN, TRANSCREATED_KR}:
        return "KR"
    return "UNKNOWN"


def group_by_task_id(results: List) -> Dict[str, Dict[str, any]]:
    """
    Group results by task_id, with variant_name as secondary key.
    
    Args:
        results: List of ConsolidatedResult or similar objects with task_id and variant_name
    
    Returns:
        Dict mapping task_id -> {variant_name -> result}
    """
    grouped = defaultdict(dict)
    for result in results:
        task_id = _get_attr(result, "task_id")
        variant_name = _get_attr(result, "variant_name")
        grouped[task_id][variant_name] = result
    return dict(grouped)


def get_paired_variants_ca() -> List[Tuple[str, str]]:
    """
    Get variant pairs for CA task comparisons.
    
    Returns pairs for Linguistic Drop calculation:
    - (V(En), V(Ko)): original_en vs translated_kr
    """
    return [(ORIGINAL_EN, TRANSLATED_KR)]


def get_paired_variants_cs() -> Dict[str, List[Tuple[str, str]]]:
    """
    Get variant pairs for CS task comparisons.
    
    Returns dict of comparison types to variant pairs:
    - contextual_en: (V(En,US), V(En,KR)) for Contextual Drop in English
    - contextual_ko: (V(Ko,US), V(Ko,KR)) for Contextual Drop in Korean
    - linguistic_us: (V(En,US), V(Ko,US)) for Linguistic comparison in US context
    - linguistic_kr: (V(En,KR), V(Ko,KR)) for Linguistic comparison in KR context
    """
    return {
        "contextual_en": [(ORIGINAL_EN, CULTURAL_ADAPTED_EN)],
        "contextual_ko": [(TRANSLATED_KR, TRANSCREATED_KR)],
        "linguistic_us": [(ORIGINAL_EN, TRANSLATED_KR)],
        "linguistic_kr": [(CULTURAL_ADAPTED_EN, TRANSCREATED_KR)],
    }


def get_comparison_pairs(task_type: str, comparison_type: str) -> List[Tuple[str, str]]:
    """
    Get variant pairs for a specific comparison type.
    
    Args:
        task_type: "CA" or "CS"
        comparison_type: Type of comparison
            - For CA: "linguistic" (only option)
            - For CS: "contextual_en", "contextual_ko", "linguistic_us", "linguistic_kr"
    
    Returns:
        List of (source_variant, target_variant) tuples
    """
    if task_type == "CA":
        return get_paired_variants_ca()
    else:
        pairs = get_paired_variants_cs()
        return pairs.get(comparison_type, [])
