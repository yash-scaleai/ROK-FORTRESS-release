"""
Deduplicate model variants: prefer reasoning over non-reasoning.

Given a list of model filenames like:
  results_anthropic_claude_opus_4_5.jsonl
  results_anthropic_claude_opus_4_5_reasoning_medium.jsonl

Keep only the reasoning variant. If no reasoning variant exists (e.g. results_hyperclovax.jsonl),
keep the non-reasoning one.
"""

import re
from pathlib import Path
from typing import Dict, List, Set


def extract_judge_name(filename: str) -> str:
    """Extract judge name from results filename, stripping prefix and extension."""
    name = filename
    if name.startswith("results_"):
        name = name[len("results_"):]
    # Remove extensions like .jsonl, .old.jsonl, .run1.jsonl
    name = re.sub(r"\.(old\d?|run\d+|failed\d?)?\.(jsonl|json)$", "", name)
    if name.endswith(".jsonl"):
        name = name[:-6]
    if name.endswith(".json"):
        name = name[:-5]
    return name


def get_base_model_name(judge_name: str) -> str:
    """Strip reasoning suffix to get the base model identity."""
    return re.sub(r"_reasoning_medium$", "", judge_name)


def deduplicate_judge_files(results_dir: Path, pattern: str = "results_*.jsonl") -> List[Path]:
    """
    Return deduplicated list of result files, preferring reasoning variants.

    For each base model, if a reasoning_medium file exists, use it.
    Otherwise use the base (non-reasoning) file.
    Skips files with .old, .run1, .failed suffixes.
    """
    candidates: Dict[str, Path] = {}
    for f in sorted(results_dir.glob(pattern)):
        name = f.name
        if re.search(r"\.(old\d?|run\d+|failed\d?)\.", name):
            continue
        judge_name = extract_judge_name(name)
        candidates[judge_name] = f

    base_to_variants: Dict[str, Dict[str, Path]] = {}
    for judge_name, path in candidates.items():
        base = get_base_model_name(judge_name)
        if base not in base_to_variants:
            base_to_variants[base] = {}
        variant_type = "reasoning" if "reasoning_medium" in judge_name else "base"
        base_to_variants[base][variant_type] = path

    selected: List[Path] = []
    for base, variants in sorted(base_to_variants.items()):
        if "reasoning" in variants:
            selected.append(variants["reasoning"])
        elif "base" in variants:
            selected.append(variants["base"])

    return selected


def deduplicate_harm_classifier_files(results_dir: Path) -> List[Path]:
    """
    Deduplicate harm classifier result files.

    Prefer reasoning variants, and for models with multiple prompt variants (A, B, C),
    keep only variant A (rubric-only, the standard comparison baseline).
    Skips .old, .run1, .failed files.
    """
    candidates: Dict[str, Path] = {}
    for f in sorted(results_dir.glob("results_*.jsonl")):
        name = f.name
        if re.search(r"\.(old\d?|run\d+|failed\d?)\.", name):
            continue
        judge_name = extract_judge_name(name)
        candidates[judge_name] = f

    base_to_variants: Dict[str, Dict[str, Path]] = {}
    for judge_name, path in candidates.items():
        variant_match = re.search(r"_variant_([ABC])$", judge_name)
        variant_letter = variant_match.group(1) if variant_match else "A"
        base_no_variant = re.sub(r"_variant_[ABC]$", "", judge_name)
        has_reasoning = "reasoning_medium" in base_no_variant
        base_model = re.sub(r"_reasoning_medium", "", base_no_variant)

        if base_model not in base_to_variants:
            base_to_variants[base_model] = {}

        priority_key = f"{'reasoning' if has_reasoning else 'base'}_{variant_letter}"
        base_to_variants[base_model][priority_key] = path

    selected: List[Path] = []
    for base_model, variants in sorted(base_to_variants.items()):
        for key in ["reasoning_A", "base_A", "reasoning_B", "base_B", "reasoning_C", "base_C"]:
            if key in variants:
                selected.append(variants[key])
                break

    return selected
