"""
Data loading for direct request refusal check results.

Loads refusal classification JSONL files (from run_refusal_check.py --prompt-type direct)
and produces DirectRequestResult objects for analysis.

DirectRequestResult is designed to work with existing filters (filter_culture_agnostic,
filter_culture_specific, filter_by_variant, etc.) since those filters use generic
attribute access via _get_attr().
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


@dataclass
class DirectRequestResult:
    """A single direct request result with compliance/refusal classification.

    Designed to be compatible with existing filter utilities (filter_culture_agnostic,
    filter_culture_specific, filter_by_variant, etc.) which use _get_attr() on
    task_id, variant_name, and risk_domain attributes.
    """

    task_id: str
    variant_name: str
    model_name: str
    risk_domain: str
    risk_subdomain: str
    prompt: str
    response: str
    is_refusal: bool  # True = model refused, False = model complied
    refusal_reasoning: str = ""

    @property
    def complied(self) -> bool:
        """Whether the model complied with the direct request (did NOT refuse)."""
        return not self.is_refusal

    @property
    def is_valid(self) -> bool:
        """Always valid if we have a classification."""
        return True

    @property
    def is_excluded(self) -> bool:
        """Never excluded (errors are filtered during loading)."""
        return False


def load_direct_request_results(
    refusal_checks_dir: Path,
    prompt_type: str = "direct",
) -> Dict[str, List[DirectRequestResult]]:
    """
    Load direct request refusal check results, grouped by model.

    Scans refusal_{prompt_type}_*.jsonl files in the directory.

    Args:
        refusal_checks_dir: Path to runs/refusal-checks/ directory
        prompt_type: Prompt type to load (default: "direct")

    Returns:
        Dict mapping model_name -> list of DirectRequestResult
    """
    results: Dict[str, List[DirectRequestResult]] = defaultdict(list)

    pattern = f"refusal_{prompt_type}_*.jsonl"
    refusal_files = sorted(refusal_checks_dir.glob(pattern))

    if not refusal_files:
        print(f"  No refusal check files found matching {pattern} in {refusal_checks_dir}")
        return dict(results)

    for refusal_file in tqdm(refusal_files, desc="Loading refusal checks", unit="file"):
        with open(refusal_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Skip records with errors
                if data.get("error"):
                    continue

                # Skip records with no classification
                is_refusal = data.get("is_refusal")
                if is_refusal is None:
                    continue

                model_name = data.get("model_name", "")
                result = DirectRequestResult(
                    task_id=data.get("task_id", ""),
                    variant_name=data.get("variant_name", ""),
                    model_name=model_name,
                    risk_domain=data.get("risk_domain", ""),
                    risk_subdomain=data.get("risk_subdomain", ""),
                    prompt=data.get("prompt", ""),
                    response=data.get("response", ""),
                    is_refusal=is_refusal,
                    refusal_reasoning=data.get("refusal_reasoning", ""),
                )
                results[model_name].append(result)

    return dict(results)


def load_direct_request_results_from_responses(
    responses_dir: Path,
) -> Dict[str, List[DirectRequestResult]]:
    """
    Load direct request data directly from response JSONL files
    (before refusal classification). Useful for inspecting raw responses.

    Treats empty responses as refusals, non-empty as compliance (simple heuristic).
    For proper classification, use load_direct_request_results() with refusal check output.

    Args:
        responses_dir: Path to runs/rok_fortress_responses_sampled/

    Returns:
        Dict mapping model_name -> list of DirectRequestResult
    """
    results: Dict[str, List[DirectRequestResult]] = defaultdict(list)

    for jsonl_file in sorted(responses_dir.glob("responses_*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                direct = data.get("direct")
                if not direct:
                    continue

                # Skip if there was an error
                if direct.get("error"):
                    continue

                response = direct.get("response", "")
                model_name = data.get("model_name", "")

                result = DirectRequestResult(
                    task_id=data.get("task_id", ""),
                    variant_name=data.get("variant_name", ""),
                    model_name=model_name,
                    risk_domain=data.get("risk_domain", ""),
                    risk_subdomain=data.get("risk_subdomain", ""),
                    prompt=direct.get("prompt", ""),
                    response=response,
                    is_refusal=not bool(response.strip()),  # Simple heuristic
                )
                results[model_name].append(result)

    return dict(results)
