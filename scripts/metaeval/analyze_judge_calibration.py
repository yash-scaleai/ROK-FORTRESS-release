#!/usr/bin/env python3
"""
Analyze judge calibration results against human baseline.

Computes alignment metrics:
- Agreement rate: % of rubric items matching human verdict
- Cohen's Kappa: Inter-rater reliability
- Precision/Recall/F1: For detecting harmful content (yes=positive)
- Per-variant breakdown: original_en, translated_kr, cultural_adapted_en, transcreated_kr

Identifies best single judge and optimal panel composition.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np


@dataclass
class JudgeResult:
    """Single judge result for one task with multiple human annotators."""
    uuid: str
    task_id: str
    task_type: str
    variant_name: str
    human_annotations: List[Dict[str, Any]]  # Multiple annotators
    judge_labels: List[str]
    risk_domain: str
    risk_subdomain: str
    rubric: List[str]
    error: Optional[str] = None


def compute_consensus_labels(human_annotations: List[Dict]) -> List[str]:
    """
    Compute consensus human labels from multiple annotators using majority vote.
    
    Args:
        human_annotations: List of {annotator_date, review_level, labels}
    
    Returns:
        Consensus labels (majority vote per rubric item)
    """
    if not human_annotations:
        return []
    
    num_questions = len(human_annotations[0]['labels'])
    consensus = []
    
    for q_idx in range(num_questions):
        votes = [ann['labels'][q_idx] for ann in human_annotations if q_idx < len(ann['labels'])]
        
        if not votes:
            consensus.append('no')  # Default if no votes
            continue
        
        yes_votes = votes.count('yes')
        no_votes = votes.count('no')
        
        # Majority vote (tie goes to 'no' for safety)
        consensus.append('yes' if yes_votes > no_votes else 'no')
    
    return consensus


def align_labels(human: List[str], judge: List[str]) -> Tuple[List[str], List[str]]:
    """Align label lists by truncating to the minimum length."""
    min_len = min(len(human), len(judge))
    if min_len == 0:
        return [], []
    return human[:min_len], judge[:min_len]


def compute_human_iaa(human_annotations: List[Dict]) -> float:
    """
    Compute inter-annotator agreement (Cohen's Kappa on the first 2 annotators).
    
    Returns Kappa if 2+ annotators, 1.0 if only 1 annotator.
    """
    if len(human_annotations) < 2:
        return 1.0  # Perfect agreement with self
    
    ann1 = human_annotations[0]['labels']
    ann2 = human_annotations[1]['labels']
    min_len = min(len(ann1), len(ann2))
    
    return cohen_kappa(ann1[:min_len], ann2[:min_len])


def compute_overall_human_alignment(results: List["JudgeResult"]) -> Dict[str, Any]:
    """
    Compute global inter-annotator agreement across all available human labels.
    """
    ann1_labels: List[str] = []
    ann2_labels: List[str] = []

    for result in results:
        if len(result.human_annotations) < 2:
            continue
        labels_1 = result.human_annotations[0].get("labels", [])
        labels_2 = result.human_annotations[1].get("labels", [])
        min_len = min(len(labels_1), len(labels_2))
        if min_len == 0:
            continue
        ann1_labels.extend(labels_1[:min_len])
        ann2_labels.extend(labels_2[:min_len])

    if not ann1_labels:
        return {
            "agreement": 0.0,
            "kappa": 0.0,
            "total_pairs": 0,
            "confusion": {"yes_yes": 0, "yes_no": 0, "no_yes": 0, "no_no": 0},
        }

    agreements = sum(1 for a, b in zip(ann1_labels, ann2_labels) if a == b)
    agreement_rate = agreements / len(ann1_labels)
    kappa = cohen_kappa(ann1_labels, ann2_labels)

    confusion = {
        "yes_yes": sum(1 for a, b in zip(ann1_labels, ann2_labels) if a == "yes" and b == "yes"),
        "yes_no": sum(1 for a, b in zip(ann1_labels, ann2_labels) if a == "yes" and b == "no"),
        "no_yes": sum(1 for a, b in zip(ann1_labels, ann2_labels) if a == "no" and b == "yes"),
        "no_no": sum(1 for a, b in zip(ann1_labels, ann2_labels) if a == "no" and b == "no"),
    }

    return {
        "agreement": agreement_rate,
        "kappa": kappa,
        "total_pairs": len(ann1_labels),
        "confusion": confusion,
    }


def cohen_kappa(human: List[str], judge: List[str]) -> float:
    """
    Compute Cohen's Kappa for inter-rater reliability.
    
    Kappa = (P_o - P_e) / (1 - P_e)
    where P_o = observed agreement, P_e = expected agreement by chance
    """
    if len(human) != len(judge) or len(human) == 0:
        return 0.0
    
    # Convert to binary
    human_bin = [1 if h == 'yes' else 0 for h in human]
    judge_bin = [1 if j == 'yes' else 0 for j in judge]
    
    # Observed agreement
    agreements = sum(1 for h, j in zip(human_bin, judge_bin) if h == j)
    p_o = agreements / len(human)
    
    # Expected agreement
    n = len(human)
    yes_human = sum(human_bin)
    yes_judge = sum(judge_bin)
    no_human = n - yes_human
    no_judge = n - yes_judge
    
    p_yes = (yes_human / n) * (yes_judge / n)
    p_no = (no_human / n) * (no_judge / n)
    p_e = p_yes + p_no
    
    if p_e >= 1.0:
        return 0.0
    
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


def compute_metrics(human: List[str], judge: List[str]) -> Dict[str, float]:
    """Compute alignment metrics between human and judge labels."""
    if len(human) == 0 or len(judge) == 0:
        return {'agreement': 0.0, 'kappa': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'n_compared': 0}
    
    # Truncate to minimum length (handle cases where judge returns fewer labels)
    min_len = min(len(human), len(judge))
    human = human[:min_len]
    judge = judge[:min_len]
    
    # Agreement rate
    agreements = sum(1 for h, j in zip(human, judge) if h == j)
    agreement_rate = agreements / len(human)
    
    # Cohen's Kappa
    kappa = cohen_kappa(human, judge)
    
    # Precision/Recall/F1 (yes = positive class)
    tp = sum(1 for h, j in zip(human, judge) if h == 'yes' and j == 'yes')
    fp = sum(1 for h, j in zip(human, judge) if h == 'no' and j == 'yes')
    fn = sum(1 for h, j in zip(human, judge) if h == 'yes' and j == 'no')
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'agreement': agreement_rate,
        'kappa': kappa,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_compared': len(human),
    }


def load_judge_results(
    results_dir: Path,
    uuid_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, List[JudgeResult]]:
    """Load all judge result files."""
    judge_results = {}
    
    for result_file in results_dir.glob('results_*.jsonl'):
        # Extract judge name from filename
        judge_name = result_file.stem.replace('results_', '').replace('_', '/')
        
        results = []
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                # Skip if error or no judge labels
                if data.get('error') or not data.get('judge_labels'):
                    continue
                
                metadata = (uuid_metadata or {}).get(data['uuid'], {})
                results.append(JudgeResult(
                    uuid=data['uuid'],
                    task_id=data['task_id'],
                    task_type=data.get('task_type') or metadata.get('task_type', ''),
                    variant_name=data.get('variant_name') or metadata.get('variant_name', ''),
                    human_annotations=data.get('human_annotations', []),
                    judge_labels=data['judge_labels'],
                    risk_domain=data.get('risk_domain') or metadata.get('risk_domain', ''),
                    risk_subdomain=data.get('risk_subdomain') or metadata.get('risk_subdomain', ''),
                    rubric=data.get('rubric', []),
                    error=data.get('error'),
                ))
        
        if results:
            judge_results[judge_name] = results
            print(f"  Loaded {len(results)} results from {judge_name}")
    
    return judge_results


def load_human_annotations(dataset_path: Path) -> List[JudgeResult]:
    """Load calibration dataset to compute human inter-annotator agreement."""
    results: List[JudgeResult] = []
    if not dataset_path.exists():
        return results

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            results.append(
                JudgeResult(
                    uuid=data["uuid"],
                    task_id=data["task_id"],
                    task_type=data.get("task_type", ""),
                    variant_name=data.get("variant_name", ""),
                    human_annotations=data.get("human_annotations", []),
                    judge_labels=[],
                    risk_domain=data.get("risk_domain", ""),
                    risk_subdomain=data.get("risk_subdomain", ""),
                    rubric=data.get("rubric", []),
                    error=None,
                )
            )

    return results


def load_uuid_metadata(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load task metadata from the calibration dataset keyed by UUID."""
    metadata: Dict[str, Dict[str, Any]] = {}
    if not dataset_path.exists():
        return metadata

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            metadata[data["uuid"]] = {
                "task_type": data.get("task_type", ""),
                "variant_name": data.get("variant_name", ""),
                "risk_domain": data.get("risk_domain", ""),
                "risk_subdomain": data.get("risk_subdomain", ""),
            }

    return metadata


def compute_group_metrics(
    results: List[JudgeResult],
    key_fn,
) -> Dict[str, Dict[str, Any]]:
    """Compute alignment metrics grouped by a key function."""
    grouped: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"human": [], "judge": [], "num_tasks": 0})

    for result in results:
        consensus = compute_consensus_labels(result.human_annotations)
        aligned_human, aligned_judge = align_labels(consensus, result.judge_labels)
        if not aligned_human:
            continue
        key = key_fn(result) or "unknown"
        grouped[key]["human"].extend(aligned_human)
        grouped[key]["judge"].extend(aligned_judge)
        grouped[key]["num_tasks"] += 1

    metrics_by_group = {}
    for key, data in grouped.items():
        metrics = compute_metrics(data["human"], data["judge"])
        metrics_by_group[key] = {
            **metrics,
            "num_tasks": data["num_tasks"],
            "num_items": len(data["human"]),
        }

    return metrics_by_group


def analyze_single_judge(judge_name: str, results: List[JudgeResult]) -> Dict[str, Any]:
    """Analyze single judge performance against consensus human baseline."""
    # Compute consensus human labels and compare to judge
    all_human_consensus = []
    all_judge = []
    all_human_iaas = []  # Track human inter-annotator agreement
    
    for result in results:
        # Compute consensus from multiple annotators
        consensus = compute_consensus_labels(result.human_annotations)
        aligned_human, aligned_judge = align_labels(consensus, result.judge_labels)
        if aligned_human:
            all_human_consensus.extend(aligned_human)
            all_judge.extend(aligned_judge)
        
        # Track human IAA
        if len(result.human_annotations) > 1:
            iaa = compute_human_iaa(result.human_annotations)
            all_human_iaas.append(iaa)
    
    # Overall metrics (judge vs consensus)
    overall_metrics = compute_metrics(all_human_consensus, all_judge)
    
    # Human IAA (ceiling for judge performance)
    avg_human_iaa = np.mean(all_human_iaas) if all_human_iaas else 1.0
    
    # Grouped metrics
    variant_metrics = compute_group_metrics(results, lambda r: r.variant_name)
    task_type_metrics = compute_group_metrics(results, lambda r: r.task_type)
    domain_metrics = compute_group_metrics(results, lambda r: r.risk_domain)
    
    return {
        'judge_name': judge_name,
        'num_tasks': len(results),
        'num_rubric_items': len(all_human_consensus),
        'num_multi_annotator_tasks': len(all_human_iaas),
        'human_iaa_kappa': avg_human_iaa,
        'overall': overall_metrics,
        'by_variant': variant_metrics,
        'by_task_type': task_type_metrics,
        'by_domain': domain_metrics,
    }


def find_best_panel(
    judge_results: Dict[str, List[JudgeResult]],
    max_panel_size: int = 3,
    panel_min_judges: Optional[int] = None,
) -> List[Dict]:
    """Find optimal judge panel by testing combinations."""
    judge_names = list(judge_results.keys())
    
    # For each panel size
    panel_scores = []
    
    from itertools import combinations
    
    for panel_size in range(2, max_panel_size + 1):
        for panel in combinations(judge_names, panel_size):
            # Get all results for this panel
            panel_data = [judge_results[j] for j in panel]
            
            # Compute ensemble agreement for each rubric item
            # Majority vote for each rubric item
            all_human = []
            all_ensemble = []
            
            # Need to align results by UUID
            uuid_to_results = defaultdict(list)
            for judge_name in panel:
                for result in judge_results[judge_name]:
                    uuid_to_results[result.uuid].append(result)
            
            effective_min_judges = panel_min_judges or panel_size
            effective_min_judges = min(effective_min_judges, panel_size)

            # For each UUID, compute ensemble vote
            for uuid, results_for_uuid in uuid_to_results.items():
                if len(results_for_uuid) < effective_min_judges:
                    continue  # Skip if not enough judges completed this task
                
                # Get consensus human labels from first result (all same UUID)
                human_consensus = compute_consensus_labels(results_for_uuid[0].human_annotations)
                
                # Compute majority vote for each rubric item
                for i in range(len(human_consensus)):
                    votes = [r.judge_labels[i] for r in results_for_uuid if i < len(r.judge_labels)]
                    if len(votes) >= effective_min_judges:
                        yes_votes = votes.count('yes')
                        ensemble_label = 'yes' if yes_votes > len(votes) / 2 else 'no'
                        
                        all_human.append(human_consensus[i])
                        all_ensemble.append(ensemble_label)
            
            if all_human:
                metrics = compute_metrics(all_human, all_ensemble)
                panel_scores.append({
                    'panel': list(panel),
                    'size': panel_size,
                    'min_judges': effective_min_judges,
                    'metrics': metrics,
                    'num_items': len(all_human),
                })
    
    # Sort by F1 score
    panel_scores.sort(key=lambda x: x['metrics']['f1'], reverse=True)
    
    return panel_scores[:10]  # Top 10 panels


def get_top_rank_marks(values: List[Optional[float]]) -> List[str]:
    """Return per-row rank marks for top 3 values (*, **, ***) in descending order."""
    marks = [""] * len(values)
    indexed = [(i, v) for i, v in enumerate(values) if v is not None]
    indexed.sort(key=lambda x: x[1], reverse=True)
    for rank, (idx, _) in enumerate(indexed[:3]):
        marks[idx] = "*" * (rank + 1)
    return marks


def build_top3_rankings(
    rows: List[Dict[str, Any]],
    metric_keys: List[str],
) -> Dict[str, Dict[str, str]]:
    rankings: Dict[str, Dict[str, str]] = {}
    for key in metric_keys:
        values = [row.get(key) for row in rows]
        marks = get_top_rank_marks(values)
        rankings[key] = {
            row["judge"]: marks[idx]
            for idx, row in enumerate(rows)
        }
    return rankings


def format_ranked_value(value: Optional[float], rank_mark: Optional[str]) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}{rank_mark or ''}"


def generate_report(
    judge_analyses: List[Dict[str, Any]],
    best_panels: List[Dict],
    human_alignment: Dict[str, Any],
    output_path: Path,
    excluded_judges: Optional[List[str]] = None,
):
    """Generate markdown calibration report."""
    
    lines = []
    lines.append("# Judge Calibration Report")
    lines.append("")
    if excluded_judges:
        excluded_str = ", ".join(excluded_judges)
        lines.append(f"**Excluded Judges:** {excluded_str}")
        lines.append("")
    lines.append("Analysis of judge model alignment with human baseline labels (consensus from multiple annotators).")
    lines.append("")
    
    # ========================================================================
    # Human Inter-Annotator Agreement (Ceiling)
    # ========================================================================
    lines.append("## Human Inter-Annotator Agreement")
    lines.append("")
    
    if human_alignment and human_alignment["total_pairs"] > 0:
        lines.append(f"**Overall Human Kappa:** {human_alignment['kappa']:.3f}")
        lines.append(f"**Raw Agreement:** {human_alignment['agreement']:.2%}")
        lines.append(f"**Total label pairs:** {human_alignment['total_pairs']}")
        lines.append("")
        lines.append("**Confusion Matrix (Human 1 vs Human 2):**")
        lines.append("")
        lines.append("| Human 1 \\ Human 2 | Yes | No |")
        lines.append("|---|---|---|")
        lines.append(
            f"| Yes | {human_alignment['confusion']['yes_yes']} | {human_alignment['confusion']['yes_no']} |"
        )
        lines.append(
            f"| No | {human_alignment['confusion']['no_yes']} | {human_alignment['confusion']['no_no']} |"
        )
        lines.append("")
        lines.append("This represents the **ceiling** for judge performance - judges cannot reliably exceed human agreement.")
        lines.append("")
    
    # ========================================================================
    # Overall Rankings
    # ========================================================================
    lines.append("## Overall Rankings")
    lines.append("")
    lines.append("Judges ranked by F1 score (harmfulness detection against consensus human baseline):")
    lines.append("")
    
    # Sort by F1
    sorted_judges = sorted(judge_analyses, key=lambda x: x['overall']['f1'], reverse=True)
    
    lines.append("| Rank | Judge | Agreement | Kappa | Precision | Recall | F1 | Tasks |")
    lines.append("|------|-------|-----------|-------|-----------|--------|----|----|")
    
    for i, analysis in enumerate(sorted_judges, 1):
        m = analysis['overall']
        lines.append(
            f"| {i} | `{analysis['judge_name']}` | "
            f"{m['agreement']:.3f} | {m['kappa']:.3f} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {analysis['num_tasks']} |"
        )
    
    lines.append("")

    # ========================================================================
    # Alignment by Task Type
    # ========================================================================
    lines.append("## Alignment by Task Type")
    lines.append("")
    lines.append("Top 3 scores per column are marked as * (1st), ** (2nd), *** (3rd).")
    lines.append("")
    lines.append("| Judge | Agnostic F1 | Agnostic Agree | Specific F1 | Specific Agree |")
    lines.append("|------|-------------|----------------|-------------|----------------|")

    task_type_keys = [
        "CULTURE_AGNOSTIC_TRANSLATABLE",
        "CULTURE_SPECIFIC_TRANSLATABLE",
    ]

    task_rows = []
    for analysis in sorted_judges:
        by_task_type = analysis.get("by_task_type", {})
        task_rows.append({
            "judge": analysis["judge_name"],
            "agnostic_f1": by_task_type.get(task_type_keys[0], {}).get("f1"),
            "agnostic_agree": by_task_type.get(task_type_keys[0], {}).get("agreement"),
            "specific_f1": by_task_type.get(task_type_keys[1], {}).get("f1"),
            "specific_agree": by_task_type.get(task_type_keys[1], {}).get("agreement"),
        })
    task_rankings = build_top3_rankings(
        task_rows,
        ["agnostic_f1", "agnostic_agree", "specific_f1", "specific_agree"],
    )
    for row in task_rows:
        row_cells = [f"`{row['judge']}`"]
        row_cells.append(format_ranked_value(row["agnostic_f1"], task_rankings["agnostic_f1"].get(row["judge"])))
        row_cells.append(format_ranked_value(row["agnostic_agree"], task_rankings["agnostic_agree"].get(row["judge"])))
        row_cells.append(format_ranked_value(row["specific_f1"], task_rankings["specific_f1"].get(row["judge"])))
        row_cells.append(format_ranked_value(row["specific_agree"], task_rankings["specific_agree"].get(row["judge"])))
        lines.append(f"| {' | '.join(row_cells)} |")
    lines.append("")

    # ========================================================================
    # Alignment by Variant
    # ========================================================================
    lines.append("## Alignment by Variant")
    lines.append("")
    lines.append("Top 3 scores per column are marked as * (1st), ** (2nd), *** (3rd).")
    lines.append("")
    lines.append("| Judge | original_en F1 | original_en Agree | translated_kr F1 | translated_kr Agree |")
    lines.append("|------|----------------|-------------------|-------------------|--------------------|")

    variant_keys = ["original_en", "translated_kr"]
    variant_rows_primary = []
    for analysis in sorted_judges:
        by_variant = analysis.get("by_variant", {})
        variant_rows_primary.append({
            "judge": analysis["judge_name"],
            "original_f1": by_variant.get(variant_keys[0], {}).get("f1"),
            "original_agree": by_variant.get(variant_keys[0], {}).get("agreement"),
            "translated_f1": by_variant.get(variant_keys[1], {}).get("f1"),
            "translated_agree": by_variant.get(variant_keys[1], {}).get("agreement"),
        })
    variant_primary_rankings = build_top3_rankings(
        variant_rows_primary,
        ["original_f1", "original_agree", "translated_f1", "translated_agree"],
    )
    for row in variant_rows_primary:
        row_cells = [f"`{row['judge']}`"]
        row_cells.append(format_ranked_value(row["original_f1"], variant_primary_rankings["original_f1"].get(row["judge"])))
        row_cells.append(format_ranked_value(row["original_agree"], variant_primary_rankings["original_agree"].get(row["judge"])))
        row_cells.append(format_ranked_value(row["translated_f1"], variant_primary_rankings["translated_f1"].get(row["judge"])))
        row_cells.append(format_ranked_value(row["translated_agree"], variant_primary_rankings["translated_agree"].get(row["judge"])))
        lines.append(f"| {' | '.join(row_cells)} |")
    lines.append("")

    lines.append("| Judge | cultural_adapted_en F1 | cultural_adapted_en Agree | transcreated_kr F1 | transcreated_kr Agree |")
    lines.append("|------|-------------------------|--------------------------|--------------------|-----------------------|")
    variant_keys = ["cultural_adapted_en", "transcreated_kr"]
    variant_rows_secondary = []
    for analysis in sorted_judges:
        by_variant = analysis.get("by_variant", {})
        variant_rows_secondary.append({
            "judge": analysis["judge_name"],
            "cultural_f1": by_variant.get(variant_keys[0], {}).get("f1"),
            "cultural_agree": by_variant.get(variant_keys[0], {}).get("agreement"),
            "transcreated_f1": by_variant.get(variant_keys[1], {}).get("f1"),
            "transcreated_agree": by_variant.get(variant_keys[1], {}).get("agreement"),
        })
    variant_secondary_rankings = build_top3_rankings(
        variant_rows_secondary,
        ["cultural_f1", "cultural_agree", "transcreated_f1", "transcreated_agree"],
    )
    for row in variant_rows_secondary:
        row_cells = [f"`{row['judge']}`"]
        row_cells.append(format_ranked_value(row["cultural_f1"], variant_secondary_rankings["cultural_f1"].get(row["judge"])))
        row_cells.append(format_ranked_value(row["cultural_agree"], variant_secondary_rankings["cultural_agree"].get(row["judge"])))
        row_cells.append(format_ranked_value(row["transcreated_f1"], variant_secondary_rankings["transcreated_f1"].get(row["judge"])))
        row_cells.append(format_ranked_value(row["transcreated_agree"], variant_secondary_rankings["transcreated_agree"].get(row["judge"])))
        lines.append(f"| {' | '.join(row_cells)} |")
    lines.append("")

    # ========================================================================
    # Alignment by Domain
    # ========================================================================
    lines.append("## Alignment by Domain")
    lines.append("")
    lines.append("Top 3 scores per column are marked as * (1st), ** (2nd), *** (3rd).")
    lines.append("")
    lines.append("| Judge | CBRNE F1 | CBRNE Agree | Political Violence F1 | Political Violence Agree | Criminal/Financial F1 | Criminal/Financial Agree |")
    lines.append("|------|----------|------------|------------------------|--------------------------|-----------------------|--------------------------|")

    domain_keys = [
        "Chemical, Biological, Radiological, Nuclear, and Explosive (CBRNE)",
        "Political Violence & Terrorism",
        "Criminal & Financial Illicit Activities",
    ]
    domain_rows = []
    for analysis in sorted_judges:
        by_domain = analysis.get("by_domain", {})
        domain_rows.append({
            "judge": analysis["judge_name"],
            "cbrne_f1": by_domain.get(domain_keys[0], {}).get("f1"),
            "cbrne_agree": by_domain.get(domain_keys[0], {}).get("agreement"),
            "pv_f1": by_domain.get(domain_keys[1], {}).get("f1"),
            "pv_agree": by_domain.get(domain_keys[1], {}).get("agreement"),
            "cf_f1": by_domain.get(domain_keys[2], {}).get("f1"),
            "cf_agree": by_domain.get(domain_keys[2], {}).get("agreement"),
        })
    domain_rankings = build_top3_rankings(
        domain_rows,
        ["cbrne_f1", "cbrne_agree", "pv_f1", "pv_agree", "cf_f1", "cf_agree"],
    )
    for row in domain_rows:
        row_cells = [f"`{row['judge']}`"]
        row_cells.append(format_ranked_value(row["cbrne_f1"], domain_rankings["cbrne_f1"].get(row["judge"])))
        row_cells.append(format_ranked_value(row["cbrne_agree"], domain_rankings["cbrne_agree"].get(row["judge"])))
        row_cells.append(format_ranked_value(row["pv_f1"], domain_rankings["pv_f1"].get(row["judge"])))
        row_cells.append(format_ranked_value(row["pv_agree"], domain_rankings["pv_agree"].get(row["judge"])))
        row_cells.append(format_ranked_value(row["cf_f1"], domain_rankings["cf_f1"].get(row["judge"])))
        row_cells.append(format_ranked_value(row["cf_agree"], domain_rankings["cf_agree"].get(row["judge"])))
        lines.append(f"| {' | '.join(row_cells)} |")
    lines.append("")
    
    # ========================================================================
    # Top 5 Detailed Analysis
    # ========================================================================
    lines.append("## Top 5 Judges - Detailed Analysis")
    lines.append("")
    
    for analysis in sorted_judges[:5]:
        lines.append(f"### {analysis['judge_name']}")
        lines.append("")
        lines.append(f"- **Tasks evaluated:** {analysis['num_tasks']}")
        lines.append(f"- **Rubric items:** {analysis['num_rubric_items']}")
        lines.append("")
        
        m = analysis['overall']
        lines.append("**Overall Metrics:**")
        lines.append(f"- Agreement: {m['agreement']:.1%}")
        lines.append(f"- Cohen's Kappa: {m['kappa']:.3f}")
        lines.append(f"- Precision: {m['precision']:.1%}")
        lines.append(f"- Recall: {m['recall']:.1%}")
        lines.append(f"- F1 Score: {m['f1']:.3f}")
        lines.append("")
        
        # Per-variant breakdown
        if analysis['by_variant']:
            lines.append("**By Variant:**")
            lines.append("")
            lines.append("| Variant | Agreement | Kappa | F1 |")
            lines.append("|---------|-----------|-------|----|")
            for variant, vm in analysis['by_variant'].items():
                lines.append(f"| {variant} | {vm['agreement']:.3f} | {vm['kappa']:.3f} | {vm['f1']:.3f} |")
            lines.append("")
        
        lines.append("")
    
    # ========================================================================
    # Best Panels
    # ========================================================================
    lines.append("## Best Judge Panels")
    lines.append("")
    lines.append("Top panels ranked by F1 score (majority voting):")
    lines.append("")
    
    for i, panel_info in enumerate(best_panels[:5], 1):
        panel_str = " + ".join([f"`{j}`" for j in panel_info['panel']])
        m = panel_info['metrics']
        min_judges = panel_info.get("min_judges", panel_info["size"])
        if min_judges == panel_info["size"]:
            item_note = "rubric items with all panel judges"
        else:
            item_note = f"rubric items with at least {min_judges} judges"
        lines.append(f"### Panel {i}: {panel_str}")
        lines.append("")
        lines.append(f"- **Size:** {panel_info['size']} judges")
        lines.append(f"- **Items evaluated:** {panel_info['num_items']} ({item_note})")
        lines.append(f"- **Agreement:** {m['agreement']:.1%}")
        lines.append(f"- **Kappa:** {m['kappa']:.3f}")
        lines.append(f"- **F1 Score:** {m['f1']:.3f}")
        lines.append("")
    
    # ========================================================================
    # Culture Analysis (if applicable)
    # ========================================================================
    lines.append("## Culture-Specific Analysis")
    lines.append("")
    
    # Group by culture (Western vs Korean)
    western_judges = [a for a in judge_analyses if 'openai' in a['judge_name'].lower() 
                      or 'anthropic' in a['judge_name'].lower() 
                      or 'gemini' in a['judge_name'].lower()
                      or 'mistral' in a['judge_name'].lower()
                      or 'llama' in a['judge_name'].lower()]
    
    korean_judges = [a for a in judge_analyses if 'exaone' in a['judge_name'].lower()
                     or 'solar' in a['judge_name'].lower()
                     or 'clovax' in a['judge_name'].lower()
                     or 'kanana' in a['judge_name'].lower()
                     or 'a.x-k1' in a['judge_name'].lower()
                     or 'a_x_k1' in a['judge_name'].lower()]
    
    if western_judges and korean_judges:
        western_avg_f1 = np.mean([a['overall']['f1'] for a in western_judges])
        korean_avg_f1 = np.mean([a['overall']['f1'] for a in korean_judges])
        
        lines.append(f"**Western Models (n={len(western_judges)}):**")
        lines.append(f"- Average F1: {western_avg_f1:.3f}")
        lines.append("")
        
        lines.append(f"**Korean Models (n={len(korean_judges)}):**")
        lines.append(f"- Average F1: {korean_avg_f1:.3f}")
        lines.append("")
    
    # ========================================================================
    # Recommendations
    # ========================================================================
    lines.append("## Recommendations")
    lines.append("")
    
    best_judge = sorted_judges[0]
    lines.append(f"**Best Single Judge:** `{best_judge['judge_name']}`")
    lines.append(f"- F1 Score: {best_judge['overall']['f1']:.3f}")
    lines.append(f"- Agreement: {best_judge['overall']['agreement']:.1%}")
    lines.append(f"- Kappa: {best_judge['overall']['kappa']:.3f}")
    lines.append("")
    
    if best_panels:
        best_panel = best_panels[0]
        panel_str = " + ".join([f"`{j}`" for j in best_panel['panel']])
        lines.append(f"**Best Panel:** {panel_str}")
        lines.append(f"- F1 Score: {best_panel['metrics']['f1']:.3f}")
        lines.append(f"- Agreement: {best_panel['metrics']['agreement']:.1%}")
        lines.append("")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def filter_judge_results(
    judge_results: Dict[str, List[JudgeResult]],
    exclude_terms: List[str],
) -> Dict[str, List[JudgeResult]]:
    if not exclude_terms:
        return judge_results
    lowered_terms = [term.lower() for term in exclude_terms]
    filtered = {}
    for judge_name, results in judge_results.items():
        judge_key = judge_name.lower()
        if any(term in judge_key for term in lowered_terms):
            continue
        filtered[judge_name] = results
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Analyze judge calibration results.")
    parser.add_argument(
        "--exclude-judges",
        type=str,
        default="",
        help="Comma-separated list of judge name substrings to exclude.",
    )
    parser.add_argument(
        "--panel-min-judges",
        type=int,
        default=0,
        help="Minimum judges required per item when scoring panels (default: all judges).",
    )
    args = parser.parse_args()
    exclude_terms = parse_csv_list(args.exclude_judges)
    panel_min_judges = args.panel_min_judges or None

    base_dir = Path(__file__).resolve().parents[2]
    results_dir = base_dir / 'runs' / 'judge-calibration'
    dataset_path = results_dir / 'calibration_dataset.jsonl'
    output_path = results_dir / 'calibration_report.md'
    summary_path = results_dir / 'calibration_summary.json'
    
    print("=" * 80)
    print("JUDGE CALIBRATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Load dataset metadata for task_type/domain enrichment
    uuid_metadata = load_uuid_metadata(dataset_path)

    # Load all judge results
    print("Loading judge results...")
    judge_results = load_judge_results(results_dir, uuid_metadata=uuid_metadata)
    judge_results = filter_judge_results(judge_results, exclude_terms)
    print(f"Loaded {len(judge_results)} judge models")
    print()
    
    if not judge_results:
        print("❌ No judge results found in", results_dir)
        return
    
    # Analyze each judge
    print("Analyzing individual judges...")
    judge_analyses = []
    for judge_name, results in judge_results.items():
        analysis = analyze_single_judge(judge_name, results)
        judge_analyses.append(analysis)
        print(f"  {judge_name}: F1={analysis['overall']['f1']:.3f}, "
              f"Agreement={analysis['overall']['agreement']:.1%}")
    
    print()
    
    # Compute overall human inter-annotator agreement
    human_results = load_human_annotations(dataset_path)
    human_alignment = compute_overall_human_alignment(human_results)
    if human_alignment["total_pairs"] > 0:
        print(
            "Human IAA: "
            f"Kappa={human_alignment['kappa']:.3f}, "
            f"Agreement={human_alignment['agreement']:.2%}, "
            f"Pairs={human_alignment['total_pairs']}"
        )
    print()

    # Find best panels
    print("Finding best judge panels...")
    best_panels = find_best_panel(
        judge_results,
        max_panel_size=3,
        panel_min_judges=panel_min_judges,
    )
    if best_panels:
        print(f"  Top panel: {' + '.join(best_panels[0]['panel'])}")
        print(f"  F1: {best_panels[0]['metrics']['f1']:.3f}")
    print()
    
    # Generate report
    print("Generating report...")
    generate_report(
        judge_analyses,
        best_panels,
        human_alignment,
        output_path,
        excluded_judges=exclude_terms,
    )
    
    # Save JSON summary
    summary = {
        'human_alignment': human_alignment,
        'judges': judge_analyses,
        'best_panels': best_panels[:5],
        'excluded_judges': exclude_terms,
        'panel_min_judges': panel_min_judges,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()
    print(f"✓ Report: {output_path}")
    print(f"✓ Summary: {summary_path}")
    print()


if __name__ == '__main__':
    main()
