#!/usr/bin/env python3
"""
Analyze harm classifier calibration against human labels.

Computes per-classifier metrics:
- Exact match rate (per-rubric)
- Cohen's Kappa (multiclass, 7 harm dimensions)
- Macro F1, per-dimension precision/recall/F1

Panel analysis:
- All 2-3 classifier combinations via majority voting
- Pairwise inter-classifier Kappa

Output:
  runs/rebuttal-calibration-analysis/harm_classifier_calibration_report.md
  analysis/experiments/figures/fig_harm_classifier_kappa.{pdf,png}
"""

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rok_fortress_eval.harm_classifier.data_models import HarmDimension, parse_harm_dimension
from rok_fortress_eval.harm_classifier.meta_eval import load_human_labels
from scripts.metaeval.dedup_models import (
    deduplicate_harm_classifier_files,
    extract_judge_name,
)

# ---- Constants ----

HUMAN_LABELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "archive"
    / "Copy of fortress private classification - 2026-01-14-1126 - META EVAL MASTER (DO NOT EDIT) (3).tsv"
)
CLASSIFIER_RESULTS_DIR = PROJECT_ROOT / "runs" / "harm-classifier-calibration"
REPORT_DIR = PROJECT_ROOT / "runs" / "rebuttal-calibration-analysis"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "experiments" / "figures"

DIMENSION_VALUES = [d.value for d in HarmDimension]


# ---- Data Loading ----


def load_classifier_predictions(filepath: Path) -> Dict[str, List[str]]:
    """
    Load classifier predictions from a JSONL file.

    Returns:
        Dict mapping task_id -> list of predicted dimension strings (one per rubric item,
        ordered by rubric_index).
    """
    predictions: Dict[str, List[Tuple[int, str]]] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = data.get("task_id", "")
            if not task_id:
                continue
            if data.get("error") or data.get("refused"):
                continue
            classifications = data.get("classifications", [])
            if not classifications:
                continue
            items: List[Tuple[int, str]] = []
            for cls in classifications:
                dim = parse_harm_dimension(cls.get("primary_harm", ""))
                if dim is None:
                    continue
                items.append((cls.get("rubric_index", 0), dim.value))
            if items:
                items.sort(key=lambda x: x[0])
                predictions[task_id] = [v for _, v in items]
    return predictions


# ---- Metrics ----


def multiclass_cohen_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    """
    Multiclass Cohen's Kappa.

    kappa = (po - pe) / (1 - pe)
    pe = sum over classes c of P(rater1=c) * P(rater2=c)
    """
    if len(labels_a) != len(labels_b) or len(labels_a) == 0:
        return 0.0
    n = len(labels_a)
    # Observed agreement
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    # Expected agreement
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    all_classes = set(counts_a.keys()) | set(counts_b.keys())
    pe = sum((counts_a.get(c, 0) / n) * (counts_b.get(c, 0) / n) for c in all_classes)
    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1 - pe)


def compute_per_dimension_metrics(
    true_labels: List[str], pred_labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-dimension precision, recall, F1 (one-vs-rest).

    Returns dict: dimension -> {precision, recall, f1}
    """
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for t, p in zip(true_labels, pred_labels):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    results: Dict[str, Dict[str, float]] = {}
    for dim in DIMENSION_VALUES:
        t = tp[dim]
        f_p = fp[dim]
        f_n = fn[dim]
        prec = t / (t + f_p) if (t + f_p) > 0 else 0.0
        rec = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[dim] = {"precision": prec, "recall": rec, "f1": f1}
    return results


def macro_f1(per_dim: Dict[str, Dict[str, float]]) -> float:
    """Compute macro F1 from per-dimension metrics."""
    f1s = [v["f1"] for v in per_dim.values()]
    return np.mean(f1s) if f1s else 0.0


def compute_classifier_metrics(
    human: Dict[str, List[HarmDimension]],
    predictions: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Compare one classifier's predictions to human labels.

    Returns dict with exact_match, kappa, macro_f1, per_dimension, n_rubrics, n_tasks.
    """
    all_true: List[str] = []
    all_pred: List[str] = []
    overlapping = set(human.keys()) & set(predictions.keys())

    for task_id in overlapping:
        h_dims = human[task_id]
        p_dims = predictions[task_id]
        min_len = min(len(h_dims), len(p_dims))
        for i in range(min_len):
            all_true.append(h_dims[i].value)
            all_pred.append(p_dims[i])

    if not all_true:
        return {
            "exact_match": 0.0,
            "kappa": 0.0,
            "macro_f1": 0.0,
            "per_dimension": {},
            "n_rubrics": 0,
            "n_tasks": 0,
        }

    exact = sum(1 for t, p in zip(all_true, all_pred) if t == p) / len(all_true)
    kappa = multiclass_cohen_kappa(all_true, all_pred)
    per_dim = compute_per_dimension_metrics(all_true, all_pred)
    mf1 = macro_f1(per_dim)

    return {
        "exact_match": exact,
        "kappa": kappa,
        "macro_f1": mf1,
        "per_dimension": per_dim,
        "n_rubrics": len(all_true),
        "n_tasks": len(overlapping),
    }


# ---- Panel (Majority Voting) ----


def majority_vote_predictions(
    panel_preds: List[Dict[str, List[str]]],
    human: Dict[str, List[HarmDimension]],
) -> Dict[str, List[str]]:
    """
    Compute majority-vote predictions across panel members.

    For each task and each rubric item, take the most common prediction.
    Only includes tasks present in ALL panel members AND in human labels.
    """
    if not panel_preds:
        return {}

    common_tasks = set(human.keys())
    for preds in panel_preds:
        common_tasks &= set(preds.keys())

    result: Dict[str, List[str]] = {}
    for task_id in common_tasks:
        h_len = len(human[task_id])
        min_len = h_len
        for preds in panel_preds:
            min_len = min(min_len, len(preds[task_id]))
        if min_len == 0:
            continue
        voted: List[str] = []
        for i in range(min_len):
            item_votes = [preds[task_id][i] for preds in panel_preds if i < len(preds[task_id])]
            if item_votes:
                voted.append(Counter(item_votes).most_common(1)[0][0])
        if voted:
            result[task_id] = voted
    return result


# ---- Report Generation ----


def format_classifier_name(name: str) -> str:
    """Make classifier names more readable."""
    name = name.replace("_", " ").replace("/", " ")
    # Shorten common prefixes
    for prefix in ["anthropic claude ", "openai ", "gemini ", "fireworks ai ", "bedrock ", "vllm "]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.strip()


def generate_report(
    classifier_metrics: Dict[str, Dict[str, Any]],
    panel_metrics: List[Tuple[str, Dict[str, Any]]],
    pairwise_kappa: Dict[Tuple[str, str], float],
    top_n_panels: int = 5,
) -> str:
    """Generate markdown report."""
    lines: List[str] = []

    lines.append("# Harm Classifier Calibration Report")
    lines.append("")
    lines.append("Comparison of LLM-based harm dimension classifiers against human labels.")
    lines.append("Metrics: exact match rate, multiclass Cohen's Kappa (7 dimensions), macro F1.")
    lines.append("")

    # --- Section 1: Individual classifier rankings ---
    lines.append("## 1. Individual Classifier Rankings")
    lines.append("")

    # Sort by macro F1 descending
    ranked = sorted(
        classifier_metrics.items(), key=lambda x: x[1]["macro_f1"], reverse=True
    )

    lines.append(
        "| Rank | Classifier | Exact Match | Kappa | Macro F1 | Tasks | Rubrics |"
    )
    lines.append(
        "|------|-----------|-------------|-------|----------|-------|---------|"
    )
    for i, (name, m) in enumerate(ranked, 1):
        display = format_classifier_name(name)
        lines.append(
            f"| {i} | {display} | {m['exact_match']:.3f} | "
            f"{m['kappa']:.3f} | {m['macro_f1']:.3f} | "
            f"{m['n_tasks']} | {m['n_rubrics']} |"
        )
    lines.append("")

    # --- Section 2: Per-dimension P/R/F1 for top 5 classifiers ---
    lines.append("## 2. Per-Dimension Metrics (Top 5 Classifiers)")
    lines.append("")

    for name, m in ranked[:5]:
        display = format_classifier_name(name)
        lines.append(f"### {display}")
        lines.append(f"Overall: exact_match={m['exact_match']:.3f}, kappa={m['kappa']:.3f}, macro_f1={m['macro_f1']:.3f}")
        lines.append("")
        lines.append("| Dimension | Precision | Recall | F1 |")
        lines.append("|-----------|-----------|--------|----|")
        per_dim = m.get("per_dimension", {})
        for dim in DIMENSION_VALUES:
            d = per_dim.get(dim, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
            lines.append(
                f"| {dim} | {d['precision']:.3f} | {d['recall']:.3f} | {d['f1']:.3f} |"
            )
        lines.append("")

    # --- Section 3: Top 5 panels (majority voting) ---
    lines.append("## 3. Top 5 Panels (Majority Voting)")
    lines.append("")
    lines.append(
        "| Rank | Panel | Exact Match | Kappa | Macro F1 | Tasks | Rubrics |"
    )
    lines.append(
        "|------|-------|-------------|-------|----------|-------|---------|"
    )
    for i, (label, m) in enumerate(panel_metrics[:top_n_panels], 1):
        lines.append(
            f"| {i} | {label} | {m['exact_match']:.3f} | "
            f"{m['kappa']:.3f} | {m['macro_f1']:.3f} | "
            f"{m['n_tasks']} | {m['n_rubrics']} |"
        )
    lines.append("")

    # --- Section 4: Pairwise inter-classifier Kappa ---
    lines.append("## 4. Pairwise Inter-Classifier Kappa (Top 20)")
    lines.append("")

    sorted_pairs = sorted(pairwise_kappa.items(), key=lambda x: x[1], reverse=True)
    lines.append("| Rank | Classifier A | Classifier B | Kappa |")
    lines.append("|------|-------------|-------------|-------|")
    for i, ((a, b), k) in enumerate(sorted_pairs[:20], 1):
        lines.append(
            f"| {i} | {format_classifier_name(a)} | {format_classifier_name(b)} | {k:.3f} |"
        )
    lines.append("")

    return "\n".join(lines)


# ---- Figure Generation ----


def generate_figure(
    classifier_metrics: Dict[str, Dict[str, Any]],
    best_panel: Optional[Tuple[str, Dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Generate horizontal bar chart of classifier Kappa values + best panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sort classifiers by kappa
    ranked = sorted(
        classifier_metrics.items(), key=lambda x: x[1]["kappa"], reverse=False
    )

    names = [format_classifier_name(n) for n, _ in ranked]
    kappas = [m["kappa"] for _, m in ranked]

    # Add best panel at the end if available
    if best_panel:
        names.append(f"PANEL: {best_panel[0]}")
        kappas.append(best_panel[1]["kappa"])

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
    colors = ["#4C72B0"] * len(ranked)
    if best_panel:
        colors.append("#C44E52")  # Red for panel

    y_pos = np.arange(len(names))
    ax.barh(y_pos, kappas, color=colors, edgecolor="white", height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Cohen's Kappa (multiclass, 7 dimensions)", fontsize=11)
    ax.set_title("Harm Classifier Calibration: Kappa vs Human Labels", fontsize=13)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)

    # Add value labels
    for i, v in enumerate(kappas):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig_harm_classifier_kappa.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved to {output_dir}/fig_harm_classifier_kappa.{{pdf,png}}")


# ---- Main ----


def main() -> None:
    print("=" * 60)
    print("Harm Classifier Calibration Analysis")
    print("=" * 60)

    # 1. Load human labels
    print(f"\nLoading human labels from:\n  {HUMAN_LABELS_PATH}")
    human_labels = load_human_labels(
        str(HUMAN_LABELS_PATH),
        task_id_column="TASK_ID",
        harm_column="ORIGINAL_RUBRIC_HARM",
    )
    print(f"  Loaded {len(human_labels)} tasks with human labels")

    # 2. Discover and deduplicate classifier files
    print(f"\nDiscovering classifier results in:\n  {CLASSIFIER_RESULTS_DIR}")
    classifier_files = deduplicate_harm_classifier_files(CLASSIFIER_RESULTS_DIR)
    print(f"  Found {len(classifier_files)} deduplicated classifier files")

    # 3. Load all classifier predictions (exclude classifiers with < min_tasks overlap)
    MIN_OVERLAP_TASKS = 10
    all_predictions: Dict[str, Dict[str, List[str]]] = {}
    for filepath in classifier_files:
        name = extract_judge_name(filepath.name)
        preds = load_classifier_predictions(filepath)
        if preds:
            overlapping = len(set(preds.keys()) & set(human_labels.keys()))
            if overlapping < MIN_OVERLAP_TASKS:
                print(f"  SKIPPED {name}: only {overlapping} overlapping tasks (< {MIN_OVERLAP_TASKS})")
                continue
            all_predictions[name] = preds
            print(f"  {name}: {len(preds)} tasks ({overlapping} overlap with human)")

    if not all_predictions:
        print("ERROR: No classifier predictions loaded.")
        return

    # 4. Compute per-classifier metrics
    print("\nComputing per-classifier metrics...")
    classifier_metrics: Dict[str, Dict[str, Any]] = {}
    for name, preds in all_predictions.items():
        metrics = compute_classifier_metrics(human_labels, preds)
        classifier_metrics[name] = metrics
        print(
            f"  {name}: exact={metrics['exact_match']:.3f} "
            f"kappa={metrics['kappa']:.3f} macroF1={metrics['macro_f1']:.3f}"
        )

    # 5. Pairwise inter-classifier Kappa
    print("\nComputing pairwise inter-classifier Kappa...")
    classifier_names = sorted(all_predictions.keys())
    pairwise_kappa: Dict[Tuple[str, str], float] = {}

    for a, b in combinations(classifier_names, 2):
        preds_a = all_predictions[a]
        preds_b = all_predictions[b]
        common = set(preds_a.keys()) & set(preds_b.keys())
        labels_a: List[str] = []
        labels_b: List[str] = []
        for tid in common:
            min_len = min(len(preds_a[tid]), len(preds_b[tid]))
            for i in range(min_len):
                labels_a.append(preds_a[tid][i])
                labels_b.append(preds_b[tid][i])
        if labels_a:
            pairwise_kappa[(a, b)] = multiclass_cohen_kappa(labels_a, labels_b)

    # 6. Panel analysis (all 2-3 combinations)
    print("\nTesting panel combinations (majority voting)...")
    panel_results: List[Tuple[str, Dict[str, Any]]] = []

    for panel_size in (2, 3):
        for combo in combinations(classifier_names, panel_size):
            panel_preds_list = [all_predictions[name] for name in combo]
            voted = majority_vote_predictions(panel_preds_list, human_labels)
            if not voted:
                continue
            metrics = compute_classifier_metrics(human_labels, voted)
            label = " + ".join(format_classifier_name(n) for n in combo)
            panel_results.append((label, metrics))

    # Sort panels by macro F1 descending
    panel_results.sort(key=lambda x: x[1]["macro_f1"], reverse=True)
    print(f"  Tested {len(panel_results)} panel combinations")

    if panel_results:
        best = panel_results[0]
        print(
            f"  Best panel: {best[0]}"
            f" (macroF1={best[1]['macro_f1']:.3f}, kappa={best[1]['kappa']:.3f})"
        )

    # 7. Generate report
    print("\nGenerating report...")
    report = generate_report(classifier_metrics, panel_results, pairwise_kappa)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "harm_classifier_calibration_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report: {report_path}")

    # 8. Generate figure
    print("\nGenerating figure...")
    best_panel = panel_results[0] if panel_results else None
    generate_figure(classifier_metrics, best_panel, FIGURES_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
