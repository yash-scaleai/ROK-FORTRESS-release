#!/usr/bin/env python3
"""
Domain-stratified ΔLing / ΔCtx with bootstrap 95% CIs.

ΔLing per task (CA): TRS(V_En) - TRS(V_Ko)            [positive => Korean more conservative]
ΔCtx per task (CS):  TRS(V_En,US) - TRS(V_En,KR)      [positive => KR context more conservative]

Stratified by risk domain (CBRNE, Political, Criminal, Information).
Aggregate-across-models view (headline) plus per-model x domain matrix (appendix).

Bootstrap: percentile method, resampling at the unit of paired (task,model) diffs (or
paired tasks within a single model for the per-model matrix). 2000 iterations.

Output:
  runs/rebuttal-calibration-analysis/domain_drops_report.md
  runs/rebuttal-calibration-analysis/domain_drops.json
  analysis/experiments/figures/fig_domain_drops.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.experiments.exp_data.filters import (
    DOMAINS,
    filter_by_domain,
    filter_culture_agnostic,
    filter_culture_specific,
)
from analysis.experiments.exp_data.loaders import (
    ConsolidatedResult,
    load_all_judgements_with_responses,
    load_harm_classifications,
    merge_harm_dimensions,
)
from analysis.experiments.exp_data.models import (
    get_model_display_name,
    is_global_model,
    is_regional_model,
)
from analysis.experiments.exp_data.variants import (
    CULTURAL_ADAPTED_EN,
    ORIGINAL_EN,
    TRANSLATED_KR,
    group_by_task_id,
)
from analysis.experiments.metrics.harm_scores import compute_harm_score_for_result


DEFAULT_JUDGEMENTS_DIR = PROJECT_ROOT / "runs" / "rok_fortress_judgements_final"
DEFAULT_RESPONSES_DIR = PROJECT_ROOT / "runs" / "rok_fortress_responses_final"
DEFAULT_HARM_CLASSIFIER = (
    PROJECT_ROOT / "runs" / "harm-classifier-final"
    / "results_gemini_gemini-3-pro-preview.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "rebuttal-calibration-analysis"
FIGURE_DIR = PROJECT_ROOT / "analysis" / "experiments" / "figures"

DOMAIN_ORDER = ["CBRNE", "Political", "Criminal", "Information"]
DOMAIN_DISPLAY = {
    "CBRNE": "CBRNE",
    "Political": "Political Violence",
    "Criminal": "Criminal Activity",
    "Information": "Information Leakage",
}


# ----------------------------------------------------------------------
# Paired-diff extraction
# ----------------------------------------------------------------------

def trs_or_none(result: ConsolidatedResult) -> Optional[float]:
    """TRS for a result, treating refusals as 0.0 and excluded responses as None."""
    if result.is_excluded:
        return None
    if not result.is_valid and not result.is_safe_refusal:
        return None
    return float(compute_harm_score_for_result(result, use_weights=True).harm_score)


@dataclass
class PairedDiff:
    """A single paired Δ for one (model, task)."""
    model: str
    task_id: str
    domain_short: str
    drop_type: str  # "ling" or "ctx"
    diff: float     # positive = English harm > Korean / KR-context harm


def extract_paired_diffs(
    results: List[ConsolidatedResult],
    model: str,
    drop_type: str,
) -> List[PairedDiff]:
    """
    Build paired ΔLing (CA) or ΔCtx (CS) per task for a single model.

    drop_type=="ling": ΔLing on CA tasks, paired (ORIGINAL_EN, TRANSLATED_KR)
    drop_type=="ctx":  ΔCtx on CS tasks, paired (ORIGINAL_EN, CULTURAL_ADAPTED_EN)
    """
    if drop_type == "ling":
        subset = filter_culture_agnostic(results)
        var_en, var_other = ORIGINAL_EN, TRANSLATED_KR
    elif drop_type == "ctx":
        subset = filter_culture_specific(results)
        var_en, var_other = ORIGINAL_EN, CULTURAL_ADAPTED_EN
    else:
        raise ValueError(drop_type)

    grouped = group_by_task_id(subset)
    diffs: List[PairedDiff] = []
    for task_id, variants in grouped.items():
        if var_en not in variants or var_other not in variants:
            continue
        en_r = variants[var_en]
        other_r = variants[var_other]
        en_trs = trs_or_none(en_r)
        other_trs = trs_or_none(other_r)
        if en_trs is None or other_trs is None:
            continue
        domain_full = en_r.risk_domain or other_r.risk_domain
        domain_short = next(
            (k for k, v in DOMAINS.items() if v == domain_full), domain_full
        )
        diffs.append(
            PairedDiff(
                model=model,
                task_id=task_id,
                domain_short=domain_short,
                drop_type=drop_type,
                diff=en_trs - other_trs,
            )
        )
    return diffs


# ----------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------

def bootstrap_mean_ci(
    values: List[float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the mean. Returns (mean, lower, upper)."""
    if not values:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    if n == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]))
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (float(arr.mean()), lower, upper)


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def aggregate_per_domain(
    diffs: List[PairedDiff],
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Aggregate-across-models view per domain.

    Returns: {domain_short: {drop_type: {mean, lo, hi, n_pairs, n_models}}}.
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dom in DOMAIN_ORDER:
        out[dom] = {}
        for drop_type in ("ling", "ctx"):
            cell = [
                d.diff for d in diffs
                if d.domain_short == dom and d.drop_type == drop_type
            ]
            n_models = len({
                d.model for d in diffs
                if d.domain_short == dom and d.drop_type == drop_type
            })
            mean, lo, hi = bootstrap_mean_ci(cell, n_boot=n_boot, seed=seed)
            out[dom][drop_type] = {
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "n_pairs": len(cell),
                "n_models": n_models,
            }
    return out


def per_model_domain_matrix(
    diffs: List[PairedDiff],
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Per-model per-domain matrix.

    Returns: {model: {domain: {drop_type: {mean, lo, hi, n_pairs}}}}.
    """
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    by_model_domain: Dict[Tuple[str, str, str], List[float]] = {}
    for d in diffs:
        by_model_domain.setdefault(
            (d.model, d.domain_short, d.drop_type), []
        ).append(d.diff)

    models = sorted({d.model for d in diffs})
    for m in models:
        out[m] = {}
        for dom in DOMAIN_ORDER:
            out[m][dom] = {}
            for drop_type in ("ling", "ctx"):
                cell = by_model_domain.get((m, dom, drop_type), [])
                mean, lo, hi = bootstrap_mean_ci(cell, n_boot=n_boot, seed=seed)
                out[m][dom][drop_type] = {
                    "mean": mean,
                    "lo": lo,
                    "hi": hi,
                    "n_pairs": len(cell),
                }
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def _fmt_pp(v: float) -> str:
    return f"{v * 100:+.2f}"


def _fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{_fmt_pp(mean)} pp [{_fmt_pp(lo)}, {_fmt_pp(hi)}]"


def write_report(
    aggregate: Dict[str, Dict[str, Dict[str, float]]],
    matrix: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    overall: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Domain-Stratified Linguistic and Contextual Drops")
    lines.append("")
    lines.append("Paired per-task TRS differences with percentile bootstrap 95% CIs (2000 resamples).")
    lines.append("")
    lines.append("- ΔLing (Culture-Agnostic): TRS(V\\_En) − TRS(V\\_Ko); positive ⇒ Korean is more conservative.")
    lines.append("- ΔCtx (Culture-Specific): TRS(V\\_En,US) − TRS(V\\_En,KR); positive ⇒ KR cultural context is more conservative.")
    lines.append("")

    # ---- Overall (sanity check, all domains pooled) ----
    lines.append("## Overall (all four domains pooled)")
    lines.append("")
    lines.append("| Drop | Mean (pp) | 95% CI (pp) | N pairs | N models |")
    lines.append("|------|----------:|:-----------:|--------:|---------:|")
    for drop_type, label in (("ling", "ΔLing"), ("ctx", "ΔCtx")):
        c = overall[drop_type]
        lines.append(
            f"| {label} | {_fmt_pp(c['mean'])} | "
            f"[{_fmt_pp(c['lo'])}, {_fmt_pp(c['hi'])}] | "
            f"{c['n_pairs']} | {c['n_models']} |"
        )
    lines.append("")

    # ---- Per-domain headline ----
    lines.append("## Per-Domain (aggregated across models)")
    lines.append("")
    lines.append("| Domain | ΔLing mean (pp) | ΔLing 95% CI | N (Ling) | ΔCtx mean (pp) | ΔCtx 95% CI | N (Ctx) |")
    lines.append("|--------|---------------:|:-------------:|---------:|---------------:|:------------:|--------:|")
    for dom in DOMAIN_ORDER:
        ling = aggregate[dom]["ling"]
        ctx = aggregate[dom]["ctx"]
        lines.append(
            f"| {DOMAIN_DISPLAY[dom]} | "
            f"{_fmt_pp(ling['mean'])} | [{_fmt_pp(ling['lo'])}, {_fmt_pp(ling['hi'])}] | "
            f"{ling['n_pairs']} | "
            f"{_fmt_pp(ctx['mean'])} | [{_fmt_pp(ctx['lo'])}, {_fmt_pp(ctx['hi'])}] | "
            f"{ctx['n_pairs']} |"
        )
    lines.append("")
    lines.append("**Caveat:** Information Leakage has only 94 source tasks; per-domain CIs there are wider and "
                 "rest on smaller paired samples (~94 × n_models).")
    lines.append("")

    # ---- Per-model x domain matrix ----
    lines.append("## Model × Domain Matrix")
    lines.append("")
    lines.append("Each cell shows mean (pp) [95% CI] / n pairs.")
    lines.append("")

    for drop_type, label in (("ling", "ΔLing (CA tasks)"), ("ctx", "ΔCtx (CS tasks)")):
        lines.append(f"### {label}")
        lines.append("")
        header = "| Model | " + " | ".join(DOMAIN_DISPLAY[d] for d in DOMAIN_ORDER) + " |"
        sep = "|-------|" + "|".join([":------:"] * len(DOMAIN_ORDER)) + "|"
        lines.append(header)
        lines.append(sep)

        # Sort models: global first then regional, alphabetical inside
        models_sorted = sorted(
            matrix.keys(),
            key=lambda m: (
                0 if is_global_model(m) else (1 if is_regional_model(m) else 2),
                get_model_display_name(m).lower(),
            ),
        )
        for m in models_sorted:
            cells = [get_model_display_name(m)]
            for dom in DOMAIN_ORDER:
                c = matrix[m][dom][drop_type]
                if c["n_pairs"] == 0:
                    cells.append("-")
                else:
                    cells.append(
                        f"{_fmt_pp(c['mean'])} [{_fmt_pp(c['lo'])}, {_fmt_pp(c['hi'])}] / n={c['n_pairs']}"
                    )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote: {output_path}")


def write_json(
    aggregate: Dict,
    matrix: Dict,
    overall: Dict,
    output_path: Path,
) -> None:
    payload = {
        "overall": overall,
        "per_domain_aggregate": aggregate,
        "per_model_per_domain": matrix,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Wrote: {output_path}")


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------

def make_figure(
    aggregate: Dict[str, Dict[str, Dict[str, float]]],
    output_name: str = "fig_domain_drops",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    domains = DOMAIN_ORDER
    x = np.arange(len(domains))
    width = 0.38

    ling_mean = np.array([aggregate[d]["ling"]["mean"] * 100 for d in domains])
    ling_lo = np.array([aggregate[d]["ling"]["lo"] * 100 for d in domains])
    ling_hi = np.array([aggregate[d]["ling"]["hi"] * 100 for d in domains])
    ctx_mean = np.array([aggregate[d]["ctx"]["mean"] * 100 for d in domains])
    ctx_lo = np.array([aggregate[d]["ctx"]["lo"] * 100 for d in domains])
    ctx_hi = np.array([aggregate[d]["ctx"]["hi"] * 100 for d in domains])

    ling_err = np.vstack([ling_mean - ling_lo, ling_hi - ling_mean])
    ctx_err = np.vstack([ctx_mean - ctx_lo, ctx_hi - ctx_mean])

    ax.bar(
        x - width / 2, ling_mean, width,
        yerr=ling_err, capsize=3,
        label=r"$\Delta_{\mathrm{ling}}$ (CA)",
        color="#9b59b6", alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    ax.bar(
        x + width / 2, ctx_mean, width,
        yerr=ctx_err, capsize=3,
        label=r"$\Delta_{\mathrm{ctx}}$ (CS)",
        color="#e67e22", alpha=0.85, edgecolor="black", linewidth=0.5,
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_DISPLAY[d] for d in domains], fontsize=9)
    ax.set_ylabel("Drop (pp, TRS)", fontsize=10)
    ax.set_title("Per-Domain Linguistic & Contextual Drops (mean ± 95% bootstrap CI)", fontsize=10)
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIGURE_DIR / f"{output_name}.{ext}"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Wrote: {path}")
    plt.close(fig)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judgements-dir", type=Path, default=DEFAULT_JUDGEMENTS_DIR)
    ap.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    ap.add_argument("--harm-classifier", type=Path, default=DEFAULT_HARM_CLASSIFIER)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 70)
    print("DOMAIN-STRATIFIED ΔLing / ΔCtx (TRS, bootstrap 95% CI)")
    print("=" * 70)
    print(f"  Judgements: {args.judgements_dir}")
    print(f"  Responses:  {args.responses_dir}")
    print(f"  Harm dims:  {args.harm_classifier}")
    print(f"  Output:     {args.output_dir}")
    print()

    print("Loading consolidated panel results...")
    data = load_all_judgements_with_responses(
        args.judgements_dir, args.responses_dir, target_models=None
    )
    if not data:
        print("ERROR: no data loaded", file=sys.stderr)
        return 1
    print(f"  Loaded {len(data)} models.")

    if args.harm_classifier.exists():
        print("Merging harm dimensions for tier weighting...")
        harm_cache = load_harm_classifications(args.harm_classifier)
        for results in data.values():
            merge_harm_dimensions(results, harm_cache)
    else:
        print(f"  WARNING: harm classifier file missing ({args.harm_classifier}); "
              "TRS will fall back to ARS where dimensions are absent.")

    print("Extracting paired diffs...")
    all_diffs: List[PairedDiff] = []
    for model, results in data.items():
        all_diffs.extend(extract_paired_diffs(results, model, "ling"))
        all_diffs.extend(extract_paired_diffs(results, model, "ctx"))
    print(f"  Total paired diffs: {len(all_diffs)}")

    # Quick sanity: domain task counts (CA + CS source)
    src_task_counts: Dict[str, int] = {}
    for d in all_diffs:
        # use CA count as a domain task count proxy across models
        # simpler: report unique task_ids per domain
        src_task_counts.setdefault(d.domain_short, set()).add(d.task_id) if False else None
    unique_tasks_per_domain: Dict[str, int] = {}
    seen: Dict[str, set] = {}
    for d in all_diffs:
        seen.setdefault(d.domain_short, set()).add(d.task_id)
    for k, v in seen.items():
        unique_tasks_per_domain[k] = len(v)
    print("  Unique tasks observed per domain (across CA+CS):")
    for dom in DOMAIN_ORDER:
        print(f"    {DOMAIN_DISPLAY[dom]}: {unique_tasks_per_domain.get(dom, 0)}")

    print("Bootstrapping per-domain aggregate...")
    aggregate = aggregate_per_domain(all_diffs, n_boot=args.n_boot, seed=args.seed)

    print("Bootstrapping overall (all domains pooled)...")
    overall: Dict[str, Dict[str, float]] = {}
    for drop_type in ("ling", "ctx"):
        vals = [d.diff for d in all_diffs if d.drop_type == drop_type]
        n_models = len({d.model for d in all_diffs if d.drop_type == drop_type})
        m, lo, hi = bootstrap_mean_ci(vals, n_boot=args.n_boot, seed=args.seed)
        overall[drop_type] = {
            "mean": m, "lo": lo, "hi": hi,
            "n_pairs": len(vals), "n_models": n_models,
        }

    print("Bootstrapping model × domain matrix...")
    matrix = per_model_domain_matrix(all_diffs, n_boot=args.n_boot, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_report(aggregate, matrix, overall, args.output_dir / "domain_drops_report.md")
    write_json(aggregate, matrix, overall, args.output_dir / "domain_drops.json")

    print("Generating figure...")
    make_figure(aggregate, output_name="fig_domain_drops")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
