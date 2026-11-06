#!/usr/bin/env python3
"""
Analyze MultiJail baseline experiment results.

Computes Unsafe Rate (URS) metrics by language, tier, and scenario,
then compares with original paper results.

Supports both single-judge and ensemble (majority voting) judgements:
- For ensemble: reads from judgements_ensemble/ and uses final_verdict
- For single-judge: reads from judgements_* and uses verdict

Usage:
    # Analyze ensemble results (default)
    python -m analysis.experiments.multijail_baseline.analyze
    
    # Analyze specific judge
    python -m analysis.experiments.multijail_baseline.analyze \
        --judgements-dir results/multijail_baseline/judgements_o4
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    LANGUAGES,
    LANGUAGE_TIERS,
    LANGUAGE_NAMES,
    MULTIJAIL_MODELS,
    NUM_PROMPTS,
    RESULTS_DIR,
)


@dataclass
class URSMetrics:
    """Unsafe Rate Statistics for a group of judgements."""
    
    total: int = 0
    safe: int = 0
    unsafe: int = 0
    invalid: int = 0
    
    @property
    def urs(self) -> float:
        """Unsafe Rate = unsafe / total (excluding invalid)."""
        valid = self.total - self.invalid
        if valid == 0:
            return 0.0
        return self.unsafe / valid * 100
    
    @property
    def safety_rate(self) -> float:
        """Safety Rate = safe / total (excluding invalid)."""
        valid = self.total - self.invalid
        if valid == 0:
            return 0.0
        return self.safe / valid * 100
    
    @property
    def invalid_rate(self) -> float:
        """Invalid Rate = invalid / total."""
        if self.total == 0:
            return 0.0
        return self.invalid / self.total * 100
    
    def to_dict(self) -> Dict:
        return {
            "total": self.total,
            "safe": self.safe,
            "unsafe": self.unsafe,
            "invalid": self.invalid,
            "urs": round(self.urs, 2),
            "safety_rate": round(self.safety_rate, 2),
            "invalid_rate": round(self.invalid_rate, 2),
        }


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    
    # Overall metrics
    overall: URSMetrics = field(default_factory=URSMetrics)
    
    # By model
    by_model: Dict[str, URSMetrics] = field(default_factory=dict)
    
    # By language
    by_language: Dict[str, URSMetrics] = field(default_factory=dict)
    
    # By tier
    by_tier: Dict[str, URSMetrics] = field(default_factory=dict)
    
    # By scenario
    by_scenario: Dict[str, URSMetrics] = field(default_factory=dict)
    
    # By model x language
    by_model_language: Dict[str, Dict[str, URSMetrics]] = field(default_factory=dict)
    
    # By model x tier
    by_model_tier: Dict[str, Dict[str, URSMetrics]] = field(default_factory=dict)
    
    # By model x scenario
    by_model_scenario: Dict[str, Dict[str, URSMetrics]] = field(default_factory=dict)
    
    # By tier x scenario (for scenario-specific tier plots)
    by_tier_scenario: Dict[str, Dict[str, URSMetrics]] = field(default_factory=dict)
    
    # By language x scenario (for scenario-specific language plots)
    by_language_scenario: Dict[str, Dict[str, URSMetrics]] = field(default_factory=dict)
    
    # By model x language x scenario (for scenario-specific heatmaps)
    by_model_language_scenario: Dict[str, Dict[str, Dict[str, URSMetrics]]] = field(default_factory=dict)
    
    # By model x tier x scenario (for scenario-specific tier comparison)
    by_model_tier_scenario: Dict[str, Dict[str, Dict[str, URSMetrics]]] = field(default_factory=dict)
    
    # Gap metrics (LRL URS - HRL URS)
    gap_by_model: Dict[str, float] = field(default_factory=dict)
    
    # Gap by scenario
    gap_by_model_scenario: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall": self.overall.to_dict(),
            "by_model": {k: v.to_dict() for k, v in self.by_model.items()},
            "by_language": {k: v.to_dict() for k, v in self.by_language.items()},
            "by_tier": {k: v.to_dict() for k, v in self.by_tier.items()},
            "by_scenario": {k: v.to_dict() for k, v in self.by_scenario.items()},
            "by_model_language": {
                m: {l: v.to_dict() for l, v in langs.items()}
                for m, langs in self.by_model_language.items()
            },
            "by_model_tier": {
                m: {t: v.to_dict() for t, v in tiers.items()}
                for m, tiers in self.by_model_tier.items()
            },
            "by_model_scenario": {
                m: {s: v.to_dict() for s, v in scenarios.items()}
                for m, scenarios in self.by_model_scenario.items()
            },
            "by_tier_scenario": {
                t: {s: v.to_dict() for s, v in scenarios.items()}
                for t, scenarios in self.by_tier_scenario.items()
            },
            "by_language_scenario": {
                l: {s: v.to_dict() for s, v in scenarios.items()}
                for l, scenarios in self.by_language_scenario.items()
            },
            "by_model_language_scenario": {
                m: {l: {s: v.to_dict() for s, v in scenarios.items()} for l, scenarios in langs.items()}
                for m, langs in self.by_model_language_scenario.items()
            },
            "by_model_tier_scenario": {
                m: {t: {s: v.to_dict() for s, v in scenarios.items()} for t, scenarios in tiers.items()}
                for m, tiers in self.by_model_tier_scenario.items()
            },
            "gap_by_model": self.gap_by_model,
            "gap_by_model_scenario": self.gap_by_model_scenario,
        }


def load_judgements(judgements_dir: Path) -> List[Dict]:
    """Load all judgement records from JSONL files."""
    records = []
    
    # Try loading merged ensemble file first
    merged_file = judgements_dir / "merged_judgements.jsonl"
    if merged_file.exists():
        with open(merged_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue
        return records
    
    # Fall back to individual judgement files
    for jsonl_file in judgements_dir.glob("judgements_*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue
    
    return records


def deduplicate_judgements(records: List[Dict]) -> List[Dict]:
    """
    Deduplicate judgements, keeping the most recent record for each unique key.
    
    Key: (prompt_id, language, scenario, model_name)
    
    For duplicates, prefers non-invalid verdicts over invalid ones.
    """
    from typing import Tuple
    
    # Group by key
    by_key: Dict[Tuple[str, str, str, str], List[Dict]] = {}
    
    for record in records:
        key = (
            record.get("prompt_id", ""),
            record.get("language", ""),
            record.get("scenario", ""),
            record.get("model_name", ""),
        )
        if key not in by_key:
            by_key[key] = []
        by_key[key].append(record)
    
    # Select best record for each key
    deduplicated = []
    duplicates_removed = 0
    
    for key, group in by_key.items():
        if len(group) == 1:
            deduplicated.append(group[0])
        else:
            # Multiple records - prefer non-invalid verdicts
            # Check both final_verdict (ensemble) and verdict (single-judge)
            def get_verdict(r):
                return r.get("final_verdict") or r.get("verdict")
            
            valid = [r for r in group if get_verdict(r) != "invalid"]
            
            if valid:
                # Take the last valid record (most recent)
                deduplicated.append(valid[-1])
            else:
                # All invalid - take the last one
                deduplicated.append(group[-1])
            
            duplicates_removed += len(group) - 1
    
    if duplicates_removed > 0:
        print(f"  Deduplicated: removed {duplicates_removed} duplicate records")
    
    return deduplicated


def compute_agreement_stats(records: List[Dict]) -> Dict:
    """Compute agreement statistics for ensemble judgements."""
    from collections import Counter
    
    agreement_counts = Counter()
    has_agreement = False
    
    for record in records:
        agreement = record.get("agreement")
        if agreement:
            has_agreement = True
            agreement_counts[agreement] += 1
    
    if not has_agreement:
        return {}
    
    total = sum(agreement_counts.values())
    stats = {
        "distribution": dict(agreement_counts),
        "total": total,
    }
    
    # Compute full agreement rate
    full_agreement = agreement_counts.get("3/3", 0)
    stats["full_agreement_rate"] = round(full_agreement / total * 100, 2) if total > 0 else 0
    
    return stats


def update_metrics(metrics: URSMetrics, verdict: str):
    """Update metrics with a verdict."""
    metrics.total += 1
    if verdict == "safe":
        metrics.safe += 1
    elif verdict == "unsafe":
        metrics.unsafe += 1
    elif verdict == "invalid":
        metrics.invalid += 1


def analyze_judgements(records: List[Dict]) -> AnalysisResult:
    """Analyze all judgement records and compute metrics."""
    result = AnalysisResult()
    
    # Initialize nested dicts
    for model in set(r["model_name"] for r in records):
        result.by_model[model] = URSMetrics()
        result.by_model_language[model] = {}
        result.by_model_tier[model] = {}
        result.by_model_scenario[model] = {}
        result.by_model_language_scenario[model] = {}
        result.by_model_tier_scenario[model] = {}
    
    for lang in LANGUAGES:
        result.by_language[lang] = URSMetrics()
        result.by_language_scenario[lang] = {}
    
    for tier in ["HRL", "MRL", "LRL"]:
        result.by_tier[tier] = URSMetrics()
        result.by_tier_scenario[tier] = {}
    
    for scenario in ["unintentional", "intentional"]:
        result.by_scenario[scenario] = URSMetrics()
    
    # Process each record
    for record in records:
        # Use final_verdict for ensemble, fall back to verdict for single-judge
        verdict = record.get("final_verdict") or record.get("verdict")
        model = record["model_name"]
        language = record["language"]
        tier = record.get("language_tier", LANGUAGE_TIERS.get(language, "unknown"))
        scenario = record["scenario"]
        
        # Overall
        update_metrics(result.overall, verdict)
        
        # By model
        update_metrics(result.by_model[model], verdict)
        
        # By language
        if language in result.by_language:
            update_metrics(result.by_language[language], verdict)
        
        # By tier
        if tier in result.by_tier:
            update_metrics(result.by_tier[tier], verdict)
        
        # By scenario
        if scenario in result.by_scenario:
            update_metrics(result.by_scenario[scenario], verdict)
        
        # By model x language
        if language not in result.by_model_language[model]:
            result.by_model_language[model][language] = URSMetrics()
        update_metrics(result.by_model_language[model][language], verdict)
        
        # By model x tier
        if tier not in result.by_model_tier[model]:
            result.by_model_tier[model][tier] = URSMetrics()
        update_metrics(result.by_model_tier[model][tier], verdict)
        
        # By model x scenario
        if scenario not in result.by_model_scenario[model]:
            result.by_model_scenario[model][scenario] = URSMetrics()
        update_metrics(result.by_model_scenario[model][scenario], verdict)
        
        # By tier x scenario
        if tier in result.by_tier_scenario:
            if scenario not in result.by_tier_scenario[tier]:
                result.by_tier_scenario[tier][scenario] = URSMetrics()
            update_metrics(result.by_tier_scenario[tier][scenario], verdict)
        
        # By language x scenario
        if language in result.by_language_scenario:
            if scenario not in result.by_language_scenario[language]:
                result.by_language_scenario[language][scenario] = URSMetrics()
            update_metrics(result.by_language_scenario[language][scenario], verdict)
        
        # By model x language x scenario
        if language not in result.by_model_language_scenario[model]:
            result.by_model_language_scenario[model][language] = {}
        if scenario not in result.by_model_language_scenario[model][language]:
            result.by_model_language_scenario[model][language][scenario] = URSMetrics()
        update_metrics(result.by_model_language_scenario[model][language][scenario], verdict)
        
        # By model x tier x scenario
        if tier not in result.by_model_tier_scenario[model]:
            result.by_model_tier_scenario[model][tier] = {}
        if scenario not in result.by_model_tier_scenario[model][tier]:
            result.by_model_tier_scenario[model][tier][scenario] = URSMetrics()
        update_metrics(result.by_model_tier_scenario[model][tier][scenario], verdict)
    
    # Compute gap metrics (LRL URS - HRL URS)
    for model in result.by_model_tier.keys():
        tiers = result.by_model_tier[model]
        if "LRL" in tiers and "HRL" in tiers:
            result.gap_by_model[model] = round(tiers["LRL"].urs - tiers["HRL"].urs, 2)
    
    # Compute gap by scenario
    for model in result.by_model_tier_scenario.keys():
        result.gap_by_model_scenario[model] = {}
        for scenario in ["unintentional", "intentional"]:
            tiers = result.by_model_tier_scenario[model]
            if "LRL" in tiers and "HRL" in tiers:
                lrl = tiers["LRL"].get(scenario, URSMetrics())
                hrl = tiers["HRL"].get(scenario, URSMetrics())
                if isinstance(lrl, dict):
                    lrl_urs = lrl.get("urs", 0) if isinstance(lrl, dict) else lrl.urs
                    hrl_urs = hrl.get("urs", 0) if isinstance(hrl, dict) else hrl.urs
                else:
                    lrl_urs = lrl.urs
                    hrl_urs = hrl.urs
                result.gap_by_model_scenario[model][scenario] = round(lrl_urs - hrl_urs, 2)
    
    return result


def generate_report(result: AnalysisResult) -> str:
    """Generate a human-readable markdown report."""
    lines = [
        "# MultiJail Baseline Experiment Results",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Overview",
        "",
        f"- Total judgements: {result.overall.total}",
        f"- Overall URS: {result.overall.urs:.2f}%",
        f"- Overall Safety Rate: {result.overall.safety_rate:.2f}%",
        f"- Invalid Rate: {result.overall.invalid_rate:.2f}%",
        "",
        "## URS by Language Tier",
        "",
        "| Tier | URS (%) | Safe | Unsafe | Invalid | Total |",
        "|------|---------|------|--------|---------|-------|",
    ]
    
    for tier in ["HRL", "MRL", "LRL"]:
        if tier in result.by_tier:
            m = result.by_tier[tier]
            lines.append(f"| {tier} | {m.urs:.2f} | {m.safe} | {m.unsafe} | {m.invalid} | {m.total} |")
    
    lines.extend([
        "",
        "## URS by Scenario",
        "",
        "| Scenario | URS (%) | Safe | Unsafe | Invalid |",
        "|----------|---------|------|--------|---------|",
    ])
    
    for scenario in ["unintentional", "intentional"]:
        if scenario in result.by_scenario:
            m = result.by_scenario[scenario]
            lines.append(f"| {scenario.title()} | {m.urs:.2f} | {m.safe} | {m.unsafe} | {m.invalid} |")
    
    lines.extend([
        "",
        "## URS by Model",
        "",
        "| Model | URS (%) | Gap (LRL-HRL) | Safe | Unsafe | Invalid |",
        "|-------|---------|---------------|------|--------|---------|",
    ])
    
    for model, metrics in sorted(result.by_model.items()):
        gap = result.gap_by_model.get(model, "N/A")
        if isinstance(gap, float):
            gap = f"{gap:+.2f}"
        lines.append(f"| {model} | {metrics.urs:.2f} | {gap} | {metrics.safe} | {metrics.unsafe} | {metrics.invalid} |")
    
    lines.extend([
        "",
        "## URS by Model x Language",
        "",
        "| Model | " + " | ".join(LANGUAGES) + " |",
        "|-------|" + "|".join(["------" for _ in LANGUAGES]) + "|",
    ])
    
    for model in sorted(result.by_model_language.keys()):
        langs = result.by_model_language[model]
        row = [f"{langs.get(l, URSMetrics()).urs:.1f}" for l in LANGUAGES]
        lines.append(f"| {model.split('/')[-1]} | " + " | ".join(row) + " |")
    
    lines.extend([
        "",
        "## Gap Analysis (LRL URS - HRL URS)",
        "",
        "A positive gap indicates higher unsafe rate in low-resource languages.",
        "",
    ])
    
    for model, gap in sorted(result.gap_by_model.items(), key=lambda x: x[1], reverse=True):
        direction = "↑" if gap > 0 else "↓" if gap < 0 else "="
        lines.append(f"- **{model}**: {gap:+.2f}% {direction}")
    
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze MultiJail baseline experiment results",
    )
    
    parser.add_argument(
        "--judgements-dir", type=str, default=None,
        help="Directory containing judgement JSONL files",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory for analysis results",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup directories
    if args.judgements_dir:
        judgements_dir = Path(args.judgements_dir)
    else:
        # Default to ensemble directory (FORTRESS convention)
        judgements_dir = RESULTS_DIR / "judgements" / "ensemble"
    
    if args.outdir:
        output_dir = Path(args.outdir)
    else:
        output_dir = RESULTS_DIR / "analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print banner
    print("=" * 70)
    print("MULTIJAIL BASELINE - ANALYSIS")
    print("=" * 70)
    print(f"  Judgements dir: {judgements_dir}")
    print(f"  Output dir: {output_dir}")
    print("=" * 70)
    
    # Load judgements
    print("\nLoading judgements...")
    records = load_judgements(judgements_dir)
    
    if not records:
        print(f"\n❌ No judgement files found in {judgements_dir}")
        print("  Run judge_responses.py first")
        return 1
    
    print(f"  Loaded {len(records)} judgement records")
    
    # Deduplicate (handles multiple worker runs with overlapping records)
    records = deduplicate_judgements(records)
    print(f"  After deduplication: {len(records)} unique records")
    
    # Check if this is ensemble data
    is_ensemble = any(r.get("final_verdict") for r in records)
    if is_ensemble:
        print("  Mode: Ensemble (majority voting)")
    else:
        print("  Mode: Single judge")
    
    # Analyze
    print("\nAnalyzing...")
    result = analyze_judgements(records)
    
    # Compute agreement stats for ensemble
    agreement_stats = compute_agreement_stats(records) if is_ensemble else {}
    
    # Save JSON result
    result_dict = result.to_dict()
    if agreement_stats:
        result_dict["agreement_stats"] = agreement_stats
    result_dict["is_ensemble"] = is_ensemble
    
    result_path = output_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {result_path}")
    
    # Generate and save report
    report = generate_report(result)
    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Data quality summary
    print(f"\n  DATA QUALITY:")
    print(f"    Total records: {result.overall.total}")
    print(f"    Valid (safe+unsafe): {result.overall.safe + result.overall.unsafe}")
    print(f"    Invalid: {result.overall.invalid} ({result.overall.invalid_rate:.2f}%)")
    print(f"    Error rate: {result.overall.invalid_rate:.2f}%")
    
    # Per-model data quality
    print(f"\n  DATA QUALITY BY MODEL:")
    for model, m in sorted(result.by_model.items()):
        valid = m.safe + m.unsafe
        print(f"    {model.split('/')[-1]}: {valid} valid, {m.invalid} invalid ({m.invalid_rate:.1f}%)")
    
    # Per-scenario data quality
    print(f"\n  DATA QUALITY BY SCENARIO:")
    for scenario, m in result.by_scenario.items():
        valid = m.safe + m.unsafe
        print(f"    {scenario}: {valid} valid, {m.invalid} invalid ({m.invalid_rate:.1f}%)")
    
    print(f"\n  RESULTS:")
    print(f"    Overall URS: {result.overall.urs:.2f}%")
    print(f"\n  URS by Tier:")
    for tier in ["HRL", "MRL", "LRL"]:
        if tier in result.by_tier:
            m = result.by_tier[tier]
            valid = m.safe + m.unsafe
            print(f"    {tier}: {m.urs:.2f}% (n={valid})")
    
    print(f"\n  Gap (LRL - HRL) by Model:")
    for model, gap in sorted(result.gap_by_model.items(), key=lambda x: x[1], reverse=True):
        print(f"    {model}: {gap:+.2f}%")
    
    # Print agreement stats for ensemble
    if agreement_stats:
        print(f"\n  JUDGE AGREEMENT (ensemble):")
        print(f"    Full agreement (3/3): {agreement_stats.get('full_agreement_rate', 0):.1f}%")
        for agreement, count in sorted(agreement_stats.get("distribution", {}).items(), reverse=True):
            pct = count / agreement_stats.get("total", 1) * 100
            print(f"    {agreement}: {count} ({pct:.1f}%)")
    
    print("\n" + "=" * 70)
    print(f"  Full report: {report_path}")
    print("  Next step: Run plots.py to generate visualizations")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
