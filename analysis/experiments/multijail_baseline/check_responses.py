#!/usr/bin/env python3
"""
Diagnostic script to check the status of collected responses.

Usage:
    python -m analysis.experiments.multijail_baseline.check_responses
    python -m analysis.experiments.multijail_baseline.check_responses --verbose
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

from .config import MULTIJAIL_MODELS, RESULTS_DIR, NUM_PROMPTS, SCENARIOS, LANGUAGES


def analyze_response_file(filepath: Path) -> Dict:
    """Analyze a single response file and return statistics."""
    stats = {
        "total": 0,
        "success": 0,
        "errors": 0,
        "empty": 0,
        "error_types": Counter(),
        "sample_errors": [],
    }
    
    if not filepath.exists():
        return stats
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                stats["total"] += 1
                
                error = record.get("error")
                response_text = record.get("response_text", "")
                
                if error:
                    stats["errors"] += 1
                    # Categorize error type
                    if "429" in str(error) or "rate" in str(error).lower():
                        stats["error_types"]["rate_limit"] += 1
                    elif "timeout" in str(error).lower():
                        stats["error_types"]["timeout"] += 1
                    elif "400" in str(error):
                        stats["error_types"]["bad_request"] += 1
                    elif "500" in str(error) or "502" in str(error) or "503" in str(error):
                        stats["error_types"]["server_error"] += 1
                    else:
                        stats["error_types"]["other"] += 1
                    
                    # Keep sample errors for debugging
                    if len(stats["sample_errors"]) < 3:
                        stats["sample_errors"].append(str(error)[:200])
                elif not response_text.strip():
                    stats["empty"] += 1
                else:
                    stats["success"] += 1
                    
            except json.JSONDecodeError:
                continue
    
    return stats


def get_model_filename(model: str) -> str:
    """Get the expected filename for a model's responses."""
    model_name = model.split("/")[-1].replace(":", "_")
    return f"responses_{model_name}.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Check response collection status")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed error info")
    parser.add_argument("--responses-dir", type=Path, 
                        default=RESULTS_DIR / "responses",
                        help="Directory containing response files")
    args = parser.parse_args()
    
    responses_dir = args.responses_dir
    expected_per_model = NUM_PROMPTS * len(SCENARIOS) * len(LANGUAGES)  # 315 * 2 * 10 = 6300
    
    print("=" * 80)
    print("MULTIJAIL RESPONSE COLLECTION STATUS")
    print("=" * 80)
    print(f"\nResponses directory: {responses_dir}")
    print(f"Expected responses per model: {expected_per_model}")
    print(f"Models configured: {len(MULTIJAIL_MODELS)}")
    print()
    
    # Summary table header
    print(f"{'Model':<50} {'Total':>7} {'OK':>7} {'Err':>6} {'Empty':>6} {'%':>6}")
    print("-" * 80)
    
    total_all = 0
    success_all = 0
    errors_all = 0
    empty_all = 0
    
    model_stats = {}
    
    for model in MULTIJAIL_MODELS:
        filename = get_model_filename(model)
        filepath = responses_dir / filename
        
        stats = analyze_response_file(filepath)
        model_stats[model] = stats
        
        total_all += stats["total"]
        success_all += stats["success"]
        errors_all += stats["errors"]
        empty_all += stats["empty"]
        
        pct = (stats["success"] / expected_per_model * 100) if expected_per_model > 0 else 0
        status = "✓" if stats["success"] >= expected_per_model else ""
        
        model_short = model.split("/")[-1][:48]
        print(f"{model_short:<50} {stats['total']:>7} {stats['success']:>7} {stats['errors']:>6} {stats['empty']:>6} {pct:>5.1f}% {status}")
    
    print("-" * 80)
    total_expected = expected_per_model * len(MULTIJAIL_MODELS)
    overall_pct = (success_all / total_expected * 100) if total_expected > 0 else 0
    print(f"{'TOTAL':<50} {total_all:>7} {success_all:>7} {errors_all:>6} {empty_all:>6} {overall_pct:>5.1f}%")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total records:    {total_all:,}")
    print(f"  Successful:       {success_all:,} ({success_all/total_all*100:.1f}%)" if total_all > 0 else "  Successful:       0")
    print(f"  Errors:           {errors_all:,} ({errors_all/total_all*100:.1f}%)" if total_all > 0 else "  Errors:           0")
    print(f"  Empty responses:  {empty_all:,} ({empty_all/total_all*100:.1f}%)" if total_all > 0 else "  Empty responses:  0")
    print()
    print(f"  Progress: {success_all:,} / {total_expected:,} ({overall_pct:.1f}%)")
    remaining = total_expected - success_all
    print(f"  Remaining: {remaining:,} successful responses needed")
    
    # Show error breakdown if verbose
    if args.verbose:
        print()
        print("=" * 80)
        print("ERROR DETAILS BY MODEL")
        print("=" * 80)
        
        for model in MULTIJAIL_MODELS:
            stats = model_stats[model]
            if stats["errors"] > 0 or stats["empty"] > 0:
                print(f"\n{model}:")
                if stats["error_types"]:
                    print(f"  Error types: {dict(stats['error_types'])}")
                if stats["sample_errors"]:
                    print(f"  Sample errors:")
                    for err in stats["sample_errors"]:
                        print(f"    - {err}...")
    
    # Check for files not in config
    print()
    print("=" * 80)
    print("FILES CHECK")
    print("=" * 80)
    expected_files = {get_model_filename(m) for m in MULTIJAIL_MODELS}
    actual_files = {f.name for f in responses_dir.glob("responses_*.jsonl")} if responses_dir.exists() else set()
    
    extra_files = actual_files - expected_files
    missing_files = expected_files - actual_files
    
    if extra_files:
        print(f"\n  Extra files (not in config): {len(extra_files)}")
        for f in sorted(extra_files):
            print(f"    - {f}")
    
    if missing_files:
        print(f"\n  Missing files: {len(missing_files)}")
        for f in sorted(missing_files):
            print(f"    - {f}")
    
    if not extra_files and not missing_files:
        print("\n  All expected files present ✓")


if __name__ == "__main__":
    main()
