#!/usr/bin/env python3
"""
Collect model responses for MultiJail baseline experiment.

Runs all target models on the MultiJail dataset across 10 languages
and 2 scenarios (unintentional + intentional with AIM).

The MultiJail.csv dataset is bundled in the multijail_baseline directory.

Usage:
    uv run python -m analysis.experiments.multijail_baseline.collect_responses \
        --outdir results/multijail_baseline/responses
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tenacity import RetryError

from rok_fortress_eval.core.api_client import call_model_once

from .config import (
    MULTIJAIL_MODELS,
    LANGUAGES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT,
    DEFAULT_CONCURRENCY,
    RESULTS_DIR,
    get_model_temperature,
)
from .data_loader import (
    MultiJailPrompt,
    load_multijail_dataset,
    get_dataset_stats,
)


@dataclass
class ResponseRecord:
    """Single response record for JSONL output."""
    
    prompt_id: str
    language: str
    language_tier: str
    scenario: str
    model_name: str
    prompt_text: str
    response_text: str
    latency_ms: float
    tokens_used: Optional[int]
    error: Optional[str]
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt_id": self.prompt_id,
            "language": self.language,
            "language_tier": self.language_tier,
            "scenario": self.scenario,
            "model_name": self.model_name,
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "error": self.error,
            "timestamp": self.timestamp,
        }


def extract_error_message(e: Exception) -> str:
    """Extract meaningful error message from exception."""
    if isinstance(e, RetryError) and hasattr(e, 'last_attempt'):
        try:
            underlying = e.last_attempt.exception()
            if underlying:
                return f"{type(underlying).__name__}: {str(underlying)}"
        except Exception:
            pass
    return f"{type(e).__name__}: {str(e)}"


def get_model_short_name(model_name: str) -> str:
    """Get short name for file naming."""
    return model_name.split("/")[-1].replace(":", "_")


class ProgressTracker:
    """Track progress across all model collections."""
    
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.errors = 0
        self.start_time = time.time()
        self.lock = asyncio.Lock()
    
    async def increment(self, success: bool = True):
        async with self.lock:
            self.completed += 1
            if not success:
                self.errors += 1
    
    def get_status(self) -> str:
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        pct = (self.completed / self.total * 100) if self.total > 0 else 0
        return (
            f"Progress: {self.completed}/{self.total} ({pct:.1f}%) | "
            f"Errors: {self.errors} | Elapsed: {elapsed_str}"
        )


async def test_model_connectivity(
    model_name: str,
    timeout: float = 30,
) -> Tuple[bool, str]:
    """Test if a model is reachable."""
    try:
        response, latency, tokens, _ = await call_model_once(
            prompt="What is 1+1? Answer with just the number.",
            model_name=model_name,
            temperature=get_model_temperature(model_name),
            max_tokens=50,
            timeout=timeout,
            reasoning_effort=None,
        )
        if response and response.strip():
            return True, f"OK ({latency:.0f}ms)"
        else:
            return False, "Empty response"
    except asyncio.TimeoutError:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, f"Error: {extract_error_message(e)[:100]}"


async def test_all_models(models: List[str]) -> Dict[str, Tuple[bool, str]]:
    """Test connectivity for all models."""
    print("\n" + "=" * 70)
    print("MODEL CONNECTIVITY TEST")
    print("=" * 70 + "\n")
    
    results = {}
    for model in models:
        success, msg = await test_model_connectivity(model)
        status = "✓" if success else "✗"
        print(f"  {status} {model:50s} | {msg}")
        results[model] = (success, msg)
        await asyncio.sleep(0.3)
    
    passed = sum(1 for s, _ in results.values() if s)
    print(f"\n  Passed: {passed}/{len(models)}")
    
    return results


async def collect_single_response(
    prompt: MultiJailPrompt,
    model_name: str,
    timeout: float,
) -> ResponseRecord:
    """Collect response for a single prompt from a model."""
    timestamp = datetime.now().isoformat()

    response_text = ""
    latency_ms = 0.0
    tokens_used = None
    error = None

    try:
        response_text, latency_ms, tokens_used, _ = await call_model_once(
            prompt=prompt.prompt_text,
            model_name=model_name,
            temperature=get_model_temperature(model_name),
            max_tokens=DEFAULT_MAX_TOKENS,
            timeout=timeout,
            reasoning_effort=None,  # No reasoning for baseline
        )
    except Exception as e:
        error = extract_error_message(e)
    
    return ResponseRecord(
        prompt_id=prompt.prompt_id,
        language=prompt.language,
        language_tier=prompt.language_tier,
        scenario=prompt.scenario,
        model_name=model_name,
        prompt_text=prompt.prompt_text,
        response_text=response_text,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        error=error,
        timestamp=timestamp,
    )


async def collect_model_responses(
    model_name: str,
    prompts: List[MultiJailPrompt],
    output_dir: Path,
    concurrency: int,
    timeout: float,
    progress: ProgressTracker,
    skip_keys: Set[Tuple[str, str, str]],
):
    """Collect responses for a single model."""
    short_name = get_model_short_name(model_name)
    output_file = output_dir / f"responses_{short_name}.jsonl"

    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()

    async def process_prompt(prompt: MultiJailPrompt):
        # Check if already completed
        key = (prompt.prompt_id, prompt.language, prompt.scenario)
        if key in skip_keys:
            return None

        async with semaphore:
            record = await collect_single_response(prompt, model_name, timeout)
            
            # Write immediately
            async with file_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            
            # Update progress
            success = record.error is None
            await progress.increment(success)
            
            return record
    
    tasks = [process_prompt(p) for p in prompts]
    await asyncio.gather(*tasks)
    
    print(f"  ✓ {model_name} completed -> {output_file.name}")


async def run_collection(
    models: List[str],
    prompts: List[MultiJailPrompt],
    output_dir: Path,
    concurrency: int,
    timeout: float,
    skip_keys_by_model: Dict[str, Set[Tuple[str, str, str]]],
):
    """Run collection for all models in parallel."""
    print(f"\n{'=' * 70}")
    print("COLLECTING RESPONSES")
    print(f"{'=' * 70}")
    print(f"  Models: {len(models)}")
    print(f"  Prompts per model: {len(prompts)}")
    print(f"  Concurrency per model: {concurrency}")
    print(f"  Output: {output_dir}\n")
    
    # Calculate total remaining prompts across all models
    total_skipped = sum(len(skip_keys_by_model.get(m, set())) for m in models)
    total = len(models) * len(prompts) - total_skipped
    progress = ProgressTracker(total)
    
    # Show per-model resume status
    for model in models:
        model_skip_count = len(skip_keys_by_model.get(model, set()))
        if model_skip_count > 0:
            print(f"  Resume: {model} - skipping {model_skip_count} completed prompts")
    
    # Progress reporter
    async def report_progress():
        while True:
            await asyncio.sleep(15)
            print(f"  {progress.get_status()}")
    
    progress_task = asyncio.create_task(report_progress())

    try:
        model_tasks = [
            collect_model_responses(
                model_name=model,
                prompts=prompts,
                output_dir=output_dir,
                concurrency=concurrency,
                timeout=timeout,
                progress=progress,
                skip_keys=skip_keys_by_model.get(model, set()),
            )
            for model in models
        ]
        await asyncio.gather(*model_tasks)
    finally:
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

    print(f"\n  {progress.get_status()}")


def load_skip_keys_per_model(output_dir: Path) -> Dict[str, Set[Tuple[str, str, str]]]:
    """
    Load already-completed keys for resume support, organized per model.
    Only skips SUCCESSFUL records (no error). Records with errors will be retried.
    
    Returns:
        Dict mapping model_name -> set of (prompt_id, language, scenario) tuples
    """
    skip_keys_by_model: Dict[str, Set[Tuple[str, str, str]]] = {}
    
    for jsonl_file in output_dir.glob("responses_*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    
                    # Only skip successful records (no error)
                    # Records with errors should be retried
                    error = record.get("error")
                    if error:  # Has an error - don't skip, will retry
                        continue
                    
                    model_name = record["model_name"]
                    key = (
                        record["prompt_id"],
                        record["language"],
                        record["scenario"],
                    )
                    
                    if model_name not in skip_keys_by_model:
                        skip_keys_by_model[model_name] = set()
                    skip_keys_by_model[model_name].add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
    
    return skip_keys_by_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect model responses for MultiJail baseline experiment",
    )
    
    parser.add_argument(
        "--csv-path", type=str, default=None,
        help="Path to MultiJail.csv (default: bundled in multijail_baseline/)",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory for responses",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated list of models (default: all 4 target models)",
    )
    parser.add_argument(
        "--languages", type=str, default=None,
        help="Comma-separated list of languages (default: all 10)",
    )
    parser.add_argument(
        "--scenario", type=str, choices=["unintentional", "intentional", "both"],
        default="both", help="Which scenario(s) to run",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Concurrent requests per model (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--skip-connectivity-test", action="store_true",
        help="Skip the initial model connectivity test",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output files",
    )

    return parser.parse_args()


async def main():
    args = parse_args()
    
    # Parse models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = MULTIJAIL_MODELS
    
    # Parse languages
    if args.languages:
        languages = [l.strip() for l in args.languages.split(",")]
    else:
        languages = LANGUAGES
    
    # Setup output directory
    if args.outdir:
        output_dir = Path(args.outdir)
    else:
        output_dir = RESULTS_DIR / "responses"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print banner
    print("=" * 70)
    print("MULTIJAIL BASELINE - RESPONSE COLLECTION")
    print("=" * 70)
    print(f"  CSV: {args.csv_path or 'bundled MultiJail.csv'}")
    print(f"  Output: {output_dir}")
    print(f"  Models: {len(models)}")
    print(f"  Languages: {len(languages)}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Resume: {args.resume}")
    print(f"  Concurrency: {args.concurrency}")
    print("=" * 70)
    
    # Load prompts
    print("\nLoading MultiJail dataset...")
    include_intentional = args.scenario in ["intentional", "both"]
    
    try:
        prompts = load_multijail_dataset(
            csv_path=Path(args.csv_path) if args.csv_path else None,
            languages=languages,
            include_intentional=include_intentional,
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    # Filter by scenario if needed
    if args.scenario == "unintentional":
        prompts = [p for p in prompts if p.scenario == "unintentional"]
    elif args.scenario == "intentional":
        prompts = [p for p in prompts if p.scenario == "intentional"]
    
    # Print stats
    stats = get_dataset_stats(prompts)
    print(f"  Loaded {stats['total_prompts']} prompts")
    print(f"  By tier: HRL={stats['by_tier']['HRL']}, MRL={stats['by_tier']['MRL']}, LRL={stats['by_tier']['LRL']}")
    print(f"  By scenario: {stats['by_scenario']}")
    
    # Load skip keys if resuming (per-model)
    skip_keys_by_model: Dict[str, Set[Tuple[str, str, str]]] = {}
    if args.resume:
        skip_keys_by_model = load_skip_keys_per_model(output_dir)
        total_skipped = sum(len(keys) for keys in skip_keys_by_model.values())
        if total_skipped:
            print(f"  Resume: found {total_skipped} completed records across {len(skip_keys_by_model)} models")
    
    # Connectivity test
    if not args.skip_connectivity_test:
        test_results = await test_all_models(models)
        
        # Save test results
        with open(output_dir / "connectivity_test.json", "w") as f:
            json.dump(
                {m: {"success": s, "message": msg} for m, (s, msg) in test_results.items()},
                f, indent=2
            )
        
        failed = [m for m, (s, _) in test_results.items() if not s]
        if failed:
            print(f"\n⚠️  {len(failed)} models failed connectivity test")
            if sys.stdin.isatty():
                response = input("\nContinue anyway? [y/N]: ").strip().lower()
                if response != "y":
                    print("Aborted.")
                    return 1
            else:
                print("  Non-interactive mode: continuing...")
    
    # Run collection
    await run_collection(
        models=models,
        prompts=prompts,
        output_dir=output_dir,
        concurrency=args.concurrency,
        timeout=args.timeout,
        skip_keys_by_model=skip_keys_by_model,
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    
    output_files = sorted(output_dir.glob("responses_*.jsonl"))
    print(f"\n  Output files ({len(output_files)} total):")
    for f in output_files:
        line_count = sum(1 for _ in open(f, "r", encoding="utf-8"))
        print(f"    - {f.name} ({line_count} records)")
    
    print("\n  Next step: Run judge_responses.py to evaluate safety")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
