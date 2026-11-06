#!/usr/bin/env python3
"""
Run safety judgement on collected MultiJail responses.

Supports multiple judge models for ensemble voting:
- openai/o4-mini (default)
- gemini/gemini-3-pro-preview
- openai/gpt-5.2

Following FORTRESS convention, outputs are organized as:
    judgements/{judge_folder}/judgements_{judge}_on_{response_model}.jsonl

Use merge_judgements.py to combine results with majority voting.

Usage:
    # Run with default o4-mini judge
    python -m analysis.experiments.multijail_baseline.judge_responses
    
    # Run with specific judge
    python -m analysis.experiments.multijail_baseline.judge_responses \
        --judge-model gemini/gemini-3-pro-preview
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from .config import (
    JUDGE_MODEL,
    JUDGE_MODELS,
    DEFAULT_TIMEOUT,
    DEFAULT_CONCURRENCY,
    RESULTS_DIR,
    get_judge_folder_name,
    get_judgement_filename,
)
from .judge import evaluate_safety, SafetyVerdict


class ProgressTracker:
    """Track judging progress."""
    
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.verdicts = {"safe": 0, "unsafe": 0, "invalid": 0}
        self.start_time = time.time()
        self.lock = asyncio.Lock()
    
    async def increment(self, verdict: str):
        async with self.lock:
            self.completed += 1
            if verdict in self.verdicts:
                self.verdicts[verdict] += 1
    
    def get_status(self) -> str:
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        pct = (self.completed / self.total * 100) if self.total > 0 else 0
        
        return (
            f"Progress: {self.completed}/{self.total} ({pct:.1f}%) | "
            f"Safe: {self.verdicts['safe']} | Unsafe: {self.verdicts['unsafe']} | "
            f"Invalid: {self.verdicts['invalid']} | Elapsed: {elapsed_str}"
        )


def load_responses(responses_dir: Path) -> List[Dict]:
    """Load all response records from JSONL files."""
    records = []
    
    for jsonl_file in responses_dir.glob("responses_*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue
    
    return records


def deduplicate_responses(records: List[Dict]) -> List[Dict]:
    """
    Deduplicate responses, keeping successful records over errors.
    
    For each (prompt_id, language, scenario, model_name) key:
    - If there's a successful record (no error), use that
    - If all records have errors, use the most recent one
    
    Returns:
        Deduplicated list of records
    """
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
            # Multiple records - prefer successful ones
            successful = [r for r in group if not r.get("error")]
            
            if successful:
                # Take the last successful record (most recent)
                deduplicated.append(successful[-1])
            else:
                # All have errors - take the last one
                deduplicated.append(group[-1])
            
            duplicates_removed += len(group) - 1
    
    if duplicates_removed > 0:
        print(f"  Deduplicated: removed {duplicates_removed} duplicate records")
    
    return deduplicated


def load_completed_keys(output_dir: Path) -> Set[Tuple[str, str, str, str]]:
    """Load already-judged keys for resume support (FORTRESS convention)."""
    skip_keys = set()
    
    # Load from all JSONL files in the judge folder
    for jsonl_file in output_dir.glob("judgements_*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    key = (
                        record["prompt_id"],
                        record["language"],
                        record["scenario"],
                        record["model_name"],
                    )
                    skip_keys.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
    
    return skip_keys


def get_output_files_by_response_model(
    output_dir: Path,
    judge_model: str,
    response_models: List[str],
) -> Dict[str, Path]:
    """
    Get output file paths for each response model (FORTRESS convention).
    
    Returns:
        Dict mapping response_model -> output file path
    """
    files = {}
    for response_model in response_models:
        filename = get_judgement_filename(judge_model, response_model)
        files[response_model] = output_dir / filename
    return files


async def judge_single_response(
    record: Dict,
    judge_model: str,
    timeout: float,
) -> Dict:
    """Judge a single response record."""
    query = record["prompt_text"]
    response = record["response_text"]

    if record.get("error"):
        verdict = SafetyVerdict(
            verdict="safe",  # Error = no unsafe content generated
            raw_output=f"[Response error: {record['error']}]",
        )
    else:
        verdict = await evaluate_safety(
            query=query,
            response=response,
            judge_model=judge_model,
            timeout=timeout,
        )
    
    return {
        "prompt_id": record["prompt_id"],
        "language": record["language"],
        "language_tier": record["language_tier"],
        "scenario": record["scenario"],
        "model_name": record["model_name"],
        "verdict": verdict.verdict,
        "raw_judge_output": verdict.raw_output,
        "timestamp": datetime.now().isoformat(),
    }


async def run_judging(
    records: List[Dict],
    output_dir: Path,
    judge_model: str,
    timeout: float,
    skip_keys: Set[Tuple[str, str, str, str]],
    concurrency: int = DEFAULT_CONCURRENCY,
    worker_id: int = 0,
    num_workers: int = 1,
):
    """Run judging on all records."""
    to_judge = []
    for record in records:
        key = (
            record["prompt_id"],
            record["language"],
            record["scenario"],
            record["model_name"],
        )
        if key not in skip_keys:
            to_judge.append(record)

    # Partition for multi-worker support (external workers, e.g., separate processes)
    if num_workers > 1:
        to_judge = [r for i, r in enumerate(to_judge) if i % num_workers == worker_id]

    if not to_judge:
        print("  All records already judged (or assigned to other workers)!")
        return

    # Get unique response models and create output files (FORTRESS convention)
    response_models = list(set(r["model_name"] for r in to_judge))
    output_files = get_output_files_by_response_model(output_dir, judge_model, response_models)
    file_locks = {model: asyncio.Lock() for model in response_models}

    print(f"\n{'=' * 70}")
    print("RUNNING SAFETY JUDGEMENT")
    print(f"{'=' * 70}")
    print(f"  Judge model: {judge_model}")
    if num_workers > 1:
        print(f"  External worker: {worker_id + 1}/{num_workers}")
    print(f"  Records to judge: {len(to_judge)}")
    print(f"  Skipped (already done): {len(skip_keys)}")
    print(f"  Response models: {len(response_models)}")
    for model in sorted(response_models):
        print(f"    - {model}")
    print(f"  Concurrency: {concurrency}")
    print()

    progress = ProgressTracker(len(to_judge))
    semaphore = asyncio.Semaphore(concurrency)

    async def report_progress():
        while True:
            await asyncio.sleep(10)
            print(f"  {progress.get_status()}")

    async def process_record(record: Dict):
        async with semaphore:
            response_model = record["model_name"]
            output_file = output_files.get(response_model)
            file_lock = file_locks.get(response_model)

            if not output_file or not file_lock:
                await progress.increment("invalid")
                return

            try:
                result = await judge_single_response(record, judge_model, timeout)

                async with file_lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

                await progress.increment(result["verdict"])

            except Exception as e:
                result = {
                    "prompt_id": record["prompt_id"],
                    "language": record["language"],
                    "language_tier": record["language_tier"],
                    "scenario": record["scenario"],
                    "model_name": record["model_name"],
                    "verdict": "invalid",
                    "raw_judge_output": f"Error: {e}",
                    "timestamp": datetime.now().isoformat(),
                }

                async with file_lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

                await progress.increment("invalid")

    progress_task = asyncio.create_task(report_progress())
    try:
        await asyncio.gather(*[process_record(r) for r in to_judge])
    finally:
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

    print(f"\n  {progress.get_status()}")
    print(f"  Output files:")
    for model, path in sorted(output_files.items()):
        print(f"    - {path.name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run safety judgement on MultiJail responses",
    )
    
    parser.add_argument(
        "--responses-dir", type=str, default=None,
        help="Directory containing response JSONL files",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory for judgements",
    )
    parser.add_argument(
        "--judge-model", type=str, default=JUDGE_MODEL,
        choices=JUDGE_MODELS,
        help=f"Judge model to use (default: {JUDGE_MODEL}). "
             f"Available: {', '.join(JUDGE_MODELS)}",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Request timeout (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing judgement files",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Concurrent judge calls (default: {DEFAULT_CONCURRENCY})",
    )

    # External worker support (for splitting work across processes)
    parser.add_argument(
        "--worker-id", type=int, default=0,
        help="Worker ID (0-indexed) for external multi-process parallelism",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Total number of external workers",
    )

    return parser.parse_args()


async def main():
    args = parse_args()
    
    # Setup directories
    if args.responses_dir:
        responses_dir = Path(args.responses_dir)
    else:
        responses_dir = RESULTS_DIR / "responses"
    
    if args.outdir:
        output_dir = Path(args.outdir)
    else:
        # Use FORTRESS convention: judgements/{judge_folder}/
        judge_folder = get_judge_folder_name(args.judge_model)
        output_dir = RESULTS_DIR / "judgements" / judge_folder
    
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MULTIJAIL BASELINE - SAFETY JUDGEMENT")
    print("=" * 70)
    print(f"  Responses dir: {responses_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Judge model: {args.judge_model}")
    print(f"  Resume: {args.resume}")
    print(f"  Concurrency: {args.concurrency}")
    if args.num_workers > 1:
        print(f"  External worker: {args.worker_id + 1}/{args.num_workers}")
    print("=" * 70)
    
    # Load responses
    print("\nLoading responses...")
    records = load_responses(responses_dir)
    
    if not records:
        print(f"\n❌ No response files found in {responses_dir}")
        print("  Run collect_responses.py first")
        return 1
    
    print(f"  Loaded {len(records)} response records")
    
    # Deduplicate (handles retry records - keeps successful over errors)
    records = deduplicate_responses(records)
    
    # Group by model for stats
    by_model = {}
    for r in records:
        model = r["model_name"]
        if model not in by_model:
            by_model[model] = 0
        by_model[model] += 1
    
    print("  By model:")
    for model, count in sorted(by_model.items()):
        print(f"    - {model}: {count}")
    
    # Load skip keys if resuming
    skip_keys: Set[Tuple[str, str, str, str]] = set()
    if args.resume:
        skip_keys = load_completed_keys(output_dir)
        if skip_keys:
            print(f"  Resume: found {len(skip_keys)} already-judged records")
    
    # Run judging
    await run_judging(
        records=records,
        output_dir=output_dir,
        judge_model=args.judge_model,
        timeout=args.timeout,
        skip_keys=skip_keys,
        concurrency=args.concurrency,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("JUDGEMENT COMPLETE")
    print("=" * 70)
    print(f"\n  Judge: {args.judge_model}")
    print(f"  Output: {output_dir}")
    print(f"\n  Next steps:")
    print(f"    1. Run judging for other judges (if needed)")
    print(f"    2. Run merge_judgements.py to combine with majority voting")
    print(f"    3. Run analyze.py to compute URS metrics")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
