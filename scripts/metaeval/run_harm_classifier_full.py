#!/usr/bin/env python3
"""
Run harm dimension classifier on ALL rubric criteria from the main pipeline.

Features:
- Resume capability: tracks completed task IDs, skips duplicates
- Proper error logging: errors saved with full details, not just retries
- Batch support: can run specific task ranges for parallel execution
- Progress tracking: real-time progress file

Usage:
    # Run all tasks
    python run_harm_classifier_full.py --model gemini/gemini-3-pro-preview

    # Run specific batch (for parallel tmux sessions)
    python run_harm_classifier_full.py --model gemini/gemini-3-pro-preview --batch 1 --total-batches 4

    # Resume from previous run
    python run_harm_classifier_full.py --model gemini/gemini-3-pro-preview --resume
"""

import asyncio
import json
import argparse
import ast
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rok_fortress_eval.harm_classifier.data_models import (
    TaskClassificationInput,
    HarmDimension,
    parse_harm_dimension,
)
from rok_fortress_eval.harm_classifier.classifier import classify_task


@dataclass
class ClassificationTask:
    task_id: str
    uuid: str
    risk_domain: str
    rubric: List[str]
    adversarial_prompt: Optional[str] = None
    benign_prompt: Optional[str] = None


def parse_rubric(value) -> List[str]:
    """Parse rubric from TSV."""
    if pd.isna(value) or not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed]
        except Exception:
            pass
        # Try splitting by question marks
        if '?, ' in value:
            questions = []
            for q in value.split('?, '):
                q = q.strip()
                if q:
                    if not q.endswith('?'):
                        q = q + '?'
                    questions.append(q)
            return questions
        # Single question
        if '?' in value:
            return [value]
    return []


def load_all_tasks(master_path: Path) -> List[ClassificationTask]:
    """Load ALL tasks from master TSV."""
    print(f"Loading tasks from {master_path.name}...")
    df = pd.read_csv(master_path, sep='\t', lineterminator='\n', encoding='utf-8')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.replace('\r', '', regex=False)
    print(f"  Loaded {len(df)} rows")

    tasks = []
    skipped = 0
    for _, row in df.iterrows():
        task_id = str(row.get('TASK_ID', '')).strip()
        if not task_id:
            skipped += 1
            continue

        rubric = parse_rubric(row.get('Rubric', ''))
        if not rubric:
            skipped += 1
            continue

        tasks.append(ClassificationTask(
            task_id=task_id,
            uuid=str(row.get('Task UUID', task_id)),
            risk_domain=str(row.get('Risk Domain', '')),
            rubric=rubric,
            adversarial_prompt=str(row.get('Adversarial Prompt', '')) if pd.notna(row.get('Adversarial Prompt')) else None,
            benign_prompt=str(row.get('Benign Prompt', '')) if pd.notna(row.get('Benign Prompt')) else None,
        ))

    print(f"  Valid tasks: {len(tasks)} (skipped {skipped})")
    return tasks


def load_completed_task_ids(output_path: Path) -> Set[str]:
    """Load task IDs that have already been processed."""
    completed = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    completed.add(data.get('task_id', ''))
                except:
                    pass
    return completed


async def classify_single_task(
    task: ClassificationTask,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    reasoning_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify a single task using the standard classifier."""

    # Create TaskClassificationInput
    task_input = TaskClassificationInput(
        task_id=task.task_id,
        risk_domain=task.risk_domain,
        rubric=task.rubric,
        adversarial_prompt=task.adversarial_prompt,
        benign_prompt=task.benign_prompt,
    )

    start_time = time.time()

    try:
        # Use the standard classifier
        result = await classify_task(
            task_input=task_input,
            model_name=model,
            variant="A",  # Rubric-only variant
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            reasoning_effort=reasoning_mode,
            max_parse_retries=3,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Convert result to dict
        if result.classification:
            classifications = []
            for c in result.classification.classifications:
                classifications.append({
                    "rubric_index": c.rubric_index,
                    "rubric_question": c.rubric_question,
                    "primary_harm": c.primary_harm.value if c.primary_harm else None,
                    "secondary_harm": c.secondary_harm.value if c.secondary_harm else None,
                    "secondary_harm_justification": c.secondary_harm_justification,
                    "confidence": c.confidence,
                    "reasoning": c.reasoning,
                })

            return {
                "task_id": task.task_id,
                "uuid": task.uuid,
                "risk_domain": task.risk_domain,
                "rubric_count": len(task.rubric),
                "model_name": model,
                "success": True,
                "error": result.error,  # May have warning but still success
                "latency_ms": latency_ms,
                "raw_output": result.raw_output[:5000] if result.raw_output else "",
                "refused": result.refused,
                "classifications": classifications,
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            return {
                "task_id": task.task_id,
                "uuid": task.uuid,
                "risk_domain": task.risk_domain,
                "rubric_count": len(task.rubric),
                "model_name": model,
                "success": False,
                "error": result.error,
                "latency_ms": latency_ms,
                "raw_output": result.raw_output[:5000] if result.raw_output else "",
                "refused": result.refused,
                "classifications": [],
                "timestamp": datetime.utcnow().isoformat(),
            }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return {
            "task_id": task.task_id,
            "uuid": task.uuid,
            "risk_domain": task.risk_domain,
            "rubric_count": len(task.rubric),
            "model_name": model,
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "latency_ms": latency_ms,
            "raw_output": "",
            "refused": False,
            "classifications": [],
            "timestamp": datetime.utcnow().isoformat(),
        }


async def run_classifier(
    tasks: List[ClassificationTask],
    model: str,
    output_path: Path,
    concurrency: int,
    timeout: float,
    temperature: float = 0.3,
    reasoning_mode: Optional[str] = None,
    progress_file: Optional[Path] = None,
):
    """Run classifier on all tasks with progress tracking."""
    semaphore = asyncio.Semaphore(concurrency)

    # Track progress
    success_count = 0
    error_count = 0
    start_time = time.time()

    async def classify_with_semaphore(task, idx):
        nonlocal success_count, error_count
        async with semaphore:
            result = await classify_single_task(
                task, model, temperature, 8192, timeout, reasoning_mode
            )

            if result.get('success'):
                success_count += 1
                status = "✓"
            else:
                error_count += 1
                status = "✗"

            elapsed = time.time() - start_time
            rate = (success_count + error_count) / elapsed * 60 if elapsed > 0 else 0

            print(f"[{idx+1}/{len(tasks)}] {status} {task.task_id} ({rate:.1f}/min)")

            return result

    print(f"\nClassifying {len(tasks)} tasks with {model}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Output: {output_path}")
    print()

    # Create coroutines
    coros = [classify_with_semaphore(task, i) for i, task in enumerate(tasks)]

    # Process and write results incrementally
    with open(output_path, 'a', encoding='utf-8') as f:
        for coro in asyncio.as_completed(coros):
            result = await coro
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()

            # Update progress file
            if progress_file:
                progress = {
                    "total": len(tasks),
                    "completed": success_count + error_count,
                    "success": success_count,
                    "errors": error_count,
                    "elapsed_seconds": time.time() - start_time,
                }
                with open(progress_file, 'w') as pf:
                    json.dump(progress, pf)

    print(f"\n{'='*60}")
    print(f"COMPLETED: {success_count} success, {error_count} errors")
    print(f"Output: {output_path}")
    print(f"Time: {time.time() - start_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run harm classifier on all rubric criteria")
    parser.add_argument('--model', required=True, help="Model name (e.g., gemini/gemini-3-pro-preview)")
    parser.add_argument('--batch', type=int, default=0, help="Batch number (1-indexed, 0=all)")
    parser.add_argument('--total-batches', type=int, default=4, help="Total number of batches")
    parser.add_argument('--concurrency', type=int, default=5, help="Concurrent requests")
    parser.add_argument('--timeout', type=float, default=180.0, help="Request timeout")
    parser.add_argument('--temperature', type=float, default=0.3, help="Model temperature")
    parser.add_argument('--reasoning-mode', choices=['low', 'medium', 'high'], default=None)
    parser.add_argument('--resume', action='store_true', help="Resume from existing output")
    parser.add_argument('--output-dir', default='runs/harm-classifier-full', help="Output directory")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    master_path = base_dir / 'data' / 'Copy of fortress private classification - 2026-01-14-1126 - MAIN PIPELINE MASTER (DO NOT EDIT) (1).tsv'

    output_dir = base_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all tasks
    all_tasks = load_all_tasks(master_path)

    # Split into batches if specified
    if args.batch > 0:
        batch_size = len(all_tasks) // args.total_batches
        start_idx = (args.batch - 1) * batch_size
        end_idx = start_idx + batch_size if args.batch < args.total_batches else len(all_tasks)
        tasks = all_tasks[start_idx:end_idx]
        batch_suffix = f"_batch{args.batch}of{args.total_batches}"
        print(f"Running batch {args.batch}/{args.total_batches}: tasks {start_idx+1}-{end_idx}")
    else:
        tasks = all_tasks
        batch_suffix = ""

    # Output file
    model_safe = args.model.replace('/', '_').replace('.', '_')
    output_path = output_dir / f'results_{model_safe}{batch_suffix}.jsonl'
    progress_file = output_dir / f'progress_{model_safe}{batch_suffix}.json'

    # Resume: filter out already completed tasks
    if args.resume or output_path.exists():
        completed_ids = load_completed_task_ids(output_path)
        original_count = len(tasks)
        tasks = [t for t in tasks if t.task_id not in completed_ids]
        print(f"Resume mode: {len(completed_ids)} already done, {len(tasks)} remaining")

    if not tasks:
        print("No tasks to process!")
        return

    print(f"\n{'='*60}")
    print(f"HARM DIMENSION CLASSIFIER - FULL RUN")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Tasks: {len(tasks)}")
    print(f"Output: {output_path}")
    print()

    asyncio.run(run_classifier(
        tasks=tasks,
        model=args.model,
        output_path=output_path,
        concurrency=args.concurrency,
        timeout=args.timeout,
        temperature=args.temperature,
        reasoning_mode=args.reasoning_mode,
        progress_file=progress_file,
    ))


if __name__ == '__main__':
    main()
