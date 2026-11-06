#!/usr/bin/env python3
"""
Classify harm dimensions for ROK-FORTRESS rubric criteria.

Uses a 3-judge LLM panel to assign harm dimensions to each rubric criterion,
with majority vote consensus.

Usage:
    # Variant A (rubric-only):
    uv run python scripts/classify_harm_dimensions.py \
        --input data/rok_fortress_public.tsv \
        --variant A \
        --outdir runs/harm_classification_A

    # Variant B (with context):
    uv run python scripts/classify_harm_dimensions.py \
        --input data/rok_fortress_public.tsv \
        --variant B \
        --outdir runs/harm_classification_B

    # With meta-eval comparison:
    uv run python scripts/classify_harm_dimensions.py \
        --input data/rok_fortress_public.tsv \
        --variant A \
        --meta-eval "data/Copy of fortress private classification - 2026-01-14-1126 - META EVAL MASTER (DO NOT EDIT) (1).tsv"
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from rok_fortress_eval.harm_classifier.data_models import (
    TaskClassificationInput,
    ConsensusResult,
)
from rok_fortress_eval.harm_classifier.judge_panel import (
    JudgePanel,
    DEFAULT_JUDGES,
)
from rok_fortress_eval.harm_classifier.meta_eval import (
    run_meta_eval,
    print_meta_eval_summary,
)
from rok_fortress_eval.core.data_loader import parse_rubric


def load_tasks_for_classification(
    input_path: str,
    max_rows: Optional[int] = None,
) -> List[TaskClassificationInput]:
    """
    Load tasks from dataset for harm dimension classification.

    Only loads original_en rubric variant (841 unique tasks).
    """
    path = Path(input_path)
    if path.suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    if max_rows:
        df = df.head(max_rows)

    # Column mapping
    task_id_col = "TASK_ID" if "TASK_ID" in df.columns else "task_id"
    risk_domain_col = "Risk Domain" if "Risk Domain" in df.columns else "risk_domain"
    rubric_col = "Rubric" if "Rubric" in df.columns else "rubric"
    adv_prompt_col = "Adversarial Prompt" if "Adversarial Prompt" in df.columns else "en_adversarial_prompt"
    benign_col = "Benign Prompt" if "Benign Prompt" in df.columns else "benign_prompt"

    tasks = []
    seen_task_ids = set()

    for _, row in df.iterrows():
        task_id = str(row.get(task_id_col, "")).strip()
        if not task_id or task_id in seen_task_ids:
            continue

        # Parse rubric
        rubric = parse_rubric(row.get(rubric_col, ""))
        if not rubric:
            continue

        risk_domain = str(row.get(risk_domain_col, "")).strip()
        adv_prompt = str(row.get(adv_prompt_col, "")).strip()
        benign_prompt = str(row.get(benign_col, "")).strip()

        task = TaskClassificationInput(
            task_id=task_id,
            risk_domain=risk_domain,
            rubric=rubric,
            adversarial_prompt=adv_prompt if adv_prompt and adv_prompt != "nan" else None,
            benign_prompt=benign_prompt if benign_prompt and benign_prompt != "nan" else None,
        )
        tasks.append(task)
        seen_task_ids.add(task_id)

    return tasks


def load_skip_keys(output_path: Path) -> Set[str]:
    """Load already-completed task IDs from existing output file."""
    skip_keys = set()
    if not output_path.exists():
        return skip_keys

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                skip_keys.add(record["task_id"])
            except (json.JSONDecodeError, KeyError):
                continue

    return skip_keys


class ProgressTracker:
    """Track classification progress."""

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
        rate = self.completed / elapsed * 60 if elapsed > 0 else 0
        return (
            f"[PROGRESS] {self.completed}/{self.total} ({pct:.1f}%) | "
            f"Errors: {self.errors} | Rate: {rate:.1f}/min | Elapsed: {elapsed_str}"
        )


async def classify_batch(
    tasks: List[TaskClassificationInput],
    panel: JudgePanel,
    variant: str,
    output_path: Path,
    progress: ProgressTracker,
    semaphore: asyncio.Semaphore,
):
    """Classify a batch of tasks and write results incrementally."""
    file_lock = asyncio.Lock()

    async def process_task(task: TaskClassificationInput):
        async with semaphore:
            try:
                result = await panel.classify_with_consensus(task, variant)

                # Write result
                async with file_lock:
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result.to_jsonl_record(), ensure_ascii=False) + "\n")

                await progress.increment(success=True)
                return result

            except Exception as e:
                await progress.increment(success=False)
                print(f"  Error classifying {task.task_id}: {e}")
                return None

    # Process all tasks
    await asyncio.gather(*[process_task(t) for t in tasks])


async def run_classification(
    input_path: str,
    output_dir: Path,
    variant: str,
    judges: List[str],
    concurrency: int,
    timeout: float,
    resume: bool,
    max_rows: Optional[int],
):
    """Run full classification pipeline."""
    # Load tasks
    print("\nLoading tasks...")
    tasks = load_tasks_for_classification(input_path, max_rows)
    print(f"  Loaded {len(tasks)} unique tasks")

    # Setup output
    output_path = output_dir / f"classifications_variant_{variant}.jsonl"

    # Load skip keys if resuming
    skip_keys = set()
    if resume and output_path.exists():
        skip_keys = load_skip_keys(output_path)
        print(f"  Resume: found {len(skip_keys)} completed tasks to skip")

    # Filter tasks
    tasks_to_run = [t for t in tasks if t.task_id not in skip_keys]
    print(f"  Tasks to classify: {len(tasks_to_run)}")

    if not tasks_to_run:
        print("  No tasks to classify!")
        return output_path

    # Initialize panel
    panel = JudgePanel(
        judges=judges,
        temperature=0.0,
        max_tokens=8192,
        timeout=timeout,
    )

    # Progress tracking
    progress = ProgressTracker(len(tasks_to_run))
    semaphore = asyncio.Semaphore(concurrency)

    # Progress reporter
    async def report_progress():
        while True:
            await asyncio.sleep(15)
            print(f"  {progress.get_status()}")

    progress_task = asyncio.create_task(report_progress())

    try:
        print(f"\nClassifying with variant {variant}...")
        print(f"  Judges: {', '.join(judges)}")
        print(f"  Concurrency: {concurrency}")
        print(f"  Timeout: {timeout}s")
        print(f"  Output: {output_path}\n")

        await classify_batch(
            tasks=tasks_to_run,
            panel=panel,
            variant=variant,
            output_path=output_path,
            progress=progress,
            semaphore=semaphore,
        )

    finally:
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

    print(f"\n  {progress.get_status()}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify harm dimensions for ROK-FORTRESS rubric criteria",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input TSV/CSV dataset",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory (default: runs/harm_classification_<timestamp>)",
    )
    parser.add_argument(
        "--variant", type=str, choices=["A", "B"], default="A",
        help="Variant A (rubric-only) or B (with-context)",
    )
    parser.add_argument(
        "--judges", type=str, default=None,
        help=f"Comma-separated judge models (default: {','.join(DEFAULT_JUDGES)})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=50,
        help="Concurrent requests (default: 50)",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output",
    )
    parser.add_argument(
        "--meta-eval", type=str, default=None,
        help="Path to meta-eval TSV for human label comparison",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Max rows to load (for testing)",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # Parse judges
    if args.judges:
        judges = [j.strip() for j in args.judges.split(",")]
    else:
        judges = DEFAULT_JUDGES

    # Setup output directory
    if args.outdir:
        output_dir = Path(args.outdir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"runs/harm_classification_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Print banner
    print("=" * 70)
    print("ROK-FORTRESS HARM DIMENSION CLASSIFIER")
    print("=" * 70)
    print(f"  Input: {args.input}")
    print(f"  Output: {output_dir}")
    print(f"  Variant: {args.variant}")
    print(f"  Judges: {len(judges)}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Resume: {args.resume}")
    print("=" * 70)

    # Run classification
    output_path = await run_classification(
        input_path=args.input,
        output_dir=output_dir,
        variant=args.variant,
        judges=judges,
        concurrency=args.concurrency,
        timeout=args.timeout,
        resume=args.resume,
        max_rows=args.max_rows,
    )

    # Run meta-eval if requested
    if args.meta_eval:
        print("\n" + "=" * 70)
        print("RUNNING META-EVALUATION")
        print("=" * 70)

        try:
            result = run_meta_eval(
                llm_classifications_path=str(output_path),
                human_labels_path=args.meta_eval,
            )

            print(print_meta_eval_summary(result))

            # Save meta-eval results
            meta_eval_path = output_dir / "meta_eval_comparison.json"
            with open(meta_eval_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nMeta-eval results saved to: {meta_eval_path}")

        except Exception as e:
            print(f"Meta-eval failed: {e}")

    # Final summary
    print("\n" + "=" * 70)
    print("CLASSIFICATION COMPLETE")
    print("=" * 70)

    # Count results
    if output_path.exists():
        line_count = sum(1 for _ in open(output_path, "r", encoding="utf-8"))
        print(f"  Output: {output_path} ({line_count} records)")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
