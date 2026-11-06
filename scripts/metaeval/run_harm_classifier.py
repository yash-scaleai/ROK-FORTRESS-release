#!/usr/bin/env python3
"""
Run single-model harm dimension classifier on the meta-eval dataset.

Classifies rubric questions into harm dimensions and compares against human baseline.
Supports running individual models to find the best single model and optimal panel.

Usage:
    # Run single model with reasoning
    python run_harm_classifier.py --model anthropic/claude-opus-4-5 --reasoning-mode medium --variant A

    # Run without reasoning
    python run_harm_classifier.py --model openai/gpt-5.2 --variant A
"""

import asyncio
import json
import argparse
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rok_fortress_eval.harm_classifier.classifier import classify_task
from rok_fortress_eval.harm_classifier.data_models import (
    TaskClassificationInput,
    JudgeResult,
    HarmDimension,
    parse_harm_dimension,
)
from rok_fortress_eval.harm_classifier.meta_eval import (
    load_human_labels,
    compare_classifications,
    print_meta_eval_summary,
    MetaEvalResult,
)


@dataclass
class ClassificationTask:
    """Task to be classified."""
    task_id: str
    uuid: str
    risk_domain: str
    rubric: List[str]
    adversarial_prompt: Optional[str] = None
    benign_prompt: Optional[str] = None


def parse_rubric(value) -> List[str]:
    """Parse rubric from TSV (comma-separated questions or Python list format)."""
    if pd.isna(value) or not value:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        value = value.strip()

        # Try Python list literal first
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed]
        except Exception:
            pass

        # Split by "?, " pattern (questions ending with ?)
        if '?, ' in value:
            questions = []
            for q in value.split('?, '):
                q = q.strip()
                if q:
                    # Add back the ? if it was removed by split
                    if not q.endswith('?'):
                        q = q + '?'
                    questions.append(q)
            return questions

        # Single string fallback
        return [value]

    return []


def load_tasks_from_master(
    master_path: Path,
    human_labels_path: Path,
) -> List[ClassificationTask]:
    """
    Load tasks from MAIN PIPELINE MASTER, filtered to those with human labels.

    Args:
        master_path: Path to MAIN PIPELINE MASTER TSV
        human_labels_path: Path to META EVAL MASTER TSV (for human labels)

    Returns:
        List of tasks that have both task data and human labels
    """
    print(f"Loading task data from {master_path.name}...")
    master_df = pd.read_csv(master_path, sep='\t', lineterminator='\n', encoding='utf-8')

    # Strip \r from string columns (CRLF handling)
    for col in master_df.select_dtypes(include=['object']).columns:
        master_df[col] = master_df[col].str.replace('\r', '', regex=False)

    print(f"  Loaded {len(master_df)} rows")

    # Load human labels to get valid task IDs
    print(f"Loading human labels from {human_labels_path.name}...")
    human_labels = load_human_labels(
        str(human_labels_path),
        task_id_column="TASK_ID",
        harm_column="ORIGINAL_RUBRIC_HARM",
    )
    print(f"  Found {len(human_labels)} tasks with human harm labels")

    # Filter master to tasks with human labels
    tasks = []
    for _, row in master_df.iterrows():
        task_id = str(row.get('TASK_ID', '')).strip()
        if not task_id or task_id not in human_labels:
            continue

        rubric = parse_rubric(row.get('Rubric', ''))
        if not rubric:
            continue

        tasks.append(ClassificationTask(
            task_id=task_id,
            uuid=str(row.get('Task UUID', task_id)),
            risk_domain=str(row.get('Risk Domain', '')),
            rubric=rubric,
            adversarial_prompt=str(row.get('Adversarial Prompt', '')) if pd.notna(row.get('Adversarial Prompt')) else None,
            benign_prompt=str(row.get('Benign Prompt', '')) if pd.notna(row.get('Benign Prompt')) else None,
        ))

    print(f"  Filtered to {len(tasks)} tasks with both data and labels")
    return tasks


async def classify_single_task(
    task: ClassificationTask,
    model_name: str,
    variant: str,
    timeout: float,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run harm classifier on a single task.

    Returns dict with task_id, classifications, and metadata.
    """
    task_input = TaskClassificationInput(
        task_id=task.task_id,
        risk_domain=task.risk_domain,
        rubric=task.rubric,
        adversarial_prompt=task.adversarial_prompt,
        benign_prompt=task.benign_prompt,
    )

    # OpenAI models (gpt-5.x, o4-mini) require temperature=1.0
    temperature = 1.0 if model_name.startswith("openai/") else 0.0

    try:
        result = await classify_task(
            task_input=task_input,
            model_name=model_name,
            variant=variant,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )
    except Exception as e:
        # Return error result with exception details
        return {
            'task_id': task.task_id,
            'uuid': task.uuid,
            'variant': variant,
            'model_name': model_name,
            'refused': False,
            'error': f'{type(e).__name__}: {str(e)}',
            'latency_ms': 0,
            'raw_output': '',
            'classifications': [],
            'agreement_scores': [],
            'timestamp': None,
        }

    # Convert to serializable dict
    output = {
        'task_id': task.task_id,
        'uuid': task.uuid,
        'variant': variant,
        'model_name': model_name,
        'refused': result.refused,
        'error': result.error,
        'latency_ms': result.latency_ms,
        'raw_output': result.raw_output,
    }

    if result.classification:
        output['classifications'] = [
            {
                'rubric_index': c.rubric_index,
                'rubric_question': c.rubric_question,
                'primary_harm': c.primary_harm.value,
                'secondary_harm': c.secondary_harm.value if c.secondary_harm else None,
                'secondary_harm_justification': c.secondary_harm_justification,
                'confidence': c.confidence,
                'reasoning': c.reasoning,
            }
            for c in result.classification.classifications
        ]
        output['agreement_scores'] = []  # Not applicable for single model
        output['timestamp'] = result.classification.timestamp
    else:
        output['classifications'] = []
        output['agreement_scores'] = []
        output['timestamp'] = None

    return output


async def run_classifier(
    tasks: List[ClassificationTask],
    model_name: str,
    variant: str,
    output_path: Path,
    concurrency: int,
    timeout: float,
    reasoning_effort: Optional[str] = None,
):
    """Run harm classifier on all tasks with concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)

    async def classify_with_semaphore(task):
        async with semaphore:
            return await classify_single_task(task, model_name, variant, timeout, reasoning_effort)

    reasoning_str = f" (reasoning: {reasoning_effort})" if reasoning_effort else ""
    print(f"\nClassifying {len(tasks)} tasks with {model_name}{reasoning_str}")
    print(f"   Variant: {variant}")
    print(f"   Concurrency: {concurrency}")
    print()

    # Run all tasks
    coros = [classify_with_semaphore(task) for task in tasks]

    success_count = 0
    error_count = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, coro in enumerate(asyncio.as_completed(coros), 1):
            result = await coro

            # Write immediately
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()

            # Status
            if result.get('error') or result.get('refused'):
                status = "X"
                error_count += 1
                error_msg = (result.get('error') or 'refused')[:60]
                task_id = result['task_id']
                print(f"[{i}/{len(tasks)}] {status} {task_id} - {error_msg}")
            else:
                status = "o"
                success_count += 1
                task_id = result['task_id']
                print(f"[{i}/{len(tasks)}] {status} {task_id}")

    print(f"\nResults: {success_count} success, {error_count} errors")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run harm dimension classifier")
    parser.add_argument('--model', required=True, help="Model name (e.g., anthropic/claude-opus-4-5)")
    parser.add_argument('--variant', choices=['A', 'B', 'C'], default='A',
                        help="Classification variant: A (rubric-only), B (with adversarial+benign), C (benign only)")
    parser.add_argument('--reasoning-mode', choices=['low', 'medium', 'high'],
                        default=None, help="Reasoning mode (low/medium/high, default: none)")
    parser.add_argument('--concurrency', type=int, default=10,
                        help="Concurrent API requests")
    parser.add_argument('--timeout', type=float, default=300.0,
                        help="Timeout per request (seconds)")
    parser.add_argument('--limit', type=int, default=0,
                        help="Limit number of tasks (0=all, for testing)")
    parser.add_argument('--output-dir', default='runs/harm-classifier-calibration',
                        help="Output directory for results")

    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).resolve().parents[2]
    master_path = base_dir / 'data' / 'archive' / 'Copy of fortress private classification - 2026-01-14-1126 - MAIN PIPELINE MASTER (DO NOT EDIT) (1).tsv'
    human_labels_path = base_dir / 'data' / 'archive' / 'Copy of fortress private classification - 2026-01-14-1126 - META EVAL MASTER (DO NOT EDIT) (3).tsv'

    output_dir = base_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename
    model_safe = args.model.replace('/', '_').replace('-', '_').replace(':', '_').replace('.', '_')
    reasoning_suffix = f"_reasoning_{args.reasoning_mode}" if args.reasoning_mode else ""
    variant_suffix = f"_variant_{args.variant}"
    output_path = output_dir / f'results_{model_safe}{reasoning_suffix}{variant_suffix}.jsonl'

    print("=" * 80)
    print("HARM DIMENSION CLASSIFIER")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Variant: {args.variant}")
    print(f"Reasoning: {args.reasoning_mode or 'none'}")
    print(f"Output: {output_path}")
    print()

    # Load tasks
    tasks = load_tasks_from_master(master_path, human_labels_path)

    if args.limit > 0 and args.limit < len(tasks):
        tasks = tasks[:args.limit]
        print(f"  -> Limited to {len(tasks)} tasks for testing")

    # Run classifier
    asyncio.run(run_classifier(
        tasks=tasks,
        model_name=args.model,
        variant=args.variant,
        output_path=output_path,
        concurrency=args.concurrency,
        timeout=args.timeout,
        reasoning_effort=args.reasoning_mode,
    ))

    # Calculate alignment with human baseline
    print("\n" + "=" * 80)
    print("ALIGNMENT WITH HUMAN BASELINE")
    print("=" * 80)

    from rok_fortress_eval.harm_classifier.meta_eval import (
        load_human_labels,
        load_llm_classifications,
        compare_classifications,
        print_meta_eval_summary,
    )

    human_labels = load_human_labels(
        str(human_labels_path),
        task_id_column="TASK_ID",
        harm_column="ORIGINAL_RUBRIC_HARM",
    )

    llm_results = load_llm_classifications(str(output_path))

    result = compare_classifications(human_labels, llm_results)

    summary = print_meta_eval_summary(result)
    print(summary)

    # Save summary
    summary_path = output_path.with_suffix('.summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
