#!/usr/bin/env python3
"""
Run single-judge calibration on meta-eval dataset with human baseline labels.

Supports both API models (via LiteLLM) and vLLM models (local inference).
Runs single judge per task (not ensemble) to calibrate against human labels.

Usage:
    # API model (parallel execution across all tasks)
    python run_judge_calibration.py --judge-model openai/gpt-5.2 --concurrency 30
    
    # vLLM model (local inference)
    python run_judge_calibration.py --judge-model vllm/k-exaone \
        --vllm-endpoints http://localhost:8000/v1 http://localhost:8001/v1
"""

import asyncio
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Reuse existing infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rok_fortress_eval.core.judge import call_single_judge, parse_judge_output
from rok_fortress_eval.core.config import DEFAULT_TIMEOUT, DEFAULT_PARSING_MODEL
from rok_fortress_eval.models import PromptVariant, ModelResponse, JudgeEvaluation, RubricEvaluation


@dataclass
class CalibrationTask:
    """Single calibration task with human baseline (multiple annotators)."""
    uuid: str
    task_id: str
    task_type: str
    variant_name: str
    prompt: str
    response: str
    rubric: List[str]
    human_annotations: List[Dict[str, Any]]  # List of {annotator_date, review_level, labels}
    risk_domain: str
    risk_subdomain: str


def load_calibration_dataset(jsonl_path: Path) -> List[CalibrationTask]:
    """Load calibration dataset from JSONL."""
    tasks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            tasks.append(CalibrationTask(
                uuid=data['uuid'],
                task_id=data['task_id'],
                task_type=data['task_type'],
                variant_name=data['variant_name'],
                prompt=data['prompt'],
                response=data['response'],
                rubric=data['rubric'],
                human_annotations=data['human_annotations'],
                risk_domain=data['risk_domain'],
                risk_subdomain=data['risk_subdomain'],
            ))
    return tasks


async def judge_single_task(
    task: CalibrationTask,
    judge_model: str,
    parsing_model: str,
    timeout: float,
) -> Dict[str, Any]:
    """
    Run single judge on one calibration task.
    
    Returns dict with judge verdicts and human baseline for comparison.
    """
    # Convert to PromptVariant and ModelResponse for existing judge infrastructure
    variant = PromptVariant(
        task_id=task.task_id,
        task_type=task.task_type,
        variant_name=task.variant_name,
        adversarial_prompt=task.prompt,
        benign_prompt="",  # Not needed for judging
        rubric=task.rubric,
        risk_domain=task.risk_domain,
        risk_subdomain=task.risk_subdomain,
    )
    
    model_response = ModelResponse(
        model_name="meta-eval-model",  # Placeholder - actual model varies
        prompt_type="adversarial",
        response_text=task.response,
        latency_ms=0,
        tokens_used=None,
    )
    
    # Call judge (reuse existing infrastructure)
    try:
        raw_output = await call_single_judge(
            variant=variant,
            model_response=model_response,
            judge_model=judge_model,
            judge_name=judge_model,
            timeout=timeout,
        )
        
        # Parse verdicts
        verdicts, reasonings = await parse_judge_output(
            raw_output=raw_output,
            num_questions=len(task.rubric),
            parsing_model=parsing_model,
            timeout=timeout,
        )
        
        # Convert bool to yes/no strings for consistency with human labels
        judge_labels = ["yes" if v else "no" for v in verdicts]
        
        return {
            'uuid': task.uuid,
            'task_id': task.task_id,
            'variant_name': task.variant_name,
            'risk_domain': task.risk_domain,
            'risk_subdomain': task.risk_subdomain,
            'rubric': task.rubric,
            'prompt': task.prompt,
            'response': task.response,
            'human_annotations': task.human_annotations,  # Multiple annotators
            'judge_labels': judge_labels,
            'judge_reasonings': reasonings,
            'judge_raw_output': raw_output,
            'error': None,
        }
    
    except Exception as e:
        return {
            'uuid': task.uuid,
            'task_id': task.task_id,
            'variant_name': task.variant_name,
            'error': str(e),
            'human_annotations': task.human_annotations,
            'judge_labels': None,
        }


async def run_judge_calibration(
    tasks: List[CalibrationTask],
    judge_model: str,
    parsing_model: str,
    output_path: Path,
    concurrency: int,
    timeout: float,
):
    """Run judge calibration with async concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)
    
    async def judge_with_semaphore(task):
        async with semaphore:
            return await judge_single_task(task, judge_model, parsing_model, timeout)
    
    print(f"\n🔍 Judging {len(tasks)} tasks with {judge_model}")
    print(f"   Concurrency: {concurrency}")
    print(f"   Parsing: {parsing_model}\n")
    
    # Run all tasks
    coros = [judge_with_semaphore(task) for task in tasks]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, coro in enumerate(asyncio.as_completed(coros), 1):
            result = await coro
            
            # Write immediately
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            
            # Status
            status = "✓" if result.get('error') is None else "❌"
            variant = result['variant_name']
            task_id = result['task_id']
            print(f"[{i}/{len(tasks)}] {status} {task_id} | {variant}")
    
    print(f"\n✅ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run single-judge calibration")
    parser.add_argument('--judge-model', required=True, help="Judge model name (e.g., openai/gpt-5.2)")
    parser.add_argument('--input', default='runs/judge-calibration/calibration_dataset.jsonl',
                        help="Input calibration dataset (JSONL)")
    parser.add_argument('--output-dir', default='runs/judge-calibration',
                        help="Output directory for results")
    parser.add_argument('--parsing-model', default=DEFAULT_PARSING_MODEL,
                        help="Model for parsing judge outputs")
    parser.add_argument('--concurrency', type=int, default=30,
                        help="Concurrent API requests")
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                        help="Timeout per request (seconds)")
    parser.add_argument('--vllm-endpoints', nargs='+',
                        help="vLLM endpoints for vllm/ prefix models")
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    input_path = base_dir / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_dir = base_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output filename based on judge model
    judge_safe_name = args.judge_model.replace('/', '_').replace('-', '_').replace(':', '_').replace('.', '_')
    output_path = output_dir / f'results_{judge_safe_name}.jsonl'
    
    # Load tasks
    print("=" * 80)
    print("JUDGE CALIBRATION")
    print("=" * 80)
    print(f"\nJudge: {args.judge_model}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    tasks = load_calibration_dataset(input_path)
    print(f"\nLoaded {len(tasks)} calibration tasks")
    
    # Configure vLLM endpoints if needed
    if args.vllm_endpoints:
        # Import vLLM router and configure
        from rok_fortress_eval.vllm.load_balancer import VLLMLoadBalancer
        balancer = VLLMLoadBalancer(args.vllm_endpoints)
        print(f"vLLM endpoints: {args.vllm_endpoints}")
        # Note: vLLM routing handled in api_client via existing infrastructure
    
    # Run calibration
    asyncio.run(run_judge_calibration(
        tasks=tasks,
        judge_model=args.judge_model,
        parsing_model=args.parsing_model,
        output_path=output_path,
        concurrency=args.concurrency,
        timeout=args.timeout,
    ))


if __name__ == '__main__':
    main()
