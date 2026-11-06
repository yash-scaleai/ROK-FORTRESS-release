#!/usr/bin/env python3
"""
Run judge model on collected model responses.

This script takes model response files (from collect_model_responses.py) and runs
a judge model to evaluate each response against its rubric.

Usage:
    # Single response file
    python run_full_judging.py \
        --judge-model openai/o4-mini \
        --response-file runs/rok_fortress_responses_final/responses_gpt-5.2_no_reasoning.jsonl \
        --output-dir runs/rok_fortress_judgements_final \
        --concurrency 40

    # vLLM judge (HyperCLOVAX)
    python run_full_judging.py \
        --judge-model vllm/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B \
        --response-file runs/rok_fortress_responses_final/responses_gpt-5.2_no_reasoning.jsonl \
        --output-dir runs/rok_fortress_judgements_final \
        --vllm-endpoints http://localhost:8000/v1,http://localhost:8001/v1 \
        --concurrency 40

    # All response files
    python run_full_judging.py \
        --judge-model openai/o4-mini \
        --response-dir runs/rok_fortress_responses_final \
        --output-dir runs/rok_fortress_judgements_final \
        --concurrency 40
"""

import asyncio
import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rok_fortress_eval.core.judge import call_single_judge, parse_judge_output
from rok_fortress_eval.core.config import DEFAULT_TIMEOUT, DEFAULT_PARSING_MODEL
from rok_fortress_eval.models import PromptVariant, ModelResponse


@dataclass
class ResponseToJudge:
    """A model response that needs to be judged."""
    task_id: str
    variant_name: str
    task_type: str
    risk_domain: str
    risk_subdomain: str
    model_name: str
    reasoning_mode: str
    adversarial_prompt: str
    adversarial_response: str
    benign_prompt: str
    benign_response: str
    rubric: List[str]
    source_file: str


def load_responses(jsonl_path: Path) -> List[ResponseToJudge]:
    """Load model responses from JSONL file."""
    responses = []

    # Extract model name and reasoning mode from filename
    # Format: responses_<model>_<reasoning>.jsonl
    filename = jsonl_path.stem
    parts = filename.replace('responses_', '').rsplit('_', 1)

    if len(parts) == 2 and parts[1] in ['reasoning', 'no']:
        # Handle cases like responses_gpt-5.2_with_reasoning or responses_gpt-5.2_no_reasoning
        pass

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            # Skip responses with errors
            adv_error = data.get('adversarial', {}).get('error')
            if adv_error:
                continue

            # Skip empty responses (successful refusals - no judging needed)
            adv_response = data.get('adversarial', {}).get('response', '')
            if not adv_response or not adv_response.strip():
                continue

            # Parse rubric
            rubric = data.get('rubric', [])
            if isinstance(rubric, str):
                # Handle comma-separated rubric string
                rubric = [q.strip() for q in rubric.split(',') if q.strip()]

            if not rubric:
                continue

            responses.append(ResponseToJudge(
                task_id=data.get('task_id', ''),
                variant_name=data.get('variant_name', ''),
                task_type=data.get('task_type', ''),
                risk_domain=data.get('risk_domain', ''),
                risk_subdomain=data.get('risk_subdomain', ''),
                model_name=data.get('model_name', ''),
                reasoning_mode=data.get('reasoning_mode', 'unknown'),
                adversarial_prompt=data.get('adversarial', {}).get('prompt', ''),
                adversarial_response=adv_response,
                benign_prompt=data.get('benign', {}).get('prompt', ''),
                benign_response=data.get('benign', {}).get('response', ''),
                rubric=rubric,
                source_file=jsonl_path.name,
            ))

    return responses


def get_completed_keys(output_path: Path) -> Set[Tuple[str, str]]:
    """Get set of already-judged (task_id, variant_name) tuples."""
    completed = set()

    if not output_path.exists():
        return completed

    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Only count as completed if no error and has labels
                if data.get('error') is None and data.get('judge_labels'):
                    key = (data.get('task_id', ''), data.get('variant_name', ''))
                    completed.add(key)
            except json.JSONDecodeError:
                continue

    return completed


async def judge_single_response(
    response: ResponseToJudge,
    judge_model: str,
    parsing_model: str,
    timeout: float,
    reasoning_effort: Optional[str] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Run judge on a single model response with retry logic.

    Returns dict with judge verdicts.
    """
    # Convert to PromptVariant for existing judge infrastructure
    variant = PromptVariant(
        task_id=response.task_id,
        task_type=response.task_type,
        variant_name=response.variant_name,
        adversarial_prompt=response.adversarial_prompt,
        benign_prompt=response.benign_prompt,
        rubric=response.rubric,
        risk_domain=response.risk_domain,
        risk_subdomain=response.risk_subdomain,
    )

    model_response = ModelResponse(
        model_name=response.model_name,
        prompt_type="adversarial",
        response_text=response.adversarial_response,
        latency_ms=0,
        tokens_used=None,
    )

    last_error = None

    for attempt in range(max_retries):
        try:
            # Call judge
            raw_output = await call_single_judge(
                variant=variant,
                model_response=model_response,
                judge_model=judge_model,
                judge_name=judge_model,
                timeout=timeout,
                reasoning_effort=reasoning_effort,
            )

            # Parse verdicts
            verdicts, reasonings = await parse_judge_output(
                raw_output=raw_output,
                num_questions=len(response.rubric),
                parsing_model=parsing_model,
                timeout=timeout,
            )

            # Convert bool to yes/no strings
            judge_labels = ["yes" if v else "no" for v in verdicts]

            return {
                'task_id': response.task_id,
                'variant_name': response.variant_name,
                'model_name': response.model_name,
                'reasoning_mode': response.reasoning_mode,
                'judge_model': judge_model.split('/')[-1],  # Short name
                'risk_domain': response.risk_domain,
                'risk_subdomain': response.risk_subdomain,
                'rubric': response.rubric,
                'adversarial_prompt': response.adversarial_prompt,
                'adversarial_response': response.adversarial_response,
                'judge_labels': judge_labels,
                'judge_reasonings': reasonings,
                'judge_raw_output': raw_output,
                'error': None,
                'timestamp': datetime.utcnow().isoformat(),
            }

        except Exception as e:
            # Unwrap RetryError to get the actual root cause
            actual_error = e
            if hasattr(e, 'last_attempt'):
                try:
                    actual_error = e.last_attempt.exception()
                except Exception:
                    pass
            last_error = f"{type(actual_error).__name__}: {actual_error}"
            if attempt < max_retries - 1:
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
                continue

    # All retries failed
    return {
        'task_id': response.task_id,
        'variant_name': response.variant_name,
        'model_name': response.model_name,
        'reasoning_mode': response.reasoning_mode,
        'judge_model': judge_model.split('/')[-1],
        'error': last_error,
        'judge_labels': None,
        'timestamp': datetime.utcnow().isoformat(),
    }


async def run_judging(
    responses: List[ResponseToJudge],
    judge_model: str,
    parsing_model: str,
    output_path: Path,
    error_path: Path,
    concurrency: int,
    timeout: float,
    reasoning_effort: Optional[str] = None,
    max_retries: int = 3,
):
    """Run judge on all responses with async concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)

    async def judge_with_semaphore(response):
        async with semaphore:
            return await judge_single_response(
                response, judge_model, parsing_model, timeout, reasoning_effort, max_retries
            )

    # Get already completed
    completed = get_completed_keys(output_path)

    # Filter to only pending responses
    pending = [r for r in responses if (r.task_id, r.variant_name) not in completed]

    if len(pending) < len(responses):
        print(f"  Resuming: {len(completed)} already completed, {len(pending)} pending")

    if not pending:
        print("  All responses already judged!")
        return

    reasoning_str = f" (reasoning: {reasoning_effort})" if reasoning_effort else ""
    print(f"\n  Judging {len(pending)} responses with {judge_model}{reasoning_str}")
    print(f"  Concurrency: {concurrency}, Max retries: {max_retries}")

    # Run all tasks
    coros = [judge_with_semaphore(r) for r in pending]

    success_count = 0
    error_count = 0

    # Open files for appending
    with open(output_path, 'a', encoding='utf-8') as f_out, \
         open(error_path, 'a', encoding='utf-8') as f_err:

        for i, coro in enumerate(asyncio.as_completed(coros), 1):
            result = await coro

            # Write result
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_out.flush()

            # Track errors
            if result.get('error'):
                error_count += 1
                f_err.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_err.flush()
                status = "X"
            else:
                success_count += 1
                status = "v"

            # Progress update every 100 or at end
            if i % 100 == 0 or i == len(pending):
                print(f"  [{i}/{len(pending)}] {status} Success: {success_count}, Errors: {error_count}")

    print(f"\n  Completed: {success_count} success, {error_count} errors")
    print(f"  Output: {output_path}")


def get_judge_short_name(judge_model: str) -> str:
    """Extract short name from judge model path."""
    # vllm/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B -> hyperclovax
    # openai/o4-mini -> o4_mini
    # fireworks_ai/kimi-k2-thinking -> kimi_k2_thinking

    name = judge_model.split('/')[-1].lower()

    # Simplify common names
    if 'hyperclovax' in name.lower():
        return 'hyperclovax'
    elif 'kimi' in name.lower():
        return 'kimi_k2_thinking'
    elif 'o4-mini' in name.lower() or 'o4_mini' in name.lower():
        return 'o4_mini'

    return name.replace('-', '_').replace('.', '_')


def get_response_file_short_name(response_file: Path) -> str:
    """Extract short name from response file."""
    # responses_gpt-5.2_no_reasoning.jsonl -> gpt-5.2_no_reasoning
    name = response_file.stem.replace('responses_', '')
    return name


def main():
    parser = argparse.ArgumentParser(description="Run judge on model responses")

    # Required
    parser.add_argument('--judge-model', required=True,
                        help="Judge model (e.g., openai/o4-mini, vllm/naver-hyperclovax/...)")

    # Input (one of these required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--response-file',
                             help="Single response JSONL file to judge")
    input_group.add_argument('--response-dir',
                             help="Directory containing response JSONL files")

    # Output
    parser.add_argument('--output-dir', default='runs/rok_fortress_judgements_final',
                        help="Output directory for judgements")

    # Configuration
    parser.add_argument('--parsing-model', default=DEFAULT_PARSING_MODEL,
                        help="Model for parsing judge outputs")
    parser.add_argument('--concurrency', type=int, default=40,
                        help="Concurrent judge requests")
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                        help="Timeout per request (seconds)")
    parser.add_argument('--max-retries', type=int, default=3,
                        help="Max retries per response")
    parser.add_argument('--reasoning-mode', choices=['low', 'medium', 'high'],
                        default=None, help="Reasoning mode for judge")

    # vLLM specific
    parser.add_argument('--vllm-endpoints',
                        help="Comma-separated vLLM endpoints for vllm/ prefix models")

    # Filtering
    parser.add_argument('--limit', type=int, default=0,
                        help="Limit responses to process (0=all, for testing)")

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create logs subdirectory
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # Get judge short name
    judge_short = get_judge_short_name(args.judge_model)

    # Collect response files
    if args.response_file:
        response_files = [Path(args.response_file)]
    else:
        response_dir = Path(args.response_dir)
        if not response_dir.is_absolute():
            response_dir = base_dir / response_dir
        response_files = sorted(response_dir.glob('responses_*.jsonl'))

    if not response_files:
        print("No response files found!")
        return

    print("=" * 80)
    print("FULL DATASET JUDGING")
    print("=" * 80)
    print(f"\nJudge: {args.judge_model}")
    print(f"Reasoning: {args.reasoning_mode or 'none'}")
    print(f"Response files: {len(response_files)}")
    print(f"Output dir: {output_dir}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max retries: {args.max_retries}")

    # Configure vLLM endpoints if provided
    if args.vllm_endpoints:
        endpoints = args.vllm_endpoints.split(',')
        print(f"vLLM endpoints: {len(endpoints)} servers")

        # Set environment variable for vLLM router
        os.environ['VLLM_ENDPOINTS'] = args.vllm_endpoints

    print()

    # Process each response file
    for response_file in response_files:
        if not response_file.exists():
            print(f"File not found: {response_file}")
            continue

        response_short = get_response_file_short_name(response_file)

        print("-" * 80)
        print(f"Processing: {response_file.name}")

        # Load responses
        responses = load_responses(response_file)
        print(f"  Loaded {len(responses)} responses to judge")

        if not responses:
            print("  No valid responses to judge, skipping")
            continue

        # Apply limit if specified
        if args.limit > 0:
            responses = responses[:args.limit]
            print(f"  Limited to {len(responses)} responses")

        # Output paths
        output_path = output_dir / f'judgements_{judge_short}_on_{response_short}.jsonl'
        error_path = output_dir / f'errors_{judge_short}.jsonl'

        # Run judging
        asyncio.run(run_judging(
            responses=responses,
            judge_model=args.judge_model,
            parsing_model=args.parsing_model,
            output_path=output_path,
            error_path=error_path,
            concurrency=args.concurrency,
            timeout=args.timeout,
            reasoning_effort=args.reasoning_mode,
            max_retries=args.max_retries,
        ))

    print()
    print("=" * 80)
    print("JUDGING COMPLETE")
    print("=" * 80)

    # Summary
    judgement_files = list(output_dir.glob(f'judgements_{judge_short}_*.jsonl'))
    total_judgements = sum(
        sum(1 for _ in open(f, 'r')) for f in judgement_files if f.exists()
    )
    print(f"\nTotal judgement files: {len(judgement_files)}")
    print(f"Total judgements: {total_judgements}")
    print(f"\nOutput directory: {output_dir}")


if __name__ == '__main__':
    main()
