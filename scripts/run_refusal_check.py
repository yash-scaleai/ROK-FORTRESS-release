#!/usr/bin/env python3
"""
Run refusal classifier on collected model responses.

This script takes model response files (from collect_model_responses.py) and runs
a refusal classifier to detect if the model refused to answer.

Usage:
    # Check benign responses for over-refusal (ORS calculation)
    python run_refusal_check.py \
        --refusal-model openai/o4-mini \
        --response-file runs/rok_fortress_responses_final/responses_gpt-5.2_no_reasoning.jsonl \
        --output-dir runs/rok_fortress_refusal_checks_final \
        --prompt-type benign \
        --concurrency 40

    # Check adversarial responses for refusal
    python run_refusal_check.py \
        --refusal-model gemini/gemini-3-flash-preview \
        --response-file runs/rok_fortress_responses_final/responses_gpt-5.2_no_reasoning.jsonl \
        --output-dir runs/rok_fortress_refusal_checks_final \
        --prompt-type adversarial \
        --concurrency 40

    # Process all response files in a directory
    python run_refusal_check.py \
        --refusal-model openai/o4-mini \
        --response-dir runs/rok_fortress_responses_final \
        --output-dir runs/rok_fortress_refusal_checks_final \
        --prompt-type benign \
        --concurrency 40
"""

import asyncio
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rok_fortress_eval.core.judge import check_for_refusal
from rok_fortress_eval.core.config import DEFAULT_TIMEOUT


@dataclass
class ResponseToCheck:
    """A model response that needs refusal checking."""
    task_id: str
    variant_name: str
    task_type: str
    risk_domain: str
    risk_subdomain: str
    model_name: str
    reasoning_mode: str
    prompt: str
    response: str
    prompt_type: str  # 'adversarial' or 'benign'
    source_file: str


def load_responses(jsonl_path: Path, prompt_type: str) -> List[ResponseToCheck]:
    """Load model responses from JSONL file.

    Args:
        jsonl_path: Path to the response JSONL file
        prompt_type: 'adversarial' or 'benign' - which prompt/response to check
    """
    responses = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            # Get the appropriate prompt/response pair
            if prompt_type == 'adversarial':
                prompt_data = data.get('adversarial', {})
            elif prompt_type == 'direct':
                prompt_data = data.get('direct', {})
                if not prompt_data:
                    continue  # Record has no direct field (older format)
            else:
                prompt_data = data.get('benign', {})

            prompt = prompt_data.get('prompt', '')
            response = prompt_data.get('response', '')
            error = prompt_data.get('error')

            # Skip if there was an error collecting the response
            if error:
                continue

            # Skip empty responses - they're already refusals
            if not response or not response.strip():
                continue

            responses.append(ResponseToCheck(
                task_id=data.get('task_id', ''),
                variant_name=data.get('variant_name', ''),
                task_type=data.get('task_type', ''),
                risk_domain=data.get('risk_domain', ''),
                risk_subdomain=data.get('risk_subdomain', ''),
                model_name=data.get('model_name', ''),
                reasoning_mode=data.get('reasoning_mode', 'unknown'),
                prompt=prompt,
                response=response,
                prompt_type=prompt_type,
                source_file=jsonl_path.name,
            ))

    return responses


def get_completed_keys(output_path: Path) -> Set[Tuple[str, str]]:
    """Get set of already-checked (task_id, variant_name) tuples."""
    completed = set()

    if not output_path.exists():
        return completed

    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Only count as completed if no error
                if data.get('error') is None:
                    key = (data.get('task_id', ''), data.get('variant_name', ''))
                    completed.add(key)
            except json.JSONDecodeError:
                continue

    return completed


async def check_single_response(
    response: ResponseToCheck,
    refusal_model: str,
    timeout: float,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Run refusal check on a single model response with retry logic.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            result = await check_for_refusal(
                prompt=response.prompt,
                response=response.response,
                refusal_model=refusal_model,
                timeout=timeout,
            )

            return {
                'task_id': response.task_id,
                'variant_name': response.variant_name,
                'model_name': response.model_name,
                'reasoning_mode': response.reasoning_mode,
                'prompt_type': response.prompt_type,
                'refusal_model': refusal_model.split('/')[-1],
                'risk_domain': response.risk_domain,
                'risk_subdomain': response.risk_subdomain,
                'prompt': response.prompt,
                'response': response.response,
                'is_refusal': result.verdict,
                'refusal_reasoning': result.reasoning,
                'error': None,
                'timestamp': datetime.utcnow().isoformat(),
            }

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue

    # All retries failed
    return {
        'task_id': response.task_id,
        'variant_name': response.variant_name,
        'model_name': response.model_name,
        'reasoning_mode': response.reasoning_mode,
        'prompt_type': response.prompt_type,
        'refusal_model': refusal_model.split('/')[-1],
        'error': last_error,
        'is_refusal': None,
        'timestamp': datetime.utcnow().isoformat(),
    }


async def run_refusal_checks(
    responses: List[ResponseToCheck],
    refusal_model: str,
    output_path: Path,
    error_path: Path,
    concurrency: int,
    timeout: float,
    max_retries: int = 3,
):
    """Run refusal check on all responses with async concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)

    async def check_with_semaphore(response):
        async with semaphore:
            return await check_single_response(
                response, refusal_model, timeout, max_retries
            )

    # Get already completed
    completed = get_completed_keys(output_path)

    # Filter to only pending responses
    pending = [r for r in responses if (r.task_id, r.variant_name) not in completed]

    if len(pending) < len(responses):
        print(f"  Resuming: {len(completed)} already completed, {len(pending)} pending")

    if not pending:
        print("  All responses already checked!")
        return

    print(f"\n  Checking {len(pending)} responses with {refusal_model}")
    print(f"  Concurrency: {concurrency}, Max retries: {max_retries}")

    # Run all tasks
    coros = [check_with_semaphore(r) for r in pending]

    success_count = 0
    error_count = 0
    refusal_count = 0

    # Open files for appending
    with open(output_path, 'a', encoding='utf-8') as f_out, \
         open(error_path, 'a', encoding='utf-8') as f_err:

        for i, coro in enumerate(asyncio.as_completed(coros), 1):
            result = await coro

            # Write result
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_out.flush()

            # Track errors and refusals
            if result.get('error'):
                error_count += 1
                f_err.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_err.flush()
                status = "X"
            else:
                success_count += 1
                if result.get('is_refusal'):
                    refusal_count += 1
                status = "R" if result.get('is_refusal') else "v"

            # Progress update every 100 or at end
            if i % 100 == 0 or i == len(pending):
                print(f"  [{i}/{len(pending)}] {status} Success: {success_count}, Refusals: {refusal_count}, Errors: {error_count}")

    print(f"\n  Completed: {success_count} success, {refusal_count} refusals, {error_count} errors")
    print(f"  Output: {output_path}")


def get_refusal_model_short_name(refusal_model: str) -> str:
    """Extract short name from refusal model path."""
    name = refusal_model.split('/')[-1].lower()

    if 'o4-mini' in name or 'o4_mini' in name:
        return 'o4_mini'
    elif 'gemini-3-flash' in name or 'gemini_3_flash' in name:
        return 'gemini_3_flash'
    elif 'gemini-2.0-flash' in name:
        return 'gemini_2_flash'

    return name.replace('-', '_').replace('.', '_')


def get_response_file_short_name(response_file: Path) -> str:
    """Extract short name from response file."""
    name = response_file.stem.replace('responses_', '')
    return name


def main():
    parser = argparse.ArgumentParser(description="Run refusal check on model responses")

    # Required
    parser.add_argument('--refusal-model', required=True,
                        help="Model for refusal detection (e.g., openai/o4-mini, gemini/gemini-3-flash-preview)")
    parser.add_argument('--prompt-type', required=True, choices=['adversarial', 'benign', 'direct'],
                        help="Which prompt/response pair to check")

    # Input (one of these required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--response-file',
                             help="Single response JSONL file to check")
    input_group.add_argument('--response-dir',
                             help="Directory containing response JSONL files")

    # Output
    parser.add_argument('--output-dir', default='runs/rok_fortress_refusal_checks_final',
                        help="Output directory for refusal check results")

    # Configuration
    parser.add_argument('--concurrency', type=int, default=40,
                        help="Concurrent requests")
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                        help="Timeout per request (seconds)")
    parser.add_argument('--max-retries', type=int, default=3,
                        help="Max retries per response")

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

    # Get refusal model short name
    refusal_short = get_refusal_model_short_name(args.refusal_model)

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
    print("REFUSAL CHECK")
    print("=" * 80)
    print(f"\nRefusal model: {args.refusal_model}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Response files: {len(response_files)}")
    print(f"Output dir: {output_dir}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max retries: {args.max_retries}")
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
        responses = load_responses(response_file, args.prompt_type)
        print(f"  Loaded {len(responses)} responses to check")

        if not responses:
            print("  No valid responses to check, skipping")
            continue

        # Apply limit if specified
        if args.limit > 0:
            responses = responses[:args.limit]
            print(f"  Limited to {len(responses)} responses")

        # Output paths
        output_path = output_dir / f'refusal_{args.prompt_type}_{refusal_short}_on_{response_short}.jsonl'
        error_path = output_dir / f'errors_{refusal_short}.jsonl'

        # Run refusal checks
        asyncio.run(run_refusal_checks(
            responses=responses,
            refusal_model=args.refusal_model,
            output_path=output_path,
            error_path=error_path,
            concurrency=args.concurrency,
            timeout=args.timeout,
            max_retries=args.max_retries,
        ))

    print()
    print("=" * 80)
    print("REFUSAL CHECK COMPLETE")
    print("=" * 80)

    # Summary
    check_files = list(output_dir.glob(f'refusal_{args.prompt_type}_{refusal_short}_*.jsonl'))
    total_checks = sum(
        sum(1 for _ in open(f, 'r')) for f in check_files if f.exists()
    )
    print(f"\nTotal check files: {len(check_files)}")
    print(f"Total checks: {total_checks}")
    print(f"\nOutput directory: {output_dir}")


if __name__ == '__main__':
    main()
