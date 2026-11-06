#!/usr/bin/env python3
"""
HyperCLOVAX-SEED-Think-32B Judge Calibration

Uses OmniServe deployment (official method from HuggingFace).
Directly uses OpenAI client for OmniServe API compatibility.

API: http://localhost:10021/v1
Model: track_a_model
Reasoning: extra_body={"thinking_token_budget": 500, "chat_template_kwargs": {"thinking": True}}
"""

import asyncio
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from openai import AsyncOpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from rok_fortress_eval.core.judge import parse_judge_output, JUDGE_EVAL_PROMPT
from rok_fortress_eval.core.config import DEFAULT_TIMEOUT, DEFAULT_PARSING_MODEL
from rok_fortress_eval.models import PromptVariant, ModelResponse


# HyperCLOVAX configuration
# Using direct vLLM endpoint (port 10021) since OmniServe's chainer requires newer driver
OMNISERVE_BASE_URL = "http://localhost:10021/v1"
OMNISERVE_MODEL = "track_a_model"


def load_calibration_dataset(jsonl_path: Path) -> List[Dict]:
    """Load calibration dataset from JSONL."""
    tasks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


async def call_hyperclovax_judge(
    client: AsyncOpenAI,
    task: Dict,
    timeout: float,
    enable_thinking: bool = True,
) -> str:
    """
    Call HyperCLOVAX via OmniServe for judging.
    
    Returns raw judge output text.
    """
    # Format the judge prompt
    questions_formatted = "\n".join([
        f"Question {i+1}: {q}" for i, q in enumerate(task['rubric'])
    ])
    
    eval_prompt = JUDGE_EVAL_PROMPT.format(
        adversarial_prompt=task['prompt'],
        model_response=task['response'],
        questions_formatted=questions_formatted,
    )
    
    messages = [{"role": "user", "content": eval_prompt}]
    
    # OmniServe extra_body for reasoning mode
    extra_body = {}
    if enable_thinking:
        extra_body = {
            "thinking_token_budget": 500,
            "chat_template_kwargs": {"thinking": True},
        }
    
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=OMNISERVE_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=3072,
                extra_body=extra_body if extra_body else None,
            ),
            timeout=timeout
        )
        return resp.choices[0].message.content or ""
    except asyncio.TimeoutError:
        raise Exception(f"OmniServe timeout after {timeout}s")
    except Exception as e:
        raise Exception(f"OmniServe API error: {e}")


async def judge_single_task(
    client: AsyncOpenAI,
    task: Dict,
    parsing_model: str,
    timeout: float,
) -> Dict[str, Any]:
    """Run HyperCLOVAX judge on one calibration task."""
    try:
        # Call HyperCLOVAX judge
        raw_output = await call_hyperclovax_judge(client, task, timeout)
        
        # Parse verdicts using existing infrastructure
        verdicts, reasonings = await parse_judge_output(
            raw_output=raw_output,
            num_questions=len(task['rubric']),
            parsing_model=parsing_model,
            timeout=timeout,
        )
        
        # Convert bool to yes/no strings
        judge_labels = ["yes" if v else "no" for v in verdicts]
        
        return {
            'uuid': task['uuid'],
            'task_id': task['task_id'],
            'variant_name': task['variant_name'],
            'risk_domain': task.get('risk_domain', ''),
            'risk_subdomain': task.get('risk_subdomain', ''),
            'rubric': task['rubric'],
            'prompt': task['prompt'],
            'response': task['response'],
            'human_annotations': task['human_annotations'],
            'judge_labels': judge_labels,
            'judge_reasonings': reasonings,
            'judge_raw_output': raw_output,
            'error': None,
        }
    
    except Exception as e:
        return {
            'uuid': task['uuid'],
            'task_id': task['task_id'],
            'variant_name': task['variant_name'],
            'error': str(e),
            'human_annotations': task['human_annotations'],
            'judge_labels': None,
        }


async def run_calibration(
    tasks: List[Dict],
    output_path: Path,
    parsing_model: str,
    concurrency: int,
    timeout: float,
    base_url: str,
):
    """Run HyperCLOVAX judge calibration."""
    # Initialize OmniServe client
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="not-needed",  # OmniServe doesn't require API key
    )
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def judge_with_semaphore(task):
        async with semaphore:
            return await judge_single_task(client, task, parsing_model, timeout)
    
    print(f"\n🔍 Judging {len(tasks)} tasks with HyperCLOVAX-SEED-Think-32B")
    print(f"   Endpoint: {base_url}")
    print(f"   Model: {OMNISERVE_MODEL}")
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
    parser = argparse.ArgumentParser(description="HyperCLOVAX Judge Calibration")
    parser.add_argument('--input', required=True, help="Input calibration dataset (JSONL)")
    parser.add_argument('--output', required=True, help="Output results file (JSONL)")
    parser.add_argument('--base-url', default=OMNISERVE_BASE_URL, 
                        help=f"OmniServe API base URL (default: {OMNISERVE_BASE_URL})")
    parser.add_argument('--parsing-model', default=DEFAULT_PARSING_MODEL,
                        help="Model for parsing judge outputs")
    parser.add_argument('--concurrency', type=int, default=5,
                        help="Concurrent requests (default: 5, keep low for local inference)")
    parser.add_argument('--timeout', type=float, default=180.0,
                        help="Timeout per request in seconds (default: 180)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("HYPERCLOVAX JUDGE CALIBRATION")
    print("=" * 80)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    
    # Load tasks
    tasks = load_calibration_dataset(input_path)
    print(f"Loaded {len(tasks)} calibration tasks")
    
    # Run calibration
    asyncio.run(run_calibration(
        tasks=tasks,
        output_path=output_path,
        parsing_model=args.parsing_model,
        concurrency=args.concurrency,
        timeout=args.timeout,
        base_url=args.base_url,
    ))


if __name__ == '__main__':
    main()
