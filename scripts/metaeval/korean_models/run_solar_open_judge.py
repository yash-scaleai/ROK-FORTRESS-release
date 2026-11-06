#!/usr/bin/env python3
"""
Solar-Open-100B Judge Calibration

Docker: upstage/vllm-solar-open:latest
Reasoning parser: solar_open
Recommended params: temperature=0.8, top_p=0.95, top_k=50
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
from rok_fortress_eval.core.config import DEFAULT_PARSING_MODEL


# Solar-Open configuration
DEFAULT_BASE_URL = "http://localhost:8090/v1"
MODEL_NAME = "upstage/Solar-Open-100B"


def load_calibration_dataset(jsonl_path: Path) -> List[Dict]:
    """Load calibration dataset from JSONL."""
    tasks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


async def call_solar_judge(
    client: AsyncOpenAI,
    task: Dict,
    timeout: float,
) -> str:
    """Call Solar-Open via vLLM for judging."""
    questions_formatted = "\n".join([
        f"Question {i+1}: {q}" for i, q in enumerate(task['rubric'])
    ])
    
    eval_prompt = JUDGE_EVAL_PROMPT.format(
        adversarial_prompt=task['prompt'],
        model_response=task['response'],
        questions_formatted=questions_formatted,
    )
    
    messages = [{"role": "user", "content": eval_prompt}]
    
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                # Solar-Open recommended params
                temperature=0.8,
                top_p=0.95,
                max_tokens=3072,
                extra_body={"top_k": 50},
            ),
            timeout=timeout
        )
        return resp.choices[0].message.content or ""
    except asyncio.TimeoutError:
        raise Exception(f"Timeout after {timeout}s")
    except Exception as e:
        raise Exception(f"API error: {e}")


async def judge_single_task(
    client: AsyncOpenAI,
    task: Dict,
    parsing_model: str,
    timeout: float,
) -> Dict[str, Any]:
    """Run Solar-Open judge on one calibration task."""
    try:
        raw_output = await call_solar_judge(client, task, timeout)
        
        verdicts, reasonings = await parse_judge_output(
            raw_output=raw_output,
            num_questions=len(task['rubric']),
            parsing_model=parsing_model,
            timeout=timeout,
        )
        
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
    """Run Solar-Open judge calibration."""
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="not-needed",
    )
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def judge_with_semaphore(task):
        async with semaphore:
            return await judge_single_task(client, task, parsing_model, timeout)
    
    print(f"\n🔍 Judging {len(tasks)} tasks with Solar-Open-100B")
    print(f"   Endpoint: {base_url}")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Concurrency: {concurrency}")
    print(f"   Parsing: {parsing_model}\n")
    
    coros = [judge_with_semaphore(task) for task in tasks]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, coro in enumerate(asyncio.as_completed(coros), 1):
            result = await coro
            
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            
            status = "✓" if result.get('error') is None else "❌"
            variant = result['variant_name']
            task_id = result['task_id']
            print(f"[{i}/{len(tasks)}] {status} {task_id} | {variant}")
    
    print(f"\n✅ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Solar-Open Judge Calibration")
    parser.add_argument('--input', required=True, help="Input calibration dataset (JSONL)")
    parser.add_argument('--output', required=True, help="Output results file (JSONL)")
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL,
                        help=f"vLLM API base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument('--parsing-model', default=DEFAULT_PARSING_MODEL,
                        help="Model for parsing judge outputs")
    parser.add_argument('--concurrency', type=int, default=5,
                        help="Concurrent requests (default: 5)")
    parser.add_argument('--timeout', type=float, default=180.0,
                        help="Timeout per request in seconds (default: 180)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("SOLAR-OPEN-100B JUDGE CALIBRATION")
    print("=" * 80)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    
    tasks = load_calibration_dataset(input_path)
    print(f"Loaded {len(tasks)} calibration tasks")
    
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
