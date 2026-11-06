#!/usr/bin/env python3
"""
Run harm dimension classifier on vLLM models.

Directly calls vLLM endpoints without going through LiteLLM proxy.

Usage:
    # HyperCLOVA X
    python run_harm_classifier_vllm.py --model hyperclovax --port 10021

    # Kanana-2
    python run_harm_classifier_vllm.py --model kanana2 --port 9100

    # Solar-Open
    python run_harm_classifier_vllm.py --model solar --port 8090

    # K-EXAONE
    python run_harm_classifier_vllm.py --model kexaone --port 8001
"""

import asyncio
import json
import argparse
import ast
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rok_fortress_eval.harm_classifier.data_models import (
    TaskClassificationInput,
    HarmDimension,
    parse_harm_dimension,
)
from rok_fortress_eval.harm_classifier.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt_variant_a,
)
from rok_fortress_eval.harm_classifier.meta_eval import (
    load_human_labels,
    load_llm_classifications,
    compare_classifications,
    print_meta_eval_summary,
)


# Model configurations
MODEL_CONFIGS = {
    "hyperclovax": {
        "model_name": "track_a_model",  # OmniServe model name
        "temperature": 0.7,
        "max_tokens": 8192,
        "display_name": "HyperCLOVA X",
    },
    "kanana2": {
        "model_name": "kakaocorp/kanana-2-30b-a3b-thinking",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 8192,
        "display_name": "Kanana-2",
    },
    "solar": {
        "model_name": "upstage/Solar-Open-100B",
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 8192,
        "display_name": "Solar-Open-100B",
    },
    "kexaone": {
        "model_name": "LGAI-EXAONE/K-EXAONE-236B-A23B",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "display_name": "K-EXAONE",
    },
}


class APIError(Exception):
    pass


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
        if '?, ' in value:
            questions = []
            for q in value.split('?, '):
                q = q.strip()
                if q:
                    if not q.endswith('?'):
                        q = q + '?'
                    questions.append(q)
            return questions
        return [value]
    return []


def load_tasks_from_master(master_path: Path, human_labels_path: Path) -> List[ClassificationTask]:
    """Load tasks from master TSV, filtered to those with human labels."""
    print(f"Loading task data from {master_path.name}...")
    master_df = pd.read_csv(master_path, sep='\t', lineterminator='\n', encoding='utf-8')
    for col in master_df.select_dtypes(include=['object']).columns:
        master_df[col] = master_df[col].str.replace('\r', '', regex=False)
    print(f"  Loaded {len(master_df)} rows")

    print(f"Loading human labels from {human_labels_path.name}...")
    human_labels = load_human_labels(
        str(human_labels_path),
        task_id_column="TASK_ID",
        harm_column="ORIGINAL_RUBRIC_HARM",
    )
    print(f"  Found {len(human_labels)} tasks with human harm labels")

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


def extract_json_from_response(raw_output: str) -> Optional[dict]:
    """Extract JSON from model response."""
    import re
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(code_block_pattern, raw_output)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try JSON object pattern
    json_pattern = r'\{[\s\S]*"classifications"[\s\S]*\}'
    matches = re.findall(json_pattern, raw_output)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None


@retry(
    retry=retry_if_exception_type((APIError, asyncio.TimeoutError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
)
async def call_vllm_model(
    client: AsyncOpenAI,
    model_name: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    **extra_params,
) -> tuple:
    """Call vLLM model with retry."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    request_kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Add top_p if provided (standard OpenAI parameter)
    if 'top_p' in extra_params and extra_params['top_p'] is not None:
        request_kwargs['top_p'] = extra_params['top_p']

    # Add top_k via extra_body (not a standard OpenAI parameter)
    if 'top_k' in extra_params and extra_params['top_k'] is not None:
        request_kwargs['extra_body'] = {'top_k': extra_params['top_k']}

    start_time = time.time()
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(**request_kwargs),
            timeout=timeout
        )
        latency_ms = (time.time() - start_time) * 1000

        # Handle reasoning models where content may be in reasoning_content
        message = response.choices[0].message
        content = message.content
        if not content:
            # Try to get reasoning_content for thinking models
            reasoning_content = getattr(message, 'reasoning_content', None)
            if reasoning_content:
                content = reasoning_content
            else:
                # Also check for reasoning field
                reasoning = getattr(message, 'reasoning', None)
                if reasoning:
                    content = reasoning
        content = content or ""
        tokens = response.usage.total_tokens if response.usage else None

        return content, latency_ms, tokens

    except asyncio.TimeoutError:
        raise
    except Exception as e:
        raise APIError(f"API call failed: {type(e).__name__}: {str(e)}")


async def classify_single_task(
    client: AsyncOpenAI,
    task: ClassificationTask,
    model_config: dict,
    timeout: float,
) -> Dict[str, Any]:
    """Classify a single task."""
    task_input = TaskClassificationInput(
        task_id=task.task_id,
        risk_domain=task.risk_domain,
        rubric=task.rubric,
        adversarial_prompt=task.adversarial_prompt,
        benign_prompt=task.benign_prompt,
    )

    user_prompt = build_user_prompt_variant_a(
        task_id=task.task_id,
        risk_domain=task.risk_domain,
        rubric=task.rubric,
    )

    max_retries = 3
    all_outputs = []
    last_error = None

    for attempt in range(max_retries):
        try:
            response_text, latency_ms, tokens = await call_vllm_model(
                client=client,
                model_name=model_config["model_name"],
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
                timeout=timeout,
                top_p=model_config.get("top_p"),
                top_k=model_config.get("top_k"),
            )
            all_outputs.append(response_text)

            # Parse response
            json_data = extract_json_from_response(response_text)
            if not json_data:
                last_error = "Failed to extract JSON from response"
                if attempt < max_retries - 1:
                    user_prompt += f"\n\n[RETRY: Please output valid JSON with exactly {len(task.rubric)} classifications.]"
                    continue
            else:
                classifications_data = json_data.get("classifications", [])
                if len(classifications_data) != len(task.rubric):
                    last_error = f"Got {len(classifications_data)} classifications, expected {len(task.rubric)}"
                    if attempt < max_retries - 1:
                        user_prompt += f"\n\n[RETRY: You output {len(classifications_data)} classifications but need exactly {len(task.rubric)}.]"
                        continue

                # Success - parse classifications
                classifications = []
                for item in classifications_data:
                    primary_harm_str = item.get("primary_harm", "")
                    primary_harm = parse_harm_dimension(primary_harm_str)
                    if not primary_harm:
                        continue

                    secondary_harm = None
                    if item.get("secondary_harm"):
                        secondary_harm = parse_harm_dimension(item["secondary_harm"])

                    rubric_index = item.get("rubric_index", len(classifications))
                    rubric_question = item.get("rubric_question", task.rubric[rubric_index] if rubric_index < len(task.rubric) else "")

                    classifications.append({
                        "rubric_index": rubric_index,
                        "rubric_question": rubric_question,
                        "primary_harm": primary_harm.value,
                        "secondary_harm": secondary_harm.value if secondary_harm else None,
                        "secondary_harm_justification": item.get("secondary_harm_justification"),
                        "confidence": float(item.get("confidence", 0.5)),
                        "reasoning": item.get("reasoning", ""),
                    })

                return {
                    "task_id": task.task_id,
                    "uuid": task.uuid,
                    "variant": "A",
                    "model_name": model_config["display_name"],
                    "refused": False,
                    "error": None,
                    "latency_ms": latency_ms,
                    "raw_output": response_text,
                    "classifications": classifications,
                    "timestamp": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            if attempt < max_retries - 1:
                continue

    # All retries failed
    return {
        "task_id": task.task_id,
        "uuid": task.uuid,
        "variant": "A",
        "model_name": model_config["display_name"],
        "refused": False,
        "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
        "latency_ms": 0,
        "raw_output": "\n---\n".join(all_outputs) if all_outputs else "",
        "classifications": [],
        "timestamp": datetime.utcnow().isoformat(),
    }


async def run_classifier(
    tasks: List[ClassificationTask],
    model_key: str,
    port: int,
    output_path: Path,
    concurrency: int,
    timeout: float,
):
    """Run classifier on all tasks."""
    model_config = MODEL_CONFIGS[model_key]

    base_url = f"http://localhost:{port}/v1"
    client = AsyncOpenAI(base_url=base_url, api_key="dummy")

    semaphore = asyncio.Semaphore(concurrency)

    async def classify_with_semaphore(task):
        async with semaphore:
            return await classify_single_task(client, task, model_config, timeout)

    print(f"\nClassifying {len(tasks)} tasks with {model_config['display_name']}")
    print(f"   Endpoint: {base_url}")
    print(f"   Concurrency: {concurrency}")
    print()

    coros = [classify_with_semaphore(task) for task in tasks]

    success_count = 0
    error_count = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, coro in enumerate(asyncio.as_completed(coros), 1):
            result = await coro

            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()

            if result.get('error'):
                status = "X"
                error_count += 1
                error_msg = result.get('error', '')[:60]
                print(f"[{i}/{len(tasks)}] {status} {result['task_id']} - {error_msg}")
            else:
                status = "o"
                success_count += 1
                print(f"[{i}/{len(tasks)}] {status} {result['task_id']}")

    print(f"\nResults: {success_count} success, {error_count} errors")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run harm classifier on vLLM models")
    parser.add_argument('--model', required=True, choices=list(MODEL_CONFIGS.keys()),
                        help="Model key (hyperclovax, kanana2, solar, kexaone)")
    parser.add_argument('--port', type=int, required=True, help="vLLM server port")
    parser.add_argument('--concurrency', type=int, default=10, help="Concurrent requests")
    parser.add_argument('--timeout', type=float, default=300.0, help="Request timeout")
    parser.add_argument('--limit', type=int, default=0, help="Limit tasks (0=all)")
    parser.add_argument('--output-dir', default='runs/harm-classifier-calibration',
                        help="Output directory")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    master_path = base_dir / 'data' / 'archive' / 'Copy of fortress private classification - 2026-01-14-1126 - MAIN PIPELINE MASTER (DO NOT EDIT) (1).tsv'
    human_labels_path = base_dir / 'data' / 'archive' / 'Copy of fortress private classification - 2026-01-14-1126 - META EVAL MASTER (DO NOT EDIT) (3).tsv'

    output_dir = base_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = MODEL_CONFIGS[args.model]
    model_safe = args.model
    output_path = output_dir / f'results_vllm_{model_safe}_variant_A.jsonl'

    print("=" * 80)
    print("HARM DIMENSION CLASSIFIER (vLLM)")
    print("=" * 80)
    print(f"\nModel: {model_config['display_name']}")
    print(f"Port: {args.port}")
    print(f"Output: {output_path}")
    print()

    tasks = load_tasks_from_master(master_path, human_labels_path)

    if args.limit > 0 and args.limit < len(tasks):
        tasks = tasks[:args.limit]
        print(f"  -> Limited to {len(tasks)} tasks")

    asyncio.run(run_classifier(
        tasks=tasks,
        model_key=args.model,
        port=args.port,
        output_path=output_path,
        concurrency=args.concurrency,
        timeout=args.timeout,
    ))

    # Calculate alignment
    print("\n" + "=" * 80)
    print("ALIGNMENT WITH HUMAN BASELINE")
    print("=" * 80)

    human_labels = load_human_labels(
        str(human_labels_path),
        task_id_column="TASK_ID",
        harm_column="ORIGINAL_RUBRIC_HARM",
    )

    llm_results = load_llm_classifications(str(output_path))
    result = compare_classifications(human_labels, llm_results)
    summary = print_meta_eval_summary(result)
    print(summary)

    summary_path = output_path.with_suffix('.summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
