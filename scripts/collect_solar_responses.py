#!/usr/bin/env python3
"""
Collect model responses from Solar-Open-100B on ROK-FORTRESS dataset.

Solar-Open-100B is a MoE model (102B total, 12B active) with 128K context.
Uses solar_open reasoning parser for thinking traces.
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, RetryError

sys.path.insert(0, str(Path(__file__).parent.parent))

from rok_fortress_eval.core.data_loader import load_and_expand_csv
from rok_fortress_eval.models import PromptVariant


# Solar-Open config - HuggingFace recommended params
MODEL_NAME = "upstage/Solar-Open-100B"
TEMPERATURE = 0.8  # HuggingFace recommended
TOP_P = 0.95  # HuggingFace recommended
TOP_K = 50  # HuggingFace recommended


class APIError(Exception):
    pass


def extract_error_message(e: Exception) -> str:
    """Extract meaningful error message from exception."""
    if isinstance(e, RetryError) and hasattr(e, 'last_attempt'):
        try:
            underlying = e.last_attempt.exception()
            if underlying:
                return f"{type(underlying).__name__}: {str(underlying)}"
        except Exception:
            pass
    return f"{type(e).__name__}: {str(e)}"


BackfillIndex = Dict[Tuple[str, str], dict]

VALID_PROMPT_TYPES = {"adversarial", "benign", "direct"}


def load_backfill_index(
    backfill_dir: Path,
    short_name: str,
) -> Tuple[BackfillIndex, Optional[str]]:
    """Load existing responses from backfill directory."""
    index: BackfillIndex = {}
    candidates = [
        (f"responses_{short_name}_with_reasoning.jsonl", "with_reasoning"),
        (f"responses_{short_name}_no_reasoning.jsonl", "no_reasoning"),
        (f"responses_{short_name}.jsonl", "no_reasoning"),
    ]
    for filename, suffix in candidates:
        jsonl_file = backfill_dir / filename
        if jsonl_file.exists():
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        key = (record["task_id"], record["variant_name"])
                        index[key] = record
                    except (json.JSONDecodeError, KeyError):
                        continue
            print(f"  Backfill: loaded {len(index)} records from {jsonl_file.name}")
            return index, suffix
    return index, None


@dataclass
class ResponseRecord:
    task_id: str
    variant_name: str
    task_type: str
    risk_domain: str
    risk_subdomain: str
    model_name: str
    reasoning_mode: str
    adversarial: Dict
    benign: Dict
    rubric: List[str]
    timestamp: str
    direct: Optional[Dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.direct is None:
            d.pop("direct", None)
        return d


class ModelLogger:
    def __init__(self, log_path: Path, model_name: str):
        self.log_path = log_path
        self.model_name = model_name
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, task_id: str, variant: str, status: str, latency_ms: float,
            tokens: Optional[int], has_reasoning: bool, error: Optional[str] = None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        reasoning_str = " + reasoning" if has_reasoning else ""
        token_str = f"{tokens} tokens{reasoning_str}" if tokens else "N/A"
        error_str = f" | {error}" if error else ""
        line = f"{ts} | {task_id} | {variant} | {status} | {latency_ms:.1f}ms | {token_str}{error_str}\n"
        with open(self.log_path, "a") as f:
            f.write(line)


class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.errors = 0
        self.start_time = time.time()
        self._lock = asyncio.Lock()

    async def increment(self, has_error: bool = False):
        async with self._lock:
            self.completed += 1
            if has_error:
                self.errors += 1

    def get_status(self) -> str:
        elapsed = time.time() - self.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        pct = 100 * self.completed / self.total if self.total > 0 else 0
        return f"[PROGRESS] {self.completed}/{self.total} ({pct:.1f}%) | Errors: {self.errors} | Elapsed: {mins}m {secs}s"


@retry(
    retry=retry_if_exception_type((APIError, asyncio.TimeoutError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
)
async def call_solar(
    client: AsyncOpenAI,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> Tuple[str, float, Optional[int], Optional[str]]:
    """Call Solar-Open with reasoning support."""

    request_kwargs = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": max_tokens,
        "extra_body": {
            "top_k": TOP_K,  # OpenAI API doesn't have top_k natively
        },
    }

    start = time.time()
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(**request_kwargs),
            timeout=timeout
        )
        latency_ms = (time.time() - start) * 1000

        content = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else None

        # Extract reasoning content (via solar_open parser)
        reasoning_content = None
        msg = resp.choices[0].message
        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
            reasoning_content = msg.reasoning_content

        return content, latency_ms, tokens, reasoning_content

    except asyncio.TimeoutError:
        raise APIError(f"Timeout after {timeout}s")
    except Exception as e:
        raise APIError(f"API error: {e}")


async def process_variant(
    client: AsyncOpenAI,
    variant: PromptVariant,
    timeout: float,
    logger: ModelLogger,
    progress: ProgressTracker,
    reasoning_mode: bool = False,
    prompt_types: Set[str] = None,
    backfill: Optional[BackfillIndex] = None,
) -> ResponseRecord:
    """Process a single variant with optional backfill and direct support."""

    if prompt_types is None:
        prompt_types = {"adversarial", "benign"}

    timestamp = datetime.now().isoformat()
    max_tokens = 8192
    model_name = f"solar/{MODEL_NAME}"
    reasoning_mode_str = "with_reasoning" if reasoning_mode else "no_reasoning"

    # Check backfill
    backfill_record = None
    if backfill:
        backfill_record = backfill.get((variant.task_id, variant.variant_name))
        if backfill_record:
            model_name = backfill_record.get("model_name", model_name)
            reasoning_mode_str = backfill_record.get("reasoning_mode", reasoning_mode_str)

    # Adversarial
    adv_dict = None
    if "adversarial" in prompt_types:
        if backfill_record and backfill_record.get("adversarial"):
            adv_dict = backfill_record["adversarial"]
        else:
            adv_response, adv_reasoning, adv_latency, adv_tokens, adv_error = "", None, 0.0, None, None
            try:
                adv_response, adv_latency, adv_tokens, adv_reasoning = await call_solar(
                    client=client, prompt=variant.adversarial_prompt,
                    max_tokens=max_tokens, timeout=timeout,
                )
            except Exception as e:
                adv_error = extract_error_message(e)
            adv_dict = {"prompt": variant.adversarial_prompt, "response": adv_response,
                        "reasoning_content": adv_reasoning, "latency_ms": adv_latency,
                        "tokens": adv_tokens, "error": adv_error}
    elif backfill_record and backfill_record.get("adversarial"):
        adv_dict = backfill_record["adversarial"]

    # Benign
    ben_dict = None
    if "benign" in prompt_types:
        if backfill_record and backfill_record.get("benign"):
            ben_dict = backfill_record["benign"]
        elif variant.benign_prompt:
            ben_response, ben_reasoning, ben_latency, ben_tokens, ben_error = "", None, 0.0, None, None
            try:
                ben_response, ben_latency, ben_tokens, ben_reasoning = await call_solar(
                    client=client, prompt=variant.benign_prompt,
                    max_tokens=max_tokens, timeout=timeout,
                )
            except Exception as e:
                ben_error = extract_error_message(e)
            ben_dict = {"prompt": variant.benign_prompt, "response": ben_response,
                        "reasoning_content": ben_reasoning, "latency_ms": ben_latency,
                        "tokens": ben_tokens, "error": ben_error}
    elif backfill_record and backfill_record.get("benign"):
        ben_dict = backfill_record["benign"]

    # Direct (always fresh, never backfilled)
    dir_dict = None
    if "direct" in prompt_types and variant.direct_prompt:
        dir_response, dir_reasoning, dir_latency, dir_tokens, dir_error = "", None, 0.0, None, None
        try:
            dir_response, dir_latency, dir_tokens, dir_reasoning = await call_solar(
                client=client, prompt=variant.direct_prompt,
                max_tokens=max_tokens, timeout=timeout,
            )
        except Exception as e:
            dir_error = extract_error_message(e)
        dir_dict = {"prompt": variant.direct_prompt, "response": dir_response,
                    "reasoning_content": dir_reasoning, "latency_ms": dir_latency,
                    "tokens": dir_tokens, "error": dir_error}

    # Log result
    all_subs = [adv_dict, ben_dict, dir_dict]
    has_error = any(s and s.get("error") for s in all_subs)
    has_reasoning = any(s and s.get("reasoning_content") for s in all_subs)
    total_latency = sum(s.get("latency_ms", 0) for s in all_subs if s)
    total_tokens = None
    for s in all_subs:
        if s and s.get("tokens"):
            total_tokens = (total_tokens or 0) + s["tokens"]

    logger.log(
        task_id=variant.task_id, variant=variant.variant_name,
        status="✗" if has_error else "✓", latency_ms=total_latency,
        tokens=total_tokens, has_reasoning=has_reasoning,
        error=next((s["error"] for s in all_subs if s and s.get("error")), None),
    )
    await progress.increment(has_error=has_error)

    return ResponseRecord(
        task_id=variant.task_id, variant_name=variant.variant_name,
        task_type=variant.task_type, risk_domain=variant.risk_domain,
        risk_subdomain=variant.risk_subdomain, model_name=model_name,
        reasoning_mode=reasoning_mode_str,
        adversarial=adv_dict or {}, benign=ben_dict or {},
        rubric=variant.rubric, timestamp=timestamp, direct=dir_dict,
    )


async def run_collection(
    client: AsyncOpenAI,
    variants: List[PromptVariant],
    output_dir: Path,
    logs_dir: Path,
    concurrency: int,
    timeout: float,
    skip_keys: Set[Tuple[str, str, str]],
    reasoning_mode: bool = False,
    prompt_types: Set[str] = None,
    backfill: Optional[BackfillIndex] = None,
    output_suffix: Optional[str] = None,
):
    """Run collection for Solar-Open."""

    if prompt_types is None:
        prompt_types = {"adversarial", "benign"}

    mode_suffix = output_suffix or ("with_reasoning" if reasoning_mode else "no_reasoning")
    output_file = output_dir / f"responses_solar_open_{mode_suffix}.jsonl"
    log_path = logs_dir / f"solar_open_{mode_suffix}.log"

    logger = ModelLogger(log_path, MODEL_NAME)

    # Filter variants that need processing
    reasoning_mode_str = "with_reasoning" if reasoning_mode else "no_reasoning"
    tasks_to_run = []
    for v in variants:
        key = (v.task_id, v.variant_name, reasoning_mode_str)
        if key not in skip_keys:
            tasks_to_run.append(v)

    if not tasks_to_run:
        print(f"  All {len(variants)} variants already completed")
        return

    print(f"\n{'=' * 70}")
    print(f"COLLECTING RESPONSES: Solar-Open-100B")
    print(f"{'=' * 70}")
    print(f"  Variants to process: {len(tasks_to_run)} (skipping {len(variants) - len(tasks_to_run)} completed)")
    print(f"  Prompt types: {', '.join(sorted(prompt_types))}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Reasoning mode: {reasoning_mode}")
    print(f"  Parameters: temp={TEMPERATURE}, top_p={TOP_P}, top_k={TOP_K}")
    if backfill:
        print(f"  Backfill: {len(backfill)} records (adversarial/benign reused)")
    print(f"  Output: {output_file}\n")

    progress = ProgressTracker(len(tasks_to_run))
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_process(variant):
        async with semaphore:
            return await process_variant(
                client=client,
                variant=variant,
                timeout=timeout,
                logger=logger,
                progress=progress,
                reasoning_mode=reasoning_mode,
                prompt_types=prompt_types,
                backfill=backfill,
            )

    # Progress reporter
    async def report_progress():
        while True:
            await asyncio.sleep(10)
            print(f"  {progress.get_status()}")

    progress_task = asyncio.create_task(report_progress())

    try:
        # Process all variants
        tasks = [bounded_process(v) for v in tasks_to_run]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Write results
        with open(output_file, "a") as f:
            for result in results:
                if isinstance(result, ResponseRecord):
                    f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    finally:
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

    print(f"\n  {progress.get_status()}")
    print(f"  ✓ Completed -> {output_file.name}")


def load_skip_keys(output_dir: Path, prompt_types: Set[str] = None) -> Set[Tuple[str, str, str]]:
    """Load already-completed keys from existing output files."""
    skip_keys = set()

    for jsonl_file in output_dir.glob("responses_solar*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if prompt_types and "direct" in prompt_types:
                        if not record.get("direct"):
                            continue
                    key = (
                        record["task_id"],
                        record["variant_name"],
                        record["reasoning_mode"],
                    )
                    skip_keys.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue

    return skip_keys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect Solar-Open-100B responses on ROK-FORTRESS dataset"
    )
    parser.add_argument("--input", type=str, required=True, help="Input TSV/CSV file")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--base-url", type=str, default="http://localhost:8090/v1",
                        help="vLLM endpoint URL")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent requests")
    parser.add_argument("--timeout", type=int, default=180, help="Request timeout")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--max-rows", type=int, default=None, help="Max rows (for testing)")
    parser.add_argument("--reasoning", action="store_true",
                        help="Enable reasoning mode (requires --reasoning-parser solar_open in vLLM)")
    parser.add_argument("--prompt-types", type=str, default="adversarial,benign",
                        help="Comma-separated prompt types: adversarial,benign,direct (default: adversarial,benign)")
    parser.add_argument("--backfill-dir", type=str, default=None,
                        help="Directory with existing response JSONL files to backfill adversarial/benign from")
    return parser.parse_args()


async def main():
    args = parse_args()

    # Parse and validate prompt types
    prompt_types = set(pt.strip() for pt in args.prompt_types.split(","))
    invalid = prompt_types - VALID_PROMPT_TYPES
    if invalid:
        print(f"Error: invalid prompt types: {invalid}. Valid: {VALID_PROMPT_TYPES}")
        return

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    client = AsyncOpenAI(base_url=args.base_url, api_key="dummy")

    # Load backfill if provided
    backfill: Optional[BackfillIndex] = None
    backfill_reasoning_suffix: Optional[str] = None
    if args.backfill_dir:
        backfill_dir = Path(args.backfill_dir)
        if backfill_dir.exists():
            backfill, backfill_reasoning_suffix = load_backfill_index(backfill_dir, "solar_open")
            if not backfill:
                print("  Warning: no backfill data found for 'solar_open'")
        else:
            print(f"  Warning: backfill directory does not exist: {backfill_dir}")

    # Print banner
    mode_str = "WITH REASONING" if args.reasoning else "NO REASONING"
    print("=" * 70)
    print(f"SOLAR-OPEN-100B MODEL RESPONSE COLLECTION ({mode_str})")
    print("=" * 70)
    print(f"  Input: {args.input}")
    print(f"  Output: {output_dir}")
    print(f"  Prompt types: {', '.join(sorted(prompt_types))}")
    print(f"  Endpoint: {args.base_url}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Resume: {args.resume}")
    print(f"  Reasoning: {args.reasoning}")
    print(f"  Parameters: temp={TEMPERATURE}, top_p={TOP_P}, top_k={TOP_K}")
    if backfill:
        print(f"  Backfill: {len(backfill)} records")
    print("=" * 70)

    # Load variants
    print("\nLoading dataset...")
    variants = load_and_expand_csv(args.input, max_rows=args.max_rows)
    print(f"  Loaded {len(variants)} variants")

    if "direct" in prompt_types:
        direct_count = sum(1 for v in variants if v.direct_prompt)
        print(f"  With direct prompts: {direct_count}/{len(variants)}")

    # Load skip keys if resuming
    skip_keys: Set[Tuple[str, str, str]] = set()
    if args.resume:
        skip_keys = load_skip_keys(output_dir, prompt_types=prompt_types)
        print(f"  Resuming: found {len(skip_keys)} completed records")

    # Verify endpoint
    print("\nVerifying Solar-Open endpoint...")
    try:
        models = await client.models.list()
        print(f"  ✓ Connected to {args.base_url}")
        print(f"    Model: {models.data[0].id}")
    except Exception as e:
        print(f"  ✗ Failed to connect: {e}")
        return

    # Run collection
    await run_collection(
        client=client,
        variants=variants,
        output_dir=output_dir,
        logs_dir=logs_dir,
        concurrency=args.concurrency,
        timeout=args.timeout,
        skip_keys=skip_keys,
        reasoning_mode=args.reasoning,
        prompt_types=prompt_types,
        backfill=backfill,
        output_suffix=backfill_reasoning_suffix,
    )

    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
