#!/usr/bin/env python3
"""
Collect model responses from API models on ROK-FORTRESS dataset.

Runs all models in parallel with and without reasoning, storing responses
for downstream judge evaluation. No judge evaluation is performed.

Supports three prompt types: adversarial, benign, and direct (simplified
direct requests stripped of jailbreak tactics). Can backfill adversarial/benign
responses from existing JSONL files to avoid redundant API calls.

Usage:
    # Original: collect adversarial + benign
    uv run python scripts/collect_model_responses.py \
        --input data/rok_fortress_public.tsv \
        --outdir runs/rok_fortress_responses

    # Collect all 3 types, backfilling adv+benign from existing data
    uv run python scripts/collect_model_responses.py \
        --input data/rok_fortress_public.tsv \
        --outdir runs/rok_fortress_responses_sampled \
        --backfill-dir runs/rok_fortress_responses \
        --prompt-types adversarial,benign,direct \
        --reasoning-only
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from tenacity import RetryError

from rok_fortress_eval.core.data_loader import load_and_expand_csv
from rok_fortress_eval.core.api_client import call_model_once
from rok_fortress_eval.models import PromptVariant


def extract_error_message(e: Exception) -> str:
    """Extract meaningful error message from exception, including RetryError."""
    if isinstance(e, RetryError) and hasattr(e, 'last_attempt'):
        try:
            underlying = e.last_attempt.exception()
            if underlying:
                return f"{type(underlying).__name__}: {str(underlying)}"
        except Exception:
            pass
    return f"{type(e).__name__}: {str(e)}"


# Models to evaluate
DEFAULT_MODELS = [
    "openai/gpt-5.2",
    "openai/o4-mini",
    "anthropic/claude-opus-4-5",
    "anthropic/claude-sonnet-4-5",
    "gemini/gemini-3.1-pro-preview",
    "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0",
    "fireworks_ai/kimi-k2-thinking",
    "bedrock/qwen.qwen3-235b-a22b-2507-v1:0",
    "fireworks_ai/deepseek-v3p2",
    "bedrock/mistral.mistral-large-3-675b-instruct",
]


@dataclass
class PromptSubRecord:
    """Response data for a single prompt type (adversarial, benign, or direct)."""
    prompt: str = ""
    response: str = ""
    reasoning_content: Optional[str] = None
    latency_ms: float = 0.0
    tokens: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "reasoning_content": self.reasoning_content,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptSubRecord":
        return cls(
            prompt=d.get("prompt", ""),
            response=d.get("response", ""),
            reasoning_content=d.get("reasoning_content"),
            latency_ms=d.get("latency_ms", 0.0),
            tokens=d.get("tokens"),
            error=d.get("error"),
        )


@dataclass
class ResponseRecord:
    """Single response record for JSONL output."""
    task_id: str
    variant_name: str
    task_type: str
    risk_domain: str
    risk_subdomain: str
    model_name: str
    reasoning_mode: str
    adversarial: Optional[PromptSubRecord]
    benign: Optional[PromptSubRecord]
    direct: Optional[PromptSubRecord]
    rubric: List[str]
    timestamp: str

    def to_dict(self) -> dict:
        d = {
            "task_id": self.task_id,
            "variant_name": self.variant_name,
            "task_type": self.task_type,
            "risk_domain": self.risk_domain,
            "risk_subdomain": self.risk_subdomain,
            "model_name": self.model_name,
            "reasoning_mode": self.reasoning_mode,
            "adversarial": self.adversarial.to_dict() if self.adversarial else None,
            "benign": self.benign.to_dict() if self.benign else None,
            "rubric": self.rubric,
            "timestamp": self.timestamp,
        }
        # Only include direct field if it was collected (keeps backward compat)
        if self.direct is not None:
            d["direct"] = self.direct.to_dict()
        return d


# Override short names when the API endpoint differs from the canonical name
# used in filenames and analysis (e.g., switching providers)
MODEL_SHORT_NAME_OVERRIDES = {
    "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0": "llama4-maverick-instruct-basic",
}

MODEL_MAX_OUTPUT_TOKENS = {
    "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0": 8192,
}


def get_model_short_name(model_name: str) -> str:
    """Get short name for log/output files (e.g., 'openai/gpt-5.2' -> 'gpt-5.2')."""
    if model_name in MODEL_SHORT_NAME_OVERRIDES:
        return MODEL_SHORT_NAME_OVERRIDES[model_name]
    return model_name.split("/")[-1]


def get_temperature(model_name: str) -> float:
    """Get appropriate temperature for model based on developer recommendations."""
    # Per-model temperature settings based on official documentation
    TEMPERATURE_MAP = {
        # OpenAI: 1.0 required for reasoning models
        "openai/gpt-5.2": 1.0,
        "openai/o4-mini": 1.0,
        # Anthropic: 1.0 default
        "anthropic/claude-opus-4-5": 1.0,
        "anthropic/claude-sonnet-4-5": 1.0,
        # Gemini: 1.0 strongly recommended (lower causes looping)
        "gemini/gemini-3-pro-preview": 1.0,
        # Llama 4: 0.7 (middle of 0.5-0.7 range)
        "fireworks_ai/llama4-maverick-instruct-basic": 0.7,
        "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0": 0.7,
        # Kimi K2 Thinking: 1.0 (thinking model)
        "fireworks_ai/kimi-k2-thinking": 1.0,
        # Qwen3: 0.6 (recommended for thinking mode)
        "bedrock/qwen.qwen3-235b-a22b-2507-v1:0": 0.6,
        # DeepSeek: 1.0 (internally maps to 0.3)
        "fireworks_ai/deepseek-v3p2": 1.0,
        # Mistral: 0.7 for consistency
        "bedrock/mistral.mistral-large-3-675b-instruct": 0.7,
    }
    return TEMPERATURE_MAP.get(model_name, 1.0)  # Default to 1.0


class ModelLogger:
    """Per-model logger that writes to a dedicated log file."""

    def __init__(self, log_path: Path, model_name: str, reasoning_mode: str):
        self.log_path = log_path
        self.model_name = model_name
        self.reasoning_mode = reasoning_mode
        self.file = open(log_path, "a", encoding="utf-8")

    def log(self, task_id: str, variant: str, status: str, latency_ms: float,
            tokens: Optional[int], has_reasoning: bool = False, error: Optional[str] = None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tokens_str = f"{tokens} tokens" if tokens else "? tokens"
        reasoning_str = " + reasoning" if has_reasoning else ""

        if error:
            line = f"{timestamp} | {task_id} | {variant} | {status} | {error}\n"
        else:
            line = f"{timestamp} | {task_id} | {variant} | {status} | {latency_ms:.1f}ms | {tokens_str}{reasoning_str}\n"

        self.file.write(line)
        self.file.flush()

    def close(self):
        self.file.close()


class ProgressTracker:
    """Track overall progress across all models."""

    def __init__(self, total_per_mode: int, num_models: int):
        self.total_per_mode = total_per_mode
        self.num_models = num_models
        self.completed = {"none": 0, "medium": 0}
        self.errors = {"none": 0, "medium": 0}
        self.start_time = time.time()
        self.lock = asyncio.Lock()

    async def increment(self, mode: str, success: bool = True):
        async with self.lock:
            self.completed[mode] += 1
            if not success:
                self.errors[mode] += 1

    def get_status(self) -> str:
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

        total = self.total_per_mode * self.num_models
        none_pct = (self.completed["none"] / total * 100) if total > 0 else 0
        medium_pct = (self.completed["medium"] / total * 100) if total > 0 else 0

        return (
            f"[PROGRESS] no_reasoning: {self.completed['none']}/{total} ({none_pct:.1f}%) | "
            f"with_reasoning: {self.completed['medium']}/{total} ({medium_pct:.1f}%) | "
            f"Errors: {self.errors['none'] + self.errors['medium']} | Elapsed: {elapsed_str}"
        )


async def test_model_connectivity(model_name: str, timeout: float = 30) -> Tuple[bool, str]:
    """Test if a model is reachable with a simple prompt."""
    prompt = "What is 1+1? Answer with just the number."

    try:
        # Use call_model_once which handles LiteLLM routing correctly
        response, latency, tokens, _ = await call_model_once(
            prompt=prompt,
            model_name=model_name,
            temperature=get_temperature(model_name),
            max_tokens=50,
            timeout=timeout,
            reasoning_effort=None,
        )

        if response and response.strip():
            return True, f"OK: {response[:30].strip()}"
        else:
            return False, "Empty response"
    except asyncio.TimeoutError:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, f"Error: {extract_error_message(e)[:100]}"


async def test_all_models(models: List[str], timeout: float = 30) -> Dict[str, Tuple[bool, str]]:
    """Test connectivity for all models."""
    print("\n" + "=" * 70)
    print("API CONNECTIVITY TEST")
    print("=" * 70 + "\n")

    results = {}
    for model in models:
        success, msg = await test_model_connectivity(model, timeout)
        status = "✓" if success else "✗"
        print(f"  {status} {model:50s} | {msg}")
        results[model] = (success, msg)
        await asyncio.sleep(0.3)  # Brief pause between tests

    passed = sum(1 for s, _ in results.values() if s)
    print(f"\n  Passed: {passed}/{len(models)}")

    return results


async def _collect_prompt(
    prompt_text: str,
    model_name: str,
    temperature: float,
    reasoning_mode: Optional[str],
    timeout: float,
) -> PromptSubRecord:
    """Collect a single prompt response from a model."""
    sub = PromptSubRecord(prompt=prompt_text)
    if not prompt_text:
        return sub
    try:
        response, latency, tokens, reasoning = await call_model_once(
            prompt=prompt_text,
            model_name=model_name,
            temperature=temperature,
            max_tokens=MODEL_MAX_OUTPUT_TOKENS.get(
                model_name, 8192 if not reasoning_mode else 32768
            ),
            timeout=timeout,
            reasoning_effort=reasoning_mode,
        )
        sub.response = response
        sub.latency_ms = latency
        sub.tokens = tokens
        sub.reasoning_content = reasoning
    except Exception as e:
        sub.error = extract_error_message(e)
    return sub


BackfillIndex = Dict[Tuple[str, str], dict]  # (task_id, variant_name) -> raw JSON record


def detect_backfill_reasoning_mode(backfill_dir: Path, model_short_name: str) -> Optional[str]:
    """Detect which reasoning mode has a backfill file for a model.

    Returns "medium" if with_reasoning file exists, None if only no_reasoning,
    or None if no backfill files exist at all.
    """
    with_reasoning = backfill_dir / f"responses_{model_short_name}_with_reasoning.jsonl"
    no_reasoning = backfill_dir / f"responses_{model_short_name}_no_reasoning.jsonl"

    if with_reasoning.exists():
        return "medium"
    elif no_reasoning.exists():
        return None  # None = no reasoning mode
    return None  # Default fallback


def detect_model_reasoning_modes(
    backfill_dir: Path,
    models: List[str],
) -> Dict[str, Optional[str]]:
    """Detect per-model reasoning mode based on available backfill files.

    Returns:
        Dict mapping model_name -> reasoning_mode ("medium" or None)
    """
    modes: Dict[str, Optional[str]] = {}
    for model in models:
        short_name = get_model_short_name(model)
        mode = detect_backfill_reasoning_mode(backfill_dir, short_name)
        modes[model] = mode
    return modes


def load_backfill_index(
    backfill_dir: Path,
    model_short_name: str,
    reasoning_mode: Optional[str] = None,
) -> BackfillIndex:
    """Load existing responses from backfill directory for a specific model.

    Tries the file matching the given reasoning_mode first, then falls back
    to whichever file exists.
    """
    index: BackfillIndex = {}

    # Build candidate file list in priority order
    if reasoning_mode:
        candidates = [
            f"responses_{model_short_name}_with_reasoning.jsonl",
            f"responses_{model_short_name}_no_reasoning.jsonl",
        ]
    else:
        candidates = [
            f"responses_{model_short_name}_no_reasoning.jsonl",
            f"responses_{model_short_name}_with_reasoning.jsonl",
        ]

    for filename in candidates:
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
            break  # Use first file found

    return index


async def collect_single_response(
    variant: PromptVariant,
    model_name: str,
    reasoning_mode: Optional[str],
    timeout: float,
    logger: ModelLogger,
    prompt_types: Set[str],
    backfill: Optional[BackfillIndex] = None,
) -> ResponseRecord:
    """Collect response for a single variant from a model.

    Args:
        prompt_types: Set of prompt types to collect ('adversarial', 'benign', 'direct')
        backfill: Optional index of existing responses to reuse instead of calling API
    """
    temperature = get_temperature(model_name)
    timestamp = datetime.now().isoformat()

    adv_sub: Optional[PromptSubRecord] = None
    ben_sub: Optional[PromptSubRecord] = None
    dir_sub: Optional[PromptSubRecord] = None

    # Check backfill for adversarial/benign
    backfill_record = None
    if backfill:
        backfill_record = backfill.get((variant.task_id, variant.variant_name))

    # Adversarial
    if "adversarial" in prompt_types:
        if backfill_record and backfill_record.get("adversarial"):
            adv_sub = PromptSubRecord.from_dict(backfill_record["adversarial"])
        else:
            adv_sub = await _collect_prompt(
                variant.adversarial_prompt, model_name, temperature, reasoning_mode, timeout
            )
    elif backfill_record and backfill_record.get("adversarial"):
        adv_sub = PromptSubRecord.from_dict(backfill_record["adversarial"])

    # Benign
    if "benign" in prompt_types:
        if backfill_record and backfill_record.get("benign"):
            ben_sub = PromptSubRecord.from_dict(backfill_record["benign"])
        else:
            ben_sub = await _collect_prompt(
                variant.benign_prompt or "", model_name, temperature, reasoning_mode, timeout
            )
    elif backfill_record and backfill_record.get("benign"):
        ben_sub = PromptSubRecord.from_dict(backfill_record["benign"])

    # Direct (always fresh - never backfilled)
    if "direct" in prompt_types and variant.direct_prompt:
        dir_sub = await _collect_prompt(
            variant.direct_prompt, model_name, temperature, reasoning_mode, timeout
        )

    # Log result
    total_latency = sum(
        s.latency_ms for s in [adv_sub, ben_sub, dir_sub] if s is not None
    )
    total_tokens = None
    for s in [adv_sub, ben_sub, dir_sub]:
        if s is not None and s.tokens is not None:
            total_tokens = (total_tokens or 0) + s.tokens
    has_error = any(
        s is not None and s.error is not None
        for s in [adv_sub, ben_sub, dir_sub]
    )
    has_reasoning = any(
        s is not None and s.reasoning_content is not None
        for s in [adv_sub, ben_sub, dir_sub]
    )
    first_error = next(
        (s.error for s in [adv_sub, ben_sub, dir_sub] if s is not None and s.error),
        None,
    )

    logger.log(
        task_id=variant.task_id,
        variant=variant.variant_name,
        status="✗" if has_error else "✓",
        latency_ms=total_latency,
        tokens=total_tokens,
        has_reasoning=has_reasoning,
        error=first_error,
    )

    return ResponseRecord(
        task_id=variant.task_id,
        variant_name=variant.variant_name,
        task_type=variant.task_type,
        risk_domain=variant.risk_domain,
        risk_subdomain=variant.risk_subdomain,
        model_name=model_name,
        reasoning_mode=reasoning_mode or "none",
        adversarial=adv_sub,
        benign=ben_sub,
        direct=dir_sub,
        rubric=variant.rubric,
        timestamp=timestamp,
    )


async def run_model_collection(
    model_name: str,
    variants: List[PromptVariant],
    reasoning_mode: Optional[str],
    output_dir: Path,
    logs_dir: Path,
    concurrency: int,
    timeout: float,
    progress: ProgressTracker,
    skip_keys: Set[Tuple[str, str, str, str]],
    prompt_types: Set[str],
    backfill_dir: Optional[Path] = None,
):
    """Run collection for a single model with its own output file."""
    mode_str = reasoning_mode or "none"
    short_name = get_model_short_name(model_name)
    log_path = logs_dir / f"{short_name}_{mode_str.replace('none', 'no')}_reasoning.log"

    # Each model gets its own output file
    mode_suffix = "with_reasoning" if reasoning_mode else "no_reasoning"
    output_file = output_dir / f"responses_{short_name}_{mode_suffix}.jsonl"

    # Load backfill index if available
    backfill: Optional[BackfillIndex] = None
    if backfill_dir and backfill_dir.exists():
        backfill = load_backfill_index(backfill_dir, short_name, reasoning_mode)
        if backfill:
            print(f"    Backfill: loaded {len(backfill)} existing records for {short_name}")

    logger = ModelLogger(log_path, model_name, mode_str)
    semaphore = asyncio.Semaphore(concurrency)

    # File lock per model (each model has its own file, so no cross-model contention)
    file_lock = asyncio.Lock()

    async def process_variant(variant: PromptVariant):
        # Check if already completed
        key = (variant.task_id, variant.variant_name, model_name, mode_str)
        if key in skip_keys:
            return None

        async with semaphore:
            record = await collect_single_response(
                variant=variant,
                model_name=model_name,
                reasoning_mode=reasoning_mode,
                timeout=timeout,
                logger=logger,
                prompt_types=prompt_types,
                backfill=backfill,
            )

            # Write to model-specific file immediately
            async with file_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

            # Update progress
            has_error = any(
                s is not None and s.error is not None
                for s in [record.adversarial, record.benign, record.direct]
            )
            await progress.increment(mode_str, not has_error)

            return record

    # Process all variants
    tasks = [process_variant(v) for v in variants]
    await asyncio.gather(*tasks)

    logger.close()
    print(f"  ✓ {model_name} completed -> {output_file.name}")


async def run_collection_pass(
    models: List[str],
    variants: List[PromptVariant],
    reasoning_mode: Optional[str],
    output_dir: Path,
    logs_dir: Path,
    concurrency: int,
    timeout: float,
    progress: ProgressTracker,
    skip_keys: Set[Tuple[str, str, str, str]],
    prompt_types: Set[str],
    backfill_dir: Optional[Path] = None,
):
    """Run collection for all models in parallel."""
    mode_str = "with_reasoning (medium)" if reasoning_mode else "no_reasoning"
    print(f"\n{'=' * 70}")
    print(f"COLLECTING RESPONSES: {mode_str}")
    print(f"{'=' * 70}")
    print(f"  Models: {len(models)}")
    print(f"  Variants: {len(variants)}")
    print(f"  Prompt types: {', '.join(sorted(prompt_types))}")
    print(f"  Concurrency per model: {concurrency}")
    print(f"  Output dir: {output_dir} (one file per model)")
    if backfill_dir:
        print(f"  Backfill dir: {backfill_dir}")
    print()

    # Run all models in parallel (each model writes to its own file)
    model_tasks = [
        run_model_collection(
            model_name=model,
            variants=variants,
            reasoning_mode=reasoning_mode,
            output_dir=output_dir,
            logs_dir=logs_dir,
            concurrency=concurrency,
            timeout=timeout,
            progress=progress,
            skip_keys=skip_keys,
            prompt_types=prompt_types,
            backfill_dir=backfill_dir,
        )
        for model in models
    ]

    # Progress reporter
    async def report_progress():
        while True:
            await asyncio.sleep(10)
            print(f"  {progress.get_status()}")

    progress_task = asyncio.create_task(report_progress())

    try:
        await asyncio.gather(*model_tasks)
    finally:
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

    print(f"\n  {progress.get_status()}")


def load_skip_keys(output_dir: Path, prompt_types: Set[str] = None) -> Set[Tuple[str, str, str, str]]:
    """Load already-completed keys from existing output files."""
    skip_keys = set()

    # Scan all responses_*.jsonl files (one per model per mode)
    for jsonl_file in output_dir.glob("responses_*.jsonl"):
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
                        record["model_name"],
                        record["reasoning_mode"],
                    )
                    skip_keys.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue

    return skip_keys


VALID_PROMPT_TYPES = {"adversarial", "benign", "direct"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect model responses from API models on ROK-FORTRESS dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input TSV/CSV file",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory (default: runs/response_collection_<timestamp>)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=20,
        help="Concurrent requests per model (default: 20)",
    )
    parser.add_argument(
        "--timeout", type=int, default=180,
        help="Request timeout in seconds (default: 180)",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated list of models (default: all 10 API models)",
    )
    parser.add_argument(
        "--skip-api-test", action="store_true",
        help="Skip the initial API connectivity test",
    )
    parser.add_argument(
        "--no-reasoning-only", action="store_true",
        help="Only run the no-reasoning pass",
    )
    parser.add_argument(
        "--reasoning-only", action="store_true",
        help="Only run the reasoning pass",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output directory",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Max rows to load from dataset (for testing)",
    )
    parser.add_argument(
        "--prompt-types", type=str, default="adversarial,benign",
        help="Comma-separated prompt types to collect: adversarial,benign,direct (default: adversarial,benign)",
    )
    parser.add_argument(
        "--backfill-dir", type=str, default=None,
        help="Directory with existing response JSONL files to backfill adversarial/benign from (avoids redundant API calls)",
    )
    parser.add_argument(
        "--match-backfill-reasoning", action="store_true",
        help="Auto-detect reasoning mode per model from backfill files (with_reasoning if available, else no_reasoning)",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # Parse models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = DEFAULT_MODELS

    # Parse prompt types
    prompt_types = set(pt.strip() for pt in args.prompt_types.split(","))
    invalid = prompt_types - VALID_PROMPT_TYPES
    if invalid:
        print(f"Error: invalid prompt types: {invalid}. Valid: {VALID_PROMPT_TYPES}")
        return 1

    # Parse backfill directory
    backfill_dir = Path(args.backfill_dir) if args.backfill_dir else None
    if backfill_dir and not backfill_dir.exists():
        print(f"Warning: backfill directory does not exist: {backfill_dir}")
        backfill_dir = None

    # Setup output directory
    if args.outdir:
        output_dir = Path(args.outdir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"runs/response_collection_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Print banner
    print("=" * 70)
    print("ROK-FORTRESS MODEL RESPONSE COLLECTION")
    print("=" * 70)
    print(f"  Input: {args.input}")
    print(f"  Output: {output_dir}")
    print(f"  Models: {len(models)}")
    print(f"  Prompt types: {', '.join(sorted(prompt_types))}")
    if backfill_dir:
        print(f"  Backfill from: {backfill_dir}")
    print(f"  Concurrency: {args.concurrency} per model")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Resume: {args.resume}")
    print("=" * 70)

    # Load variants
    print("\nLoading dataset...")
    variants = load_and_expand_csv(args.input, max_rows=args.max_rows)
    print(f"  Loaded {len(variants)} variants")

    # Validate variant distribution
    from collections import Counter
    variant_counts = Counter(v.variant_name for v in variants)
    task_ids = set(v.task_id for v in variants)
    print(f"  Unique tasks: {len(task_ids)}")
    print(f"  By variant type:")
    for name, count in sorted(variant_counts.items()):
        print(f"    {name}: {count}")

    # Report direct prompt availability
    if "direct" in prompt_types:
        direct_count = sum(1 for v in variants if v.direct_prompt)
        print(f"  With direct prompts: {direct_count}/{len(variants)}")

    # Load skip keys if resuming
    skip_keys: Set[Tuple[str, str, str, str]] = set()
    if args.resume:
        skip_keys = load_skip_keys(output_dir, prompt_types=prompt_types)
        if skip_keys:
            print(f"  Resume: found {len(skip_keys)} completed records to skip")

    # API connectivity test (skip if only backfilling)
    needs_api = "direct" in prompt_types or (
        not backfill_dir and ("adversarial" in prompt_types or "benign" in prompt_types)
    )
    if not args.skip_api_test and needs_api:
        test_results = await test_all_models(models, timeout=30)

        # Save test results
        with open(output_dir / "api_test_results.json", "w") as f:
            json.dump({m: {"success": s, "message": msg} for m, (s, msg) in test_results.items()}, f, indent=2)

        # Check for failures
        failed_models = [m for m, (s, _) in test_results.items() if not s]
        if failed_models:
            print(f"\n⚠️  {len(failed_models)} models failed connectivity test:")
            for m in failed_models:
                print(f"     - {m}")

            # Auto-continue in non-interactive mode, prompt in interactive mode
            if sys.stdin.isatty():
                response = input("\nContinue anyway? [y/N]: ").strip().lower()
                if response != "y":
                    print("Aborted.")
                    return 1
            else:
                print("\n  Non-interactive mode: continuing with working models...")

    # Initialize progress tracker
    progress = ProgressTracker(total_per_mode=len(variants), num_models=len(models))

    # Determine reasoning modes
    if args.match_backfill_reasoning and backfill_dir:
        # Auto-detect per-model reasoning mode from backfill files
        model_reasoning_modes = detect_model_reasoning_modes(backfill_dir, models)

        reasoning_models = [m for m, mode in model_reasoning_modes.items() if mode == "medium"]
        no_reasoning_models = [m for m, mode in model_reasoning_modes.items() if mode is None]

        print(f"\n  Per-model reasoning modes (from backfill):")
        for m in reasoning_models:
            print(f"    with_reasoning: {get_model_short_name(m)}")
        for m in no_reasoning_models:
            print(f"    no_reasoning:   {get_model_short_name(m)}")

        # Run with_reasoning models
        if reasoning_models:
            await run_collection_pass(
                models=reasoning_models,
                variants=variants,
                reasoning_mode="medium",
                output_dir=output_dir,
                logs_dir=logs_dir,
                concurrency=args.concurrency,
                timeout=args.timeout,
                progress=progress,
                skip_keys=skip_keys,
                prompt_types=prompt_types,
                backfill_dir=backfill_dir,
            )

        # Run no_reasoning models
        if no_reasoning_models:
            await run_collection_pass(
                models=no_reasoning_models,
                variants=variants,
                reasoning_mode=None,
                output_dir=output_dir,
                logs_dir=logs_dir,
                concurrency=args.concurrency,
                timeout=args.timeout,
                progress=progress,
                skip_keys=skip_keys,
                prompt_types=prompt_types,
                backfill_dir=backfill_dir,
            )
    else:
        # Original behavior: run passes based on --reasoning-only / --no-reasoning-only flags

        # Pass 1: No reasoning
        if not args.reasoning_only:
            await run_collection_pass(
                models=models,
                variants=variants,
                reasoning_mode=None,
                output_dir=output_dir,
                logs_dir=logs_dir,
                concurrency=args.concurrency,
                timeout=args.timeout,
                progress=progress,
                skip_keys=skip_keys,
                prompt_types=prompt_types,
                backfill_dir=backfill_dir,
            )

        # Pass 2: With reasoning (medium)
        if not args.no_reasoning_only:
            await run_collection_pass(
                models=models,
                variants=variants,
                reasoning_mode="medium",
                output_dir=output_dir,
                logs_dir=logs_dir,
                concurrency=args.concurrency,
                timeout=args.timeout,
                progress=progress,
                skip_keys=skip_keys,
                prompt_types=prompt_types,
                backfill_dir=backfill_dir,
            )

    # Final summary
    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)
    print(f"  {progress.get_status()}")

    # List all output files
    output_files = sorted(output_dir.glob("responses_*.jsonl"))
    print(f"\n  Output files ({len(output_files)} total):")
    for f in output_files:
        line_count = sum(1 for _ in open(f, "r", encoding="utf-8"))
        print(f"    - {f.name} ({line_count} records)")

    print(f"\n  Log files: {logs_dir}/")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
