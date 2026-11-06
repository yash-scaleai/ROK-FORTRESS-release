#!/usr/bin/env python3
"""
Collect model responses from vLLM-hosted models on ROK-FORTRESS dataset.

Launches a vLLM server, collects responses for all variants, and outputs
to JSONL format compatible with the downstream judge evaluation pipeline.

Supports three prompt types (adversarial, benign, direct) and can backfill
adversarial/benign from existing JSONL files to produce merged output with
all three fields.

Usage:
<<<<<<<< HEAD:analysis/experiments/multijail_baseline/collect_vllm_responses.py
    # Single model on specific GPU
    CUDA_VISIBLE_DEVICES=1 uv run python -m analysis.experiments.multijail_baseline.collect_vllm_responses \
========
    # Original: collect adversarial + benign (backward compatible)
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/collect_vllm_responses.py \
>>>>>>>> origin/direct-requests:scripts/collect_vllm_responses.py
        --model meta-llama/Llama-3.1-8B-Instruct \
        --port 8001 \
        --input data/rok_fortress_public.tsv \
        --outdir runs/rok_fortress_responses \
        --concurrency 100

<<<<<<<< HEAD:analysis/experiments/multijail_baseline/collect_vllm_responses.py
    # With custom max context length
    CUDA_VISIBLE_DEVICES=2 uv run python -m analysis.experiments.multijail_baseline.collect_vllm_responses \
        --model google/gemma-2-9b-it \
        --port 8002 \
        --outdir results/multijail_baseline/vllm_responses \
        --max-model-len 8192
========
    # Collect direct requests, backfilling adv+benign from existing data
    CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/collect_vllm_responses.py \
        --model naver-hyperclovax/HyperCLOVAX-SEED-Think-32B \
        --input data/rok_fortress_public.tsv \
        --outdir runs/rok_fortress_responses_full \
        --backfill-dir runs/rok_fortress_responses \
        --short-name hyperclovax \
        --prompt-types direct \
        --port 8000
>>>>>>>> origin/direct-requests:scripts/collect_vllm_responses.py
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from rok_fortress_eval.core.data_loader import load_and_expand_csv
from rok_fortress_eval.models import PromptVariant
from rok_fortress_eval.vllm.model_configs import get_model_config


@dataclass
class PromptSubRecord:
    """Response data for a single prompt type."""
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


def get_model_short_name(model_name: str) -> str:
    """Get short name for file naming (e.g., 'meta-llama/Llama-3.1-8B-Instruct' -> 'Llama-3.1-8B-Instruct')."""
    return model_name.split("/")[-1]


class VLLMServer:
    """Manages a vLLM server process."""

    def __init__(
        self,
        model: str,
        port: int,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = False,
        enforce_eager: bool = False,
        use_v0_engine: bool = False,
        attention_backend: Optional[str] = None,
        log_file: Optional[Path] = None,
    ):
        self.model = model
        self.port = port
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.enforce_eager = enforce_eager
        self.use_v0_engine = use_v0_engine
        self.attention_backend = attention_backend
        self.log_file = log_file
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://localhost:{port}"

    def launch(self) -> None:
        """Launch the vLLM server."""
        cmd = [
            "vllm", "serve", self.model,
            "--port", str(self.port),
            "--dtype", "auto",
            "--gpu-memory-utilization", "0.9",
        ]

        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])

        if self.trust_remote_code:
            cmd.append("--trust-remote-code")

        if self.enforce_eager:
            cmd.append("--enforce-eager")

        print(f"  Launching vLLM server: {' '.join(cmd)}")

        # Set up environment
        env = os.environ.copy()

        # Handle special environment variables for compatibility
        if self.use_v0_engine:
            env["VLLM_USE_V1"] = "0"
            print(f"  Using V0 engine (VLLM_USE_V1=0)")

        if self.attention_backend:
            env["VLLM_ATTENTION_BACKEND"] = self.attention_backend
            print(f"  Using attention backend: {self.attention_backend}")

        # Set torchinductor cache dir to avoid /tmp noexec issues
        cache_dir = os.path.expanduser("~/.cache/torchinductor")
        os.makedirs(cache_dir, exist_ok=True)
        env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

        # Open log file if specified
        log_handle = None
        if self.log_file:
            log_handle = open(self.log_file, "w")
            print(f"  Server logs: {self.log_file}")

        self.process = subprocess.Popen(
            cmd,
            stdout=log_handle or subprocess.DEVNULL,
            stderr=subprocess.STDOUT if log_handle else subprocess.DEVNULL,
            env=env,
        )

    async def wait_for_health(self, timeout: float = 300) -> bool:
        """Wait for the server to become healthy."""
        health_url = f"{self.base_url}/health"
        start = time.time()

        print(f"  Waiting for server health at {health_url}...")

        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout:
                try:
                    response = await client.get(health_url, timeout=5)
                    if response.status_code == 200:
                        elapsed = time.time() - start
                        print(f"  Server healthy after {elapsed:.1f}s")
                        return True
                except (httpx.RequestError, httpx.HTTPStatusError):
                    pass

                await asyncio.sleep(2)

        print(f"  ERROR: Server did not become healthy within {timeout}s")
        return False

    def stop(self) -> None:
        """Stop the vLLM server."""
        if self.process:
            print("  Stopping vLLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


class VLLMClient:
    """Client for making requests to a vLLM server."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.chat_url = f"{base_url}/v1/chat/completions"

    async def chat(
        self,
        prompt: str,
        max_tokens: int = 8192,  # Reduced from 8192 to fit within context limits
        temperature: float = 1.0,
        timeout: float = 180,
        stop_tokens: Optional[List[str]] = None,
    ) -> Tuple[str, float, int]:
        """Send a chat request and return (response, latency_ms, tokens)."""
        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if stop_tokens:
            payload["stop"] = stop_tokens

        start = time.time()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.chat_url,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()

        latency_ms = (time.time() - start) * 1000
        data = response.json()

        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", 0)

        return text, latency_ms, tokens


class ProgressTracker:
    """Track collection progress."""

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

        rate = self.completed / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.completed) / rate if rate > 0 else 0
        eta_str = f"{int(remaining // 60)}m {int(remaining % 60)}s" if rate > 0 else "?"

        return (
            f"Progress: {self.completed}/{self.total} ({pct:.1f}%) | "
            f"Errors: {self.errors} | Elapsed: {elapsed_str} | ETA: {eta_str}"
        )


BackfillIndex = Dict[Tuple[str, str], dict]


def load_backfill_index(
    backfill_dir: Path,
    short_name: str,
) -> Tuple[BackfillIndex, Optional[str]]:
    """Load existing responses from backfill directory for a specific model.

    Scans for responses_{short_name}_*.jsonl files and loads the first match.

    Returns:
        Tuple of (index keyed by (task_id, variant_name), reasoning_suffix)
        where reasoning_suffix is e.g. "with_reasoning" or "no_reasoning".
    """
    index: BackfillIndex = {}

    # Try with_reasoning first, then no_reasoning
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


async def _collect_prompt_from_vllm(
    prompt_text: str,
    client: VLLMClient,
    max_tokens: int,
    temperature: float,
    timeout: float,
    stop_tokens: Optional[List[str]],
) -> PromptSubRecord:
    """Collect a single prompt response from the vLLM server."""
    sub = PromptSubRecord(prompt=prompt_text)
    if not prompt_text:
        return sub
    try:
        response, latency, tokens = await client.chat(
            prompt=prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            stop_tokens=stop_tokens,
        )
        sub.response = response
        sub.latency_ms = latency
        sub.tokens = tokens if tokens else None
    except Exception as e:
        sub.error = f"{type(e).__name__}: {str(e)}"
    return sub


async def collect_single_response(
    variant: PromptVariant,
    client: Optional[VLLMClient],
    model_name: str,
    timeout: float,
    stop_tokens: Optional[List[str]],
    prompt_types: Set[str],
    backfill: Optional[BackfillIndex] = None,
    max_tokens: int = 8192,
    temperature: float = 1.0,
) -> ResponseRecord:
    """Collect response for a single variant.

    Args:
        prompt_types: Set of prompt types to collect from vLLM ('adversarial', 'benign', 'direct')
        backfill: Optional index of existing responses to reuse for adversarial/benign
    """
    timestamp = datetime.now().isoformat()

    adv_sub: Optional[PromptSubRecord] = None
    ben_sub: Optional[PromptSubRecord] = None
    dir_sub: Optional[PromptSubRecord] = None
    reasoning_mode = "none"

    # Check backfill for existing data
    backfill_record = None
    if backfill:
        backfill_record = backfill.get((variant.task_id, variant.variant_name))
        if backfill_record:
            reasoning_mode = backfill_record.get("reasoning_mode", "none")
            # Preserve original model_name from backfill for consistency
            model_name = backfill_record.get("model_name", model_name)

    # Adversarial
    if "adversarial" in prompt_types:
        if backfill_record and backfill_record.get("adversarial"):
            adv_sub = PromptSubRecord.from_dict(backfill_record["adversarial"])
        elif client:
            adv_sub = await _collect_prompt_from_vllm(
                variant.adversarial_prompt, client, max_tokens, temperature, timeout, stop_tokens,
            )
    elif backfill_record and backfill_record.get("adversarial"):
        # Not in prompt_types but backfill has it -- include for merged output
        adv_sub = PromptSubRecord.from_dict(backfill_record["adversarial"])

    # Benign
    if "benign" in prompt_types:
        if backfill_record and backfill_record.get("benign"):
            ben_sub = PromptSubRecord.from_dict(backfill_record["benign"])
        elif client and variant.benign_prompt:
            ben_sub = await _collect_prompt_from_vllm(
                variant.benign_prompt, client, max_tokens, temperature, timeout, stop_tokens,
            )
    elif backfill_record and backfill_record.get("benign"):
        ben_sub = PromptSubRecord.from_dict(backfill_record["benign"])

    # Direct (always collected fresh from vLLM, never backfilled)
    if "direct" in prompt_types and variant.direct_prompt and client:
        dir_sub = await _collect_prompt_from_vllm(
            variant.direct_prompt, client, max_tokens, temperature, timeout, stop_tokens,
        )

    return ResponseRecord(
        task_id=variant.task_id,
        variant_name=variant.variant_name,
        task_type=variant.task_type,
        risk_domain=variant.risk_domain,
        risk_subdomain=variant.risk_subdomain,
        model_name=model_name,
        reasoning_mode=reasoning_mode,
        adversarial=adv_sub,
        benign=ben_sub,
        direct=dir_sub,
        rubric=variant.rubric,
        timestamp=timestamp,
    )


async def run_collection(
    variants: List[PromptVariant],
    client: Optional[VLLMClient],
    model_name: str,
    output_file: Path,
    concurrency: int,
    timeout: float,
    stop_tokens: Optional[List[str]],
    skip_keys: Set[Tuple[str, str]],
    prompt_types: Set[str],
    backfill: Optional[BackfillIndex] = None,
    max_tokens: int = 8192,
    temperature: float = 1.0,
) -> None:
    """Run collection for all variants."""
    progress = ProgressTracker(total=len(variants))
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()

    async def process_variant(variant: PromptVariant):
        # Check if already completed
        key = (variant.task_id, variant.variant_name)
        if key in skip_keys:
            return

        async with semaphore:
            record = await collect_single_response(
                variant=variant,
                client=client,
                model_name=model_name,
                timeout=timeout,
                stop_tokens=stop_tokens,
                prompt_types=prompt_types,
                backfill=backfill,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Write to output file immediately
            async with file_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

            # Determine success: check all collected prompt types for errors
            has_error = any(
                s is not None and s.error is not None
                for s in [record.adversarial, record.benign, record.direct]
            )
            await progress.increment(not has_error)

    # Process all variants
    tasks = [process_variant(v) for v in variants]

    # Progress reporter
    async def report_progress():
        while True:
            await asyncio.sleep(10)
            print(f"  {progress.get_status()}")

    progress_task = asyncio.create_task(report_progress())

    try:
        await asyncio.gather(*tasks)
    finally:
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

    print(f"\n  Final: {progress.get_status()}")


def load_skip_keys(output_file: Path) -> Set[Tuple[str, str]]:
    """Load already-completed keys from existing output file."""
    skip_keys = set()

    if not output_file.exists():
        return skip_keys

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                key = (record["task_id"], record["variant_name"])
                skip_keys.add(key)
            except (json.JSONDecodeError, KeyError):
                continue

    return skip_keys


VALID_PROMPT_TYPES = {"adversarial", "benign", "direct"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect model responses from vLLM-hosted models on ROK-FORTRESS dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for vLLM server (default: 8000)",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input TSV/CSV file",
    )
    parser.add_argument(
        "--outdir", type=str, required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--concurrency", type=int, default=100,
        help="Concurrent requests (default: 100)",
    )
    parser.add_argument(
        "--timeout", type=int, default=180,
        help="Request timeout in seconds (default: 180)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None,
        help="Maximum context length (default: from model config)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8192,
        help="Maximum output tokens per response (default: 8192)",
    )
    parser.add_argument(
        "--skip-server-launch", action="store_true",
        help="Skip launching vLLM server (use existing server)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file",
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
        help="Directory with existing response JSONL files to backfill adversarial/benign from",
    )
    parser.add_argument(
        "--short-name", type=str, default=None,
        help="Override short name for file naming (default: last segment of --model)",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # Parse and validate prompt types
    prompt_types = set(pt.strip() for pt in args.prompt_types.split(","))
    invalid = prompt_types - VALID_PROMPT_TYPES
    if invalid:
        print(f"Error: invalid prompt types: {invalid}. Valid: {VALID_PROMPT_TYPES}")
        return 1

    # Determine if we need a vLLM server (only if collecting fresh responses)
    needs_vllm = bool(prompt_types - {"adversarial", "benign"}) or not args.backfill_dir

    # Load environment variables (for HF_TOKEN)
    load_dotenv()

    # Verify HF_TOKEN for gated models
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if "gemma" in args.model.lower() and not hf_token:
        print("ERROR: Gemma 2 is a gated model requiring HF_TOKEN")
        print("  Set HF_TOKEN in .env file or environment")
        return 1

    # Setup directories
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Get model config
    model_config = get_model_config(args.model)
    short_name = args.short_name or get_model_short_name(args.model)

    # Determine max_model_len
    max_model_len = args.max_model_len or model_config.max_model_len or 8192

    # Check for trust_remote_code
    trust_remote_code = model_config.extra_launch_args.get("trust_remote_code", False)
    if "exaone" in args.model.lower():
        trust_remote_code = True

    # Check for enforce_eager
    enforce_eager = model_config.extra_launch_args.get("enforce_eager", False)

    # Gemma 2 requires special handling due to softcapping in attention
    # It needs TRITON_ATTN backend and V0 engine
    use_v0_engine = False
    attention_backend = None
    if "gemma-2" in args.model.lower() or "gemma2" in args.model.lower():
        use_v0_engine = True
        attention_backend = "TRITON_ATTN"
        enforce_eager = True

    # Load backfill index if provided
    backfill: Optional[BackfillIndex] = None
    backfill_reasoning_suffix: Optional[str] = None
    backfill_dir = Path(args.backfill_dir) if args.backfill_dir else None
    if backfill_dir:
        if not backfill_dir.exists():
            print(f"Warning: backfill directory does not exist: {backfill_dir}")
            backfill_dir = None
        else:
            backfill, backfill_reasoning_suffix = load_backfill_index(
                backfill_dir, short_name,
            )
            if not backfill:
                print(f"  Warning: no backfill data found for '{short_name}' in {backfill_dir}")

    # Output file naming:
    # - With backfill: match the backfill file's reasoning suffix
    # - Default: _no_reasoning
    reasoning_suffix = backfill_reasoning_suffix or "no_reasoning"
    output_file = output_dir / f"responses_{short_name}_{reasoning_suffix}.jsonl"

    # Print banner
    print("=" * 70)
    print("ROK-FORTRESS vLLM RESPONSE COLLECTION")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Short name: {short_name}")
    print(f"  Prompt types: {', '.join(sorted(prompt_types))}")
    print(f"  Port: {args.port}")
    print(f"  Input: {args.input}")
    print(f"  Output: {output_file}")
    if backfill_dir:
        print(f"  Backfill dir: {backfill_dir}")
        print(f"  Backfill records: {len(backfill) if backfill else 0}")
    print(f"  Max model length: {max_model_len}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Trust remote code: {trust_remote_code}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Needs vLLM: {needs_vllm}")
    print(f"  HF_TOKEN: {'set' if hf_token else 'NOT SET'}")
    print("=" * 70)

    # Load variants
    print("\nLoading dataset...")
    variants = load_and_expand_csv(args.input, max_rows=args.max_rows)
    print(f"  Loaded {len(variants)} variants")

    # Report direct prompt availability
    if "direct" in prompt_types:
        direct_count = sum(1 for v in variants if v.direct_prompt)
        print(f"  With direct prompts: {direct_count}/{len(variants)}")

    # Load skip keys if resuming
    skip_keys: Set[Tuple[str, str]] = set()
    if args.resume:
        skip_keys = load_skip_keys(output_file)
        if skip_keys:
            print(f"  Resume: found {len(skip_keys)} completed records to skip")

    # Calculate how many to process
    to_process = len(variants) - len(skip_keys)
    print(f"  To process: {to_process}")

    if to_process == 0:
        print("\n  All variants already processed. Nothing to do.")
        return 0

    # Launch or connect to vLLM server (only if needed)
    server = None
    client = None
    if needs_vllm:
        if not args.skip_server_launch:
            print("\n" + "=" * 70)
            print("LAUNCHING vLLM SERVER")
            print("=" * 70)

            log_file = logs_dir / f"{short_name}_vllm.log"
            server = VLLMServer(
                model=args.model,
                port=args.port,
                max_model_len=max_model_len,
                trust_remote_code=trust_remote_code,
                enforce_eager=enforce_eager,
                use_v0_engine=use_v0_engine,
                attention_backend=attention_backend,
                log_file=log_file,
            )

            server.launch()

            if not await server.wait_for_health(timeout=300):
                print("ERROR: Server failed to start")
                server.stop()
                return 1

        # Create client
        base_url = f"http://localhost:{args.port}"
        client = VLLMClient(base_url=base_url, model=args.model)

    # Get stop tokens and temperature from model config
    stop_tokens = model_config.stop_tokens
    temperature = model_config.default_temperature

    # Run collection
    print("\n" + "=" * 70)
    print("COLLECTING RESPONSES")
    print("=" * 70)
    print(f"  Variants: {len(variants)}")
    print(f"  Prompt types: {', '.join(sorted(prompt_types))}")
    print(f"  Stop tokens: {stop_tokens}")
    print(f"  Temperature: {temperature}")
    if backfill:
        print(f"  Backfill: {len(backfill)} records (adversarial/benign reused)")
    print("")

    try:
        await run_collection(
            variants=variants,
            client=client,
            model_name=f"vllm/{args.model}",
            output_file=output_file,
            concurrency=args.concurrency,
            timeout=args.timeout,
            stop_tokens=stop_tokens,
            skip_keys=skip_keys,
            prompt_types=prompt_types,
            backfill=backfill,
            max_tokens=args.max_tokens,
            temperature=temperature,
        )
    finally:
        if server:
            server.stop()

    # Final summary
    print("\n" + "=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)

    line_count = sum(1 for _ in open(output_file, "r", encoding="utf-8"))
    print(f"  Output: {output_file}")
    print(f"  Records: {line_count}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)