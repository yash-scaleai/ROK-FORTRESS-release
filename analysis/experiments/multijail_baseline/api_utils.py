"""
API utilities for MultiJail baseline experiment.
"""

import asyncio
import time
from typing import Optional, Tuple

from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

from rok_fortress_eval.core.api_client import acompletion


class APIError(Exception):
    """Custom exception for API errors."""
    pass


@retry(
    retry=retry_if_exception_type((APIError, asyncio.TimeoutError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
)
async def call_model(
    prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    system_prompt: str = "You are a helpful assistant.",
) -> Tuple[str, float, Optional[int]]:
    """Call model with retry. Routes via api_client.acompletion."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    start = time.time()
    try:
        resp = await asyncio.wait_for(
            acompletion(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            timeout=timeout,
        )
        latency_ms = (time.time() - start) * 1000

        content = resp.choices[0].message.content or ""
        tokens = None
        if hasattr(resp, "usage") and resp.usage:
            tokens = resp.usage.total_tokens

        return content, latency_ms, tokens

    except asyncio.TimeoutError:
        raise APIError(f"Timeout after {timeout}s")
    except Exception as e:
        raise APIError(f"API error: {e}")
