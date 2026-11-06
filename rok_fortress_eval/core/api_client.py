"""
Async API client for model calls with retry logic.

Two routing modes, selected by whether ``BASE_URL`` is set in the environment:

* **Proxy mode** (``BASE_URL`` set) -- the OpenAI SDK talks to a single
  OpenAI-compatible endpoint (e.g. a LiteLLM proxy you launched) and the
  provider/model prefix is forwarded as the model name. This is what
  produced the paper artifacts.
* **Direct mode** (``BASE_URL`` unset) -- ``litellm.acompletion`` reads
  per-provider keys from the environment (``OPENAI_API_KEY``,
  ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``, ``FIREWORKS_AI_API_KEY``,
  ``AWS_*`` for Bedrock, ...) and dispatches to each provider's API
  natively. No proxy required.

Local vLLM servers (``vllm/`` prefix) are routed through
``rok_fortress_eval.vllm.router`` instead and are not handled here.
"""

import asyncio
import time
from typing import Optional, Tuple

import litellm
from openai import AsyncOpenAI
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    RetryError,
)

from .config import API_KEY, BASE_URL, DEFAULT_TIMEOUT, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from ..models import ModelResponse, PromptVariant


class APIError(Exception):
    """Custom exception for API errors"""
    pass


# Proxy client is built only when BASE_URL is set; otherwise direct litellm.
client = (
    AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY or "MISSING_API_KEY")
    if BASE_URL
    else None
)


async def acompletion(**kwargs):
    """Single dispatch point for chat completions.

    Routes to the configured proxy when ``BASE_URL`` is set; otherwise calls
    ``litellm.acompletion`` directly. The litellm SDK takes provider-specific
    reasoning kwargs (``thinking``, ``reasoning_effort``) as top-level args
    rather than wrapped in ``extra_body``, so we unwrap when going direct.
    """
    if client is not None:
        return await client.chat.completions.create(**kwargs)
    extra = kwargs.pop("extra_body", None)
    if extra:
        # ``allowed_openai_params`` is OpenAI-SDK metadata; meaningless to litellm.
        extra.pop("allowed_openai_params", None)
        kwargs.update(extra)
    return await litellm.acompletion(**kwargs)


@retry(
    retry=retry_if_exception_type((APIError, asyncio.TimeoutError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
)
async def call_model_once(
    prompt: str,
    model_name: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT,
    system_prompt: str = "You are a helpful assistant.",
    reasoning_effort: Optional[str] = None,
) -> Tuple[str, float, Optional[int], Optional[str]]:
    """
    Call model once with retry logic.
    
    Args:
        prompt: User prompt
        model_name: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        system_prompt: System message
        reasoning_effort: Reasoning mode - "low", "medium", "high", or None (disabled)
        
    Returns:
        Tuple of (response_text, latency_ms, tokens_used, reasoning_content)
        
    Raises:
        APIError: If API call fails after retries
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    # Build request kwargs
    request_kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    # Add reasoning/thinking if specified
    # Different providers need different configurations
    # Following fortress pattern: budget_tokens=16384, max_tokens=32768 for medium
    if reasoning_effort:
        # Consistent budget mapping across all providers (fortress pattern)
        budget_map = {"low": 8192, "medium": 16384, "high": 24576}
        max_tokens_map = {"low": 16384, "medium": 32768, "high": 49152}
        reasoning_max = max_tokens_map.get(reasoning_effort, 32768)
        # Respect caller's cap if they set a lower limit (e.g. model-specific limits)
        capped_max = min(reasoning_max, max_tokens) if max_tokens < reasoning_max else reasoning_max
        
        if model_name.startswith("anthropic/"):
            # Anthropic extended thinking: use thinking block with budget_tokens
            budget_tokens = budget_map.get(reasoning_effort, 16384)
            request_kwargs["max_tokens"] = capped_max
            request_kwargs["extra_body"] = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": min(budget_tokens, capped_max - 1),
                },
                "allowed_openai_params": ["thinking"],
            }
        elif model_name.startswith("openai/"):
            # OpenAI reasoning models (o4-mini, gpt-5.2)
            request_kwargs["max_tokens"] = capped_max
            request_kwargs["extra_body"] = {
                "reasoning_effort": reasoning_effort,
            }
        elif model_name.startswith("gemini/"):
            # Gemini 3 uses thinking_level (not thinking_budget)
            level_map = {"low": "low", "medium": "medium", "high": "high"}
            request_kwargs["max_tokens"] = capped_max
            request_kwargs["extra_body"] = {
                "thinking": {"thinking_level": level_map.get(reasoning_effort, "medium")},
            }
        elif "deepseek" in model_name:
            request_kwargs["max_tokens"] = capped_max
        elif "kimi" in model_name:
            request_kwargs["max_tokens"] = capped_max
        elif "qwen" in model_name.lower():
            request_kwargs["max_tokens"] = capped_max
        else:
            # llama4, mistral, etc.
            request_kwargs["max_tokens"] = capped_max
    
    start = time.time()
    try:
        resp = await asyncio.wait_for(
            acompletion(**request_kwargs),
            timeout=timeout
        )
        latency_ms = (time.time() - start) * 1000
        
        content = resp.choices[0].message.content or ""
        tokens = None
        if hasattr(resp, 'usage') and resp.usage:
            tokens = resp.usage.total_tokens
        
        # Extract reasoning_content if present (from thinking models)
        # Different providers return this in different fields
        reasoning_content = None
        msg = resp.choices[0].message

        # Try direct reasoning_content field (Anthropic, DeepSeek, Kimi)
        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
            reasoning_content = msg.reasoning_content
        # Try thinking_blocks (Anthropic alternative format)
        elif hasattr(msg, 'thinking_blocks') and msg.thinking_blocks:
            thinking_texts = [b.get('thinking', '') for b in msg.thinking_blocks if isinstance(b, dict)]
            if thinking_texts:
                reasoning_content = '\n'.join(thinking_texts)
        # Try reasoning field (OpenAI format)
        elif hasattr(msg, 'reasoning') and msg.reasoning:
            if isinstance(msg.reasoning, str):
                reasoning_content = msg.reasoning
            elif isinstance(msg.reasoning, dict) and msg.reasoning.get('content'):
                reasoning_content = msg.reasoning['content']

        return content, latency_ms, tokens, reasoning_content
        
    except asyncio.TimeoutError:
        raise APIError(f"Timeout after {timeout}s")
    except Exception as e:
        raise APIError(f"API error: {e}")


async def evaluate_variant_with_model(
    variant: PromptVariant,
    model_name: str,
    include_benign: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT,
    reasoning_effort: Optional[str] = None,
) -> Tuple[ModelResponse, Optional[ModelResponse]]:
    """
    Evaluate a single variant with a model (adversarial + optional benign).
    
    Args:
        variant: Prompt variant to evaluate
        model_name: Model to evaluate with
        include_benign: Whether to also evaluate benign prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        timeout: Request timeout
        reasoning_effort: Reasoning mode - "low", "medium", "high", or None (disabled)
        
    Returns:
        Tuple of (adversarial_response, benign_response)
    """
    
    # Adversarial
    try:
        adv_text, adv_latency, adv_tokens, adv_reasoning = await call_model_once(
            prompt=variant.adversarial_prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
        )
        adv_response = ModelResponse(
            model_name=model_name,
            prompt_type="adversarial",
            response_text=adv_text,
            latency_ms=adv_latency,
            tokens_used=adv_tokens,
            reasoning_content=adv_reasoning,
        )
    except (RetryError, Exception) as e:
        # Extract underlying exception from RetryError if present
        error_msg = str(e)
        if isinstance(e, RetryError) and hasattr(e, 'last_attempt'):
            try:
                error_msg = str(e.last_attempt.exception())
            except:
                pass
        adv_response = ModelResponse(
            model_name=model_name,
            prompt_type="adversarial",
            response_text="",
            latency_ms=0.0,
            error=error_msg,
        )
    
    # Benign
    benign_response = None
    if include_benign and variant.benign_prompt:
        try:
            ben_text, ben_latency, ben_tokens, ben_reasoning = await call_model_once(
                prompt=variant.benign_prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                reasoning_effort=reasoning_effort,
            )
            benign_response = ModelResponse(
                model_name=model_name,
                prompt_type="benign",
                response_text=ben_text,
                latency_ms=ben_latency,
                tokens_used=ben_tokens,
                reasoning_content=ben_reasoning,
            )
        except (RetryError, Exception) as e:
            error_msg = str(e)
            if isinstance(e, RetryError) and hasattr(e, 'last_attempt'):
                try:
                    error_msg = str(e.last_attempt.exception())
                except:
                    pass
            benign_response = ModelResponse(
                model_name=model_name,
                prompt_type="benign",
                response_text="",
                latency_ms=0.0,
                error=error_msg,
            )
    
    return adv_response, benign_response


