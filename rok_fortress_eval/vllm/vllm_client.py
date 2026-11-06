"""
vLLM API client that mirrors the interface of core/api_client.py.

This module provides the same function signatures as api_client.py but routes
requests to local vLLM servers via the load balancer, with model-specific
configurations applied.

The existing api_client.py remains UNCHANGED - this is a parallel implementation.
"""

import asyncio
import time
from typing import Optional, Tuple, List

from openai import AsyncOpenAI
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    RetryError,
)

from .load_balancer import VLLMLoadBalancer
from .model_configs import get_model_config, ModelConfig
from ..models import ModelResponse, PromptVariant


class VLLMAPIError(Exception):
    """Custom exception for vLLM API errors."""
    pass


# Global load balancer instance (set via configure_vllm)
_load_balancer: Optional[VLLMLoadBalancer] = None


def configure_vllm(endpoints: List[str]) -> VLLMLoadBalancer:
    """
    Configure the vLLM load balancer with endpoints.
    
    Must be called before using vLLM client functions.
    
    Args:
        endpoints: List of vLLM server URLs (e.g., ["http://localhost:8000/v1"])
        
    Returns:
        Configured VLLMLoadBalancer instance
    """
    global _load_balancer
    _load_balancer = VLLMLoadBalancer(endpoints)
    return _load_balancer


def get_load_balancer() -> VLLMLoadBalancer:
    """Get the configured load balancer, raising if not configured."""
    if _load_balancer is None:
        raise RuntimeError(
            "vLLM not configured. Call configure_vllm(endpoints) first, "
            "or pass --vllm-endpoints to the calling script."
        )
    return _load_balancer


class VLLMClient:
    """
    Client for making requests to vLLM servers with model-specific configs.
    
    Usage:
        client = VLLMClient(
            endpoints=["http://localhost:8000/v1", "http://localhost:8001/v1"],
            model_name="vllm/hyperclovax"
        )
        text, latency, tokens, reasoning = await client.call_model_once(prompt)
    """
    
    def __init__(self, endpoints: List[str], model_name: str):
        """
        Initialize vLLM client.
        
        Args:
            endpoints: List of vLLM server URLs
            model_name: Model identifier (e.g., "vllm/hyperclovax")
        """
        self._balancer = VLLMLoadBalancer(endpoints)
        self._model_name = model_name
        self._model_config = get_model_config(model_name)
    
    @property
    def model_config(self) -> ModelConfig:
        """Get the model configuration."""
        return self._model_config
    
    async def call_model_once(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 3072,
        timeout: float = 120.0,
        system_prompt: str = "You are a helpful assistant.",
    ) -> Tuple[str, float, Optional[int], Optional[str]]:
        """
        Call vLLM model once.
        
        Same signature as api_client.call_model_once for compatibility.
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            system_prompt: System message
            
        Returns:
            Tuple of (response_text, latency_ms, tokens_used, reasoning_content)
        """
        client = self._balancer.get_client()
        
        # Build messages based on model's system prompt support
        if self._model_config.supports_system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            # Prepend system prompt to user message for models without system support
            combined_prompt = f"{system_prompt}\n\n{prompt}"
            messages = [{"role": "user", "content": combined_prompt}]
        
        # Build request kwargs
        request_kwargs = {
            "model": self._model_name.replace("vllm/", ""),  # Strip prefix for vLLM
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add model-specific stop tokens
        if self._model_config.stop_tokens:
            request_kwargs["stop"] = self._model_config.stop_tokens
        
        # Handle extra_request_params - some go in extra_body for vLLM-specific params
        if self._model_config.extra_request_params:
            extra_body = {}
            for key, val in self._model_config.extra_request_params.items():
                if key in ("chat_template_kwargs", "skip_special_tokens", "guided_json", "guided_regex"):
                    extra_body[key] = val
                else:
                    request_kwargs[key] = val
            if extra_body:
                request_kwargs["extra_body"] = extra_body
        
        start = time.time()
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(**request_kwargs),
                timeout=timeout
            )
            latency_ms = (time.time() - start) * 1000
            
            content = resp.choices[0].message.content or ""
            tokens = None
            if hasattr(resp, 'usage') and resp.usage:
                tokens = resp.usage.total_tokens
            
            # vLLM doesn't typically return reasoning_content, but check anyway
            reasoning_content = None
            msg = resp.choices[0].message
            if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                reasoning_content = msg.reasoning_content
            
            return content, latency_ms, tokens, reasoning_content
            
        except asyncio.TimeoutError:
            raise VLLMAPIError(f"vLLM timeout after {timeout}s")
        except Exception as e:
            raise VLLMAPIError(f"vLLM API error: {e}")


# Module-level functions that use the global load balancer
# These mirror api_client.py function signatures exactly

@retry(
    retry=retry_if_exception_type((VLLMAPIError, asyncio.TimeoutError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
)
async def call_vllm_model_once(
    prompt: str,
    model_name: str,
    temperature: float = 1.0,
    max_tokens: int = 3072,
    timeout: float = 120.0,
    system_prompt: str = "You are a helpful assistant.",
    reasoning_effort: Optional[str] = None,  # Ignored for vLLM, kept for interface compat
) -> Tuple[str, float, Optional[int], Optional[str]]:
    """
    Call vLLM model once with retry logic.
    
    Mirrors api_client.call_model_once signature exactly.
    Uses global load balancer configured via configure_vllm().
    
    Args:
        prompt: User prompt
        model_name: Model identifier (e.g., "vllm/hyperclovax")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        system_prompt: System message
        reasoning_effort: Ignored (kept for interface compatibility)
        
    Returns:
        Tuple of (response_text, latency_ms, tokens_used, reasoning_content)
    """
    balancer = get_load_balancer()
    client = balancer.get_client()
    model_config = get_model_config(model_name)
    
    # Build messages
    if model_config.supports_system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        combined_prompt = f"{system_prompt}\n\n{prompt}"
        messages = [{"role": "user", "content": combined_prompt}]
    
    # Build request
    request_kwargs = {
        "model": model_name.replace("vllm/", ""),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    if model_config.stop_tokens:
        request_kwargs["stop"] = model_config.stop_tokens
    
    # Handle extra_request_params - some go in extra_body for vLLM-specific params
    if model_config.extra_request_params:
        extra_body = {}
        for key, val in model_config.extra_request_params.items():
            if key in ("chat_template_kwargs", "skip_special_tokens", "guided_json", "guided_regex"):
                # vLLM-specific params go in extra_body
                extra_body[key] = val
            else:
                request_kwargs[key] = val
        if extra_body:
            request_kwargs["extra_body"] = extra_body
    
    start = time.time()
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(**request_kwargs),
            timeout=timeout
        )
        latency_ms = (time.time() - start) * 1000
        
        content = resp.choices[0].message.content or ""
        tokens = None
        if hasattr(resp, 'usage') and resp.usage:
            tokens = resp.usage.total_tokens
        
        reasoning_content = None
        msg = resp.choices[0].message
        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
            reasoning_content = msg.reasoning_content
        
        return content, latency_ms, tokens, reasoning_content
        
    except asyncio.TimeoutError:
        raise VLLMAPIError(f"vLLM timeout after {timeout}s")
    except Exception as e:
        raise VLLMAPIError(f"vLLM API error: {e}")


async def evaluate_variant_with_vllm(
    variant: PromptVariant,
    model_name: str,
    include_benign: bool = True,
    temperature: float = 1.0,
    max_tokens: int = 3072,
    timeout: float = 120.0,
    reasoning_effort: Optional[str] = None,
) -> Tuple[ModelResponse, Optional[ModelResponse]]:
    """
    Evaluate a single variant with a vLLM model (adversarial + optional benign).
    
    Mirrors api_client.evaluate_variant_with_model signature exactly.
    
    Args:
        variant: Prompt variant to evaluate
        model_name: vLLM model identifier
        include_benign: Whether to also evaluate benign prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        timeout: Request timeout
        reasoning_effort: Ignored (kept for interface compatibility)
        
    Returns:
        Tuple of (adversarial_response, benign_response)
    """
    # Adversarial
    try:
        adv_text, adv_latency, adv_tokens, adv_reasoning = await call_vllm_model_once(
            prompt=variant.adversarial_prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
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
            ben_text, ben_latency, ben_tokens, ben_reasoning = await call_vllm_model_once(
                prompt=variant.benign_prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
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

