"""
Router for dispatching model requests to appropriate backends.

Routes requests based on model name prefix:
- `vllm/` prefix → Local vLLM servers (via vllm_client)
- Other prefixes (openai/, anthropic/, gemini/) → Remote API (existing api_client)

This enables hybrid setups where some models run locally and others use remote APIs.
"""

from typing import Optional, Tuple, List, Callable, Any
from ..models import ModelResponse, PromptVariant


def is_vllm_model(model_name: str) -> bool:
    """
    Check if a model should be routed to vLLM.
    
    Args:
        model_name: Model identifier
        
    Returns:
        True if model should use vLLM backend
    """
    return model_name.lower().startswith("vllm/")


def get_client_for_model(
    model_name: str,
    vllm_endpoints: Optional[List[str]] = None,
):
    """
    Get the appropriate client/caller for a model.
    
    This is a factory function that returns the right call_model_once function
    based on the model name prefix.
    
    Args:
        model_name: Model identifier (e.g., "vllm/hyperclovax", "openai/gpt-5.1")
        vllm_endpoints: List of vLLM endpoints (required for vllm/ models)
        
    Returns:
        Tuple of (call_function, evaluate_variant_function)
        
    Raises:
        ValueError: If vllm/ model requested but no endpoints configured
    """
    if is_vllm_model(model_name):
        if not vllm_endpoints:
            raise ValueError(
                f"Model {model_name} requires vLLM but no endpoints configured. "
                "Pass --vllm-endpoints to the calling script."
            )
        
        # Import here to avoid circular imports and keep core unchanged
        from .vllm_client import (
            call_vllm_model_once,
            evaluate_variant_with_vllm,
            configure_vllm,
        )
        
        # Configure the global load balancer
        configure_vllm(vllm_endpoints)
        
        return call_vllm_model_once, evaluate_variant_with_vllm
    else:
        # Use existing api_client (unchanged)
        from ..core.api_client import (
            call_model_once,
            evaluate_variant_with_model,
        )
        
        return call_model_once, evaluate_variant_with_model


async def call_model_routed(
    prompt: str,
    model_name: str,
    vllm_endpoints: Optional[List[str]] = None,
    temperature: float = 1.0,
    max_tokens: int = 3072,
    timeout: float = 120.0,
    system_prompt: str = "You are a helpful assistant.",
    reasoning_effort: Optional[str] = None,
) -> Tuple[str, float, Optional[int], Optional[str]]:
    """
    Call a model, routing to appropriate backend based on model name.
    
    This is the main entry point for hybrid routing.
    
    Args:
        prompt: User prompt
        model_name: Model identifier
        vllm_endpoints: vLLM endpoints (for vllm/ models)
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        timeout: Request timeout
        system_prompt: System message
        reasoning_effort: Reasoning mode (ignored for vLLM)
        
    Returns:
        Tuple of (response_text, latency_ms, tokens_used, reasoning_content)
    """
    call_fn, _ = get_client_for_model(model_name, vllm_endpoints)
    
    return await call_fn(
        prompt=prompt,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        system_prompt=system_prompt,
        reasoning_effort=reasoning_effort,
    )


async def evaluate_variant_routed(
    variant: PromptVariant,
    model_name: str,
    vllm_endpoints: Optional[List[str]] = None,
    include_benign: bool = True,
    temperature: float = 1.0,
    max_tokens: int = 3072,
    timeout: float = 120.0,
    reasoning_effort: Optional[str] = None,
) -> Tuple[ModelResponse, Optional[ModelResponse]]:
    """
    Evaluate a variant with a model, routing to appropriate backend.
    
    Args:
        variant: Prompt variant to evaluate
        model_name: Model identifier
        vllm_endpoints: vLLM endpoints (for vllm/ models)
        include_benign: Whether to evaluate benign prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        timeout: Request timeout
        reasoning_effort: Reasoning mode (ignored for vLLM)
        
    Returns:
        Tuple of (adversarial_response, benign_response)
    """
    _, evaluate_fn = get_client_for_model(model_name, vllm_endpoints)
    
    return await evaluate_fn(
        variant=variant,
        model_name=model_name,
        include_benign=include_benign,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )


def parse_vllm_endpoints(endpoints_str: Optional[str]) -> Optional[List[str]]:
    """
    Parse comma-separated vLLM endpoints string.
    
    Args:
        endpoints_str: Comma-separated endpoints (e.g., "http://localhost:8000/v1,http://localhost:8001/v1")
        
    Returns:
        List of endpoint URLs or None if not provided
    """
    if not endpoints_str:
        return None
    
    endpoints = [e.strip() for e in endpoints_str.split(",") if e.strip()]
    
    # Ensure endpoints end with /v1
    normalized = []
    for ep in endpoints:
        if not ep.endswith("/v1"):
            if ep.endswith("/"):
                ep = ep + "v1"
            else:
                ep = ep + "/v1"
        normalized.append(ep)
    
    return normalized if normalized else None

