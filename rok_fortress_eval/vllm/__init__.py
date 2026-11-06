"""
vLLM integration subpackage for ROK-FORTRESS.

This module provides local GPU-based model inference via vLLM with:
- Multi-GPU load balancing
- Model-specific configurations (stop tokens, chat templates)
- OpenAI-compatible API interface

Usage:
    Models prefixed with 'vllm/' are routed to local vLLM servers.
    Example: --judge-models "vllm/hyperclovax,openai/gpt-5.1"
"""

from .vllm_client import VLLMClient, call_vllm_model_once, evaluate_variant_with_vllm, configure_vllm
from .load_balancer import VLLMLoadBalancer
from .model_configs import ModelConfig, get_model_config, MODEL_REGISTRY, register_model_config
from .router import (
    get_client_for_model,
    is_vllm_model,
    call_model_routed,
    evaluate_variant_routed,
    parse_vllm_endpoints,
)
from .vllm_manager import VLLMServerManager, start_vllm_servers, check_endpoints_health

__all__ = [
    # Client
    "VLLMClient",
    "call_vllm_model_once",
    "evaluate_variant_with_vllm",
    "configure_vllm",
    # Load balancer
    "VLLMLoadBalancer",
    # Model configs
    "ModelConfig",
    "get_model_config",
    "MODEL_REGISTRY",
    "register_model_config",
    # Router
    "get_client_for_model",
    "is_vllm_model",
    "call_model_routed",
    "evaluate_variant_routed",
    "parse_vllm_endpoints",
    # Server manager
    "VLLMServerManager",
    "start_vllm_servers",
    "check_endpoints_health",
]

