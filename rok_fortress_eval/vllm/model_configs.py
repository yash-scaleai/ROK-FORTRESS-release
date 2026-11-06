"""
Model-specific configurations for vLLM-hosted models.

This registry handles model-specific parameters like stop tokens, chat templates,
and context lengths without affecting the existing remote API code path.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for a specific model hosted via vLLM."""
    
    name: str
    # vLLM server launch options
    chat_template: Optional[str] = None  # Path to chat template file
    max_model_len: Optional[int] = None  # Maximum context length
    gpu_memory_utilization: float = 0.9  # Fraction of GPU memory to use
    dtype: str = "auto"  # Data type: auto, float16, bfloat16
    
    # Request-time options
    stop_tokens: Optional[List[str]] = None  # Model-specific stop sequences
    supports_system_prompt: bool = True  # Whether model handles system prompts well
    default_temperature: float = 1.0
    
    # Extra parameters passed to vLLM
    extra_launch_args: Dict[str, Any] = field(default_factory=dict)
    extra_request_params: Dict[str, Any] = field(default_factory=dict)


# Registry of known models and their configurations
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # HyperCLOVAX SEED Think 14B - Korean reasoning model from Naver
    # Ref: https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B
    # Note: Requires --enforce-eager due to triton/CUDA linking issues
    # VRAM: ~27.5 GiB on H100
    # IMPORTANT: Use --reasoning-parser hcx when serving!
    "hyperclovax": ModelConfig(
        name="hyperclovax",
        stop_tokens=["<|im_end|><|stop|>", "<|im_end|><|endofturn|>"],  # Correct stop tokens
        max_model_len=131072,  # 128k context
        supports_system_prompt=True,
        extra_launch_args={
            "enforce_eager": True,  # Required for stability
            "trust_remote_code": True,  # Contains custom code
            "reasoning_parser": "hcx",  # Requires hcx-vllm-plugin
        },
        extra_request_params={
            "chat_template_kwargs": {"force_reasoning": True},  # Force reasoning mode
            "skip_special_tokens": False,
        },
    ),
    "llama-3": ModelConfig(
        name="llama-3",
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
        max_model_len=8192,
        supports_system_prompt=True,
    ),
    "llama-2": ModelConfig(
        name="llama-2",
        stop_tokens=["</s>"],
        max_model_len=4096,
        supports_system_prompt=True,
    ),
    "mistral": ModelConfig(
        name="mistral",
        stop_tokens=["</s>", "[/INST]"],
        max_model_len=8192,
        supports_system_prompt=True,
    ),
    "qwen": ModelConfig(
        name="qwen",
        stop_tokens=["<|endoftext|>", "<|im_end|>"],
        max_model_len=8192,
        supports_system_prompt=True,
    ),
    # SKT A.X-4.0 - Korean LLM based on Qwen2.5-72B
    # Ref: https://huggingface.co/skt/A.X-4.0
    # VRAM: ~140 GiB (needs 2-4x H100 with tensor parallelism)
    # No reasoning mode - uses standard Qwen2.5 chat format
    "a.x-4.0": ModelConfig(
        name="a.x-4.0",
        stop_tokens=["<|endoftext|>", "<|im_end|>"],  # Qwen2.5 tokens
        max_model_len=131072,  # 128k context
        supports_system_prompt=True,
        # No enforce_eager by default - add only if CUDA compilation issues occur
    ),
    # EXAONE-4.0-32B - LG AI Research multilingual reasoning model
    # Ref: https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B
    # VRAM: ~60 GiB (needs 2x H100 with tensor parallelism)
    # Supports reasoning mode
    "exaone-4.0": ModelConfig(
        name="exaone-4.0",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
        max_model_len=32768,
        supports_system_prompt=True,
        extra_request_params={
            "chat_template_kwargs": {"force_reasoning": True},
        },
    ),
    "gemma": ModelConfig(
        name="gemma",
        stop_tokens=["<end_of_turn>", "<eos>"],
        max_model_len=8192,
        supports_system_prompt=True,
    ),
    
    # ========================================================================
    # OmniServe-deployed models (HyperCLOVAX official deployment)
    # ========================================================================
    
    # HyperCLOVAX-SEED-Think-32B via OmniServe
    # Ref: https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B
    # Uses OmniServe Docker deployment - NOT vLLM directly
    # API endpoint: http://localhost:8000/a/v1
    # Model name in requests: track_a_model
    "omniserve-hyperclovax": ModelConfig(
        name="omniserve-hyperclovax",
        stop_tokens=["<|im_end|><|stop|>", "<|im_end|><|endofturn|>"],
        max_model_len=131072,  # 128k context
        supports_system_prompt=True,
        extra_request_params={
            "thinking_token_budget": 500,  # OmniServe reasoning budget
            "chat_template_kwargs": {"thinking": True},  # Enable reasoning mode
        },
    ),
    
    # ========================================================================
    # Korean Models for Judge Calibration
    # ========================================================================
    
    # K-EXAONE - LG AI Research 236B MoE model
    # Ref: https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B
    # VRAM: ~140 GiB (needs 4x H100 with TP=4)
    # Supports reasoning mode
    "k-exaone": ModelConfig(
        name="k-exaone",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
        max_model_len=32768,
        supports_system_prompt=True,
        extra_request_params={
            "chat_template_kwargs": {"force_reasoning": True},
        },
    ),
    
    # Solar-Open - Upstage 100B MoE model
    # Ref: https://huggingface.co/upstage/Solar-Open-100B
    # VRAM: ~60 GiB (needs 2x H100 with TP=2)
    # No reasoning mode
    "solar-open": ModelConfig(
        name="solar-open",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
        max_model_len=32768,
        supports_system_prompt=True,
    ),
    
    # Kanana-2 - Kakao 30B MoE thinking model
    # Ref: https://huggingface.co/kakaocorp/kanana-2-30b-a3b-thinking
    # VRAM: ~32 GiB (needs 2x H100 with TP=2)
    # DeepSeek-V3 architecture with thinking mode - uses deepseek_r1 parser
    "kanana-2": ModelConfig(
        name="kanana-2",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
        max_model_len=131072,  # 128k context
        supports_system_prompt=True,
        extra_launch_args={
            "reasoning_parser": "deepseek_r1",  # DeepSeek-V3 based architecture
        },
        extra_request_params={
            "chat_template_kwargs": {"enable_thinking": True},
        },
    ),
    
    # A.X-K1 - SK Telecom 519B MoE model with hybrid reasoning
    # Ref: https://huggingface.co/skt/A.X-K1
    # VRAM: ~280 GiB (needs 8x H100 with TP=8)
    # DeepSeek-V3 based, supports Think/Non-Think mode via chat template
    "a.x-k1": ModelConfig(
        name="a.x-k1",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
        max_model_len=131072,  # 128k context
        supports_system_prompt=True,
        extra_launch_args={
            "reasoning_parser": "deepseek_r1",  # DeepSeek-V3 based architecture
        },
        extra_request_params={
            "chat_template_kwargs": {"enable_thinking": True},  # Enable Think mode
        },
    ),
    
    # Fallback for unknown models - uses vLLM defaults
    "default": ModelConfig(
        name="default",
        stop_tokens=None,  # Let vLLM handle it
        max_model_len=None,
        supports_system_prompt=True,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    Get configuration for a model by name.
    
    Matches against known model patterns in the registry.
    Falls back to default config for unknown models.
    
    Args:
        model_name: Full model name (e.g., 'vllm/hyperclovax-large')
        
    Returns:
        ModelConfig for the matched model or default config
    """
    # Strip vllm/ prefix if present
    clean_name = model_name.lower()
    if clean_name.startswith("vllm/"):
        clean_name = clean_name[5:]
    
    # Check registry for matching pattern
    for key in MODEL_REGISTRY:
        if key != "default" and key in clean_name:
            return MODEL_REGISTRY[key]
    
    return MODEL_REGISTRY["default"]


def register_model_config(key: str, config: ModelConfig) -> None:
    """
    Register a new model configuration.
    
    Use this to add custom configurations at runtime.
    
    Args:
        key: Lookup key (matched against model names)
        config: ModelConfig instance
    """
    MODEL_REGISTRY[key] = config

