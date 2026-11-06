"""
Configuration for ROK-FORTRESS evaluation.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paper-panel default: Korean-aware reasoning judge + frontier reasoning judges.
# HyperCLOVA X requires a local vLLM server (see scripts/launch_vllm.sh).
# Override with --judge-models to use an API-only panel.
DEFAULT_JUDGE_MODELS = [
    "vllm/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
    "openai/o4-mini",
    "fireworks_ai/kimi-k2-thinking",
]

# Parsing model for extracting structured verdicts from judge outputs
DEFAULT_PARSING_MODEL = "gemini/gemini-2.0-flash"

# Refusal detection model
DEFAULT_REFUSAL_MODEL = "openai/gpt-4o-mini-2024-07-18"

# Model parameters
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 3072  # Match FORTRESS
DEFAULT_CONCURRENCY = 30
DEFAULT_TIMEOUT = 120  # Increased from 60s
DEFAULT_REASONING_EFFORT = None  # None, "low", "medium", "high"

# OpenAI models require temp=1.0
OPENAI_REQUIRED_TEMP = 1.0

def get_model_temperature(model_name: str, default_temp: float = DEFAULT_TEMPERATURE) -> float:
    """Get appropriate temperature for a model. OpenAI models require temp=1.0."""
    if model_name.startswith("openai/"):
        return OPENAI_REQUIRED_TEMP
    return default_temp

# API configuration.
#
# Two routing modes (see rok_fortress_eval/core/api_client.py):
#   1. Proxy mode -- BASE_URL points at a LiteLLM (or other OpenAI-compatible)
#      proxy that fans out to multiple providers. We accept either ``BASE_URL``
#      or ``OPENAI_BASE_URL`` (the OpenAI SDK's canonical name).
#   2. Direct mode -- BASE_URL unset; ``litellm.acompletion`` uses per-provider
#      env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, ...).
BASE_URL = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")


def require_api_key() -> str:
    """Return the configured API key, raising if none is set.

    Validation is deferred to call time so the package can be imported (and the
    data loader used) without any API key configured.
    """
    if not API_KEY:
        raise ValueError("API_KEY or OPENAI_API_KEY must be set in environment")
    return API_KEY
