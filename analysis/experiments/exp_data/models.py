"""
Model categorization for Global vs Regional analysis.

Global Models: Frontier models from US/international companies
Regional Models: Korean-developed models (Naver, Kakao, LG, etc.)
"""

from typing import Dict, List, Literal, Optional


ModelCategory = Literal["global", "regional", "unknown"]


# Global frontier models
GLOBAL_MODELS = [
    # OpenAI
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "o4-mini",
    # Anthropic
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    # Google
    "gemini-3-pro-preview",
    "gemini-3-pro",
    # Meta
    "llama4-maverick-instruct-basic",
    # Mistral
    "mistral.mistral-large-3-675b-instruct",
    # DeepSeek
    "deepseek-v3p2",
    # Qwen
    "qwen.qwen3-235b-a22b-2507-v1:0",
    # Kimi (Moonshot)
    "kimi-k2-thinking",
]

# Regional Korean models
REGIONAL_MODELS = [
    # Naver
    "hyperclovax",
    "hyperclova-x",
    # Kakao
    "kanana2",
    "kanana-2",
    # LG
    "kexaone",
    "exaone",
    # Upstage
    "solar_open",
    "solar-open",
]

# Closed-source global models (proprietary APIs)
CLOSED_SOURCE_MODELS = [
    "gpt-5.2",
    "o4-mini",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "gemini-3-pro-preview",
]

# Open-source models (weights available)
OPEN_SOURCE_MODELS = [
    # Global open-source
    "kimi-k2-thinking",
    "deepseek-v3p2",
    "llama4-maverick-instruct-basic",
    "qwen.qwen3-235b-a22b-2507-v1:0",
    "mistral.mistral-large-3-675b-instruct",
    # Korean regional models
    "hyperclovax",
    "kanana2",
    "solar_open",
    "kexaone",
]

# Normalized model name mappings (handle variations)
MODEL_ALIASES: Dict[str, str] = {
    # OpenAI variations
    "openai/gpt-5.2": "gpt-5.2",
    "openai/gpt-5.1": "gpt-5.1",
    "openai/o4-mini": "o4-mini",
    # Anthropic variations
    "anthropic/claude-opus-4-5": "claude-opus-4-5",
    "anthropic/claude-sonnet-4-5": "claude-sonnet-4-5",
    # Google variations
    "google/gemini-3-pro-preview": "gemini-3-pro-preview",
    "gemini/gemini-3-pro-preview": "gemini-3-pro-preview",
    # Naver variations
    "naver/hyperclova-x": "hyperclovax",
    "vllm/skt/A.X-4.0": "hyperclovax",
    "hyperclovax/track_a_model": "hyperclovax",
    # Kakao variations
    "kakao/kanana-2": "kanana2",
    "kanana2/kakaocorp/kanana-2-30b-a3b-thinking": "kanana2",
    # LG variations
    "lg/exaone": "kexaone",
    "vllm/LGAI-EXAONE/EXAONE-4.0-32B": "kexaone",
    "vllm/LGAI-EXAONE/K-EXAONE-236B-A23B": "kexaone",
    # Upstage variations
    "solar/upstage/Solar-Open-100B": "solar_open",
    # Bedrock variations
    "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0": "llama4-maverick-instruct-basic",
}


def normalize_model_name(model_name: str) -> str:
    """
    Normalize a model name to its canonical form.
    
    Handles variations like:
    - Provider prefixes (openai/, anthropic/, etc.)
    - Alias mappings
    
    Args:
        model_name: Raw model name
    
    Returns:
        Normalized model name
    """
    # Check alias mapping first
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]
    
    # Remove common provider prefixes
    for prefix in ["openai/", "anthropic/", "google/", "gemini/", "naver/", "kakao/", "lg/", "vllm/", "bedrock/", "fireworks_ai/"]:
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    
    return model_name


def is_global_model(model_name: str) -> bool:
    """Check if a model is a global frontier model."""
    normalized = normalize_model_name(model_name)
    return normalized in GLOBAL_MODELS or any(
        normalized.startswith(g) or g.startswith(normalized)
        for g in GLOBAL_MODELS
    )


def is_regional_model(model_name: str) -> bool:
    """Check if a model is a regional Korean model."""
    normalized = normalize_model_name(model_name)
    return normalized in REGIONAL_MODELS or any(
        normalized.startswith(r) or r.startswith(normalized)
        for r in REGIONAL_MODELS
    )


def is_closed_source_model(model_name: str) -> bool:
    """Check if a model is a closed-source (proprietary API) model."""
    normalized = normalize_model_name(model_name)
    return normalized in CLOSED_SOURCE_MODELS or any(
        normalized.startswith(c) or c.startswith(normalized)
        for c in CLOSED_SOURCE_MODELS
    )


def is_open_source_model(model_name: str) -> bool:
    """Check if a model is an open-source (weights available) model."""
    normalized = normalize_model_name(model_name)
    return normalized in OPEN_SOURCE_MODELS or any(
        normalized.startswith(o) or o.startswith(normalized)
        for o in OPEN_SOURCE_MODELS
    )


def get_model_category(model_name: str) -> ModelCategory:
    """
    Get the category of a model.
    
    Args:
        model_name: Model name (raw or normalized)
    
    Returns:
        "global", "regional", or "unknown"
    """
    if is_global_model(model_name):
        return "global"
    elif is_regional_model(model_name):
        return "regional"
    return "unknown"


def filter_global_models(model_names: List[str]) -> List[str]:
    """Filter a list of model names to only global models."""
    return [m for m in model_names if is_global_model(m)]


def filter_regional_models(model_names: List[str]) -> List[str]:
    """Filter a list of model names to only regional models."""
    return [m for m in model_names if is_regional_model(m)]


def categorize_models(model_names: List[str]) -> Dict[ModelCategory, List[str]]:
    """
    Categorize a list of models into global, regional, and unknown.
    
    Args:
        model_names: List of model names
    
    Returns:
        Dict with keys "global", "regional", "unknown" -> list of model names
    """
    result: Dict[ModelCategory, List[str]] = {
        "global": [],
        "regional": [],
        "unknown": [],
    }
    
    for model in model_names:
        category = get_model_category(model)
        result[category].append(model)
    
    return result


def get_model_display_name(model_name: str) -> str:
    """
    Get a display-friendly name for a model.
    
    Args:
        model_name: Raw or normalized model name
    
    Returns:
        Human-readable display name
    """
    display_names = {
        "gpt-5.2": "GPT-5.2",
        "gpt-5.1": "GPT-5.1",
        "o4-mini": "O4-mini",
        "claude-opus-4-5": "Claude Opus 4.5",
        "claude-sonnet-4-5": "Claude Sonnet 4.5",
        "gemini-3-pro-preview": "Gemini 3 Pro",
        "hyperclovax": "HyperCLOVA X",
        "kanana2": "Kanana 2",
        "kexaone": "K-EXAONE",
        "solar_open": "Solar Open",
        "deepseek-v3p2": "DeepSeek V3.2",
        "kimi-k2-thinking": "Kimi K2",
        "llama4-maverick-instruct-basic": "Llama 4 Maverick",
        "mistral.mistral-large-3-675b-instruct": "Mistral Large 3",
        "qwen.qwen3-235b-a22b-2507-v1:0": "Qwen3-235B-A22B",
    }
    
    normalized = normalize_model_name(model_name)
    return display_names.get(normalized, model_name)
