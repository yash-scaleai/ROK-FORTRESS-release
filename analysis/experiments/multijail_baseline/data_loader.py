"""
Data loader for MultiJail dataset.

Loads the 315 harmful prompts across 10 languages and creates both
unintentional (raw) and intentional (AIM-wrapped) variants.

Dataset source: https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Iterator
import csv

from .config import (
    LANGUAGES,
    LANGUAGE_TIERS,
    LANGUAGE_NAMES,
    AIM_PREFIX,
    DATA_DIR,
    NUM_PROMPTS,
    MULTIJAIL_CSV_PATH,
)


@dataclass
class MultiJailPrompt:
    """Single prompt from the MultiJail dataset."""
    
    prompt_id: str           # Unique identifier (e.g., "001")
    language: str            # Language code (e.g., "en", "sw")
    language_tier: str       # Resource tier: HRL, MRL, LRL
    prompt_text: str         # The actual prompt text
    scenario: str            # "unintentional" or "intentional"
    safety_tag: Optional[str] = None  # Original safety category if available
    
    @property
    def language_name(self) -> str:
        """Human-readable language name."""
        return LANGUAGE_NAMES.get(self.language, self.language)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt_id": self.prompt_id,
            "language": self.language,
            "language_tier": self.language_tier,
            "language_name": self.language_name,
            "prompt_text": self.prompt_text,
            "scenario": self.scenario,
            "safety_tag": self.safety_tag,
        }


def apply_aim_template(prompt: str) -> str:
    """
    Apply the AIM (Always Intelligent and Machiavellian) jailbreak template.
    
    This creates the "intentional" scenario where malicious users combine
    the harmful prompt with an English jailbreak prefix.
    
    Args:
        prompt: The raw harmful prompt
        
    Returns:
        The prompt wrapped in the AIM template
    """
    return f"{AIM_PREFIX} '{prompt}'"


def load_multijail_dataset(
    data_dir: Optional[Path] = None,
    languages: Optional[List[str]] = None,
    include_intentional: bool = True,
    csv_path: Optional[Path] = None,
) -> List[MultiJailPrompt]:
    """
    Load MultiJail dataset and expand into prompt variants.
    
    The dataset contains harmful prompts translated into 10 languages.
    This function creates two variants per prompt per language:
    - Unintentional: Raw prompt (simulating non-English users)
    - Intentional: AIM-wrapped prompt (simulating adversarial attack)
    
    Args:
        data_dir: Path to MultiJail data directory. If None, uses default.
        languages: List of language codes to load. If None, loads all.
        include_intentional: Whether to include AIM-wrapped intentional variants.
        csv_path: Direct path to MultiJail.csv. If None, checks default location.
        
    Returns:
        List of MultiJailPrompt objects
    """
    if languages is None:
        languages = LANGUAGES
    
    prompts: List[MultiJailPrompt] = []
    
    # Priority 1: Check for MultiJail.csv in the module directory (bundled with code)
    if csv_path is None:
        csv_path = MULTIJAIL_CSV_PATH
    
    if csv_path.exists():
        prompts.extend(_load_from_wide_csv(csv_path, languages, include_intentional))
        return prompts
    
    # Priority 2: Check data_dir for various formats
    if data_dir is None:
        data_dir = DATA_DIR
    
    # Format 1: Single JSON file with all languages
    json_path = data_dir / "multijail.json"
    if json_path.exists():
        prompts.extend(_load_from_json(json_path, languages, include_intentional))
        return prompts
    
    # Format 2: Separate files per language
    for lang in languages:
        lang_file = data_dir / f"{lang}.json"
        if lang_file.exists():
            prompts.extend(_load_language_file(lang_file, lang, include_intentional))
            continue
        
        # Try CSV format
        csv_file = data_dir / f"{lang}.csv"
        if csv_file.exists():
            prompts.extend(_load_language_csv(csv_file, lang, include_intentional))
            continue
    
    # Format 3: JSONL format
    jsonl_path = data_dir / "multijail.jsonl"
    if jsonl_path.exists():
        prompts.extend(_load_from_jsonl(jsonl_path, languages, include_intentional))
        return prompts
    
    if not prompts:
        raise FileNotFoundError(
            f"No MultiJail data found. Checked:\n"
            f"  - {csv_path}\n"
            f"  - {data_dir}\n"
            f"Please ensure MultiJail.csv is in the multijail_baseline directory."
        )
    
    return prompts


def _load_from_wide_csv(
    csv_path: Path,
    languages: List[str],
    include_intentional: bool,
) -> List[MultiJailPrompt]:
    """
    Load from wide-format CSV where each row contains prompts for all languages.
    
    Expected CSV format:
        id,source,tags,en,zh,it,vi,ar,ko,th,bn,sw,jv
        0,anthropics,"['tag1', 'tag2']",English prompt,Chinese prompt,...
    
    Args:
        csv_path: Path to the CSV file
        languages: List of language codes to load
        include_intentional: Whether to include AIM-wrapped variants
        
    Returns:
        List of MultiJailPrompt objects
    """
    prompts = []
    
    # Use utf-8-sig to strip BOM (Byte Order Mark) from Excel exports
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            prompt_id = str(row.get("id", ""))
            safety_tag = row.get("tags", row.get("source", ""))
            
            # Iterate through each language column
            for lang in languages:
                if lang not in row:
                    continue
                
                prompt_text = row[lang]
                if not prompt_text or not prompt_text.strip():
                    continue
                
                # Unintentional variant (raw prompt)
                prompts.append(MultiJailPrompt(
                    prompt_id=prompt_id,
                    language=lang,
                    language_tier=LANGUAGE_TIERS.get(lang, "unknown"),
                    prompt_text=prompt_text.strip(),
                    scenario="unintentional",
                    safety_tag=safety_tag,
                ))
                
                # Intentional variant (AIM-wrapped)
                if include_intentional:
                    prompts.append(MultiJailPrompt(
                        prompt_id=prompt_id,
                        language=lang,
                        language_tier=LANGUAGE_TIERS.get(lang, "unknown"),
                        prompt_text=apply_aim_template(prompt_text.strip()),
                        scenario="intentional",
                        safety_tag=safety_tag,
                    ))
    
    return prompts


def _load_from_json(
    json_path: Path,
    languages: List[str],
    include_intentional: bool,
) -> List[MultiJailPrompt]:
    """Load from single JSON file containing all languages."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    prompts = []
    
    # Expected format: {"en": [{"id": "001", "prompt": "...", "tag": "..."}], ...}
    for lang in languages:
        if lang not in data:
            continue
        
        for item in data[lang]:
            prompt_id = str(item.get("id", item.get("prompt_id", "")))
            prompt_text = item.get("prompt", item.get("text", item.get("prompt_text", "")))
            safety_tag = item.get("tag", item.get("safety_tag", item.get("category")))
            
            if not prompt_text:
                continue
            
            # Unintentional variant (raw prompt)
            prompts.append(MultiJailPrompt(
                prompt_id=prompt_id,
                language=lang,
                language_tier=LANGUAGE_TIERS.get(lang, "unknown"),
                prompt_text=prompt_text,
                scenario="unintentional",
                safety_tag=safety_tag,
            ))
            
            # Intentional variant (AIM-wrapped)
            if include_intentional:
                prompts.append(MultiJailPrompt(
                    prompt_id=prompt_id,
                    language=lang,
                    language_tier=LANGUAGE_TIERS.get(lang, "unknown"),
                    prompt_text=apply_aim_template(prompt_text),
                    scenario="intentional",
                    safety_tag=safety_tag,
                ))
    
    return prompts


def _load_from_jsonl(
    jsonl_path: Path,
    languages: List[str],
    include_intentional: bool,
) -> List[MultiJailPrompt]:
    """Load from JSONL file."""
    prompts = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            lang = item.get("language", item.get("lang", ""))
            
            if lang not in languages:
                continue
            
            prompt_id = str(item.get("id", item.get("prompt_id", "")))
            prompt_text = item.get("prompt", item.get("text", item.get("prompt_text", "")))
            safety_tag = item.get("tag", item.get("safety_tag", item.get("category")))
            
            if not prompt_text:
                continue
            
            # Unintentional variant
            prompts.append(MultiJailPrompt(
                prompt_id=prompt_id,
                language=lang,
                language_tier=LANGUAGE_TIERS.get(lang, "unknown"),
                prompt_text=prompt_text,
                scenario="unintentional",
                safety_tag=safety_tag,
            ))
            
            # Intentional variant
            if include_intentional:
                prompts.append(MultiJailPrompt(
                    prompt_id=prompt_id,
                    language=lang,
                    language_tier=LANGUAGE_TIERS.get(lang, "unknown"),
                    prompt_text=apply_aim_template(prompt_text),
                    scenario="intentional",
                    safety_tag=safety_tag,
                ))
    
    return prompts


def _load_language_file(
    file_path: Path,
    language: str,
    include_intentional: bool,
) -> List[MultiJailPrompt]:
    """Load prompts from a single language JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    prompts = []
    items = data if isinstance(data, list) else data.get("prompts", [])
    
    for item in items:
        prompt_id = str(item.get("id", item.get("prompt_id", "")))
        prompt_text = item.get("prompt", item.get("text", item.get("prompt_text", "")))
        safety_tag = item.get("tag", item.get("safety_tag", item.get("category")))
        
        if not prompt_text:
            continue
        
        # Unintentional variant
        prompts.append(MultiJailPrompt(
            prompt_id=prompt_id,
            language=language,
            language_tier=LANGUAGE_TIERS.get(language, "unknown"),
            prompt_text=prompt_text,
            scenario="unintentional",
            safety_tag=safety_tag,
        ))
        
        # Intentional variant
        if include_intentional:
            prompts.append(MultiJailPrompt(
                prompt_id=prompt_id,
                language=language,
                language_tier=LANGUAGE_TIERS.get(language, "unknown"),
                prompt_text=apply_aim_template(prompt_text),
                scenario="intentional",
                safety_tag=safety_tag,
            ))
    
    return prompts


def _load_language_csv(
    file_path: Path,
    language: str,
    include_intentional: bool,
) -> List[MultiJailPrompt]:
    """Load prompts from a CSV file."""
    prompts = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            prompt_id = row.get("id", row.get("prompt_id", str(i + 1)))
            prompt_text = row.get("prompt", row.get("text", row.get("prompt_text", "")))
            safety_tag = row.get("tag", row.get("safety_tag", row.get("category")))
            
            if not prompt_text:
                continue
            
            # Unintentional variant
            prompts.append(MultiJailPrompt(
                prompt_id=prompt_id,
                language=language,
                language_tier=LANGUAGE_TIERS.get(language, "unknown"),
                prompt_text=prompt_text,
                scenario="unintentional",
                safety_tag=safety_tag,
            ))
            
            # Intentional variant
            if include_intentional:
                prompts.append(MultiJailPrompt(
                    prompt_id=prompt_id,
                    language=language,
                    language_tier=LANGUAGE_TIERS.get(language, "unknown"),
                    prompt_text=apply_aim_template(prompt_text),
                    scenario="intentional",
                    safety_tag=safety_tag,
                ))
    
    return prompts


def iter_prompts_by_language(
    prompts: List[MultiJailPrompt],
) -> Iterator[tuple[str, List[MultiJailPrompt]]]:
    """Group prompts by language for iteration."""
    by_lang: Dict[str, List[MultiJailPrompt]] = {}
    
    for prompt in prompts:
        if prompt.language not in by_lang:
            by_lang[prompt.language] = []
        by_lang[prompt.language].append(prompt)
    
    for lang in LANGUAGES:
        if lang in by_lang:
            yield lang, by_lang[lang]


def iter_prompts_by_scenario(
    prompts: List[MultiJailPrompt],
) -> Iterator[tuple[str, List[MultiJailPrompt]]]:
    """Group prompts by scenario for iteration."""
    by_scenario: Dict[str, List[MultiJailPrompt]] = {}
    
    for prompt in prompts:
        if prompt.scenario not in by_scenario:
            by_scenario[prompt.scenario] = []
        by_scenario[prompt.scenario].append(prompt)
    
    for scenario in ["unintentional", "intentional"]:
        if scenario in by_scenario:
            yield scenario, by_scenario[scenario]


def get_dataset_stats(prompts: List[MultiJailPrompt]) -> Dict:
    """Get summary statistics for a list of prompts."""
    stats = {
        "total_prompts": len(prompts),
        "by_language": {},
        "by_tier": {"HRL": 0, "MRL": 0, "LRL": 0},
        "by_scenario": {"unintentional": 0, "intentional": 0},
    }
    
    for prompt in prompts:
        # By language
        if prompt.language not in stats["by_language"]:
            stats["by_language"][prompt.language] = 0
        stats["by_language"][prompt.language] += 1
        
        # By tier
        if prompt.language_tier in stats["by_tier"]:
            stats["by_tier"][prompt.language_tier] += 1
        
        # By scenario
        if prompt.scenario in stats["by_scenario"]:
            stats["by_scenario"][prompt.scenario] += 1
    
    return stats
