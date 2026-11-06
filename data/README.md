---
license: cc-by-4.0
language:
  - en
  - ko
pretty_name: ROK-FORTRESS
size_categories:
  - n<1K
task_categories:
  - text-generation
tags:
  - safety
  - red-teaming
  - adversarial
  - bilingual
  - korean
  - transcreation
  - benchmark
configs:
  - config_name: default
    data_files:
      - split: tasks
        path: rok_fortress_public.tsv
    sep: "\t"
extra_gated_prompt: |
  ROK-FORTRESS contains adversarial prompts designed to elicit unsafe LLM behavior under a controlled, defensive safety-evaluation framework.

  By requesting access, you agree to the following:

  1. **Defensive use only.** You will use this dataset solely for research on LLM safety, red-teaming, alignment, refusal calibration, or related defensive purposes.
  2. **No misuse-oriented training.** You will not use this dataset to train, fine-tune, or steer models toward producing harmful, illegal, or operationally dangerous content.
  3. **No redistribution of harmful artifacts.** You will not publicly redistribute unsafe model completions elicited from these prompts.
  4. **Attribution.** You will cite the ROK-FORTRESS paper in any publication or system that uses this dataset.
  5. **Compliance.** You will comply with all applicable laws and with the CC-BY-4.0 license under which this dataset is released.
extra_gated_button_content: Acknowledge and access
---

# ROK-FORTRESS Public Dataset

This directory contains the public ROK-FORTRESS evaluation dataset.

**Hugging Face:** <https://huggingface.co/datasets/ScaleAI/ROK-FORTRESS_public>

## Files

- `rok_fortress_public.tsv` (download from Hugging Face; not bundled in git) — 791 adversarial tasks across 4 NSPS risk domains, with English/Korean translations and US/Korean cultural adaptations.
- `croissant.json` — Croissant ML dataset metadata (schema, provenance, calibration κ values).

## Schema

| Column | Description |
|---|---|
| `TASK_ID` | Unique task identifier |
| `Phase` | Dataset phase / version tag |
| `Task Type` | `Culture Agnostic` (2 variants per task) or `Culture Specific` (4 variants per task) |
| `Tactic` | Adversarial tactic class |
| `Risk Domain` / `Risk Sub-Domain` | NSPS risk taxonomy |
| `Adversarial Prompt` / `Benign Prompt` | English (US-context) prompts |
| `Rubric` | Python-list-formatted YES/NO rubric questions |
| `Simplified Direct Request Variant` | Short, casual adversarial direct-request variant |
| `[KR] Translated - *` | Korean-language US-context variants |
| `[EN] Cultural Adapted - *` | English Korean-context variants (culture-specific only) |
| `[KR] Transcreated - *` | Korean-language Korean-context variants (culture-specific only) |
| `Harm Dimensions` | 7-dimension harm taxonomy labels |

## Loading

```python
from rok_fortress_eval.core.data_loader import load_and_expand_csv
variants = load_and_expand_csv("data/rok_fortress_public.tsv")
```

The loader expands each row into 2 variants (culture-agnostic) or 4 variants (culture-specific).

## License

The dataset is released under **CC-BY-4.0** (separate from the codebase license — the repository `LICENSE` covers the code under MIT).
