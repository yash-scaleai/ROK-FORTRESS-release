"""
Data loading utilities for ROK-FORTRESS TSV dataset.
"""

import ast
import re
from typing import List, Optional
import pandas as pd

from ..models import PromptVariant


COLUMN_MAPPING = {
    'TASK_ID': 'task_id',
    'Task Type': 'task_type',
    'Adversarial Prompt': 'en_adversarial_prompt',
    'Risk Domain': 'risk_domain',
    'Risk Sub-Domain': 'risk_subdomain',
    'Benign Prompt': 'benign_prompt',
    'Rubric': 'rubric',
    '[KR] Translated - Adversarial Prompt': 'kr_prompt_translated',
    '[KR] Translated - Benign Prompt': 'kr_benign_prompt',
    '[KR] Translated - Rubric': 'rubric_translated',
    '[EN] Cultural Adapted - Adversarial Prompt': 'en_prompt_cultural_adapted',
    '[EN] Cultural Adapted - Benign Prompt': 'en_prompt_cultural_adapted_benign',
    '[EN] Cultural Adapted - Rubric': 'rubric_cultural_adapted',
    '[KR] Transcreated - Adversarial Prompt': 'kr_prompt_transcreated',
    '[KR] Transcreated - Benign Prompt': 'kr_benign_prompt_transcreated',
    '[KR] Transcreated - Rubric': 'rubric_transcreated',
    'Simplified Direct Request Variant': 'en_direct_request',
    '[KR] Translated - Simplified Direct Request Variant': 'kr_direct_request_translated',
    '[EN] Cultural Adapted - Simplified Direct Request Variant': 'en_direct_request_cultural_adapted',
    '[KR] Transcreated - Simplified Direct Request Variant': 'kr_direct_request_transcreated',
}

CURLY_OPEN = '‘'
CURLY_CLOSE = '’'
STRAIGHT = "'"


def _try_split_quoted_list(s: str) -> List[str]:
    """Try to split a string that looks like quoted list items."""
    sep_curly = CURLY_CLOSE + ", " + CURLY_OPEN
    sep_straight = "', '"
    sep_mixed1 = CURLY_CLOSE + ", '"
    sep_mixed2 = "', " + CURLY_OPEN
    sep_space = "' '"
    sep_space_curly = CURLY_CLOSE + " " + CURLY_OPEN

    for sep in [sep_curly, sep_straight, sep_mixed1, sep_mixed2, sep_space, sep_space_curly]:
        if sep in s:
            parts = s.split(sep)
            if len(parts) > 1:
                questions = []
                for p in parts:
                    p = p.strip()
                    p = p.replace(CURLY_OPEN, "'").replace(CURLY_CLOSE, "'")
                    if p:
                        questions.append(p)
                if questions:
                    return questions
    return []


def parse_rubric(rubric_str) -> List[str]:
    """Parse rubric from CSV (handles various formats)."""
    if pd.isna(rubric_str) or not rubric_str:
        return []
    if isinstance(rubric_str, list):
        return rubric_str
    if isinstance(rubric_str, str):
        rubric_str = rubric_str.strip()
        rubric_str = rubric_str.replace('“', '"').replace('”', '"')

        if rubric_str.startswith("'[") and rubric_str.endswith("]'"):
            inner = rubric_str[1:-1]
            try:
                parsed = ast.literal_eval(inner)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed]
            except Exception:
                pass

        if rubric_str.startswith('[') and rubric_str.endswith(']'):
            inner = rubric_str[1:-1].strip()

            if not inner.startswith((STRAIGHT, CURLY_OPEN, '"')):
                parts = [p.strip() for p in inner.split(',') if p.strip()]
                if len(parts) > 1:
                    return parts

            if inner and inner[0] in (STRAIGHT, CURLY_OPEN, '"'):
                inner = inner[1:]
            if inner and inner[-1] in (STRAIGHT, CURLY_CLOSE, '"'):
                inner = inner[:-1]

            result = _try_split_quoted_list(inner)
            if result:
                return result

        if (rubric_str.startswith(STRAIGHT) or rubric_str.startswith(CURLY_OPEN)) and \
           (rubric_str.endswith(STRAIGHT) or rubric_str.endswith(CURLY_CLOSE)):
            inner = rubric_str
            if inner[0] in (STRAIGHT, CURLY_OPEN):
                inner = inner[1:]
            if inner[-1] in (STRAIGHT, CURLY_CLOSE):
                inner = inner[:-1]

            result = _try_split_quoted_list(inner)
            if result:
                return result

        if '?, ' in rubric_str or '가?, ' in rubric_str or '나요?, ' in rubric_str:
            parts = re.split(r'\?\s*,\s*', rubric_str)
            if len(parts) > 1:
                questions = []
                for p in parts:
                    p = p.strip()
                    p = p.replace(CURLY_OPEN, "'").replace(CURLY_CLOSE, "'")
                    if p and not p.endswith('?'):
                        p = p + '?'
                    if p:
                        questions.append(p)
                if questions:
                    return questions

        if '\n' in rubric_str:
            lines = [line.strip().replace(CURLY_OPEN, "'").replace(CURLY_CLOSE, "'")
                     for line in rubric_str.split('\n') if line.strip()]
            if len(lines) > 1:
                return lines

        sanitized = rubric_str.replace(CURLY_OPEN, "'").replace(CURLY_CLOSE, "'")
        try:
            parsed = ast.literal_eval(sanitized)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
        except Exception:
            pass

        parts = re.split(r'(\d+\.\s+)', rubric_str)
        questions = []
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                num_part = parts[i]
                q_part = parts[i+1].strip().rstrip("']").rstrip()
                questions.append(num_part + q_part)
        if questions:
            return questions

        return [rubric_str.replace(CURLY_OPEN, "'").replace(CURLY_CLOSE, "'")]
    return []


def _str_or_empty(value) -> str:
    s = str(value).strip()
    return '' if s == 'nan' else s


def load_and_expand_csv(csv_path: str, max_rows: Optional[int] = None) -> List[PromptVariant]:
    """Load ROK-FORTRESS CSV/TSV and expand into prompt variants."""
    if csv_path.endswith('.tsv'):
        df = pd.read_csv(csv_path, sep='\t')
    else:
        df = pd.read_csv(csv_path)
    if max_rows:
        df = df.head(max_rows)

    df = df.rename(columns=COLUMN_MAPPING)

    variants = []
    for _, row in df.iterrows():
        task_id = str(row.get('task_id', ''))
        task_type = str(row.get('task_type', ''))
        if not task_id:
            continue

        en_adv = _str_or_empty(row.get('en_adversarial_prompt', ''))
        en_benign = _str_or_empty(row.get('benign_prompt', ''))
        en_rubric = parse_rubric(row.get('rubric', ''))
        en_direct = _str_or_empty(row.get('en_direct_request', ''))
        if en_adv and en_rubric:
            variants.append(PromptVariant(
                task_id=task_id, task_type=task_type, variant_name='original_en',
                adversarial_prompt=en_adv, benign_prompt=en_benign, rubric=en_rubric,
                risk_domain=str(row.get('risk_domain', '')),
                risk_subdomain=str(row.get('risk_subdomain', '')),
                direct_prompt=en_direct,
            ))

        kr_adv = _str_or_empty(row.get('kr_prompt_translated', ''))
        kr_benign = _str_or_empty(row.get('kr_benign_prompt', ''))
        kr_rubric = parse_rubric(row.get('rubric_translated', ''))
        kr_direct = _str_or_empty(row.get('kr_direct_request_translated', ''))
        if kr_adv and kr_rubric:
            variants.append(PromptVariant(
                task_id=task_id, task_type=task_type, variant_name='translated_kr',
                adversarial_prompt=kr_adv, benign_prompt=kr_benign, rubric=kr_rubric,
                risk_domain=str(row.get('risk_domain', '')),
                risk_subdomain=str(row.get('risk_subdomain', '')),
                direct_prompt=kr_direct,
            ))

        if 'specific' in task_type.lower():
            ca_adv = _str_or_empty(row.get('en_prompt_cultural_adapted', ''))
            ca_benign = _str_or_empty(row.get('en_prompt_cultural_adapted_benign', ''))
            ca_rubric = parse_rubric(row.get('rubric_cultural_adapted', ''))
            ca_direct = _str_or_empty(row.get('en_direct_request_cultural_adapted', ''))
            if ca_adv and ca_rubric:
                variants.append(PromptVariant(
                    task_id=task_id, task_type=task_type, variant_name='cultural_adapted_en',
                    adversarial_prompt=ca_adv, benign_prompt=ca_benign, rubric=ca_rubric,
                    risk_domain=str(row.get('risk_domain', '')),
                    risk_subdomain=str(row.get('risk_subdomain', '')),
                    direct_prompt=ca_direct,
                ))

            tc_adv = _str_or_empty(row.get('kr_prompt_transcreated', ''))
            tc_benign = _str_or_empty(row.get('kr_benign_prompt_transcreated', ''))
            tc_rubric = parse_rubric(row.get('rubric_transcreated', ''))
            tc_direct = _str_or_empty(row.get('kr_direct_request_transcreated', ''))
            if tc_adv and tc_rubric:
                variants.append(PromptVariant(
                    task_id=task_id, task_type=task_type, variant_name='transcreated_kr',
                    adversarial_prompt=tc_adv, benign_prompt=tc_benign, rubric=tc_rubric,
                    risk_domain=str(row.get('risk_domain', '')),
                    risk_subdomain=str(row.get('risk_subdomain', '')),
                    direct_prompt=tc_direct,
                ))

    return variants
