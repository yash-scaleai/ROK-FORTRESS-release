#!/usr/bin/env python3
"""
Prepare judge calibration dataset by merging human baseline labels with meta-eval task data.

Joins on UUID column and maps human labels to appropriate variants:
- ORIGINAL_RUBRIC_LABELS → original_en
- KR_TRANSLATION_ANSWERS → translated_kr
- KR_LOCALIZATION_ANSWERS → transcreated_kr (CULTURE_SPECIFIC)
- EN_LOCALIZATION_ANSWERS → cultural_adapted_en (CULTURE_SPECIFIC)
"""

import json
import ast
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


def parse_json_array(value) -> Optional[List[str]]:
    """Parse JSON array string from CSV, handling various formats."""
    if pd.isna(value) or value == 'undefined':
        return None
    
    if isinstance(value, list):
        return value
    
    if isinstance(value, str):
        value = value.strip()
        if not value or value == 'undefined':
            return None
        
        # Try parsing as JSON/Python literal
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
    
    return None


def parse_rubric(value) -> List[str]:
    """Parse rubric questions from CSV (reuse existing logic)."""
    if pd.isna(value) or not value:
        return []
    
    if isinstance(value, list):
        return value
    
    if isinstance(value, str):
        value = value.strip()
        # Try parsing as JSON array
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed]
        except Exception:
            pass
        
        # Fallback: return as single string
        return [value]
    
    return []


def load_human_baseline(csv_path: Path) -> pd.DataFrame:
    """Load human baseline CSV with answer columns."""
    print(f"Loading human baseline from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")
    
    # Key columns
    required = ['UUID', 'ORIGINAL_RUBRIC_LABELS', 'KR_TRANSLATION_ANSWERS', 
                'KR_LOCALIZATION_ANSWERS', 'EN_LOCALIZATION_ANSWERS']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def load_metaeval_tasks(csv_path: Path) -> pd.DataFrame:
    """Load meta-eval tasks CSV."""
    print(f"Loading meta-eval tasks from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")
    
    # Rename column
    if '(New) UUID' in df.columns:
        df = df.rename(columns={'(New) UUID': 'UUID'})
    
    required = ['UUID', 'Task UID', 'Task Type', 'Adversarial Response', 'Rubric']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def get_human_value(row: pd.Series, column: str):
    """Prefer human-suffixed columns from merged rows when available."""
    human_col = f"{column}_human"
    if human_col in row:
        return row[human_col]
    return row.get(column)


def create_variant_records(
    task_row: pd.Series,
    human_rows: List[pd.Series],
) -> List[Dict]:
    """
    Create calibration records for all variants of a task.
    
    Args:
        task_row: Meta-eval task data
        human_rows: List of human annotation rows (one per annotator)
    """
    task_id = str(task_row['Task UID'])
    task_type = str(task_row['Task Type'])
    uuid = str(task_row['UUID'])
    
    rubric = parse_rubric(task_row['Rubric'])
    if not rubric:
        return []
    
    variants = []
    
    # 1. Original EN
    if pd.notna(task_row.get('Adversarial Prompt')) and task_row['Adversarial Prompt'].strip():
        # Collect all human annotations for this variant
        human_annotations = []
        for human_row in human_rows:
            labels = parse_json_array(get_human_value(human_row, 'ORIGINAL_RUBRIC_LABELS'))
            if labels:
                human_annotations.append({
                    'annotator_date': str(human_row.get('Date', '')),
                    'review_level': str(human_row.get('Review Level', '')),
                    'labels': labels
                })
        
        if human_annotations:
            variants.append({
                'uuid': uuid,
                'task_id': task_id,
                'task_type': task_type,
                'variant_name': 'original_en',
                'prompt': str(task_row['Adversarial Prompt']),
                'response': str(task_row.get('Adversarial Response', '')),
                'rubric': rubric,
                'human_annotations': human_annotations,  # List of all annotators
                'risk_domain': str(task_row.get('Risk Domain', '')),
                'risk_subdomain': str(task_row.get('Risk Sub-Domain', '')),
            })
    
    # 2. Translated KR
    if pd.notna(task_row.get('[KR] Translated - Adversarial Prompt')) and task_row['[KR] Translated - Adversarial Prompt'].strip():
        kr_rubric = parse_rubric(task_row.get('[KR] Translated - Rubric', ''))
        
        # Collect all human annotations
        human_annotations = []
        for human_row in human_rows:
            labels = parse_json_array(get_human_value(human_row, 'KR_TRANSLATION_ANSWERS'))
            if labels:
                human_annotations.append({
                    'annotator_date': str(human_row.get('Date', '')),
                    'review_level': str(human_row.get('Review Level', '')),
                    'labels': labels
                })
        
        if human_annotations:
            variants.append({
                'uuid': uuid,
                'task_id': task_id,
                'task_type': task_type,
                'variant_name': 'translated_kr',
                'prompt': str(task_row['[KR] Translated - Adversarial Prompt']),
                'response': str(task_row.get('[KR] Translated - Adversarial Response', '')),
                'rubric': kr_rubric if kr_rubric else rubric,
                'human_annotations': human_annotations,
                'risk_domain': str(task_row.get('Risk Domain', '')),
                'risk_subdomain': str(task_row.get('Risk Sub-Domain', '')),
            })
    
    # 3. Cultural Adapted EN (CULTURE_SPECIFIC only)
    if 'SPECIFIC' in task_type:
        if pd.notna(task_row.get('[EN] Cultural Adapted - Adversarial Prompt')) and task_row['[EN] Cultural Adapted - Adversarial Prompt'].strip():
            ca_rubric = parse_rubric(task_row.get('[EN] Cultural Adapted - Rubric', ''))
            
            # Collect all human annotations
            human_annotations = []
            for human_row in human_rows:
                labels = parse_json_array(get_human_value(human_row, 'EN_LOCALIZATION_ANSWERS'))
                if labels:
                    human_annotations.append({
                        'annotator_date': str(human_row.get('Date', '')),
                        'review_level': str(human_row.get('Review Level', '')),
                        'labels': labels
                    })
            
            if human_annotations:
                variants.append({
                    'uuid': uuid,
                    'task_id': task_id,
                    'task_type': task_type,
                    'variant_name': 'cultural_adapted_en',
                    'prompt': str(task_row['[EN] Cultural Adapted - Adversarial Prompt']),
                    'response': str(task_row.get('[EN] Cultural Adapted - Adversarial Response', '')),
                    'rubric': ca_rubric if ca_rubric else rubric,
                    'human_annotations': human_annotations,
                    'risk_domain': str(task_row.get('Risk Domain', '')),
                    'risk_subdomain': str(task_row.get('Risk Sub-Domain', '')),
                })
        
        # 4. Transcreated KR (CULTURE_SPECIFIC only)
        if pd.notna(task_row.get('[KR] Transcreated - Adversarial Prompt')) and task_row['[KR] Transcreated - Adversarial Prompt'].strip():
            tc_rubric = parse_rubric(task_row.get('[KR] Transcreated - Rubric', ''))
            
            # Collect all human annotations
            human_annotations = []
            for human_row in human_rows:
                labels = parse_json_array(get_human_value(human_row, 'KR_LOCALIZATION_ANSWERS'))
                if labels:
                    human_annotations.append({
                        'annotator_date': str(human_row.get('Date', '')),
                        'review_level': str(human_row.get('Review Level', '')),
                        'labels': labels
                    })
            
            if human_annotations:
                variants.append({
                    'uuid': uuid,
                    'task_id': task_id,
                    'task_type': task_type,
                    'variant_name': 'transcreated_kr',
                    'prompt': str(task_row['[KR] Transcreated - Adversarial Prompt']),
                    'response': str(task_row.get('[KR] Transcreated - Adversarial Response', '')),
                    'rubric': tc_rubric if tc_rubric else rubric,
                    'human_annotations': human_annotations,
                    'risk_domain': str(task_row.get('Risk Domain', '')),
                    'risk_subdomain': str(task_row.get('Risk Sub-Domain', '')),
                })
    
    return variants


def main():
    base_dir = Path(__file__).resolve().parents[2]
    
    # Input files
    human_baseline_path = base_dir / 'data' / 'PHASE 1 & 2 ROK (ANALYSIS) - Meta Eval Data Review - Consolidated (1_7).csv'
    metaeval_path = base_dir / 'data' / 'PHASE 1 & 2 ROK (ANALYSIS) - meta_eval_final_new_uuid.csv'
    
    # Output
    output_dir = base_dir / 'runs' / 'judge-calibration'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'calibration_dataset.jsonl'
    
    print("=" * 80)
    print("PREPARE JUDGE CALIBRATION DATASET")
    print("=" * 80)
    print()
    
    # Load data
    human_df = load_human_baseline(human_baseline_path)
    metaeval_df = load_metaeval_tasks(metaeval_path)
    
    print()
    print(f"Merging on UUID column...")
    
    # Merge on UUID
    merged = metaeval_df.merge(human_df, on='UUID', how='inner', suffixes=('', '_human'))
    print(f"  Found {len(merged)} matching UUIDs")
    
    if len(merged) == 0:
        raise ValueError("No matching UUIDs found - check data alignment")
    
    print()
    print("Creating calibration records (deduplicating by UUID+variant)...")
    
    # Group merged rows by UUID to aggregate multiple human annotators
    from collections import defaultdict
    uuid_groups = defaultdict(list)
    for _, row in merged.iterrows():
        uuid_groups[row['UUID']].append(row)
    
    print(f"  Found {len(uuid_groups)} unique UUIDs")
    print(f"  Average annotators per UUID: {len(merged) / len(uuid_groups):.1f}")
    
    # Create variant records - one per unique (UUID, variant_name)
    all_variants = []
    skipped = 0
    
    for uuid, rows in uuid_groups.items():
        # Use first row for task data (all rows have same task data, just different human annotations)
        task_row = rows[0]
        
        # Pass all human rows for aggregation
        variants = create_variant_records(task_row, rows)
        if variants:
            all_variants.extend(variants)
        else:
            skipped += 1
    
    print(f"  Created {len(all_variants)} unique calibration records (deduplicated)")
    print(f"  Skipped {skipped} tasks with missing data")
    
    # Filter out records without human annotations
    variants_with_labels = [v for v in all_variants if v.get('human_annotations')]
    variants_without_labels = len(all_variants) - len(variants_with_labels)
    
    print(f"  Records with human annotations: {len(variants_with_labels)}")
    print(f"  Records without human annotations: {variants_without_labels} (excluded)")
    
    # Write JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for variant in variants_with_labels:
            json.dump(variant, f, ensure_ascii=False)
            f.write('\n')
    
    # Statistics
    print()
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print()
    
    variant_counts = {}
    for v in variants_with_labels:
        vname = v['variant_name']
        variant_counts[vname] = variant_counts.get(vname, 0) + 1
    
    print("Variants:")
    for vname in ['original_en', 'translated_kr', 'cultural_adapted_en', 'transcreated_kr']:
        count = variant_counts.get(vname, 0)
        if count > 0:
            print(f"  {vname:20s}: {count:4d}")
    
    print()
    
    # Unique tasks
    unique_uuids = len(set(v['uuid'] for v in variants_with_labels))
    unique_tasks = len(set(v['task_id'] for v in variants_with_labels))
    print(f"Unique UUIDs: {unique_uuids}")
    print(f"Unique Task IDs: {unique_tasks}")
    
    print()
    print(f"✓ Output: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    main()
