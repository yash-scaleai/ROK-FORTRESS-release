#!/usr/bin/env python3
"""
Sample meta-evaluation dataset and include ALL variants for each selected task.

For each selected task:
- CULTURE_AGNOSTIC_TRANSLATABLE: Include original_en + translated_kr (2 variants)
- CULTURE_SPECIFIC_TRANSLATABLE: Include all 4 variants (original_en, translated_kr, cultural_adapted_en, transcreated_kr)
"""

import json
import random
import csv
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List


# Target distribution (excluding Information Leakage domain)
TARGET_DISTRIBUTION = {
    # CBRNE
    ('Chemical, Biological, Radiological, Nuclear, and Explosive (CBRNE)', 
     'Chemical: chemical synthesis guidance, disclosure of exposure pathways, assistance to obtain restricted chemicals, strategies to overcome decontamination measures, etc.',
     'CULTURE_AGNOSTIC_TRANSLATABLE'): 4,
    ('Chemical, Biological, Radiological, Nuclear, and Explosive (CBRNE)',
     'Chemical: chemical synthesis guidance, disclosure of exposure pathways, assistance to obtain restricted chemicals, strategies to overcome decontamination measures, etc.',
     'CULTURE_SPECIFIC_TRANSLATABLE'): 3,
    
    ('Chemical, Biological, Radiological, Nuclear, and Explosive (CBRNE)',
     'Biological: pathogen engineering assistance for malicious purposes, bioweapon design assistance, dissemination strategy optimization, synthetic biology automation',
     'CULTURE_AGNOSTIC_TRANSLATABLE'): 4,
    ('Chemical, Biological, Radiological, Nuclear, and Explosive (CBRNE)',
     'Biological: pathogen engineering assistance for malicious purposes, bioweapon design assistance, dissemination strategy optimization, synthetic biology automation',
     'CULTURE_SPECIFIC_TRANSLATABLE'): 3,
    
    ('Chemical, Biological, Radiological, Nuclear, and Explosive (CBRNE)',
     'Radiological and Nuclear (WMD): nuclear material acquisition advice, weapon design assistance, enhanced radiation weapon development',
     'CULTURE_AGNOSTIC_TRANSLATABLE'): 4,
    ('Chemical, Biological, Radiological, Nuclear, and Explosive (CBRNE)',
     'Radiological and Nuclear (WMD): nuclear material acquisition advice, weapon design assistance, enhanced radiation weapon development',
     'CULTURE_SPECIFIC_TRANSLATABLE'): 3,
    
    ('Chemical, Biological, Radiological, Nuclear, and Explosive (CBRNE)',
     'Explosives: improvised explosive device assistance, detonation method assistance, target analysis and impact maximization',
     'CULTURE_AGNOSTIC_TRANSLATABLE'): 5,
    ('Chemical, Biological, Radiological, Nuclear, and Explosive (CBRNE)',
     'Explosives: improvised explosive device assistance, detonation method assistance, target analysis and impact maximization',
     'CULTURE_SPECIFIC_TRANSLATABLE'): 4,
    
    # Political Violence & Terrorism
    ('Political Violence & Terrorism',
     'Terrorism: financing guidance, recruitment assistance, target selection, attack planning, counter-terrorism evasion',
     'CULTURE_AGNOSTIC_TRANSLATABLE'): 4,
    ('Political Violence & Terrorism',
     'Terrorism: financing guidance, recruitment assistance, target selection, attack planning, counter-terrorism evasion',
     'CULTURE_SPECIFIC_TRANSLATABLE'): 9,
    
    ('Political Violence & Terrorism',
     'Political Violence: disinformation campaigns, riot instigation, violence incitement',
     'CULTURE_AGNOSTIC_TRANSLATABLE'): 4,
    ('Political Violence & Terrorism',
     'Political Violence: disinformation campaigns, riot instigation, violence incitement',
     'CULTURE_SPECIFIC_TRANSLATABLE'): 6,
    
    ('Political Violence & Terrorism',
     'Illegal Weapons: 3d-printed weapons, illegal weapon modification, illegal weapon acquisition, illegal gun development',
     'CULTURE_AGNOSTIC_TRANSLATABLE'): 4,
    ('Political Violence & Terrorism',
     'Illegal Weapons: 3d-printed weapons, illegal weapon modification, illegal weapon acquisition, illegal gun development',
     'CULTURE_SPECIFIC_TRANSLATABLE'): 3,
    
    # Criminal & Financial
    ('Criminal & Financial Illicit Activities',
     'Coordination of Illicit Activities: drug trafficking, human trafficking, weapons trafficking, smuggling, criminal operations',
     'CULTURE_AGNOSTIC_TRANSLATABLE'): 6,
    ('Criminal & Financial Illicit Activities',
     'Coordination of Illicit Activities: drug trafficking, human trafficking, weapons trafficking, smuggling, criminal operations',
     'CULTURE_SPECIFIC_TRANSLATABLE'): 6,
    
    ('Criminal & Financial Illicit Activities',
     'Fraud: counterfeit (i.e., identity and document forgery), financial fraud (i.e., help with payment fraud) impersonation (i.e., mimicking another person)',
     'CULTURE_AGNOSTIC_TRANSLATABLE'): 6,
    ('Criminal & Financial Illicit Activities',
     'Fraud: counterfeit (i.e., identity and document forgery), financial fraud (i.e., help with payment fraud) impersonation (i.e., mimicking another person)',
     'CULTURE_SPECIFIC_TRANSLATABLE'): 4,
    
    ('Criminal & Financial Illicit Activities',
     'Privacy/Scams: data poisoning, model inversion, privacy leaks, data negligence, phishing attacks',
     'CULTURE_AGNOSTIC_TRANSLATABLE'): 4,
    ('Criminal & Financial Illicit Activities',
     'Privacy/Scams: data poisoning, model inversion, privacy leaks, data negligence, phishing attacks',
     'CULTURE_SPECIFIC_TRANSLATABLE'): 4,
}


def load_complete_tasks(merged_file: Path) -> List[Dict]:
    """Load all complete task variants."""
    tasks = []
    with open(merged_file, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def validate_variants(variants: List[Dict], task_type: str) -> bool:
    """
    Validate that a task has the correct number of variants.
    - AGNOSTIC: 2 variants (original_en, translated_kr)
    - SPECIFIC: 4 variants (original_en, translated_kr, cultural_adapted_en, transcreated_kr)
    """
    expected_count = 2 if 'AGNOSTIC' in task_type else 4
    actual_count = len(variants)
    
    if actual_count != expected_count:
        return False
    
    # Check for expected variant names
    variant_names = {v['variant_name'] for v in variants}
    if 'AGNOSTIC' in task_type:
        expected_names = {'original_en', 'translated_kr'}
    else:
        expected_names = {'original_en', 'translated_kr', 'cultural_adapted_en', 'transcreated_kr'}
    
    return variant_names == expected_names


def group_by_task_and_model(tasks: List[Dict]) -> Dict:
    """
    Group variants by (task_id, phase, model).
    Returns: {(task_id, domain, subdomain, task_type, phase): {model: [variants_for_that_model]}}
    
    NOTE: Phase is critical! Same task_id can appear in both Phase 1 and Phase 2
    with different task types and variant structures.
    
    Also deduplicates variants (some source data has duplicates).
    """
    grouped = defaultdict(lambda: defaultdict(dict))  # Use dict to dedupe by variant_name
    
    for task in tasks:
        key = (
            task['task_id'],
            task['risk_domain'],
            task['risk_subdomain'],
            task['task_type'],
            task['meta_eval_phase']  # CRITICAL: Include phase to separate P1/P2
        )
        model = task['meta_eval_model']
        variant_name = task['variant_name']
        
        # Use variant_name as key to automatically deduplicate
        grouped[key][model][variant_name] = task
    
    # Convert back to lists and validate
    result = {}
    skipped_count = 0
    
    for key, models in grouped.items():
        task_id, domain, subdomain, task_type, phase = key
        valid_models = {}
        
        for model, variant_dict in models.items():
            variant_list = list(variant_dict.values())
            
            # Validate variant count and names
            if validate_variants(variant_list, task_type):
                valid_models[model] = variant_list
            else:
                skipped_count += 1
                print(f"⚠️  Skipping incomplete/corrupt: Task {task_id} ({phase}) - {model}: "
                      f"{len(variant_list)} variants (expected {2 if 'AGNOSTIC' in task_type else 4})")
        
        # Only include task if at least one model has valid data
        if valid_models:
            result[key] = valid_models
    
    if skipped_count > 0:
        print(f"\nSkipped {skipped_count} incomplete/corrupt task-model combinations")
        print()
    
    return result


def sample_tasks(grouped_tasks: Dict, target_dist: Dict, seed: int = 42) -> List[Dict]:
    """
    Sample tasks at the TASK level (not variant level).
    For each selected task:
    1. Choose ONE model (with balancing at both global and subdomain level)
    2. Include ALL variants from that model
    
    Model selection uses a combined weight that considers:
    - Global model distribution (overall balance)
    - Subdomain-specific model distribution (local balance)
    """
    random.seed(seed)
    
    all_models = [
        'openai/gpt-5',
        'anthropic/claude-opus-4-5',
        'gemini/gemini-3-pro-preview',
        'vllm/skt/A.X-4.0',
        'vllm/LGAI-EXAONE/EXAONE-4.0-32B'
    ]
    
    model_counts = Counter()  # Global model counts
    subdomain_model_counts = defaultdict(Counter)  # Per-subdomain model counts
    selected_variants = []
    selected_task_ids = set()  # Track already-selected task_ids to avoid duplicates
    
    # EXCLUDED TASKS: Tasks with data corruption or other issues
    excluded_task_ids = {'80'}  # Task 80 (phase2): Has mixed data from two different tasks
    
    # Group tasks by (domain, subdomain, task_type)
    # Note: We keep phase separate in task identity but pool across phases for sampling
    category_tasks = defaultdict(list)
    for (task_id, domain, subdomain, task_type, phase), model_variants in grouped_tasks.items():
        category_key = (domain, subdomain, task_type)
        # Count how many models responded
        models_responded = len(model_variants)
        category_tasks[category_key].append((task_id, phase, model_variants, models_responded))
    
    # Process each category
    for category_key, target_count in sorted(target_dist.items()):
        domain, subdomain, task_type = category_key
        available_tasks = category_tasks.get(category_key, [])
        
        if not available_tasks:
            print(f"⚠️  No tasks for {task_type} in {subdomain[:50]}...")
            continue
        
        # Filter out already-selected task_ids and excluded tasks
        available_tasks = [(tid, p, mv, c) for tid, p, mv, c in available_tasks 
                          if tid not in selected_task_ids and tid not in excluded_task_ids]
        
        # Sort by model coverage (descending) - prioritize tasks where more models responded
        available_tasks.sort(key=lambda x: x[3], reverse=True)  # x[3] is coverage, x[0]=task_id, x[1]=phase, x[2]=model_variants, x[3]=coverage
        
        print(f"\n{task_type} | {subdomain[:60]}...")
        print(f"  Target: {target_count} tasks, Available: {len(available_tasks)}")
        
        # Sample target_count tasks
        selected_count = min(target_count, len(available_tasks))
        
        for task_id, phase, model_variants, coverage in available_tasks[:selected_count]:
            # Mark this task_id as selected
            selected_task_ids.add(task_id)
            # Choose ONE model for this task
            available_models = list(model_variants.keys())
            
            # Combined weighting:
            # - Global weight: favor models with fewer tasks overall
            # - Subdomain weight: favor models with fewer tasks in this subdomain
            weights = []
            for m in available_models:
                global_weight = 1.0 / ((model_counts[m] + 1) ** 2)
                subdomain_weight = 1.0 / ((subdomain_model_counts[subdomain][m] + 1) ** 1.5)
                # Combine both weights (multiplicative)
                combined_weight = global_weight * subdomain_weight
                weights.append(combined_weight)
            
            chosen_model = random.choices(available_models, weights=weights)[0]
            
            # Include ALL variants from this model
            task_variants = model_variants[chosen_model]
            selected_variants.extend(task_variants)
            model_counts[chosen_model] += 1
            subdomain_model_counts[subdomain][chosen_model] += 1
            
            print(f"    Task {task_id} ({phase}): {len(task_variants)} variants from {chosen_model.split('/')[-1]}")
        
        print(f"  Selected: {selected_count} tasks")
    
    return selected_variants


def flatten_for_csv(record: dict) -> dict:
    """Flatten record for CSV with only requested columns."""
    flat = {}
    
    # Core columns
    flat['task_id'] = record.get('task_id')
    flat['task_type'] = record.get('task_type')
    flat['variant_name'] = record.get('variant_name')
    flat['risk_domain'] = record.get('risk_domain')
    flat['risk_subdomain'] = record.get('risk_subdomain')
    flat['phase'] = record.get('meta_eval_phase')
    flat['model_name'] = record.get('meta_eval_model')
    
    # Prompts and responses
    flat['adversarial_prompt'] = record.get('adversarial_prompt')
    adv_resp = record.get('adversarial_response') or {}
    flat['adversarial_response_reasoning_content'] = adv_resp.get('reasoning_content')
    flat['adversarial_response_text'] = adv_resp.get('text', '')
    
    flat['benign_prompt'] = record.get('benign_prompt')
    benign_resp = record.get('benign_response') or {}
    flat['benign_response_text'] = benign_resp.get('text', '')
    
    benign_eval = record.get('benign_evaluation') or {}
    flat['benign_refusal_reason'] = benign_eval.get('refusal_reason')
    
    flat['rubric'] = '|'.join(record.get('rubric', [])) if record.get('rubric') else ''
    
    # Additional columns (adversarial evaluation details)
    adv_eval = record.get('adversarial_evaluation') or {}
    flat['adversarial_refused'] = adv_eval.get('refused')
    flat['adversarial_refusal_reason'] = adv_eval.get('refusal_reason')
    flat['adversarial_final_score'] = adv_eval.get('final_score')
    
    # Judge evaluation
    judge_evals = adv_eval.get('judge_evaluations', {}) or {}
    for judge_model, eval_data in judge_evals.items():
        judge_short = judge_model.split('/')[-1]
        flat[f'judge_{judge_short}_score'] = eval_data.get('score')
        flat[f'judge_{judge_short}_yes_percentage'] = eval_data.get('yes_percentage')
    
    return flat


def main():
    base_dir = Path(__file__).resolve().parents[2] / 'runs' / 'meta-eval'
    merged_file = base_dir / 'merged_complete_tasks.jsonl'
    output_jsonl = base_dir / 'meta_eval_final_with_all_variants.jsonl'
    output_csv = base_dir / 'meta_eval_final_with_all_variants.csv'
    
    print("=" * 80)
    print("META-EVALUATION DATASET: SAMPLING WITH ALL VARIANTS")
    print("=" * 80)
    print()
    
    # Load tasks
    print("Loading complete tasks...")
    all_tasks = load_complete_tasks(merged_file)
    print(f"Loaded {len(all_tasks)} variants")
    print()
    
    # Group by task and model
    print("Grouping by task and model...")
    grouped = group_by_task_and_model(all_tasks)
    print(f"Found {len(grouped)} unique tasks")
    print()
    
    # Sample
    print("Sampling tasks (will include ALL variants for each)...")
    print("=" * 80)
    
    sampled = sample_tasks(grouped, TARGET_DISTRIBUTION)
    
    # Write JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for variant in sampled:
            json.dump(variant, f, ensure_ascii=False)
            f.write('\n')
    
    # Write CSV
    flat_records = [flatten_for_csv(v) for v in sampled]
    if flat_records:
        fieldnames = list(flat_records[0].keys())
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_records)
    
    # Statistics
    print()
    print("=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print()
    
    print(f"Total variants: {len(sampled)}")
    
    unique_tasks = len(set(v['task_id'] for v in sampled))
    print(f"Unique tasks: {unique_tasks}")
    print()
    
    # Model distribution (at task level)
    task_to_model = {}
    for v in sampled:
        task_to_model[v['task_id']] = v['meta_eval_model']
    
    task_model_counts = Counter(task_to_model.values())
    print("Tasks by model (task-level distribution):")
    for model in sorted(task_model_counts.keys()):
        short_name = model.split('/')[-1]
        count = task_model_counts[model]
        pct = count / unique_tasks * 100
        print(f"  {short_name:30s}: {count:3d} tasks ({pct:5.1f}%)")
    print()
    
    # Variant distribution by model
    variant_counts = Counter(v['meta_eval_model'] for v in sampled)
    print("Variants by model:")
    for model in sorted(variant_counts.keys()):
        short_name = model.split('/')[-1]
        count = variant_counts[model]
        print(f"  {short_name:30s}: {count:4d} variants")
    print()
    
    # Phase distribution
    phase_counts = Counter(v['meta_eval_phase'] for v in sampled)
    print("Variants by phase:")
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase}: {count}")
    print()
    
    # Variant distribution
    variant_counts = Counter(v['variant_name'] for v in sampled)
    print("By variant:")
    for variant, count in sorted(variant_counts.items()):
        print(f"  {variant}: {count}")
    print()
    
    print(f"✓ JSONL: {output_jsonl}")
    print(f"  Size: {output_jsonl.stat().st_size / (1024*1024):.2f} MB")
    print(f"✓ CSV: {output_csv}")
    print(f"  Size: {output_csv.stat().st_size / (1024*1024):.2f} MB")


if __name__ == '__main__':
    main()

