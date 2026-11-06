#!/usr/bin/env python3
"""
Sample meta-evaluation dataset with strategic model selection and balancing.

Strategy:
1. For each (subdomain, task_type), rank tasks by model coverage (5 models > 4 > 3...)
2. Sample N tasks per subdomain following target distribution
3. For each task, randomly select ONE model from those available
4. Balance model distribution within ±5% (target: ~18 tasks per model for 90 total)
5. Maintain phase ratio per subdomain
"""

import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set


# Target distribution (excluding Information Leakage domain)
# Using actual subdomain names from the data
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


def group_by_task_and_category(tasks: List[Dict]) -> Dict:
    """
    Group variants by (task_id, subdomain, task_type) and track which models responded.
    Returns: {(task_id, domain, subdomain, task_type): {model: [variants]}}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    
    for task in tasks:
        key = (
            task['task_id'],
            task['risk_domain'],
            task['risk_subdomain'],
            task['task_type']
        )
        model = task['meta_eval_model']
        grouped[key][model].append(task)
    
    return grouped


def calculate_model_coverage(task_models: Dict) -> int:
    """Return number of models that responded to this task."""
    return len(task_models)


def sample_tasks_with_balancing(
    grouped_tasks: Dict,
    target_dist: Dict,
    seed: int = 42
) -> List[Dict]:
    """
    Sample tasks following the target distribution with model balancing.
    """
    random.seed(seed)
    
    # Track selected tasks and model counts
    selected_variants = []
    model_counts = Counter()
    
    # All models
    all_models = [
        'openai/gpt-5',
        'anthropic/claude-opus-4-5',
        'gemini/gemini-3-pro-preview',
        'vllm/skt/A.X-4.0',
        'vllm/LGAI-EXAONE/EXAONE-4.0-32B'
    ]
    
    total_target = sum(target_dist.values())
    target_per_model = total_target / len(all_models)  # ~18 for 90 tasks
    tolerance = 0.10 * total_target  # ±10%
    
    print(f"Target tasks: {total_target}")
    print(f"Target per model: {target_per_model:.1f} ± {tolerance:.1f}")
    print()
    
    # Group tasks by (domain, subdomain, task_type)
    category_tasks = defaultdict(list)
    for (task_id, domain, subdomain, task_type), models in grouped_tasks.items():
        category_key = (domain, subdomain, task_type)
        category_tasks[category_key].append((task_id, models))
    
    # Process each category
    for category_key, target_count in sorted(target_dist.items()):
        domain, subdomain, task_type = category_key
        available_tasks = category_tasks.get(category_key, [])
        
        if not available_tasks:
            print(f"⚠️  No tasks for {task_type} in {subdomain[:50]}...")
            continue
        
        # Sort by model coverage (descending) - prioritize tasks where all models responded
        available_tasks.sort(key=lambda x: calculate_model_coverage(x[1]), reverse=True)
        
        print(f"\n{task_type} | {subdomain[:60]}...")
        print(f"  Target: {target_count}, Available: {len(available_tasks)}")
        
        # Sample exactly target_count tasks with weighted model selection
        selected_for_category = []
        
        for i, (task_id, task_models) in enumerate(available_tasks[:target_count]):
            # Choose model: prefer underrepresented models
            available_models = list(task_models.keys())
            
            # Weight by inverse of current count (strongly favor underrepresented)
            # Use squared inverse to make the preference stronger
            weights = [1.0 / ((model_counts[m] + 1) ** 2) for m in available_models]
            chosen_model = random.choices(available_models, weights=weights)[0]
            
            # Pick one random variant from this model
            variant = random.choice(task_models[chosen_model])
            selected_for_category.append((task_id, chosen_model, variant))
            model_counts[chosen_model] += 1
        
        # Add to final selection
        selected_variants.extend([v for _, _, v in selected_for_category])
        
        # Show category stats
        category_models = Counter(m for _, m, _ in selected_for_category)
        print(f"  Selected: {len(selected_for_category)} tasks")
        for model in all_models:
            short_name = model.split('/')[-1]
            print(f"    {short_name}: {category_models[model]}")
    
    return selected_variants


def main():
    base_dir = Path(__file__).resolve().parents[2] / 'runs' / 'meta-eval'
    merged_file = base_dir / 'merged_complete_tasks.jsonl'
    output_file = base_dir / 'meta_eval_sampled_dataset.jsonl'
    
    print("=" * 80)
    print("META-EVALUATION DATASET SAMPLING")
    print("=" * 80)
    print()
    
    # Load tasks
    print("Loading complete tasks...")
    all_tasks = load_complete_tasks(merged_file)
    print(f"Loaded {len(all_tasks)} variants")
    print()
    
    # Group by task and category
    print("Grouping by task and category...")
    grouped = group_by_task_and_category(all_tasks)
    print(f"Found {len(grouped)} unique tasks")
    print()
    
    # Sample with balancing
    print("Sampling with model balancing...")
    print("=" * 80)
    
    sampled = sample_tasks_with_balancing(grouped, TARGET_DISTRIBUTION)
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for variant in sampled:
            json.dump(variant, f, ensure_ascii=False)
            f.write('\n')
    
    print()
    print("=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print()
    
    # Overall stats
    print(f"Total sampled variants: {len(sampled)}")
    print()
    
    # Model distribution
    model_counts = Counter(v['meta_eval_model'] for v in sampled)
    print("Model distribution:")
    for model in sorted(model_counts.keys()):
        short_name = model.split('/')[-1]
        count = model_counts[model]
        pct = count / len(sampled) * 100
        print(f"  {short_name:30s}: {count:3d} ({pct:5.1f}%)")
    print()
    
    # Phase distribution
    phase_counts = Counter(v['meta_eval_phase'] for v in sampled)
    print("Phase distribution:")
    for phase in sorted(phase_counts.keys()):
        count = phase_counts[phase]
        pct = count / len(sampled) * 100
        print(f"  {phase}: {count:3d} ({pct:5.1f}%)")
    print()
    
    # Task type distribution
    task_type_counts = Counter(v['task_type'] for v in sampled)
    print("Task type distribution:")
    for task_type in sorted(task_type_counts.keys()):
        count = task_type_counts[task_type]
        print(f"  {task_type}: {count}")
    print()
    
    # Count unique tasks
    unique_tasks = len(set(v['task_id'] for v in sampled))
    print(f"Unique tasks: {unique_tasks}")
    print()
    
    print(f"✓ Output saved to: {output_file}")
    print(f"  Size: {output_file.stat().st_size / (1024*1024):.2f} MB")


if __name__ == '__main__':
    main()

