"""
Experiment 1: Global Alignment Overfitting

Goal: Show that Global Models fail due to Contextual Blindness, not Linguistic Inability.

Method:
- Linguistic Drop = HarmScore(V(Ko)) - HarmScore(V(En)) on CA tasks
  Hypothesis: Small drop indicates linguistic capability preserved
  
- Contextual Drop = HarmScore(V(En,KR)) - HarmScore(V(En,US)) on CS tasks
  Hypothesis: Large drop indicates contextual blindness (alignment overfitting)

Conclusion: If Contextual Drop >> Linguistic Drop, we show Alignment Overfitting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .base import BaseExperiment, ExperimentConfig, ExperimentResult, create_experiment_result
from .exp_data.loaders import ConsolidatedResult
from .exp_data.filters import (
    filter_culture_agnostic,
    filter_culture_specific,
    filter_by_variant,
    classify_all_tasks,
)
from .exp_data.variants import (
    ORIGINAL_EN,
    TRANSLATED_KR,
    CULTURAL_ADAPTED_EN,
    TRANSCREATED_KR,
    group_by_task_id,
)
from .exp_data.models import (
    is_global_model,
    is_closed_source_model,
    is_open_source_model,
    get_model_display_name,
    GLOBAL_MODELS,
    CLOSED_SOURCE_MODELS,
    OPEN_SOURCE_MODELS,
)
from .exp_data.loaders import get_available_target_models
from .metrics.harm_scores import compute_harm_score_for_result, HarmScoreResult
from .metrics.alignment_metrics import (
    compute_linguistic_drop,
    compute_contextual_drop,
    compute_averaged_contextual_drop,
    compute_alignment_overfitting_score,
    DropResult,
)
from .visualization.plots import plot_drop_comparison, plot_harm_scores_by_variant


class GlobalAlignmentExperiment(BaseExperiment):
    """
    Experiment 1: Global Alignment Overfitting Analysis.
    
    Tests whether global models fail on Korean contexts due to:
    - Linguistic inability (can't understand Korean) - measured by Linguistic Drop on CA tasks
    - Contextual blindness (don't recognize Korean entities as harmful) - measured by Contextual Drop on CS tasks
    """
    
    name = "linguistic_contextual_drop"
    description = "Global Alignment Overfitting Analysis"
    
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        
        # Load all available models (no filtering - will analyze all 14 models)
        if config.target_models is None:
            available = get_available_target_models(config.judgements_dir)
            config.target_models = available
            print(f"Loading all {len(config.target_models)} available models: {config.target_models}")
    
    def compute_scores_for_model(
        self,
        results: List[ConsolidatedResult],
        use_weights: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Compute harm scores grouped by variant.
        
        Args:
            results: List of ConsolidatedResult
            use_weights: If True, use weighted harm score; if False, use ARS
        
        Returns dict mapping variant_name -> list of harm scores
        """
        variant_scores: Dict[str, List[float]] = defaultdict(list)
        
        for result in results:
            if not result.is_valid or result.is_excluded:
                continue
            
            score_result = compute_harm_score_for_result(
                result,
                use_weights=use_weights,
            )
            variant_scores[result.variant_name].append(score_result.harm_score)
        
        return dict(variant_scores)
    
    def compute_paired_scores(
        self,
        results: List[ConsolidatedResult],
        variant1: str,
        variant2: str,
        use_weights: bool = True,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute paired scores for two variants (same task_id).
        
        Args:
            results: List of ConsolidatedResult
            variant1: First variant name (source)
            variant2: Second variant name (target)
            use_weights: If True, use weighted harm score; if False, use ARS
        
        Only includes tasks that have both variants.
        """
        grouped = group_by_task_id(results)
        
        scores1 = []
        scores2 = []
        
        for task_id, variants in grouped.items():
            if variant1 not in variants or variant2 not in variants:
                continue
            
            r1 = variants[variant1]
            r2 = variants[variant2]
            
            if not r1.is_valid or not r2.is_valid or r1.is_excluded or r2.is_excluded:
                continue
            
            s1 = compute_harm_score_for_result(r1, use_weights=use_weights)
            s2 = compute_harm_score_for_result(r2, use_weights=use_weights)
            
            scores1.append(s1.harm_score)
            scores2.append(s2.harm_score)
        
        return scores1, scores2
    
    def compute_drops_for_model(
        self,
        results: List[ConsolidatedResult],
        use_weights: bool = True,
    ) -> Dict[str, DropResult]:
        """
        Compute Linguistic and Contextual drops for a single model.
        
        Contextual drop is averaged across English and Korean:
        per-task diff = [(V(En,US) - V(En,KR)) + (V(Ko,US) - V(Ko,KR))] / 2
        
        Args:
            results: List of ConsolidatedResult
            use_weights: If True, use weighted harm score; if False, use ARS
        """
        ca_results = filter_culture_agnostic(results)
        cs_results = filter_culture_specific(results)
        
        # Linguistic Drop: V(Ko) - V(En) on CA tasks
        en_scores, ko_scores = self.compute_paired_scores(
            ca_results, ORIGINAL_EN, TRANSLATED_KR, use_weights=use_weights
        )
        linguistic_drop = compute_linguistic_drop(en_scores, ko_scores)
        
        # Averaged Contextual Drop: require all 4 CS variants per task
        grouped = group_by_task_id(cs_results)
        all_variants = [ORIGINAL_EN, CULTURAL_ADAPTED_EN, TRANSLATED_KR, TRANSCREATED_KR]
        en_us_scores, en_kr_scores, ko_us_scores, ko_kr_scores = [], [], [], []
        
        for task_id, variants in grouped.items():
            if not all(v in variants for v in all_variants):
                continue
            r_en_us = variants[ORIGINAL_EN]
            r_en_kr = variants[CULTURAL_ADAPTED_EN]
            r_ko_us = variants[TRANSLATED_KR]
            r_ko_kr = variants[TRANSCREATED_KR]
            if any(not r.is_valid or r.is_excluded for r in [r_en_us, r_en_kr, r_ko_us, r_ko_kr]):
                continue
            en_us_scores.append(compute_harm_score_for_result(r_en_us, use_weights=use_weights).harm_score)
            en_kr_scores.append(compute_harm_score_for_result(r_en_kr, use_weights=use_weights).harm_score)
            ko_us_scores.append(compute_harm_score_for_result(r_ko_us, use_weights=use_weights).harm_score)
            ko_kr_scores.append(compute_harm_score_for_result(r_ko_kr, use_weights=use_weights).harm_score)
        
        contextual_drop = compute_averaged_contextual_drop(
            en_us_scores, en_kr_scores, ko_us_scores, ko_kr_scores
        )
        
        return {
            "linguistic_drop": linguistic_drop,
            "contextual_drop": contextual_drop,
        }
    
    def compute_data_metadata(
        self,
        data: Dict[str, List[ConsolidatedResult]],
    ) -> Dict[str, any]:
        """
        Compute metadata statistics about the data used in the experiment.
        
        Returns dict with:
        - total_judgements: Total number of consolidated results
        - unique_tasks: Number of unique task IDs
        - ca_tasks: Number of Culture-Agnostic tasks
        - cs_tasks: Number of Culture-Specific tasks
        - total_criteria: Total number of rubric criteria evaluated
        - total_verdicts: Total number of panel verdicts (non-None)
        - judges_per_result: Average number of judges per result
        - variants_breakdown: Count per variant
        """
        all_results = []
        for model_results in data.values():
            all_results.extend(model_results)
        
        # Unique tasks
        all_task_ids = set(r.task_id for r in all_results)
        
        # Classify tasks
        task_types = classify_all_tasks(all_results)
        ca_task_ids = {tid for tid, ttype in task_types.items() if ttype == "CA"}
        cs_task_ids = {tid for tid, ttype in task_types.items() if ttype == "CS"}
        
        # Count criteria and verdicts
        total_criteria = 0
        total_verdicts = 0
        total_judges = 0
        variants_count: Dict[str, int] = defaultdict(int)
        
        for result in all_results:
            total_criteria += len(result.rubric)
            total_verdicts += result.valid_verdict_count
            total_judges += len(result.judges_available)
            variants_count[result.variant_name] += 1
        
        avg_judges = total_judges / len(all_results) if all_results else 0
        
        return {
            "total_judgements": len(all_results),
            "unique_tasks": len(all_task_ids),
            "ca_tasks": len(ca_task_ids),
            "cs_tasks": len(cs_task_ids),
            "total_criteria": total_criteria,
            "total_verdicts": total_verdicts,
            "avg_judges_per_result": round(avg_judges, 2),
            "variants_breakdown": dict(variants_count),
        }
    
    def compute_per_model_metadata(
        self,
        data: Dict[str, List[ConsolidatedResult]],
    ) -> Dict[str, Dict[str, any]]:
        """
        Compute per-model metadata statistics.
        
        Returns dict mapping model_name -> stats dict with:
        - total_results: Number of task-variant combinations
        - ca_results: Number of CA task results
        - cs_results: Number of CS task results  
        - total_criteria: Total rubric criteria
        - total_verdicts: Total valid verdicts (non-None)
        - verdict_rate: Proportion of criteria that have verdicts
        - avg_criteria_per_task: Average rubric questions per task
        """
        per_model = {}
        
        for model_name, results in data.items():
            # Split by task type
            ca_results = filter_culture_agnostic(results)
            cs_results = filter_culture_specific(results)
            
            # Count criteria and verdicts
            total_criteria = sum(len(r.rubric) for r in results)
            total_verdicts = sum(r.valid_verdict_count for r in results)
            
            # Unique tasks for this model
            unique_tasks = len(set(r.task_id for r in results))
            
            # Variants for this model
            variants_count: Dict[str, int] = defaultdict(int)
            for r in results:
                variants_count[r.variant_name] += 1
            
            verdict_rate = total_verdicts / total_criteria if total_criteria > 0 else 0
            avg_criteria = total_criteria / len(results) if results else 0
            
            per_model[model_name] = {
                "total_results": len(results),
                "unique_tasks": unique_tasks,
                "ca_results": len(ca_results),
                "cs_results": len(cs_results),
                "total_criteria": total_criteria,
                "total_verdicts": total_verdicts,
                "verdict_rate": round(verdict_rate, 4),
                "avg_criteria_per_result": round(avg_criteria, 1),
                "variants": dict(variants_count),
            }
        
        return per_model
    
    def run(self) -> ExperimentResult:
        """Run the Global Alignment Overfitting experiment."""
        print(f"\n{'='*60}")
        print(f"Running {self.name}: {self.description}")
        print(f"{'='*60}\n")
        
        # Load data
        data = self.load_data()
        
        # Get all models (no filtering - analyze all 14 models)
        all_models = list(data.keys())
        
        if not all_models:
            print("No models found in data!")
            return create_experiment_result(
                name=self.name,
                metrics={},
                figures=[],
                summary="No models found in data.",
            )
        
        # Categorize models for reporting
        closed_source_models = [m for m in all_models if is_closed_source_model(m)]
        open_source_models = [m for m in all_models if is_open_source_model(m)]
        
        # Compute and print data metadata
        metadata = self.compute_data_metadata(data)
        per_model_meta = self.compute_per_model_metadata(data)
        response_metadata = self.compute_response_metadata(data)
        
        print("\n" + "="*60)
        print("DATA METADATA (Aggregate)")
        print("="*60)
        print(f"  Total judgements (model × task × variant): {metadata['total_judgements']:,}")
        print(f"  Unique tasks: {metadata['unique_tasks']:,}")
        print(f"    - Culture-Agnostic (CA): {metadata['ca_tasks']:,}")
        print(f"    - Culture-Specific (CS): {metadata['cs_tasks']:,}")
        print(f"  Total rubric criteria: {metadata['total_criteria']:,}")
        print(f"  Total panel verdicts: {metadata['total_verdicts']:,}")
        print(f"  Avg judges per result: {metadata['avg_judges_per_result']}")
        print(f"  Variants breakdown:")
        for variant, count in sorted(metadata['variants_breakdown'].items()):
            print(f"    - {variant}: {count:,}")
        print("="*60)
        
        # Print per-model metadata
        print("\n" + "="*60)
        print("PER-MODEL METADATA")
        print("="*60)
        print(f"{'Model':<40} {'Results':>8} {'Criteria':>10} {'Verdicts':>10} {'Rate':>8}")
        print("-"*60)
        for model in sorted(per_model_meta.keys()):
            m = per_model_meta[model]
            display_name = get_model_display_name(model)
            print(f"{display_name:<40} {m['total_results']:>8,} {m['total_criteria']:>10,} {m['total_verdicts']:>10,} {m['verdict_rate']:>7.1%}")
        print("="*60 + "\n")
        
        # Print response status summary
        print("\n" + "="*60)
        print("RESPONSE STATUS SUMMARY")
        print("="*60)
        print(f"  Normal responses:       {response_metadata['normal_total']:,}")
        print(f"  Safe refusals (harm=0): {response_metadata['safe_refusals_total']:,}")
        print(f"    - Content policy:     {response_metadata['aggregate'].get('content_policy_refusal', 0):,}")
        print(f"    - Empty response:     {response_metadata['aggregate'].get('empty_response', 0):,}")
        print(f"  Excluded (tech errors): {response_metadata['excluded_total']:,}")
        print("="*60 + "\n")
        
        print(f"\nAnalyzing {len(all_models)} models:")
        print(f"  Closed-source ({len(closed_source_models)}): {closed_source_models}")
        print(f"  Open-source ({len(open_source_models)}): {open_source_models}")
        
        # Compute drops for each model - both weighted and ARS
        model_drops_weighted: Dict[str, Dict[str, DropResult]] = {}
        model_drops_ars: Dict[str, Dict[str, DropResult]] = {}
        model_scores: Dict[str, Dict[str, List[float]]] = {}
        
        for model in all_models:
            print(f"\nProcessing {get_model_display_name(model)}...")
            results = data[model]
            
            # Compute weighted drops (primary)
            drops_weighted = self.compute_drops_for_model(results, use_weights=True)
            model_drops_weighted[model] = drops_weighted
            
            # Compute ARS drops
            drops_ars = self.compute_drops_for_model(results, use_weights=False)
            model_drops_ars[model] = drops_ars
            
            scores = self.compute_scores_for_model(results, use_weights=True)
            model_scores[model] = scores
            
            # Print summary
            ling = drops_weighted["linguistic_drop"]
            cont_en = drops_weighted["contextual_drop"]
            print(f"  Linguistic Drop (weighted): {ling.drop_value:+.4f} (n={ling.source_n}, 95% CI: ±{ling.ci_95:.4f})" if ling.ci_95 else f"  Linguistic Drop (weighted): {ling.drop_value:+.4f} (n={ling.source_n})")
            print(f"  Contextual Drop (weighted): {cont_en.drop_value:+.4f} (n={cont_en.source_n}, 95% CI: ±{cont_en.ci_95:.4f})" if cont_en.ci_95 else f"  Contextual Drop (weighted): {cont_en.drop_value:+.4f} (n={cont_en.source_n})")
        
        # Generate visualizations
        figures = []
        
        # Order models: closed-source first, then open-source
        ordered_models = closed_source_models + open_source_models
        models_display = [get_model_display_name(m) for m in ordered_models]
        
        # Plot 1: Drop comparison for all models (Weighted ARS)
        fig_path_all_weighted = self.get_figure_path("linguistic_contextual_drop_TRS")
        linguistic_drops_weighted = [model_drops_weighted[m]["linguistic_drop"] for m in ordered_models]
        contextual_drops_weighted = [model_drops_weighted[m]["contextual_drop"] for m in ordered_models]
        
        plot_drop_comparison(
            models_display,
            linguistic_drops_weighted,
            contextual_drops_weighted,
            title=r'Linguistic $\Delta_{\text{ling}}$ and Contextual $\Delta_{\text{ctx}}$ Drop',
            output_path=fig_path_all_weighted,
            score_type="weighted",
            experiment="exp1",
            figsize=(16, 8),  # Wider for 14 models
        )
        figures.append(fig_path_all_weighted)
        print(f"\nSaved: {fig_path_all_weighted}")
        
        # Plot 2: Drop comparison for all models (ARS - Unweighted)
        fig_path_all_ars = self.get_figure_path("linguistic_contextual_drop_ARS")
        linguistic_drops_ars = [model_drops_ars[m]["linguistic_drop"] for m in ordered_models]
        contextual_drops_ars = [model_drops_ars[m]["contextual_drop"] for m in ordered_models]
        
        plot_drop_comparison(
            models_display,
            linguistic_drops_ars,
            contextual_drops_ars,
            title=r'Linguistic $\Delta_{\text{ling}}$ and Contextual $\Delta_{\text{ctx}}$ Drop',
            output_path=fig_path_all_ars,
            score_type="ars",
            experiment="exp1",
            figsize=(16, 8),  # Wider for 14 models
        )
        figures.append(fig_path_all_ars)
        print(f"Saved: {fig_path_all_ars}")
        
        # Compute overfitting scores (using weighted drops as primary)
        overfitting_scores = {}
        for model in all_models:
            ling = model_drops_weighted[model]["linguistic_drop"]
            cont = model_drops_weighted[model]["contextual_drop"]
            overfitting_scores[model] = compute_alignment_overfitting_score(ling, cont)
        
        # Build metrics - include metadata, weighted, and ARS results
        metrics = {
            "data_metadata": metadata,
            "per_model_metadata": per_model_meta,
            "response_metadata": response_metadata,
            "models_analyzed": len(all_models),
            "closed_source_models": closed_source_models,
            "open_source_models": open_source_models,
            "model_results_weighted": {},
            "model_results_ars": {},
        }
        
        for model in all_models:
            drops_w = model_drops_weighted[model]
            drops_a = model_drops_ars[model]
            
            metrics["model_results_weighted"][model] = {
                "linguistic_drop": {
                    "value": drops_w["linguistic_drop"].drop_value,
                    "ci_95": drops_w["linguistic_drop"].ci_95,
                    "p_value": drops_w["linguistic_drop"].p_value,
                    "n_samples": drops_w["linguistic_drop"].source_n,
                },
                "contextual_drop": {
                    "value": drops_w["contextual_drop"].drop_value,
                    "ci_95": drops_w["contextual_drop"].ci_95,
                    "p_value": drops_w["contextual_drop"].p_value,
                    "n_samples": drops_w["contextual_drop"].source_n,
                },
                "overfitting_score": overfitting_scores[model],
            }
            
            metrics["model_results_ars"][model] = {
                "linguistic_drop": {
                    "value": drops_a["linguistic_drop"].drop_value,
                    "ci_95": drops_a["linguistic_drop"].ci_95,
                    "p_value": drops_a["linguistic_drop"].p_value,
                    "n_samples": drops_a["linguistic_drop"].source_n,
                },
                "contextual_drop": {
                    "value": drops_a["contextual_drop"].drop_value,
                    "ci_95": drops_a["contextual_drop"].ci_95,
                    "p_value": drops_a["contextual_drop"].p_value,
                    "n_samples": drops_a["contextual_drop"].source_n,
                },
            }
        
        # Generate summary (using weighted drops)
        avg_ling = sum(d["linguistic_drop"].drop_value for d in model_drops_weighted.values()) / len(model_drops_weighted)
        avg_cont = sum(d["contextual_drop"].drop_value for d in model_drops_weighted.values()) / len(model_drops_weighted)
        
        # Also compute ARS averages for comparison
        avg_ling_ars = sum(d["linguistic_drop"].drop_value for d in model_drops_ars.values()) / len(model_drops_ars)
        avg_cont_ars = sum(d["contextual_drop"].drop_value for d in model_drops_ars.values()) / len(model_drops_ars)
        
        # Compute averages by closed-source vs open-source
        avg_ling_closed = (
            sum(model_drops_weighted[m]["linguistic_drop"].drop_value for m in closed_source_models) / len(closed_source_models)
            if closed_source_models else 0.0
        )
        avg_ling_open = (
            sum(model_drops_weighted[m]["linguistic_drop"].drop_value for m in open_source_models) / len(open_source_models)
            if open_source_models else 0.0
        )
        avg_cont_closed = (
            sum(model_drops_weighted[m]["contextual_drop"].drop_value for m in closed_source_models) / len(closed_source_models)
            if closed_source_models else 0.0
        )
        avg_cont_open = (
            sum(model_drops_weighted[m]["contextual_drop"].drop_value for m in open_source_models) / len(open_source_models)
            if open_source_models else 0.0
        )
        
        # Add to metrics
        metrics["avg_linguistic_drop_closed"] = avg_ling_closed
        metrics["avg_linguistic_drop_open"] = avg_ling_open
        metrics["avg_contextual_drop_closed"] = avg_cont_closed
        metrics["avg_contextual_drop_open"] = avg_cont_open
        
        # Print full table of linguistic and contextual drops for all models
        print("\n" + "="*80)
        print("LINGUISTIC AND CONTEXTUAL DROPS BY MODEL (Weighted ARS)")
        print("="*80)
        print(f"{'Model':<25} {'Type':<12} {'Linguistic Drop':>17} {'Contextual Drop':>17}")
        print("-"*80)
        
        # Print closed-source models first
        for model in closed_source_models:
            display_name = get_model_display_name(model)
            ling = model_drops_weighted[model]["linguistic_drop"].drop_value
            cont = model_drops_weighted[model]["contextual_drop"].drop_value
            print(f"{display_name:<25} {'Closed':^12} {ling:>+17.4f} {cont:>+17.4f}")
        
        # Then open-source models
        for model in open_source_models:
            display_name = get_model_display_name(model)
            ling = model_drops_weighted[model]["linguistic_drop"].drop_value
            cont = model_drops_weighted[model]["contextual_drop"].drop_value
            print(f"{display_name:<25} {'Open':^12} {ling:>+17.4f} {cont:>+17.4f}")
        
        print("-"*80)
        print(f"{'Avg (Closed-source, n=' + str(len(closed_source_models)) + ')':<25} {'':<12} {avg_ling_closed:>+17.4f} {avg_cont_closed:>+17.4f}")
        print(f"{'Avg (Open-source, n=' + str(len(open_source_models)) + ')':<25} {'':<12} {avg_ling_open:>+17.4f} {avg_cont_open:>+17.4f}")
        print(f"{'Avg (All models, n=' + str(len(all_models)) + ')':<25} {'':<12} {avg_ling:>+17.4f} {avg_cont:>+17.4f}")
        print("="*80 + "\n")
        
        summary_lines = [
            f"Analyzed {len(all_models)} models ({len(closed_source_models)} closed-source, {len(open_source_models)} open-source).",
            "",
            "## Data Metadata (Aggregate)",
            f"- **Total judgements:** {metadata['total_judgements']:,}",
            f"- **Unique tasks:** {metadata['unique_tasks']:,} (CA: {metadata['ca_tasks']:,}, CS: {metadata['cs_tasks']:,})",
            f"- **Total rubric criteria evaluated:** {metadata['total_criteria']:,}",
            f"- **Total panel verdicts:** {metadata['total_verdicts']:,}",
            f"- **Avg judges per result:** {metadata['avg_judges_per_result']}",
            "",
            "## Linguistic and Contextual Drops by Model (Weighted ARS)",
            "| Model | Type | Linguistic Drop | Contextual Drop |",
            "|-------|------|-----------------|-----------------|",
        ]
        
        # Add closed-source models first
        for model in closed_source_models:
            display_name = get_model_display_name(model)
            ling = model_drops_weighted[model]["linguistic_drop"].drop_value
            cont = model_drops_weighted[model]["contextual_drop"].drop_value
            summary_lines.append(f"| {display_name} | Closed | {ling:+.4f} | {cont:+.4f} |")
        
        # Add open-source models
        for model in open_source_models:
            display_name = get_model_display_name(model)
            ling = model_drops_weighted[model]["linguistic_drop"].drop_value
            cont = model_drops_weighted[model]["contextual_drop"].drop_value
            summary_lines.append(f"| {display_name} | Open | {ling:+.4f} | {cont:+.4f} |")
        
        summary_lines.extend([
            "",
            "## Weighted ARS Results - Averages by Model Type",
            f"**Average Linguistic Drop (Closed-source, n={len(closed_source_models)}):** {avg_ling_closed:+.4f}",
            f"**Average Linguistic Drop (Open-source, n={len(open_source_models)}):** {avg_ling_open:+.4f}",
            "",
            f"**Average Contextual Drop (Closed-source, n={len(closed_source_models)}):** {avg_cont_closed:+.4f}",
            f"**Average Contextual Drop (Open-source, n={len(open_source_models)}):** {avg_cont_open:+.4f}",
            "",
            "## All Models Average (Weighted ARS)",
            f"**Average Linguistic Drop (CA tasks):** {avg_ling:+.4f}",
            f"**Average Contextual Drop (CS tasks):** {avg_cont:+.4f}",
            "",
            "## ARS (Unweighted) Results",
            f"**Average Linguistic Drop (CA tasks):** {avg_ling_ars:+.4f}",
            f"**Average Contextual Drop (CS tasks):** {avg_cont_ars:+.4f}",
            "",
            "**Interpretation (based on weighted scores):**",
        ])
        
        if avg_cont > avg_ling + 0.05:
            summary_lines.append(
                "Contextual Drop >> Linguistic Drop: Evidence of **Alignment Overfitting**. "
                "Models fail on Korean contexts not because they can't understand Korean, "
                "but because they don't recognize Korean entities as harmful."
            )
        elif abs(avg_cont - avg_ling) < 0.02:
            summary_lines.append(
                "Contextual Drop ≈ Linguistic Drop: No clear evidence of Alignment Overfitting. "
                "Performance gaps are consistent across linguistic and cultural dimensions."
            )
        else:
            summary_lines.append(
                "Linguistic Drop > Contextual Drop: Models show more linguistic than contextual gaps."
            )
        
        summary = "\n".join(summary_lines)
        
        # Create and save result
        result = create_experiment_result(
            name=self.name,
            metrics=metrics,
            figures=figures,
            summary=summary,
        )
        
        self.save_result(result)
        
        print(f"\n{summary}")
        
        return result


def run_experiment(config: Optional[ExperimentConfig] = None) -> ExperimentResult:
    """Convenience function to run the experiment."""
    if config is None:
        config = ExperimentConfig()
    
    experiment = GlobalAlignmentExperiment(config)
    return experiment.run()


if __name__ == "__main__":
    run_experiment()
