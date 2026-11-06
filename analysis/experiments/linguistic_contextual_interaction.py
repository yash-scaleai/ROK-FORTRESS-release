"""
Interaction Analysis: Language × Context Effects

This analysis computes the full 2×2 interaction matrix to properly test
whether Korean context offsets or amplifies language-driven suppression.

Convention: Positive drop = Korean/KR-context is more conservative (lower harm scores)

We compare:
1. Linguistic effect in US context: V(En,US) − V(Ko,US) 
2. Linguistic effect in KR context: V(En,KR) − V(Ko,KR)

Positive Ling = Korean is more conservative (suppression)

If context "mitigates" language suppression:
Ling(KR) < Ling(US) → suppression is smaller when Korean context is added

We also compute:
3. Contextual effect in English: V(En,US) − V(En,KR)
4. Contextual effect in Korean: V(Ko,US) − V(Ko,KR)

And the interaction term:
- Interaction = Ling(US) - Ling(KR)
  > 0 means context MITIGATES suppression (Korean conservatism is smaller in KR context)
  < 0 means context AMPLIFIES suppression (Korean conservatism is larger in KR context)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .base import BaseExperiment, ExperimentConfig, ExperimentResult, create_experiment_result
from .exp_data.loaders import ConsolidatedResult
from .exp_data.filters import filter_culture_specific
from .exp_data.variants import (
    ORIGINAL_EN,
    TRANSLATED_KR,
    CULTURAL_ADAPTED_EN,
    TRANSCREATED_KR,
    group_by_task_id,
)
from .exp_data.models import get_model_display_name, GLOBAL_MODELS
from .exp_data.loaders import get_available_target_models
from .metrics.harm_scores import compute_harm_score_for_result
from .metrics.alignment_metrics import compute_drop, DropResult


def bonferroni_correction(pvalues: List[float], alpha: float = 0.05) -> List[bool]:
    """Bonferroni family-wise error correction.

    Returns a boolean list (aligned to the input order) marking which
    hypotheses are rejected at family-wise error rate ``alpha``. This is the
    most conservative standard correction (reject if ``p <= alpha / n``) and
    gives the strongest family-wise guarantee.
    """
    n = len(pvalues)
    if n == 0:
        return []
    threshold = alpha / n
    return [p <= threshold for p in pvalues]


@dataclass
class InteractionResult:
    """Results of the 2×2 interaction analysis for one model.
    
    Convention: Positive drop = Korean/KR-context is more conservative.
    """
    
    # Linguistic drops (language effect) - positive = Korean more conservative
    linguistic_drop_us_context: DropResult  # V(En,US) - V(Ko,US) on CS tasks
    linguistic_drop_kr_context: DropResult  # V(En,KR) - V(Ko,KR) on CS tasks
    
    # Contextual drops (context effect) - positive = KR context more conservative
    contextual_drop_english: DropResult     # V(En,US) - V(En,KR) on CS tasks
    contextual_drop_korean: DropResult      # V(Ko,US) - V(Ko,KR) on CS tasks
    
    # Bootstrapped CI for interaction term (optional)
    interaction_ci_95: Optional[Tuple[float, float]] = None  # (lower, upper)
    interaction_is_significant: bool = False  # True if CI excludes zero
    interaction_p_value: Optional[float] = None  # two-sided bootstrap p-value
    
    @property
    def interaction_term(self) -> float:
        """
        Interaction = Ling(US) - Ling(KR)
        
        Positive: Suppression is larger in US context → Context MITIGATES
        Negative: Suppression is larger in KR context → Context AMPLIFIES
        Near zero: No interaction (effects are additive)
        """
        return self.linguistic_drop_us_context.drop_value - self.linguistic_drop_kr_context.drop_value
    
    @property
    def interaction_ci_half_width(self) -> float:
        """Return half-width of CI for error bar plotting."""
        if self.interaction_ci_95 is None:
            return 0.0
        return (self.interaction_ci_95[1] - self.interaction_ci_95[0]) / 2
    
    @property
    def model_type(self) -> str:
        """
        Classify the model based on the interaction pattern.
        
        Type I: Stable (both linguistic effects ~0, no suppression)
        Type II: Context Mitigates (interaction > 0, suppression reduced by KR context)
        Type III: Context Amplifies (interaction < 0, suppression increased by KR context)
        
        Uses bootstrapped CI for significance if available, otherwise falls back to heuristic.
        """
        ling_us = self.linguistic_drop_us_context.drop_value
        ling_kr = self.linguistic_drop_kr_context.drop_value
        interaction = self.interaction_term
        
        # Check for "stable" (Type I) - both linguistic drops near zero
        if abs(ling_us) < 0.03 and abs(ling_kr) < 0.03:
            return "Type I: Stable"
        
        # Use bootstrapped significance if available
        if self.interaction_ci_95 is not None:
            if self.interaction_is_significant:
                if interaction > 0:
                    return "Type II: Context Mitigates"
                else:
                    return "Type III: Context Amplifies"
            else:
                return "Additive (No Interaction)"
        
        # Fallback: heuristic threshold
        if interaction > 0.02:  # Context mitigates suppression
            return "Type II: Context Mitigates"
        elif interaction < -0.02:  # Context amplifies suppression
            return "Type III: Context Amplifies"
        else:
            return "Additive (No Interaction)"


class InteractionAnalysis(BaseExperiment):
    """
    Interaction Analysis: Language × Context Effects.
    
    Computes the full 2×2 matrix of effects on Culture-Specific tasks:
    
    |                  | US Context (original) | KR Context (adapted) |
    |------------------|----------------------|---------------------|
    | English          | V(En,US)             | V(En,KR)            |
    | Korean           | V(Ko,US)             | V(Ko,KR)            |
    
    Convention: Positive = more conservative (lower harm scores)
    
    Effects:
    - Linguistic (US): V(En,US) - V(Ko,US) = Language suppression in US context
    - Linguistic (KR): V(En,KR) - V(Ko,KR) = Language suppression in KR context
    - Contextual (EN): V(En,US) - V(En,KR) = Context suppression in English
    - Contextual (KO): V(Ko,US) - V(Ko,KR) = Context suppression in Korean
    - Interaction: Ling(US) - Ling(KR) = How context moderates language suppression
      - Positive: Context mitigates (suppression smaller in KR context)
      - Negative: Context amplifies (suppression larger in KR context)
    """
    
    name = "linguistic_contextual_interaction"
    description = "Language × Context Interaction Analysis"
    
    def compute_paired_scores_cs(
        self,
        results: List[ConsolidatedResult],
        variant1: str,
        variant2: str,
        use_weights: bool = True,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute paired scores for two variants on CS tasks only.
        
        Only includes tasks that have both variants.
        """
        # Filter to CS tasks only
        cs_results = filter_culture_specific(results)
        grouped = group_by_task_id(cs_results)
        
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
    
    def compute_interaction_for_model(
        self,
        results: List[ConsolidatedResult],
        use_weights: bool = True,
    ) -> InteractionResult:
        """
        Compute the full 2×2 interaction matrix for one model.
        
        Convention: Positive = Korean has higher harm scores (less conservative)
        Linguistic Drop = V(Ko) - V(En), so positive means Korean is less safe
        """
        # Linguistic Drop in US Context: V(Ko,US) - V(En,US)
        # compute_drop(source, target) = source - target, so source=ko, target=en
        en_us_scores, ko_us_scores = self.compute_paired_scores_cs(
            results, ORIGINAL_EN, TRANSLATED_KR, use_weights=use_weights
        )
        linguistic_drop_us = compute_drop(ko_us_scores, en_us_scores, paired=True)
        
        # Linguistic Drop in KR Context: V(Ko,KR) - V(En,KR)
        en_kr_scores, ko_kr_scores = self.compute_paired_scores_cs(
            results, CULTURAL_ADAPTED_EN, TRANSCREATED_KR, use_weights=use_weights
        )
        linguistic_drop_kr = compute_drop(ko_kr_scores, en_kr_scores, paired=True)
        
        # Contextual Drop in English: V(En,US) - V(En,KR)
        en_us_scores_ctx, en_kr_scores_ctx = self.compute_paired_scores_cs(
            results, ORIGINAL_EN, CULTURAL_ADAPTED_EN, use_weights=use_weights
        )
        contextual_drop_en = compute_drop(en_us_scores_ctx, en_kr_scores_ctx, paired=True)
        
        # Contextual Drop in Korean: V(Ko,US) - V(Ko,KR)
        ko_us_scores_ctx, ko_kr_scores_ctx = self.compute_paired_scores_cs(
            results, TRANSLATED_KR, TRANSCREATED_KR, use_weights=use_weights
        )
        contextual_drop_ko = compute_drop(ko_us_scores_ctx, ko_kr_scores_ctx, paired=True)
        
        return InteractionResult(
            linguistic_drop_us_context=linguistic_drop_us,
            linguistic_drop_kr_context=linguistic_drop_kr,
            contextual_drop_english=contextual_drop_en,
            contextual_drop_korean=contextual_drop_ko,
        )
    
    def bootstrap_interaction_ci(
        self,
        results: List[ConsolidatedResult],
        use_weights: bool = True,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        random_seed: int = 42,
    ) -> Tuple[Tuple[float, float], bool, float]:
        """
        Bootstrap confidence interval for the interaction term.
        
        The interaction term is: Ling(US) - Ling(KR)
        where Ling(X) = V(En,X) - V(Ko,X)
        
        We resample paired observations (tasks) and recompute the interaction
        for each bootstrap sample.
        
        Args:
            results: ConsolidatedResult list for one model
            use_weights: Use weighted harm scores if True
            n_bootstrap: Number of bootstrap iterations
            confidence: Confidence level (default 0.95 for 95% CI)
            random_seed: Random seed for reproducibility
        
        Returns:
            Tuple of ((ci_lower, ci_upper), is_significant, p_value)
            where is_significant is True if CI excludes zero and p_value is the
            two-sided bootstrap p-value for the interaction term differing from 0.
        """
        np.random.seed(random_seed)
        
        # Get CS results and group by task
        cs_results = filter_culture_specific(results)
        grouped = group_by_task_id(cs_results)
        
        # Collect tasks that have all 4 variants
        complete_tasks = []
        for task_id, variants in grouped.items():
            if all(v in variants for v in [ORIGINAL_EN, TRANSLATED_KR, CULTURAL_ADAPTED_EN, TRANSCREATED_KR]):
                r_en_us = variants[ORIGINAL_EN]
                r_ko_us = variants[TRANSLATED_KR]
                r_en_kr = variants[CULTURAL_ADAPTED_EN]
                r_ko_kr = variants[TRANSCREATED_KR]
                
                # Check all are valid and not excluded
                if all(r.is_valid and not r.is_excluded for r in [r_en_us, r_ko_us, r_en_kr, r_ko_kr]):
                    # Compute scores for each cell
                    s_en_us = compute_harm_score_for_result(r_en_us, use_weights=use_weights).harm_score
                    s_ko_us = compute_harm_score_for_result(r_ko_us, use_weights=use_weights).harm_score
                    s_en_kr = compute_harm_score_for_result(r_en_kr, use_weights=use_weights).harm_score
                    s_ko_kr = compute_harm_score_for_result(r_ko_kr, use_weights=use_weights).harm_score
                    complete_tasks.append((s_en_us, s_ko_us, s_en_kr, s_ko_kr))
        
        if len(complete_tasks) < 10:
            print(f"Warning: Only {len(complete_tasks)} complete tasks for bootstrap, CI may be unreliable")
            if len(complete_tasks) < 2:
                return (0.0, 0.0), False, 1.0
        
        n_tasks = len(complete_tasks)
        task_array = np.array(complete_tasks)  # Shape: (n_tasks, 4)
        
        # Bootstrap
        bootstrap_interactions = []
        for _ in range(n_bootstrap):
            # Resample tasks with replacement
            indices = np.random.choice(n_tasks, size=n_tasks, replace=True)
            sample = task_array[indices]
            
            # Compute means for each cell
            mean_en_us = sample[:, 0].mean()
            mean_ko_us = sample[:, 1].mean()
            mean_en_kr = sample[:, 2].mean()
            mean_ko_kr = sample[:, 3].mean()
            
            # Ling(US) = V(Ko,US) - V(En,US) (flipped: positive = Korean less conservative)
            ling_us = mean_ko_us - mean_en_us
            # Ling(KR) = V(Ko,KR) - V(En,KR)
            ling_kr = mean_ko_kr - mean_en_kr
            
            # Interaction = Ling(US) - Ling(KR)
            interaction = ling_us - ling_kr
            bootstrap_interactions.append(interaction)
        
        bootstrap_interactions = np.array(bootstrap_interactions)
        
        # Compute percentile CI
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_interactions, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_interactions, 100 * (1 - alpha / 2))
        
        # Significant if CI excludes zero
        is_significant = (ci_lower > 0) or (ci_upper < 0)

        # Two-sided bootstrap p-value: 2 * min(fraction <= 0, fraction >= 0).
        # Floored at 1/n_bootstrap so a value of exactly 0 is reported as the
        # resolution limit rather than an impossible p=0.
        frac_le_0 = float(np.mean(bootstrap_interactions <= 0))
        frac_ge_0 = float(np.mean(bootstrap_interactions >= 0))
        p_value = min(1.0, 2.0 * min(frac_le_0, frac_ge_0))
        p_value = max(p_value, 1.0 / n_bootstrap)

        return (ci_lower, ci_upper), is_significant, p_value
    
    def compute_interaction_with_ci(
        self,
        results: List[ConsolidatedResult],
        use_weights: bool = True,
        n_bootstrap: int = 1000,
    ) -> InteractionResult:
        """
        Compute interaction result with bootstrapped CI for the interaction term.
        
        This is the preferred method as it provides proper significance testing.
        """
        # Get base interaction result
        base_result = self.compute_interaction_for_model(results, use_weights=use_weights)
        
        # Bootstrap CI + p-value for interaction
        ci, is_significant, p_value = self.bootstrap_interaction_ci(
            results, use_weights=use_weights, n_bootstrap=n_bootstrap
        )
        
        # Return updated result with CI
        return InteractionResult(
            linguistic_drop_us_context=base_result.linguistic_drop_us_context,
            linguistic_drop_kr_context=base_result.linguistic_drop_kr_context,
            contextual_drop_english=base_result.contextual_drop_english,
            contextual_drop_korean=base_result.contextual_drop_korean,
            interaction_ci_95=ci,
            interaction_is_significant=is_significant,
            interaction_p_value=p_value,
        )
    
    def plot_interaction_comparison(
        self,
        models: List[str],
        interactions: Dict[str, InteractionResult],
        output_path: Path,
        use_weights: bool = True,
    ) -> plt.Figure:
        """
        Plot comparison of linguistic drops across models with interaction effect.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Enable LaTeX-style rendering for math
        plt.rcParams['mathtext.fontset'] = 'stix'
        
        # Panel 1: Linguistic Drop in US vs KR Context
        ax1 = axes[0]
        x = np.arange(len(models))
        width = 0.35
        
        # Scale values by 100 for display (already computed as Ko - En, positive = Korean less conservative)
        ling_us_values = [interactions[m].linguistic_drop_us_context.drop_value * 100 for m in models]
        ling_kr_values = [interactions[m].linguistic_drop_kr_context.drop_value * 100 for m in models]
        ling_us_ci = [(interactions[m].linguistic_drop_us_context.ci_95 or 0) * 100 for m in models]
        ling_kr_ci = [(interactions[m].linguistic_drop_kr_context.ci_95 or 0) * 100 for m in models]
        
        bars1 = ax1.bar(
            x - width/2, ling_us_values, width,
            label=r"$\Delta_{\mathrm{ling}}$ (US Context)",
            color="steelblue",
            yerr=ling_us_ci,
            capsize=3,
            alpha=0.8,
        )
        bars2 = ax1.bar(
            x + width/2, ling_kr_values, width,
            label=r"$\Delta_{\mathrm{ling}}$ (KR Context)",
            color="coral",
            yerr=ling_kr_ci,
            capsize=3,
            alpha=0.8,
        )
        
        # Add significance markers
        for i, (int_us, int_kr) in enumerate(zip(
            [interactions[m].linguistic_drop_us_context for m in models],
            [interactions[m].linguistic_drop_kr_context for m in models]
        )):
            if int_us.is_significant:
                y_pos = ling_us_values[i] + ling_us_ci[i] + 2
                ax1.annotate("*", xy=(x[i] - width/2, y_pos), ha="center", fontsize=14, color="steelblue")
            if int_kr.is_significant:
                y_pos = ling_kr_values[i] + ling_kr_ci[i] + 2
                ax1.annotate("*", xy=(x[i] + width/2, y_pos), ha="center", fontsize=14, color="coral")
        
        ax1.set_ylabel("Drop value")
        ax1.set_xlabel("Model")
        ax1.set_title(r"Linguistic Drop $\Delta_{\mathrm{ling}}$ by Cultural Context")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha="right")
        ax1.legend(loc="upper left")
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax1.text(
            0.02, 0.02,
            "Ling(US) > Ling(KR) → Context mitigates (Type II)\n"
            "Ling(US) < Ling(KR) → Context amplifies (Type III)\n"
            "* = p < 0.05",
            transform=ax1.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
        )
        
        # Panel 2: Interaction Term with Bootstrapped CI
        ax2 = axes[1]
        # Scale interaction values by 100 for display (already computed as Ling(US) - Ling(KR))
        interaction_values = [interactions[m].interaction_term * 100 for m in models]
        
        # Get error bars from bootstrapped CI (if available), scaled by 100
        interaction_ci = [interactions[m].interaction_ci_half_width * 100 for m in models]
        has_ci = any(c > 0 for c in interaction_ci)
        
        # Determine significance based on CI (if available) or classification
        is_significant = []
        for m in models:
            if interactions[m].interaction_ci_95 is not None:
                is_significant.append(interactions[m].interaction_is_significant)
            else:
                # Fallback to heuristic
                is_significant.append(abs(interactions[m].interaction_term) > 0.02)
        
        # Color by classification
        colors = []
        for m in models:
            mtype = interactions[m].model_type
            if "Stable" in mtype:
                colors.append("green")
            elif "Mitigates" in mtype:
                colors.append("steelblue")
            elif "Amplifies" in mtype:
                colors.append("coral")
            else:
                colors.append("gray")
        
        # Plot horizontal bars with error bars
        y_pos = np.arange(len(models))
        bars = ax2.barh(
            y_pos, interaction_values, 
            xerr=interaction_ci if has_ci else None,
            color=colors, alpha=0.8, zorder=2,
            capsize=3,
        )
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(models)
        ax2.set_xlabel(r"Interaction Term = Ling(US) - Ling(KR)")
        ax2.set_title(r"Interaction Effect: Cultural Context's Moderation of Linguistic Drop $\Delta_{\mathrm{ling}}$")
        ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5, zorder=1)
        
        # Add significance markers
        for i, (m, val, sig) in enumerate(zip(models, interaction_values, is_significant)):
            if sig:
                # Position the asterisk at the end of the bar (plus CI if available)
                offset = (interaction_ci[i] + 1) if has_ci else 1
                x_pos = val + offset if val >= 0 else val - offset
                ax2.annotate("*", xy=(x_pos, i), ha="left" if val >= 0 else "right", 
                            va="center", fontsize=14, fontweight="bold", color=colors[i])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="green", alpha=0.8, label="Type I: Stable (both ~0)"),
            Patch(facecolor="steelblue", alpha=0.8, label="Type II: Context Mitigates (+)"),
            Patch(facecolor="coral", alpha=0.8, label="Type III: Context Amplifies (-)"),
            Patch(facecolor="gray", alpha=0.8, label="Additive (CI includes 0)"),
        ]
        if has_ci:
            legend_elements.append(
                plt.Line2D([0], [0], color='black', linestyle='-', marker='|', 
                           markersize=10, label="95% CI (bootstrap)")
            )
        ax2.legend(handles=legend_elements, loc="lower right", fontsize=8)
        
        plt.tight_layout()
        fig.suptitle("Language x Cultural Context Interaction Analysis", fontsize=14, y=1.02)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        return fig
    
    def run(self, include_regional: bool = False) -> ExperimentResult:
        """Run the interaction analysis.
        
        Args:
            include_regional: If True, include regional Korean models. 
                              If False (default), only analyze global models.
        """
        print(f"\n{'='*60}")
        print(f"Running {self.name}: {self.description}")
        print(f"{'='*60}\n")
        
        # Load data
        data = self.load_data()
        
        # Filter models based on include_regional flag
        from .exp_data.models import is_global_model, is_regional_model
        
        if include_regional:
            # Include all models (global + regional)
            target_models = [m for m in data.keys() if is_global_model(m) or is_regional_model(m)]
            model_scope = "all (global + regional)"
        else:
            # Global models only
            target_models = [m for m in data.keys() if is_global_model(m)]
            model_scope = "global"
        
        if not target_models:
            print(f"No {model_scope} models found!")
            return create_experiment_result(
                name=self.name,
                metrics={},
                figures=[],
                summary=f"No {model_scope} models found in data.",
            )
        
        print(f"Analyzing {len(target_models)} {model_scope} models: {target_models}")
        
        # Compute interactions for each model
        model_interactions: Dict[str, InteractionResult] = {}
        
        for model in target_models:
            display_name = get_model_display_name(model)
            print(f"\nProcessing {display_name}...")
            results = data[model]
            
            # Compute interaction with bootstrapped CI (weighted)
            interaction = self.compute_interaction_with_ci(results, use_weights=True, n_bootstrap=1000)
            model_interactions[model] = interaction
            
            # Print summary with CI
            print(f"  Linguistic Drop (US context): {interaction.linguistic_drop_us_context.drop_value:+.4f}")
            print(f"  Linguistic Drop (KR context): {interaction.linguistic_drop_kr_context.drop_value:+.4f}")
            interaction_val = interaction.interaction_term
            if interaction.interaction_ci_95 is not None:
                ci_lo, ci_hi = interaction.interaction_ci_95
                sig_marker = "*" if interaction.interaction_is_significant else ""
                print(f"  Interaction Term: {interaction_val:+.4f} [95% CI: {ci_lo:+.4f}, {ci_hi:+.4f}]{sig_marker}")
            else:
                print(f"  Interaction Term: {interaction_val:+.4f}")
            print(f"  Classification: {interaction.model_type}")
        
        # Generate visualizations
        figures = []
        models_display = [get_model_display_name(m) for m in target_models]
        interactions_display = {get_model_display_name(m): model_interactions[m] for m in target_models}
        
        # Plot 1: Main interaction comparison
        fig_path_main = self.get_figure_path("linguistic_contextual_interaction")
        self.plot_interaction_comparison(
            models_display,
            interactions_display,
            fig_path_main,
        )
        figures.append(fig_path_main)
        print(f"\nSaved: {fig_path_main}")
        
        # Build metrics
        metrics = {
            "models_analyzed": len(target_models),
            "model_results": {},
        }
        
        for model in target_models:
            interaction = model_interactions[model]
            metrics["model_results"][model] = {
                "linguistic_drop_us_context": {
                    "value": interaction.linguistic_drop_us_context.drop_value,
                    "ci_95": interaction.linguistic_drop_us_context.ci_95,
                    "p_value": interaction.linguistic_drop_us_context.p_value,
                    "n_samples": interaction.linguistic_drop_us_context.source_n,
                },
                "linguistic_drop_kr_context": {
                    "value": interaction.linguistic_drop_kr_context.drop_value,
                    "ci_95": interaction.linguistic_drop_kr_context.ci_95,
                    "p_value": interaction.linguistic_drop_kr_context.p_value,
                    "n_samples": interaction.linguistic_drop_kr_context.source_n,
                },
                "contextual_drop_english": {
                    "value": interaction.contextual_drop_english.drop_value,
                    "ci_95": interaction.contextual_drop_english.ci_95,
                    "p_value": interaction.contextual_drop_english.p_value,
                    "n_samples": interaction.contextual_drop_english.source_n,
                },
                "contextual_drop_korean": {
                    "value": interaction.contextual_drop_korean.drop_value,
                    "ci_95": interaction.contextual_drop_korean.ci_95,
                    "p_value": interaction.contextual_drop_korean.p_value,
                    "n_samples": interaction.contextual_drop_korean.source_n,
                },
                "interaction_term": float(interaction.interaction_term),
                "interaction_ci_95": list(interaction.interaction_ci_95) if interaction.interaction_ci_95 else None,
                "interaction_is_significant": bool(interaction.interaction_is_significant),
                "interaction_p_value": interaction.interaction_p_value,
                "model_type": interaction.model_type,
            }

        # ------------------------------------------------------------------
        # Population-level and multiple-comparison analysis 
        #
        # Two distinct claims tested:
        #   1. Per-model claim: which individual models show a significant
        #      interaction. We apply a Bonferroni family-wise correction across
        #      the N per-model tests (the most conservative standard correction).
        #   2. Population claim: does Korean context systematically mitigate
        #      language suppression across models? Tested directly with a
        #      sign/binomial test on the direction of the interaction terms,
        #      with a Wilcoxon signed-rank test (which uses magnitude) as a
        #      more-powerful backup.
        # Caveat: these tests treat the N models as independent units; the
        # interaction estimates share the same task set, so nominal p-values
        # are slightly optimistic.
        # ------------------------------------------------------------------
        interaction_terms = [model_interactions[m].interaction_term for m in target_models]
        per_model_pvals = [
            model_interactions[m].interaction_p_value
            if model_interactions[m].interaction_p_value is not None else 1.0
            for m in target_models
        ]
        n_models = len(interaction_terms)
        n_positive = int(sum(1 for t in interaction_terms if t > 0))
        n_negative = int(sum(1 for t in interaction_terms if t < 0))

        # Primary: sign/binomial test on direction of interaction terms vs 50/50
        sign_two_sided = float(stats.binomtest(n_positive, n_models, 0.5, alternative="two-sided").pvalue)
        sign_one_sided = float(stats.binomtest(n_positive, n_models, 0.5, alternative="greater").pvalue)

        # Secondary: Bonferroni family-wise correction over per-model p-values
        bonferroni_significant = bonferroni_correction(per_model_pvals, alpha=0.05)
        bonferroni_significant_models = [
            get_model_display_name(m) for m, sig in zip(target_models, bonferroni_significant) if sig
        ]

        # Backup: Wilcoxon signed-rank on interaction magnitudes (median > 0)
        try:
            wilcoxon_two_sided = float(stats.wilcoxon(interaction_terms).pvalue)
            wilcoxon_one_sided = float(stats.wilcoxon(interaction_terms, alternative="greater").pvalue)
        except ValueError:
            wilcoxon_two_sided = float("nan")
            wilcoxon_one_sided = float("nan")

        for model, sig in zip(target_models, bonferroni_significant):
            metrics["model_results"][model]["interaction_bonferroni_significant"] = bool(sig)

        metrics["population_tests"] = {
            "n_models": n_models,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "sign_test_two_sided_p": sign_two_sided,
            "sign_test_one_sided_p": sign_one_sided,
            "wilcoxon_two_sided_p": wilcoxon_two_sided,
            "wilcoxon_one_sided_p": wilcoxon_one_sided,
            "bonferroni_significant_models": bonferroni_significant_models,
        }

        print(f"\n{'='*60}")
        print("POPULATION-LEVEL INTERACTION ANALYSIS")
        print(f"{'='*60}")
        print(f"Models analyzed: {n_models}")
        print(f"Positive interaction terms (context mitigates): {n_positive}/{n_models}")
        print(f"Negative interaction terms (context amplifies): {n_negative}/{n_models}")
        print("Sign/binomial test vs 50/50:")
        print(f"  two-sided p = {sign_two_sided:.4f}")
        print(f"  one-sided p (greater) = {sign_one_sided:.4f}")
        print("Wilcoxon signed-rank (median interaction > 0):")
        print(f"  two-sided p = {wilcoxon_two_sided:.4f}")
        print(f"  one-sided p (greater) = {wilcoxon_one_sided:.4f}")
        print(f"Per-model Bonferroni (alpha=0.05/{n_models}) significant: "
              f"{bonferroni_significant_models if bonferroni_significant_models else 'none'}")
        print("Per-model interaction terms and bootstrap p-values:")
        for model in target_models:
            ir = model_interactions[model]
            pv = ir.interaction_p_value if ir.interaction_p_value is not None else float("nan")
            ci_flag = "CI95!=0" if ir.interaction_is_significant else ""
            print(f"  {get_model_display_name(model):<22} term={ir.interaction_term:+.4f}  p={pv:.4f}  {ci_flag}")
        print(f"{'='*60}\n")

        # Generate summary
        type_counts = defaultdict(list)
        for model in target_models:
            mtype = model_interactions[model].model_type.split(":")[0]
            type_counts[mtype].append(get_model_display_name(model))
        
        summary_lines = [
            f"# Interaction Analysis: Language × Context Effects",
            "",
            f"Analyzed {len(target_models)} {model_scope} models on Culture-Specific (CS) tasks.",
            "",
            "## Key Question",
            "Does Korean sociocultural context mitigate or amplify language-driven suppression?",
            "",
            "## Method",
            "Computed the full 2×2 matrix of effects (positive = more conservative):",
            "- **Linguistic Drop (US)**: V(En,US) - V(Ko,US) = Language suppression in US context",
            "- **Linguistic Drop (KR)**: V(En,KR) - V(Ko,KR) = Language suppression in KR context",
            "- **Interaction**: Ling(US) - Ling(KR)",
            "  - Positive: Context MITIGATES suppression (less conservative in KR context)",
            "  - Negative: Context AMPLIFIES suppression (more conservative in KR context)",
            "",
            "**Significance Testing**: 95% CI computed via 1000 bootstrap iterations.",
            "Interaction is significant if CI excludes zero.",
            "",
            "## Model Classifications",
        ]
        
        for mtype in ["Type I", "Type II", "Type III", "Additive"]:
            if mtype in type_counts:
                models_in_type = ", ".join(type_counts[mtype])
                summary_lines.append(f"- **{mtype}**: {models_in_type}")
        
        summary_lines.extend([
            "",
            "## Detailed Results",
            "",
            "| Model | Ling(US) | Ling(KR) | Interaction | 95% CI | Sig | Classification |",
            "|-------|----------|----------|-------------|--------|-----|----------------|",
        ])
        
        for model in target_models:
            display_name = get_model_display_name(model)
            interaction = model_interactions[model]
            ling_us = interaction.linguistic_drop_us_context.drop_value
            ling_kr = interaction.linguistic_drop_kr_context.drop_value
            inter = interaction.interaction_term
            mtype = interaction.model_type.split(":")[0]
            
            if interaction.interaction_ci_95 is not None:
                ci_lo, ci_hi = interaction.interaction_ci_95
                ci_str = f"[{ci_lo:+.3f}, {ci_hi:+.3f}]"
                sig_str = "Yes" if interaction.interaction_is_significant else "No"
            else:
                ci_str = "N/A"
                sig_str = "N/A"
            
            summary_lines.append(f"| {display_name} | {ling_us:+.3f} | {ling_kr:+.3f} | {inter:+.3f} | {ci_str} | {sig_str} | {mtype} |")

        pt = metrics["population_tests"]
        summary_lines.extend([
            "",
            "## Population-Level Tests (multiple-comparison aware)",
            "",
            f"- **Direction of interaction**: {pt['n_positive']}/{pt['n_models']} models positive "
            f"(context mitigates), {pt['n_negative']}/{pt['n_models']} negative (amplifies).",
            f"- **Sign/binomial test vs 50/50** (primary, population claim): "
            f"two-sided p = {pt['sign_test_two_sided_p']:.4f}, one-sided p = {pt['sign_test_one_sided_p']:.4f}.",
            f"- **Wilcoxon signed-rank** (median interaction > 0, backup): "
            f"two-sided p = {pt['wilcoxon_two_sided_p']:.4f}, one-sided p = {pt['wilcoxon_one_sided_p']:.4f}.",
            f"- **Bonferroni correction (alpha=0.05/{pt['n_models']})** over per-model interaction p-values (per-model claim): "
            f"significant = {', '.join(pt['bonferroni_significant_models']) if pt['bonferroni_significant_models'] else 'none'}.",
            "- Caveat: tests treat models as independent units; shared task set makes nominal p-values slightly optimistic.",
            "",
            "## Interpretation",
            "",
            "**Type II (Context Mitigates)**: Models where Korean context reduces the language-driven suppression effect.",
            "This supports the hypothesis that 'context provides information that allows limited escape from language-gated safety response.'",
            "",
            "**Type III (Context Amplifies)**: Models where Korean context intensifies the suppression.",
            "This suggests context is treated as confirming or heightening perceived risk.",
            "",
            "**Type I (Stable)**: Models that show minimal language or context effects.",
        ])
        
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


def run_interaction_analysis(
    config: Optional[ExperimentConfig] = None,
    include_regional: bool = True,
) -> ExperimentResult:
    """Convenience function to run the interaction analysis.
    
    Args:
        config: Experiment configuration
        include_regional: If True, include regional Korean models (14 total).
                          If False (default), only analyze global models (10).
    """
    if config is None:
        config = ExperimentConfig()
    
    # Use phase 2 to match exp1
    config.phase_filter = "2"
    
    analysis = InteractionAnalysis(config)
    return analysis.run(include_regional=include_regional)


if __name__ == "__main__":
    run_interaction_analysis(include_regional=True)
