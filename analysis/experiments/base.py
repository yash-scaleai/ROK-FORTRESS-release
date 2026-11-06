"""
Base experiment class and shared utilities.

Provides common infrastructure for all experiments:
- Data loading and caching
- Output directory management
- Report generation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from .exp_data.loaders import (
    load_all_judgements,
    load_all_judgements_with_responses,
    load_harm_classifications,
    merge_harm_dimensions,
    get_response_stats,
    ConsolidatedResult,
)
from .exp_data.filters import classify_all_tasks
from .exp_data.models import get_model_display_name


@dataclass
class ExperimentConfig:
    """Configuration for running an experiment."""
    
    judgements_dir: Path = field(default_factory=lambda: Path("runs/rok_fortress_judgements_final"))
    responses_dir: Path = field(default_factory=lambda: Path("runs/rok_fortress_responses_final"))
    harm_classifier_path: Path = field(default_factory=lambda: Path("runs/harm-classifier-final/results_gemini_gemini-3-pro-preview.jsonl"))
    output_dir: Path = field(default_factory=lambda: Path("results"))
    target_models: Optional[List[str]] = None  # None = all models
    use_harm_weights: bool = True  # Use tier-weighted scoring
    use_response_files: bool = True  # Whether to load response files for refusal detection
    exclude_tasks: Optional[List[str]] = field(
        default_factory=lambda: ["final_main_master_243", "final_main_master_694"]
    )  # Task IDs to exclude from analysis
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.judgements_dir, str):
            self.judgements_dir = Path(self.judgements_dir)
        if isinstance(self.responses_dir, str):
            self.responses_dir = Path(self.responses_dir)
        if isinstance(self.harm_classifier_path, str):
            self.harm_classifier_path = Path(self.harm_classifier_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class ExperimentResult:
    """Result of running an experiment."""
    
    name: str
    timestamp: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    figures: List[Path] = field(default_factory=list)
    summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "figures": [str(f) for f in self.figures],
            "summary": self.summary,
        }
    
    def save(self, output_path: Path) -> None:
        """Save result to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class BaseExperiment(ABC):
    """
    Base class for all experiments.
    
    Provides:
    - Data loading with panel consolidation
    - Harm dimension merging
    - Output directory management
    - Result saving
    """
    
    name: str = "base"
    description: str = "Base experiment"
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._data_cache: Optional[Dict[str, List[ConsolidatedResult]]] = None
        self._harm_cache: Optional[Dict[str, List[Dict]]] = None
        self._task_types_cache: Optional[Dict[str, str]] = None
    
    @property
    def output_dir(self) -> Path:
        """Experiment-specific output directory."""
        return self.config.output_dir / self.name
    
    def load_data(self) -> Dict[str, List[ConsolidatedResult]]:
        """
        Load panel-consolidated judgements for all target models.
        
        If use_response_files is enabled in config, also loads response file data
        to identify refusals and errors:
        - content_policy_refusal: Model refused due to content policy → harm = 0
        - empty_response: Model returned empty (e.g., Claude safety filter) → harm = 0
        - rate_limit, timeout, other_error: Technical errors → excluded from analysis
        
        Returns:
            Dict mapping model_name -> list of ConsolidatedResult
        """
        if self._data_cache is not None:
            return self._data_cache
        
        # Choose loading method based on config
        if self.config.use_response_files and self.config.responses_dir.exists():
            print(f"Loading judgements with response files from {self.config.judgements_dir}...")
            print(f"  Response files: {self.config.responses_dir}")
            self._data_cache = load_all_judgements_with_responses(
                self.config.judgements_dir,
                self.config.responses_dir,
                self.config.target_models,
            )
        else:
            print(f"Loading judgements from {self.config.judgements_dir}...")
            self._data_cache = load_all_judgements(
                self.config.judgements_dir,
                self.config.target_models,
            )
        
        # Merge harm dimensions if available
        if self.config.harm_classifier_path.exists():
            print(f"Loading harm classifications from {self.config.harm_classifier_path}...")
            self._harm_cache = load_harm_classifications(self.config.harm_classifier_path)
            
            for model, results in self._data_cache.items():
                merge_harm_dimensions(results, self._harm_cache)
        
        # Filter out excluded tasks if specified
        if self.config.exclude_tasks:
            exclude_set = set(self.config.exclude_tasks)
            for model in self._data_cache:
                self._data_cache[model] = [
                    r for r in self._data_cache[model]
                    if r.task_id not in exclude_set
                ]
            print(f"Excluded {len(exclude_set)} tasks: {sorted(exclude_set)}")
        
        # Print summary
        total_results = sum(len(r) for r in self._data_cache.values())
        print(f"Loaded {total_results} results across {len(self._data_cache)} models")
        
        # Print response status summary if using response files
        if self.config.use_response_files:
            self._print_response_summary()
        
        return self._data_cache
    
    def _print_response_summary(self) -> None:
        """Print summary of response statuses across all models."""
        if self._data_cache is None:
            return
        
        from collections import Counter
        
        all_statuses = Counter()
        for model, results in self._data_cache.items():
            stats = get_response_stats(results)
            for status, count in stats.items():
                all_statuses[status] += count
        
        print("\nResponse Status Summary:")
        print(f"  Normal responses (use verdicts):  {all_statuses.get('normal', 0):,}")
        print(f"  Safe refusals (harm = 0):")
        print(f"    - Content policy refusal:       {all_statuses.get('content_policy_refusal', 0):,}")
        print(f"    - Empty response (safety):      {all_statuses.get('empty_response', 0):,}")
        print(f"  Excluded (technical errors):")
        print(f"    - Rate limit:                   {all_statuses.get('rate_limit', 0):,}")
        print(f"    - Timeout:                      {all_statuses.get('timeout', 0):,}")
        print(f"    - Other error:                  {all_statuses.get('other_error', 0):,}")
        print(f"  Missing response file data:       {all_statuses.get('missing', 0):,}")
    

    def compute_response_metadata(
        self,
        data: Dict[str, List[ConsolidatedResult]],
    ) -> Dict[str, Any]:
        """
        Compute metadata about response statuses across all models.
        
        Returns dict with counts of each response status type.
        """
        from collections import Counter
        
        all_statuses = Counter()
        per_model_statuses: Dict[str, Dict[str, int]] = {}
        
        for model, results in data.items():
            stats = get_response_stats(results)
            per_model_statuses[model] = stats
            for status, count in stats.items():
                all_statuses[status] += count
        
        return {
            "aggregate": dict(all_statuses),
            "per_model": per_model_statuses,
            "safe_refusals_total": all_statuses.get("content_policy_refusal", 0) + all_statuses.get("empty_response", 0),
            "excluded_total": all_statuses.get("rate_limit", 0) + all_statuses.get("timeout", 0) + all_statuses.get("other_error", 0),
            "normal_total": all_statuses.get("normal", 0),
        }

    def get_task_types(self) -> Dict[str, str]:
        """
        Get task type classification (CA/CS) for all tasks.
        
        Returns:
            Dict mapping task_id -> "CA" or "CS"
        """
        if self._task_types_cache is not None:
            return self._task_types_cache
        
        # Load data if not cached
        data = self.load_data()
        
        # Flatten all results
        all_results = []
        for results in data.values():
            all_results.extend(results)
        
        self._task_types_cache = classify_all_tasks(all_results)
        return self._task_types_cache
    
    def setup_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_figure_path(self, name: str, ext: str = "png") -> Path:
        """Get path for a figure file."""
        self.setup_output_dir()
        return self.output_dir / f"{name}.{ext}"
    
    def get_result_path(self) -> Path:
        """Get path for the result JSON file."""
        self.setup_output_dir()
        return self.output_dir / "result.json"
    
    @abstractmethod
    def run(self) -> ExperimentResult:
        """
        Run the experiment.
        
        Returns:
            ExperimentResult with metrics, figures, and summary
        """
        pass
    
    def generate_report(self, result: ExperimentResult) -> str:
        """
        Generate a human-readable report from the result.
        
        Args:
            result: ExperimentResult from run()
        
        Returns:
            Markdown-formatted report string
        """
        lines = [
            f"# {self.name}: {self.description}",
            "",
            f"**Timestamp:** {result.timestamp}",
            "",
            "## Summary",
            result.summary,
            "",
            "## Metrics",
        ]
        
        for key, value in result.metrics.items():
            if isinstance(value, dict):
                lines.append(f"### {key}")
                for k, v in value.items():
                    lines.append(f"- {k}: {v}")
            else:
                lines.append(f"- **{key}:** {value}")
        
        if result.figures:
            lines.extend([
                "",
                "## Figures",
            ])
            for fig_path in result.figures:
                lines.append(f"- {fig_path.name}")
        
        return "\n".join(lines)
    
    def save_result(self, result: ExperimentResult) -> None:
        """Save experiment result and generate report."""
        # Save JSON result
        result.save(self.get_result_path())
        
        # Save markdown report
        report = self.generate_report(result)
        report_path = self.output_dir / "report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"Results saved to {self.output_dir}")


def create_experiment_result(
    name: str,
    metrics: Dict[str, Any],
    figures: List[Path],
    summary: str,
) -> ExperimentResult:
    """Helper to create an ExperimentResult with current timestamp."""
    return ExperimentResult(
        name=name,
        timestamp=datetime.now().isoformat(),
        metrics=metrics,
        figures=figures,
        summary=summary,
    )
