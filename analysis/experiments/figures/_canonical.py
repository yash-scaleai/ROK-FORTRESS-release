"""Canonical data layer for the figure scripts.

Replaces the dead bespoke loaders in gen_*.py (which pointed at non-existent
dirs and scored TRS from a single judge's labels). Uses the real 3-judge
panel-consolidated, refusal-aware scoring library. Fails loud on missing inputs.
"""
import json
from collections import defaultdict
from pathlib import Path

from analysis.experiments.exp_data.loaders import (
    load_all_judgements_with_responses,
    load_harm_classifications,
    merge_harm_dimensions,
)
from analysis.experiments.exp_data.models import normalize_model_name
from analysis.experiments.metrics.harm_scores import compute_harm_score_for_result

REPO = Path(__file__).resolve().parents[3]
JUDGEMENTS_DIR = REPO / "runs/rok_fortress_judgements_final"
RESPONSES_DIR = REPO / "runs/rok_fortress_responses_final"
HARM_CLASSIFIER_FILE = REPO / "runs/harm-classifier-final/results_gemini_gemini-3-pro-preview.jsonl"
BENIGN_REFUSAL_DIR = REPO / "runs/rok_fortress_refusal_checks_final"
BENIGN_REFUSAL_GLOB = "refusal_benign_gemini_3_1_pro_preview_*.jsonl"


def load_results_by_model():
    """{model_token: {variant: [ConsolidatedResult]}} with harm dimensions merged."""
    for p in (JUDGEMENTS_DIR, RESPONSES_DIR, HARM_CLASSIFIER_FILE):
        if not p.exists():
            raise FileNotFoundError(f"Required input missing: {p}")
    data = load_all_judgements_with_responses(JUDGEMENTS_DIR, RESPONSES_DIR)
    harm = load_harm_classifications(HARM_CLASSIFIER_FILE)
    out = defaultdict(lambda: defaultdict(list))
    for model, results in data.items():
        merge_harm_dimensions(results, harm)
        for r in results:
            out[model][r.variant_name].append(r)
    return {m: dict(v) for m, v in out.items()}


def response_trs(result):
    """Tier-weighted harm score (0-1) for one consolidated response."""
    return compute_harm_score_for_result(result, use_weights=True).harm_score


def trs_percent(results):
    """Mean tier-weighted harm score (%) over valid, non-excluded responses."""
    scores = [response_trs(r) for r in results if r.is_valid and not r.is_excluded]
    return (sum(scores) / len(scores) * 100) if scores else 0.0


def load_benign_ors_counts():
    """{model_token: {variant: (refusals, total)}} from benign refusal checks."""
    if not BENIGN_REFUSAL_DIR.exists():
        raise FileNotFoundError(f"Benign refusal dir missing: {BENIGN_REFUSAL_DIR}")
    files = list(BENIGN_REFUSAL_DIR.glob(BENIGN_REFUSAL_GLOB))
    if not files:
        raise FileNotFoundError(f"No {BENIGN_REFUSAL_GLOB} in {BENIGN_REFUSAL_DIR}")
    ref = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                rec = json.loads(line)
                m = normalize_model_name(rec.get("model_name", ""))
                v = rec.get("variant_name", "")
                ref[m][v][1] += 1
                if rec.get("is_refusal"):
                    ref[m][v][0] += 1
    return {m: {v: tuple(c) for v, c in vs.items()} for m, vs in ref.items()}
