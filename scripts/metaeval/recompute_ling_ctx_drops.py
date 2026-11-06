"""Independent recompute of linguistic/contextual drops on the corrected 3-judge pipeline.

Run from repo root:  uv run python scripts/metaeval/recompute_ling_ctx_drops.py
Uses the canonical loader (BaseExperiment.load_data) so panel voting, refusal
handling, harm-dimension weights and task exclusions match the paper pipeline.
"""
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from analysis.experiments.base import BaseExperiment, ExperimentConfig
from analysis.experiments.exp_data.variants import (
    ORIGINAL_EN, TRANSLATED_KR, CULTURAL_ADAPTED_EN, TRANSCREATED_KR,
)
from analysis.experiments.exp_data.loaders import JUDGE_NAMES
from analysis.experiments.metrics.harm_scores import compute_harm_score_for_result, DOMAIN_NAME_MAP

EN_US, KO_US, EN_KR, KO_KR = ORIGINAL_EN, TRANSLATED_KR, CULTURAL_ADAPTED_EN, TRANSCREATED_KR


class _Loader(BaseExperiment):
    name = "recompute_drops"
    def run(self):
        raise NotImplementedError


def load():
    data = _Loader(ExperimentConfig()).load_data()
    assert len(data) == 14, f"expected 14 models, got {len(data)}: {sorted(data)}"
    # sanity: every result must carry all 3 judges, domain must map, dims present
    judge_counts, bad_domain, no_dims = Counter(), 0, 0
    for rs in data.values():
        for r in rs:
            judge_counts[len(r.judges_available)] += 1
            if DOMAIN_NAME_MAP.get(r.risk_domain) is None:
                bad_domain += 1
            if not r.is_safe_refusal and not r.harm_dimensions:
                no_dims += 1
    print(f"judges-per-result: {dict(judge_counts)}  (JUDGE_NAMES={JUDGE_NAMES})")
    print(f"results with unmapped domain: {bad_domain}; non-refusal results without harm dims: {no_dims}")
    return data


def scores(data, use_weights):
    """S[model][task][variant] = TRS (0-1). Only valid, non-excluded results."""
    S = {}
    for m, rs in data.items():
        d = defaultdict(dict)
        for r in rs:
            if r.is_valid and not r.is_excluded:
                d[r.task_id][r.variant_name] = compute_harm_score_for_result(r, use_weights=use_weights).harm_score
        S[m] = d
    return S


def task_sets(data):
    """CA = tasks whose variant set is exactly {EN_US, KO_US}; CS = tasks that have any adapted variant."""
    vs = defaultdict(set)
    for rs in data.values():
        for r in rs:
            vs[r.task_id].add(r.variant_name)
    ca = {t for t, v in vs.items() if v <= {EN_US, KO_US}}
    cs = {t for t, v in vs.items() if v & {EN_KR, KO_KR}}
    assert not (ca & cs)
    return ca, cs


def diff(d, a, b):
    return d[a] - d[b] if a in d and b in d else None


def two_term(d, p1, p2, complete4):
    """½[(p1a-p1b)+(p2a-p2b)]; complete4 requires all four, else average whichever pairs exist."""
    if complete4 and not all(v in d for v in (EN_US, KO_US, EN_KR, KO_KR)):
        return None
    xs = [x for x in (diff(d, *p1), diff(d, *p2)) if x is not None]
    if not xs or (complete4 and len(xs) != 2):
        return None
    return sum(xs) / len(xs)


LING_CS = ((EN_US, KO_US), (EN_KR, KO_KR))
CTX_CS = ((EN_US, EN_KR), (KO_US, KO_KR))


def per_model_task_values(S, tasks, fn):
    """{model: {task: value}} for tasks where fn returns a value."""
    out = {}
    for m, d in S.items():
        out[m] = {t: v for t in tasks if t in d and (v := fn(d[t])) is not None}
    return out


def macro(vals):
    return float(np.mean([np.mean(list(v.values())) for v in vals.values()]))


def pooled(vals):
    return float(np.mean([x for v in vals.values() for x in v.values()]))


def macro_unpaired(S, tasks, a_variants, b_variants):
    """Def 3.2 literal: per-model TRS(m,V) = mean over all valid results of V, then difference, then mean over models.
    a_variants/b_variants: list of (plus, minus) pairs averaged."""
    per_model = []
    for m, d in S.items():
        terms = []
        for plus, minus in zip(a_variants, b_variants):
            p = [d[t][plus] for t in tasks if t in d and plus in d[t]]
            q = [d[t][minus] for t in tasks if t in d and minus in d[t]]
            terms.append(np.mean(p) - np.mean(q))
        per_model.append(np.mean(terms))
    return float(np.mean(per_model))


def bootstrap_macro(S, tasks, fns, n=2000, seed=42):
    """Task-level bootstrap of macro (per-model then mean) for several fns; returns CIs incl. ratio of fns[0]/fns[1]."""
    rng = np.random.default_rng(seed)
    models = sorted(S)
    tl = sorted(tasks)
    # matrix [fn][model][task] with nan where missing
    M = np.full((len(fns), len(models), len(tl)), np.nan)
    for i, fn in enumerate(fns):
        for j, m in enumerate(models):
            d = S[m]
            for k, t in enumerate(tl):
                if t in d and (v := fn(d[t])) is not None:
                    M[i, j, k] = v
    boots = np.empty((n, len(fns)))
    for b in range(n):
        idx = rng.integers(0, len(tl), len(tl))
        boots[b] = np.nanmean(np.nanmean(M[:, :, idx], axis=2), axis=1)
    ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return [ci(boots[:, i]) for i in range(len(fns))], ci(boots[:, 0] / boots[:, 1])


def main():
    data = load()
    ca, cs = task_sets(data)
    print(f"CA tasks={len(ca)}  CS tasks={len(cs)}")

    for use_weights in (True, False):
        S = scores(data, use_weights)
        tag = "TIER-WEIGHTED TRS" if use_weights else "UNIFORM WEIGHTS (ARS)"
        print(f"\n===== {tag} =====")

        ling_ca = per_model_task_values(S, ca, lambda d: diff(d, EN_US, KO_US))
        rows = []
        for c4 in (True, False):
            lab = "complete-4" if c4 else "per-pair"
            lcs = per_model_task_values(S, cs, lambda d, c4=c4: two_term(d, *LING_CS, c4))
            ccs = per_model_task_values(S, cs, lambda d, c4=c4: two_term(d, *CTX_CS, c4))
            rows += [
                (f"Δling CS  macro  {lab}", macro(lcs)), (f"Δctx  CS  macro  {lab}", macro(ccs)),
                (f"Δling CS  pooled {lab}", pooled(lcs)), (f"Δctx  CS  pooled {lab}", pooled(ccs)),
            ]
            if c4:
                c4_counts = [len(v) for v in lcs.values()]
        rows += [
            ("Δling CA  macro  paired", macro(ling_ca)),
            ("Δling CA  pooled paired", pooled(ling_ca)),
            ("Δling CA  macro  unpaired (Def3.2)", macro_unpaired(S, ca, [EN_US], [KO_US])),
            ("Δctx  CS  macro  unpaired (Def3.2)", macro_unpaired(S, cs, [EN_US, KO_US], [EN_KR, KO_KR])),
            ("Δling CS  macro  unpaired", macro_unpaired(S, cs, [EN_US, EN_KR], [KO_US, KO_KR])),
        ]
        for k, v in rows:
            print(f"  {k:<40} {100*v:7.2f} pp")
        print(f"  complete-4 CS tasks per model: min={min(c4_counts)} max={max(c4_counts)}")

        f_l = lambda d: two_term(d, *LING_CS, True)
        f_c = lambda d: two_term(d, *CTX_CS, True)
        (cl, cc), cr = bootstrap_macro(S, cs, [f_l, f_c])
        lm = macro(per_model_task_values(S, cs, f_l)); cm = macro(per_model_task_values(S, cs, f_c))
        print(f"  HEADLINE CS complete-4 macro: Δling={100*lm:.2f} [{100*cl[0]:.2f},{100*cl[1]:.2f}]  "
              f"Δctx={100*cm:.2f} [{100*cc[0]:.2f},{100*cc[1]:.2f}]  ratio={lm/cm:.2f}x [{cr[0]:.2f},{cr[1]:.2f}]")

        if use_weights:
            print("  per-model CS complete-4 (Δling, Δctx, Ling(US)-Ling(KR) interaction):")
            for m in sorted(S):
                d = S[m]
                c4t = [t for t in cs if t in d and all(v in d[t] for v in (EN_US, KO_US, EN_KR, KO_KR))]
                a = np.array([[d[t][EN_US], d[t][KO_US], d[t][EN_KR], d[t][KO_KR]] for t in c4t])
                lus, lkr = (a[:, 0] - a[:, 1]).mean(), (a[:, 2] - a[:, 3]).mean()
                ctx = ((a[:, 0] - a[:, 2]) + (a[:, 1] - a[:, 3])).mean() / 2
                print(f"    {m:<45} n={len(c4t):4d}  ling={100*(lus+lkr)/2:6.2f}  ctx={100*ctx:6.2f}  inter={100*(lus-lkr):6.2f}")


if __name__ == "__main__":
    main()
