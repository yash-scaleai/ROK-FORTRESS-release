"""Sensitivity to empty adversarial responses that still carry a reasoning trace (appendix tab:empty_sensitivity).

Main results score every empty adversarial response as a refusal (TRS 0). Here, empties whose
`reasoning_content` is non-empty (truncated generation) are treated as missing instead.
Reports per-model Δling,CS / Δctx / interaction on complete-4 culture-specific tasks, the 14-model
means and ratio, and a paired-task bootstrap p for the interaction (20,000 resamples, seed 42).

Needs results/scores_cache.json (written by scripts/metaeval/domain_drops.py).
Usage: uv run python scripts/metaeval/empty_response_sensitivity.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from scripts.metaeval.recompute_ling_ctx_drops import EN_KR, EN_US, KO_KR, KO_US

RESPONSES = Path("runs/rok_fortress_responses_final")
C4 = (EN_US, KO_US, EN_KR, KO_KR)
N_BOOT = 20_000


def truncated_empties(models):
    drop = {m: set() for m in models}
    for f in RESPONSES.glob("responses_*.jsonl"):
        m = f.name[len("responses_"):].rsplit("_", 2)[0]
        if m not in drop:
            continue
        for line in open(f):
            r = json.loads(line)
            x = r["adversarial"]
            if not (x.get("response") or "").strip() and x.get("error") is None and (x.get("reasoning_content") or "").strip():
                drop[m].add((r["task_id"], r["variant_name"]))
    return drop


def stats(d, cs):
    a = np.array([[d[t][v] for v in C4] for t in cs if t in d and all(v in d[t] for v in C4)])
    it = a[:, 0] - a[:, 1] - a[:, 2] + a[:, 3]
    b = it[np.random.default_rng(42).integers(0, len(it), (N_BOOT, len(it)))].mean(1)
    ling = ((a[:, 0] - a[:, 1]) + (a[:, 2] - a[:, 3])).mean() / 2
    ctx = ((a[:, 0] - a[:, 2]) + (a[:, 1] - a[:, 3])).mean() / 2
    return 100 * ling, 100 * ctx, 100 * it.mean(), 2 * min((b <= 0).mean(), (b >= 0).mean())


def main():
    c = json.loads(Path("results/scores_cache.json").read_text())
    S, cs = c["S"], set(c["cs"])
    assert len(S) == 14, f"expected 14 models, got {len(S)}"
    drop = truncated_empties(S)
    agg = {"scored": [], "missing": []}
    print(f"{'model':<22}{'dropped':>8}  {'ling scored/missing':>20}  {'ctx scored/missing':>19}  interaction scored (p) / missing (p)")
    for m in sorted(S):
        rows = {}
        for k, ex in (("scored", set()), ("missing", drop[m])):
            d = {t: {v: s for v, s in vs.items() if (t, v) not in ex} for t, vs in S[m].items()}
            rows[k] = stats(d, cs)
            agg[k].append(rows[k])
        if drop[m]:
            s, x = rows["scored"], rows["missing"]
            print(f"{m[:22]:<22}{len(drop[m]):>8}  {s[0]:>9.2f}/{x[0]:<9.2f}  {s[1]:>9.2f}/{x[1]:<9.2f}  "
                  f"{s[2]:+.2f} ({s[3]:.4f}) / {x[2]:+.2f} ({x[3]:.4f})")
    for k, rs in agg.items():
        ling, ctx = np.mean([r[0] for r in rs]), np.mean([r[1] for r in rs])
        print(f"MEAN(14) {k:<8} d_ling_CS={ling:.2f}  d_ctx={ctx:.2f}  ratio={ling / ctx:.2f}x  "
              f"Bonferroni-significant interactions={sum(r[3] < 0.05 / 14 for r in rs)}")


if __name__ == "__main__":
    main()
