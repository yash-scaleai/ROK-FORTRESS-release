"""Per-domain drops on the corrected 3-judge pipeline (replaces the uncommitted Table 18 / fig_domain_drops generator).

Δling(CA)  = En − Ko on culture-agnostic tasks                      (Def 3.2 eq. 1)
Δling(CS)  = ½[(EnUS−KoUS) + (EnKR−KoKR)] on complete-4 CS tasks    (rebuttal W1)
Δctx       = ½[(EnUS−EnKR) + (KoUS−KoKR)] on complete-4 CS tasks    (Def 3.2 eq. 2)
Pooled over paired (task, model) differences; percentile bootstrap 95% CI, 2,000 resamples, seed 42.
Also caches S/domain to scores_cache.pkl for later scripts.
"""
import pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(Path.cwd()))
from scripts.metaeval.recompute_ling_ctx_drops import load, scores, task_sets, EN_US, KO_US, EN_KR, KO_KR

cache = Path("results/scores_cache.pkl")
if cache.exists():
    S, Su, ca, cs, dom = pickle.loads(cache.read_bytes())
else:
    data = load()
    ca, cs = task_sets(data)
    S, Su = scores(data, True), scores(data, False)
    dom = {}
    for rs in data.values():
        for r in rs:
            if r.risk_domain:
                assert dom.setdefault(r.task_id, r.risk_domain) == r.risk_domain
    missing = (ca | cs) - set(dom)
    assert not missing, f"tasks with no domain: {len(missing)}"
    cache.write_bytes(pickle.dumps((S, Su, ca, cs, dom)))

SHORT = {"Chemical, Biological, Radiological, Nuclear, and Explosive (CBRNE)": "CBRNE",
         "Political Violence & Terrorism": "Political Violence",
         "Criminal & Financial Illicit Activities": "Criminal Activity",
         "Information Leakage": "Information Leakage"}
assert set(dom.values()) == set(SHORT), set(dom.values())
ORDER = ["CBRNE", "Political Violence", "Criminal Activity", "Information Leakage"]
C4 = (EN_US, KO_US, EN_KR, KO_KR)


def rows(kind):
    out = []
    for m, dm in S.items():
        for t, d in dm.items():
            if kind == "ling_ca" and t in ca and EN_US in d and KO_US in d:
                out.append((SHORT[dom[t]], d[EN_US] - d[KO_US]))
            elif kind in ("ling_cs", "ctx") and t in cs and all(v in d for v in C4):
                v = ((d[EN_US] - d[KO_US]) + (d[EN_KR] - d[KO_KR])) / 2 if kind == "ling_cs" \
                    else ((d[EN_US] - d[EN_KR]) + (d[KO_US] - d[KO_KR])) / 2
                out.append((SHORT[dom[t]], v))
    return out


def est(x, rng):
    x = np.asarray(x)
    b = x[rng.integers(0, len(x), (2000, len(x)))].mean(axis=1)
    return 100 * x.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5), len(x)


rng = np.random.default_rng(42)
res = {}
for kind in ("ling_ca", "ling_cs", "ctx"):
    r = rows(kind)
    for g in ORDER + ["Pooled"]:
        xs = [v for d, v in r if g == "Pooled" or d == g]
        res[(kind, g)] = est(xs, rng)

print(f"{'Domain':<20} {'Δling CA':>26} {'Δling CS':>26} {'Δctx':>26}")
for g in ORDER + ["Pooled"]:
    print(f"{g:<20}" + "".join(f"  {m:+6.2f} [{lo:+.2f},{hi:+.2f}] n={n:<5}" for m, lo, hi, n in (res[(k, g)] for k in ("ling_ca", "ling_cs", "ctx"))))
ntask = {g: (len({t for t in ca if SHORT[dom[t]] == g}), len({t for t in cs if SHORT[dom[t]] == g})) for g in ORDER}
print("tasks per domain (CA, CS):", ntask)

print("\nLaTeX rows:")
for g in ORDER + ["Pooled"]:
    cells = " & ".join(f"${m:+.2f}$ & [${lo:+.2f}$, ${hi:+.2f}$]" for m, lo, hi, _ in (res[(k, g)] for k in ("ling_ca", "ling_cs", "ctx")))
    print(f"{g} & {cells} \\\\")

fig, ax = plt.subplots(figsize=(6.4, 2.8))
w, x = 0.27, np.arange(len(ORDER))
for i, (k, lab, c) in enumerate([("ling_ca", r"$\Delta_{\rm ling}$ (CA)", "#a970c9"),
                                 ("ling_cs", r"$\Delta_{\rm ling}$ (CS)", "#6a3d9a"),
                                 ("ctx", r"$\Delta_{\rm ctx}$ (CS)", "#ef9a48")]):
    m = np.array([res[(k, g)][0] for g in ORDER])
    lo = np.array([res[(k, g)][1] for g in ORDER]); hi = np.array([res[(k, g)][2] for g in ORDER])
    ax.bar(x + (i - 1) * w, m, w, yerr=[m - lo, hi - m], capsize=3, color=c, edgecolor="black", linewidth=0.6, label=lab)
ax.set_xticks(x, ORDER, fontsize=8); ax.set_ylabel("Drop (pp, TRS)"); ax.axhline(0, color="black", lw=0.6)
ax.legend(fontsize=7, frameon=False); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(Path("results/fig_domain_drops.pdf")); print("wrote", Path("results/fig_domain_drops.pdf"))
