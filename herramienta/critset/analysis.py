"""Resumen estadístico de un directorio de resultados.

- Promedio y mediana por regla de cada medida para un presupuesto k.
- Semanas ganadas: estrictas (máximo exacto) y con tolerancia (a menos de
  ``tol`` del máximo de la semana; varias reglas pueden ganar la misma semana).
- Diferencias pareadas por semana entre la mejor regla (por promedio) y cada
  otra, con intervalo bootstrap percentil del 95 % remuestreando semanas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MAIN_RULES = ["fuerza", "intermediacion", "pagerank", "articulacion", "cds", "espectral", "voraz"]
METRICS = ["L_est", "L_pond", "F_rel", "L_single", "L_multi"]
COMPARED = ["L_est", "L_pond", "F_rel"]


def summarize(res_dir: str, k: int = 10, tol: float = 0.01, B: int = 5000, seed: int = 0):
    res = Path(res_dir)
    cur = pd.read_csv(res / "curvas.csv", dtype={"week": str})
    cur = cur[cur["k"] == k]
    rules = [r for r in MAIN_RULES if r in set(cur["rule"])]
    sem = pd.read_csv(res / "semanas.csv", dtype={"week": str}).set_index("week")
    rows = []
    for r in rules + ["aleatorio"]:
        d = cur[cur["rule"] == r]
        row = dict(rule=r, weeks=len(d))
        for m in METRICS:
            if m in d:
                row[f"{m}_mean"] = d[m].mean()
                row[f"{m}_median"] = d[m].median()
        rows.append(row)
    out = pd.DataFrame(rows).set_index("rule")
    piv = {m: cur[cur["rule"].isin(rules)].pivot(index="week", columns="rule", values=m)[rules]
           for m in COMPARED if m in cur}
    for m, p in piv.items():
        best = p.max(axis=1)
        out[f"{m}_wins"] = (p.eq(best, axis=0)).sum().reindex(out.index)
        out[f"{m}_wins_tol"] = (p.ge(best - tol, axis=0)).sum().reindex(out.index)
    out["k_over_C_mean"] = float((k / sem["C"]).mean())
    out.to_csv(res / f"resumen_k{k}.csv")

    rng = np.random.default_rng(seed)
    brows = []
    for m, p in piv.items():
        leader = p.mean().idxmax()
        n = len(p)
        idx = rng.integers(0, n, size=(B, n))
        for r in rules:
            if r == leader:
                continue
            diff = (p[leader] - p[r]).to_numpy()
            boots = diff[idx].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            brows.append(dict(metric=m, leader=leader, other=r, mean_diff=diff.mean(),
                              ci_lo=lo, ci_hi=hi, weeks_leader_ahead=int((diff > 0).sum()), n=n))
    boot = pd.DataFrame(brows)
    boot.to_csv(res / f"bootstrap_k{k}.csv", index=False)
    return out, boot
