"""Protocolo experimental (entrega 2, Sec. VI-C) y caracterización del panel.

- Semanas: primera semana disponible de cada trimestre 2022–2024 y última
  instantánea anterior a cada episodio. Duplicados eliminados.
- Presupuestos k ∈ {1,2,5,10,20} fijados antes de observar resultados.
- Cada regla produce su secuencia una vez (k_max) y se evalúan sus prefijos
  sobre una copia intacta de la instantánea.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from . import rules
from .graph import Snapshot, build_snapshot, snapshot_from_panel, two_core_names
from .metrics import Evaluator, jaccard

K_LIST = [1, 2, 5, 10, 20]
N_RANDOM = 20

# Fechas de inicio de los episodios. Terra y FTX: las del event study de
# DeXposure-FM (Shu et al. 2026, §6.4). USDC: pérdida de paridad del 11-mar-2023.
EVENTS = {
    "terra": "2022-05-09",
    "ftx": "2022-11-07",
    "usdc": "2023-03-11",
}
# Indicador complementario de afectación, fijado antes del cotejo:
# caída >= 50 % de la fuerza de exposición entre la semana previa y 4 semanas después.
AFFECT_WINDOW_WEEKS = 4
AFFECT_DROP = 0.5


def select_weeks(weeks: list[str]) -> dict[str, list[str]]:
    w = sorted(weeks)
    sel: dict[str, list[str]] = {}
    for y in (2022, 2023, 2024):
        for q in range(4):
            months = [f"{y}-{m:02d}" for m in range(3 * q + 1, 3 * q + 4)]
            first = next((x for x in w if x[:7] in months), None)
            if first:
                sel.setdefault(first, []).append(f"T{q + 1}-{y}")
    for ev, d in EVENTS.items():
        prev = [x for x in w if x < d]
        if prev:
            sel.setdefault(prev[-1], []).append(f"pre-{ev}")
    return {k: sel[k] for k in sorted(sel)}


def load_tvl(nodes_csvs: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for p in nodes_csvs:
        d = pd.read_csv(p, dtype={"time": str, "id": str})
        d = d[np.isfinite(d["size"])]
        for wk, g in d.groupby("time"):
            out.setdefault(wk, {}).update(g.groupby("id")["size"].sum().to_dict())
    return out


def run_rule(name, fn, snap, ev, kmax, **kw):
    info: dict = {}
    tracemalloc.start()
    t0 = time.perf_counter()
    seq = fn(snap, kmax, ev=ev, info=info, **kw)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return seq, dt, peak / 2**20, info


def run_week(args) -> dict:
    """Corre todas las reglas sobre una semana. Pensado para multiprocessing."""
    panel_path, week, labels, tvl_week, kmax, with_raw, *rest = args
    only = rest[0] if rest else None        # subconjunto de reglas (None = todas)
    exclude = rest[1] if len(rest) > 1 else ()  # identificadores excluidos del panel
    core2 = rest[2] if len(rest) > 2 else False  # restringir C_t a su 2-núcleo
    panel = pd.read_csv(panel_path, dtype={"time": str, "source": str, "target": str})
    snap = snapshot_from_panel(panel, week, exclude=exclude, core2=core2)
    ev = Evaluator(snap)
    rows, seqs, extra = [], {}, {}
    todo = dict(rules.RULES)
    todo["voraz"] = rules.rule_greedy
    if tvl_week is not None:
        cover = np.mean([s in tvl_week for s in snap.names])
        extra["tvl_coverage"] = float(cover)
        if cover == 1.0:
            todo["tvl"] = rules.rule_tvl
            todo["sis"] = rules.rule_sis
    if only:
        todo = {n: f for n, f in todo.items() if n in only}
    if with_raw and (not only or "espectral_crudo" in only):
        todo["espectral_crudo"] = lambda s, k, **kw: rules.rule_spectral(s, k, raw=True, **kw)
    for name, fn in todo.items():
        seq, dt, mem, info = run_rule(name, fn, snap, ev, kmax, tvl=tvl_week)
        seqs[name] = [str(snap.names[i]) for i in seq]
        if "cds_size" in info:
            extra["cds_size"] = info["cds_size"]
        if "rho_trace" in info and name == "espectral":
            extra["rho_trace"] = info["rho_trace"]
        for k in K_LIST:
            if k > len(seq) or k >= snap.n:
                continue
            d = ev.evaluate_set(seq[:k])
            rows.append(dict(week=week, rule=name, k=k, **d.row(), seconds=dt, peak_mb=mem))
    rows += random_rows(snap, ev, week)
    meta = dict(week=week, labels=labels, V=snap.n_vertices, E=snap.n_arcs,
                ncomp=snap.n_components, C=snap.n, k_over_C=20 / snap.n,
                W0=snap.W0, W_all=snap.W_total_all, **extra)
    return dict(rows=rows, seqs=seqs, meta=meta)


def random_rows(snap: Snapshot, ev: Evaluator, week: str) -> list[dict]:
    """Línea base aleatoria: media de N_RANDOM permutaciones con semillas 0..N_RANDOM-1."""
    rows = []
    for k in K_LIST:
        ds = [ev.evaluate_set(rules.rule_random(snap, k, seed=s)).row() for s in range(N_RANDOM)]
        rows.append(dict(week=week, rule="aleatorio", k=k,
                         **{c: float(np.mean([d[c] for d in ds])) for c in ds[0]},
                         seconds=np.nan, peak_mb=np.nan))
    return rows


def reevaluate_week(args) -> dict:
    """Recalcula todas las medidas de daño a partir de las secuencias ya guardadas
    (sin volver a correr las reglas). Devuelve filas de curvas y diagnósticos de la semana."""
    panel_path, week, seqs, exclude, core2 = args
    panel = pd.read_csv(panel_path, dtype={"time": str, "source": str, "target": str})
    snap = snapshot_from_panel(panel, week, exclude=exclude, core2=core2)
    ev = Evaluator(snap)
    rows = []
    for name, ids in seqs.items():
        seq = snap.index_of(ids)
        assert len(seq) == len(ids), f"{week}/{name}: identificadores fuera de C_t"
        for k in K_LIST:
            if k > len(seq) or k >= snap.n:
                continue
            rows.append(dict(week=week, rule=name, k=k, **ev.evaluate_set(seq[:k]).row()))
    rows += random_rows(snap, ev, week)
    # diagnósticos: PageRank frente a fuerza, hojas
    from scipy.stats import spearmanr
    s = rules.strength(snap)
    pr = rules.pagerank(snap.W)
    deg = np.diff(snap.U.indptr)
    diag = dict(week=week, C=snap.n, leaves=int((deg == 1).sum()),
                core2=len(two_core_names(snap)),
                spearman_pr_fuerza=float(spearmanr(pr, s).statistic))
    return dict(rows=rows, diag=diag)


# --------------------------------------------------------------- caracterización
def characterize(snap: Snapshot) -> dict:
    s = rules.strength(snap)
    top10 = np.sort(snap.w)[::-1][:10].sum() / snap.W0  # 10 aristas de mayor peso
    s_sorted = np.sort(s)[::-1]
    # fracción de la fuerza total en los 10 vértices de mayor fuerza
    top10_strength = s_sorted[:10].sum() / s.sum()
    deg = np.diff(snap.U.indptr)
    lam2 = np.nan
    if snap.n >= 2:
        L = np.diag(deg.astype(float)) - snap.U.toarray().astype(float)
        vals = eigh(L, eigvals_only=True, subset_by_index=[0, 1])
        lam2 = float(vals[1])
    recip = np.nan
    if snap.n_arcs:
        pairs = set(zip(snap.src.tolist(), snap.dst.tolist()))
        recip = sum((b, a) in pairs for a, b in pairs) / len(pairs)
    return dict(week=snap.week, V=snap.n_vertices, E=snap.n_arcs, ncomp=snap.n_components,
                C=snap.n, giant_frac=snap.n / snap.n_vertices, lambda2=lam2,
                deg_mean=float(deg.mean()), deg_max=int(deg.max()), leaves=int((deg == 1).sum()),
                n_articulation=None, top10_arcs_share=float(top10),
                top10_strength_share=float(top10_strength), reciprocity=float(recip),
                W0=snap.W0, W_all=snap.W_total_all)


def characterize_week(args) -> dict:
    panel_path, week = args
    panel = pd.read_csv(panel_path, dtype={"time": str, "source": str, "target": str})
    snap = build_snapshot(panel[panel["time"] == week], week)
    out = characterize(snap)
    import igraph as ig

    g = ig.Graph(n=snap.n, edges=list(zip(snap.usrc.tolist(), snap.udst.tolist())))
    out["n_articulation"] = len(g.articulation_points())
    return out


def ar1_neff(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    rho = float((x[:-1] * x[1:]).sum() / (x * x).sum())
    n = len(x)
    return rho, n * (1 - rho) / (1 + rho)


def affected_set(panel: pd.DataFrame, pre_week: str, names: np.ndarray) -> set[str]:
    weeks = sorted(panel["time"].unique())
    i = weeks.index(pre_week)
    post = weeks[min(i + AFFECT_WINDOW_WEEKS, len(weeks) - 1)]

    def strength_by_name(wk):
        d = panel[panel["time"] == wk]
        d = d[(d["size"] > 0) & (d["source"] != d["target"])]
        a = d.groupby("source")["size"].sum()
        b = d.groupby("target")["size"].sum()
        return a.add(b, fill_value=0)

    s0, s1 = strength_by_name(pre_week), strength_by_name(post)
    out = set()
    for nm in names:
        before = s0.get(nm, 0.0)
        after = s1.get(nm, 0.0)
        if before > 0 and after <= (1 - AFFECT_DROP) * before:
            out.add(str(nm))
    return out


def save(results: list[dict], out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for res in results for r in res["rows"]]
    pd.DataFrame(rows).to_csv(out / "curvas.csv", index=False)
    pd.DataFrame([res["meta"] | {"rho_trace": json.dumps(res["meta"].get("rho_trace", []))}
                  for res in results]).drop(columns=["labels"]).assign(
        labels=[";".join(res["meta"]["labels"]) for res in results]).to_csv(out / "semanas.csv", index=False)
    (out / "conjuntos.json").write_text(
        json.dumps({res["meta"]["week"]: res["seqs"] for res in results}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    # Jaccard por semana y k
    jrows = []
    for res in results:
        seqs = res["seqs"]
        names = sorted(seqs)
        for k in K_LIST:
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    if len(seqs[a]) >= k and len(seqs[b]) >= k:
                        jrows.append(dict(week=res["meta"]["week"], k=k, a=a, b=b,
                                          jaccard=jaccard(seqs[a][:k], seqs[b][:k])))
    pd.DataFrame(jrows).to_csv(out / "jaccard.csv", index=False)
