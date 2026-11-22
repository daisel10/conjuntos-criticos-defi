"""Interfaz de línea de comandos.

critset run         --panel CSV --week AAAA-MM-DD [--k 1 2 5 10 20] [--out DIR]
critset experiment  --panel CSV [--nodes CSV ...] [--out DIR] [--jobs N]
critset characterize --panel CSV [--out DIR] [--jobs N]
critset case        --panel CSV --results DIR
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, rules
from .experiment import (EVENTS, K_LIST, affected_set, ar1_neff, characterize_week,
                         load_tvl, run_week, save, select_weeks)
from .graph import load_panel, snapshot_from_panel
from .metrics import Evaluator


def cmd_run(a):
    panel = load_panel(a.panel)
    snap = snapshot_from_panel(panel, a.week)
    ev = Evaluator(snap)
    kmax = max(a.k)
    out = {"week": a.week, "V": snap.n_vertices, "C": snap.n, "ncomp": snap.n_components,
           "W0": snap.W0, "reglas": {}}
    todo = {n: f for n, f in rules.RULES.items() if not a.rules or n in a.rules}
    if a.greedy:
        todo["voraz"] = rules.rule_greedy
    for name, fn in todo.items():
        t0 = time.perf_counter()
        seq = fn(snap, kmax, ev=ev)
        dt = time.perf_counter() - t0
        curves = []
        for k in a.k:
            if k >= snap.n:
                continue
            d = ev.evaluate_set(seq[:k])
            curves.append(dict(k=k, R_est=d.R_est, R_pond=d.R_pond, L_est=d.L_est,
                               L_pond=d.L_pond, L_directa=d.L_directa, L_frag=d.L_frag))
        out["reglas"][name] = {"seleccion": [str(snap.names[i]) for i in seq],
                               "curvas": curves, "segundos": dt}
        print(f"{name:15s} {dt:7.2f}s  " + "  ".join(
            f"k={c['k']}: Lest={c['L_est']:.3f} Lpond={c['L_pond']:.3f}" for c in curves[-2:]),
            file=sys.stderr)
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f"critset_{a.week}.json").write_text(text, encoding="utf-8")
        rows = [dict(regla=n, **c) for n, r in out["reglas"].items() for c in r["curvas"]]
        pd.DataFrame(rows).to_csv(Path(a.out) / f"curvas_{a.week}.csv", index=False)
    else:
        print(text)


def cmd_experiment(a):
    panel = pd.read_csv(a.panel, usecols=["time"], dtype={"time": str})
    sel = select_weeks(sorted(panel["time"].unique()))
    if a.weeks:
        sel = {w: sel.get(w, ["manual"]) for w in a.weeks}
    tvl = load_tvl(a.nodes) if a.nodes else {}
    parts = Path(a.out) / "por_semana"
    parts.mkdir(parents=True, exist_ok=True)
    results = []
    for w in sel:  # reanudar: las semanas ya guardadas no se recalculan
        f = parts / f"{w}.json"
        if f.exists():
            results.append(json.loads(f.read_text(encoding="utf-8")))
    done = {r["meta"]["week"] for r in results}
    tasks = [(a.panel, w, labs, tvl.get(w), max(K_LIST), any(l.startswith("pre-") for l in labs),
              a.rules, tuple(a.exclude))
             for w, labs in sel.items() if w not in done]
    print(f"{len(sel)} semanas, {len(tasks)} por calcular: {[t[1] for t in tasks]}", file=sys.stderr)
    if tasks:
        with Pool(min(a.jobs, len(tasks))) as pool:
            for r in pool.imap_unordered(run_week, tasks):
                (parts / f"{r['meta']['week']}.json").write_text(
                    json.dumps(r, ensure_ascii=False, default=float), encoding="utf-8")
                print(f"  hecha {r['meta']['week']}", file=sys.stderr, flush=True)
                results.append(r)
    results.sort(key=lambda r: r["meta"]["week"])
    save(results, Path(a.out))


def cmd_characterize(a):
    panel = pd.read_csv(a.panel, usecols=["time"], dtype={"time": str})
    weeks = sorted(panel["time"].unique())
    with Pool(a.jobs) as pool:
        rows = pool.map(characterize_week, [(a.panel, w) for w in weeks])
    df = pd.DataFrame(rows).sort_values("week")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / "caracterizacion.csv", index=False)
    m = df[df["week"] >= "2021-01-01"]
    diag = []
    for col in ["lambda2", "ncomp", "giant_frac", "top10_strength_share"]:
        x = m[col].to_numpy(dtype=float)
        if col == "ncomp":
            x = x / m["V"].to_numpy(dtype=float)
            col = "ncomp/V"
        rho, neff = ar1_neff(x)
        diag.append(dict(serie=col, n=len(x), rho1=rho, n_ef=neff))
    pd.DataFrame(diag).to_csv(Path(a.out) / "autocorrelacion.csv", index=False)


def cmd_case(a):
    panel = load_panel(a.panel)
    res = Path(a.results)
    sets = json.loads((res / "conjuntos.json").read_text(encoding="utf-8"))
    weeks = pd.read_csv(res / "semanas.csv", dtype={"week": str})
    rows = []
    for ev, d in EVENTS.items():
        pre = weeks[weeks["labels"].str.contains(f"pre-{ev}")]["week"]
        if pre.empty:
            continue
        wk = pre.iloc[0]
        snap = snapshot_from_panel(panel, wk)
        aff = affected_set(panel, wk, snap.names)
        base = len(aff) / snap.n
        for rule, seq in sets[wk].items():
            for k in K_LIST:
                if len(seq) < k:
                    continue
                hit = sum(s in aff for s in seq[:k])
                rows.append(dict(evento=ev, semana=wk, regla=rule, k=k, aciertos=hit,
                                 precision=hit / k, tasa_base=base, n_afectados=len(aff), C=snap.n))
        (res / f"afectados_{ev}.txt").write_text("\n".join(sorted(aff)), encoding="utf-8")
    pd.DataFrame(rows).to_csv(res / "estudio_caso.csv", index=False)


def cmd_tvl(a):
    """Análisis exploratorio: TVL y SIS con el TVL faltante en cero (desvío declarado)."""
    panel = load_panel(a.panel)
    tvl = load_tvl(a.nodes)
    res = Path(a.results)
    weeks = pd.read_csv(res / "semanas.csv", dtype={"week": str})
    curves = pd.read_csv(res / "curvas.csv", dtype={"week": str})
    rows = []
    for wk in weeks["week"]:
        if wk not in tvl:
            continue
        snap = snapshot_from_panel(panel, wk)
        ev = Evaluator(snap)
        tw = tvl[wk]
        has = np.array([s in tw for s in snap.names])
        s_in = rules.strength(snap)
        cov_w = float(s_in[has].sum() / s_in.sum())
        for name, fn in (("tvl", rules.rule_tvl), ("sis", rules.rule_sis)):
            seq = fn(snap, max(K_LIST), tvl=tw, fill_missing=True)
            for k in K_LIST:
                d = ev.evaluate_set(seq[:k])
                rows.append(dict(week=wk, rule=name, k=k, L_est=d.L_est, L_pond=d.L_pond,
                                 L_directa=d.L_directa, L_frag=d.L_frag,
                                 cobertura=float(has.mean()), cobertura_fuerza=cov_w))
    df = pd.DataFrame(rows)
    ref = curves[curves["week"].isin(df["week"].unique())][
        ["week", "rule", "k", "L_est", "L_pond", "L_directa", "L_frag"]]
    pd.concat([df, ref]).sort_values(["week", "k", "rule"]).to_csv(
        res / "tvl_exploratorio.csv", index=False)
    print(df.groupby(["week"])[["cobertura", "cobertura_fuerza"]].first(), file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(prog="critset", description="Conjuntos críticos en el grafo de exposición DeFi")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Selecciona conjuntos y curvas de daño para una instantánea")
    r.add_argument("--panel", required=True, help="CSV con columnas time,source,target,size")
    r.add_argument("--week", required=True)
    r.add_argument("--k", type=int, nargs="+", default=K_LIST)
    r.add_argument("--rules", nargs="*", choices=list(rules.RULES), default=None)
    r.add_argument("--greedy", action="store_true", help="incluir el control voraz (lento)")
    r.add_argument("--out", default=None)
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("experiment", help="Protocolo experimental completo")
    e.add_argument("--panel", required=True)
    e.add_argument("--nodes", nargs="*", default=[], help="CSV de nodos con TVL (time,id,size,...)")
    e.add_argument("--weeks", nargs="*", default=None)
    e.add_argument("--out", default="resultados")
    e.add_argument("--jobs", type=int, default=4)
    e.add_argument("--rules", nargs="*", default=None,
                   help="subconjunto de reglas y controles (por omisión, todos)")
    e.add_argument("--exclude", nargs="*", default=[],
                   help="identificadores a quitar del panel antes de construir C_t")
    e.set_defaults(fn=cmd_experiment)

    c = sub.add_parser("characterize", help="Invariantes de conectividad de todas las semanas")
    c.add_argument("--panel", required=True)
    c.add_argument("--out", default="resultados")
    c.add_argument("--jobs", type=int, default=4)
    c.set_defaults(fn=cmd_characterize)

    s = sub.add_parser("case", help="Estudio de caso de los tres episodios")
    s.add_argument("--panel", required=True)
    s.add_argument("--results", default="resultados")
    s.set_defaults(fn=cmd_case)

    t = sub.add_parser("tvl", help="Control exploratorio por TVL y SIS (faltantes en cero)")
    t.add_argument("--panel", required=True)
    t.add_argument("--nodes", nargs="+", required=True)
    t.add_argument("--results", default="resultados")
    t.set_defaults(fn=cmd_tvl)

    f = sub.add_parser("plots", help="Figuras PDF a partir de los resultados")
    f.add_argument("--results", default="resultados")
    f.add_argument("--out", default="figuras")
    f.set_defaults(fn=lambda a: __import__("critset.plots", fromlist=["x"]).all_figures(a.results, a.out))

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
