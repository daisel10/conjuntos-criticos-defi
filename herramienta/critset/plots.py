"""Figuras para el documento (PDF vectorial, impresión en blanco y negro posible:
cada regla tiene color de la paleta categórica en orden fijo + marcador propio)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paleta categórica de referencia, slots 1..6 en orden fijo.
ORDER = ["fuerza", "intermediacion", "pagerank", "articulacion", "cds", "espectral"]
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MARK = dict(zip(ORDER, ["o", "s", "^", "D", "v", "P"]))
LABEL = {"fuerza": "Fuerza", "intermediacion": "Intermediación", "pagerank": "PageRank",
         "articulacion": "Articulación", "cds": "CDS", "espectral": "Espectral",
         "voraz": "Voraz (control)", "aleatorio": "Aleatorio", "tvl": "TVL", "sis": "SIS"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e2dc"
REF = {"voraz": dict(color="#3d3d3a", ls="--", marker="x"),
       "aleatorio": dict(color="#9a988f", ls=":", marker=".")}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.5, "lines.markersize": 4,
})
COLW = 3.5  # ancho de columna IEEE en pulgadas


def _style(rule):
    if rule in COLOR:
        return dict(color=COLOR[rule], marker=MARK[rule], ls="-")
    return REF[rule]


def fig_curvas(cur: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(2 * COLW, 2.3), sharex=True)
    for ax, col, title in [(axes[0], "L_est", "Pérdida estructural $L_{\\mathrm{est}}$"),
                           (axes[1], "L_pond", "Pérdida ponderada $L_{\\mathrm{pond}}$")]:
        for rule in ORDER + ["voraz", "aleatorio"]:
            g = cur[cur["rule"] == rule].groupby("k")[col]
            if g.ngroups == 0:
                continue
            med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            st = _style(rule)
            ax.plot(med.index, med.values, label=LABEL[rule], **st)
            if rule in COLOR:
                ax.fill_between(med.index, q1.values, q3.values, color=st["color"], alpha=0.10, lw=0)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.set_xlabel("Presupuesto $k$")
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Mediana entre semanas")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig_curvas.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_descomposicion(cur: pd.DataFrame, out: Path, k=10):
    d = cur[(cur["k"] == k) & cur["rule"].isin(ORDER + ["voraz", "aleatorio"])]
    med = d.groupby("rule")[["L_directa", "L_frag", "L_est"]].median()
    rules_ = [r for r in ORDER + ["voraz", "aleatorio"] if r in med.index]
    fig, axes = plt.subplots(1, 2, figsize=(2 * COLW, 2.1), gridspec_kw={"width_ratios": [1, 1]})
    y = np.arange(len(rules_))
    ax = axes[0]
    ax.barh(y, med.loc[rules_, "L_directa"], color="#2a78d6", height=0.6, label="$L_{\\mathrm{directa}}$")
    ax.barh(y, med.loc[rules_, "L_frag"], left=med.loc[rules_, "L_directa"], color="#eb6834",
            height=0.6, label="$L_{\\mathrm{frag}}$", edgecolor="white", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[r] for r in rules_])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Pérdida ponderada, $k={k}$ (mediana)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=7)
    ax = axes[1]
    ax.barh(y, med.loc[rules_, "L_frag"], color="#eb6834", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("$L_{\\mathrm{frag}}$ ampliada")
    for i, r in enumerate(rules_):
        ax.text(med.loc[r, "L_frag"], i, f" {med.loc[r, 'L_frag']:.4f}", va="center", fontsize=6.5, color=INK)
    ax.set_xlim(0, med["L_frag"].max() * 1.45 if med["L_frag"].max() > 0 else 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    fig.tight_layout()
    fig.savefig(out / "fig_descomposicion.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_jaccard(jac: pd.DataFrame, out: Path, k=10):
    names = ORDER + ["voraz"]
    d = jac[(jac["k"] == k) & jac["a"].isin(names) & jac["b"].isin(names)]
    m = d.groupby(["a", "b"])["jaccard"].mean()
    M = np.eye(len(names))
    for (a, b), v in m.items():
        i, j = names.index(a), names.index(b)
        M[i, j] = M[j, i] = v
    fig, ax = plt.subplots(figsize=(COLW, 2.9))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([LABEL[n].replace(" (control)", "") for n in names], rotation=45, ha="right")
    ax.set_yticklabels([LABEL[n].replace(" (control)", "") for n in names])
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"Jaccard medio, $k={k}$")
    fig.tight_layout()
    fig.savefig(out / "fig_jaccard.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_tiempos(cur: pd.DataFrame, sem: pd.DataFrame, out: Path):
    d = cur[cur["k"] == 1][["week", "rule", "seconds"]].merge(sem[["week", "C"]], on="week")
    fig, ax = plt.subplots(figsize=(COLW, 2.7))
    for rule in ORDER + ["voraz"]:
        g = d[d["rule"] == rule].sort_values("C")
        if g.empty:
            continue
        ax.plot(g["C"], g["seconds"].clip(lower=1e-3), label=LABEL[rule], **_style(rule))
    ax.set_yscale("log")
    ax.set_xlabel("$|C_t|$ (vértices de la componente gigante)")
    ax.set_ylabel("Segundos para $k_{\\max}=20$")
    ax.legend(fontsize=6, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              columnspacing=0.8, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(out / "fig_tiempos.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_caracterizacion(car: pd.DataFrame, out: Path, events: dict):
    c = car[car["week"] >= "2021-01-01"].copy()
    c["t"] = pd.to_datetime(c["week"])
    fig, axes = plt.subplots(2, 2, figsize=(2 * COLW, 3.0), sharex=True)
    panels = [("V", "$|V_t|$ y $|C_t|$"), ("ncomp", "Componentes conexas"),
              ("lambda2", "$\\lambda_2$ de $H_t$"), ("top10_strength_share", "Fuerza en los 10 mayores")]
    for ax, (col, title) in zip(axes.ravel(), panels):
        ax.plot(c["t"], c[col], color="#2a78d6", lw=1.2, label="$|V_t|$" if col == "V" else None)
        if col == "V":
            ax.plot(c["t"], c["C"], color="#eb6834", lw=1.2, label="$|C_t|$")
            ax.legend(fontsize=6.5)
        for ev, d in events.items():
            ax.axvline(pd.Timestamp(d), color=MUTED, lw=0.6, ls="--")
        ax.set_title(title, fontsize=8, color=INK)
    for ev, d in events.items():
        axes[0, 0].text(pd.Timestamp(d), axes[0, 0].get_ylim()[1], ev.upper(), fontsize=5.5,
                        color=MUTED, rotation=90, va="top", ha="right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "fig_caracterizacion.pdf", bbox_inches="tight")
    plt.close(fig)


def all_figures(res_dir: str, out_dir: str):
    from .experiment import EVENTS

    res, out = Path(res_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cur = pd.read_csv(res / "curvas.csv", dtype={"week": str})
    sem = pd.read_csv(res / "semanas.csv", dtype={"week": str})
    jac = pd.read_csv(res / "jaccard.csv", dtype={"week": str})
    fig_curvas(cur, out)
    fig_descomposicion(cur, out)
    fig_jaccard(jac, out)
    fig_tiempos(cur, sem, out)
    car_p = res / "caracterizacion.csv"
    if car_p.exists():
        fig_caracterizacion(pd.read_csv(car_p, dtype={"week": str}), out, EVENTS)


if __name__ == "__main__":
    import sys

    all_figures(sys.argv[1], sys.argv[2])
