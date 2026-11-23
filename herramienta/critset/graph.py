"""Construcción de una instantánea G_t, su proyección no dirigida y la componente gigante C_t.

Convenciones (entrega 2, Sec. V-A):
- Se agregan registros repetidos por par dirigido sumando ``size``.
- Se rechazan pesos no finitos o no positivos y se eliminan bucles.
- V_t son los extremos de los registros de la semana.
- {u,v} es arista de la proyección si w(u,v)+w(v,u) > 0.
- C_t es la componente de mayor cardinalidad; los empates se resuelven por la
  lista ordenada de identificadores (gana la lexicográficamente menor).
- Los vértices se indexan en orden lexicográfico de identificador, de modo que
  "desempate lexicográfico" equivale a "menor índice".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


@dataclass
class Snapshot:
    week: str
    names: np.ndarray            # identificadores de C_t, orden lexicográfico
    W: sp.csr_matrix             # exposición dirigida cruda en G_t[C_t], W[u,v] = w(u,v)
    U: sp.csr_matrix             # adyacencia binaria simétrica de H_t = proyección[C_t]
    n_vertices: int              # |V_t|
    n_arcs: int                  # |E_t|
    n_components: int            # componentes de la proyección completa
    W_total_all: float           # exposición total de G_t (todas las componentes)
    # arrays auxiliares de aristas dirigidas en C_t
    src: np.ndarray = field(repr=False, default=None)
    dst: np.ndarray = field(repr=False, default=None)
    w: np.ndarray = field(repr=False, default=None)
    # arrays auxiliares de la proyección (cada arista no dirigida una vez, u<v)
    usrc: np.ndarray = field(repr=False, default=None)
    udst: np.ndarray = field(repr=False, default=None)

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def W0(self) -> float:
        return float(self.w.sum())

    def index_of(self, ids) -> np.ndarray:
        pos = {s: i for i, s in enumerate(self.names)}
        return np.array([pos[s] for s in ids if s in pos], dtype=np.int64)


def clean_week(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y agrega los registros de una semana."""
    d = df[["source", "target", "size"]].copy()
    d["source"] = d["source"].astype(str)
    d["target"] = d["target"].astype(str)
    d["size"] = pd.to_numeric(d["size"], errors="coerce")
    d = d[np.isfinite(d["size"]) & (d["size"] > 0)]
    d = d[d["source"] != d["target"]]
    d = d.groupby(["source", "target"], as_index=False, sort=True)["size"].sum()
    return d


def build_snapshot(df_week: pd.DataFrame, week: str) -> Snapshot:
    d = clean_week(df_week)
    if d.empty:
        raise ValueError(f"La semana {week} no tiene aristas válidas.")
    names_all = np.array(sorted(set(d["source"]) | set(d["target"])))
    pos = {s: i for i, s in enumerate(names_all)}
    s_idx = d["source"].map(pos).to_numpy()
    t_idx = d["target"].map(pos).to_numpy()
    w_all = d["size"].to_numpy(dtype=float)
    N = len(names_all)

    # proyección no dirigida
    A = sp.coo_matrix((np.ones_like(w_all), (s_idx, t_idx)), shape=(N, N)).tocsr()
    Usym = ((A + A.T) > 0).astype(np.int8)
    ncomp, labels = connected_components(Usym, directed=False)

    # componente gigante con desempate por lista ordenada de identificadores
    sizes = np.bincount(labels)
    best = np.flatnonzero(sizes == sizes.max())
    if len(best) > 1:
        keyed = sorted(best, key=lambda c: list(names_all[labels == c]))
        giant = keyed[0]
    else:
        giant = best[0]
    keep = np.flatnonzero(labels == giant)  # índices crecientes = orden lexicográfico

    remap = -np.ones(N, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    m = (remap[s_idx] >= 0) & (remap[t_idx] >= 0)
    src, dst, w = remap[s_idx[m]], remap[t_idx[m]], w_all[m]
    n = len(keep)
    W = sp.csr_matrix((w, (src, dst)), shape=(n, n))
    U = Usym[keep][:, keep].tocsr()
    U.data[:] = 1
    uc = sp.triu(U, k=1).tocoo()

    return Snapshot(
        week=week,
        names=names_all[keep],
        W=W,
        U=U,
        n_vertices=N,
        n_arcs=len(d),
        n_components=int(ncomp),
        W_total_all=float(w_all.sum()),
        src=src,
        dst=dst,
        w=w,
        usrc=uc.row.astype(np.int64),
        udst=uc.col.astype(np.int64),
    )


def load_panel(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"time": str, "source": str, "target": str})
    missing = {"time", "source", "target", "size"} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el panel: {sorted(missing)}")
    return df


def two_core_names(snap: Snapshot) -> np.ndarray:
    """Identificadores del 2-núcleo de H_t = proyección[C_t] (poda iterativa de hojas).

    El 2-núcleo de un grafo conexo con algún ciclo es conexo: quitar una hoja no
    desconecta nada, así que la poda conserva la conexidad."""
    deg = np.diff(snap.U.indptr).astype(np.int64)
    alive = np.ones(snap.n, dtype=bool)
    stack = list(np.flatnonzero(deg <= 1))
    while stack:
        v = stack.pop()
        if not alive[v]:
            continue
        alive[v] = False
        for u in snap.U.indices[snap.U.indptr[v]:snap.U.indptr[v + 1]]:
            if alive[u]:
                deg[u] -= 1
                if deg[u] == 1:
                    stack.append(u)
    return snap.names[alive]


def snapshot_from_panel(panel: pd.DataFrame, week: str, exclude=(), core2: bool = False,
                        min_weight: float = 0.0) -> Snapshot:
    """Instantánea de una semana. Variantes de sensibilidad:
    ``exclude``: identificadores retirados del panel antes de construir G_t;
    ``core2``: restringe C_t a su 2-núcleo (sin hojas);
    ``min_weight``: descarta aristas dirigidas agregadas con peso menor al umbral."""
    dw = panel[panel["time"] == week]
    if dw.empty:
        raise ValueError(f"La semana {week} no está en el panel.")
    if len(exclude):
        dw = dw[~dw["source"].isin(exclude) & ~dw["target"].isin(exclude)]
    if min_weight > 0:
        d = clean_week(dw)
        dw = d[d["size"] >= min_weight]
    snap = build_snapshot(dw, week)
    if core2:
        core = set(two_core_names(snap))
        d = clean_week(dw)
        d = d[d["source"].isin(core) & d["target"].isin(core)]
        full = snap
        snap = build_snapshot(d, week)
        # conservar las dimensiones de G_t completo para el reporte
        snap.n_vertices, snap.n_arcs = full.n_vertices, full.n_arcs
        snap.n_components, snap.W_total_all = full.n_components, full.W_total_all
    return snap
