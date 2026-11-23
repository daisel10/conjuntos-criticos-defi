"""Medidas de daño (entrega 2, ecs. medidas, retención y descomposición).

R_est(S)  = max_Q |Q| / |C_t|
R_pond(S) = max_Q W_t(Q) / W_0
L_j = 1 - R_j,   L_pond = L_directa + L_frag

Medidas añadidas en la revisión final (ninguna regla las optimiza):
F_rel(S)  = 1 - W*/W_{-S}   fragmentación relativa de la exposición que sobrevive
L_est     = k/|C_t| + L_single + L_multi
            (vértices quitados + separados como singletons + separados en
             fragmentos de dos o más vértices)

Hecho usado para acelerar: toda arista dirigida con ambos extremos vivos está,
por construcción de la proyección, dentro de una sola componente de H_t - S.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .graph import Snapshot

NAN = float("nan")


@dataclass(frozen=True)
class Damage:
    R_est: float
    R_pond: float
    L_directa: float
    L_frag: float
    F_rel: float = NAN
    L_single: float = NAN
    L_multi: float = NAN
    n_single: int = 0
    n_multi: int = 0

    @property
    def L_est(self) -> float:
        return 1.0 - self.R_est

    @property
    def L_pond(self) -> float:
        return 1.0 - self.R_pond

    def row(self) -> dict:
        return dict(R_est=self.R_est, R_pond=self.R_pond, L_est=self.L_est, L_pond=self.L_pond,
                    L_directa=self.L_directa, L_frag=self.L_frag, F_rel=self.F_rel,
                    L_single=self.L_single, L_multi=self.L_multi,
                    n_single=self.n_single, n_multi=self.n_multi)


class Evaluator:
    def __init__(self, snap: Snapshot):
        self.snap = snap
        self.n = snap.n
        self.W0 = snap.W0
        self.us, self.ud = snap.usrc, snap.udst
        self.src, self.dst, self.w = snap.src, snap.dst, snap.w

    def components(self, alive: np.ndarray):
        """Etiquetas de componentes de H_t - S (los removidos quedan con etiqueta -1)."""
        keep = alive[self.us] & alive[self.ud]
        r, c = self.us[keep], self.ud[keep]
        M = sp.coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(self.n, self.n))
        _, lab = connected_components(M, directed=False)
        lab = lab.astype(np.int64)
        lab[~alive] = -1
        return lab

    def evaluate(self, alive: np.ndarray) -> Damage:
        lab = self.components(alive)
        live = lab >= 0
        if not live.any():
            return Damage(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
        # renumerar etiquetas vivas
        uniq, inv = np.unique(lab[live], return_inverse=True)
        sizes = np.bincount(inv)
        D_est = sizes.max()
        rest = np.delete(sizes, int(np.argmax(sizes)))  # fragmentos separados de la gigante
        n_single, n_multi = int((rest == 1).sum()), int((rest >= 2).sum())
        L_single = n_single / self.n
        L_multi = float(rest[rest >= 2].sum()) / self.n
        keep = alive[self.src] & alive[self.dst]
        W_minus = float(self.w[keep].sum())
        if self.W0 <= 0:
            return Damage(D_est / self.n, NAN, NAN, NAN, NAN, L_single, L_multi, n_single, n_multi)
        lab_live = -np.ones(self.n, dtype=np.int64)
        lab_live[live] = inv
        comp_w = np.bincount(lab_live[self.src[keep]], weights=self.w[keep], minlength=len(sizes))
        W_star = float(comp_w.max()) if len(comp_w) else 0.0
        return Damage(
            R_est=D_est / self.n,
            R_pond=W_star / self.W0,
            L_directa=(self.W0 - W_minus) / self.W0,
            L_frag=(W_minus - W_star) / self.W0,
            F_rel=(1.0 - W_star / W_minus) if W_minus > 0 else 0.0,
            L_single=L_single,
            L_multi=L_multi,
            n_single=n_single,
            n_multi=n_multi,
        )

    def evaluate_set(self, S) -> Damage:
        alive = np.ones(self.n, dtype=bool)
        alive[np.asarray(list(S), dtype=np.int64)] = False
        return self.evaluate(alive)


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
