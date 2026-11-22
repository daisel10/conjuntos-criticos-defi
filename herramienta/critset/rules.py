"""Las seis reglas de selección, el control voraz y los controles por tamaño.

Cada regla devuelve una secuencia ordenada de índices de C_t; el conjunto
evaluado para presupuesto k es su prefijo de longitud k. Los índices siguen el
orden lexicográfico de identificadores, así que "desempate lexicográfico" es
"menor índice". Especificación: entrega 2, Sec. VI-A.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from .graph import Snapshot
from .metrics import Evaluator

TOL_REL = 1e-8
TOL_ABS = 1e-10


def _rank(scores: np.ndarray, k: int) -> list[int]:
    """Orden descendente por puntaje, empates por índice."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    return [int(i) for i in order[:k]]


# ---------------------------------------------------------------- 1. fuerza
def strength(snap: Snapshot) -> np.ndarray:
    W = snap.W
    return np.asarray(W.sum(axis=0)).ravel() + np.asarray(W.sum(axis=1)).ravel()


def rule_strength(snap: Snapshot, k: int, **_) -> list[int]:
    return _rank(strength(snap), k)


# ---------------------------------------------------------- 2. intermediación
def _igraph(n: int, us: np.ndarray, ud: np.ndarray):
    import igraph as ig

    return ig.Graph(n=n, edges=list(zip(us.tolist(), ud.tolist())), directed=False)


def betweenness_unweighted(n: int, us: np.ndarray, ud: np.ndarray) -> np.ndarray:
    g = _igraph(n, us, ud)
    return np.asarray(g.betweenness(directed=False), dtype=float)


def rule_betweenness(snap: Snapshot, k: int, **_) -> list[int]:
    return _rank(betweenness_unweighted(snap.n, snap.usrc, snap.udst), k)


# ------------------------------------------------------------------ 3. PageRank
def pagerank(W: sp.csr_matrix, alpha=0.85, tol=1e-10, max_iter=1000) -> np.ndarray:
    n = W.shape[0]
    out = np.asarray(W.sum(axis=1)).ravel()
    dangling = out == 0
    inv = np.zeros(n)
    inv[~dangling] = 1.0 / out[~dangling]
    P = sp.diags(inv) @ W  # filas estocásticas (salvo colgantes)
    PT = P.T.tocsr()
    x = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        x_new = alpha * (PT @ x + x[dangling].sum() / n) + (1 - alpha) / n
        if np.abs(x_new - x).sum() < tol:
            return x_new
        x = x_new
    raise RuntimeError("PageRank no convergió en 1000 iteraciones; no se emite ranking.")


def rule_pagerank(snap: Snapshot, k: int, **_) -> list[int]:
    return _rank(pagerank(snap.W), k)


# --------------------------------------------------- 4. articulación adaptativa
def _ap_max_piece(indptr, indices, in_comp: np.ndarray, root: int, comp_size: int) -> dict[int, int]:
    """Para cada punto de articulación v de la componente, tamaño de la mayor
    pieza que queda al suprimirlo (Tarjan iterativo con tamaños de subárbol)."""
    n = len(in_comp)
    disc = np.full(n, -1, dtype=np.int64)
    low = np.zeros(n, dtype=np.int64)
    parent = np.full(n, -1, dtype=np.int64)
    sub = np.zeros(n, dtype=np.int64)
    sep: dict[int, list[int]] = {}
    t = 0
    disc[root] = low[root] = t
    sub[root] = 1
    t += 1
    stack = [[root, indptr[root]]]
    while stack:
        top = stack[-1]
        v, p = top
        if p < indptr[v + 1]:
            top[1] = p + 1
            u = indices[p]
            if not in_comp[u]:
                continue
            if disc[u] == -1:
                parent[u] = v
                disc[u] = low[u] = t
                t += 1
                sub[u] = 1
                stack.append([u, indptr[u]])
            elif u != parent[v]:
                if disc[u] < low[v]:
                    low[v] = disc[u]
        else:
            stack.pop()
            pv = parent[v]
            if pv != -1:
                if low[v] < low[pv]:
                    low[pv] = low[v]
                sub[pv] += sub[v]
                if low[v] >= disc[pv]:
                    sep.setdefault(int(pv), []).append(int(sub[v]))
    res: dict[int, int] = {}
    for v, pieces in sep.items():
        if v == root:
            if len(pieces) >= 2:
                res[v] = max(pieces)
        else:
            rest = comp_size - 1 - sum(pieces)
            res[v] = max(max(pieces), rest)
    return res


def rule_articulation(snap: Snapshot, k: int, ev: Evaluator | None = None, **_) -> list[int]:
    ev = ev or Evaluator(snap)
    indptr, indices = snap.U.indptr, snap.U.indices
    alive = np.ones(snap.n, dtype=bool)
    S: list[int] = []
    for _ in range(min(k, snap.n - 1)):
        lab = ev.components(alive)
        live = lab >= 0
        sizes = np.bincount(lab[live])
        # componente mayor; empate -> la que contiene el menor índice
        big = sizes.max()
        cands = np.flatnonzero(sizes == big)
        first_idx = {c: int(np.flatnonzero(lab == c)[0]) for c in cands}
        g = min(cands, key=lambda c: first_idx[c])
        in_comp = lab == g
        others = np.delete(sizes, g)
        second = int(others.max()) if len(others) else 0
        aps = _ap_max_piece(indptr, indices, in_comp, first_idx[g], int(big))
        if aps:
            v = min(aps, key=lambda a: (max(aps[a], second), a))
        else:
            nodes = np.flatnonzero(in_comp)
            pos = -np.ones(snap.n, dtype=np.int64)
            pos[nodes] = np.arange(len(nodes))
            m = in_comp[snap.usrc] & in_comp[snap.udst]
            b = betweenness_unweighted(len(nodes), pos[snap.usrc[m]], pos[snap.udst[m]])
            v = int(nodes[_rank(b, 1)[0]])
        S.append(int(v))
        alive[v] = False
    return S


# ------------------------------------------------------------------- 5. CDS
def guha_khuller_cds(snap: Snapshot) -> list[int]:
    """Algoritmo I modificado de Guha–Khuller (1998), Sec. 2: escaneo de un
    vértice gris o de un par adyacente gris–blanco, el que domine más vértices
    nuevos (empates: vértice simple antes que par, luego menor índice)."""
    n = snap.n
    indptr, indices = snap.U.indptr, snap.U.indices
    nbrs = [indices[indptr[i]:indptr[i + 1]] for i in range(n)]
    WHITE, GRAY, BLACK = 0, 1, 2
    color = np.zeros(n, dtype=np.int8)
    deg = np.diff(indptr)
    wdeg = deg.copy()  # vecinos blancos
    black: list[int] = []

    def whiten_out(x):  # x deja de ser blanco
        for y in nbrs[x]:
            wdeg[y] -= 1

    def make_black(x):
        if color[x] == WHITE:
            whiten_out(x)
        color[x] = BLACK
        black.append(int(x))
        for y in nbrs[x]:
            if color[y] == WHITE:
                color[y] = GRAY
                whiten_out(y)

    if n == 1:
        return [0]
    v0 = int(_rank(deg.astype(float), 1)[0])
    make_black(v0)
    n_white = int((color == WHITE).sum())
    while n_white > 0:
        best_y, best = -1, None  # best = ("s", u) o ("p", u, w)
        grays = np.flatnonzero((color == GRAY) & (wdeg > 0))
        # vértices simples primero (preferidos en empate)
        for u in grays:
            y = int(wdeg[u])
            if y > best_y:
                best_y, best = y, ("s", int(u))
        for u in grays:
            Wu = None
            wu = int(wdeg[u])
            for w in nbrs[u]:
                if color[w] != WHITE or wu + int(wdeg[w]) <= best_y:
                    continue
                if Wu is None:
                    Wu = {int(x) for x in nbrs[u] if color[x] == WHITE}
                y = len(Wu | {int(x) for x in nbrs[w] if color[x] == WHITE})
                if y > best_y:
                    best_y, best = y, ("p", int(u), int(w))
        if best is None:
            raise RuntimeError("CDS: no hay candidato; ¿grafo desconectado?")
        if best[0] == "s":
            make_black(best[1])
        else:
            make_black(best[1])
            make_black(best[2])
        n_white = int((color == WHITE).sum())
    return black


def greedy_pond(ev: Evaluator, k: int, candidates: np.ndarray | None = None,
                fallback_all: bool = True) -> list[int]:
    """Voraz: en cada paso, el vértice que maximiza la pérdida ponderada
    marginal (mínima R_pond); empates por menor índice."""
    n = ev.n
    alive = np.ones(n, dtype=bool)
    pool = np.arange(n) if candidates is None else np.unique(np.asarray(candidates))
    S: list[int] = []
    for _ in range(min(k, n - 1)):
        avail = pool[alive[pool]]
        if len(avail) == 0:
            if not fallback_all:
                break
            pool = np.arange(n)
            avail = pool[alive[pool]]
        best, best_key = None, None
        for c in avail:
            alive[c] = False
            d = ev.evaluate(alive)
            alive[c] = True
            key = (round(d.R_pond, 12), int(c))
            if best_key is None or key < best_key:
                best, best_key = int(c), key
        S.append(best)
        alive[best] = False
    return S


def rule_cds(snap: Snapshot, k: int, ev: Evaluator | None = None, info: dict | None = None, **_) -> list[int]:
    ev = ev or Evaluator(snap)
    B = guha_khuller_cds(snap)
    if info is not None:
        info["cds_size"] = len(B)
    return greedy_pond(ev, k, candidates=np.array(B), fallback_all=True)


def rule_greedy(snap: Snapshot, k: int, ev: Evaluator | None = None, **_) -> list[int]:
    return greedy_pond(ev or Evaluator(snap), k)


# --------------------------------------------------- 6. inmunización espectral
def _rho_dense(M: np.ndarray) -> float:
    if M.size == 0:
        return 0.0
    return float(np.abs(np.linalg.eigvals(M)).max())


def _rho(M: sp.csr_matrix) -> float:
    m = M.shape[0]
    if m == 0 or M.nnz == 0:
        return 0.0
    if m <= 600:
        return _rho_dense(M.toarray())
    try:
        vals, vecs = eigs(M, k=1, which="LM", v0=np.ones(m), tol=1e-12, maxiter=50 * m)
        lam = vals[0]
        x = vecs[:, 0]
        res = np.linalg.norm(M @ x - lam * x) / (abs(lam) * np.linalg.norm(x))
        if res < 1e-8:
            return float(abs(lam))
    except ArpackNoConvergence:
        pass
    return _rho_dense(M.toarray())


def _scc_radii(A: sp.csr_matrix, alive: np.ndarray):
    """Radio espectral por componente fuertemente conexa no trivial.
    rho(A) = max_SCC rho(A[SCC]) para matrices no negativas."""
    idx = np.flatnonzero(alive)
    sub = A[idx][:, idx].tocsr()
    ncomp, lab = connected_components(sub, directed=True, connection="strong")
    out = []
    sizes = np.bincount(lab)
    for c in np.flatnonzero(sizes > 1):
        nodes = idx[lab == c]
        out.append((nodes, _rho(A[nodes][:, nodes].tocsr())))
    return out


def log_matrix(W: sp.csr_matrix, w_ref: float = 1.0) -> sp.csr_matrix:
    A = W.copy().astype(float)
    A.data = np.log1p(A.data / w_ref)
    A.eliminate_zeros()
    return A


def rule_spectral(snap: Snapshot, k: int, raw: bool = False, info: dict | None = None, **_) -> list[int]:
    """Quita en cada paso el vértice de mayor caída del radio espectral de la
    matriz dirigida (log(1+w) por defecto, cruda si raw=True). Evaluación exacta:
    solo un vértice de una SCC que alcanza el máximo puede reducir rho; el
    resto tiene caída nula, y se compara igual."""
    A = snap.W.astype(float) if raw else log_matrix(snap.W)
    n = snap.n
    alive = np.ones(n, dtype=bool)
    S: list[int] = []
    trace = []
    for _ in range(min(k, n - 1)):
        radii = _scc_radii(A, alive)
        rho = max((r for _, r in radii), default=0.0)
        trace.append(rho)
        tol = max(TOL_ABS, TOL_REL * rho)
        best_drop, best = 0.0, None
        top = [(nodes, r) for nodes, r in radii if r >= rho - tol]
        rest = max((r for _, r in radii if r < rho - tol), default=0.0)
        if len(top) == 1 and rho > 0:
            nodes, _ = top[0]
            for c in nodes:  # índice creciente
                a2 = np.zeros(n, dtype=bool)
                a2[nodes] = True
                a2[c] = False
                # radio de la SCC sin c (puede partirse en varias SCC)
                sub_r = max((r for _, r in _scc_radii(A, a2)), default=0.0)
                new_rho = max(sub_r, rest)
                drop = rho - new_rho
                if drop > best_drop + max(TOL_ABS, TOL_REL * rho):
                    best_drop, best = drop, int(c)
        if best is None:  # caída nula para todos: menor índice vivo
            best = int(np.flatnonzero(alive)[0])
        S.append(best)
        alive[best] = False
    if info is not None:
        info["rho_trace"] = trace
    return S


# ------------------------------------------------------ controles por tamaño
def _tvl_values(snap: Snapshot, tvl: dict | None, fill_missing: bool) -> np.ndarray:
    """TVL por vértice de C_t. Con fill_missing, el faltante vale 0 (análisis exploratorio)."""
    if tvl is None:
        raise ValueError("TVL no disponible para esta semana")
    vals = np.array([tvl.get(s, np.nan) for s in snap.names], dtype=float)
    if np.isnan(vals).any():
        if not fill_missing:
            raise ValueError("TVL incompleto para C_t")
        vals = np.nan_to_num(vals, nan=0.0)
    return vals


def rule_tvl(snap: Snapshot, k: int, tvl: dict | None = None, fill_missing: bool = False,
             **_) -> list[int]:
    return _rank(_tvl_values(snap, tvl, fill_missing), k)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)


def tail5(W: sp.csr_matrix) -> np.ndarray:
    n = W.shape[0]
    out = np.zeros(n)
    for i in range(n):
        row = W.data[W.indptr[i]:W.indptr[i + 1]]
        if len(row) and row.sum() > 0:
            out[i] = np.sort(row)[-5:].sum() / row.sum()
    return out


def rule_sis(snap: Snapshot, k: int, tvl: dict | None = None, fill_missing: bool = False,
             **_) -> list[int]:
    vals = _tvl_values(snap, tvl, fill_missing)
    sis = (_minmax(pagerank(snap.W)) + tail5(snap.W) + _minmax(np.log1p(vals))) / 3.0
    return _rank(sis, k)


def rule_random(snap: Snapshot, k: int, seed: int = 0, **_) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(i) for i in rng.permutation(snap.n)[:k]]


RULES = {
    "fuerza": rule_strength,
    "intermediacion": rule_betweenness,
    "pagerank": rule_pagerank,
    "articulacion": rule_articulation,
    "cds": rule_cds,
    "espectral": rule_spectral,
}
CONTROLS = {
    "voraz": rule_greedy,
    "tvl": rule_tvl,
    "sis": rule_sis,
}
