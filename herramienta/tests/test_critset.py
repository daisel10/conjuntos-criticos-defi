import networkx as nx
import numpy as np
import pandas as pd
import pytest

from critset import rules
from critset.graph import build_snapshot
from critset.metrics import Evaluator


def random_snapshot(seed, n=60, p=0.05):
    rng = np.random.default_rng(seed)
    g = nx.gnp_random_graph(n, p, seed=seed, directed=True)
    # asegurar una componente grande: añadir un camino
    for i in range(n - 1):
        if rng.random() < 0.5:
            g.add_edge(i, i + 1)
    rows = [(f"p{u:03d}", f"p{v:03d}", float(rng.lognormal(10, 2))) for u, v in g.edges()]
    rows.append(("p000", "p000", 5.0))  # bucle: debe descartarse
    df = pd.DataFrame(rows, columns=["source", "target", "size"])
    return build_snapshot(df, "t")


def to_nx(snap, alive=None):
    g = nx.Graph()
    alive = np.ones(snap.n, bool) if alive is None else alive
    g.add_nodes_from(np.flatnonzero(alive))
    for a, b in zip(snap.usrc, snap.udst):
        if alive[a] and alive[b]:
            g.add_edge(int(a), int(b))
    return g


@pytest.mark.parametrize("seed", range(8))
def test_giant_connected_and_no_loops(seed):
    s = random_snapshot(seed)
    assert nx.is_connected(to_nx(s))
    assert (s.src != s.dst).all()


@pytest.mark.parametrize("seed", range(8))
def test_decomposition_identity(seed):
    s = random_snapshot(seed)
    ev = Evaluator(s)
    rng = np.random.default_rng(seed)
    for _ in range(10):
        S = rng.choice(s.n, size=rng.integers(1, 6), replace=False)
        d = ev.evaluate_set(S)
        assert d.L_directa >= -1e-12 and d.L_frag >= -1e-12
        assert abs(d.L_pond - (d.L_directa + d.L_frag)) < 1e-12
        # contra networkx
        alive = np.ones(s.n, bool)
        alive[S] = False
        comps = list(nx.connected_components(to_nx(s, alive)))
        assert d.R_est == pytest.approx(max(len(c) for c in comps) / s.n)
        wq = []
        for c in comps:
            m = np.isin(s.src, list(c)) & np.isin(s.dst, list(c))
            wq.append(s.w[m].sum())
        assert d.R_pond == pytest.approx(max(wq) / s.W0)


@pytest.mark.parametrize("seed", range(8))
def test_articulation_pieces_bruteforce(seed):
    s = random_snapshot(seed)
    g = to_nx(s)
    in_comp = np.ones(s.n, bool)
    res = rules._ap_max_piece(s.U.indptr, s.U.indices, in_comp, 0, s.n)
    assert set(res) == set(nx.articulation_points(g))
    for v, m in res.items():
        h = g.copy()
        h.remove_node(v)
        assert m == max(len(c) for c in nx.connected_components(h))


@pytest.mark.parametrize("seed", range(8))
def test_cds_valid(seed):
    s = random_snapshot(seed)
    B = rules.guha_khuller_cds(s)
    g = to_nx(s)
    assert nx.is_connected(g.subgraph(B))
    assert nx.is_dominating_set(g, B)


def test_pagerank_matches_networkx():
    s = random_snapshot(3)
    pr = rules.pagerank(s.W)
    g = nx.DiGraph()
    g.add_nodes_from(range(s.n))
    for a, b, w in zip(s.src, s.dst, s.w):
        g.add_edge(int(a), int(b), weight=float(w))
    ref = nx.pagerank(g, alpha=0.85, weight="weight", tol=1e-13, max_iter=5000)
    assert np.allclose(pr, [ref[i] for i in range(s.n)], atol=1e-8)


def test_spectral_exact_vs_bruteforce():
    s = random_snapshot(5, n=40, p=0.08)
    A = rules.log_matrix(s.W).toarray()
    seq = rules.rule_spectral(s, 3)
    alive = np.ones(s.n, bool)
    for v in seq:
        base = np.abs(np.linalg.eigvals(A[np.ix_(alive, alive)])).max()
        drops = {}
        for c in np.flatnonzero(alive):
            a2 = alive.copy(); a2[c] = False
            drops[c] = base - np.abs(np.linalg.eigvals(A[np.ix_(a2, a2)])).max()
        best = max(drops.values())
        assert drops[v] == pytest.approx(best, abs=1e-8)
        alive[v] = False


def test_greedy_is_monotone():
    s = random_snapshot(2)
    ev = Evaluator(s)
    seq = rules.rule_greedy(s, 5, ev=ev)
    vals = [ev.evaluate_set(seq[:k]).R_pond for k in range(1, 6)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))
