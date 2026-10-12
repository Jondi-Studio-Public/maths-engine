"""Exam question on a small simplicial complex given by its maximal faces: edges, Euler characteristic, Betti numbers.

The stem gives the vertices 1, ..., V and the maximal faces: some filled triangles, some extra edges that lie in no
triangle, and any vertex that lies in nothing else. Four parts:

(a) the number of edges E (a shared edge counts once)              number (whole), 1 mark
(b) the Euler characteristic V - E + F                             number (whole), 1 mark, follow_through V - part_a + F
(c) b0, the number of connected components                          number (whole), 1 mark
(d) b1, given that X has no 2-dimensional hole (b2 = 0)             number (whole), 2 marks, follow_through part_c - part_b

The draw keeps only complexes whose b2 is 0, so the stated assumption is true. Every key is checked a second way:
the Betti numbers are computed from the ranks of the boundary matrices over the rationals
(b_k = dim C_k - rank d_k - rank d_(k+1)) and must agree with the Euler relation b0 - b1 + b2 = V - E + F.
"""

from __future__ import annotations

import random
from fractions import Fraction
from itertools import combinations

from ..core import attempts, check, generator, phrase
from .exd_x_r2h_support import build_part

_VERTICES = (8, 9, 10, 11, 12, 13)

_STEM = (
    "A simplicial complex X has vertices 1, 2, ..., {v}. Its maximal faces are {faces}. Every face of a maximal face "
    "also belongs to X, and the triangles are filled in.",
    "Let X be the simplicial complex on the vertices 1 to {v} whose maximal faces are {faces}. Triangles are "
    "filled, and every edge and vertex of a listed face belongs to X.",
    "The simplicial complex X is built on the vertices 1, ..., {v}. Its maximal faces are {faces}; each listed "
    "triangle is a filled 2-simplex and all its edges and vertices are in X.",
)
_PROMPT_A = (
    "How many edges does X have? An edge shared by two faces is counted once.",
    "Find the number of edges (1-simplices) of X, counting a shared edge once.",
)
_PROMPT_B = (
    "Hence find the Euler characteristic of X.",
    "Use your count of edges to find the Euler characteristic of X, that is, vertices - edges + triangles.",
)
_PROMPT_C = (
    "Find b0, the number of connected components of X.",
    "How many connected components does X have (the Betti number b0)?",
)
_PROMPT_D = (
    "Given that X has no 2-dimensional hole (b2 = 0), use the Euler characteristic to find b1.",
    "Assuming b2 = 0, find b1, the number of independent 1-dimensional holes of X.",
)


def _rank(rows: list) -> int:
    """Rank of a matrix of Fractions by Gaussian elimination."""
    m = [list(map(Fraction, r)) for r in rows]
    rank = 0
    for col in range(len(m[0]) if m else 0):
        pivot = next((i for i in range(rank, len(m)) if m[i][col] != 0), None)
        if pivot is None:
            continue
        m[rank], m[pivot] = m[pivot], m[rank]
        for i in range(len(m)):
            if i != rank and m[i][col] != 0:
                factor = m[i][col] / m[rank][col]
                m[i] = [a - factor * b for a, b in zip(m[i], m[rank])]
        rank += 1
    return rank


def _betti(vertices: list, edges: list, triangles: list) -> tuple[int, int, int]:
    """b0, b1, b2 from the ranks of the boundary matrices."""
    d1 = [[(-1 if v == e[0] else 1 if v == e[1] else 0) for e in edges] for v in vertices]
    rank1 = _rank(d1) if edges else 0
    d2 = []
    for e in edges:
        row = []
        for t in triangles:
            a, b, c = t
            if e == (b, c):
                row.append(1)
            elif e == (a, c):
                row.append(-1)
            elif e == (a, b):
                row.append(1)
            else:
                row.append(0)
        d2.append(row)
    rank2 = _rank(d2) if edges and triangles else 0
    return len(vertices) - rank1, len(edges) - rank1 - rank2, len(triangles) - rank2


def _components(vertices: list, edges: list) -> int:
    parent = {v: v for v in vertices}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b in edges:
        parent[find(a)] = find(b)
    return len({find(v) for v in vertices})


def _build(rng: random.Random) -> dict:
    v = rng.choice(_VERTICES)
    vertices = list(range(1, v + 1))
    for _ in attempts(20000, "a complex with b2 = 0 and a visible hole"):
        shuffled = rng.sample(vertices, v)
        cuts = sorted(rng.sample(range(2, v - 1), rng.choice((0, 0, 0, 1, 1, 2)))) if v > 4 else []
        groups = [shuffled[i:j] for i, j in zip([0, *cuts], [*cuts, v])]
        big = [g for g in groups if len(g) >= 3]
        if not big:
            continue
        triangles = []
        for _t in range(rng.randint(3, 8)):
            tri = tuple(sorted(rng.sample(rng.choice(big), 3)))
            if tri not in triangles:
                triangles.append(tri)
        tri_edges = {e for t in triangles for e in combinations(t, 2)}
        extra = []
        for _e in range(rng.randint(3, 10)):
            edge = tuple(sorted(rng.sample(rng.choice([g for g in groups if len(g) >= 2]), 2)))
            if edge not in tri_edges and edge not in extra:
                extra.append(edge)
        edges = sorted(tri_edges | set(extra))
        used = {x for e in edges for x in e}
        isolated = [x for x in vertices if x not in used]
        b0, b1, b2 = _betti(vertices, edges, triangles)
        if b2 == 0 and 1 <= b1 <= 6 and b0 <= 3 and len(isolated) <= 1 and len(triangles) >= 3:
            break

    n_edges, n_tri = len(edges), len(triangles)
    chi = v - n_edges + n_tri
    check(chi == b0 - b1 + b2, "Euler characteristic disagrees with the Betti numbers")
    check(b0 == _components(vertices, edges), "b0 disagrees with the component count")
    check(b1 == b0 - chi, "b1 disagrees with the Euler relation")

    def face_text(face) -> str:
        return "{" + ", ".join(str(i) for i in face) + "}"

    maximal = [*triangles, *extra, *[(x,) for x in isolated]]
    rng.shuffle(maximal)
    faces_text = ", ".join(face_text(f) for f in maximal)
    stem = phrase(rng, _STEM, v=v, faces=faces_text)

    listed_edges = len(extra) + 3 * n_tri
    a_wrong = [
        (listed_edges, "double-counts-shared-edges"),
        (len(extra), "forgets-edges-of-triangles"),
        (n_edges + 1, "off-by-one-count"),
        (n_edges - 1, "off-by-one-count"),
        (n_edges + n_tri, "counts-triangles-as-edges"),
    ]
    b_wrong = [
        (v - n_edges, "forgets-the-triangles"),
        (v - n_edges - n_tri, "wrong-signs-in-euler-characteristic"),
        (n_edges - v - n_tri, "answer-sign-reversed"),
        (v + n_tri - listed_edges, "double-counts-shared-edges"),
        (v + n_edges - n_tri, "wrong-signs-in-euler-characteristic"),
    ]
    c_wrong = [
        (v, "counts-vertices-as-components"),
        (1, "assumes-connected"),
        (b0 + 1, "off-by-one-count"),
        (b0 - 1 if b0 > 1 else None, "off-by-one-count"),
        (len(triangles), "counts-triangles-as-components"),
        (b0 + 2, "off-by-one-count"),
        (b0 - len(isolated) if isolated else None, "ignores-isolated-vertex"),
    ]
    d_wrong = [
        (b0 + chi, "euler-relation-sign-slip"),
        (n_edges - v + 1, "assumes-connected"),
        (-chi, "drops-b0-from-euler-relation"),
        (b1 + 1, "off-by-one-count"),
        (b1 - 1 if b1 > 1 else None, "off-by-one-count"),
    ]
    parts = [
        build_part(
            phrase(rng, _PROMPT_A),
            n_edges,
            kind="number",
            marks=1,
            working=[
                f"The {n_tri} triangles give edges {sorted(tri_edges)} (shared edges once); "
                f"the extra listed edges are {extra}.",
                f"Total edges: {n_edges}.",
            ],
            wrong=a_wrong,
            sensible_range=(0, None),
        ),
        build_part(
            phrase(rng, _PROMPT_B),
            chi,
            kind="number",
            marks=1,
            working=[f"V - E + F = {v} - {n_edges} + {n_tri} = {chi}."],
            wrong=b_wrong,
            follow_through=f"{v} - part_a + {n_tri}",
        ),
        build_part(
            phrase(rng, _PROMPT_C),
            b0,
            kind="number",
            marks=1,
            working=[f"Joining vertices along edges gives {b0} connected piece(s)."],
            wrong=c_wrong,
            sensible_range=(0, None),
        ),
        build_part(
            phrase(rng, _PROMPT_D),
            b1,
            kind="number",
            marks=2,
            working=[f"chi = b0 - b1 + b2 with b2 = 0, so b1 = b0 - chi = {b0} - ({chi}) = {b1}."],
            wrong=d_wrong,
            follow_through="part_c - part_b",
        ),
    ]
    return dict(text=stem, latex=f"plain:{stem}", parts=parts)


@generator("exd_x_r2h_simplicial_euler_chain", level=19, presets={"exam": dict(tier="exam")})
def exd_x_r2h_simplicial_euler_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Edge count, Euler characteristic, components and first Betti number of a complex given by maximal faces."""
    check(tier == "exam", "exd_x_r2h_simplicial_euler_chain: only the exam tier exists")
    return _build(rng)
