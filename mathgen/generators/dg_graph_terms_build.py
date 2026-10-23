"""Generates `dg_graph_terms` questions (easy, medium, hard) and the exam chain `dg_graph_terms_chain`.

``graph-terms-from-a-network`` (``financial-discrete``, answer ``number``): the words of graph theory (vertex, edge,
adjacent, degree, connected, component, subgraph, loop, multiple edge, simple graph) used on a network that is drawn,
and on a multigraph that cannot be drawn with the ``network`` figure (that kind refuses loops and repeated edges, so
a multigraph is given as a list of its edges, each written once per copy).

* easy: a simple connected network is drawn. Count its edges, the vertices adjacent to one vertex, the vertices not
  adjacent to it, or the greatest degree.
* medium: a network is drawn in pieces (or with a thick subgraph). Count its connected components, the fewest edges to
  add to make it connected, the vertices of a thick subgraph, or the edges of the network that the subgraph leaves out.
* hard: a multigraph is given as an edge list. Count its loops, its multiple (repeated) edges, the edges that remain
  once it is made a simple graph, or the degree of a vertex that carries a loop (a loop adds 2).
* exam (``dg_graph_terms_chain``): either a drawn chain (degree of a vertex, the edges of the network, then the edges
  and components left when that vertex is deleted) or a multigraph chain (loops, multiple edges, the simple graph's
  edges, a degree). Part (b) of the drawn chain is a typed set of adjacent vertices.

Every answer is found from the edge list the figure is drawn from; the reader recomputes it from the figure spec
alone, and degrees are also checked against the handshake sum.
"""

from __future__ import annotations

import re
from collections import Counter

from ..answers import Distractor
from ..core import attempts, check, generator, phrase
from .. import figures
from ._network_fig import graph_figure
from ._snet_kit import degrees, int_distractors, neighbours, norm
from ._w9_graph_kit import TooFewDistractors, enough, retry_draw

LETTERS = "ABCDEFGHIJ"
AREA = "graph-terms-from-a-network"

MISCONCEPTIONS = [
    ("dg-gt-adjacent-counts-itself", "Counts the vertex itself as one of its own adjacent vertices.", AREA),
    (
        "dg-gt-adjacent-all-others",
        "Takes every other vertex as adjacent, ignoring which are actually joined by an edge.",
        AREA,
    ),
    (
        "dg-gt-adjacent-for-nonadjacent",
        "Counts the adjacent vertices when the vertices NOT adjacent were asked for.",
        AREA,
    ),
    (
        "dg-gt-degree-of-wrong-vertex",
        "Reads the degree of a different vertex, or the second greatest degree, not the greatest.",
        AREA,
    ),
    (
        "dg-gt-edges-as-degree-sum",
        "Gives the sum of the degrees as the number of edges, so every edge is counted twice.",
        AREA,
    ),
    (
        "dg-gt-edges-miscount",
        "Miscounts the edges by one, usually missing one that is hidden behind a crossing or counted from both ends.",
        AREA,
    ),
    ("dg-gt-vertices-as-edges", "Gives the number of vertices where the number of edges was asked for.", AREA),
    (
        "dg-gt-components-isolated-only",
        "Counts only the isolated vertices as separate pieces and treats every larger piece as part of one.",
        AREA,
    ),
    (
        "dg-gt-components-as-edges-needed",
        "Gives the number of components as the number of edges to add, forgetting that joining c pieces takes c - 1 edges.",
        AREA,
    ),
    (
        "dg-gt-components-edges-per-vertex",
        "Adds one edge for every vertex that is not in the largest piece, instead of one edge per piece.",
        AREA,
    ),
    ("dg-gt-components-miscount", "Miscounts the separate pieces of the network by one.", AREA),
    (
        "dg-gt-subgraph-all-vertices",
        "Counts every vertex of the network in the subgraph, not only the ends of the thick edges.",
        AREA,
    ),
    (
        "dg-gt-subgraph-counts-its-own-edges",
        "Gives the edges of the subgraph when the edges of the network that it leaves out were asked for.",
        AREA,
    ),
    (
        "dg-gt-subgraph-vertices-of-network",
        "Gives the number of vertices of the whole network where the subgraph's vertices were asked for.",
        AREA,
    ),
    (
        "dg-gt-loop-counted-once",
        "Counts a loop once towards the degree of its vertex; a loop leaves the vertex at both of its ends, so it adds 2.",
        AREA,
    ),
    ("dg-gt-loop-not-counted", "Ignores the loop when finding the degree of its vertex.", AREA),
    (
        "dg-gt-multiple-all-copies",
        "Counts every copy of a repeated edge as a multiple edge, instead of only the extra copies.",
        AREA,
    ),
    ("dg-gt-loops-as-multiple", "Counts the loops as multiple edges, or the multiple edges as loops.", AREA),
    (
        "dg-gt-simple-keeps-repeats",
        "Removes the loops but leaves the repeated edges when making the graph simple.",
        AREA,
    ),
    ("dg-gt-simple-keeps-loops", "Removes the repeated edges but leaves the loops when making the graph simple.", AREA),
    ("dg-gt-delete-vertex-keeps-edges", "Deletes a vertex but keeps the edges that were at it.", AREA),
    (
        "dg-gt-delete-vertex-uses-sum",
        "Takes away the number of vertices adjacent to the deleted vertex twice, once for each end of its edges.",
        AREA,
    ),
]


# ----------------------------------------------------------------------------------------------- graphs


def components(n, edges):
    """The connected components of the graph on ``0..n-1`` as sorted tuples of vertices."""
    nbr = neighbours(n, edges)
    seen, out = set(), []
    for start in range(n):
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            v = stack.pop()
            if v in comp:
                continue
            comp.add(v)
            stack.extend(nbr[v] - comp)
        seen |= comp
        out.append(tuple(sorted(comp)))
    return out


def _tree_plus(rng, vertices, extra):
    """A connected simple graph on ``vertices`` (a random tree plus ``extra`` edges)."""
    vs = list(vertices)
    rng.shuffle(vs)
    edges = {norm(vs[i], vs[rng.randrange(i)]) for i in range(1, len(vs))}
    rest = [norm(a, b) for i, a in enumerate(vs) for b in vs[i + 1 :] if norm(a, b) not in edges]
    rng.shuffle(rest)
    edges.update(rest[: max(0, extra)])
    return edges


def _connected(rng, n, m):
    return sorted(_tree_plus(rng, range(n), m - (n - 1)))


def _in_pieces(rng, n):
    """A simple graph on ``n`` vertices in two or three pieces (one may be a lone vertex), and its component sizes."""
    k = rng.choice((2, 2, 3))
    while True:
        cuts = sorted(rng.sample(range(1, n), k - 1))
        sizes = [b - a for a, b in zip([0, *cuts], [*cuts, n])]
        if sorted(sizes)[-1] >= 3:
            break
    order = list(range(n))
    rng.shuffle(order)
    edges, start = set(), 0
    for size in sizes:
        group = order[start : start + size]
        start += size
        if size > 1:
            edges |= _tree_plus(rng, group, rng.randint(0, 2 if size >= 4 else 1))
    return sorted(edges)


def _figure(n, edges, highlight=(), alt=None):
    return graph_figure(
        n,
        edges,
        highlight_edges=highlight,
        alt=alt or f"A network of {n} vertices labelled {LETTERS[0]} to {LETTERS[n - 1]} joined by straight edges.",
    )


def _draw(n, edges, highlight=()):
    try:
        return _figure(n, edges, highlight)
    except ValueError as exc:  # an illegible layout: the caller redraws
        raise TooFewDistractors(str(exc)) from exc


def _by_matrix(n, edges):
    import sympy

    m = sympy.zeros(n, n)
    for u, v in edges:
        m[u, v] = m[v, u] = 1
    return [sum(m.row(i)) for i in range(n)]


def _checked_degrees(n, edges):
    deg = degrees(n, edges)
    check(deg == _by_matrix(n, edges), "degrees by counting and by the adjacency matrix differ")
    check(sum(deg) == 2 * len(edges), "the degrees do not add to twice the edges")
    return deg


def _ints(answer, candidates, n=3, *, lo=0):
    """Tagged integer distractors; a candidate that is None or below ``lo`` is skipped."""
    return int_distractors(answer, [c for c in candidates if c[0] is not None and c[0] >= lo], n, lo=lo)


def _q(text, figure, answer, working, distractors):
    enough(distractors)
    return dict(
        text=text,
        figure=figure,
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=distractors,
    )


def _hi(edges, thick):
    return [e for e in edges if e in thick]


# ----------------------------------------------------------------------------------------------- easy

_ADJ = (
    "In the network drawn, how many vertices are adjacent to vertex {x}? (Two vertices are adjacent when one edge joins them.)",
    "Two vertices are adjacent if an edge joins them. How many vertices of the drawn network are adjacent to vertex {x}?",
)
_NONADJ = (
    "How many vertices of the drawn network are not adjacent to vertex {x}? Do not count {x} itself.",
    "Vertex {x} is adjacent to some of the other vertices of the drawn network. How many of the other vertices is it not adjacent to?",
)
_EDGES = (
    "How many edges does the network drawn have?",
    "Count the edges of the network drawn.",
)
_MAXDEG = (
    "The degree of a vertex is the number of edges at it. What is the greatest degree of any vertex of the network drawn?",
    "What is the greatest degree among the vertices of the drawn network? (The degree of a vertex is the number of edges at it.)",
)


def _easy(rng):
    for _ in attempts(300, "an easy graph-terms network"):
        n = rng.randint(5, 7)
        edges = _connected(rng, n, rng.randint(n, n + 3))
        deg = _checked_degrees(n, edges)
        shape = rng.choice(("adjacent", "nonadjacent", "edges", "maxdeg"))
        x = rng.choice([v for v in range(n) if 1 <= deg[v] <= n - 2])
        d = deg[x]
        try:
            fig = _draw(n, edges)
        except TooFewDistractors:
            continue
        L = LETTERS
        if shape == "adjacent":
            text = phrase(rng, _ADJ, x=L[x])
            nbrs = sorted(neighbours(n, edges)[x])
            answer = d
            check(answer == len(nbrs), "adjacent count")
            cands = [
                (d + 1, "dg-gt-adjacent-counts-itself"),
                (n - 1, "dg-gt-adjacent-all-others"),
                (len(edges), "dg-gt-edges-miscount"),
                (d - 1, "dg-gt-edges-miscount"),
            ]
            working = [
                f"The edges at {L[x]} join it to {', '.join(L[v] for v in nbrs)}.",
                f"That is {d} adjacent vertices.",
            ]
        elif shape == "nonadjacent":
            text = phrase(rng, _NONADJ, x=L[x])
            answer = n - 1 - d
            others = sorted(set(range(n)) - set(neighbours(n, edges)[x]) - {x})
            check(answer == len(others), "non-adjacent count")
            cands = [
                (d, "dg-gt-adjacent-for-nonadjacent"),
                (n - d, "dg-gt-adjacent-counts-itself"),
                (n - 1, "dg-gt-adjacent-all-others"),
                (answer + 1, "dg-gt-edges-miscount"),
            ]
            working = [
                f"{L[x]} is joined to {d} of the other {n - 1} vertices.",
                f"So it is not adjacent to {n - 1} - {d} = {answer} of them.",
            ]
        elif shape == "edges":
            text = phrase(rng, _EDGES)
            answer = len(edges)
            cands = [
                (sum(deg), "dg-gt-edges-as-degree-sum"),
                (n, "dg-gt-vertices-as-edges"),
                (answer + 1, "dg-gt-edges-miscount"),
                (answer - 1, "dg-gt-edges-miscount"),
            ]
            working = [f"The edges are {', '.join(L[u] + L[v] for u, v in edges)}.", f"There are {answer}."]
        else:
            text = phrase(rng, _MAXDEG)
            answer = max(deg)
            second = sorted(set(deg))[-2] if len(set(deg)) > 1 else None
            cands = [
                (second, "dg-gt-degree-of-wrong-vertex"),
                (answer + 1, "dg-gt-edges-miscount"),
                (min(deg), "dg-gt-degree-of-wrong-vertex"),
                (len(edges), "dg-gt-edges-as-degree-sum"),
            ]
            working = [
                "The degrees are " + ", ".join(f"{L[i]} {v}" for i, v in enumerate(deg)) + ".",
                f"The greatest is {answer}.",
            ]
            if len([v for v in deg if v == answer]) > 3:
                continue
        ds = _ints(answer, [c for c in cands if c[0] is not None])
        if len(ds) < 3:
            continue
        return _q(text, fig, answer, working, ds)
    raise AssertionError  # pragma: no cover


# ----------------------------------------------------------------------------------------------- medium

_COMP = (
    "A network is drawn in separate pieces. A piece is a connected component: every vertex in it can be reached from every other by following edges. How many components does the network have?",
    "How many connected components does the drawn network have? (Each separate piece is one component, and a vertex with no edges is a component of its own.)",
)
_CONNECT = (
    "The drawn network is not connected. What is the fewest number of edges that could be added to make it connected?",
    "The drawn network is in more than one piece. At least how many new edges must be added so that it becomes connected?",
)
_SUBV = (
    "The thick edges, together with the vertices at their ends, form a subgraph H of the network drawn. How many vertices does H have?",
    "In the drawn network the thick edges and the vertices they join make a subgraph H. How many vertices are in H?",
)
_SUBE = (
    "The thick edges, together with the vertices at their ends, form a subgraph H of the network drawn. How many edges of the network are not in H?",
    "The subgraph H is made of the thick edges and the vertices at their ends. How many edges of the whole network are left out of H?",
)


def _medium(rng):
    for _ in attempts(400, "a medium graph-terms network"):
        shape = rng.choice(("components", "connect", "subvertices", "subedges"))
        if shape in ("components", "connect"):
            n = rng.randint(6, 8)
            edges = _in_pieces(rng, n)
            comps = components(n, edges)
            c = len(comps)
            try:
                fig = _draw(n, edges)
            except TooFewDistractors:
                continue
            isolated = sum(1 for comp in comps if len(comp) == 1)
            big = max(len(comp) for comp in comps)
            if shape == "components":
                text = phrase(rng, _COMP)
                answer = c
                cands = [
                    (isolated if isolated else None, "dg-gt-components-isolated-only"),
                    (c + 1, "dg-gt-components-miscount"),
                    (c - 1, "dg-gt-components-miscount"),
                    (n - big + 1, "dg-gt-components-edges-per-vertex"),
                ]
            else:
                text = phrase(rng, _CONNECT)
                answer = c - 1
                cands = [
                    (c, "dg-gt-components-as-edges-needed"),
                    (n - big, "dg-gt-components-edges-per-vertex"),
                    (c + 1, "dg-gt-components-miscount"),
                    (len(comps[0]) if len(comps[0]) != answer else None, "dg-gt-components-miscount"),
                ]
            working = [
                "The pieces are " + "; ".join("{" + ", ".join(LETTERS[v] for v in comp) + "}" for comp in comps) + ".",
                f"That is {c} components."
                if shape == "components"
                else f"Joining {c} pieces takes {c} - 1 = {answer} edge{'' if answer == 1 else 's'} at the least.",
            ]
            ds = _ints(answer, [x for x in cands if x[0] is not None and x[0] >= 0], lo=0)
            if answer < 1 or len(ds) < 3:
                continue
            return _q(text, fig, answer, working, ds)
        n = rng.randint(6, 8)
        edges = _connected(rng, n, rng.randint(n + 1, n + 3))
        k = rng.randint(max(2, len(edges) // 3), len(edges) - 2)
        thick = set(rng.sample(edges, k))
        ends = sorted({v for e in thick for v in e})
        if len(ends) == n:
            continue
        try:
            fig = _draw(n, edges, thick)
        except TooFewDistractors:
            continue
        if shape == "subvertices":
            text = phrase(rng, _SUBV)
            answer = len(ends)
            cands = [
                (n, "dg-gt-subgraph-all-vertices"),
                (len(thick), "dg-gt-vertices-as-edges"),
                (answer + 1, "dg-gt-edges-miscount"),
                (answer - 1, "dg-gt-edges-miscount"),
            ]
            working = [
                f"The thick edges are {', '.join(LETTERS[u] + LETTERS[v] for u, v in sorted(thick))}.",
                f"They end at {', '.join(LETTERS[v] for v in ends)}, so H has {answer} vertices.",
            ]
        else:
            text = phrase(rng, _SUBE)
            answer = len(edges) - len(thick)
            cands = [
                (len(thick), "dg-gt-subgraph-counts-its-own-edges"),
                (len(edges), "dg-gt-edges-miscount"),
                (n - len(ends), "dg-gt-subgraph-vertices-of-network"),
                (answer + 1, "dg-gt-edges-miscount"),
            ]
            working = [
                f"The network has {len(edges)} edges and H has {len(thick)} edge{'' if len(thick) == 1 else 's'} (the thick ones).",
                f"{len(edges)} - {len(thick)} = {answer} edge{'' if answer == 1 else 's'} not in H.",
            ]
        ds = _ints(answer, cands, lo=0)
        if len(ds) < 3 or answer < 1:
            continue
        return _q(text, fig, answer, working, ds)
    raise AssertionError  # pragma: no cover


# ----------------------------------------------------------------------------------------------- multigraphs


def _pl(n, word, plural=None):
    """`n word` with the right number: "1 loop", "2 loops"."""
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def _multigraph(rng, n, loops, extras, base):
    """A connected multigraph on ``n`` vertices: a simple connected base, ``loops`` loops and ``extras`` repeated
    copies of base edges. Returns the edge list (pairs, loops as ``(v, v)``) in a shuffled order."""
    simple = _connected(rng, n, base)
    es = list(simple)
    for v in rng.sample(range(n), loops):
        es.append((v, v))
    for e in rng.sample(simple, extras):
        es.append(e)
    rng.shuffle(es)
    return es, simple


def _edge_list_text(es):
    return ", ".join(LETTERS[u] + LETTERS[v] for u, v in es)


def _multi_facts(n, es):
    loops = [e for e in es if e[0] == e[1]]
    counts = Counter(norm(*e) for e in es if e[0] != e[1])
    extras = sum(c - 1 for c in counts.values())
    repeated_pairs = [e for e, c in counts.items() if c > 1]
    in_repeated = sum(c for c in counts.values() if c > 1)
    simple = sorted(counts)
    deg = [0] * n
    for u, v in es:
        deg[u] += 1
        deg[v] += 1  # a loop adds 2 because both ends are at the vertex
    return dict(
        loops=len(loops),
        extras=extras,
        pairs=len(repeated_pairs),
        in_repeated=in_repeated,
        simple=simple,
        deg=deg,
        loop_vertices=[u for u, _ in loops],
    )


_MG_INTRO = (
    "A multigraph has vertices {a} to {z} and these edges, each written once per copy: {es}.",
    "The edges of a multigraph on vertices {a} to {z} are {es}. An edge written twice is two separate edges, and a pair such as {ex} is a loop.",
)
_MG_LOOPS = ("How many loops does the multigraph have? (A loop is an edge that starts and ends at the same vertex.)",)
_MG_MULT = (
    "How many extra edges are there beyond one between each pair of vertices? (Edges joining the same two vertices are multiple edges; count the copies beyond the first of each.)",
)
_MG_PAIRS = (
    "Two vertices joined by more than one edge are joined by multiple edges. How many pairs of vertices are joined by multiple edges?",
)
_MG_SIMPLE = (
    "Loops and repeated edges are removed, keeping one edge between each pair of vertices that was joined, to leave a simple graph. How many edges does the simple graph have?",
)
_MG_DEG = ("What is the degree of vertex {x}? (A loop adds 2 to the degree of its vertex.)",)


def _hard(rng):
    for _ in attempts(400, "a multigraph question"):
        n = rng.randint(5, 6)
        loops = rng.randint(1, 3)
        extras = rng.randint(1, 3)
        es, simple = _multigraph(rng, n, loops, extras, rng.randint(n, n + 1))
        f = _multi_facts(n, es)
        check(sum(f["deg"]) == 2 * len(es), "handshake sum of the multigraph")
        check(len(f["simple"]) == len(simple), "simple graph edges")
        shape = rng.choice(("loops", "multiple", "pairs", "simple", "degree"))
        ex = LETTERS[f["loop_vertices"][0]] * 2
        intro = phrase(rng, _MG_INTRO, a=LETTERS[0], z=LETTERS[n - 1], es=_edge_list_text(es), ex=ex)
        if shape == "loops":
            text = intro + " " + _MG_LOOPS[0]
            answer = f["loops"]
            cands = [
                (f["extras"], "dg-gt-loops-as-multiple"),
                (f["loops"] + f["extras"], "dg-gt-loops-as-multiple"),
                (answer + 1, "dg-gt-edges-miscount"),
                (answer - 1, "dg-gt-edges-miscount"),
            ]
            working = [
                f"The loops are {', '.join(LETTERS[v] * 2 for v in f['loop_vertices'])}, each joining a vertex to itself.",
                f"There are {_pl(answer, 'loop')}.",
            ]
        elif shape == "multiple":
            text = intro + " " + _MG_MULT[0]
            answer = f["extras"]
            cands = [
                (f["in_repeated"], "dg-gt-multiple-all-copies"),
                (f["loops"], "dg-gt-loops-as-multiple"),
                (f["pairs"] if f["pairs"] != answer else None, "dg-gt-multiple-all-copies"),
                (answer + 1, "dg-gt-edges-miscount"),
            ]
            working = [
                f"The repeated pairs give {f['in_repeated']} edges between them, of which one per pair is the first copy.",
                f"The extra copies number {f['in_repeated']} - {f['pairs']} = {answer}.",
            ]
        elif shape == "pairs":
            text = intro + " " + _MG_PAIRS[0]
            answer = f["pairs"]
            cands = [
                (f["extras"], "dg-gt-multiple-all-copies"),
                (f["in_repeated"], "dg-gt-multiple-all-copies"),
                (f["loops"], "dg-gt-loops-as-multiple"),
                (answer + 1, "dg-gt-edges-miscount"),
            ]
            working = [
                f"Count the pairs of vertices that appear more than once in the list: {f['pairs']}.",
                "A pair joined by three edges still counts as one pair.",
            ]
        elif shape == "simple":
            text = intro + " " + _MG_SIMPLE[0]
            answer = len(f["simple"])
            cands = [
                (len(es) - f["loops"], "dg-gt-simple-keeps-repeats"),
                (len(es) - f["extras"], "dg-gt-simple-keeps-loops"),
                (len(es), "dg-gt-simple-keeps-repeats"),
                (answer + 1, "dg-gt-edges-miscount"),
            ]
            working = [
                f"The multigraph has {len(es)} edges. Remove the {_pl(f['loops'], 'loop')} and the {_pl(f['extras'], 'extra copy', 'extra copies')}.",
                f"{len(es)} - {f['loops']} - {f['extras']} = {answer}.",
            ]
        else:
            x = rng.choice(f["loop_vertices"])
            text = intro + " " + phrase(rng, _MG_DEG, x=LETTERS[x])
            answer = f["deg"][x]
            nonloop = sum(1 for u, v in es if u != v and x in (u, v))
            check(answer == nonloop + 2 * sum(1 for u, v in es if u == v == x), "degree with loops")
            cands = [
                (answer - 1, "dg-gt-loop-counted-once"),
                (nonloop, "dg-gt-loop-not-counted"),
                (answer + 1, "dg-gt-edges-miscount"),
                (answer - 2 if answer - 2 != nonloop else None, "dg-gt-loop-not-counted"),
            ]
            working = [
                f"{_pl(nonloop, 'edge')} join{'s' if nonloop == 1 else ''} {LETTERS[x]} to other vertices.",
                f"Each loop at {LETTERS[x]} adds 2, so the degree is {nonloop} + 2 x {(answer - nonloop) // 2} = {answer}.",
            ]
        ds = _ints(answer, [c for c in cands if c[0] is not None], lo=0)
        if len(ds) < 3:
            continue
        d = _q(text, None, answer, working, ds)
        d.pop("figure")
        return d
    raise AssertionError  # pragma: no cover


@generator("dg_graph_terms", level=12, presets={t: {"tier": t} for t in ("easy", "medium", "hard")})
def dg_graph_terms(rng, tier="easy"):
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


# ----------------------------------------------------------------------------------------------- exam chain


def _set_fields(vs):
    items = sorted(vs)
    text = "{" + ", ".join(items) + "}"
    return dict(answer_kind="set", answer=items, answer_text=text, answer_latex="plain:" + text)


def _set_dist(answer_items, candidates):
    from ..grading import GRADERS
    from types import SimpleNamespace

    probe = SimpleNamespace(
        answer=sorted(answer_items),
        answer_text="{" + ", ".join(sorted(answer_items)) + "}",
        answer_kind="set",
        parts=None,
    )
    seen = {probe.answer_text}
    out = []
    for items, tag in candidates:
        if not items:
            continue
        text = "{" + ", ".join(sorted(items)) + "}"
        if text in seen:
            continue
        if GRADERS["set"](probe, text).correct:
            continue
        seen.add(text)
        out.append(Distractor(text, tag, sorted(items)))
    return out[:3]


def _p(prompt, fields, marks, working, distractors, **extra):
    fields = {k: v for k, v in fields.items() if k != "answer_kind"}
    kind = "set" if isinstance(fields.get("answer"), list) else "number"
    return dict(
        prompt=prompt, answer_kind=kind, marks=marks, working=list(working), distractors=distractors, **fields, **extra
    )


def _num(answer):
    return dict(answer=answer, answer_text=str(answer), answer_latex=str(answer))


def _chain_drawn(rng):
    for _ in attempts(400, "a drawn graph-terms chain"):
        n = rng.randint(6, 8)
        edges = _connected(rng, n, rng.randint(n - 1, n + 1))
        deg = _checked_degrees(n, edges)
        x = rng.choice([v for v in range(n) if 2 <= deg[v] <= 4])
        rest = [e for e in edges if x not in e]
        nbrs = sorted(neighbours(n, edges)[x])
        left = [v for v in range(n) if v != x]
        # components of the graph with x deleted, found on relabelled vertices
        idx = {v: i for i, v in enumerate(left)}
        comps = components(n - 1, [(idx[u], idx[v]) for u, v in rest])
        c = len(comps)
        if not (2 <= c <= 3) or len(rest) < 3:
            continue
        try:
            fig = _draw(n, edges)
        except TooFewDistractors:
            continue
        L = LETTERS
        m, d = len(edges), deg[x]
        check(len(rest) == m - d, "edges after deleting the vertex two ways")
        pa = _num(d)
        da = _ints(
            d,
            [
                (d + 1, "dg-gt-adjacent-counts-itself"),
                (n - 1, "dg-gt-adjacent-all-others"),
                (d - 1, "dg-gt-edges-miscount"),
                (m, "dg-gt-edges-miscount"),
            ],
        )
        sa = [L[v] for v in nbrs]
        non = [L[v] for v in left if v not in nbrs]
        db = _set_dist(
            sa,
            [
                (sa + [L[x]], "dg-gt-adjacent-counts-itself"),
                (non, "dg-gt-adjacent-for-nonadjacent"),
                (sa[:-1], "dg-gt-edges-miscount"),
                ([L[v] for v in left], "dg-gt-adjacent-all-others"),
            ],
        )
        dc = _ints(
            m,
            [
                (sum(deg), "dg-gt-edges-as-degree-sum"),
                (n, "dg-gt-vertices-as-edges"),
                (m + 1, "dg-gt-edges-miscount"),
                (m - 1, "dg-gt-edges-miscount"),
            ],
        )
        dd = _ints(
            m - d,
            [
                (m, "dg-gt-delete-vertex-keeps-edges"),
                (m - 2 * d, "dg-gt-delete-vertex-uses-sum"),
                (m - d + 1, "dg-gt-edges-miscount"),
                (d, "dg-gt-delete-vertex-keeps-edges"),
            ],
            lo=0,
        )
        de = _ints(
            c,
            [
                (c + 1, "dg-gt-components-miscount"),
                (c - 1, "dg-gt-components-miscount"),
                (sum(1 for comp in comps if len(comp) == 1) or None, "dg-gt-components-isolated-only"),
                (c + 2, "dg-gt-components-miscount"),
            ],
            lo=1,
        )
        if min(len(x_) for x_ in (da, db, dc, dd, de)) < 2:
            continue
        stem = (
            f"A network of {n} vertices, {L[0]} to {L[n - 1]}, is drawn. Two vertices are adjacent when one edge joins them, "
            f"and the degree of a vertex is the number of edges at it."
        )
        return dict(
            text=stem,
            figure=fig,
            parts=[
                _p(
                    f"Find the degree of vertex {L[x]}.",
                    pa,
                    1,
                    [f"Count the edges at {L[x]}: {d}."],
                    da,
                    sensible_range=(0, n - 1),
                ),
                _p(
                    f"List the vertices adjacent to {L[x]}.",
                    _set_fields(sa),
                    1,
                    [f"The edges at {L[x]} end at {', '.join(sa)}."],
                    db,
                ),
                _p(
                    "How many edges does the network have?",
                    _num(m),
                    1,
                    [f"The edges are {', '.join(L[u] + L[v] for u, v in edges)}."],
                    dc,
                    sensible_range=(0, None),
                ),
                _p(
                    f"Vertex {L[x]} and every edge at it are deleted. How many edges remain?",
                    _num(m - d),
                    2,
                    [f"The {_pl(d, 'edge')} at {L[x]} go: {m} - {d} = {m - d}."],
                    dd,
                    follow_through=f"part_c - {d}",
                    sensible_range=(-1, None),
                ),
                _p(
                    "How many connected components does the remaining graph have?",
                    _num(c),
                    2,
                    [
                        "The pieces left are "
                        + "; ".join("{" + ", ".join(L[left[i]] for i in comp) + "}" for comp in comps)
                        + "."
                    ],
                    de,
                    sensible_range=(0, n),
                ),
            ],
        )
    raise AssertionError  # pragma: no cover


def _chain_multi(rng):
    for _ in attempts(400, "a multigraph chain"):
        n = rng.randint(5, 6)
        loops, extras = rng.randint(1, 3), rng.randint(1, 3)
        es, simple = _multigraph(rng, n, loops, extras, rng.randint(n, n + 1))
        f = _multi_facts(n, es)
        total = len(es)
        simple_m = len(f["simple"])
        check(simple_m == total - f["loops"] - f["extras"], "simple edges")
        x = f["loop_vertices"][0]
        nonloop = sum(1 for u, v in es if u != v and x in (u, v))
        degx = f["deg"][x]
        loops_at_x = sum(1 for u, v in es if u == v == x)
        check(degx == nonloop + 2 * loops_at_x, "degree with loops")
        if f["extras"] == f["loops"] or degx == total:
            continue
        L = LETTERS
        da = _ints(
            f["loops"],
            [
                (f["extras"], "dg-gt-loops-as-multiple"),
                (f["loops"] + 1, "dg-gt-edges-miscount"),
                (f["loops"] - 1, "dg-gt-edges-miscount"),
                (f["loops"] + f["extras"], "dg-gt-loops-as-multiple"),
            ],
        )
        db = _ints(
            f["extras"],
            [
                (f["in_repeated"], "dg-gt-multiple-all-copies"),
                (f["loops"], "dg-gt-loops-as-multiple"),
                (f["extras"] + 1, "dg-gt-edges-miscount"),
                (f["pairs"], "dg-gt-multiple-all-copies"),
            ],
        )
        dc = _ints(
            simple_m,
            [
                (total - f["loops"], "dg-gt-simple-keeps-repeats"),
                (total - f["extras"], "dg-gt-simple-keeps-loops"),
                (total, "dg-gt-simple-keeps-repeats"),
                (simple_m + 1, "dg-gt-edges-miscount"),
            ],
        )
        dd = _ints(
            degx,
            [
                (degx - loops_at_x, "dg-gt-loop-counted-once"),
                (nonloop, "dg-gt-loop-not-counted"),
                (degx + 1, "dg-gt-edges-miscount"),
                (simple_m, "dg-gt-edges-miscount"),
            ],
            lo=0,
        )
        if min(len(v) for v in (da, db, dc, dd)) < 2:
            continue
        stem = (
            f"A multigraph has vertices {L[0]} to {L[n - 1]} and {total} edges: {_edge_list_text(es)}. "
            f"An edge written twice is two separate edges, and a pair such as {L[x] * 2} is a loop."
        )
        return dict(
            text=stem,
            parts=[
                _p(
                    "How many loops does the multigraph have?",
                    _num(f["loops"]),
                    1,
                    [f"The loops are {', '.join(L[v] * 2 for v in f['loop_vertices'])}."],
                    da,
                    sensible_range=(0, total),
                ),
                _p(
                    "How many extra edges are there beyond one between each pair of joined vertices?",
                    _num(f["extras"]),
                    1,
                    [
                        f"Edges between repeated pairs: {f['in_repeated']}; one per pair is the first copy ({f['pairs']}), so {f['extras']} are extra."
                    ],
                    db,
                    sensible_range=(0, total),
                ),
                _p(
                    "Loops and extra copies are removed to leave a simple graph. How many edges does it have?",
                    _num(simple_m),
                    2,
                    [f"{total} - {f['loops']} - {f['extras']} = {simple_m}."],
                    dc,
                    follow_through=f"{total} - part_a - part_b",
                    sensible_range=(0, total),
                ),
                _p(
                    f"Find the degree of vertex {L[x]} in the original multigraph.",
                    _num(degx),
                    2,
                    [
                        f"{_pl(nonloop, 'edge')} leave{'s' if nonloop == 1 else ''} {L[x]}, and {loops_at_x} loop{'s' if loops_at_x != 1 else ''} add 2 each: {nonloop} + {2 * loops_at_x} = {degx}."
                    ],
                    dd,
                    sensible_range=(0, 2 * total),
                ),
            ],
        )
    raise AssertionError  # pragma: no cover


@generator("dg_graph_terms_chain", level=12, presets={"exam": {"tier": "exam"}})
def dg_graph_terms_chain(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _chain_drawn(rng) if rng.random() < 0.6 else _chain_multi(rng)


# ----------------------------------------------------------------------------------------------- the reader


def _spec_graph(fig):
    names = [v["id"] for v in fig["vertices"]]
    idx = {v: i for i, v in enumerate(names)}
    edges = [norm(idx[e["from"]], idx[e["to"]]) for e in fig["edges"]]
    thick = {norm(idx[e["from"]], idx[e["to"]]) for e in fig["edges"] if e.get("highlight")}
    return names, edges, thick


def _read_graph_terms(question) -> str:
    """The answer recomputed from the figure's vertices and edges and the question's words alone."""
    fig, text = question.figure, question.text
    names, edges, thick = _spec_graph(fig)
    n = len(names)
    deg = degrees(n, edges)
    found = re.search(r"[Vv]ertex ([A-J])\b", text)
    x = names.index(found.group(1)) if found else None
    if "adjacent" in text and "not adjacent" not in text:
        return str(deg[x])
    if "not adjacent" in text:
        return str(n - 1 - deg[x])
    if "How many edges does the network drawn have" in text or "Count the edges" in text:
        return str(len(edges))
    if "greatest" in text:
        return str(max(deg))
    if "fewest number of edges" in text or "At least how many new edges" in text:
        return str(len(components(n, edges)) - 1)
    if "component" in text:
        return str(len(components(n, edges)))
    if "How many vertices" in text and thick:
        return str(len({v for e in thick for v in e}))
    if "not in H" in text or "left out of H" in text:
        return str(len(edges) - len(thick))
    raise AssertionError(f"unrecognised question: {text}")


figures.register_reader("dg_graph_terms", _read_graph_terms)
