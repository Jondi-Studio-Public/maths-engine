"""Generates `snet_degrees` questions (easy, medium, hard) and the exam chain `snet_degrees_chain`.

``graph-vertex-degrees-handshaking`` (``financial-discrete``, answer ``number``): the degree of a vertex, and the
handshaking lemma: the degrees of a network add to twice its number of edges.

* easy: the degree of one vertex of a network given as an edge list, the number of edges when every degree is listed,
  or how many vertices have odd degree;
* medium: the degrees of every vertex but one are listed with the number of edges, and the last degree is found;
  or a longer degree list gives the number of edges.
* hard: two kinds of reasoning with no network drawn. A network has v vertices, each of degree d1 or d2, and e
  edges: how many have degree d2? Or: what is the largest number of vertices a network with e edges can have when
  every vertex has degree at least k?
* exam (``snet_degrees_chain``): the degrees of all but one of the vertices and the number of edges are given.
  (a) the sum of all the degrees; (b) the missing degree; (c) how many vertices have odd degree; (d) how many edges
  remain when the unlisted vertex and its edges are deleted.

Every network behind a question is a real drawn graph; its degrees are counted from the edge list and re-derived from
the adjacency matrix with SymPy, and the handshake sum is checked against the edge count.
"""

from __future__ import annotations

import sympy

from ..core import attempts, check, generator, phrase
from ._snet_kit import (
    LETTERS,
    adjacency,
    degrees,
    describe_network,
    edge_name,
    counted,
    int_distractors,
    pick_context,
    random_connected,
)


def _check_degrees(n, edges):
    """Degrees by counting, then by SymPy (row sums of the adjacency matrix); the handshake sum is twice the edges."""
    deg = degrees(n, edges)
    matrix = sympy.Matrix(adjacency(n, edges))
    by_matrix = [sum(matrix.row(i)) for i in range(n)]
    check(deg == by_matrix, "degrees by counting and by the adjacency matrix differ")
    check(sum(deg) == 2 * len(edges), "the degrees do not add to twice the edges")
    return deg


def _degree_list_text(deg, ctx, hidden=None):
    parts = [f"{LETTERS[i]}: {d}" for i, d in enumerate(deg) if i != hidden]
    return ", ".join(parts)


# -- easy -------------------------------------------------------------------------------------------------------

_EASY_DEGREE = (
    "What is the degree of vertex {x}?",
    "The degree of a vertex is the number of {es} that meet there. Find the degree of vertex {x}.",
    "Find the degree of vertex {x}.",
)
_EASY_EDGES = (
    "The degree of each vertex is: {degs}. How many {es} does the network have?",
    "A network's vertex degrees are {degs}. How many edges does it have?",
)
_EASY_ODD = (
    "How many of the {n} vertices have an odd degree?",
    "Count the vertices of odd degree.",
)


def _easy(rng):
    ctx = pick_context(rng)
    for _ in attempts(200, "a network with varied degrees"):
        n = rng.randint(5, 7)
        edges = random_connected(rng, n, rng.randint(n + 1, n + 3))
        deg = _check_degrees(n, edges)
        if len(set(deg)) >= 3 and any(2 <= d <= 5 for d in deg):
            break
    shape = rng.choice(["degree", "degree", "edges", "odd"])
    if shape == "degree":
        x = rng.choice([v for v in range(n) if 2 <= deg[v] <= 5])
        d = deg[x]
        text = describe_network(ctx, n, edges) + " " + phrase(rng, _EASY_DEGREE, x=LETTERS[x], es=ctx["es"])
        at_x = [edge_name(e) for e in edges if x in e]
        other = rng.choice([v for v in range(n) if deg[v] != d])
        candidates = [
            (d + 1, "net-degree-miscount"),
            (d - 1, "net-degree-miscount"),
            (deg[other], "net-degree-wrong-vertex"),
            (len(edges), "net-degree-as-edge-total"),
        ]
        working = [
            f"The edges at {LETTERS[x]} are {', '.join(at_x)}.",
            f"That is {d}, so the degree of {LETTERS[x]} is {d}.",
        ]
        answer = d
    elif shape == "edges":
        text = f"{ctx['intro'].format(n=n, a=LETTERS[0], z=LETTERS[n - 1])} " + phrase(
            rng, _EASY_EDGES, degs=_degree_list_text(deg, ctx), es=ctx["es"]
        )
        answer = len(edges)
        candidates = [
            (sum(deg), "net-degree-sum-as-edges"),
            (sum(deg) // 4 if sum(deg) % 4 == 0 else sum(deg) // 2 + 1, "net-handshake-slip"),
            (sum(deg) // 2 - 1, "net-handshake-slip"),
            (n, "net-vertices-as-edges"),
        ]
        working = [
            f"The degrees add to {' + '.join(str(d) for d in deg)} = {sum(deg)}.",
            f"Each edge adds 1 to two degrees, so the sum is twice the number of edges: {sum(deg)} / 2 = {answer}.",
        ]
    else:
        text = describe_network(ctx, n, edges) + " " + phrase(rng, _EASY_ODD, n=n)
        odd = sum(d % 2 for d in deg)
        answer = odd
        candidates = [
            (n - odd, "net-odd-even-swapped"),
            (sum(d for d in deg if d % 2), "net-odd-degrees-summed"),
            (odd + 1, "net-degree-miscount"),
            (odd - 1, "net-degree-miscount"),
            (odd + 2, "net-degree-miscount"),
        ]
        working = [
            "The degrees are " + ", ".join(f"{LETTERS[i]} {d}" for i, d in enumerate(deg)) + ".",
            f"{odd} of them are odd.",
        ]
    distractors = int_distractors(answer, candidates)
    check(len(distractors) >= 2, "too few distractors")
    return _single(text, answer, working, distractors)


def _single(text, answer, working, distractors):
    return dict(
        text=text,
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=distractors,
    )


# -- medium -----------------------------------------------------------------------------------------------------

_MEDIUM_MISSING = (
    "A network has {n} vertices, A to {z}, and {m} {es}. The degrees of the vertices other than {z} are {listed}. "
    "What is the degree of vertex {z}?",
    "A network has {m} {es} joining {n} vertices, A to {z}. The degrees of all the vertices except {z} are {listed}. "
    "Find the degree of vertex {z}.",
)
_MEDIUM_EDGES = (
    "A network has {n} vertices with degrees {degs}. How many {es} does it have?",
    "The vertices of a network have degrees {degs}. How many {es} are there?",
)


def _medium(rng):
    ctx = pick_context(rng)
    n = rng.randint(6, 8)
    edges = random_connected(rng, n, rng.randint(n + 2, n + 5))
    deg = _check_degrees(n, edges)
    shape = rng.choice(["missing", "missing", "edges"])
    if shape == "edges":
        degs = ", ".join(str(d) for d in deg)
        answer = len(edges)
        text = phrase(rng, _MEDIUM_EDGES, n=n, degs=degs, es=ctx["es"])
        total = sum(deg)
        candidates = [
            (total, "net-degree-sum-as-edges"),
            (total // 4 if total % 4 == 0 else total // 2 + 1, "net-handshake-slip"),
            (total // 2 + 1, "net-handshake-slip"),
            (total // 2 - 1, "net-handshake-slip"),
            (n, "net-vertices-as-edges"),
        ]
        working = [
            f"The degrees add to {total}.",
            f"The sum of the degrees is twice the number of edges, so there are {total} / 2 = {answer} {ctx['es']}.",
        ]
    else:
        hidden = n - 1
        listed = _degree_list_text(deg, ctx, hidden)
        known = sum(deg[:hidden])
        answer = deg[hidden]
        m = len(edges)
        text = phrase(rng, _MEDIUM_MISSING, n=n, listed=listed, z=LETTERS[hidden], m=m, es=ctx["es"])
        average = round(known / hidden)
        candidates = [
            (m - known if m > known else -1, "net-edges-as-degree-sum"),
            ((2 * m - known) / 2, "net-half-of-missing"),
            (average, "net-average-degree-used"),
            (answer + 1, "net-handshake-slip"),
            (answer - 1, "net-handshake-slip"),
            (2 * m - known + 2, "net-handshake-slip"),
        ]
        working = [
            f"The degrees add to twice the number of edges: 2 x {m} = {2 * m}.",
            f"The listed degrees add to {known}, so vertex {LETTERS[hidden]} has degree {2 * m} - {known} = {answer}.",
        ]
    distractors = int_distractors(answer, candidates, lo=1)
    check(len(distractors) >= 2, "too few distractors")
    return _single(text, answer, working, distractors)


# -- hard -------------------------------------------------------------------------------------------------------

_HARD_TWO = (
    "A network has {v} vertices and {m} {es}. Every vertex has degree {d1} or degree {d2}. "
    "How many vertices have degree {d2}?",
    "In a network with {v} vertices and {m} edges, each vertex has degree {d1} or {d2}. "
    "How many of the vertices have degree {d2}?",
)
_HARD_LEAST = (
    "A network has {m} {es}, and every vertex has degree at least {k}. What is the greatest number of vertices "
    "the network can have?",
    "Every vertex of a network has degree at least {k}, and the network has {m} edges. "
    "What is the largest possible number of vertices?",
)


def _hard(rng):
    ctx = pick_context(rng)
    if rng.random() < 0.6:
        d1, d2 = rng.choice([(2, 4), (3, 5), (2, 5), (3, 4), (1, 3), (2, 6), (3, 6)])
        for _ in range(500):
            a, b = rng.randint(2, 8), rng.randint(2, 8)
            if (d1 * a + d2 * b) % 2 == 0 and a != b:
                break
        else:
            raise AssertionError  # pragma: no cover
        v, m = a + b, (d1 * a + d2 * b) // 2
        answer = b
        text = phrase(rng, _HARD_TWO, v=v, m=m, es=ctx["es"], d1=d1, d2=d2)
        # Independent route: solve d1 a + d2 b = 2m, a + b = v with SymPy.
        sa, sb = sympy.symbols("a b")
        solution = sympy.solve([d1 * sa + d2 * sb - 2 * m, sa + sb - v], [sa, sb])
        check(solution[sb] == b and solution[sa] == a, "SymPy's solution differs")
        candidates = [
            (a, "net-degree-class-swapped"),
            (v // 2 if v % 2 == 0 else -1, "net-assumes-equal-split"),
            (2 * m - d1 * v, "net-forgot-divide-by-gap"),
            (m - d1 * v, "net-edges-as-degree-sum"),
            (b + 1, "net-handshake-slip"),
            (b - 1, "net-handshake-slip"),
        ]
        working = [
            f"Let b be the number of vertices of degree {d2}. The other {v} - b have degree {d1}.",
            f"The degrees add to twice the edges: {d1}({v} - b) + {d2}b = {2 * m}.",
            f"{d1 * v} + {d2 - d1}b = {2 * m}, so b = {2 * m - d1 * v} / {d2 - d1} = {b}.",
        ]
    else:
        k = rng.choice([2, 3, 3, 4])
        m = rng.randint(14, 40)
        if (2 * m) % k == 0:
            m += 1
        answer = (2 * m) // k
        # Independent route: the largest v with k v <= 2m, found by search.
        best = max(v for v in range(1, 2 * m + 1) if k * v <= 2 * m)
        check(best == answer, "search for the largest number of vertices differs")
        text = phrase(rng, _HARD_LEAST, m=m, es=ctx["es"], k=k)
        candidates = [
            (m // k, "net-edges-as-degree-sum"),
            (-(-(2 * m) // k), "net-rounded-up-not-down"),
            (2 * m, "net-degree-sum-as-vertices"),
            (m // 2, "net-halved-instead-of-doubled"),
            (answer - 1, "net-handshake-slip"),
        ]
        working = [
            f"The degrees add to 2 x {m} = {2 * m}, and each of the v vertices has degree at least {k}, so {k}v <= {2 * m}.",
            f"v <= {2 * m} / {k} = {(2 * m) / k:.2f}, and v is a whole number, so the greatest value is {answer}.",
        ]
    distractors = int_distractors(answer, candidates, lo=1)
    check(len(distractors) >= 2, "too few distractors")
    return _single(text, answer, working, distractors)


@generator(
    "snet_degrees",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def snet_degrees(rng, tier="easy"):
    """Vertex degrees and the handshaking lemma; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------

_CHAIN_STEMS = (
    "A network joins {n} {vs}, {a} to {z}, and has {m} {es}. The degree of each vertex except {z} is: {listed}.",
    "In a network of {n} {vs} ({a} to {z}) there are {m} {es} altogether. The degrees of all the vertices except "
    "{z} are: {listed}.",
)


@generator("snet_degrees_chain", level=12, presets={"exam": dict(tier="exam")})
def snet_degrees_chain(rng, tier="exam"):
    """Exam chain: the degree sum, the missing degree, the odd vertices and the edges left after deleting a vertex."""
    check(tier == "exam", f"unknown tier {tier!r}")
    ctx = pick_context(rng)
    for _ in range(300):
        n = rng.randint(7, 8)
        edges = random_connected(rng, n, rng.randint(n + 3, n + 9))
        deg = _check_degrees(n, edges)
        hidden = n - 1
        known = sum(deg[:hidden])
        listed_odd = sum(d % 2 for d in deg[:hidden])
        if 1 <= deg[hidden] <= 6 and 3 <= len(set(deg[:hidden])) and 2 * len(edges) - known >= 1:
            break
    else:
        raise AssertionError  # pragma: no cover
    m = len(edges)
    d_z = deg[hidden]
    z = LETTERS[hidden]
    total = 2 * m
    odd = listed_odd + d_z % 2
    remaining = m - d_z
    # Independent route: delete the vertex from the edge list and count what is left.
    check(len([e for e in edges if hidden not in e]) == remaining, "edges left after deleting the vertex differ")
    check(sum(1 for d in deg if d % 2) == odd, "odd vertices counted two ways differ")
    stem = phrase(
        rng,
        _CHAIN_STEMS,
        n=n,
        vs=ctx["vs"],
        a=LETTERS[0],
        z=z,
        m=m,
        es=ctx["es"],
        listed=_degree_list_text(deg, ctx, hidden),
    )

    a_dist = int_distractors(
        total,
        [
            (m, "net-edges-as-degree-sum"),
            (known, "net-listed-sum-only"),
            (m + known, "net-handshake-slip"),
            (total + 2, "net-handshake-slip"),
        ],
    )
    b_dist = int_distractors(
        d_z,
        [
            ((total - known) / 2, "net-half-of-missing"),
            (round(known / hidden), "net-average-degree-used"),
            (d_z + 1, "net-handshake-slip"),
            (d_z - 1, "net-handshake-slip"),
            (m - known if m > known else -1, "net-edges-as-degree-sum"),
        ],
        lo=1,
    )
    c_dist = int_distractors(
        odd,
        [
            (listed_odd, "net-hidden-vertex-ignored"),
            (n - odd, "net-odd-even-swapped"),
            (odd + 1, "net-handshake-slip"),
            (odd - 1, "net-handshake-slip"),
        ],
    )
    d_dist = int_distractors(
        remaining,
        [
            (m, "net-deleted-edges-kept"),
            (m - 2 * d_z, "net-edges-removed-twice"),
            (d_z, "net-answered-removed-edges"),
            (m - d_z + 1, "net-handshake-slip"),
            (m - d_z - 1, "net-handshake-slip"),
        ],
    )
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the sum of the degrees of all {n} vertices.",
                answer_kind="number",
                answer=total,
                answer_text=str(total),
                answer_latex=str(total),
                marks=1,
                sensible_range=(0, None),
                working=[f"Each edge adds 1 to two degrees, so the sum of the degrees is 2 x {m} = {total}."],
                distractors=a_dist,
            ),
            dict(
                prompt=f"Hence find the degree of vertex {z}.",
                answer_kind="number",
                answer=d_z,
                answer_text=str(d_z),
                answer_latex=str(d_z),
                marks=2,
                sensible_range=(0, n),
                follow_through=f"part_a - {known}",
                working=[f"The listed degrees add to {known}, so vertex {z} has degree {total} - {known} = {d_z}."],
                distractors=b_dist,
            ),
            dict(
                prompt="How many of the vertices have an odd degree?",
                answer_kind="number",
                answer=odd,
                answer_text=str(odd),
                answer_latex=str(odd),
                marks=1,
                sensible_range=(-1, n + 1),
                follow_through=f"(1 - (-1)**part_b)/2 + {listed_odd}",
                working=[
                    f"{listed_odd} of the listed degrees are odd, and vertex {z} has degree {d_z}, "
                    f"which is {'odd' if d_z % 2 else 'even'}.",
                    f"So {odd} vertices have an odd degree.",
                ],
                distractors=c_dist,
            ),
            dict(
                prompt=f"Vertex {z} and every {ctx['e']} at {z} are removed from the network. "
                f"How many {ctx['es']} remain?",
                answer_kind="number",
                answer=remaining,
                answer_text=str(remaining),
                answer_latex=str(remaining),
                marks=2,
                sensible_range=(-1, None),
                follow_through=f"{m} - part_b",
                working=[
                    f"Vertex {z} has degree {d_z}, so {counted(d_z, ctx['e'], ctx['es'])} removed: {m} - {d_z} = {remaining}."
                ],
                distractors=d_dist,
            ),
        ],
    )
