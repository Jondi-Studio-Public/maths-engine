"""Generates `uc2-chromatic-polynomial-bipartite-hall` questions (easy, medium, hard) and the exam chain
`exd_x_r3l_chromatic_polynomial_hall`.

``uc2-chromatic-polynomial-bipartite-hall`` (``discrete-mathematics``, answer ``number``): proper colourings of a small
graph through its chromatic polynomial, and Hall's condition for matching in a bipartite graph.

* easy: the number of proper colourings of a connected chordal graph on five vertices with k colours. Every vertex that
  is added joins an existing clique, so the chromatic polynomial factorises as k(k - s_2)(k - s_3)... where s is the size
  of the clique it joins; the graph is shown in a shuffled order so the elimination order is not given away.
* medium: the coefficient of k^2 in the chromatic polynomial of a connected five-vertex graph that is not chordal (it
  contains an induced 4-cycle or 5-cycle), by deletion and contraction.
* hard: either the number of proper colourings with k colours of a six-vertex graph that is not chordal, or the
  largest number of workers who can be matched to different jobs they can do, when Hall's condition fails (the
  answer is |X| minus the largest deficiency |S| - |N(S)|).
* exam (``exd_x_r3l_chromatic_polynomial_hall``): one of two chains. Colouring: (a) the chromatic polynomial,
  (b) its value at k = 4 (follows through from (a)), (c) the chromatic number. Matching: (a) the set of workers that
  violates Hall's condition (multiple choice), (b) the deficiency, (c) the largest matching (follows through from (b)).

Every chromatic polynomial comes from deletion-contraction and is checked against a brute-force count of the proper
colourings with 1, 2, 3, 4 and 5 colours; every matching answer comes from augmenting paths and is checked against the
deficiency by enumeration of all subsets of workers.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, paper_notation, pick_distractors

_k = sympy.symbols("k")
_LABELS = "ABCDEF"


# -- graphs and chromatic polynomials ---------------------------------------------------------------------------------


def _edge(u, v):
    return (u, v) if u < v else (v, u)


def _poly(nv, edges):
    """The chromatic polynomial of the graph on vertices 0..nv-1 by deletion-contraction (a SymPy expression in k)."""
    edges = frozenset(_edge(*e) for e in edges)
    if not edges:
        return _k**nv
    u, v = sorted(edges)[0]
    deleted = edges - {(u, v)}
    # contract v into u, relabel so the vertices stay 0..nv-2
    merged = set()
    for a, b in deleted:
        a2, b2 = (u if a == v else a), (u if b == v else b)
        if a2 != b2:
            merged.add(_edge(a2, b2))
    relabel = {x: i for i, x in enumerate(sorted(set(range(nv)) - {v}))}
    contracted = frozenset(_edge(relabel[a], relabel[b]) for a, b in merged)
    return sympy.expand(_poly(nv, deleted) - _poly(nv - 1, contracted))


def _slip_polys(nv, edges):
    edges = frozenset(_edge(*e) for e in edges)
    u, v = sorted(edges)[0]
    deleted = edges - {(u, v)}
    merged = set()
    for a, b in deleted:
        a2, b2 = (u if a == v else a), (u if b == v else b)
        if a2 != b2:
            merged.add(_edge(a2, b2))
    relabel = {x: i for i, x in enumerate(sorted(set(range(nv)) - {v}))}
    contracted = frozenset(_edge(relabel[a], relabel[b]) for a, b in merged)
    return _poly(nv, deleted), _poly(nv - 1, contracted)


def _brute_count(nv, edges, colours):
    return sum(1 for col in product(range(colours), repeat=nv) if all(col[a] != col[b] for a, b in edges))


def _verify_poly(nv, edges, poly):
    for colours in range(1, 6):
        check(
            poly.subs(_k, colours) == _brute_count(nv, edges, colours),
            "chromatic polynomial differs from a brute-force count",
        )


def _connected(nv, edges):
    seen, stack = {0}, [0]
    while stack:
        x = stack.pop()
        for a, b in edges:
            for p, q in ((a, b), (b, a)):
                if p == x and q not in seen:
                    seen.add(q)
                    stack.append(q)
    return len(seen) == nv


def _is_chordal(nv, edges):
    """Perfect elimination test: repeatedly remove a vertex whose neighbours form a clique."""
    adj = {i: set() for i in range(nv)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    remaining = set(range(nv))
    while remaining:
        for x in sorted(remaining):
            nbrs = adj[x] & remaining
            if all(b in adj[a] for a, b in combinations(sorted(nbrs), 2)):
                remaining.remove(x)
                break
        else:
            return False
    return True


def _chordal_graph(rng, nv):
    """(edges, sizes): a connected chordal graph built by joining each new vertex to an existing clique of size s."""
    edges, sizes = set(), [0]
    cliques = [[0]]
    for v in range(1, nv):
        s = rng.choice([1, 1, 2, 2, 3])
        options = [c for c in cliques if len(c) >= s]
        if not options:
            s = 1
            options = cliques
        base = rng.choice(options)
        join = rng.sample(base, s)
        for u in join:
            edges.add(_edge(u, v))
        sizes.append(s)
        cliques.append(sorted(join + [v]))
        cliques.append([v])
    return sorted(edges), sizes


def _shuffled(rng, nv, edges):
    perm = list(range(nv))
    rng.shuffle(perm)
    out = sorted(_edge(perm[a], perm[b]) for a, b in edges)
    return out


def _edge_text(edges) -> str:
    return ", ".join(f"{_LABELS[a]}{_LABELS[b]}" for a, b in edges)


def _graph_stem(nv, edges) -> str:
    return f"A graph has vertices {', '.join(_LABELS[:nv])} and edges {_edge_text(edges)}."


def _non_chordal(rng, nv, m_range):
    for _ in attempts(3000, "a connected non-chordal graph"):
        m = rng.randint(*m_range)
        pairs = list(combinations(range(nv), 2))
        edges = sorted(rng.sample(pairs, m))
        if not _connected(nv, edges) or _is_chordal(nv, edges):
            continue
        return edges
    raise AssertionError  # pragma: no cover


def _int(v) -> int:
    v = sympy.nsimplify(v)
    check(v == int(v), "non-integral value")
    return int(v)


def _coef(poly, power):
    return _int(sympy.Poly(poly, _k).coeff_monomial(_k**power))


# -- Hall's condition -----------------------------------------------------------------------------------------------


_WORKERS = ("Alex", "Bree", "Cam", "Dana", "Eli")
_JOBS = ("painting", "plumbing", "roofing", "tiling", "wiring")


def _max_matching(adj, nx):
    match = {}

    def try_assign(x, seen):
        for y in adj[x]:
            if y in seen:
                continue
            seen.add(y)
            if y not in match or try_assign(match[y], seen):
                match[y] = x
                return True
        return False

    return sum(1 for x in range(nx) if try_assign(x, set()))


def _neighbourhood(adj, subset):
    return set().union(*(adj[x] for x in subset)) if subset else set()


def _deficiency(adj, nx):
    return max(len(s) - len(_neighbourhood(adj, s)) for r in range(1, nx + 1) for s in combinations(range(nx), r))


def _hall_instance(rng, deficiency=1):
    """(adjacency lists, violating set S) for five workers and five jobs where Hall's condition fails."""
    for _ in attempts(2000, "a Hall-violating instance"):
        size = rng.choice([3, 3, 4]) if deficiency == 1 else 3
        s = sorted(rng.sample(range(5), size))
        shared = sorted(rng.sample(range(5), size - deficiency))
        adj = {}
        for x in range(5):
            if x in s:
                adj[x] = sorted(rng.sample(shared, rng.randint(1, len(shared))))
            else:
                adj[x] = sorted(rng.sample(range(5), rng.randint(2, 3)))
        if any(not adj[x] for x in range(5)):
            continue
        if _deficiency(adj, 5) != deficiency:
            continue
        violating = [t for r in range(1, 6) for t in combinations(range(5), r) if len(_neighbourhood(adj, t)) < len(t)]
        if len(violating) != 1:
            continue
        check(_max_matching(adj, 5) == 5 - deficiency, "matching size differs from 5 - deficiency")
        return adj, list(violating[0])
    raise AssertionError  # pragma: no cover


def _hall_stem(adj) -> str:
    lines = [f"{_WORKERS[x]} can do {', '.join(_JOBS[y] for y in adj[x])}" for x in range(5)]
    return (
        "A builder has five jobs to hand out, one to each of five workers, and each worker can only do some of them: "
        + "; ".join(lines)
        + "."
    )


# -- tiers -------------------------------------------------------------------------------------------------------------


def _easy(rng):
    nv = 5
    for _ in attempts(200, "a chordal graph with a triangle"):
        edges, sizes = _chordal_graph(rng, nv)
        if max(sizes) >= 2 and len(edges) >= 5:
            break
    edges = _shuffled(rng, nv, edges)
    poly = _poly(nv, edges)
    _verify_poly(nv, edges, poly)
    check(_is_chordal(nv, edges), "easy graph should be chordal")
    expected = sympy.expand(_k * sympy.prod([_k - s for s in sizes[1:]]))
    check(sympy.expand(poly - expected) == 0, "chordal factorisation differs from deletion-contraction")
    colours = rng.choice([4, 4, 5]) if max(sizes) >= 3 else rng.choice([3, 4])
    value = _int(poly.subs(_k, colours))
    check(value > 0 and value == _brute_count(nv, edges, colours), "colouring count differs from brute force")
    tree = colours * (colours - 1) ** (nv - 1)
    greedy_pend = colours * (colours - 1) * (colours - 2) ** (nv - 2)
    cands = [
        Distractor(str(tree), "r3l-chromatic-tree-formula", tree),
        Distractor(str(colours**nv), "r3l-chromatic-no-constraint", colours**nv),
        Distractor(str(greedy_pend), "r3l-chromatic-pendant-choices", greedy_pend),
    ]
    falling = 1
    for i in range(nv):
        falling *= colours - i
    cands.append(Distractor(str(falling), "r3l-chromatic-complete-graph-formula", falling))
    degrees = [sum(1 for e in edges if v in e) for v in range(nv)]
    wrong = colours
    for d in degrees:
        wrong *= max(colours - d, 1)
    cands.append(Distractor(str(wrong), "r3l-chromatic-degree-for-earlier-neighbours", wrong))
    distractors = pick_distractors(str(value), cands, 4)
    check(len(distractors) >= 3, "too few distractors")
    return dict(
        text=f"{_graph_stem(nv, edges)} In how many ways can its vertices be coloured with {colours} colours so that adjacent vertices always get different colours?",
        answer_kind="number",
        answer=value,
        answer_text=str(value),
        answer_latex=str(value),
        working=[
            "Colour the vertices one at a time, each joined to the earlier vertices that it touches: a vertex joined to s mutually adjacent earlier vertices has (k − s) choices.",
            f"P(G, k) = {paper_notation(str(sympy.factor(poly)))}.",
            f"At k = {colours}: {value}.",
        ],
        distractors=distractors,
    )


def _medium(rng):
    nv = 5
    edges = _non_chordal(rng, nv, (5, 7))
    poly = _poly(nv, edges)
    _verify_poly(nv, edges, poly)
    answer = _coef(poly, 2)
    deleted, contracted = _slip_polys(nv, edges)
    cands = []
    for label, p in (
        ("r3l-chromatic-contraction-omitted", deleted),
        ("r3l-chromatic-contraction-sign", sympy.expand(deleted + contracted)),
        ("r3l-chromatic-contraction-only", contracted),
    ):
        v = _coef(p, 2)
        cands.append(Distractor(str(v), label, v))
    cands.append(Distractor(str(-answer), "r3l-chromatic-coefficient-sign", -answer))
    for power, label in ((3, "r3l-chromatic-wrong-coefficient"), (1, "r3l-chromatic-wrong-coefficient")):
        v = _coef(poly, power)
        cands.append(Distractor(str(v), label, v))
    distractors = pick_distractors(str(answer), cands, 4)
    check(len(distractors) >= 3, "too few distractors")
    pretty = str(poly).replace("**", "^")
    return dict(
        text=f"{_graph_stem(nv, edges)} Find the coefficient of k² in its chromatic polynomial P(G, k).",
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            "Use P(G, k) = P(G − e, k) − P(G / e, k) on an edge e, repeating until no edges remain (an edgeless graph on t vertices has k^t).",
            f"P(G, k) = {paper_notation(pretty)}, so the coefficient of k² is {answer}.",
        ],
        distractors=distractors,
    )


def _hard_polynomial_value(rng):
    nv = 6
    edges = _non_chordal(rng, nv, (6, 8))
    poly = _poly(nv, edges)
    _verify_poly(nv, edges, poly)
    colours = rng.choice([3, 4])
    value = _int(poly.subs(_k, colours))
    if value <= 0:
        raise AssertionError  # pragma: no cover
    deleted, contracted = _slip_polys(nv, edges)
    cands = []
    for label, p in (
        ("r3l-chromatic-contraction-omitted", deleted),
        ("r3l-chromatic-contraction-sign", sympy.expand(deleted + contracted)),
    ):
        v = _int(p.subs(_k, colours))
        cands.append(Distractor(str(v), label, v))
    tree = colours * (colours - 1) ** (nv - 1)
    cands.append(Distractor(str(tree), "r3l-chromatic-tree-formula", tree))
    cands.append(Distractor(str(colours**nv), "r3l-chromatic-no-constraint", colours**nv))
    cands.append(
        Distractor(
            str(_int(contracted.subs(_k, colours))),
            "r3l-chromatic-contraction-only",
            _int(contracted.subs(_k, colours)),
        )
    )
    return edges, nv, colours, value, cands, poly


def _hard(rng):
    if rng.random() < 0.5:
        edges, nv, colours, value, cands, poly = _hard_polynomial_value(rng)
        distractors = pick_distractors(str(value), cands, 4)
        check(len(distractors) >= 3, "too few distractors")
        return dict(
            text=f"{_graph_stem(nv, edges)} In how many ways can its vertices be coloured with {colours} colours so that adjacent vertices always get different colours?",
            answer_kind="number",
            answer=value,
            answer_text=str(value),
            answer_latex=str(value),
            working=[
                "Apply P(G, k) = P(G − e, k) − P(G / e, k) until every remaining graph is chordal or edgeless.",
                f"P(G, k) = {paper_notation(str(poly))}; at k = {colours} this is {value}.",
            ],
            distractors=distractors,
        )
    adj, s = _hall_instance(rng)
    best = 5 - 1
    cands = [
        Distractor("5", "r3l-hall-assumes-perfect-matching", 5),
        Distractor(str(len(s)), "r3l-hall-violating-set-size", len(s)),
        Distractor(str(len(_neighbourhood(adj, s))), "r3l-hall-neighbourhood-size", len(_neighbourhood(adj, s))),
        Distractor("1", "r3l-hall-deficiency-for-matching", 1),
        Distractor(str(best - 1), "r3l-hall-matching-off-by-one", best - 1),
    ]
    distractors = pick_distractors(str(best), cands, 4)
    check(len(distractors) >= 3, "too few distractors")
    names = [_WORKERS[x] for x in s]
    return dict(
        text=f"{_hall_stem(adj)} What is the greatest number of the five jobs that can be given out, each to a different worker who can do it?",
        answer_kind="number",
        answer=best,
        answer_text=str(best),
        answer_latex=str(best),
        working=[
            f"{', '.join(names)} can between them only do {len(_neighbourhood(adj, s))} jobs ({', '.join(_JOBS[y] for y in sorted(_neighbourhood(adj, s)))}), so Hall's condition fails and at least {len(s) - len(_neighbourhood(adj, s))} of them is left out.",
            f"The deficiency is {len(s) - len(_neighbourhood(adj, s))}, so the largest matching has 5 − 1 = {best} jobs.",
        ],
        distractors=distractors,
    )


@generator(
    "r3l_chromatic_polynomial_hall",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3l_chromatic_polynomial_hall(rng, tier="easy"):
    """Chromatic polynomials and Hall's condition; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------------


def _chromatic_number(nv, edges):
    for colours in range(1, nv + 1):
        if _brute_count(nv, edges, colours) > 0:
            return colours
    raise AssertionError  # pragma: no cover


def _exam_colouring(rng):
    nv = 5
    edges = _non_chordal(rng, nv, (5, 7))
    poly = _poly(nv, edges)
    _verify_poly(nv, edges, poly)
    chi = _chromatic_number(nv, edges)
    colours = 4
    at4 = _int(poly.subs(_k, colours))
    check(at4 == _brute_count(nv, edges, colours), "brute force differs at k = 4")
    deleted, contracted = _slip_polys(nv, edges)
    text = str(poly).replace("**", "^")
    a_cands = []
    for p, tag in (
        (deleted, "r3l-chromatic-contraction-omitted"),
        (sympy.expand(deleted + contracted), "r3l-chromatic-contraction-sign"),
        (sympy.expand(_k * (_k - 1) ** (nv - 1)), "r3l-chromatic-tree-formula"),
        (contracted * _k, "r3l-chromatic-contraction-only"),
    ):
        t = str(p).replace("**", "^")
        if sympy.expand(p - poly) != 0:
            a_cands.append(Distractor(t, tag, t))
    a_dist = pick_distractors(text, a_cands, 3)
    b_cands = [
        Distractor(
            str(_int(deleted.subs(_k, colours))), "r3l-chromatic-contraction-omitted", _int(deleted.subs(_k, colours))
        ),
        Distractor(
            str(_int(sympy.expand(deleted + contracted).subs(_k, colours))),
            "r3l-chromatic-contraction-sign",
            _int(sympy.expand(deleted + contracted).subs(_k, colours)),
        ),
        Distractor(
            str(colours * (colours - 1) ** (nv - 1)), "r3l-chromatic-tree-formula", colours * (colours - 1) ** (nv - 1)
        ),
        Distractor(str(colours**nv), "r3l-chromatic-no-constraint", colours**nv),
    ]
    b_dist = pick_distractors(str(at4), b_cands, 3)
    c_cands = [
        Distractor(str(chi + 1), "r3l-chromatic-number-off-by-one", chi + 1),
        Distractor(str(max(chi - 1, 1)), "r3l-chromatic-number-off-by-one", max(chi - 1, 1)),
        Distractor(
            str(max(sum(1 for e in edges if v in e) for v in range(nv)) + 1),
            "r3l-chromatic-max-degree-plus-one",
            max(sum(1 for e in edges if v in e) for v in range(nv)) + 1,
        ),
        Distractor(str(nv), "r3l-chromatic-number-is-vertex-count", nv),
    ]
    c_dist = pick_distractors(str(chi), c_cands, 3)
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist)), "too few distractors")
    return dict(
        text=_graph_stem(nv, edges),
        parts=[
            dict(
                prompt="Find the chromatic polynomial P(G, k) of G, as an expression in k.",
                answer_kind="expression",
                answer=text,
                answer_text=text,
                marks=4,
                working=["Use P(G) = P(G − e) − P(G / e) repeatedly.", f"P(G, k) = {paper_notation(text)}."],
                distractors=a_dist,
            ),
            dict(
                prompt="Hence find the number of proper colourings of G with 4 colours.",
                answer_kind="number",
                answer=at4,
                answer_text=str(at4),
                marks=1,
                follow_through="Subs(part_a, (k,), (4,))",
                sensible_range=(0, None),
                working=[f"Substitute k = 4 into P(G, k): {at4}."],
                distractors=b_dist,
            ),
            dict(
                prompt="What is the chromatic number of G: the least number of colours that can properly colour it?",
                answer_kind="number",
                answer=chi,
                answer_text=str(chi),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"P(G, {chi - 1}) = 0 or G cannot be coloured with fewer colours, while P(G, {chi}) > 0, so the chromatic number is {chi}."
                ],
                distractors=c_dist,
            ),
        ],
    )


def _exam_hall(rng):
    adj, s = _hall_instance(rng)
    n_s = _neighbourhood(adj, s)
    deficiency = len(s) - len(n_s)
    best = 5 - deficiency
    names = lambda subset: "{" + ", ".join(_WORKERS[x] for x in subset) + "}"  # noqa: E731
    non_violating = [
        list(t)
        for r in (2, 3, 4)
        for t in combinations(range(5), r)
        if len(_neighbourhood(adj, t)) >= len(t) and list(t) != s
    ]
    rng.shuffle(non_violating)
    check(len(non_violating) >= 3, "not enough non-violating sets")
    a_dist = [Distractor(names(t), "r3l-hall-set-satisfies-condition", names(t)) for t in non_violating[:4]]
    b_dist = pick_distractors(
        str(deficiency),
        [
            Distractor(str(len(s)), "r3l-hall-violating-set-size", len(s)),
            Distractor(str(len(n_s)), "r3l-hall-neighbourhood-size", len(n_s)),
            Distractor(str(deficiency + 1), "r3l-hall-deficiency-off-by-one", deficiency + 1),
            Distractor("0", "r3l-hall-assumes-perfect-matching", 0),
        ],
        3,
    )
    c_dist = pick_distractors(
        str(best),
        [
            Distractor("5", "r3l-hall-assumes-perfect-matching", 5),
            Distractor(str(deficiency), "r3l-hall-deficiency-for-matching", deficiency),
            Distractor(str(best - 1), "r3l-hall-matching-off-by-one", best - 1),
            Distractor(str(len(n_s)), "r3l-hall-neighbourhood-size", len(n_s)),
        ],
        3,
    )
    check(len(b_dist) >= 2 and len(c_dist) >= 2, "too few distractors")
    return dict(
        text=_hall_stem(adj),
        parts=[
            dict(
                prompt="Hall's condition says that every set S of workers can between them do at least |S| jobs. Which of these sets of workers shows that the condition fails?",
                answer_kind="mcq",
                answer=names(s),
                answer_text=names(s),
                marks=2,
                working=[f"{names(s)} can only do {len(n_s)} jobs between them, which is fewer than {len(s)}."],
                distractors=a_dist,
            ),
            dict(
                prompt="Find the deficiency: the largest value of |S| − (number of jobs that S can do), over all sets S of workers.",
                answer_kind="number",
                answer=deficiency,
                answer_text=str(deficiency),
                marks=2,
                working=[f"|S| − |N(S)| = {len(s)} − {len(n_s)} = {deficiency} for {names(s)}, and no set does worse."],
                distractors=b_dist,
            ),
            dict(
                prompt="What is the greatest number of the five jobs that can be given out, each to a different worker who can do it?",
                answer_kind="number",
                answer=best,
                answer_text=str(best),
                marks=2,
                follow_through="5 - part_b",
                sensible_range=(0, 5),
                working=[f"The largest matching has 5 − {deficiency} = {best} jobs."],
                distractors=c_dist,
            ),
        ],
    )


@generator("exd_x_r3l_chromatic_polynomial_hall", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r3l_chromatic_polynomial_hall(rng, tier="exam"):
    """Exam chain: colouring a graph (polynomial, a value, chromatic number) or Hall's condition (set, deficiency, matching)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _exam_colouring(rng) if rng.random() < 0.5 else _exam_hall(rng)
