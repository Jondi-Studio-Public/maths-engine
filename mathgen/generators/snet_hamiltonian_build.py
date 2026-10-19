"""Generates `snet_hamiltonian` questions (easy, medium, hard) and the exam chain `snet_hamiltonian_chain`.

``graph-hamiltonian-paths-cycles`` (``financial-discrete``, answer ``number``): a Hamiltonian path visits every vertex
exactly once; a Hamiltonian cycle also returns to its start. Both answers here are counts, found by trying every
ordering of the vertices.

* easy: a five-vertex network and four routes, each written as the vertices visited in order. How many of the routes
  are Hamiltonian cycles? The routes that are not fail in one stated way: they skip a vertex, repeat a vertex, do not
  return to the start, or use a pair of vertices that no edge joins;
* medium: how many Hamiltonian cycles does a five-vertex network have (a cycle, its reverse, and the same cycle
  started elsewhere count as one)?
* hard: how many Hamiltonian paths start at a named vertex of a six-vertex network?
* exam (``snet_hamiltonian_chain`` ): a network of roads with distances. (a) how many Hamiltonian cycles; (b) how
  many tours if the start and direction matter, following through from (a); (c) the shortest total distance of
  a tour; (d) the fuel cost of that tour at a stated price per kilometre, following through from (c).

Every count is made from the permutations of the vertices; each route is also checked against the definition one
step at a time. The exam's cheapest tour is the unique minimum over every Hamiltonian cycle.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations

from ..answers import Distractor
from ..core import check, generator, numeric_answer, phrase, pick_distractors
from ._snet_kit import (
    LETTERS,
    CONTEXTS,
    describe_network,
    edge_name,
    hamiltonian_cycles,
    hamiltonian_paths_from,
    int_distractors,
    is_walk,
    norm,
    pick_context,
    random_connected,
    route_of_class,
    route_text,
)
from ._network_fig import with_network_figure


def is_hamiltonian_cycle(route, n, edges) -> bool:
    """Straight from the definition: n + 1 vertices, closed, each vertex once, every step an edge."""
    route = list(route)
    return (
        len(route) == n + 1
        and route[0] == route[-1]
        and len(set(route[:-1])) == n
        and all(norm(route[i], route[i + 1]) in set(edges) for i in range(n))
    )


# -- easy -------------------------------------------------------------------------------------------------------

_TAG = {
    "skip": "net-hamilton-skipped-vertex-accepted",
    "repeat": "net-hamilton-repeat-accepted",
    "open": "net-hamilton-path-counted-as-cycle",
    "nonedge": "net-hamilton-non-edge-accepted",
}
_EASY_STEMS = (
    "{net} A Hamiltonian cycle visits every {v} exactly once and returns to the {v} it started at. "
    "Four routes are written as the {vs} visited in order: {routes}. How many of the four routes are Hamiltonian cycles?",
    "{net} Four routes are written as the {vs} visited in order: {routes}. "
    "A Hamiltonian cycle visits every {v} exactly once and ends back at its start. "
    "How many of these routes are Hamiltonian cycles of the network?",
)


def _failing_route(rng, kind, n, edges):
    """A route that is not a Hamiltonian cycle, failing in the stated way."""
    if kind == "skip":
        for _ in range(100):
            route = route_of_class(rng, n, edges, "cycle", lo=3, hi=n)
            if len(set(route)) < n:
                return route
        return None
    if kind == "repeat":
        for _ in range(100):
            route = route_of_class(rng, n, edges, "closed walk", lo=4, hi=n + 1)
            if len(set(route[:-1])) < len(route) - 1:
                return route
        return None
    if kind == "open":
        start = rng.randrange(n)
        paths = hamiltonian_paths_from(n, edges, start)
        return list(rng.choice(paths)) if paths else None
    for _ in range(100):  # a closed tour of every vertex once that is not made of edges
        order = list(range(n))
        rng.shuffle(order)
        route = [*order, order[0]]
        if not is_walk(route, edges):
            return route
    return None


def _easy(rng):
    ctx = pick_context(rng, travel=True)
    n = 5
    for _ in range(100):
        edges = random_connected(rng, n, rng.randint(6, 8))
        cycles = hamiltonian_cycles(n, edges)
        if cycles:
            break
    else:
        raise AssertionError  # pragma: no cover
    k = rng.randint(0, min(3, len(cycles)))
    kinds = (
        rng.sample(["skip", "repeat", "open", "nonedge"], 4 - k)
        if k >= 1
        else rng.sample(["skip", "repeat", "open", "nonedge"], 4)
    )
    routes: list[tuple[list[int], str | None]] = []
    for cyc in rng.sample(cycles, k):
        start = rng.randrange(n)
        rotated = list(cyc[start:] + cyc[:start])
        if rng.random() < 0.5:
            rotated = [rotated[0], *reversed(rotated[1:])]
        routes.append(([*rotated, rotated[0]], None))
    for kind in kinds:
        route = _failing_route(rng, kind, n, edges)
        if route is None:
            return _easy(rng)
        routes.append((route, kind))
    rng.shuffle(routes)
    if len({route_text(r) for r, _ in routes}) < 4:
        return _easy(rng)
    for r, kind in routes:
        check(is_hamiltonian_cycle(r, n, edges) == (kind is None), "route checked against the definition differs")
    good = sum(kind is None for _, kind in routes)
    check(good == k, "count of Hamiltonian cycles differs")
    fails = [kind for _, kind in routes if kind]
    candidates = [(good + sum(1 for f in fails if f == kind), _TAG[kind]) for kind in dict.fromkeys(fails)]
    candidates += [
        (good - 1, "net-hamilton-valid-rejected"),
        (good + 1, "net-hamilton-miscount"),
        (good + 2, "net-hamilton-miscount"),
    ]
    distractors = int_distractors(good, candidates)
    check(len(distractors) >= 2, "too few distractors")
    reasons = {
        "skip": "misses a vertex",
        "repeat": "repeats a vertex",
        "open": "does not return to its start",
        "nonedge": "uses a step that no edge joins",
    }
    working = [
        f"{route_text(r)}: "
        + ("a Hamiltonian cycle." if kind is None else f"not a Hamiltonian cycle, because it {reasons[kind]}.")
        for r, kind in routes
    ]
    working.append(f"So the number of Hamiltonian cycles among the four routes is {good}.")
    text = phrase(
        rng,
        _EASY_STEMS,
        net=describe_network(ctx, n, edges),
        v=ctx["v"],
        vs=ctx["vs"],
        routes=", ".join(route_text(r) for r, _ in routes),
    )
    return _number(text, good, working, distractors)


def _number(text, answer, working, distractors):
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

_MEDIUM_STEMS = (
    "{net} A Hamiltonian cycle visits every {v} exactly once and returns to its start. A cycle, the same cycle "
    "travelled in reverse, and the same cycle started at a different {v} all count as one cycle. "
    "How many Hamiltonian cycles does the network have?",
    "{net} How many different Hamiltonian cycles does the network have? A Hamiltonian cycle visits every {v} once "
    "and returns to its start; a cycle in reverse, or started at another {v}, is the same cycle.",
)


def _medium(rng):
    ctx = pick_context(rng, travel=True)
    for _ in range(200):
        n = rng.choice([5, 5, 6])
        edges = random_connected(rng, n, rng.randint(n + 2, n + 4))
        cycles = hamiltonian_cycles(n, edges)
        if 1 <= len(cycles) <= 6:
            break
    else:
        raise AssertionError  # pragma: no cover
    h = len(cycles)
    # Independent route: count the ordered tours starting at A, then divide by the two directions.
    ordered = sum(
        1 for p in permutations(range(1, n)) if all(norm(a, b) in set(edges) for a, b in zip((0, *p), (*p, 0)))
    )
    check(ordered == 2 * h, "ordered tours are not twice the cycles")
    candidates = [
        (2 * h, "net-hamilton-direction-counted"),
        (n * h, "net-hamilton-start-counted"),
        (2 * n * h, "net-hamilton-all-orderings-counted"),
        (h + 1, "net-hamilton-miscount"),
        (h - 1, "net-hamilton-miscount"),
    ]
    distractors = int_distractors(h, candidates, lo=1)
    check(len(distractors) >= 2, "too few distractors")
    working = ["The Hamiltonian cycles, each written once, are:"] + [route_text([*c, c[0]]) for c in cycles]
    working.append(f"So the number of Hamiltonian cycles is {h}.")
    text = phrase(rng, _MEDIUM_STEMS, net=describe_network(ctx, n, edges), v=ctx["v"])
    return _number(text, h, working, distractors)


# -- hard -------------------------------------------------------------------------------------------------------

_HARD_STEMS = (
    "{net} A Hamiltonian path visits every {v} exactly once and need not return to its start. "
    "Two paths are different if they visit the {vs} in a different order. "
    "How many Hamiltonian paths start at {s}?",
    "{net} How many Hamiltonian paths start at {s}? A Hamiltonian path visits every {v} exactly once and does not "
    "have to return; paths that visit the {vs} in a different order are different.",
)


def _hard(rng):
    ctx = pick_context(rng, travel=True)
    n = 6
    for _ in range(300):
        edges = random_connected(rng, n, rng.randint(8, 10))
        s = rng.randrange(n)
        paths = hamiltonian_paths_from(n, edges, s)
        closing = [p for p in paths if norm(p[-1], s) in set(edges)]
        if 2 <= len(paths) <= 8 and closing:
            break
    else:
        raise AssertionError  # pragma: no cover
    answer = len(paths)
    # Independent route: depth-first search that extends a path one vertex at a time.
    nbr = {v: [w for w in range(n) if norm(v, w) in set(edges)] for v in range(n)}

    def extend(path):
        if len(path) == n:
            return 1
        return sum(extend([*path, w]) for w in nbr[path[-1]] if w not in path)

    check(extend([s]) == answer, "depth-first count differs")
    any_start = sum(len(hamiltonian_paths_from(n, edges, v)) for v in range(n))
    candidates = [
        (len(closing), "net-hamilton-path-cycle-confused"),
        (any_start if any_start <= 3 * answer else -1, "net-hamilton-any-start-counted"),
        (answer + 1, "net-hamilton-miscount"),
        (answer - 1, "net-hamilton-miscount"),
        (answer * 2, "net-hamilton-direction-counted"),
        (answer + 2, "net-hamilton-miscount"),
    ]
    distractors = int_distractors(answer, candidates, lo=1)
    check(len(distractors) >= 2, "too few distractors")
    working = [f"The Hamiltonian paths that start at {LETTERS[s]} are:"] + [route_text(p) for p in paths]
    working.append(f"So the number of Hamiltonian paths that start at {LETTERS[s]} is {answer}.")
    text = phrase(rng, _HARD_STEMS, net=describe_network(ctx, n, edges), v=ctx["v"], vs=ctx["vs"], s=LETTERS[s])
    return _number(text, answer, working, distractors)


@generator(
    "snet_hamiltonian",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@with_network_figure
def snet_hamiltonian(rng, tier="easy"):
    """Hamiltonian paths and cycles as counts; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------

_TOUR_TOWNS = CONTEXTS[0]
_TOUR_INTROS = (
    "A coach company is planning a tour of {n} country towns, {a} to {z}, using direct roads between them.",
    "A delivery driver must call at {n} country towns, {a} to {z}, using the direct roads between them.",
)
_PRICE = Fraction(15, 100)  # dollars per kilometre


def _cycle_length(cycle, weight):
    n = len(cycle)
    return sum(weight[norm(cycle[i], cycle[(i + 1) % n])] for i in range(n))


def _money(value: Fraction) -> str:
    return f"{float(value):.2f}"


@generator("snet_hamiltonian_chain", level=12, presets={"exam": dict(tier="exam")})
@with_network_figure
def snet_hamiltonian_chain(rng, tier="exam"):
    """Exam chain on a network of roads with distances: tours counted, then the cheapest tour and its fuel cost."""
    check(tier == "exam", f"unknown tier {tier!r}")
    ctx = _TOUR_TOWNS
    target = rng.choice((1, 2, 3, 4, 5, 6))
    for _ in range(3000):
        n = rng.randint(5, 7)
        edges = random_connected(rng, n, rng.randint(n + 1, min(n + 6, n * (n - 1) // 2)))
        cycles = hamiltonian_cycles(n, edges)
        if len(cycles) != target:
            continue
        km = rng.sample(range(20, 96), len(edges))
        weight = {e: w for e, w in zip(edges, km)}
        lengths = sorted(_cycle_length(c, weight) for c in cycles)
        if len(lengths) == 1 or lengths[0] != lengths[1]:
            break
    else:
        raise AssertionError  # pragma: no cover
    h = len(cycles)
    best = lengths[0]
    cost = best * _PRICE
    # Independent route: every ordering of the towns, summing the distances, keeping the cheapest complete tour.
    tours = []
    for p in permutations(range(1, n)):
        tour = (0, *p)
        if all(norm(tour[i], tour[(i + 1) % n]) in weight for i in range(n)):
            tours.append(_cycle_length(tour, weight))
    check(len(tours) == 2 * h and min(tours) == best, "the search over every ordering differs")
    check(best * Fraction(15, 100) == Fraction(3 * best, 20), "fuel cost arithmetic differs")

    # Wrong values for (c): a spanning tree, the dearest tour, the next-dearest tour, and the cheapest n roads.
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = 0
    for e in sorted(edges, key=lambda e: weight[e]):
        a, b = find(e[0]), find(e[1])
        if a != b:
            parent[a] = b
            tree += weight[e]
    cheapest_n = sum(sorted(weight.values())[:n])
    stem = (
        f"{rng.choice(_TOUR_INTROS).format(n=n, a=LETTERS[0], z=LETTERS[n - 1])} Each pair of letters names the two "
        "towns one road joins, and the distances in kilometres are: "
        + ", ".join(f"{edge_name(e)} {weight[e]}" for e in edges)
        + ". A tour visits every town exactly once and returns to the town it started at."
    )
    a_dist = int_distractors(
        h,
        [
            (2 * h, "net-hamilton-direction-counted"),
            (n * h, "net-hamilton-start-counted"),
            (h + 1, "net-hamilton-miscount"),
            (h - 1, "net-hamilton-miscount"),
        ],
        lo=1,
    )
    b_dist = int_distractors(
        2 * n * h,
        [
            (n * h, "net-hamilton-direction-ignored"),
            (2 * h, "net-hamilton-start-ignored"),
            (h, "net-hamilton-start-ignored"),
            (2 * n * (h + 1), "net-hamilton-miscount"),
        ],
        lo=1,
    )
    c_dist = int_distractors(
        best,
        [
            (tree, "net-spanning-tree-instead-of-tour"),
            (lengths[-1], "net-longest-tour-chosen"),
            (lengths[1] if h > 1 else best + 10, "net-second-cheapest-tour-chosen"),
            (cheapest_n, "net-cheapest-roads-chosen"),
        ],
        lo=1,
    )
    d_values = [
        (best * 15, "net-cents-as-dollars"),
        (best / _PRICE, "net-divided-by-price"),
        (2 * cost, "net-doubled-return"),
        (best * Fraction(15, 1000), "net-price-decimal-slip"),
    ]
    d_dist = []
    seen = {_money(cost)}
    for value, tag in d_values:
        text = _money(Fraction(value))
        if text not in seen and Fraction(value) > 0:
            seen.add(text)
            d_dist.append(Distractor(text, tag, numeric_answer(Fraction(text))))
    d_dist = pick_distractors(_money(cost), d_dist, 3)
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"How many different Hamiltonian cycles does the network have? A cycle, the same cycle in reverse and "
                f"the same cycle started at another {ctx['v']} count as one.",
                answer_kind="number",
                answer=h,
                answer_text=str(h),
                answer_latex=str(h),
                marks=2,
                sensible_range=(0, None),
                working=["The Hamiltonian cycles, each written once, are:"] + [route_text([*c, c[0]]) for c in cycles],
                distractors=a_dist,
            ),
            dict(
                prompt=f"Hence find how many different tours there are if tours are told apart by the {ctx['v']} they "
                "start at and the direction they travel.",
                answer_kind="number",
                answer=2 * n * h,
                answer_text=str(2 * n * h),
                answer_latex=str(2 * n * h),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_a*{2 * n}",
                working=[
                    f"Each cycle can start at any of {n} {ctx['vs']} and go either way: {h} x {n} x 2 = {2 * n * h}."
                ],
                distractors=b_dist,
            ),
            dict(
                prompt="Find the shortest total distance, in kilometres, of a tour.",
                answer_kind="number",
                answer=best,
                answer_text=str(best),
                answer_latex=str(best),
                marks=2,
                sensible_range=(0, None),
                working=[
                    "The total distance of each Hamiltonian cycle: "
                    + ", ".join(f"{route_text([*c, c[0]])} = {_cycle_length(c, weight)} km" for c in cycles)
                    + ".",
                    f"The shortest is {best} km.",
                ],
                distractors=c_dist,
            ),
            dict(
                prompt="Fuel for the vehicle costs $0.15 per kilometre. Hence find the fuel cost of the shortest tour, in dollars, "
                "correct to 2 decimal places.",
                answer_kind="number",
                answer=numeric_answer(Fraction(_money(cost))),
                answer_text=_money(cost),
                answer_latex=_money(cost),
                marks=1,
                sensible_range=(0, None),
                follow_through="0.15*part_c",
                working=[f"{best} x 0.15 = {_money(cost)}, so the fuel costs ${_money(cost)}."],
                distractors=d_dist,
            ),
        ],
    )
