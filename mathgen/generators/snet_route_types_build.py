"""Generates `snet_route_types` questions (easy, medium, hard) and the exam chain `snet_route_types_chain`.

``graph-walks-trails-paths-cycles`` (``financial-discrete``, answer ``mcq``): name a route through a network as a
walk, trail, path, closed walk, closed trail (circuit) or cycle.

A route is the list of vertices it visits in order, ``B-C-A-D-A``. It is a walk if each step uses an edge of the
network; closed if it ends where it starts. The six names, most specific last on each side:

* open: walk (an edge may repeat), trail (no repeated edge), path (no repeated vertex);
* closed: closed walk (an edge may repeat), closed trail or circuit (no repeated edge), cycle (no repeated vertex
  apart from the end meeting the start).

* easy: name an open route, with all six names on offer and the most specific one wanted;
* medium: name a route that may be open or closed;
* hard: four routes are listed, each of a different kind, and the one that is a stated kind is chosen;
* exam (``snet_route_types_chain``): a route that re-uses an edge. (a) the steps it takes, (b) the different edges
  it uses, (c) how often it re-uses an edge (follows through from (a) and (b)), (d) its most specific name.

Every network is built to contain all six kinds of route (two triangles sharing a vertex, a closing vertex and a few
more edges). The kind of a route is found by ``classify_route`` and again by a direct definition in this module.
"""

from __future__ import annotations

from ..answers import Distractor
from ..core import check, generator, phrase
from ._snet_kit import (
    CLASS_OPTION,
    CLOSED_CLASSES,
    OPEN_CLASSES,
    all_pairs,
    classify_route,
    describe_network,
    int_distractors,
    is_walk,
    norm,
    pick_context,
    route_of_class,
    route_text,
)

ORDER = ("walk", "trail", "path", "closed walk", "closed trail", "cycle")


def route_network(rng):
    """(6, edges): two triangles sharing a vertex, a vertex closing a five-cycle, and one or two more edges, with the
    vertices shuffled. Every kind of route exists in it."""
    n = 6
    base = {(0, 1), (1, 2), (0, 2), (0, 3), (3, 4), (0, 4), (2, 5), (4, 5)}
    extra = [p for p in all_pairs(n) if p not in base and p != (0, 5)]
    rng.shuffle(extra)
    base.update(extra[: rng.randint(1, 2)])
    perm = list(range(n))
    rng.shuffle(perm)
    return n, sorted(norm(perm[u], perm[v]) for u, v in base)


def _direct_class(route, edges) -> str:
    """The kind of route straight from the definitions, as a check on `classify_route`."""
    check(is_walk(route, edges), "the route is not a walk of the network")
    steps = [frozenset(route[i : i + 2]) for i in range(len(route) - 1)]
    closed = route[0] == route[-1]
    no_repeat_edge = len(set(steps)) == len(steps)
    interior = route[:-1] if closed else route
    no_repeat_vertex = len(set(interior)) == len(interior)
    if closed:
        return "closed walk" if not no_repeat_edge else ("cycle" if no_repeat_vertex else "closed trail")
    return "walk" if not no_repeat_edge else ("path" if no_repeat_vertex else "trail")


def wrong_class_tag(true: str, picked: str) -> str:
    """The error behind calling a route of kind ``true`` by the name ``picked``."""
    if (true in OPEN_CLASSES) != (picked in OPEN_CLASSES):
        return "net-open-closed-confused"
    scale = OPEN_CLASSES if true in OPEN_CLASSES else CLOSED_CLASSES
    if scale.index(picked) < scale.index(true):
        return "net-route-too-general"
    if true in ("walk", "closed walk"):
        return "net-trail-allows-repeated-edge"
    if scale is OPEN_CLASSES:
        return "net-path-allows-repeated-vertex"
    return "net-circuit-cycle-confused"


_NAME_STEMS = (
    "{net} A {who} follows the route {route}. Which name describes this route most precisely?",
    "{net} The route {route} is followed by a {who}. Choose the most specific name for the route.",
)
_WHO = ("walker", "cyclist", "visitor", "courier")


def _name_options(true):
    key = CLASS_OPTION[true]
    wrong = [
        Distractor(CLASS_OPTION[c], wrong_class_tag(true, c)) for c in ORDER if c != true and CLASS_OPTION[c] != key
    ]
    return key, wrong


def _name_working(route, edges, cls):
    steps = [f"{route_text(edge)}" for edge in [(route[i], route[i + 1]) for i in range(len(route) - 1)]]
    closed = route[0] == route[-1]
    return [
        f"The route {route_text(route)} {'ends where it starts, so it is closed' if closed else 'ends at a different vertex from where it starts, so it is open'}.",
        f"Its steps are {', '.join(steps)}.",
        f"Checking for a repeated edge and then a repeated vertex, the most specific name is: {CLASS_OPTION[cls].lower()}.",
    ]


def _name_question(rng, classes):
    ctx = pick_context(rng, travel=True)
    n, edges = route_network(rng)
    cls = rng.choice(classes)
    route = route_of_class(rng, n, edges, cls)
    check(classify_route(route) == _direct_class(route, edges) == cls, "the two classifications differ")
    key, wrong = _name_options(cls)
    text = phrase(rng, _NAME_STEMS, net=describe_network(ctx, n, edges), who=rng.choice(_WHO), route=route_text(route))
    return dict(
        text=text,
        answer_kind="mcq",
        answer=key,
        answer_text=key,
        working=_name_working(route, edges, cls),
        distractors=wrong,
    )


_HARD_TARGETS = {
    "walk": "a walk that is not a trail",
    "trail": "a trail that is not a path",
    "path": "a path",
    "closed walk": "a closed walk that is not a closed trail",
    "closed trail": "a closed trail that is not a cycle",
    "cycle": "a cycle",
}
_HARD_STEMS = (
    "{net} Four routes are described by the vertices they visit in order. Which one of them is {target}?",
    "{net} Which of these four routes is {target}?",
)


def _hard(rng):
    ctx = pick_context(rng, travel=True)
    for _ in range(50):
        n, edges = route_network(rng)
        target = rng.choice(ORDER)
        others = rng.sample([c for c in ORDER if c != target], 3)
        routes = {c: route_of_class(rng, n, edges, c) for c in [target, *others]}
        if len({route_text(r) for r in routes.values()}) == 4:
            break
    else:
        raise AssertionError  # pragma: no cover
    for c, r in routes.items():
        check(classify_route(r) == _direct_class(r, edges) == c, "the two classifications differ")
    key = route_text(routes[target])
    wrong = [Distractor(route_text(routes[c]), wrong_class_tag(c, target)) for c in others]
    text = phrase(rng, _HARD_STEMS, net=describe_network(ctx, n, edges), target=_HARD_TARGETS[target])
    working = [
        f"{route_text(routes[c])} is {'' if c == target else 'not '}{_HARD_TARGETS[target]}: it is a {CLASS_OPTION[c].lower()}."
        for c in [target, *others]
    ]
    return dict(
        text=text,
        answer_kind="mcq",
        answer=key,
        answer_text=key,
        working=working,
        distractors=wrong,
    )


@generator(
    "snet_route_types",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def snet_route_types(rng, tier="easy"):
    """Name a route through a network; the tiers are described in the module docstring."""
    if tier == "easy":
        return _name_question(rng, OPEN_CLASSES)
    if tier == "medium":
        return _name_question(rng, ORDER)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("snet_route_types_chain", level=12, presets={"exam": dict(tier="exam")})
def snet_route_types_chain(rng, tier="exam"):
    """Exam chain on one route that re-uses an edge: its steps, its different edges, the re-uses, its name."""
    check(tier == "exam", f"unknown tier {tier!r}")
    ctx = pick_context(rng, travel=True)
    n, edges = route_network(rng)
    cls = rng.choice(["walk", "closed walk"])
    for _ in range(200):
        route = route_of_class(rng, n, edges, cls, lo=4, hi=7)
        steps = [norm(route[i], route[i + 1]) for i in range(len(route) - 1)]
        if len(set(steps)) < len(steps) <= 7 and len(set(route)) >= 3:
            break
    else:
        raise AssertionError  # pragma: no cover
    check(classify_route(route) == _direct_class(route, edges) == cls, "the two classifications differ")
    a = len(steps)
    b = len(set(steps))
    c = a - b
    distinct_vertices = len(set(route))
    stem = (
        f"{describe_network(ctx, n, edges)} A {rng.choice(_WHO)} follows the route {route_text(route)}, "
        f"which is written as the {ctx['vs']} visited in order."
    )
    key, wrong = _name_options(cls)

    a_dist = int_distractors(
        a,
        [
            (len(route), "net-vertices-counted-as-edges"),
            (b, "net-repeats-not-counted"),
            (a - 1, "net-handshake-slip"),
            (a + 1, "net-handshake-slip"),
        ],
    )
    b_dist = int_distractors(
        b,
        [
            (a, "net-repeats-counted-as-different"),
            (distinct_vertices, "net-vertices-counted-as-edges"),
            (b + 1, "net-handshake-slip"),
            (len(edges), "net-all-network-edges-counted"),
        ],
    )
    c_dist = int_distractors(
        c,
        [
            (a, "net-repeats-counted-as-different"),
            (b, "net-different-edges-for-reuse"),
            (len(route) - distinct_vertices, "net-repeated-vertices-for-edges"),
            (c + 1, "net-handshake-slip"),
        ],
    )
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist)), "too few distractors")
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="How many steps (edges, counting any that are used more than once) does the route take?",
                answer_kind="number",
                answer=a,
                answer_text=str(a),
                answer_latex=str(a),
                marks=1,
                sensible_range=(0, None),
                working=[
                    f"The route {route_text(route)} visits {len(route)} {ctx['vs']} in order, so it takes {a} steps."
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=f"How many different {ctx['es']} does the route use?",
                answer_kind="number",
                answer=b,
                answer_text=str(b),
                answer_latex=str(b),
                marks=2,
                sensible_range=(0, None),
                working=[
                    "The steps are " + ", ".join(route_text(s) for s in steps) + ".",
                    f"Counting each {ctx['e']} once, {b} different {ctx['es']} are used.",
                ],
                distractors=b_dist,
            ),
            dict(
                prompt=f"Hence find how many times the route uses a {ctx['e']} that it has already used.",
                answer_kind="number",
                answer=c,
                answer_text=str(c),
                answer_latex=str(c),
                marks=1,
                sensible_range=(-1, None),
                follow_through="part_a - part_b",
                working=[
                    f"The route takes {a} steps but uses only {b} different {ctx['es']}, so {a} - {b} = {c} is the number of steps that use a {ctx['e']} already used."
                ],
                distractors=c_dist,
            ),
            dict(
                prompt="Give the most specific name for the route.",
                answer_kind="mcq",
                answer=key,
                answer_text=key,
                marks=2,
                working=_name_working(route, edges, cls),
                distractors=wrong,
            ),
        ],
    )
