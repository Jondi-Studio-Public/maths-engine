"""Generates `graph-planar-faces` questions (easy, medium, hard) and the exam chain `sr2a_planar_faces_chain`.

Planar graphs and the edge-count test (ACMGM081, `financial-discrete`, answer ``number``). No figure is drawn: a
network is stated by the roads (or cables, links) that join its places, written as pairs of letters, and the prompt
says it is drawn on a map with no two roads crossing.

* easy: the regions (faces) of a drawn network given as an edge list, the outside region counted. The network is a
  ring of places with some non-crossing cross-links, so the learner counts places and links from the list and applies
  Euler's formula;
* medium: the edge-count test. A connected simple planar graph with v places (v at least 3) has at most 3v - 6
  roads, or at most 2v - 4 when no three places are joined in a triangle (every place of one group is joined to every
  place of the other). Each prompt states the rule it needs. For a complete network of n places, or one joining
  every place of one group to every place of another, find the fewest roads that must be removed before the network
  could be planar (the excess over the bound);
* hard: three reasoning questions that use the bound and Euler's formula together: how many more roads can be added
  to a stated planar network before another would have to cross; how many regions the network has once no more can be
  added (the answer does not depend on how many roads it started with); and how many places a network has when every
  region, the outside one included, is bounded by the same number of roads;
* exam (``sr2a_planar_faces_chain``): a complete network of n places: (a) its roads, (b) the fewest to remove, (c) the
  regions of the network left when exactly that many are removed and every region is a triangle, (d) whether the
  original network can be drawn with no crossings.

Every answer is found twice: by the construction that built the network, and by a direct count over its edge list
(or, for the bound questions, by solving Euler's formula with SymPy).
"""

from __future__ import annotations

from itertools import combinations

import sympy

from ..core import attempts, check, generator, phrase
from ._sr2a_kit import (
    LETTERS,
    edges_text,
    mcq_part,
    norm,
    number_part,
    single,
)

_PLACES = (
    dict(vs="towns", es="roads", e="road", intro="Towns {a} to {z} in a region are joined by roads."),
    dict(vs="islands", es="ferry routes", e="ferry route", intro="A ferry company serves islands {a} to {z}."),
    dict(vs="stations", es="rail links", e="rail link", intro="A rail network joins stations {a} to {z}."),
    dict(
        vs="suburbs", es="bike paths", e="bike path", intro="A council has built bike paths between suburbs {a} to {z}."
    ),
)


def _euler_faces(v: int, e: int) -> int:
    """Faces of a connected plane graph from v - e + f = 2, solved by SymPy."""
    f = sympy.Symbol("f")
    (solution,) = sympy.solve(sympy.Eq(v - e + f, 2), f)
    return int(solution)


def _crosses(p, q) -> bool:
    (a, b), (c, d) = sorted(p), sorted(q)
    return a < c < b < d or c < a < d < b


def _ring_with_chords(rng, n: int, k: int):
    """A ring on 0..n-1 plus ``k`` pairwise non-crossing chords (an outerplanar, so planar, graph)."""
    ring = [norm(i, (i + 1) % n) for i in range(n)]
    diagonals = [(a, b) for a, b in combinations(range(n), 2) if b - a not in (1, n - 1)]
    for _ in attempts(500, "non-crossing chords"):
        pool = diagonals[:]
        rng.shuffle(pool)
        chosen: list = []
        for d in pool:
            if len(chosen) < k and not any(_crosses(d, c) for c in chosen):
                chosen.append(d)
        if len(chosen) == k:
            return ring, chosen
    raise AssertionError  # pragma: no cover


def _relabel(rng, n: int, edges):
    perm = list(range(n))
    rng.shuffle(perm)
    return sorted(norm(perm[u], perm[v]) for u, v in edges)


# -- easy -------------------------------------------------------------------------------------------------------

_EASY_ASK = (
    "The network is drawn on a map so that no two {es} cross. Into how many regions does it divide the map, "
    "counting the region outside the network?",
    "Drawn with no crossing {es}, the network divides the map into regions, one of which is the unbounded region "
    "around the outside. How many regions are there altogether?",
)


def _easy(rng):
    ctx = rng.choice(_PLACES)
    for _ in attempts(200, "an easy planar network"):
        k = rng.randint(2, 5)
        n = rng.randint(max(7, k + 3), 9)
        ring, chords = _ring_with_chords(rng, n, k)
        edges = _relabel(rng, n, ring + chords)
        e = len(edges)
        faces = k + 2
        # Second route: each chord splits one inside region in two, starting from one inside region and the outside.
        check(_euler_faces(n, e) == faces == 1 + (k + 1), "faces by Euler's formula and by splitting differ")
        wrong = [
            (faces - 1, "sr2a-planar-forgot-outer"),
            (e - n, "sr2a-planar-faces-edges-minus-vertices"),
            (e, "sr2a-planar-faces-as-edges"),
            (n + e - 2, "sr2a-planar-euler-signs"),
            (faces + 1, "sr2a-planar-euler-signs"),
        ]
        intro = ctx["intro"].format(a=LETTERS[0], z=LETTERS[n - 1])
        text = f"{intro} The {ctx['es']} are: {edges_text(edges)}. " + phrase(rng, _EASY_ASK, es=ctx["es"])
        working = [
            f"There are {n} {ctx['vs']} and {e} {ctx['es']}.",
            f"Euler's formula v - e + f = 2 gives f = 2 - {n} + {e} = {faces}, and this count includes the outside region.",
        ]
        q = single(text, faces, working, wrong, lo=1)
        if q:
            return q
    raise AssertionError  # pragma: no cover


# -- medium -----------------------------------------------------------------------------------------------------

_RULE_ANY = "A connected simple planar graph with v vertices (v at least 3) has at most 3v - 6 edges."
_RULE_BIPARTITE = (
    "A connected simple planar graph with v vertices (v at least 3) in which no three vertices are joined in a "
    "triangle has at most 2v - 4 edges."
)

_COMPLETE_ASK = (
    "At least how many {es} must be left out before the network could be drawn with no crossing {es}?",
    "What is the least number of these {es} that must be removed before it is possible to draw the network with no "
    "{e} crossing another?",
)


def _bound_sizes(rng):
    """``("complete", n)`` or ``("bipartite", a, b)`` with a positive excess over the bound."""
    if rng.random() < 0.55:
        return ("complete", rng.randint(5, 8))
    return ("bipartite", *rng.choice([(3, 3), (3, 4), (3, 5), (4, 4), (3, 6), (4, 5)]))


def _bound_facts(shape):
    if shape[0] == "complete":
        n = shape[1]
        v, e, bound = n, n * (n - 1) // 2, 3 * n - 6
        check(e == len(list(combinations(range(n), 2))), "edge count of the complete network differs")
    else:
        a, b = shape[1], shape[2]
        v, e, bound = a + b, a * b, 2 * (a + b) - 4
        check(e == sum(1 for _ in ((i, j) for i in range(a) for j in range(b))), "edge count differs")
    return v, e, bound


def _medium(rng):
    ctx = rng.choice(_PLACES)
    for _ in attempts(200, "a medium bound question"):
        shape = _bound_sizes(rng)
        v, e, bound = _bound_facts(shape)
        excess = e - bound
        if excess < 1:
            continue
        if shape[0] == "complete":
            setup = (
                f"{v} {ctx['vs']} are to be joined so that every {ctx['vs'][:-1]} has its own direct {ctx['e']} "
                f"to every other. {_RULE_ANY}"
            )
            rule_bound = 3 * v - 6
            other = 2 * v - 4
        else:
            a, b = shape[1], shape[2]
            setup = (
                f"A council plans a separate cable from each of {a} houses to each of {b} depots, {a * b} cables "
                f"in all, with no house joined to another house and no depot to another depot. {_RULE_BIPARTITE}"
            )
            rule_bound = 2 * v - 4
            other = 3 * v - 6
        text = (
            setup
            + " "
            + phrase(
                rng,
                _COMPLETE_ASK,
                es="cables" if shape[0] == "bipartite" else ctx["es"],
                e="cable" if shape[0] == "bipartite" else ctx["e"],
            )
        )
        wrong = [
            (rule_bound, "sr2a-planar-bound-as-removals"),
            (e, "sr2a-planar-edges-not-excess"),
            (e - other if other != rule_bound else None, "sr2a-planar-wrong-bound"),
            (excess + 1, "sr2a-planar-bound-arithmetic"),
            (excess - 1, "sr2a-planar-bound-arithmetic"),
            (e - (3 * v - 18) if shape[0] == "complete" else e - (2 * v - 8), "sr2a-planar-bound-arithmetic"),
        ]
        kind_word = "roads" if shape[0] == "complete" else "cables"
        working = [
            f"There are {v} vertices, so a planar network can have at most {rule_bound} {kind_word}.",
            f"The network has {e}, so at least {e} - {rule_bound} = {excess} must be removed.",
        ]
        q = single(text, excess, working, wrong, lo=1)
        if q:
            return q
    raise AssertionError  # pragma: no cover


# -- hard -------------------------------------------------------------------------------------------------------

_ADD_ASK = (
    "Using the rule, what is the greatest number of further {es} that could be built without any two crossing?",
    "No two {es} may cross. At most how many more {es} can be added to the network?",
)
_FINAL_ASK = (
    "Further {es} are added, none crossing, until no more can be added. Into how many regions, counting the one "
    "outside, does the finished network divide the map?",
    "The council keeps adding {es} that cross no other, until it is impossible to add another. How many regions "
    "(the outside one included) does the finished network have?",
)
_UNIFORM_ASK = (
    "A connected network is drawn on a map with no crossing {es}. Every region, including the outside one, is "
    "bounded by exactly {k} {es}, and every {e} borders two regions. The network has {m} {es}. How many "
    "{vs} does it have?",
    "Every region of a connected plane network, the outside region included, is enclosed by exactly {k} {es}, and "
    "each {e} lies between two regions. There are {m} {es} altogether. How many {vs} are there?",
)

_UNIFORM_CHOICES = ((3, (9, 12, 15, 18, 24)), (4, (8, 10, 12, 14, 16, 20)), (5, (30,)))


def _hard(rng):
    ctx = rng.choice(_PLACES)
    shape = rng.choice(["add", "final", "uniform", "add", "final"])
    for _ in attempts(300, "a hard planar question"):
        if shape in ("add", "final"):
            v = rng.randint(7, 12)
            e = rng.randint(v + 2, 2 * v - 3)
            full = 3 * v - 6
            intro = (
                f"A connected network of {v} {ctx['vs']} has {e} {ctx['es']}, drawn with none crossing. "
                f"{_RULE_ANY} The bound is reached exactly when every region, the outside one included, is a "
                "triangle."
            )
            if shape == "add":
                correct = full - e
                text = intro + " " + phrase(rng, _ADD_ASK, es=ctx["es"])
                wrong = [
                    (full, "sr2a-planar-bound-as-removals"),
                    (e - full if e > full else None, "sr2a-planar-excess-direction"),
                    (correct + 1, "sr2a-planar-bound-arithmetic"),
                    (correct - 1, "sr2a-planar-bound-arithmetic"),
                    (3 * v - e, "sr2a-planar-bound-missing-constant"),
                ]
                working = [
                    f"The most {ctx['es']} a planar network of {v} {ctx['vs']} can have is 3 x {v} - 6 = {full}.",
                    f"It has {e} already, so at most {full} - {e} = {correct} more can be added.",
                ]
            else:
                correct = _euler_faces(v, full)
                check(correct == 2 * v - 4, "faces of the finished network differ from 2v - 4")
                text = intro + " " + phrase(rng, _FINAL_ASK, es=ctx["es"])
                wrong = [
                    (_euler_faces(v, e), "sr2a-planar-uses-original-edges"),
                    (full, "sr2a-planar-faces-as-edges"),
                    (correct - 1, "sr2a-planar-forgot-outer"),
                    (full - v, "sr2a-planar-faces-edges-minus-vertices"),
                    (v, "sr2a-planar-faces-as-vertices"),
                ]
                working = [
                    f"No more can be added when there are 3 x {v} - 6 = {full} {ctx['es']}.",
                    f"Euler's formula gives f = 2 - {v} + {full} = {correct} regions, outside region included.",
                ]
        else:
            k, edge_choices = rng.choice(_UNIFORM_CHOICES)
            m = rng.choice(edge_choices)
            faces = 2 * m // k
            check(faces * k == 2 * m, "the boundary count does not split into whole regions")
            correct = m - faces + 2
            text = phrase(rng, _UNIFORM_ASK, es=ctx["es"], e=ctx["e"], vs=ctx["vs"], k=k, m=m)
            wrong = [
                (m - faces, "sr2a-planar-faces-edges-minus-vertices"),
                (m // k + 2, "sr2a-planar-boundary-count-once"),
                (m - m // k + 2, "sr2a-planar-boundary-count-once"),
                (faces, "sr2a-planar-faces-as-vertices"),
                (correct - 1, "sr2a-planar-forgot-outer"),
                (correct + 1, "sr2a-planar-euler-signs"),
            ]
            working = [
                f"Each region has {k} {ctx['es']} on its boundary and each {ctx['e']} borders two regions, so {k}f = 2 x {m}.",
                f"That gives f = {faces}. Euler's formula v - e + f = 2 then gives v = 2 + {m} - {faces} = {correct}.",
            ]
        q = single(text, correct, working, wrong, lo=1)
        if q:
            return q
    raise AssertionError  # pragma: no cover


@generator(
    "sr2a_planar_faces",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def sr2a_planar_faces(rng, tier="easy"):
    """Faces of a planar network, and the edge-count test for planarity; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------

_CHAIN_STEM = (
    "A council wants to join {n} {vs} so that every {v} has its own direct {e} to every other {v}. "
    "A connected simple planar graph with v vertices (v at least 3) has at most 3v - 6 edges.",
    "Plans show {n} {vs} that are each to be joined directly to every other by a separate {e}. "
    "A connected simple planar graph with v vertices (v at least 3) has at most 3v - 6 edges.",
)
_YES_ALWAYS = "Yes: every complete network can be drawn with no crossings."
_NO_BOUND = "No: it has more than 3v - 6 edges, so it cannot be planar."
_YES_FEWER = "Yes: it has fewer than 3v edges, so it satisfies the rule."
_NOT_SURE = "Cannot tell: the rule only shows a network is planar when it is satisfied."


@generator("sr2a_planar_faces_chain", level=12, presets={"exam": dict(tier="exam")})
def sr2a_planar_faces_chain(rng, tier="exam"):
    """Exam chain on a complete network: its roads, the fewest to remove, the regions left, and a planarity verdict."""
    check(tier == "exam", f"unknown tier {tier!r}")
    ctx = rng.choice(_PLACES)
    singular = {"towns": "town", "islands": "island", "stations": "station", "suburbs": "suburb"}[ctx["vs"]]
    for _ in attempts(100, "a complete network chain with enough distractors"):
        n = rng.randint(6, 9)
        total = n * (n - 1) // 2
        check(total == len(list(combinations(range(n), 2))), "pairs counted two ways differ")
        bound = 3 * n - 6
        removed = total - bound
        left = _euler_faces(n, bound)
        check(left == 2 * n - 4, "regions of the left-over network differ from 2n - 4")
        stem = phrase(rng, _CHAIN_STEM, n=n, vs=ctx["vs"], v=singular, e=ctx["e"])
        part_a = number_part(
            f"How many {ctx['es']} are needed altogether?",
            total,
            1,
            [f"Each pair of {ctx['vs']} has one {ctx['e']}: {n} x {n - 1} / 2 = {total}."],
            [
                (n * (n - 1), "sr2a-net-ordered-pairs"),
                (n * (n - 1) // 2 - n, "sr2a-net-ring-only"),
                (n, "sr2a-net-places-as-links"),
                (total + n, "sr2a-net-pairs-miscount"),
            ],
            lo=1,
        )
        part_b = number_part(
            f"At least how many of these {ctx['es']} must be left out for the network to be drawn with no crossings?",
            removed,
            2,
            [
                f"A planar network of {n} {ctx['vs']} has at most 3 x {n} - 6 = {bound} {ctx['es']}.",
                f"{total} - {bound} = {removed}.",
            ],
            [
                (bound, "sr2a-planar-bound-as-removals"),
                (total - (3 * n - 2), "sr2a-planar-bound-arithmetic"),
                (total - 3 * n, "sr2a-planar-bound-missing-constant"),
                (removed + 1, "sr2a-planar-bound-arithmetic"),
                (total, "sr2a-planar-edges-not-excess"),
            ],
            follow=f"part_a - {bound}",
            lo=1,
        )
        part_c = number_part(
            f"Exactly that many {ctx['es']} are left out and the {ctx['es']} that remain are drawn with none "
            "crossing and every region, the outside one included, a triangle. How many regions does the map have?",
            left,
            2,
            [
                f"{total} - {removed} = {bound} {ctx['es']} remain.",
                f"Euler's formula v - e + f = 2 gives f = 2 - {n} + {bound} = {left}.",
            ],
            [
                (left - 1, "sr2a-planar-forgot-outer"),
                (bound, "sr2a-planar-faces-as-edges"),
                (total - n + 2, "sr2a-planar-uses-original-edges"),
                (n, "sr2a-planar-faces-as-vertices"),
                (bound - n, "sr2a-planar-faces-edges-minus-vertices"),
            ],
            follow=f"(part_a - part_b) - {n} + 2",
            lo=1,
        )
        part_d = mcq_part(
            f"Can all {total} {ctx['es']} be drawn with no two crossing?",
            _NO_BOUND,
            1,
            [f"{total} is more than {bound}, so the edge-count test rules planarity out."],
            [
                (_YES_ALWAYS, "sr2a-planar-complete-always-planar"),
                (_YES_FEWER, "sr2a-planar-wrong-bound"),
                (_NOT_SURE, "sr2a-planar-bound-sufficient"),
            ],
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
    raise AssertionError  # pragma: no cover
