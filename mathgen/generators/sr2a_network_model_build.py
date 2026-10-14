"""Generates `network-model-from-situation` questions (easy, medium, hard) and the exam chain `sr2a_network_model_chain`.

Modelling a practical situation as a network (ACMGM079, `financial-discrete`, answer ``number``). The situation is
described in words, never as a drawn graph or a list of letter pairs: the learner decides what the vertices are (the
people, the cities, the islands) and what the edges are (a friendship, a direct flight, a ferry run), and then reads
a number off the model.

* easy: friendships written as "each person's friends" lists, so every friendship appears twice. Either how many
  friendships there are, or how many friends the person with the most friends has;
* medium: airline or ferry routes written as a chain of stops, planes running both ways between neighbouring stops,
  two routes sharing some of their links. Either the number of different direct links (a link used by two routes
  is one link) or the fewest flights between two places across the whole network;
* hard: either how many new direct links are needed so that every place can be reached from every other (the
  routes leave separate groups of places, some places on no route at all), or the fewest flights between two places
  once one named route is cancelled;
* exam (``sr2a_network_model_chain``): from a list of routes, (a) the places served, (b) the different direct links,
  (c) how many direct links there would be if every pair of places were joined, (d) how many more links that needs.

Each network is built from its routes or lists; its numbers are re-derived from the pair list the routes spell
(distinct pairs by set union, distances by breadth-first search, groups by component search).
"""

from __future__ import annotations

from itertools import combinations

from ..core import attempts, check, generator, phrase
from ._sr2a_kit import components, distance, norm, number_part, random_connected, single

_PEOPLE = ("Ana", "Ben", "Cara", "Dev", "Ella", "Finn", "Grace", "Hugo", "Isla", "Jack")
_CITIES = (
    "Perth",
    "Adelaide",
    "Melbourne",
    "Sydney",
    "Brisbane",
    "Darwin",
    "Hobart",
    "Cairns",
    "Canberra",
    "Alice Springs",
)
_ISLANDS = ("Hamilton", "Daydream", "Lindeman", "Hayman", "Dunk", "Magnetic", "Fitzroy", "Heron", "Orpheus")


def _pick_names(rng, pool, n):
    names = list(pool)
    rng.shuffle(names)
    return names[:n]


# -- easy -------------------------------------------------------------------------------------------------------

_EASY_SETS = (
    dict(
        lead="{n} friends keep a list of the people they message regularly. Messaging goes both ways, so anyone on "
        "a list also has that person on their own list. {lists}",
        edges="How many different pairs of friends message each other?",
        most="What is the greatest number of people that any one friend messages regularly?",
    ),
    dict(
        lead="Each of {n} classmates wrote down the classmates they study with. Studying together goes both ways, so "
        "anyone on a list also has that classmate on their own list. {lists}",
        edges="How many different pairs of classmates study together?",
        most="What is the greatest number of classmates that any one person studies with?",
    ),
)


def _easy(rng):
    ctx = rng.choice(_EASY_SETS)
    for _ in attempts(300, "an easy people network"):
        n = rng.randint(6, 7)
        m = rng.randint(n, n + 3)
        edges = random_connected(rng, n, m)
        names = _pick_names(rng, _PEOPLE, n)
        nbrs = [
            sorted(names[w] for u, w in edges if u == v) + sorted(names[u] for u, w in edges if w == v)
            for v in range(n)
        ]
        degs = [len(x) for x in nbrs]
        check(sum(degs) == 2 * len(set(edges)), "the lists do not add to twice the links")
        top = max(degs)
        if degs.count(top) != 1 and rng.random() < 0.5:
            continue
        order = sorted(range(n), key=lambda v: names[v])
        lists = " ".join(f"{names[v]}: {', '.join(sorted(nbrs[v]))}." for v in order)
        if rng.random() < 0.55:
            key = m
            text = ctx["lead"].format(n=n, lists=lists) + " " + ctx["edges"]
            wrong = [
                (2 * m, "sr2a-net-lists-not-halved"),
                (n, "sr2a-net-places-as-links"),
                (m - 1, "sr2a-net-count-slip"),
                (m + 1, "sr2a-net-count-slip"),
                (top, "sr2a-net-max-degree-as-links"),
            ]
            working = [
                "Treat each person as a vertex and each pair of friends as an edge.",
                f"The lists have {' + '.join(str(d) for d in degs)} = {sum(degs)} names, and every pair appears on two lists, "
                f"so there are {sum(degs)} / 2 = {m} pairs.",
            ]
        else:
            key = top
            text = ctx["lead"].format(n=n, lists=lists) + " " + ctx["most"]
            wrong = [
                (m, "sr2a-net-links-as-max-degree"),
                (top - 1, "sr2a-net-count-slip"),
                (top + 1, "sr2a-net-count-slip"),
                (sorted(degs)[-2] if sorted(degs)[-2] != top else None, "sr2a-net-wrong-person"),
                (n, "sr2a-net-places-as-links"),
            ]
            working = [
                "The number of friends a person has is the number of names on their list.",
                f"The list lengths are {', '.join(str(d) for d in degs)}, and the greatest is {top}.",
            ]
        q = single(text, key, working, wrong, lo=1)
        if q:
            return q
    raise AssertionError  # pragma: no cover


# -- routes -----------------------------------------------------------------------------------------------------


def _route_pairs(route):
    return [norm(route[i], route[i + 1]) for i in range(len(route) - 1)]


def _routes_text(routes, names, label):
    return " ".join(f"{label} {i + 1}: {' - '.join(names[v] for v in r)}." for i, r in enumerate(routes))


_ROUTE_CONTEXTS = (
    dict(
        pool=_CITIES,
        intro="An airline flies the routes below. On each route a plane flies directly between neighbouring stops, in both directions.",
        label="Route",
        flight="flight",
        flights="flights",
        place="city",
        places="cities",
        link="direct flight",
        links="direct flights",
        move="fly",
    ),
    dict(
        pool=_ISLANDS,
        intro="A ferry company runs the routes below. On each route a ferry runs directly between neighbouring stops, in both directions.",
        label="Run",
        flight="ride",
        flights="rides",
        place="island",
        places="islands",
        link="direct ferry link",
        links="direct ferry links",
        move="travel",
    ),
)


def _random_routes(rng, n: int, count: int, *, lengths=(3, 4)):
    """``count`` routes (paths of distinct places) over places 0..n-1, as lists of place numbers."""
    routes = []
    for _ in range(count):
        size = rng.randint(*lengths)
        routes.append(rng.sample(range(n), size))
    return routes


def _union(routes):
    return sorted({p for r in routes for p in _route_pairs(r)})


def _shared_links(routes) -> int:
    pairs = [p for r in routes for p in _route_pairs(r)]
    return len(pairs) - len(set(pairs))


def _medium_network(rng, ctx):
    for _ in attempts(2000, "a connected route network that shares links"):
        n = rng.randint(6, 8)
        routes = _random_routes(rng, n, rng.randint(3, 4))
        edges = _union(routes)
        used = {v for r in routes for v in r}
        if len(used) < 6 or _shared_links(routes) < 1:
            continue
        comps = [c for c in components(n, edges) if c & used]
        if len(comps) != 1:
            continue
        return n, routes, edges, sorted(used)
    raise AssertionError  # pragma: no cover


_MEDIUM_LINKS = (
    "How many different {links} are there? A {link} used by two routes counts once.",
    "Counting a {link} that appears on two routes only once, how many different {links} are there altogether?",
)
_MEDIUM_FEWEST = (
    "What is the fewest number of {flights} needed to {move} from {a} to {b}? Changing routes is allowed.",
    "A passenger wants to {move} from {a} to {b}, changing between routes where necessary. What is the fewest number "
    "of {flights} that can be used?",
)


def _medium(rng):
    ctx = rng.choice(_ROUTE_CONTEXTS)
    for _ in attempts(300, "a medium route question"):
        n, routes, edges, used = _medium_network(rng, ctx)
        names = _pick_names(rng, ctx["pool"], n)
        text = ctx["intro"] + " " + _routes_text(routes, names, ctx["label"])
        total_listed = sum(len(r) - 1 for r in routes)
        if rng.random() < 0.5:
            key = len(edges)
            check(key == len(set(edges)), "links counted two ways differ")
            text += " " + phrase(rng, _MEDIUM_LINKS, link=ctx["link"], links=ctx["links"])
            wrong = [
                (total_listed, "sr2a-net-routes-not-merged"),
                (len(used), "sr2a-net-places-as-links"),
                (key - 1, "sr2a-net-count-slip"),
                (key + 1, "sr2a-net-count-slip"),
                (len(routes), "sr2a-net-routes-as-links"),
            ]
            working = [
                "Each stop is a vertex and each direct flight between neighbouring stops is an edge.",
                f"The routes have {' + '.join(str(len(r) - 1) for r in routes)} = {total_listed} links, but "
                f"{total_listed - key} of them appear on two routes, so there are {key} different links.",
            ]
        else:
            pairs = [(a, b) for a, b in combinations(used, 2)]
            rng.shuffle(pairs)
            found = None
            for a, b in pairs:
                d = distance(n, edges, a, b)
                single_route = [r.index(b) - r.index(a) if a in r and b in r else None for r in routes]
                single_route = [abs(x) for x in single_route if x]
                if d is not None and d >= 3 and (not single_route or min(single_route) != d):
                    found = (a, b, d)
                    break
            if found is None:
                continue
            a, b, d = found
            key = d
            text += " " + phrase(rng, _MEDIUM_FEWEST, flights=ctx["flights"], move=ctx["move"], a=names[a], b=names[b])
            wrong = [
                (d + 1, "sr2a-net-stops-for-flights"),
                (d - 1, "sr2a-net-count-slip"),
                (d + 2, "sr2a-net-single-route-distance"),
                (len(used) - 1, "sr2a-net-single-route-distance"),
                (len(routes), "sr2a-net-routes-as-links"),
            ]
            working = [
                "Model the places as vertices and the direct flights as edges; the fewest flights is the shortest walk from "
                f"{names[a]} to {names[b]}.",
                f"Searching outwards from {names[a]} first reaches {names[b]} after {d} flights.",
            ]
        q = single(text, key, working, wrong, lo=1)
        if q:
            return q
    raise AssertionError  # pragma: no cover


# -- hard -------------------------------------------------------------------------------------------------------

_HARD_JOIN = (
    "The company wants every {place} to be reachable from every other, changing between routes if needed. What is the "
    "fewest number of new {links} that must be added?",
    "A manager wants a passenger to be able to {move} between any two of the {places}, changing routes where "
    "necessary. How few new {links} would make that possible?",
)
_HARD_CANCEL = (
    "{cancelled} is cancelled. What is the fewest number of {flights} needed to {move} from {a} to {b} now?",
    "After {cancelled} is cancelled, what is the fewest number of {flights} that a passenger needs to {move} from "
    "{a} to {b}?",
)


_GROUP_SIZES = {
    3: [(3, 3, 2), (4, 3, 1), (4, 2, 2), (4, 4, 1), (3, 3, 3)],
    4: [(3, 3, 1, 1), (3, 2, 2, 1), (3, 2, 2, 2), (4, 2, 1, 1)],
    5: [(2, 2, 2, 2, 1), (3, 2, 2, 1, 1), (2, 2, 2, 1, 1)],
}


def _separate_groups(rng, ctx):
    """Routes inside 3-5 separate groups of places (some places on no route): returns n, routes, group count."""
    for _ in attempts(2000, "separate groups of places"):
        groups = rng.choice([3, 4, 5])
        sizes = rng.choice(_GROUP_SIZES[groups])
        n = sum(sizes)
        places = list(range(n))
        rng.shuffle(places)
        routes, at = [], 0
        for s in sizes:
            group = places[at : at + s]
            at += s
            if s >= 2:
                routes.append(group[:])
        extra = []
        for r in routes:
            if len(r) >= 4 and rng.random() < 0.6:
                extra.append(r[:3])  # a second, shorter run that shares links with the first
        routes += extra
        rng.shuffle(routes)
        edges = _union(routes)
        comps = components(n, edges)
        if len(comps) == len(sizes):
            return n, routes, len(comps)
    raise AssertionError  # pragma: no cover


def _hard_join(rng, ctx):
    n, routes, groups = _separate_groups(rng, ctx)
    names = _pick_names(rng, ctx["pool"], n)
    on_a_route = {v for r in routes for v in r}
    off = [names[v] for v in range(n) if v not in on_a_route]
    text = ctx["intro"] + " " + _routes_text(routes, names, ctx["label"])
    if off:
        text += f" {', '.join(off)} {'is' if len(off) == 1 else 'are'} not on any route."
    text += " " + phrase(
        rng, _HARD_JOIN, place=ctx["place"], places=ctx["places"], links=ctx["links"], move=ctx["move"]
    )
    comps = components(n, _union(routes))
    key = len(comps) - 1
    isolated = len(off)
    wrong = [
        (len(comps), "sr2a-net-components-not-minus-one"),
        (isolated, "sr2a-net-isolated-only"),
        (len(routes) - 1, "sr2a-net-routes-as-links"),
        (key + 1, "sr2a-net-components-not-minus-one"),
        (key - 1, "sr2a-net-count-slip"),
        (n - len(on_a_route) + 1, "sr2a-net-isolated-only"),
        (key + 2, "sr2a-net-count-slip"),
        (len(comps) + 1, "sr2a-net-components-not-minus-one"),
    ]
    working = [
        "Places that can be reached from one another form a group; the places on no route are groups of one.",
        f"There are {len(comps)} separate groups. Joining {len(comps)} groups into one needs {len(comps)} - 1 = {key} new links.",
    ]
    return single(text, key, working, wrong, lo=1)


def _hard_cancel(rng, ctx):
    for _ in attempts(3000, "a network where a cancellation lengthens a trip"):
        n = rng.randint(7, 9)
        routes = _random_routes(rng, n, rng.randint(4, 5), lengths=(3, 4))
        edges = _union(routes)
        used = sorted({v for r in routes for v in r})
        if len(used) < 7 or len(components(n, edges)) - (n - len(used)) != 1:
            continue
        victim = rng.randrange(len(routes))
        left = [r for i, r in enumerate(routes) if i != victim]
        left_edges = _union(left)
        if len(left_edges) == len(edges):
            continue
        pairs = list(combinations(used, 2))
        rng.shuffle(pairs)
        want = rng.choice([3, 4, 4, 5])
        found = []
        for a, b in pairs:
            before = distance(n, edges, a, b)
            after = distance(n, left_edges, a, b)
            if before is not None and after is not None and after >= before + 1 and after >= 3:
                found.append((a, b, before, after))
        found.sort(key=lambda t: abs(t[3] - want))
        if not found:
            continue
        a, b, before, after = found[0]
        names = _pick_names(rng, ctx["pool"], n)
        cancelled = f"{ctx['label']} {victim + 1}"
        text = ctx["intro"] + " " + _routes_text(routes, names, ctx["label"])
        text += " " + phrase(
            rng,
            _HARD_CANCEL,
            cancelled=cancelled,
            flights=ctx["flights"],
            move=ctx["move"],
            a=names[a],
            b=names[b],
        )
        wrong = [
            (before, "sr2a-net-cancelled-ignored"),
            (after + 1, "sr2a-net-stops-for-flights"),
            (after - 1, "sr2a-net-count-slip"),
            (after + 2, "sr2a-net-single-route-distance"),
            (len(used) - 1, "sr2a-net-single-route-distance"),
        ]
        working = [
            f"Remove {cancelled} and keep only the links on the other routes.",
            f"With {cancelled} gone the shortest trip from {names[a]} to {names[b]} takes {after} {ctx['flights']} "
            f"(it was {before} before).",
        ]
        q = single(text, after, working, wrong, lo=1)
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _hard(rng):
    ctx = rng.choice(_ROUTE_CONTEXTS)
    for _ in attempts(100, "a hard route question"):
        q = _hard_join(rng, ctx) if rng.random() < 0.5 else _hard_cancel(rng, ctx)
        if q:
            return q
    raise AssertionError  # pragma: no cover


@generator(
    "sr2a_network_model",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def sr2a_network_model(rng, tier="easy"):
    """Read a number off the network a described situation models; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("sr2a_network_model_chain", level=12, presets={"exam": dict(tier="exam")})
def sr2a_network_model_chain(rng, tier="exam"):
    """Exam chain from a list of routes: places served, different links, links if all pairs joined, links still needed."""
    check(tier == "exam", f"unknown tier {tier!r}")
    ctx = rng.choice(_ROUTE_CONTEXTS)
    for _ in attempts(300, "a route chain with enough distractors"):
        n, routes, edges, used = _medium_network(rng, ctx)
        names = _pick_names(rng, ctx["pool"], n)
        v = len(used)
        m = len(edges)
        all_pairs = v * (v - 1) // 2
        check(all_pairs == len(list(combinations(used, 2))), "pairs counted two ways differ")
        extra = all_pairs - m
        listed = sum(len(r) - 1 for r in routes)
        stem = ctx["intro"] + " " + _routes_text(routes, names, ctx["label"])
        stops = sum(len(r) for r in routes)
        part_a = number_part(
            f"How many different {ctx['places']} are served?",
            v,
            1,
            [
                f"The routes list {stops} stops, but some {ctx['places']} are on more than one route. Counting each name once gives {v}."
            ],
            [
                (stops, "sr2a-net-routes-not-merged"),
                (len(routes), "sr2a-net-routes-as-links"),
                (v - 1, "sr2a-net-count-slip"),
                (v + 1, "sr2a-net-count-slip"),
            ],
            lo=1,
        )
        part_b = number_part(
            f"How many different {ctx['links']} are there? A {ctx['link']} used by two routes counts once.",
            m,
            2,
            [f"The routes have {listed} links in all, but {listed - m} are shared, so there are {m} different links."],
            [
                (listed, "sr2a-net-routes-not-merged"),
                (v, "sr2a-net-places-as-links"),
                (m - 1, "sr2a-net-count-slip"),
                (m + 1, "sr2a-net-count-slip"),
            ],
            lo=1,
        )
        part_c = number_part(
            f"How many {ctx['links']} would there be if every pair of these {ctx['places']} were joined directly?",
            all_pairs,
            2,
            [
                f"Each of the {v} {ctx['places']} joins to the other {v - 1}, and each pair is counted twice: {v} x {v - 1} / 2 = {all_pairs}."
            ],
            [
                (v * (v - 1), "sr2a-net-ordered-pairs"),
                (v * v, "sr2a-net-ordered-pairs"),
                (v * (v - 1) // 2 - 1, "sr2a-net-pairs-miscount"),
                (v * (v - 1) // 2 + v, "sr2a-net-pairs-miscount"),
            ],
            follow="part_a*(part_a - 1)/2",
            lo=1,
        )
        part_d = number_part(
            f"How many new {ctx['links']} would have to be added to the existing ones for every pair to be joined directly?",
            extra,
            2,
            [f"{all_pairs} - {m} = {extra}."],
            [
                (all_pairs, "sr2a-net-pairs-for-extra"),
                (m, "sr2a-net-existing-for-extra"),
                (extra + 1, "sr2a-net-count-slip"),
                (extra - 1, "sr2a-net-count-slip"),
            ],
            follow="part_c - part_b",
            lo=0,
            sensible=(0, None),
        )
        if None in (part_a, part_b, part_c, part_d) or extra < 1:
            continue
        return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
    raise AssertionError  # pragma: no cover
