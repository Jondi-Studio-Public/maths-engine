"""Generates `snet_eulerian` questions (easy, medium, hard) and the exam chain `snet_eulerian_chain`.

``graph-eulerian-trails-circuits`` (``financial-discrete``, answer ``mcq``): decide from the degrees whether a network
has an Eulerian trail (a trail using every edge exactly once) or an Eulerian circuit (one that starts and ends at the
same vertex), and where a trail must start and end.

A connected network has an Eulerian circuit exactly when every vertex has even degree, and an Eulerian trail that is
not a circuit exactly when it has exactly two odd vertices, which are the trail's two ends.

* easy: a connected network is given by its edges; the statement that is true says whether it has an Eulerian circuit,
  an Eulerian trail only, or neither;
* medium: a connected network with exactly two odd vertices; the pair of vertices an Eulerian trail must start and
  end at is chosen from four pairs;
* hard: four networks are listed by their edges and the one that has an Eulerian circuit is chosen. The wrong ones
  have two odd vertices, four odd vertices, or every degree even but two separate pieces;
* exam (``snet_eulerian_chain``): a connected network with 0, 2, 4 or 6 odd vertices. (a) how many odd vertices; (b) the fewest
  new edges (each between two vertices not already joined) that give an Eulerian circuit, following through from (a);
  (c) the number of edges in that circuit, following through from (b); (d) the fewest new edges that give an Eulerian
  trail, following through from (a).

The kind of each network is read from its degrees and connectedness, and again by a backtracking search that really
tries to use every edge once. The edge counts in the exam chain are found by trying every set of new edges.
"""

from __future__ import annotations

from itertools import combinations

from ..answers import Distractor
from ..core import check, generator, phrase
from ._snet_kit import (
    LETTERS,
    degrees,
    describe_network,
    edges_text,
    euler_trail_exists,
    euler_trail_starts,
    graph_with_odd_vertices,
    int_distractors,
    is_connected,
    min_edges_to_add,
    norm,
    odd_vertices,
    pick_context,
    cycle_edges,
)
from ._network_fig import with_network_figure

CIRCUIT = "It has an Eulerian circuit."
TRAIL = "It has an Eulerian trail but no Eulerian circuit."
NEITHER = "It has neither an Eulerian trail nor an Eulerian circuit."

_TAGS = {
    (CIRCUIT, TRAIL): "net-euler-circuit-trail-confused",
    (CIRCUIT, NEITHER): "net-euler-parity-reversed",
    (TRAIL, CIRCUIT): "net-euler-circuit-trail-confused",
    (TRAIL, NEITHER): "net-euler-two-odd-rejected",
    (NEITHER, CIRCUIT): "net-euler-odd-vertices-ignored",
    (NEITHER, TRAIL): "net-euler-many-odd-accepted",
}


def euler_kind(n, edges) -> str:
    """Which statement is true of a connected network, from the number of odd vertices."""
    odd = len(odd_vertices(n, edges))
    check(is_connected(n, edges), "the network is not connected")
    return CIRCUIT if odd == 0 else TRAIL if odd == 2 else NEITHER


_WHO = ("walker", "courier", "tourist", "cyclist")
_EASY_STEMS = (
    "{net} A {who} wants to use every {e} exactly once. Which statement about the network is true?",
    "{net} Which of these statements about the network is true? An Eulerian trail uses every {e} exactly once.",
)


def _easy(rng):
    ctx = pick_context(rng, travel=True)
    n = rng.randint(5, 6)
    odd = rng.choice([0, 2, 4])
    edges = graph_with_odd_vertices(rng, n, odd, 6, 9)
    key = euler_kind(n, edges)
    has_trail, has_circuit = euler_trail_exists(n, edges)
    check(
        (has_trail, has_circuit) == {CIRCUIT: (True, True), TRAIL: (True, False), NEITHER: (False, False)}[key],
        "search differs",
    )
    wrong = [Distractor(o, _TAGS[(key, o)]) for o in (CIRCUIT, TRAIL, NEITHER) if o != key]
    odds = [LETTERS[v] for v in odd_vertices(n, edges)]
    working = [
        "The degrees are " + ", ".join(f"{LETTERS[i]} {d}" for i, d in enumerate(degrees(n, edges))) + ".",
        "No vertex has an odd degree." if not odds else f"The odd vertices are {', '.join(odds)}.",
        {
            CIRCUIT: "Every degree is even and the network is connected, so it has an Eulerian circuit.",
            TRAIL: "Exactly two vertices are odd, so there is an Eulerian trail, but it starts and ends at different vertices.",
            NEITHER: "More than two vertices are odd, so no trail can use every edge exactly once.",
        }[key],
    ]
    text = phrase(rng, _EASY_STEMS, net=describe_network(ctx, n, edges), who=rng.choice(_WHO), e=ctx["e"])
    return _mcq(text, key, wrong, working)


def _mcq(text, key, wrong, working):
    return dict(text=text, answer_kind="mcq", answer=key, answer_text=key, working=working, distractors=wrong)


_MEDIUM_STEMS = (
    "{net} An Eulerian trail of this network uses every {e} exactly once and does not return to its start. "
    "Between which two {vs} must it start and end?",
    "{net} A {who} uses every {e} exactly once and finishes at a different {v} from the one they started at. "
    "Which two {vs} are the start and the finish?",
)


def _pair(vs):
    a, b = sorted(vs)
    return f"{LETTERS[a]} and {LETTERS[b]}"


def _medium(rng):
    ctx = pick_context(rng, travel=True)
    for _ in range(100):
        n = rng.randint(5, 6)
        edges = graph_with_odd_vertices(rng, n, 2, 6, 9)
        deg = degrees(n, edges)
        odd = odd_vertices(n, edges)
        even = [v for v in range(n) if v not in odd]
        top = sorted(range(n), key=lambda v: (-deg[v], v))[:2]
        if set(top) != set(odd):
            break
    else:
        raise AssertionError  # pragma: no cover
    check(euler_trail_starts(n, edges) == set(odd), "the search finds different start vertices")
    key = _pair(odd)
    mixed = [(o, e) for o in rng.sample(odd, 2) for e in rng.sample(even, len(even))]
    two_even = [tuple(p) for p in combinations(rng.sample(even, len(even)), 2)]
    # One of each kind of wrong pair first, then more of each kind in case two coincide.
    candidates = [
        (top, "net-euler-start-highest-degree"),
        (mixed[0], "net-euler-one-odd-one-even"),
        (two_even[0], "net-euler-start-even-vertices"),
    ]
    candidates += [(p, "net-euler-one-odd-one-even") for p in mixed[1:]] + [
        (p, "net-euler-start-even-vertices") for p in two_even[1:]
    ]
    wrong, seen = [], {key}
    for pair, tag in candidates:
        text = _pair(pair)
        if text not in seen:
            seen.add(text)
            wrong.append(Distractor(text, tag))
        if len(wrong) == 3:
            break
    check(len(wrong) == 3, "need three different pairs")
    working = [
        "The degrees are " + ", ".join(f"{LETTERS[i]} {d}" for i, d in enumerate(deg)) + ".",
        f"The only odd vertices are {LETTERS[odd[0]]} and {LETTERS[odd[1]]}, so the trail must start at one and end at the other.",
    ]
    text = phrase(
        rng,
        _MEDIUM_STEMS,
        net=describe_network(ctx, n, edges),
        e=ctx["e"],
        v=ctx["v"],
        vs=ctx["vs"],
        who=rng.choice(_WHO),
    )
    return _mcq(text, key, wrong, working)


_HARD_STEMS = (
    "A planner is comparing four networks of {vs}. Each pair of letters names the two {vs} one {e} joins. {listing} "
    "A {who} wants to use every {e} of one network exactly once and finish where they started. "
    "Which network has an Eulerian circuit?",
    "Four networks of {vs} are listed by the pairs of {vs} that each {e} joins. {listing} "
    "Which of the four networks has an Eulerian circuit?",
    "{listing} These are four networks of {vs}, each {e} written as the pair of {vs} it joins. "
    "Which network has an Eulerian circuit, that is, a route using every {e} exactly once and ending where it began?",
)
_LABELS = ("P", "Q", "R", "S")


def _relabel(rng, n, edges):
    perm = list(range(n))
    rng.shuffle(perm)
    return sorted(norm(perm[u], perm[v]) for u, v in edges)


def _hard(rng):
    ctx = pick_context(rng, travel=True)
    for _ in range(100):
        n = rng.randint(5, 6)
        circuit = graph_with_odd_vertices(rng, n, 0, 6, 9)
        trail = graph_with_odd_vertices(rng, rng.randint(5, 6), 2, 6, 9)
        many = graph_with_odd_vertices(rng, rng.randint(5, 6), 4, 6, 9)
        sizes = rng.choice([(3, 3), (3, 4)])
        split = cycle_edges([0, 1, 2]) + [norm(a + 3, b + 3) for a, b in cycle_edges(list(range(sizes[1])))]
        split_n = 3 + sizes[1]
        networks = [
            (circuit, n, None),
            (trail, max(max(e) for e in trail) + 1, "net-euler-circuit-trail-confused"),
            (many, max(max(e) for e in many) + 1, "net-euler-odd-vertices-ignored"),
            (split, split_n, "net-euler-ignores-connectedness"),
        ]
        shown = [(edges_text(_relabel(rng, m, e)), tag) for e, m, tag in networks]
        if len({s for s, _ in shown}) == 4:
            break
    else:
        raise AssertionError  # pragma: no cover
    # Independent route: each network really does (or does not) have a circuit when we try to use every edge once.
    for (edges, m, tag), (text, _t) in zip(networks, shown):
        has_circuit = euler_trail_exists(m, edges)[1]
        check(has_circuit == (tag is None), "the search differs from the degree rule")
    order = list(range(4))
    rng.shuffle(order)  # order[k] is the kind of network shown under label k
    label = {kind: f"Network {_LABELS[k]}" for k, kind in enumerate(order)}
    listing = " ".join(f"{label[kind]}: {shown[kind][0]}." for kind in sorted(order, key=lambda kind: label[kind]))
    key = label[0]
    wrong = [Distractor(label[kind], shown[kind][1]) for kind in (1, 2, 3)]
    working = [
        "An Eulerian circuit needs the network to be connected and every degree to be even.",
        f"{label[0]}: connected, every degree even.",
        f"{label[1]}: connected, but exactly two vertices are odd.",
        f"{label[2]}: connected, but four vertices are odd.",
        f"{label[3]}: every degree is even, but the network is in two separate pieces.",
    ]
    text = phrase(rng, _HARD_STEMS, vs=ctx["vs"], e=ctx["e"], listing=listing, who=rng.choice(_WHO))
    return _mcq(text, key, wrong, working)


@generator(
    "snet_eulerian",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@with_network_figure
def snet_eulerian(rng, tier="easy"):
    """Eulerian trails and circuits from the degrees; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("snet_eulerian_chain", level=12, presets={"exam": dict(tier="exam")})
@with_network_figure
def snet_eulerian_chain(rng, tier="exam"):
    """Exam chain on a network with 0, 2, 4 or 6 odd vertices: odd vertices, edges to add for a circuit, its length,
    and edges to add for a trail that ends somewhere else."""
    check(tier == "exam", f"unknown tier {tier!r}")
    ctx = pick_context(rng, travel=True)
    odd = rng.choice((0, 2, 4, 6))
    for _ in range(200):
        n = rng.randint(6, 7)
        edges = graph_with_odd_vertices(rng, n, odd, 7, 12)
        # Independent route: try every set of new edges for each goal.
        circuit_k = min_edges_to_add(n, edges, 0)
        trail_k = min_edges_to_add(n, edges, 2)
        # The pairing rule (each new edge fixes two odd vertices) must hold, so the follow-through formulas are exact.
        if circuit_k == odd // 2 and trail_k == abs(odd // 2 - 1):
            break
    else:
        raise AssertionError  # pragma: no cover
    m = len(edges)
    check(len(odd_vertices(n, edges)) == odd, "odd-vertex count differs from the construction")
    k, t = circuit_k, trail_k
    stem = (
        f"{describe_network(ctx, n, edges)} A {rng.choice(_WHO)} would like to use every {ctx['e']} "
        f"exactly once. New {ctx['es']} can be built, each joining two {ctx['vs']} that are not joined already."
    )
    a_dist = int_distractors(
        odd,
        [
            (odd // 2, "net-euler-circuit-trail-confused"),
            (n - odd, "net-odd-even-swapped"),
            (odd + 1, "net-degree-miscount"),
            (odd + 2, "net-degree-miscount"),
            (odd - 1, "net-degree-miscount"),
            (odd - 2, "net-degree-miscount"),
        ],
    )
    b_dist = int_distractors(
        k,
        [
            (odd, "net-euler-edge-per-odd-vertex"),
            (k + 1, "net-euler-circuit-trail-confused"),
            (k + 2, "net-degree-miscount"),
            (k - 1, "net-degree-miscount"),
            (0, "net-euler-parity-reversed"),
        ],
    )
    c_dist = int_distractors(
        m + k,
        [
            (m, "net-euler-added-edges-not-counted"),
            (m + odd, "net-euler-edge-per-odd-vertex"),
            (m + k + 1, "net-degree-miscount"),
            (m + k - 1, "net-degree-miscount"),
            (m + k + 2, "net-degree-miscount"),
        ],
    )
    d_dist = int_distractors(
        t,
        [
            (k, "net-euler-circuit-trail-confused"),
            (t + 1, "net-euler-circuit-trail-confused"),
            (odd, "net-euler-edge-per-odd-vertex"),
            (t + 2, "net-degree-miscount"),
            (0, "net-euler-parity-reversed"),
        ],
    )
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")
    odd_names = ", ".join(LETTERS[v] for v in odd_vertices(n, edges))

    def part(prompt, value, marks, working, dists, rng_=(0, None), ft=None):
        out = dict(
            prompt=prompt,
            answer_kind="number",
            answer=value,
            answer_text=str(value),
            answer_latex=str(value),
            marks=marks,
            sensible_range=rng_,
            working=working,
            distractors=dists,
        )
        if ft:
            out["follow_through"] = ft
        return out

    if odd:
        a_work = [f"The odd vertices are {odd_names}, which is {odd} {ctx['vs']}."]
        b_work = [
            "Every vertex must end up with even degree, and each new edge changes the degree of two vertices.",
            f"{odd} odd vertices can be paired up into {k} {'pair' if k == 1 else 'pairs'}, so {odd} / 2 = {k} new "
            f"{'edge is' if k == 1 else 'edges are'} needed "
            f"(and the pairs can be chosen so they are not joined already).",
        ]
    else:
        a_work = ["Every vertex has even degree, so no vertex has an odd degree."]
        b_work = ["Every vertex already has even degree and the network is connected, so no new edges are needed."]
    if odd == 2:
        d_work = [
            f"A trail that ends elsewhere needs exactly two odd vertices, and the network already has them ({odd_names}).",
            "No new edges are needed: 2 / 2 - 1 = 0.",
        ]
    elif odd == 0:
        d_work = [
            "With no odd vertices a trail would have to be a circuit, so two vertices must be made odd.",
            "One new edge between two vertices that are not joined already does it.",
        ]
    else:
        d_work = [
            f"A trail needs exactly two odd vertices, so only {odd} - 2 = {odd - 2} odd vertices need fixing.",
            f"Each new edge between two odd vertices that are not joined already fixes two: {odd} / 2 - 1 = {t}.",
        ]

    return dict(
        text=stem,
        parts=[
            part(f"How many {ctx['vs']} have an odd degree?", odd, 1, a_work, a_dist, (-1, n + 1)),
            part(
                f"Hence find the fewest new {ctx['es']} that must be built so that the network has an Eulerian circuit.",
                k,
                2,
                b_work,
                b_dist,
                (-1, None),
                "part_a/2",
            ),
            part(
                f"How many {ctx['es']} will the Eulerian circuit then use?",
                m + k,
                1,
                [
                    f"The circuit uses every edge once: the {m} existing {ctx['es']} plus the {k} new {'one' if k == 1 else 'ones'}, {m + k}."
                ],
                c_dist,
                (-1, None),
                f"{m} + part_b",
            ),
            part(
                f"Instead, the traveller would be happy to finish at a different {ctx['v']} from where they started. "
                f"Find the fewest new {ctx['es']} needed for an Eulerian trail.",
                t,
                2,
                d_work,
                d_dist,
                (-1, None),
                "Abs(part_a/2 - 1)",
            ),
        ],
    )
