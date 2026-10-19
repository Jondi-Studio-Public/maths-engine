"""Generates `w7a_network_cut` questions (easy, medium, hard) and the exam chain `exd_x_w7a_network_cut`.

``network-cut-capacity-max-flow`` (``financial-discrete``, answer ``number``): the capacity of a cut in a directed network
described in words, finding a minimum cut, and the maximum-flow minimum-cut theorem. A network is a source S, a sink T and
junctions A, B, C, ... joined by one-way links, each with a capacity.

* easy: the capacity of a stated cut of a four-node network (only links from the source side to the sink side count).
* medium: the capacity of a stated cut when a link also runs back from the sink side to the source side; or the maximum flow,
  when three cuts are stated and one of them is a minimum cut.
* hard: the maximum flow of a six-node network with no cut stated (the smallest cut has to be found); or the maximum flow
  after one link is widened, when a different cut then becomes the smallest.
* exam (``exd_x_w7a_network_cut``): (a) the capacity of one stated cut, (b) the capacity of a second cut that has a return link,
  (c) the maximum flow, (d) the capacity out of the source that cannot be used.

Every maximum flow is found twice: by augmenting paths (Edmonds-Karp), whose flow is checked to obey conservation at every
junction, and as the smallest capacity over every cut (every subset of the junctions on the source side).
"""

from __future__ import annotations

from collections import deque
from itertools import combinations

from ..core import attempts, check, draw_until, generator, phrase
from ._w6a_kit import enough, num_dist, num_fields, part, retry_draw
from ._network_fig import with_network_figure

TOPIC = "w7a_network_cut"
EXAM_TOPIC = "exd_x_w7a_network_cut"

AREA = "networks"
MISCONCEPTIONS = [
    (
        "flow-single-path-only",
        "Gives the capacity of the best single route from the source to the sink as the maximum flow, ignoring that flow can use several routes at once.",
        AREA,
    ),
    (
        "flow-increase-passes-fully",
        "Adds the whole widening of a link to the maximum flow, forgetting that another cut then limits the flow.",
        AREA,
    ),
    (
        "flow-widening-ignored",
        "Gives the old maximum flow, as if widening one link could not help.",
        AREA,
    ),
    (
        "flow-spare-reports-flow",
        "Gives the maximum flow where the capacity out of the source that cannot be used was asked for.",
        AREA,
    ),
    (
        "flow-spare-from-sink-side",
        "Subtracts the maximum flow from the capacity into the sink where the capacity out of the source was meant.",
        AREA,
    ),
    (
        "cut-smallest-of-listed-ignored",
        "Gives the middle or the largest of several stated cut capacities as the maximum flow, rather than the smallest.",
        AREA,
    ),
    (
        "cut-counts-both-ends",
        "Adds the capacity of every link that touches the source side of the cut, those inside it as well as those crossing it.",
        AREA,
    ),
]

_ORDINAL = ("first", "second", "third")

_CONTEXTS = (
    dict(
        noun="pipe",
        unit="litres per minute",
        intro="Water flows from a reservoir S to a town T through a network of one-way pipes.",
    ),
    dict(
        noun="cable",
        unit="gigabits per second",
        intro="Data flows from a server S to a client T through a network of one-way cables.",
    ),
    dict(
        noun="road",
        unit="hundreds of vehicles per hour",
        intro="Traffic flows from a depot S to a stadium T along one-way roads.",
    ),
)


# -- the network -------------------------------------------------------------------------------------------------------


def _max_flow(arcs: dict, source: str = "S", sink: str = "T"):
    """(value, flow dict) by Edmonds-Karp on a dict {(u, v): capacity}."""
    residual = dict(arcs)
    for u, v in arcs:
        residual.setdefault((v, u), 0)
    nodes = sorted({x for e in residual for x in e})
    flow = {e: 0 for e in arcs}
    total = 0
    for _ in range(sum(arcs.values()) + 1):
        parent = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            u = queue.popleft()
            for v in nodes:
                if v not in parent and residual.get((u, v), 0) > 0:
                    parent[v] = u
                    queue.append(v)
        if sink not in parent:
            break
        path, v = [], sink
        while parent[v] is not None:
            path.append((parent[v], v))
            v = parent[v]
        push = min(residual[e] for e in path)
        for u, v in path:
            residual[(u, v)] -= push
            residual[(v, u)] += push
            if (u, v) in flow:
                flow[(u, v)] += push
            else:
                flow[(v, u)] -= push
        total += push
    for node in nodes:
        if node not in (source, sink):
            inn = sum(f for (u, v), f in flow.items() if v == node)
            out = sum(f for (u, v), f in flow.items() if u == node)
            check(inn == out, "flow is not conserved at a junction")
    check(all(0 <= f <= arcs[e] for e, f in flow.items()), "a link carries more than its capacity or less than nothing")
    return total, flow


def _cut_capacity(arcs: dict, source_side) -> int:
    """The capacity of the cut: links from the source side to the sink side only."""
    side = set(source_side)
    return sum(c for (u, v), c in arcs.items() if u in side and v not in side)


def _return_capacity(arcs: dict, source_side) -> int:
    side = set(source_side)
    return sum(c for (u, v), c in arcs.items() if u not in side and v in side)


def _inside_capacity(arcs: dict, source_side) -> int:
    side = set(source_side)
    return sum(c for (u, v), c in arcs.items() if u in side and v in side)


def _junctions(arcs: dict) -> list[str]:
    return sorted({x for e in arcs for x in e} - {"S", "T"})


def _all_cuts(arcs: dict):
    """{frozenset(source side): capacity} over every subset of the junctions."""
    mids = _junctions(arcs)
    out = {}
    for k in range(len(mids) + 1):
        for chosen in combinations(mids, k):
            side = frozenset(("S", *chosen))
            out[side] = _cut_capacity(arcs, side)
    return out


def _min_cut_value(arcs: dict) -> int:
    value, _ = _max_flow(arcs)
    cuts = _all_cuts(arcs)
    check(value == min(cuts.values()), "augmenting paths and the smallest cut disagree")
    return value


def _layered(rng, layers, lo: int = 3, hi: int = 15, extra: float = 0.35, back: int = 0) -> dict:
    """A one-way network whose links run from each layer to the next, every junction with a link in and a link out, a few
    extras, and `back` links that run from a later junction back to an earlier one."""
    arcs: dict = {}
    for i in range(len(layers) - 1):
        for u in layers[i]:
            arcs[(u, rng.choice(layers[i + 1]))] = rng.randint(lo, hi)
        for v in layers[i + 1]:
            if not any(b == v for _, b in arcs):
                arcs[(rng.choice(layers[i]), v)] = rng.randint(lo, hi)
        for u in layers[i]:
            for v in layers[i + 1]:
                if (u, v) not in arcs and rng.random() < extra:
                    arcs[(u, v)] = rng.randint(lo, hi)
    mids = [x for layer in layers[1:-1] for x in layer]
    for _ in range(back):
        for _ in attempts(100, "a return link"):
            i, j = sorted(rng.sample(range(len(mids)), 2))
            u, v = mids[j], mids[i]
            if (u, v) not in arcs and (v, u) not in arcs:
                arcs[(u, v)] = rng.randint(lo, hi)
                break
    return arcs


def _links_text(arcs: dict) -> str:
    return ", ".join(f"{u}→{v} {c}" for (u, v), c in sorted(arcs.items()))


def _intro(ctx: dict, arcs: dict) -> str:
    n_junctions = len(_junctions(arcs))
    names = ", ".join(_junctions(arcs))
    return (
        f"{ctx['intro']} There are {n_junctions} junctions, {names}. Each {ctx['noun']} carries flow one way only. "
        f"The capacities, in {ctx['unit']}, are {_links_text(arcs)}, where X→Y means a {ctx['noun']} from X to Y."
    )


def _side_text(side, arcs: dict) -> str:
    inside = sorted(side, key=_order)
    outside = sorted((set(_junctions(arcs)) | {"S", "T"}) - set(side), key=_order)
    return f"puts {_join(inside)} on the source side and {_join(outside)} on the sink side"


def _order(name: str):
    return (name != "S", name == "T", name)


def _join(names) -> str:
    names = sorted(names, key=_order)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _cut_stem(rng, ctx: dict, arcs: dict, side) -> str:
    ask = phrase(
        rng,
        (
            "What is the capacity of this cut, in {u}?",
            "Find the capacity of the cut, in {u}.",
            "In {u}, what is the capacity of this cut?",
        ),
        u=ctx["unit"],
    )
    return f"{_intro(ctx, arcs)} A cut {_side_text(side, arcs)}. {ask}"


def _naive_bounds(arcs: dict) -> tuple[int, int]:
    return sum(c for (u, _), c in arcs.items() if u == "S"), sum(c for (_, v), c in arcs.items() if v == "T")


def _best_single_route(arcs: dict) -> int:
    """The largest bottleneck over all simple routes from S to T."""
    best = 0

    def walk(node, seen, bottleneck):
        nonlocal best
        if node == "T":
            best = max(best, bottleneck)
            return
        for (u, v), c in arcs.items():
            if u == node and v not in seen:
                walk(v, seen | {v}, min(bottleneck, c))

    walk("S", {"S"}, 10**9)
    return best


# -- easy -----------------------------------------------------------------------------------------------------------------


def _easy_cut(rng):
    ctx = rng.choice(_CONTEXTS)
    arcs = _layered(rng, [["S"], ["A", "B"], ["T"]], extra=0.0)
    side = rng.choice((("S", "A"), ("S", "B")))
    if side == ("S", "A") and rng.random() < 0.8:
        arcs[("A", "B")] = rng.randint(3, 15)  # crosses the cut from A to B, never back
    arcs.setdefault(("S", "A"), rng.randint(3, 15))
    arcs.setdefault(("S", "B"), rng.randint(3, 15))
    arcs.setdefault(("A", "T"), rng.randint(3, 15))
    arcs.setdefault(("B", "T"), rng.randint(3, 15))
    crossing = [(e, c) for e, c in arcs.items() if e[0] in side and e[1] not in side]
    check(len(crossing) >= 2, "a cut with a single link")
    answer = sum(c for _, c in crossing)
    check(answer == _cut_capacity(arcs, side), "cut capacity routes disagree")
    inside = _inside_capacity(arcs, side)
    drop_one = answer - min(c for _, c in crossing)
    cands = [
        (inside if inside else None, "cut-sums-pipes-on-one-side"),
        (drop_one, "cut-misses-a-crossing-pipe"),
        (answer + inside if inside else None, "cut-counts-both-ends"),
        (_naive_bounds(arcs)[0], "cut-sums-source-pipes"),
        (_naive_bounds(arcs)[1], "sum-all-sink-incoming"),
    ]
    dist = num_dist(answer, [(v, t) for v, t in cands if v and v > 1])
    enough(dist, 3)
    stem = _cut_stem(rng, ctx, arcs, side)
    working = [
        f"Only links that go from the source side {{{', '.join(sorted(side, key=_order))}}} to the sink side count: "
        + ", ".join(f"{u}→{v} ({c})" for (u, v), c in sorted(crossing))
        + ".",
        f"The capacity of the cut is {' + '.join(str(c) for _, c in sorted(crossing))} = {answer}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


def _easy_cut_three(rng):
    ctx = rng.choice(_CONTEXTS)
    for _ in attempts(500, "a cut with no return link"):
        arcs = _layered(rng, [["S"], ["A", "B"], ["C"], ["T"]], extra=0.3)
        side = frozenset(["S", *rng.sample(["A", "B", "C"], rng.randint(1, 2))])
        crossing = [c for (u, v), c in arcs.items() if u in side and v not in side]
        if _return_capacity(arcs, side) == 0 and len(crossing) >= 2:
            break
    answer = sum(crossing)
    check(answer == _cut_capacity(arcs, side), "cut capacity routes disagree")
    inside = _inside_capacity(arcs, side)
    cands = [
        (inside if inside else None, "cut-sums-pipes-on-one-side"),
        (answer - min(crossing), "cut-misses-a-crossing-pipe"),
        (answer + inside, "cut-counts-both-ends"),
        (_naive_bounds(arcs)[0], "cut-sums-source-pipes"),
        (_naive_bounds(arcs)[1], "sum-all-sink-incoming"),
    ]
    dist = num_dist(answer, [(v, t) for v, t in cands if v and v > 1])
    enough(dist, 3)
    working = [
        "Only links that go from the source side to the sink side count: "
        + ", ".join(f"{u}→{v} ({c})" for (u, v), c in sorted(arcs.items()) if u in side and v not in side)
        + ".",
        f"The capacity of the cut is {' + '.join(str(c) for c in crossing)} = {answer}.",
    ]
    return dict(
        text=_cut_stem(rng, ctx, arcs, side),
        working=working,
        distractors=dist,
        **num_fields(answer),
        answer_kind="number",
    )


# -- medium ---------------------------------------------------------------------------------------------------------------


def _network_with_return(rng, back: int = 1):
    """(arcs, source side) where a link runs from the sink side back to the source side."""
    for _ in attempts(500, "a network with a return link across a cut"):
        arcs = _layered(rng, [["S"], ["A", "B"], ["C"], ["T"]], extra=0.4, back=back)
        side = frozenset(["S", *rng.sample(["A", "B", "C"], rng.randint(1, 2))])
        if (
            _return_capacity(arcs, side) > 0
            and _cut_capacity(arcs, side) > 0
            and len([e for e in arcs if e[0] in side and e[1] not in side]) >= 2
        ):
            return arcs, side
    raise AssertionError  # pragma: no cover


def _medium_return(rng):
    ctx = rng.choice(_CONTEXTS)
    arcs, side = _network_with_return(rng)
    answer = _cut_capacity(arcs, side)
    back = _return_capacity(arcs, side)
    inside = _inside_capacity(arcs, side)
    crossing = [c for (u, v), c in arcs.items() if u in side and v not in side]
    cands = [
        (answer + back, "cut-counts-backward-pipes"),
        (answer - min(crossing), "cut-misses-a-crossing-pipe"),
        (inside if inside else None, "cut-sums-pipes-on-one-side"),
        (_naive_bounds(arcs)[0], "cut-sums-source-pipes"),
    ]
    dist = num_dist(answer, [(v, t) for v, t in cands if v and v > 1])
    enough(dist, 3)
    stem = _cut_stem(rng, ctx, arcs, side)
    cross_links = sorted(e for e in arcs if e[0] in side and e[1] not in side)
    return_links = sorted(e for e in arcs if e[0] not in side and e[1] in side)
    working = [
        "Count only links from the source side to the sink side: "
        + ", ".join(f"{u}→{v} ({arcs[(u, v)]})" for u, v in cross_links)
        + ".",
        "A link from the sink side back to the source side ("
        + ", ".join(f"{u}→{v}" for u, v in return_links)
        + f") is not part of the cut. The capacity is {answer}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(answer), answer_kind="number")


def _medium_three_cuts(rng):
    ctx = rng.choice(_CONTEXTS)
    for _ in attempts(500, "a network with three cuts, one of them minimum"):
        arcs = _layered(rng, [["S"], ["A", "B"], ["C"], ["T"]], extra=0.4, back=rng.randint(0, 1))
        flow, _ = _max_flow(arcs)
        cuts = _all_cuts(arcs)
        check(flow == min(cuts.values()), "augmenting paths and the smallest cut disagree")
        minimum = [s for s, c in cuts.items() if c == flow]
        others = {}
        for s, c in cuts.items():
            if c > flow:
                others.setdefault(c, []).append(s)
        if len(others) >= 2 and flow < min(_naive_bounds(arcs)):
            chosen = [rng.choice(minimum)]
            for c in rng.sample(sorted(others), 2):
                chosen.append(rng.choice(others[c]))
            break
    else:
        raise AssertionError  # pragma: no cover
    rng.shuffle(chosen)
    caps = sorted(cuts[s] for s in chosen)
    check(caps[0] == flow and caps[1] > flow, "the minimum of the three is not the maximum flow")
    cands = [
        (caps[1], "cut-smallest-of-listed-ignored"),
        (caps[2], "flow-equals-capacity-of-a-non-minimum-cut"),
        (_naive_bounds(arcs)[0], "sum-all-source-outgoing"),
        (_naive_bounds(arcs)[1], "sum-all-sink-incoming"),
        (min(arcs.values()), "minimum-pipe-capacity"),
    ]
    dist = num_dist(flow, [(v, t) for v, t in cands if v and v > 1])
    enough(dist, 3)
    cut_lines = " ".join(f"The {_ORDINAL[i]} cut {_side_text(s, arcs)}." for i, s in enumerate(chosen))
    stem = (
        f"{_intro(ctx, arcs)} Three cuts are considered. {cut_lines} One of the three is a minimum cut. "
        f"What is the maximum flow from S to T, in {ctx['unit']}?"
    )
    working = [
        "Find each cut's capacity, counting only links from its source side to its sink side: "
        + ", ".join(f"the {_ORDINAL[i]} cut is {cuts[s]}" for i, s in enumerate(chosen))
        + ".",
        f"By the maximum-flow minimum-cut theorem the maximum flow equals the capacity of a minimum cut. The smallest of these is {flow}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(flow), answer_kind="number")


# -- hard -----------------------------------------------------------------------------------------------------------------


def _hard_network(rng):
    for _ in attempts(500, "a six-node network whose smallest cut is hidden"):
        arcs = _layered(rng, [["S"], ["A", "B"], ["C", "D"], ["T"]], extra=0.45, back=rng.randint(0, 1))
        flow = _min_cut_value(arcs)
        out_s, in_t = _naive_bounds(arcs)
        cuts = _all_cuts(arcs)
        second = min((c for c in cuts.values() if c > flow), default=None)
        bounds = {out_s, in_t, min(arcs.values()), second, _best_single_route(arcs)}
        if (
            flow < min(out_s, in_t)
            and second is not None
            and flow not in bounds - {None}
            and cuts[frozenset(["S"])] > flow
        ):
            return arcs, flow, second
    raise AssertionError  # pragma: no cover


def _hard_maxflow(rng):
    ctx = rng.choice(_CONTEXTS)
    arcs, flow, second = _hard_network(rng)
    out_s, in_t = _naive_bounds(arcs)
    cands = [
        (out_s, "sum-all-source-outgoing"),
        (in_t, "sum-all-sink-incoming"),
        (second, "flow-equals-capacity-of-a-non-minimum-cut"),
        (_best_single_route(arcs), "flow-single-path-only"),
        (min(arcs.values()), "minimum-pipe-capacity"),
    ]
    dist = num_dist(flow, [(v, t) for v, t in cands if v and v > 1])
    enough(dist, 3)
    stem = f"{_intro(ctx, arcs)} What is the maximum flow from S to T, in {ctx['unit']}?"
    cuts = _all_cuts(arcs)
    best = min(cuts, key=lambda s: (cuts[s], len(s)))
    working = [
        "By the maximum-flow minimum-cut theorem, the maximum flow equals the capacity of the smallest cut, so look for the cut with the least capacity.",
        f"The smallest cut puts {_join(sorted(best))} on the source side; its capacity, from links crossing to the sink side only, is {flow}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(flow), answer_kind="number")


def _hard_widen(rng):
    ctx = rng.choice(_CONTEXTS)
    for _ in attempts(1500, "a widening that helps, but only partly"):
        arcs = _layered(rng, [["S"], ["A", "B"], ["C", "D"], ["T"]], extra=0.45, back=rng.randint(0, 1))
        old = _min_cut_value(arcs)
        cuts = _all_cuts(arcs)
        link = rng.choice(sorted(arcs))
        k = rng.randint(4, 9)
        wider = dict(arcs)
        wider[link] += k
        new = _min_cut_value(wider)
        if 0 < new - old < k and old >= 6:
            break
    out_s, _ = _naive_bounds(wider)
    cands = [
        (old + k, "flow-increase-passes-fully"),
        (old, "flow-widening-ignored"),
        (out_s, "sum-all-source-outgoing"),
        (_best_single_route(wider), "flow-single-path-only"),
        (old + k - 1, "flow-increase-passes-fully"),
    ]
    dist = num_dist(new, [(v, t) for v, t in cands if v and v > 1])
    enough(dist, 3)
    u, v = link
    stem = (
        f"{_intro(ctx, arcs)} The capacity of the {ctx['noun']} {u}→{v} is increased by {k}. "
        f"What is the maximum flow from S to T after the change, in {ctx['unit']}?"
    )
    working = [
        f"Before the change the smallest cut has capacity {old}, so the maximum flow is {old}.",
        f"After {u}→{v} is widened by {k} another cut limits the flow: the smallest cut now has capacity {new}, so the maximum flow rises by only {new - old}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(new), answer_kind="number")


_EASY = (_easy_cut, _easy_cut_three)
_MEDIUM = (_medium_return, _medium_three_cuts)
_HARD = (_hard_maxflow, _hard_widen)


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@with_network_figure
def w7a_network_cut(rng, tier="easy"):
    """Cut capacity, minimum cuts and maximum flow in a described network; the tiers are described in the module docstring."""
    pool = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}.get(tier)
    if not (pool is not None):
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# -- exam -----------------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
@with_network_figure
def exd_x_w7a_network_cut(rng, tier="exam"):
    """Exam chain: capacity of two stated cuts (one with a return link), the maximum flow, the source capacity left unused."""
    if not (tier == "exam"):
        raise ValueError(f"unknown tier {tier!r}")
    ctx = rng.choice(_CONTEXTS)

    def build(rng):
        for _ in attempts(1000, "a network and two cuts for the chain"):
            arcs = _layered(rng, [["S"], ["A", "B"], ["C"], ["T"]], extra=0.4, back=1)
            flow = _min_cut_value(arcs)
            out_s, in_t = _naive_bounds(arcs)
            cuts = _all_cuts(arcs)
            first = [s for s in cuts if len(s) == 2 and _return_capacity(arcs, s) == 0 and cuts[s] > flow]
            second = [s for s in cuts if 2 <= len(s) <= 3 and _return_capacity(arcs, s) > 0 and cuts[s] != flow]
            if first and second and flow < out_s and flow < in_t:
                s1, s2 = rng.choice(sorted(first, key=sorted)), rng.choice(sorted(second, key=sorted))
                if len({cuts[s1], cuts[s2], flow}) == 3:
                    break
        else:
            raise AssertionError  # pragma: no cover
        c1, c2 = cuts[s1], cuts[s2]
        back = _return_capacity(arcs, s2)
        spare = out_s - flow
        a_dist = num_dist(
            c1,
            [
                (_inside_capacity(arcs, s1) or None, "cut-sums-pipes-on-one-side"),
                (out_s, "cut-sums-source-pipes"),
                (c1 + _return_capacity(arcs, s1), "cut-counts-backward-pipes"),
                (c1 - min(c for (u, v), c in arcs.items() if u in s1 and v not in s1), "cut-misses-a-crossing-pipe"),
            ],
            3,
        )
        b_dist = num_dist(
            c2,
            [
                (c2 + back, "cut-counts-backward-pipes"),
                (c2 - min(c for (u, v), c in arcs.items() if u in s2 and v not in s2), "cut-misses-a-crossing-pipe"),
                (_inside_capacity(arcs, s2) or None, "cut-sums-pipes-on-one-side"),
                (out_s, "cut-sums-source-pipes"),
            ],
            3,
        )
        c_dist = num_dist(
            flow,
            [
                (min(c1, c2), "flow-equals-capacity-of-a-non-minimum-cut"),
                (out_s, "sum-all-source-outgoing"),
                (in_t, "sum-all-sink-incoming"),
                (_best_single_route(arcs), "flow-single-path-only"),
                (min(arcs.values()), "minimum-pipe-capacity"),
            ],
            3,
        )
        d_dist = num_dist(
            spare,
            [
                (in_t - flow if in_t != out_s else None, "flow-spare-from-sink-side"),
                (flow, "flow-spare-reports-flow"),
                (out_s, "sum-all-source-outgoing"),
                (spare + 1, "graph-count-off-by-one"),
            ],
            3,
        )
        for d in (a_dist, b_dist, c_dist, d_dist):
            enough(d, 2)
        return arcs, s1, s2, c1, c2, flow, out_s, spare, a_dist, b_dist, c_dist, d_dist

    arcs, s1, s2, c1, c2, flow, out_s, spare, a_dist, b_dist, c_dist, d_dist = retry_draw(rng, build)
    cut_line = lambda s: (
        f"puts {_join(s)} on the source side and {_join((set(_junctions(arcs)) | {'S', 'T'}) - set(s))} on the sink side"
    )
    stem = f"{_intro(ctx, arcs)} The first cut {cut_line(s1)}. The second cut {cut_line(s2)}."
    return dict(
        text=stem,
        parts=[
            part(
                f"What is the capacity of the first cut, in {ctx['unit']}?",
                "number",
                num_fields(c1),
                1,
                [f"Add the links from the first cut's source side to its sink side: {c1}."],
                a_dist,
                sensible_range=(0, None),
            ),
            part(
                f"What is the capacity of the second cut, in {ctx['unit']}?",
                "number",
                num_fields(c2),
                1,
                [
                    f"Count only links from the source side to the sink side. The link running back is not counted: {c2}."
                ],
                b_dist,
                sensible_range=(0, None),
            ),
            part(
                f"What is the maximum flow from S to T, in {ctx['unit']}?",
                "number",
                num_fields(flow),
                2,
                [
                    f"The maximum flow equals the capacity of the smallest cut over all cuts, which is {flow} (less than both {c1} and {c2})."
                ],
                c_dist,
                sensible_range=(0, None),
            ),
            part(
                f"The {ctx['noun']}s leaving S have a total capacity of {out_s}. How much of that capacity cannot be used?",
                "number",
                num_fields(spare),
                1,
                [f"{out_s} - {flow} = {spare}."],
                d_dist,
                follow_through=f"{out_s} - part_c",
                sensible_range=(0, None),
            ),
        ],
    )
