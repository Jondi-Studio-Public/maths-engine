"""Exam-tier network questions: a constrained minimal connector, and the effect of upgrading one pipe.

  exd_x_g6_connector_constraint  Seven towns and eleven possible cables.  One cable is either REQUIRED (it must
                                 be laid) or UNAVAILABLE (it cannot be laid), and the learner must find the
                                 cheapest network that obeys that.  The requirement is drawn so that it matters:
                                 a required cable is never in the unconstrained cheapest network, and an
                                 unavailable cable is always in it.
  exd_x_g6_flow_upgrade          A six-node pipe network.  One pipe is upgraded by u kL per hour; the learner
                                 reports how much the MAXIMUM FLOW rises.  The draw guarantees the rise is
                                 positive but smaller than u, so neither "no change" nor "the full upgrade" is
                                 correct, and the learner must locate the new bottleneck.

Weights in the connector questions are all different, so the cheapest tree is unique.
"""

from __future__ import annotations

import random
import string
from itertools import combinations

from mathgen.core import check, check_equivalent, generator, latex_answer, pick_distractors

# --- minimal connector with a constraint --------------------------------------------------------------------

_TOWNS = string.ascii_uppercase[:7]


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _kruskal(edges, forced=None):
    """(total weight, chosen edges) of the cheapest spanning tree that contains ``forced`` (an edge or None)."""
    parent = {t: t for t in _TOWNS}
    total, chosen = 0, []
    if forced is not None:
        parent[_find(parent, forced[0])] = _find(parent, forced[1])
        total, chosen = forced[2], [forced]
    for e in sorted(edges, key=lambda e: e[2]):
        a, b = _find(parent, e[0]), _find(parent, e[1])
        if a != b:
            parent[a] = b
            total += e[2]
            chosen.append(e)
    connected = len({_find(parent, t) for t in _TOWNS}) == 1
    return (total if connected else None), chosen


def _subset_search(edges, banned=None, required=None):
    """Independent route: try every set of six cables."""
    best = None
    for combo in combinations(edges, len(_TOWNS) - 1):
        if banned is not None and banned in combo:
            continue
        if required is not None and required not in combo:
            continue
        parent = {t: t for t in _TOWNS}
        ok = True
        for u, v, _w in combo:
            a, b = _find(parent, u), _find(parent, v)
            if a == b:
                ok = False
                break
            parent[a] = b
        if ok:
            total = sum(w for _u, _v, w in combo)
            if best is None or total < best:
                best = total
    return best


def _draw_graph(rng):
    order = list(_TOWNS)
    rng.shuffle(order)
    pairs = {tuple(sorted((order[i], order[i + 1]))) for i in range(len(order) - 1)}
    rest = [p for p in combinations(_TOWNS, 2) if p not in pairs]
    rng.shuffle(rest)
    pairs |= set(rest[:5])
    pairs = sorted(pairs)
    weights = rng.sample(range(4, 46), len(pairs))
    return [(u, v, w) for (u, v), w in zip(pairs, weights)]


@generator("exd_x_g6_connector_constraint", level=12, presets={"exam": dict(tier="exam")})
def connector_constraint(rng: random.Random, tier: str = "exam") -> dict:
    """Cheapest cable network when one cable is required or one is unavailable."""
    for _ in range(2000):
        edges = _draw_graph(rng)
        base, tree = _kruskal(edges)
        require = rng.random() < 0.5
        if require:
            outside = [e for e in edges if e not in tree]
            e = rng.choice(outside)
            answer, _ = _kruskal(edges, forced=e)
            naive = [base, base + e[2], base + e[2] - max(w for _u, _v, w in tree)]
            check_equivalent(_subset_search(edges, required=e), answer, "subset search disagrees (required)")
        else:
            e = rng.choice(tree)
            rest = [x for x in edges if x != e]
            answer, _ = _kruskal(rest)
            if answer is None:
                continue
            unused = [x for x in edges if x not in tree]
            cheapest_unused = min(w for _u, _v, w in unused)
            naive = [base, base - e[2], base - e[2] + cheapest_unused]
            check_equivalent(_subset_search(edges, banned=e), answer, "subset search disagrees (unavailable)")
        if answer <= base:
            continue
        break
    else:
        raise RuntimeError("exd_x_g6_connector_constraint: no valid draw")
    options = " ".join(f"Cable option: {u}-{v}, cost ${w}." for u, v, w in edges)
    rule = (
        f"The cable {e[0]}-{e[1]} must be laid."
        if require
        else f"The cable {e[0]}-{e[1]} cannot be laid because the ground there is unsuitable."
    )
    text = (
        "A power company is connecting seven towns, A to G, by cable. Every town must be connected to the others, "
        f"directly or through other towns. {options} {rule} "
        "What is the minimum total cost, in dollars, of a cable network that connects all seven towns?"
    )
    answer_text = str(answer)
    distractors = pick_distractors(answer_text, [str(c) for c in naive])
    check(len(distractors) >= 2, "need at least two distractors")
    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer),
        working=[f"The cheapest network that obeys the rule costs ${answer}; without the rule it would cost ${base}."],
        distractors=distractors,
    )


# --- maximum flow with one upgraded pipe ----------------------------------------------------------------------

_FLOW_NODES = ("S", "A", "B", "C", "D", "T")
_FLOW_SHAPES = (
    (("S", "A"), ("S", "B"), ("A", "C"), ("A", "D"), ("B", "D"), ("C", "T"), ("D", "T"), ("C", "D")),
    (("S", "A"), ("S", "B"), ("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("C", "T"), ("D", "T")),
    (("S", "A"), ("S", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "T"), ("D", "T"), ("A", "D")),
)


def _max_flow_augment(caps):
    """Augmenting paths found breadth-first on the residual network."""
    res = {}
    for (u, v), c in caps.items():
        res[(u, v)] = res.get((u, v), 0) + c
        res.setdefault((v, u), 0)
    flow = 0
    while True:
        prev = {"S": None}
        queue = ["S"]
        while queue and "T" not in prev:
            u = queue.pop(0)
            for v in _FLOW_NODES:
                if res.get((u, v), 0) > 0 and v not in prev:
                    prev[v] = u
                    queue.append(v)
        if "T" not in prev:
            return flow
        bottleneck, v = None, "T"
        while prev[v] is not None:
            c = res[(prev[v], v)]
            bottleneck = c if bottleneck is None else min(bottleneck, c)
            v = prev[v]
        v = "T"
        while prev[v] is not None:
            res[(prev[v], v)] -= bottleneck
            res[(v, prev[v])] += bottleneck
            v = prev[v]
        flow += bottleneck


def _min_cut(caps):
    """Independent route: smallest capacity leaving any set of nodes that contains S but not T."""
    middle = [n for n in _FLOW_NODES if n not in ("S", "T")]
    best = None
    for mask in range(1 << len(middle)):
        side = {"S"} | {n for i, n in enumerate(middle) if mask >> i & 1}
        cut = sum(c for (u, v), c in caps.items() if u in side and v not in side)
        best = cut if best is None else min(best, cut)
    return best


@generator("exd_x_g6_flow_upgrade", level=12, presets={"exam": dict(tier="exam")})
def flow_upgrade(rng: random.Random, tier: str = "exam") -> dict:
    """Increase in maximum flow when one pipe's capacity rises by u."""
    for _ in range(4000):
        shape = rng.choice(_FLOW_SHAPES)
        caps = {e: rng.randint(5, 20) for e in shape}
        before = _max_flow_augment(caps)
        e = rng.choice(shape)
        u = rng.randint(3, 9)
        upgraded = dict(caps)
        upgraded[e] += u
        after = _max_flow_augment(upgraded)
        rise = after - before
        if not 0 < rise < u:
            continue
        if len({rise, u, after, before}) < 4:
            continue
        break
    else:
        raise RuntimeError("exd_x_g6_flow_upgrade: no valid draw")
    check_equivalent(_min_cut(caps), before, "max-flow/min-cut disagree before")
    check_equivalent(_min_cut(upgraded), after, "max-flow/min-cut disagree after")
    pipes = " ".join(f"Pipe from {a} to {b}: capacity {c} kL per hour." for (a, b), c in caps.items())
    text = (
        "An oil storage tank (source S) is connected to a terminal (sink T) through a network of pipes. "
        f"{pipes} The pipe from {e[0]} to {e[1]} is replaced by one whose capacity is {u} kL per hour larger. "
        "By how many kL per hour does the maximum flow from S to T increase?"
    )
    answer_text = str(rise)
    distractors = pick_distractors(answer_text, [str(u), str(after), str(before)])
    check(len(distractors) >= 2, "need at least two distractors")
    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=rise,
        answer_text=answer_text,
        answer_latex=latex_answer(rise),
        working=[f"Maximum flow rises from {before} to {after} kL per hour, an increase of {rise}."],
        distractors=distractors,
    )
