"""Exam source of `senior-use-minimal-spanning-trees-to-solve-minimal-connector`: a cable fails and must be re-routed.

The practice tiers find the cheapest set of cables joining every town. An exam question first asks for that network,
then breaks one of its cables and asks what the failure costs:

(a) the minimum total cost of connecting the seven towns;
(b) when one cable of that network cannot be laid, the cost of the cheapest cable that re-joins the two groups of
    towns it leaves apart (not simply the cheapest spare cable, which may join towns that are already connected);
(c) the new minimum total cost, from (a), the cost of the failed cable and (b).

Part (c) follows through from (a) and (b), so a wrong earlier answer still earns it.

## Construction

Seven towns, eleven cables with all-different costs (so the cheapest network is unique). The failed cable is always
in the cheapest network, and the draw keeps only graphs where the cheapest spare cable overall is NOT the one that
re-joins the groups, so (b) cannot be answered by looking for the cheapest unused cable. The key of every part is
checked by brute force over all sets of six cables.
"""

from __future__ import annotations

import string
from itertools import combinations

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors
from ._exd_distinct_parts import distinct_part_answers
from ._network_fig import with_network_figure

_TOWNS = string.ascii_uppercase[:7]


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _kruskal(edges):
    parent = {t: t for t in _TOWNS}
    tree = []
    for e in sorted(edges, key=lambda e: e[2]):
        a, b = _find(parent, e[0]), _find(parent, e[1])
        if a != b:
            parent[a] = b
            tree.append(e)
    return tree if len(tree) == len(_TOWNS) - 1 else None


def _is_spanning_tree(combo) -> bool:
    parent = {t: t for t in _TOWNS}
    for u, v, _w in combo:
        ru, rv = _find(parent, u), _find(parent, v)
        if ru == rv:
            return False
        parent[ru] = rv
    return True


def _brute_force(edges):
    """Independent route: the cheapest of every set of six cables that joins all seven towns."""
    totals = [
        sum(w for _u, _v, w in combo) for combo in combinations(edges, len(_TOWNS) - 1) if _is_spanning_tree(combo)
    ]
    return min(totals) if totals else None


def _draw_graph(rng):
    order = list(_TOWNS)
    rng.shuffle(order)
    pairs = {tuple(sorted((order[i], order[i + 1]))) for i in range(len(order) - 1)}
    rest = [p for p in combinations(_TOWNS, 2) if p not in pairs]
    rng.shuffle(rest)
    pairs |= set(rest[:5])
    weights = rng.sample(range(4, 46), len(pairs))
    return [(u, v, w) for (u, v), w in zip(sorted(pairs), weights)]


def _component(tree, start, skip):
    seen, stack = {start}, [start]
    while stack:
        t = stack.pop()
        for u, v, _ in tree:
            if (u, v, _) == skip:
                continue
            for a, b in ((u, v), (v, u)):
                if a == t and b not in seen:
                    seen.add(b)
                    stack.append(b)
    return seen


def _build(rng):
    for _ in attempts(3000, "a graph where the failed cable's replacement is not the cheapest spare cable"):
        edges = _draw_graph(rng)
        tree = _kruskal(edges)
        if tree is None:
            continue
        failed = rng.choice(tree)
        side = _component(tree, failed[0], failed)
        crossing = [e for e in edges if e != failed and ((e[0] in side) != (e[1] in side))]
        if not crossing:
            continue
        replacement = min(crossing, key=lambda e: e[2])
        spare = [e for e in edges if e not in tree]
        cheapest_spare = min(spare, key=lambda e: e[2])
        if cheapest_spare == replacement or replacement[2] == failed[2]:
            continue
        base = sum(w for _u, _v, w in tree)
        remaining = [e for e in edges if e != failed]
        new_tree = _kruskal(remaining)
        if new_tree is None:
            continue
        new_total = sum(w for _u, _v, w in new_tree)
        if new_total != base - failed[2] + replacement[2]:
            continue
        return edges, tree, failed, replacement, cheapest_spare, base, new_total
    raise RuntimeError("no graph found")


@generator("exd_x_r3a_connector_reroute", level=12, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
@with_network_figure
def exd_x_r3a_connector_reroute(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "enough distractors in every part"):
        edges, tree, failed, replacement, cheapest_spare, base, new_total = _build(rng)
        check(_brute_force(edges) == base, "brute force disagrees on the cheapest network")
        check(_brute_force([e for e in edges if e != failed]) == new_total, "brute force disagrees after the failure")
        w_failed, w_new = failed[2], replacement[2]
        sorted_w = sorted(e[2] for e in edges)
        a_d = pick_distractors(
            str(base),
            [
                Distractor(str(sum(e[2] for e in edges)), "all-cables-summed", sum(e[2] for e in edges)),
                Distractor(str(sum(sorted_w[:6])), "cheapest-cables-without-cycle-check", sum(sorted_w[:6])),
                Distractor(
                    str(base - max(e[2] for e in tree)), "one-town-left-unconnected", base - max(e[2] for e in tree)
                ),
                Distractor(
                    str(base + min(e[2] for e in edges if e not in tree)),
                    "extra-spare-cable-added",
                    base + min(e[2] for e in edges if e not in tree),
                ),
            ],
            3,
        )
        b_d = pick_distractors(
            str(w_new),
            [
                Distractor(str(cheapest_spare[2]), "cheapest-spare-cable-not-checked", cheapest_spare[2]),
                Distractor(str(w_failed), "failed-cable-cost-repeated", w_failed),
                Distractor(str(base - w_failed), "network-cost-for-cable-cost", base - w_failed),
                Distractor(str(w_new + w_failed), "costs-added-for-replacement", w_new + w_failed),
            ],
            3,
        )
        c_d = pick_distractors(
            str(new_total),
            [
                Distractor(str(base - w_failed), "failed-cable-not-replaced", base - w_failed),
                Distractor(str(base), "failure-ignored", base),
                Distractor(str(base + w_new), "replacement-added-failed-kept", base + w_new),
                Distractor(
                    str(base - w_failed + cheapest_spare[2]),
                    "cheapest-spare-cable-not-checked",
                    base - w_failed + cheapest_spare[2],
                ),
            ],
            3,
        )
        if min(len(a_d), len(b_d), len(c_d)) >= 2 and len({str(base), str(w_new), str(new_total)}) == 3:
            break

    options = " ".join(f"{u}-{v}, ${w}." for u, v, w in edges)
    stem = (
        "A power company is connecting seven towns, A to G, by cable. Every town must be connected to the others, "
        f"directly or through other towns. The possible cables and their costs, in thousands of dollars, are: {options}"
    )
    rule = f"The cable {failed[0]}-{failed[1]} in this network cannot be laid because the ground there is unsuitable."
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the minimum total cost, in thousands of dollars, of a cable network that connects all "
                "seven towns.",
                answer_kind="number",
                answer=base,
                answer_text=str(base),
                answer_latex=latex_answer(str(base)),
                marks=3,
                sensible_range=(0, None),
                working=[
                    "Choose the cheapest cable that does not close a cycle, six times: "
                    + ", ".join(f"{u}-{v} (${w})" for u, v, w in sorted(tree, key=lambda e: e[2]))
                    + f". Total ${base}."
                ],
                distractors=a_d,
            ),
            dict(
                prompt=f"{rule} Removing it leaves two groups of towns that are not connected to each other. "
                "Find the cost, in thousands of dollars, of the cheapest cable that joins the two groups.",
                answer_kind="number",
                answer=w_new,
                answer_text=str(w_new),
                answer_latex=latex_answer(str(w_new)),
                marks=3,
                sensible_range=(0, None),
                working=[f"The cheapest cable between the two groups is {replacement[0]}-{replacement[1]}, ${w_new}."],
                distractors=b_d,
            ),
            dict(
                prompt="Using your answers to (a) and (b), find the new minimum total cost, in thousands of dollars, "
                f"of a network that connects all seven towns without the cable {failed[0]}-{failed[1]}.",
                answer_kind="number",
                answer=new_total,
                answer_text=str(new_total),
                answer_latex=latex_answer(str(new_total)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a - {w_failed} + part_b",
                working=[f"${base} - ${w_failed} + ${w_new} = ${new_total}."],
                distractors=c_d,
            ),
        ],
    )
