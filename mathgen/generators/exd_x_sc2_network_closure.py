"""Exam-tier source for `shortest-network-path`: a road network, a closed road, and the extra distance.

A delivery driver travels between depots joined by roads of stated lengths. Three connected parts:

(a) the shortest distance from the first depot to the last;
(b) the shortest distance once one road on that shortest route is closed (the learner must re-search the network);
(c) how much longer the trip is because of the closure (follow-through from (a) and (b)).

The network has six depots and nine or ten roads. It is drawn so the shortest route in (a) is unique (so the
closed road is unambiguously "on the route"), and so a route still exists after the closure and is longer. Both
distances come from Dijkstra's algorithm; an independent route enumerates every simple path.
"""

from __future__ import annotations

import heapq
import random
from fractions import Fraction

from ..answers import Distractor
from ..core import attempts, check, generator, latex_answer, pick_distractors
from ._network_fig import with_network_figure

_NODES = "ABCDEF"
_PAIRS = [(a, b) for i, a in enumerate(_NODES) for b in _NODES[i + 1 :]]


def _dijkstra(edges: dict, source: str, target: str):
    """Shortest distance from source to target and the number of distinct shortest paths."""
    adj: dict[str, list] = {n: [] for n in _NODES}
    for (u, v), w in edges.items():
        adj[u].append((v, w))
        adj[v].append((u, w))
    best = {n: None for n in _NODES}
    count = {n: 0 for n in _NODES}
    best[source], count[source] = 0, 1
    heap = [(0, source)]
    done = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v, w in adj[u]:
            nd = d + w
            if best[v] is None or nd < best[v]:
                best[v], count[v] = nd, count[u]
                heapq.heappush(heap, (nd, v))
            elif nd == best[v]:
                count[v] += count[u]
    return best[target], count[target]


def _simple_paths(edges: dict, source: str, target: str):
    """Every simple path from source to target as (total, [nodes]) -- the independent route."""
    adj: dict[str, list] = {n: [] for n in _NODES}
    for (u, v), w in edges.items():
        adj[u].append((v, w))
        adj[v].append((u, w))
    out = []

    def walk(node, seen, total):
        if node == target:
            out.append((total, list(seen)))
            return
        for nxt, w in adj[node]:
            if nxt not in seen:
                seen.append(nxt)
                walk(nxt, seen, total + w)
                seen.pop()

    walk(source, [source], 0)
    return out


def _greedy(edges: dict, source: str, target: str):
    """Nearest-neighbour walk: always take the shortest road to an unvisited depot. None if it gets stuck."""
    adj: dict[str, list] = {n: [] for n in _NODES}
    for (u, v), w in edges.items():
        adj[u].append((v, w))
        adj[v].append((u, w))
    node, seen, total = source, {source}, 0
    while node != target:
        options = sorted((w, v) for v, w in adj[node] if v not in seen)
        if not options:
            return None
        w, node = options[0]
        seen.add(node)
        total += w
    return total


def _edge_text(edges: dict) -> str:
    return ", ".join(f"{u}{v} {w}" for (u, v), w in edges.items())


@generator("exd_x_sc2_network_closure", level=9, presets={"exam": dict(tier="exam")})
@with_network_figure
def exd_x_sc2_network_closure(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: shortest route, shortest route after a closure, and the extra distance."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(5000, "a network with a unique shortest route and a longer detour"):
        n_edges = rng.choice((9, 10))
        chosen = rng.sample(_PAIRS, n_edges)
        edges = {pair: rng.randint(3, 16) for pair in sorted(chosen)}
        short, ways = _dijkstra(edges, "A", "F")
        if short is None or ways != 1:
            continue
        paths = _simple_paths(edges, "A", "F")
        best_total, best_path = min(paths)
        if best_total != short or sum(1 for t, _ in paths if t == short) != 1:
            continue
        if len(best_path) < 4:
            continue  # want a route of at least three roads so the closure has real choices
        route_edges = [tuple(sorted(pair)) for pair in zip(best_path, best_path[1:])]
        closed = rng.choice(route_edges)
        after_edges = {p: w for p, w in edges.items() if p != closed}
        after, _ = _dijkstra(after_edges, "A", "F")
        if after is None or after <= short:
            continue
        # Independent re-derivation of (b) by enumerating the remaining simple paths.
        after_paths = _simple_paths(after_edges, "A", "F")
        if min(t for t, _ in after_paths) != after:
            continue
        hops = min(len(p) - 1 for _, p in paths)
        fewest = [t for t, p in paths if len(p) - 1 == hops]
        fewest_total = max(fewest)
        greedy = _greedy(edges, "A", "F")
        break

    closed_weight = edges[closed]
    gap = after - short
    check(gap > 0 and gap == after - short, "extra distance disagrees")

    ordered = sorted(edges.items())
    stem = (
        "A delivery driver travels between six depots, A to F. The roads between depots and their lengths in "
        f"kilometres are: {', '.join(f'{u}{v} {w}' for (u, v), w in ordered)}. "
        "The driver wants the shortest route from depot A to depot F."
    )
    closed_text = f"{closed[0]}{closed[1]}"

    a_cands = [
        Distractor(str(fewest_total), "fewest-edges-route", Fraction(fewest_total)) if fewest_total != short else None,
        Distractor(str(greedy), "nearest-neighbour-route", Fraction(greedy)) if greedy is not None else None,
        Distractor(str(sorted(t for t, _ in paths)[1]), None, Fraction(sorted(t for t, _ in paths)[1])),
        Distractor(str(short + 1), None, Fraction(short + 1)),
        Distractor(str(short - 1), None, Fraction(short - 1)),
        Distractor(str(short + 2), None, Fraction(short + 2)),
    ]
    b_cands = [
        Distractor(str(short), "ignored-closure", Fraction(short)),
        Distractor(str(short - closed_weight), "removed-closed-road-length", Fraction(short - closed_weight))
        if short - closed_weight > 0
        else None,
        Distractor(str(after + closed_weight), None, Fraction(after + closed_weight)),
        Distractor(str(after - 1), None, Fraction(after - 1)),
        Distractor(str(after + 2), None, Fraction(after + 2)),
    ]
    c_cands = [
        Distractor(str(after), "gave-new-total-not-difference", Fraction(after)),
        Distractor(str(closed_weight), "gave-closed-road-length", Fraction(closed_weight)),
        Distractor(str(after + short), "added-instead-of-subtracted", Fraction(after + short)),
        Distractor(str(gap + 1), None, Fraction(gap + 1)),
        Distractor(str(gap + 2), None, Fraction(gap + 2)),
        Distractor(str(gap - 1), None, Fraction(gap - 1)) if gap > 1 else None,
    ]
    a_dis = pick_distractors(str(short), [c for c in a_cands if c], 3)
    b_dis = pick_distractors(str(after), [c for c in b_cands if c], 3)
    c_dis = pick_distractors(str(gap), [c for c in c_cands if c], 3)
    check(len(a_dis) >= 2 and len(b_dis) >= 2 and len(c_dis) >= 2, "too few distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the length of the shortest route from A to F, in kilometres.",
                answer_kind="number",
                answer=short,
                answer_text=str(short),
                answer_latex=latex_answer(str(short)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    "Compare every route from A to F and add the road lengths of each.",
                    f"The shortest is {'-'.join(best_path)}, with length {short} km.",
                ],
                distractors=a_dis,
            ),
            dict(
                prompt=f"The road {closed_text} is closed. Find the length of the shortest route from A to F "
                "that does not use it, in kilometres.",
                answer_kind="number",
                answer=after,
                answer_text=str(after),
                answer_latex=latex_answer(str(after)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Remove road {closed_text} ({closed_weight} km) and compare the remaining routes.",
                    f"The shortest route that does not use {closed_text} is {after} km long.",
                ],
                distractors=b_dis,
            ),
            dict(
                prompt="By how many kilometres is the shortest trip longer because of the closure?",
                answer_kind="number",
                answer=gap,
                answer_text=str(gap),
                answer_latex=latex_answer(str(gap)),
                marks=1,
                sensible_range=(0, None),
                follow_through="part_b - part_a",
                working=[f"{after} - {short} = {gap} km."],
                distractors=c_dis,
            ),
        ],
    )
