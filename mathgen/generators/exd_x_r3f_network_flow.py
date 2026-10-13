"""Generates `exd_x_r3f_network_flow` questions: an exam chain on a small pipe network and an upgrade.

An oil network joins a source S to a sink T through four junctions A, B, C, D. Two pipes cross between the
junctions of a layer (A-B and C-D), each in a random direction. The chain:

(a) the capacity of a stated cut                                       number, 1 mark
(b) the maximum flow from S to T                                       number, 2 marks
(c) the maximum flow after one pipe is upgraded by a stated amount     number, 2 marks
(d) the increase in the maximum flow                                   number, 1 mark    part_c - part_b

Part (d) follows through from (b) and (c). Each maximum flow is found by augmenting paths and re-derived as the
smallest cut capacity over all sixteen ways to split the junctions, so the max-flow min-cut theorem is the
independent check on every key.
"""

from __future__ import annotations

import random
from collections import deque
from itertools import product

from ..core import VerificationError, attempts, check, generator
from ._exd_r3f_parts import number_part
from ._exd_distinct_parts import distinct_part_answers

_JUNCTIONS = ("A", "B", "C", "D")


def _max_flow(capacity: dict) -> int:
    """Edmonds-Karp on a dict {(u, v): capacity}."""
    residual = dict(capacity)
    for u, v in capacity:
        residual.setdefault((v, u), 0)
    nodes = {n for edge in residual for n in edge}
    total = 0
    # Every augmenting path pushes at least one unit, so the total capacity bounds the number of rounds.
    for _ in range(sum(capacity.values()) + 1):
        parent = {"S": None}
        queue = deque(["S"])
        while queue and "T" not in parent:
            u = queue.popleft()
            for v in sorted(nodes):
                if v not in parent and residual.get((u, v), 0) > 0:
                    parent[v] = u
                    queue.append(v)
        if "T" not in parent:
            return total
        path, v = [], "T"
        while parent[v] is not None:
            path.append((parent[v], v))
            v = parent[v]
        push = min(residual[e] for e in path)
        for u, v in path:
            residual[(u, v)] -= push
            residual[(v, u)] += push
        total += push
    raise VerificationError("the flow did not converge")


def _cut_capacity(capacity: dict, source_side: set) -> int:
    return sum(c for (u, v), c in capacity.items() if u in source_side and v not in source_side)


def _min_cut(capacity: dict) -> int:
    best = None
    for picks in product((False, True), repeat=len(_JUNCTIONS)):
        side = {"S"} | {n for n, chosen in zip(_JUNCTIONS, picks) if chosen}
        value = _cut_capacity(capacity, side)
        best = value if best is None else min(best, value)
    return best


def _brace(nodes) -> str:
    """The nodes as a list in words ("S, A and C"), S first and the junctions in order."""
    names = sorted(nodes, key=lambda n: (n != "S", n))
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


@generator("exd_x_r3f_network_flow", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3f_network_flow(rng: random.Random, tier: str = "exam") -> dict:
    """A cut capacity, the maximum flow, the maximum flow after an upgrade, and the increase."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a network whose cut is not the whole story"):
        cap = {
            ("S", "A"): rng.randint(6, 20),
            ("S", "B"): rng.randint(6, 20),
            ("A", "C"): rng.randint(6, 20),
            ("B", "D"): rng.randint(6, 20),
            ("C", "T"): rng.randint(6, 20),
            ("D", "T"): rng.randint(6, 20),
        }
        cross_one = ("A", "B") if rng.random() < 0.5 else ("B", "A")
        cross_two = ("C", "D") if rng.random() < 0.5 else ("D", "C")
        cap[cross_one] = rng.randint(4, 14)
        cap[cross_two] = rng.randint(4, 14)
        picks = [rng.random() < 0.5 for _ in _JUNCTIONS]
        side = {"S"} | {n for n, chosen in zip(_JUNCTIONS, picks) if chosen}
        if len(side) in (1, 5):  # the cut {S} or everything-but-T is the bare source or sink pipes: too easy
            continue
        bump = rng.randint(2, 9)
        base = _max_flow(cap)
        outcomes = {}
        for candidate in sorted(cap):
            bumped = dict(cap)
            bumped[candidate] += bump
            outcomes[candidate] = _max_flow(bumped)
        helpful = [e for e in sorted(cap) if outcomes[e] > base]
        if not helpful:
            continue
        # The upgrade always widens a bottleneck: an upgrade that changes nothing would leave (c) equal to (b) and
        # (d) zero, so two parts would share an answer.
        edge = rng.choice(helpful)
        upgraded = dict(cap)
        upgraded[edge] += bump
        after = outcomes[edge]
        cut = _cut_capacity(cap, side)
        if cut != base or rng.random() < 0.25:  # usually the stated cut is not simply the answer to (b)
            break
    increase = after - base
    check(base == _min_cut(cap) and after == _min_cut(upgraded), "max-flow min-cut check failed")
    check(0 <= increase <= bump and cut >= base, "inconsistent flows")

    t_side = {n for n in ("S", *_JUNCTIONS, "T") if n not in side}
    source_pipes = sum(c for (u, _v), c in cap.items() if u == "S")
    sink_pipes = sum(c for (_u, v), c in cap.items() if v == "T")
    backward = sum(c for (u, v), c in cap.items() if u in t_side and v in side)
    crossing = [c for (u, v), c in cap.items() if u in side and v in t_side]
    inside = sum(c for (u, v), c in cap.items() if u in side and v in side)

    cut_candidates = [
        (cut + backward if backward else None, "cut-counts-backward-pipes"),
        (source_pipes, "cut-sums-source-pipes"),
        (cut - min(crossing), "cut-misses-a-crossing-pipe"),
        (inside, "cut-sums-pipes-on-one-side"),
    ]
    flow_candidates = [
        (source_pipes, "sum-all-source-outgoing"),
        (sink_pipes, "sum-all-sink-incoming"),
        (cut, "flow-equals-capacity-of-a-non-minimum-cut"),
        (min(cap.values()), "minimum-pipe-capacity"),
    ]
    after_candidates = [
        (base + bump, "upgrade-increase-assumed-in-full"),
        (base, "upgrade-ignored"),
        (sum(c for (u, _v), c in upgraded.items() if u == "S"), "sum-all-source-outgoing"),
        (cut + bump, "flow-equals-capacity-of-a-non-minimum-cut"),
    ]
    increase_candidates = [
        (bump, "increase-equals-upgrade"),
        (after, "reports-new-flow-not-increase"),
        (base, "reports-old-flow-not-increase"),
        (0 if increase else None, "upgrade-ignored"),
    ]

    lines = " ".join(
        f"Pipe from {u} to {v}: capacity {c} kL per hour."
        for (u, v), c in sorted(cap.items(), key=lambda item: (item[0][0] != "S", item[0]))
    )
    stem = (
        "An oil storage tank (source S) is connected to a terminal (sink T) through a network of pipes, "
        f"and oil flows along each pipe only in the direction stated. {lines}"
    )
    return dict(
        text=stem,
        parts=[
            number_part(
                "The capacity of a cut is the total capacity of the pipes that carry oil from the side containing S "
                f"to the side containing T. Find the capacity of the cut that separates {_brace(side)} from "
                f"{_brace(t_side)}, in kL per hour.",
                cut,
                cut_candidates,
                marks=1,
                sensible_range=(0, None),
                working=[f"The pipes from {_brace(side)} to {_brace(t_side)} total {cut} kL per hour."],
            ),
            number_part(
                "Find the maximum flow from S to T, in kL per hour.",
                base,
                flow_candidates,
                marks=2,
                sensible_range=(0, None),
                working=[f"The smallest cut has capacity {base} kL per hour, so the maximum flow is {base}."],
            ),
            number_part(
                f"The pipe from {edge[0]} to {edge[1]} is replaced by one with capacity {bump} kL per hour larger. "
                "Find the new maximum flow from S to T, in kL per hour.",
                after,
                after_candidates,
                marks=2,
                sensible_range=(0, None),
                working=[f"With the upgrade the smallest cut has capacity {after} kL per hour."],
            ),
            number_part(
                "Hence find the increase in the maximum flow, in kL per hour.",
                increase,
                increase_candidates,
                marks=1,
                sensible_range=(-1, None),
                follow_through="part_c - part_b",
                working=[f"{after} - {base} = {increase} kL per hour."],
            ),
        ],
    )
