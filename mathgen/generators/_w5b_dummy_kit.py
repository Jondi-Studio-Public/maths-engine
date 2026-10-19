"""Shared engine for the precedence-table skill: the fewest dummy activities an activity-on-arc network needs.

Nothing here registers a topic. An activity network is drawn with each activity as an arrow between two events. A
*dummy* activity is a zero-duration arrow that carries a precedence from one event to another. Given a precedence table
(each activity with the activities that must finish first), `fewest_dummies` finds the network with the fewest dummies.

## The model

Take the closure of the table (an activity's predecessors are all its ancestors). An event is described by the set S of
activities that must be complete for the event to occur:

* the start event has S empty;
* the activities with the same closed predecessor set P start at one event with S = P;
* the final event has S = all activities.

Every activity ends at an event whose S contains it and its own predecessors, and no two activities may share both
their start and end events (they would be indistinguishable arrows). An activity may also end at a private event of its
own (S = its predecessors plus itself), which a dummy then connects onward. An event is reached exactly when the
activities arriving at it, whether directly or along dummies, give precisely S. The search tries every choice of end
event and, for each event, the smallest set of dummies that makes its arrivals exactly S.

`network_table` reads an explicit network back into a precedence table by following dummy-only paths, which is a second
route to the table the search started from: the two must agree.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations, product

from ..core import check


def _closure(table):
    acts = list(table)
    bit = {a: 1 << i for i, a in enumerate(acts)}
    anc = {}
    for a in acts:
        mask = 0
        for p in table[a]:
            mask |= anc[p] | bit[p]
        anc[a] = mask
    return acts, bit, anc


def _search(table):
    acts, bit, anc = _closure(table)
    full = (1 << len(acts)) - 1
    has_succ = {a: any(a in table[b] for b in acts) for a in acts}
    start = ("start", 0)
    final = ("final", full)
    pnodes = {m: ("p", m) for m in {anc[a] for a in acts if anc[a]}}
    node_of_start = {a: (start if anc[a] == 0 else pnodes[anc[a]]) for a in acts}

    def private(a):
        mask = anc[a] | bit[a]
        return pnodes.get(mask, ("priv", mask))

    options = {}
    for a in acts:
        closed = anc[a] | bit[a]
        if has_succ[a]:
            opts = [n for m, n in pnodes.items() if m & closed == closed and m != full]
            opts.append(private(a))
        else:
            opts = [final, private(a)]
        options[a] = sorted(set(opts), key=repr)

    best = None
    for choice in product(*(options[a] for a in acts)):
        end = dict(zip(acts, choice))
        if len({(node_of_start[a], end[a]) for a in acts}) < len(acts):
            continue
        arrivals = {}
        for a in acts:
            arrivals.setdefault(end[a], 0)
            arrivals[end[a]] |= anc[a] | bit[a]
        used = {n for n in arrivals} | set(pnodes.values()) | {final}
        total, dummies, ok = 0, [], True
        for node in used:
            if node == start:
                continue
            mask = node[1]
            have = arrivals.get(node, 0)
            if have == mask:
                continue
            sources = [m for m in used if m != start and m[1] != mask and m[1] & mask == m[1]]
            found = None
            for k in range(1, len(sources) + 1):
                for pick in combinations(sources, k):
                    cover = have
                    for m in pick:
                        cover |= m[1]
                    if cover == mask:
                        found = pick
                        break
                if found:
                    break
            if found is None:
                ok = False
                break
            total += len(found)
            dummies.extend((m, node) for m in found)
        if ok and (best is None or total < best[0]):
            best = (total, end, dummies, node_of_start)
    check(best is not None, "no activity network fits the table")
    return acts, best


def fewest_dummies(table) -> int:
    return _cached(_freeze(table))[0]


def _freeze(table):
    return tuple((a, tuple(table[a])) for a in table)


@lru_cache(maxsize=None)
def _cached(frozen):
    table = {a: list(p) for a, p in frozen}
    acts, (total, end, dummies, node_of_start) = _search(table)
    network = ([(a, node_of_start[a], end[a]) for a in acts], dummies)
    return total, network


def network_of(table):
    return _cached(_freeze(table))[1]


def network_table(network):
    """The closed precedence table read back from an explicit network: activity -> sorted predecessors of any depth."""
    reals, dummies = network
    out_dummy: dict = {}
    for src, dst in dummies:
        out_dummy.setdefault(src, set()).add(dst)

    def reach(node):
        seen, stack = {node}, [node]
        while stack:
            for nxt in out_dummy.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    direct = {a: [p for p, _s, e in reals if s in reach(e)] for a, s, _e in reals}

    def ancestors(a, seen=None):
        out = set()
        for p in direct[a]:
            out |= {p} | ancestors(p)
        return out

    return {a: sorted(ancestors(a)) for a, _s, _e in reals}


def closure_table(table):
    """The table with every activity's predecessors closed under 'is a predecessor of a predecessor'."""
    acts, bit, anc = _closure(table)
    return {a: [b for b in acts if anc[a] & bit[b]] for a in acts}
