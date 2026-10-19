"""Generates `w4_graph_bridges` questions (easy, medium, hard) and the exam chain `w4_graph_bridges_chain`.

``graph-bridges`` (``financial-discrete``): a bridge is an edge whose removal leaves the network in two separate
pieces. Every network here is drawn with the ``network`` figure and is not listed in the text, so the picture is the
only source of the edges. It is built from blocks (a single vertex, a triangle, a four-cycle with or without a
diagonal, a five-cycle) joined in a tree by single edges, so the bridges are exactly the joining edges and no others; the key is
then found a second way, by removing each edge in turn and testing connectedness.

* easy (``number``): how many bridges the network has. Wrong counts are the leaves, the cut vertices, the edges on
  cycles and one more or fewer than the answer;
* medium (``number``): a named bridge is removed; how many places can still be reached from A. The learner must see
  that the edge is a bridge and count the piece that holds A;
* hard (``number``): each edge carries a cost and two edges are to be removed to save their costs, so long as the
  network stays in one piece. No bridge can go, and the two dearest edges on cycles must not be a pair that cuts the
  network (checked by trying every pair); what is the greatest total saving?;
* exam (``w4_graph_bridges_chain``): (a) the number of bridges, (b) the pieces left when every bridge is removed
  (carried forward from (a)), (c) the bridges left after a stated new edge is added, (d) the fewest new edges that
  leave the network with no bridge, found by trying every set of new edges and checked against the leaf count of the
  tree of blocks.
"""

from __future__ import annotations

from itertools import combinations

from ..core import attempts, check, generator
from ._network_fig import graph_figure
from ._snet_kit import LETTERS, is_connected, neighbours, norm, pick_context
from ._sr2a_kit import number_part, single


# -- graphs and their bridges -------------------------------------------------------------------------------------


def bridges_of(n, edges):
    """The edges whose removal disconnects the (connected) network: found by removing each edge in turn."""
    return sorted(e for e in edges if not is_connected(n, [f for f in edges if f != e]))


def cut_vertices(n, edges):
    out = []
    for v in range(n):
        keep = [u for u in range(n) if u != v]
        index = {u: i for i, u in enumerate(keep)}
        rest = [(index[a], index[b]) for a, b in edges if v not in (a, b)]
        if not is_connected(n - 1, rest):
            out.append(v)
    return out


def pieces_without(n, edges, gone):
    """How many connected pieces are left when the edges in ``gone`` are removed."""
    nbr = neighbours(n, [e for e in edges if e not in gone])
    seen, count = set(), 0
    for s in range(n):
        if s in seen:
            continue
        count += 1
        stack = [s]
        seen.add(s)
        while stack:
            for w in nbr[stack.pop()]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
    return count


_BLOCKS = (1, 1, 1, 3, 3, 4, 4, 5)


def block_graph(rng, n_lo, n_hi, blocks_lo, blocks_hi, need_cycle=True):
    """A connected network built from blocks joined in a tree: ``(n, edges, bridges_by_construction)``.

    The blocks are a single vertex, a triangle, a four-cycle (sometimes with a diagonal) or a five-cycle. The edges that join
    blocks are the bridges, and no edge inside a block is one, so the bridges are known from the construction."""
    for _ in attempts(500, "a block graph"):
        sizes = [rng.choice(_BLOCKS) for _ in range(rng.randint(blocks_lo, blocks_hi))]
        n = sum(sizes)
        if not n_lo <= n <= n_hi or (need_cycle and max(sizes) < 3):
            continue
        inside, firsts, start = [], [], 0
        for size in sizes:
            firsts.append(list(range(start, start + size)))
            ring = list(range(start, start + size))
            if size >= 3:
                inside += [(ring[i], ring[(i + 1) % size]) for i in range(size)]
                if size == 4 and rng.random() < 0.4:
                    inside.append((ring[0], ring[2]))
            start += size
        joins = []
        for i in range(1, len(sizes)):
            j = rng.randrange(i)
            joins.append((rng.choice(firsts[i]), rng.choice(firsts[j])))
        perm = list(range(n))
        rng.shuffle(perm)
        edges = sorted(norm(perm[a], perm[b]) for a, b in inside + joins)
        bridges = sorted(norm(perm[a], perm[b]) for a, b in joins)
        return n, edges, bridges
    raise AssertionError  # pragma: no cover


def _name(e):
    return LETTERS[e[0]] + LETTERS[e[1]]


def _picture(n, edges, ctx, weights=None):
    return graph_figure(
        n,
        edges,
        weights=weights,
        alt=f"A network of {ctx['vs']} drawn as labelled vertices joined by straight "
        + ("edges, each labelled with a cost." if weights else "edges."),
    )


_DEFINITION = (
    "A bridge is a {e} that, if it were removed, would leave some {vs} that can no longer be reached from the others."
)


def _intro(ctx, n):
    return f"{ctx['intro'].format(n=n, a=LETTERS[0], z=LETTERS[n - 1])} The diagram shows which {ctx['vs']} are joined."


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(300, "an easy bridge count"):
        ctx = pick_context(rng)
        n, edges, built = block_graph(rng, 5, 8, 1, 5)
        found = bridges_of(n, edges)
        check(found == built, "bridges by removal differ from the construction")
        k = len(found)
        deg = [len(x) for x in neighbours(n, edges)]
        leaves = deg.count(1)
        cuts = len(cut_vertices(n, edges))
        wrong = [
            (leaves, "w4-bridge-counts-leaves"),
            (cuts, "w4-bridge-counts-cut-vertices"),
            (len(edges) - k, "w4-bridge-counts-cycle-edges"),
            (k + 1, "w4-bridge-pieces-off-by-one"),
            (k - 1, "w4-bridge-pieces-off-by-one"),
        ]
        working = [
            "Remove each edge in turn and see whether every place can still be reached.",
            f"Removing {', '.join(_name(e) for e in found)} splits the network, so there are {k}."
            if k
            else "No single edge splits the network.",
            "An edge on a cycle never splits the network, because the rest of the cycle still joins its ends.",
        ]
        text = f"{_intro(ctx, n)} {_DEFINITION.format(**ctx)} How many bridges does the network have?"
        q = single(text, k, working, wrong, lo=0, figure=_picture(n, edges, ctx))
        if q:
            return q
    raise AssertionError  # pragma: no cover


# -- medium -------------------------------------------------------------------------------------------------------


def _reach(n, edges, start=0):
    """The vertices that can be reached from ``start``, by a search."""
    nbr = neighbours(n, edges)
    seen, stack = {start}, [start]
    while stack:
        for w in nbr[stack.pop()]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return seen


def _medium(rng):
    for _ in attempts(500, "a bridge-removal network"):
        ctx = pick_context(rng)
        n, edges, built = block_graph(rng, 5, 8, 2, 5)
        found = bridges_of(n, edges)
        check(found == built, "bridges by removal differ from the construction")
        gone = rng.choice(found)
        rest = [e for e in edges if e != gone]
        side = len(_reach(n, rest))
        # Second way: the two sides are the two pieces, and the edge joined them.
        check(0 < side < n and pieces_without(n, edges, {gone}) == 2, "removing the bridge must leave two pieces")
        wrong = [
            (n, "w4-bridge-removal-ignored"),
            (n - side, "w4-bridge-wrong-side"),
            (n - 1, "w4-bridge-only-endpoint-lost"),
            (side + 1, "w4-bridge-wrong-side"),
            (side - 1, "w4-bridge-wrong-side"),
            (n - 2, "w4-bridge-only-endpoint-lost"),
        ]
        name = _name(gone)
        working = [
            f"{name} is a bridge: with it removed the {ctx['vs']} fall into two separate pieces.",
            f"Counting the {ctx['vs']} in the piece that holds {LETTERS[0]}, including {LETTERS[0]} itself, gives {side}.",
        ]
        text = (
            f"{_intro(ctx, n)} The {ctx['e']} {name} is removed. Counting {LETTERS[0]} itself, how many {ctx['vs']} "
            f"can still be reached from {LETTERS[0]} by following the {ctx['es']} that are left?"
        )
        q = single(text, side, working, wrong, lo=1, figure=_picture(n, edges, ctx))
        if q:
            return q
    raise AssertionError  # pragma: no cover


# -- hard ---------------------------------------------------------------------------------------------------------


def _connected_after(n, edges, gone):
    return is_connected(n, [e for e in edges if e not in gone])


def _hard(rng):
    for _ in attempts(2000, "a costed bridge network"):
        ctx = pick_context(rng)
        n, edges, built = block_graph(rng, 6, 8, 3, 4)
        found = bridges_of(n, edges)
        check(found == built, "bridges by removal differ from the construction")
        cyclic = [e for e in edges if e not in found]
        if len(cyclic) < 4 or len(found) < 2:
            continue
        costs = rng.sample(range(3, 41), len(edges))
        weight = dict(zip(edges, costs))
        # Second way: try every pair of edges.
        pairs = [(a, b) for a, b in combinations(edges, 2) if _connected_after(n, edges, {a, b})]
        if not pairs:
            continue
        best = max(weight[a] + weight[b] for a, b in pairs)
        # The pair of dearest removable edges must be the trap: removing both would cut the network.
        by_cost = sorted(cyclic, key=lambda e: -weight[e])
        top_two = (by_cost[0], by_cost[1])
        if _connected_after(n, edges, set(top_two)):
            continue
        two_dearest_overall = sum(sorted(costs, reverse=True)[:2])
        check(all(a in cyclic and b in cyclic for a, b in pairs), "a bridge cannot be removed")
        wrong = [
            (weight[top_two[0]] + weight[top_two[1]], "w4-bridge-pair-cuts-network"),
            (two_dearest_overall, "w4-bridge-heaviest-edge"),
            (sum(weight[e] for e in cyclic), "w4-bridge-sum-not-max"),
            (sum(weight[e] for e in found), "w4-bridge-sum-of-bridges"),
            (weight[by_cost[0]], "w4-bridge-one-edge-only"),
            (best - min(weight[a] for a, _ in pairs), "w4-bridge-pair-cuts-network"),
        ]
        chosen = max(pairs, key=lambda p: weight[p[0]] + weight[p[1]])
        working = [
            f"The bridges are {', '.join(_name(e) for e in found)}; none of them can be removed.",
            f"The two dearest edges that lie on cycles, {_name(top_two[0])} and {_name(top_two[1])}, cannot both go: "
            "removing both cuts the network.",
            f"The best pair that still leaves one piece is {_name(chosen[0])} and {_name(chosen[1])}, saving {best}.",
        ]
        text = (
            f"{_intro(ctx, n)} The number on each {ctx['e']} is the cost, in dollars, of keeping it. "
            f"Two {ctx['es']} are to be removed to save their costs, but every {ctx['v']} must still be able to reach "
            f"every other {ctx['v']} afterwards. What is the greatest total saving, in dollars?"
        )
        q = single(text, best, working, wrong, lo=1, figure=_picture(n, edges, ctx, weights=weight))
        if q:
            return q
    raise AssertionError  # pragma: no cover


@generator(
    "w4_graph_bridges",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w4_graph_bridges(rng, tier="easy"):
    """Bridges of a drawn network; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


def fewest_new_edges_for_no_bridge(n, edges, limit=3):
    """The fewest new edges after which the network has no bridge, by trying every set of new edges."""
    absent = [p for p in combinations(range(n), 2) if p not in set(edges)]
    for k in range(0, limit + 1):
        for extra in combinations(absent, k):
            if not bridges_of(n, [*edges, *extra]):
                return k
    return None


@generator("w4_graph_bridges_chain", level=12, presets={"exam": dict(tier="exam")})
def w4_graph_bridges_chain(rng, tier="exam"):
    """Exam chain: bridges, pieces left without them, bridges after a new edge, new edges to remove every bridge."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(600, "a bridge exam network"):
        ctx = pick_context(rng)
        n, edges, built = block_graph(rng, 7, 8, 4, 5)
        found = bridges_of(n, edges)
        check(found == built, "bridges by removal differ from the construction")
        k = len(found)
        if k not in (3, 4):
            continue
        absent = [p for p in combinations(range(n), 2) if p not in set(edges)]
        choices = []
        for p in absent:
            after = len(bridges_of(n, [*edges, p]))
            if 0 < after < k:
                choices.append((p, after))
        if not choices:
            continue
        new, after = rng.choice(choices)
        deg = [len(x) for x in neighbours(n, edges)]
        leaf_blocks = pieces_leaf_count(n, edges, found)
        fewest = fewest_new_edges_for_no_bridge(n, edges)
        if fewest is None:
            continue
        check(fewest == (leaf_blocks + 1) // 2, "fewest new edges differs from half the leaf blocks, rounded up")
        pieces = pieces_without(n, edges, set(found))
        check(pieces == k + 1, "pieces left differ from bridges plus one")
        new_name = LETTERS[new[0]] + LETTERS[new[1]]
        stem = (
            f"{_intro(ctx, n)} {_DEFINITION.format(**ctx)} "
            f"A {ctx['e']} {new_name} is to be built as well; it is not in the diagram."
        )
        cut_count = len(cut_vertices(n, edges))
        part_a = number_part(
            f"How many bridges does the network in the diagram have, before {new_name} is built?",
            k,
            1,
            [
                f"Removing each edge in turn, {k} of them leave some {ctx['vs']} cut off: {', '.join(_name(e) for e in found)}."
            ],
            [
                (deg.count(1), "w4-bridge-counts-leaves"),
                (cut_count, "w4-bridge-counts-cut-vertices"),
                (len(edges) - k, "w4-bridge-counts-cycle-edges"),
                (k + 1, "w4-bridge-pieces-off-by-one"),
                (k - 1, "w4-bridge-pieces-off-by-one"),
                (k + 2, "w4-bridge-pieces-off-by-one"),
                (cut_count + 1, "w4-bridge-counts-cut-vertices"),
                (deg.count(1) + 1, "w4-bridge-counts-leaves"),
            ],
            lo=0,
        )
        part_b = number_part(
            "Every bridge is removed from the network in the diagram. How many separate pieces are left?",
            pieces,
            2,
            ["Each removed bridge splits one piece in two, so the pieces number one more than the bridges."],
            [
                (k, "w4-bridge-pieces-off-by-one"),
                (k + 2, "w4-bridge-pieces-off-by-one"),
                (n, "w4-bridge-pieces-all-vertices"),
                (k - 1, "w4-bridge-pieces-off-by-one"),
            ],
            follow="part_a + 1",
            lo=1,
        )
        part_c = number_part(
            f"After {new_name} is built, how many bridges does the network have?",
            after,
            2,
            [
                f"{new_name} makes a new cycle with the path between its ends, and every bridge on that path stops "
                f"being a bridge. {after} of the {k} bridges are left."
            ],
            [
                (k, "w4-bridge-new-road-ignored"),
                (k - 1, "w4-bridge-new-road-removes-one"),
                (0, "w4-bridge-new-road-clears-all"),
                (after + 1, "w4-bridge-new-road-removes-one"),
                (k + 1, "w4-bridge-new-road-ignored"),
            ],
            lo=0,
        )
        part_d = number_part(
            f"Ignore {new_name}. What is the fewest new {ctx['es']} that must be built so that the network in the "
            "diagram has no bridge?",
            fewest,
            3,
            [
                f"Joining the ends of the tree of pieces pairs up its {leaf_blocks} end pieces, "
                f"so ({leaf_blocks} + 1) / 2 rounded down = {fewest} new {ctx['es']} are needed."
            ],
            [
                (leaf_blocks, "w4-bridge-repair-leaves"),
                (k, "w4-bridge-repair-per-bridge"),
                (fewest + 1, "w4-bridge-repair-leaves"),
                (fewest - 1, "w4-bridge-repair-leaves"),
                (k - 1, "w4-bridge-repair-per-bridge"),
                (k + 1, "w4-bridge-repair-per-bridge"),
                (fewest + 2, "w4-bridge-repair-leaves"),
                (leaf_blocks + 1, "w4-bridge-repair-leaves"),
            ],
            lo=1,
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        return dict(text=stem, figure=_picture(n, edges, ctx), parts=[part_a, part_b, part_c, part_d])
    raise AssertionError  # pragma: no cover


def pieces_leaf_count(n, edges, bridges):
    """The number of pieces (left when every bridge is removed) that are joined to the rest by exactly one bridge."""
    nbr = neighbours(n, [e for e in edges if e not in set(bridges)])
    piece = {}
    for s in range(n):
        if s in piece:
            continue
        stack = [s]
        piece[s] = s
        while stack:
            for w in nbr[stack.pop()]:
                if w not in piece:
                    piece[w] = s
                    stack.append(w)
    attach = {}
    for a, b in bridges:
        attach[piece[a]] = attach.get(piece[a], 0) + 1
        attach[piece[b]] = attach.get(piece[b], 0) + 1
    return sum(1 for c in attach.values() if c == 1)
