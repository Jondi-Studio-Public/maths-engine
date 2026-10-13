"""Generates `uc2-tree-counting-prufer` questions (easy, medium, hard) and the exam chain
`exd_x_r4j_tree_counting_prufer`.

``uc2-tree-counting-prufer`` (``discrete-mathematics``, answer ``ordered_list``): moving between a labelled tree and its
Prüfer code. Every stem states the encoding rule (delete the leaf with the smallest label, write down its neighbour,
stop when two vertices remain); decoding is for the learner to work out.

* easy: the Prüfer code of a tree given by its edges (6 or 7 vertices).
* medium: from a code, rebuild the tree (7 or 8 vertices) and list the neighbours of one vertex in increasing order.
* hard: from a code on 9 or 10 vertices, give the unique path between two vertices, in order from the first to the
  second.
* exam (``exd_x_r4j_tree_counting_prufer``): from a code, (a) the leaves in increasing order, (b) the degree of a vertex,
  (c) the number of labelled trees on the same vertices in which a named vertex is a leaf, (d) the number in which it
  is not, which follows through from (c).

Every tree is built by decoding, then encoded again, and the code must come back. The tests add SymPy's own Prüfer
routines as a second route and count the trees of a small case by brute force.
"""

from __future__ import annotations

from math import comb

from ..answers import Distractor
from ..core import attempts, check, generator
from ._r4j_kit import keep_distinct, say

_TAGS = {
    "leaf": "prufer-records-leaf",
    "largest": "prufer-largest-leaf",
    "long": "prufer-code-too-long",
    "reversed": "prufer-code-reversed",
    "decode_largest": "prufer-decode-largest",
    "last_edge": "prufer-last-edge-missed",
    "code_only": "prufer-neighbours-from-code-positions-only",
    "path_reversed": "path-direction-reversed",
    "path_ends": "path-endpoints-omitted",
    "code_vertices": "prufer-leaves-are-code-entries",
    "leaf_missing": "prufer-leaf-left-out",
    "degree_count": "prufer-degree-is-occurrence-count",
    "degree_plus": "prufer-degree-overcounted",
    "all_trees": "prufer-all-trees-counted",
    "exponent": "prufer-code-length-miscounted",
    "complement": "prufer-complement-not-taken",
}

_RULE = (
    "The Prüfer code of a labelled tree is found by repeatedly deleting the leaf with the smallest label and writing "
    "down the label of the vertex it was joined to, until only two vertices are left."
)


# -- encoding and decoding -----------------------------------------------------------------------------------------


def decode(code, n, *, largest=False):
    """The edges (as sorted pairs) of the tree on 1..n whose Prüfer code is `code`. `largest` picks the largest free leaf
    each time instead of the smallest (a deliberately wrong rule)."""
    degree = {v: 1 for v in range(1, n + 1)}
    for c in code:
        degree[c] += 1
    edges = []
    for c in code:
        leaves = [v for v in sorted(degree) if degree[v] == 1]
        leaf = leaves[-1] if largest else leaves[0]
        edges.append(tuple(sorted((leaf, c))))
        degree[leaf] = 0
        degree[c] -= 1
    last = [v for v in sorted(degree) if degree[v] == 1]
    check(len(last) == 2, "decoding did not leave two vertices")
    edges.append(tuple(last))
    return sorted(edges)


def encode(edges, n, *, largest=False, record_leaf=False):
    """The Prüfer code of a tree. `largest` and `record_leaf` are deliberately wrong rules, for distractors."""
    adjacent = {v: set() for v in range(1, n + 1)}
    for a, b in edges:
        adjacent[a].add(b)
        adjacent[b].add(a)
    out = []
    for _ in range(n - 2):
        leaves = sorted(v for v in adjacent if len(adjacent[v]) == 1)
        leaf = leaves[-1] if largest else leaves[0]
        (nb,) = adjacent[leaf]
        out.append(leaf if record_leaf else nb)
        adjacent[nb].discard(leaf)
        del adjacent[leaf]
    return out


def _neighbours(edges, v):
    return sorted({b if a == v else a for a, b in edges if v in (a, b)})


def _path(edges, n, u, w):
    adjacent = {v: _neighbours(edges, v) for v in range(1, n + 1)}
    parent, frontier = {u: None}, [u]
    while frontier:
        nxt = []
        for x in frontier:
            for y in adjacent[x]:
                if y not in parent:
                    parent[y] = x
                    nxt.append(y)
        frontier = nxt
    out, x = [], w
    while x is not None:
        out.append(x)
        x = parent[x]
    return out[::-1]


def _random_code(rng, n, need_distinct):
    for _ in attempts(500, "a code with enough different entries"):
        code = [rng.randint(1, n) for _ in range(n - 2)]
        if len(set(code)) >= need_distinct:
            return code
    raise AssertionError  # pragma: no cover


def _tree(rng, n, need_distinct):
    code = _random_code(rng, n, need_distinct)
    edges = decode(code, n)
    check(encode(edges, n) == code, "encode(decode(code)) is not the code")
    check(len(edges) == n - 1, "a tree on n vertices has n - 1 edges")
    return code, edges


def _list_text(values):
    return ", ".join(str(v) for v in values)


def _code_text(code):
    return "(" + _list_text(code) + ")"


def _edges_text(edges):
    return ", ".join(f"{a}–{b}" for a, b in edges)


def _list_distractors(answer, items):
    return keep_distinct(
        answer,
        [(v, tag, _list_text(v)) for v, tag in items if v],
        key=tuple,
    )


def _single(text, latex, answer, distractors, working):
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=text,
        latex=latex,
        answer_kind="ordered_list",
        answer=list(answer),
        answer_text=_list_text(answer),
        working=working,
        distractors=distractors,
    )


# -- easy: tree to code -----------------------------------------------------------------------------------------------


def _easy(rng):
    n = rng.choice([6, 7])
    for _ in attempts(500, "a tree whose code has no repeated-entry shortcut"):
        code, edges = _tree(rng, n, 3)
        if encode(edges, n, largest=True) != code and encode(edges, n, record_leaf=True) != code:
            break
    form = rng.randrange(3)
    if form == 0:
        text, latex = say(
            f"A tree has vertices 1 to {n} and edges {_edges_text(edges)}. {_RULE} Write down the Prüfer code of the tree, "
            "in order."
        )
    elif form == 1:
        text, latex = say(
            f"{_RULE} Find the Prüfer code of the labelled tree on vertices 1 to {n} with edges {_edges_text(edges)}."
        )
    else:
        text, latex = say(
            f"The vertices 1 to {n} are joined by the edges {_edges_text(edges)} to make a tree. {_RULE} List its Prüfer code."
        )
    items = [
        (encode(edges, n, record_leaf=True), _TAGS["leaf"]),
        (encode(edges, n, largest=True), _TAGS["largest"]),
        (code + [n], _TAGS["long"]),
        (code[::-1], _TAGS["reversed"]),
    ]
    deg = {v: sum(v in e for e in edges) for v in range(1, n + 1)}
    working = [
        f"Degrees: " + ", ".join(f"{v}: {deg[v]}" for v in range(1, n + 1)) + ".",
        "Delete the smallest leaf each time and record its neighbour: the entries are " + _list_text(code) + ".",
        f"The code has {n - 2} entries, one fewer than the number of edges, because two vertices are left at the end.",
    ]
    return _single(text, latex, code, _list_distractors(code, items), working)


# -- medium: code to neighbours -------------------------------------------------------------------------------------


def _medium(rng):
    n = rng.choice([7, 8])
    for _ in attempts(500, "a vertex with several neighbours that the shortcuts get wrong"):
        code, edges = _tree(rng, n, 3)
        v = rng.choice(sorted(set(code)))
        answer = _neighbours(edges, v)
        wrong_big = _neighbours(decode(code, n, largest=True), v)
        if len(answer) >= 3 and wrong_big != answer:
            break
    form = rng.randrange(3)
    c = _code_text(code)
    if form == 0:
        text, latex = say(
            f"{_RULE} A labelled tree on vertices 1 to {n} has Prüfer code {c}. List the vertices adjacent to vertex {v} "
            "in increasing order."
        )
    elif form == 1:
        text, latex = say(
            f"The code {c} is the Prüfer code of a tree on vertices 1 to {n}. {_RULE} Which vertices are joined to vertex {v}? "
            "Give them in increasing order."
        )
    else:
        text, latex = say(
            f"{_RULE} Rebuild the tree on vertices 1 to {n} whose Prüfer code is {c}, and list the neighbours of vertex {v} "
            "in increasing order."
        )
    code_only = sorted({leaf for leaf, c2 in _pairs(code, n) if c2 == v})
    items = [
        (wrong_big, _TAGS["decode_largest"]),
        (
            _neighbours(edges[:-1] if edges[-1] not in _pairs(code, n) else _without_last(code, n), v),
            _TAGS["last_edge"],
        ),
        (code_only, _TAGS["code_only"]),
        (_neighbours(decode(code[::-1], n), v), _TAGS["reversed"]),
    ]
    working = [
        f"Occurrences of vertex {v} in the code: {code.count(v)}, so its degree is {code.count(v) + 1}.",
        "At each step the smallest vertex that is neither used up nor still in the rest of the code is joined to the next code entry; the last two vertices are then joined.",
        f"The edges are {_edges_text(edges)}, so the neighbours of {v} are {_list_text(answer)}.",
    ]
    return _single(text, latex, answer, _list_distractors(answer, items), working)


def _pairs(code, n):
    """(leaf, code entry) for each decoding step, in order, without the final edge."""
    degree = {v: 1 for v in range(1, n + 1)}
    for c in code:
        degree[c] += 1
    out = []
    for c in code:
        leaf = min(v for v in degree if degree[v] == 1)
        out.append((leaf, c))
        degree[leaf] = 0
        degree[c] -= 1
    return out


def _without_last(code, n):
    return [tuple(sorted(p)) for p in _pairs(code, n)]


# -- hard: code to path --------------------------------------------------------------------------------------------


def _hard(rng):
    n = rng.choice([9, 10])
    for _ in attempts(2000, "a long path that the shortcuts get wrong"):
        code, edges = _tree(rng, n, 4)
        u, w = rng.sample(range(1, n + 1), 2)
        path = _path(edges, n, u, w)
        wrong_big = _path(decode(code, n, largest=True), n, u, w)
        wrong_rev = _path(decode(code[::-1], n), n, u, w)
        if len(path) >= 5 and wrong_big != path and wrong_rev != path:
            break
    c = _code_text(code)
    form = rng.randrange(3)
    if form == 0:
        text, latex = say(
            f"{_RULE} A labelled tree on vertices 1 to {n} has Prüfer code {c}. List the vertices on the path from vertex {u} "
            f"to vertex {w}, in order, starting with {u}."
        )
    elif form == 1:
        text, latex = say(
            f"The Prüfer code of a tree on vertices 1 to {n} is {c}. {_RULE} Give the unique path from vertex {u} to vertex {w} "
            f"as a list of vertices, beginning at {u} and ending at {w}."
        )
    else:
        text, latex = say(
            f"{_RULE} Rebuild the tree with Prüfer code {c} on vertices 1 to {n}. In the tree, how do you get from vertex {u} to "
            f"vertex {w}? List every vertex you pass through, including {u} and {w}, in order."
        )
    items = [
        (wrong_big, _TAGS["decode_largest"]),
        (path[::-1], _TAGS["path_reversed"]),
        (path[1:-1], _TAGS["path_ends"]),
        (wrong_rev, _TAGS["reversed"]),
    ]
    working = [
        f"Decode the code: the edges are {_edges_text(edges)}.",
        f"There is exactly one path between two vertices of a tree: {_list_text(path)}.",
    ]
    return _single(text, latex, path, _list_distractors(path, items), working)


@generator(
    "r4j_tree_counting_prufer",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4j_tree_counting_prufer(rng, tier="easy"):
    """Prüfer code to tree and back; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ----------------------------------------------------------------------------------------------------------


def _trees_with_degree(n, d):
    """Labelled trees on n vertices in which a named vertex has degree d."""
    return comb(n - 2, d - 1) * (n - 1) ** (n - 1 - d)


@generator("exd_x_r4j_tree_counting_prufer", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4j_tree_counting_prufer(rng, tier="exam"):
    """Exam chain: leaves from a code, a degree, then two counts of labelled trees."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n = rng.choice([7, 8])
    code, edges = _tree(rng, n, 3)
    v = rng.choice([x for x in sorted(set(code)) if code.count(x) >= 1])
    d = code.count(v) + 1
    leaves = sorted(set(range(1, n + 1)) - set(code))
    check(
        leaves == [x for x in range(1, n + 1) if sum(x in e for e in edges) == 1],
        "leaves disagree with the decoded tree",
    )
    check(d == sum(v in e for e in edges), "degree disagrees with the decoded tree")
    check(sum(_trees_with_degree(n, k) for k in range(1, n)) == n ** (n - 2), "degree counts do not sum to n^(n-2)")
    leaf_count = _trees_with_degree(n, 1)
    check(leaf_count == (n - 1) ** (n - 2), "leaf count disagrees with (n - 1)^(n - 2)")
    c = _code_text(code)
    form = rng.randrange(3)
    if form == 0:
        stem, stem_tex = say(f"A labelled tree on vertices 1 to {n} has Prüfer code {c}.")
    elif form == 1:
        stem, stem_tex = say(f"The Prüfer code of a tree whose vertices are labelled 1 to {n} is {c}.")
    else:
        stem, stem_tex = say(
            f"Vertices 1 to {n} form a labelled tree. Its Prüfer code, found by deleting the smallest leaf each time, is {c}."
        )

    def part(prompt, kind, value, text, marks, items, working, follow=None, rng_=None):
        d_ = dict(
            prompt=prompt,
            answer_kind=kind,
            answer=value,
            answer_text=text,
            marks=marks,
            working=working,
            distractors=items,
        )
        if follow:
            d_["follow_through"] = follow
        if rng_:
            d_["sensible_range"] = rng_
        return d_

    def nums(answer, pairs):
        return keep_distinct(answer, [(x, tag, str(x)) for x, tag in pairs if x > 0])

    only_once = sorted(x for x in set(code) if code.count(x) == 1)
    parts = [
        part(
            "List the leaves of the tree in increasing order.",
            "ordered_list",
            leaves,
            _list_text(leaves),
            2,
            _list_distractors(
                leaves,
                [
                    (sorted(set(code)), _TAGS["code_vertices"]),
                    (leaves[:-1], _TAGS["leaf_missing"]),
                    (only_once, _TAGS["code_vertices"]),
                    (leaves[1:], _TAGS["leaf_missing"]),
                ],
            ),
            [
                f"A vertex is a leaf exactly when it never appears in the code. The vertices missing from {c} are {_list_text(leaves)}."
            ],
        ),
        part(
            f"What is the degree of vertex {v} in the tree?",
            "number",
            d,
            str(d),
            1,
            nums(d, [(d - 1, _TAGS["degree_count"]), (d + 1, _TAGS["degree_plus"]), (d + 2, _TAGS["degree_plus"])]),
            [f"Occurrences of vertex {v} in the code: {d - 1}, and degree = occurrences + 1 = {d}."],
            rng_=(1, None),
        ),
        part(
            f"How many labelled trees on vertices 1 to {n} have vertex {v} as a leaf?",
            "number",
            leaf_count,
            str(leaf_count),
            2,
            nums(
                leaf_count,
                [
                    (n ** (n - 2), _TAGS["all_trees"]),
                    ((n - 1) ** (n - 3), _TAGS["exponent"]),
                    ((n - 1) ** (n - 1), _TAGS["exponent"]),
                    (n ** (n - 3), _TAGS["exponent"]),
                ],
            ),
            [
                f"A leaf never appears in the code, so the {n - 2} entries are chosen from the other {n - 1} vertices: {(n - 1)}^{n - 2} = {leaf_count}."
            ],
            rng_=(0, None),
        ),
        part(
            f"How many labelled trees on vertices 1 to {n} do not have vertex {v} as a leaf?",
            "number",
            n ** (n - 2) - leaf_count,
            str(n ** (n - 2) - leaf_count),
            2,
            nums(
                n ** (n - 2) - leaf_count,
                [
                    (n ** (n - 2), _TAGS["all_trees"]),
                    (leaf_count, _TAGS["complement"]),
                    (n ** (n - 2) + leaf_count, _TAGS["complement"]),
                    (n ** (n - 3) - leaf_count if n ** (n - 3) > leaf_count else n ** (n - 3), _TAGS["exponent"]),
                ],
            ),
            [
                f"By Cayley's formula there are {n}^{n - 2} = {n ** (n - 2)} labelled trees, so {n ** (n - 2)} − {leaf_count} = {n ** (n - 2) - leaf_count}."
            ],
            follow=f"{n}**{n - 2} - part_c",
            rng_=(0, None),
        ),
    ]
    check(all(len(p["distractors"]) >= 2 for p in parts), "too few distractors in a part")
    return dict(text=stem, latex=stem_tex, parts=parts)
