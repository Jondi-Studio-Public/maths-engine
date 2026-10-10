"""Generates `p14b_tree_edge_count` questions: the number of edges of a tree, using the fact that a tree on n
vertices has n - 1 edges.

Ucv1-tree-spanning-edge-count.

Kernel: a tree on n vertices has exactly n - 1 edges. That single subtraction
is the WHOLE mathematical content -- the escalation across tiers is entirely
in what has to be worked out FIRST to get n (or the edge count), never a
bigger n on its own:

- easy: n is stated directly.
- medium: n must be derived by reading a full stated DEGREE SEQUENCE (one
  degree per vertex) -- the direct route counts the list to get n, the
  handshake-lemma route sums the list and halves it to get n - 1 (edges)
  straight away, since for a tree sum(degrees) = 2(n - 1).
- hard: n must be derived from a DESCRIBED STRUCTURE (a full binary tree --
  every node is a leaf or has exactly two children -- given its number of
  LEAVES L), via the vertex-count fact n = 2L - 1 -- a genuinely different
  first step (recalling/deriving a structural formula) from medium's
degree-sequence sum.
  complete-binary-tree family, depth 2..9, produced only 8 distinct answers
  over 200 seeds -- memorisable. The leaf-count family is linear in L rather
  than exponential in depth, so a human-scale range of L produces dozens of
  distinct answers instead of 8; see the generator's leaf-count range below.)
- exam: hard's structural derivation PLUS a genuine connected-vs-not-a-tree
  comparison -- the described network has SOME extra connections beyond the
  full binary tree (never a stated count -- see below) and a stated TOTAL
  edge count, and the learner must recognise n - 1 as a BASELINE to compare
  the stated total edge count against, not something to apply directly.
  count literally in the prompt -- "8 extra direct connections... giving 38
  edges in total" then asked for the extra count, so the answer was already
  printed. The prompt now states only the structural description (leaf
  count L) and the TOTAL edge count; the learner must derive n = 2L - 1,
  then n - 1, then subtract from the stated total to get the extra count.)

Never a weighted-graph optimisation shape (that is
`senior-use-minimal-spanning-trees-to-solve-minimal-connector`'s job, an
entirely different skill: it chooses which edges of a WEIGHTED graph to keep
to minimise total cost; this skill never has edge weights and never
chooses anything -- it only counts).
"""

from __future__ import annotations

from ..core import check, generator, phrase, pick_distractors


# --- medium: a random tree's degree sequence -----------------------------


def _random_tree_degrees(rng, n: int) -> list[int]:
    """Build a random labelled tree on n vertices (vertex i > 0 attaches to
    a uniformly random earlier vertex) and return its degree sequence. This
    always yields a genuinely realizable tree, so sum(degrees) == 2*(n-1)
    and every degree >= 1 by construction -- never an impossible sequence
    (e.g. every degree 1 for n > 2)."""
    degree = [0] * n
    for i in range(1, n):
        parent = rng.randint(0, i - 1)
        degree[i] += 1
        degree[parent] += 1
    return degree


# --- text templates --------------------------------------------------------

_EASY_TEXT = (
    "A tree has {n} vertices. How many edges does it have?",
    "A connected, acyclic graph (a tree) has {n} vertices in total. How many edges does the tree have?",
    "Consider a tree with {n} vertices. What is the number of edges in the tree?",
)

_MEDIUM_TEXT = (
    "A tree's vertices have the following degrees: {degrees}. How many edges does the tree have?",
    "In a tree, every vertex's degree is listed here: {degrees}. How many edges does the tree have?",
    "A tree has one vertex for each degree listed: {degrees}. How many edges does it have?",
)

_HARD_TEXT = (
    "A full binary tree has {L} leaves (every node is either a leaf, or an internal node with exactly two children). How many edges does the tree have?",
    "Consider a full binary tree with {L} leaves: every node is a leaf or has exactly two children. How many edges does this tree have?",
    "A full binary tree contains {L} leaves in total (every internal node has exactly two children). Find the number of edges in the tree.",
    "In a full binary tree with {L} leaves (every internal node has exactly two children), how many edges connect its vertices?",
)

_EXAM_TEXT = (
    "A network starts as a full binary tree with {L} leaves (every internal node has exactly two children), then some extra direct connections are added between existing nodes, giving {m} edges in total. How many of the network's edges are extra, beyond what a spanning tree of the network would need?",
    "Take a full binary tree with {L} leaves (every internal node has exactly two children) and add some extra direct connections between existing nodes, so the network now has {m} edges altogether. How many edges are extra, beyond what a spanning tree would require?",
    "A full binary tree with {L} leaves (every internal node has exactly two children) has some extra direct connections added between existing nodes, for {m} edges in total. Determine how many of these edges are extra beyond a spanning tree.",
)


@generator(
    "p14b_tree_edge_count",
    level=16,
    presets={
        "easy": dict(variant="stated_n", n_lo=5, n_hi=40),
        "medium": dict(variant="degree_sequence", n_lo=5, n_hi=14),
        "hard": dict(variant="described_structure", L_lo=10, L_hi=50),
        "exam": dict(variant="extra_edges", L_lo=10, L_hi=50, extra_lo=3, extra_hi=9),
    },
)
def tree_edge_count(rng, variant="stated_n", n_lo=5, n_hi=40, L_lo=10, L_hi=50, extra_lo=3, extra_hi=9):
    if variant == "stated_n":
        return _stated_n(rng, n_lo, n_hi)
    if variant == "degree_sequence":
        return _degree_sequence(rng, n_lo, n_hi)
    if variant == "described_structure":
        return _described_structure(rng, L_lo, L_hi)
    if variant == "extra_edges":
        return _extra_edges(rng, L_lo, L_hi, extra_lo, extra_hi)
    raise ValueError(f"unknown variant {variant!r}")


def _stated_n(rng, n_lo, n_hi):
    n = rng.randint(n_lo, n_hi)
    edges = n - 1

    # Independent route: re-derive n from the answer (edges + 1 == n), the
    # opposite arithmetic direction from construction's own n - 1.
    check(edges + 1 == n, "stated_n: edges + 1 does not recover n")

    uses_n = str(n)
    uses_n_plus_1 = str(n + 1)
    check(edges != n and edges != n + 1, "stated_n: answer collides with a misconception")
    cands = pick_distractors(str(edges), [uses_n, uses_n_plus_1])

    text = phrase(rng, _EASY_TEXT, n=n)
    return dict(
        text=text,
        latex=text,
        answer=edges,
        answer_text=str(edges),
        answer_latex=str(edges),
        working=[f"A tree on n = {n} vertices has edges = n - 1.", f"edges = {n} - 1 = {edges}"],
        distractors=cands,
    )


def _degree_sequence(rng, n_lo, n_hi):
    n = rng.randint(n_lo, n_hi)
    degrees = _random_tree_degrees(rng, n)
    rng.shuffle(degrees)
    total = sum(degrees)
    edges = n - 1

    # Independent route: re-sum the STATED degree list and halve it
    # (handshake lemma), never reusing construction's own len(degrees) - 1
    # arithmetic.
    check(total == 2 * (n - 1), "degree_sequence: sum(degrees) != 2(n-1)")
    check(total % 2 == 0 and total // 2 == edges, "degree_sequence: sum(degrees)/2 does not match n - 1")
    check(all(d >= 1 for d in degrees), "degree_sequence: a degree below 1 cannot belong to a tree")

    uses_n = str(n)  # forgets to subtract 1 after counting the list
    uses_n_plus_1 = str(n + 1)
    uses_raw_sum = str(total)  # forgot to divide the degree sum by 2
    check_candidates = [uses_n, uses_n_plus_1, uses_raw_sum]
    for c in check_candidates:
        check(c != str(edges), "degree_sequence: a distractor equals the answer")
    cands = pick_distractors(str(edges), check_candidates)

    degrees_text = ", ".join(str(d) for d in degrees)
    text = phrase(rng, _MEDIUM_TEXT, degrees=degrees_text)
    return dict(
        text=text,
        latex=text,
        answer=edges,
        answer_text=str(edges),
        answer_latex=str(edges),
        working=[
            f"{n} degrees are listed, so the tree has n = {n} vertices.",
            f"Sum of degrees = {total} = 2(n - 1), so edges = {total} / 2 = {edges}.",
        ],
        distractors=cands,
    )


def _vertex_count_by_leaves(L: int) -> int:
    """Independent route for a full binary tree's vertex count: every
    internal node has exactly two children, so a full binary tree with L
    leaves has exactly L - 1 internal nodes (a standard, separately-derived
    fact -- summing internal and leaf counts directly), rather than
    construction's own closed-form n = 2*L - 1 evaluation."""
    internal_nodes = L - 1
    return L + internal_nodes


def _described_structure(rng, L_lo, L_hi):
    # A full binary tree's vertex count is LINEAR in its leaf count (n = 2L - 1), so a human-scale
    # range of L produces dozens of distinct answers instead of 8.
    L = rng.randint(L_lo, L_hi)
    n = 2 * L - 1
    edges = n - 1

    check(
        _vertex_count_by_leaves(L) == n,
        "described_structure: leaf/internal-node vertex count disagrees with the closed form",
    )
    check(edges + 1 == n, "described_structure: edges + 1 does not recover n")

    uses_n = str(n)
    uses_n_plus_1 = str(n + 1)
    uses_internal_nodes_only = str(L - 1)  # mistakes the internal-node count for the edge count
    check_candidates = [uses_n, uses_n_plus_1, uses_internal_nodes_only]
    for c in check_candidates:
        check(c != str(edges), "described_structure: a distractor equals the answer")
    cands = pick_distractors(str(edges), check_candidates)

    text = phrase(rng, _HARD_TEXT, L=L)
    return dict(
        text=text,
        latex=text,
        answer=edges,
        answer_text=str(edges),
        answer_latex=str(edges),
        working=[
            f"A full binary tree with L = {L} leaves has n = 2L - 1 = {n} vertices.",
            f"edges = n - 1 = {n} - 1 = {edges}",
        ],
        distractors=cands,
    )


def _extra_edges(rng, L_lo, L_hi, extra_lo, extra_hi):
    # The prompt now states only the structural description (leaf count L) and the TOTAL edge count
    # m; it never states `extra` anywhere in question.text. The learner must derive n = 2L - 1, then
    # n - 1, then subtract from m.
    L = rng.randint(L_lo, L_hi)
    n = 2 * L - 1
    extra = rng.randint(extra_lo, extra_hi)
    m = (n - 1) + extra

    check(
        _vertex_count_by_leaves(L) == n, "extra_edges: leaf/internal-node vertex count disagrees with the closed form"
    )
    check(m - (n - 1) == extra, "extra_edges: m - (n - 1) does not recover extra")
    # The only numbers printed in question.text are L and m. Enforce by
    # construction (every seed, unconditionally) that neither can ever
    # collide with `extra` as a literal token -- L_hi/L_lo are set well
    # above extra_hi, and m is always much larger than extra since
    # m = (2L - 2) + extra with L >= L_lo, so this must hold for any preset
    # respecting that ordering; check it explicitly rather than trust it.
    check(extra != L, "extra_edges: extra collides with the stated leaf count L (answer leak)")
    check(extra != m, "extra_edges: extra collides with the stated total edge count m (answer leak)")

    reports_given_edges = str(m)  # the exam-specific misconception
    off_by_one_low = str(m - n)  # compares against n instead of n - 1
    off_by_one_high = str(m - (n + 1))  # compares against n + 1 instead of n - 1
    check_candidates = [reports_given_edges, off_by_one_low, off_by_one_high]
    for c in check_candidates:
        check(c != str(extra), "extra_edges: a distractor equals the answer")
    cands = pick_distractors(str(extra), check_candidates)

    text = phrase(rng, _EXAM_TEXT, L=L, m=m)
    return dict(
        text=text,
        latex=text,
        answer=extra,
        answer_text=str(extra),
        answer_latex=str(extra),
        working=[
            f"A full binary tree with L = {L} leaves has n = 2L - 1 = {n} vertices, so a spanning tree needs n - 1 = {n - 1} edges.",
            f"The network has m = {m} edges in total, so the extra edges beyond a spanning tree are {m} - {n - 1} = {extra}.",
        ],
        distractors=cands,
    )
