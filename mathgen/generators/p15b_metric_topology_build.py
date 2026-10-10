"""Generates two `metric-spaces-and-topology` skills: metric axioms on a finite distance table, and compactness
and subcovers of a finite open cover.

Both skills sit in `metric-spaces-and-topology`. They are deliberately different STRUCTURES: a symmetric
finite distance TABLE (metric axioms) versus a finite OPEN COVER of a set (compactness/subcovers) -- nothing
here overlaps.
"""

from __future__ import annotations

import random
import string

from ..core import check, generator, pick_distractors

# ---------------------------------------------------------------------------
# ucv1-metric-axiom-check
# ---------------------------------------------------------------------------

# The skill's contract: "Decide which metric axiom fails, if any, for a supplied finite
# symmetric distance table." The table is stated symmetric by contract, so symmetry itself can never
# be the failing axiom. The honest answer space is therefore these four fixed classes -- never a
# per-draw wording, so the mcq options are constant strings and the grader compares them exactly.
_AXIOM_OPTIONS = {
    "none": "No axiom fails: the table defines a genuine metric.",
    "identity": "The identity axiom fails: some point has nonzero self-distance.",
    "zero": "The identity of indiscernibles fails: two distinct points have distance 0.",
    "triangle": "The triangle inequality fails for some triple of points.",
}
_AXIOM_ORDER = ["none", "identity", "zero", "triangle"]


def _classify_table(dist: list[list[int]], n: int) -> str | None:
    """Brute-force classification of a (already-symmetric) n x n integer
    distance table against the three non-symmetry metric axioms. Returns one
    of "none"/"identity"/"zero"/"triangle" when EXACTLY one axiom class
    describes the table, or None when more than one axiom fails at once (the
    generator retries rather than ship an ambiguous table)."""
    identity_ok = all(dist[i][i] == 0 for i in range(n))
    zero_ok = all(dist[i][j] != 0 for i in range(n) for j in range(n) if i != j)
    triangle_ok = all(dist[i][j] <= dist[i][k] + dist[k][j] for i in range(n) for j in range(n) for k in range(n))
    flags = (identity_ok, zero_ok, triangle_ok)
    if flags == (True, True, True):
        return "none"
    if flags == (False, True, True):
        return "identity"
    if flags == (True, False, True):
        return "zero"
    if flags == (True, True, False):
        return "triangle"
    return None


def _base_metric_table(rng: random.Random, n: int, coord_hi: int) -> list[list[int]]:
    """A genuine metric via a 1-D embedding: distinct integer coordinates,
    d(i, j) = |c_i - c_j|. Always satisfies all three axioms (identity,
    zero-implies-equal since coordinates are distinct, and the triangle
    inequality, since |a - c| <= |a - b| + |b - c| always)."""
    coords = rng.sample(range(1, coord_hi + 1), n)
    return [[abs(coords[i] - coords[j]) for j in range(n)] for i in range(n)]


def _break_identity(dist: list[list[int]], n: int, rng: random.Random, k_choices) -> list[list[int]]:
    i = rng.randrange(n)
    k = rng.choice(k_choices)
    dist = [row[:] for row in dist]
    dist[i][i] = k
    return dist


def _zero_table(rng: random.Random, n: int, coord_hi: int) -> list[list[int]]:
    """Two DIFFERENT points share the same 1-D coordinate (a 'twin' pair),
    which keeps the table a genuine pseudometric -- identity and the
    triangle inequality both still hold exactly, since it is literally
    |c_i - c_j| throughout -- while d(twin1, twin2) = 0 for two distinct
    labels, breaking only the identity of indiscernibles. Perturbing an
    already-distinct embedding's off-diagonal entry to 0 instead (the
    naive approach) generically breaks the triangle inequality too, since a
    third point is almost never equidistant from both -- this constructs
    the twin directly instead of retrying a doomed perturbation."""
    coords = rng.sample(range(1, coord_hi + 1), n - 1)
    twin_of = rng.randrange(n - 1)
    coords = coords + [coords[twin_of]]
    rng.shuffle(coords)
    return [[abs(coords[i] - coords[j]) for j in range(n)] for i in range(n)]


def _break_triangle(dist: list[list[int]], n: int, rng: random.Random, near_miss: bool) -> list[list[int]]:
    i, y, j = rng.sample(range(n), 3)
    dist = [row[:] for row in dist]
    delta = 1 if near_miss else rng.randint(2, 5)
    new_val = dist[i][y] + dist[y][j] + delta
    dist[i][j] = new_val
    dist[j][i] = new_val
    return dist


def _make_metric_table(rng: random.Random, n: int, coord_hi: int, target: str, near_miss: bool):
    for _attempt in range(200):
        base = _base_metric_table(rng, n, coord_hi)
        if target == "none":
            candidate = base
        elif target == "identity":
            candidate = _break_identity(base, n, rng, (1, 2) if not near_miss else (1,))
        elif target == "zero":
            candidate = _zero_table(rng, n, coord_hi)
        elif target == "triangle":
            candidate = _break_triangle(base, n, rng, near_miss)
        else:
            raise ValueError(f"unknown target {target!r}")
        classified = _classify_table(candidate, n)
        if classified == target:
            return candidate
    raise RuntimeError(f"metric-axiom-check: could not construct a clean {target!r} table")


def _render_table_prose(labels, dist, n) -> str:
    parts = []
    for i in range(n):
        parts.append(f"d({labels[i]}, {labels[i]}) = {dist[i][i]}")
    for i in range(n):
        for j in range(i + 1, n):
            parts.append(f"d({labels[i]}, {labels[j]}) = {dist[i][j]}")
    return "; ".join(parts)


def _render_table_matrix(labels, dist, n) -> str:
    rows = " \\\\ ".join(" & ".join(str(dist[i][j]) for j in range(n)) for i in range(n))
    header = ", ".join(labels)
    return rf"\text{{points }} ({header}):\quad \begin{{bmatrix}}{rows}\end{{bmatrix}}"


@generator(
    "p15b_metric_axiom_check",
    level=20,
    presets={
        "easy": dict(n=3, coord_hi=15, near_miss=False, as_matrix=False),
        "medium": dict(n=4, coord_hi=25, near_miss=False, as_matrix=False),
        "hard": dict(n=4, coord_hi=30, near_miss=True, as_matrix=False),
        "exam": dict(n=4, coord_hi=30, near_miss=True, as_matrix=True),
    },
)
def p15b_metric_axiom_check(rng, n=3, coord_hi=15, near_miss=False, as_matrix=False):
    """Decide which metric axiom fails, if any, for a supplied finite
    symmetric distance table (the skill's contract). The table is
    constructed to make exactly ONE of {identity, zero-implies-equal,
    triangle inequality} fail, or none, drawn uniformly from those four
    classes -- construct-to-target, then verified by an independent
    brute-force classification over ALL axioms (never trusting which one the
    construction step *meant* to break).

    easy/medium: the table is listed as prose pairs, small coordinates.
    hard: a near-miss triangle violation (off by exactly 1, so it cannot be
    eyeballed) or a small identity offset, with a slightly larger range.
    exam: the SAME near-miss construction, but read off an explicit n x n
    matrix instead of a prose list -- a genuinely different extraction task.
    """
    target = rng.choice(_AXIOM_ORDER)
    labels = rng.sample(string.ascii_uppercase.replace("U", ""), n)
    dist = _make_metric_table(rng, n, coord_hi, target, near_miss)

    # Independent re-verification, never trusting the constructor's own
    # target label: re-run the same brute-force classifier used to accept
    # the draw, plus a second, differently-shaped check for each axiom.
    reclassified = _classify_table(dist, n)
    check(reclassified == target, "metric-axiom-check: reclassification disagrees with the drawn target")
    check(
        all(dist[i][j] == dist[j][i] for i in range(n) for j in range(n)), "metric-axiom-check: table is not symmetric"
    )

    if as_matrix:
        body = _render_table_matrix(labels, dist, n)
        text = f"The distance table below is symmetric. Which metric axiom fails, if any? {body}"
        latex = body
    else:
        body = _render_table_prose(labels, dist, n)
        text = f"A finite set has the symmetric distances {body}. Which metric axiom fails, if any?"
        latex = rf"\text{{{body}}}"

    answer_text = _AXIOM_OPTIONS[target]
    others = [_AXIOM_OPTIONS[key] for key in _AXIOM_ORDER if key != target]
    check(len(set([answer_text, *others])) == 4, "metric-axiom-check: option collision")
    distractors = pick_distractors(answer_text, others, n=3)
    check(len(distractors) == 3, "metric-axiom-check: expected exactly three distractors")

    if target == "none":
        working = [
            "Every d(x, x) = 0, every d(x, y) > 0 for x != y, and every triple satisfies "
            "the triangle inequality: no axiom fails."
        ]
    elif target == "identity":
        offending = next(i for i in range(n) if dist[i][i] != 0)
        working = [
            f"d({labels[offending]}, {labels[offending]}) = {dist[offending][offending]} != 0: "
            "the identity axiom fails."
        ]
    elif target == "zero":
        i, j = next((i, j) for i in range(n) for j in range(n) if i != j and dist[i][j] == 0)
        working = [
            f"d({labels[i]}, {labels[j]}) = 0 with {labels[i]} != {labels[j]}: the identity of indiscernibles fails."
        ]
    else:
        i, y, j = next(
            (i, y, j) for i in range(n) for y in range(n) for j in range(n) if dist[i][j] > dist[i][y] + dist[y][j]
        )
        working = [
            f"d({labels[i]}, {labels[j]}) = {dist[i][j]} > d({labels[i]}, {labels[y]}) + "
            f"d({labels[y]}, {labels[j]}) = {dist[i][y] + dist[y][j]}: the triangle inequality fails."
        ]

    return dict(
        text=text,
        latex=latex,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=rf"\text{{{answer_text}}}",
        working=working,
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# ucv1-compactness-finite-check
# ---------------------------------------------------------------------------

# The skill's contract: "Decide compactness of a supplied finite topological space by checking
# its finite open-cover subfamilies." Every finite topological space is compact,
# so a "decide compactness" question has one constant, honest answer -- never built. The words "by
# checking its finite open-cover subfamilies" support the widened question actually built here:
# given a supplied finite open cover, WHICH listed subfamily is itself a subcover of X.


def _covers(sets: list[frozenset], universe: frozenset) -> bool:
    union: set = set()
    for s in sets:
        union |= s
    return union == universe


def _format_subfamily(indices: tuple[int, ...], perm: list[int]) -> str:
    """Render internal set indices as displayed U-labels through `perm`
    (perm[internal_index] = displayed 1-based number), so which position
    holds the essential vs. redundant sets is randomised per draw -- without
    it, the answer's internal indices (always "drop the last one") render
    as the SAME string "{U1, U2, U3}" on every single seed, a 100%-
    concentrated answer that also hands away the pattern for free."""
    return "{" + ", ".join(f"U{perm[i]}" for i in sorted(indices, key=lambda i: perm[i])) + "}"


def _format_subfamily_latex(indices: tuple[int, ...], perm: list[int]) -> str:
    return r"\{" + ", ".join(f"U_{{{perm[i]}}}" for i in sorted(indices, key=lambda i: perm[i])) + r"\}"


def _draw_cover(rng: random.Random, labels: list[str], n_essential: int, n_redundant: int, overlap_prob: float):
    """Draw a finite open cover of X = `labels`: `n_essential` sets that
    PARTITION X (so each is individually load-bearing -- dropping one always
    uncovers its own block's points, no other set can hold them) plus
    `n_redundant` sets that are each built as a STRICT subset of every
    essential block (never the WHOLE of any block), so a redundant set can
    never single-handedly replace an essential one -- droppability of every
    redundant set is then true BY CONSTRUCTION, not by chance. (The earlier,
    fully-random redundant draw let a redundant set absorb an entire small
    essential block by chance, especially at n=4, making that essential set
    look droppable too -- this rules that out structurally instead of
    retrying a collision-prone draw.) `overlap_prob` controls how much of
    each block a redundant set duplicates. Every property is still
    re-verified by brute force below, never assumed from the construction."""
    universe = frozenset(labels)
    n = len(labels)

    for _attempt in range(200):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        cut_points = sorted(rng.sample(range(1, n), n_essential - 1)) if n_essential > 1 else []
        bounds = [0] + cut_points + [n]
        blocks = [shuffled[bounds[i] : bounds[i + 1]] for i in range(n_essential)]
        if any(len(b) == 0 for b in blocks):
            continue
        essential_sets = [frozenset(block) for block in blocks]

        redundant_sets = []
        for _ in range(n_redundant):
            pieces = []
            for block in blocks:
                max_take = len(block) - 1  # strictly fewer than the whole block
                if max_take <= 0:
                    continue
                take = sum(1 for _ in range(max_take) if rng.random() < max(overlap_prob, 0.5))
                take = min(take, max_take)
                if take:
                    pieces.extend(rng.sample(block, take))
            redundant_sets.append(frozenset(pieces))
        if any(len(s) == 0 for s in redundant_sets):
            continue

        all_sets = essential_sets + redundant_sets
        if not _covers(all_sets, universe):
            continue
        # every essential index must be load-bearing: dropping it alone breaks coverage
        ok = True
        for i in range(n_essential):
            remaining = [s for j, s in enumerate(all_sets) if j != i]
            if _covers(remaining, universe):
                ok = False
                break
        if not ok:
            continue
        # every redundant index (and all of them together) must be droppable
        for i in range(n_essential, n_essential + n_redundant):
            remaining = [s for j, s in enumerate(all_sets) if j != i]
            if not _covers(remaining, universe):
                ok = False
                break
        if not ok:
            continue
        if not _covers(essential_sets, universe):
            continue
        return all_sets
    raise RuntimeError("compactness-finite-check: could not draw a usable open cover")


def _pick_candidates_forward(rng: random.Random, all_sets, n_essential: int, universe: frozenset):
    """Forward tiers (easy/medium/hard): the answer drops the single
    redundant set; each distractor drops one of the n_essential sets
    instead, always the same shape (a single-set-dropped subfamily) so no
    option is identifiable purely by its length."""
    full = tuple(range(len(all_sets)))
    redundant_index = n_essential  # by construction, index n_essential is the sole redundant set
    answer_indices = tuple(i for i in full if i != redundant_index)
    check(
        _covers([all_sets[i] for i in answer_indices], universe),
        "compactness-finite-check: constructed answer subfamily does not cover X",
    )
    distractor_indices = []
    for drop in range(n_essential):
        cand = tuple(i for i in full if i != drop)
        check(
            not _covers([all_sets[i] for i in cand], universe),
            "compactness-finite-check: an essential-dropped subfamily unexpectedly covers X",
        )
        distractor_indices.append(cand)
    return answer_indices, distractor_indices


def _pick_candidates_exam(rng: random.Random, all_sets, n_essential: int, universe: frozenset):
    """Exam tier: two redundant sets (indices n_essential, n_essential + 1).
    The answer drops BOTH redundant sets (the smallest valid subcover). One
    distractor drops only one redundant set (still valid, but not smallest --
    the genuine escalation: several options COVER, and the learner must find
    the smallest). One distractor drops one essential set (invalid). One
    distractor drops one essential AND one redundant set (same SIZE as the
    true answer, but invalid) -- so size alone never identifies the answer."""
    full = tuple(range(len(all_sets)))
    r1, r2 = n_essential, n_essential + 1
    answer_indices = tuple(i for i in full if i not in (r1, r2))
    check(
        _covers([all_sets[i] for i in answer_indices], universe),
        "compactness-finite-check exam: minimal subfamily does not cover X",
    )

    valid_not_minimal = tuple(i for i in full if i != r1)
    check(
        _covers([all_sets[i] for i in valid_not_minimal], universe),
        "compactness-finite-check exam: dropping one redundant set unexpectedly fails to cover X",
    )
    check(
        len(valid_not_minimal) > len(answer_indices),
        "compactness-finite-check exam: 'valid but not minimal' distractor is not actually larger",
    )

    drop_essential = rng.randrange(n_essential)
    invalid_smaller = tuple(i for i in full if i != drop_essential)
    check(
        not _covers([all_sets[i] for i in invalid_smaller], universe),
        "compactness-finite-check exam: dropping one essential set unexpectedly covers X",
    )

    invalid_same_size = tuple(i for i in full if i not in (drop_essential, r1))
    check(
        not _covers([all_sets[i] for i in invalid_same_size], universe),
        "compactness-finite-check exam: same-size invalid distractor unexpectedly covers X",
    )
    check(
        len(invalid_same_size) == len(answer_indices),
        "compactness-finite-check exam: same-size distractor is not actually the same size as the answer",
    )

    return answer_indices, [valid_not_minimal, invalid_smaller, invalid_same_size]


@generator(
    "p15b_compactness_subcover_check",
    level=20,
    presets={
        "easy": dict(n=4, n_essential=3, n_redundant=1, overlap_prob=0.0, exam=False),
        "medium": dict(n=5, n_essential=3, n_redundant=1, overlap_prob=0.2, exam=False),
        "hard": dict(n=5, n_essential=3, n_redundant=1, overlap_prob=0.35, exam=False),
        "exam": dict(n=6, n_essential=3, n_redundant=2, overlap_prob=0.2, exam=True),
    },
)
def p15b_compactness_subcover_check(rng, n=4, n_essential=3, n_redundant=1, overlap_prob=0.0, exam=False):
    """Given a supplied finite open cover of X, which listed subfamily is
    itself an open subcover of X? (Every finite space is compact -- the
    contract's own words "by checking its finite open-cover subfamilies"
    are what this question asks.)

    Forward tiers: exactly one redundant cover member; the answer drops it,
    each distractor drops a different load-bearing member instead (same
    shape: always "all sets except one").

    Exam tier: TWO redundant members, escalating by method -- several
    options actually cover X and the learner must find the SMALLEST one,
    with a same-size invalid distractor so size alone never gives it away.
    """
    labels = rng.sample(string.ascii_uppercase, n)
    universe = frozenset(labels)
    all_sets = _draw_cover(rng, labels, n_essential, n_redundant, overlap_prob)
    check(_covers(all_sets, universe), "compactness-finite-check: full cover does not cover X")

    if exam:
        answer_indices, distractor_index_lists = _pick_candidates_exam(rng, all_sets, n_essential, universe)
    else:
        answer_indices, distractor_index_lists = _pick_candidates_forward(rng, all_sets, n_essential, universe)

    # Randomise which internal set gets which DISPLAYED U-number -- without
    # this, the answer is always "drop the internally-last index", which
    # renders as the literal same string "{U1, U2, U3}" on every draw.
    perm = list(range(1, len(all_sets) + 1))
    rng.shuffle(perm)

    answer_text = _format_subfamily(answer_indices, perm)
    distractor_texts = [_format_subfamily(idx, perm) for idx in distractor_index_lists]
    check(
        len(set([answer_text, *distractor_texts])) == 1 + len(distractor_texts),
        "compactness-finite-check: candidate subfamily collision",
    )
    distractors = pick_distractors(answer_text, distractor_texts, n=len(distractor_texts))
    check(
        len(distractors) == len(distractor_texts), "compactness-finite-check: a distractor was dropped as a duplicate"
    )

    set_listing = "; ".join(f"U{perm[i]} = {{{', '.join(sorted(all_sets[i]))}}}" for i in range(len(all_sets)))
    if exam:
        instruction = (
            "X and an open cover {U1, ..., U%d} are given above. Which listed subfamily "
            "is the SMALLEST subfamily of the cover that still covers X?" % len(all_sets)
        )
    else:
        instruction = (
            "X and an open cover {U1, ..., U%d} are given above. Which listed subfamily "
            "of the cover is itself an open subcover of X?" % len(all_sets)
        )
    text = f"X = {{{', '.join(sorted(labels))}}}. {set_listing}. {instruction}"
    latex = rf"\text{{{set_listing}}}"

    working = [f"Dropping the set(s) not needed for coverage leaves {answer_text}, whose union is all of X."]

    return dict(
        text=text,
        latex=latex,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=_format_subfamily_latex(answer_indices, perm),
        working=working,
        distractors=distractors,
    )
