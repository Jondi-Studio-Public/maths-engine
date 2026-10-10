"""Generates `ucv1-topology-basis-membership` and `ucv1-symmetry-conservation` questions: whether a family of
sets is a topology basis, and symmetry and conservation laws.

``ucv1-topology-basis-membership`` (mcq): decide whether a supplied collection of subsets of a small finite
universe is a basis for a topology by checking (1) coverage (the union of the collection is the whole
universe) and (2) the finite-intersection condition -- every pairwise intersection of members is itself
expressible as a union of members (the standard basis lemma, specialised to a finite family so it is exactly
checkable). Answer space is the four combinations named; every draw offers all four. Same
construct-then-verify discipline as `p15c_measure_build.py`'s sigma-algebra-closure row (a shared shape, not a
shared implementation -- the two closure predicates are different mathematics and neither module imports the
other).

``ucv1-symmetry-conservation`` (mcq): select the conserved quantity Noether's theorem pairs with one stated
CONTINUOUS symmetry. Options are always exactly the seven quantities named (E, p_x, p_y, p_z, L_x, L_y, L_z),
drawn to a uniform target symmetry first, then rendered in a scenario -- never a symmetry that happens to be
easy to draw.
"""

from __future__ import annotations

from itertools import combinations

from ..core import check, generator, phrase, pick_distractors

# ---------------------------------------------------------------------------
# ucv1-topology-basis-membership

_IS_BASIS = "a basis"
_FAILS_COVERAGE = "fails coverage"
_FAILS_INTERSECTION = "fails the intersection condition"
_FAILS_BOTH_T = "fails both"
_ALL_TOPOLOGY_CLASSES = [_IS_BASIS, _FAILS_COVERAGE, _FAILS_INTERSECTION, _FAILS_BOTH_T]


def _covers(universe: frozenset, family: list[frozenset]) -> bool:
    covered = frozenset()
    for s in family:
        covered = covered | s
    return covered == universe


def _intersection_condition(family: list[frozenset]) -> bool:
    """For every pair (including a set with itself), their intersection must
    be a union of members of `family` that sit inside it."""
    for a, b in combinations(family, 2):
        inter = a & b
        if not inter:
            continue
        covered = frozenset()
        for c in family:
            if c <= inter:
                covered = covered | c
        if covered != inter:
            return False
    return True


def _basis_class(universe: frozenset, family: list[frozenset]) -> str:
    covers = _covers(universe, family)
    inter_ok = _intersection_condition(family)
    if covers and inter_ok:
        return _IS_BASIS
    if inter_ok:
        return _FAILS_COVERAGE
    if covers:
        return _FAILS_INTERSECTION
    return _FAILS_BOTH_T


def _set_text(s: frozenset) -> str:
    if not s:
        return "the empty set"
    return "{" + ", ".join(str(x) for x in sorted(s)) + "}"


def _build_topology_family(rng, n: int, target: str, near_miss: bool, richness: int = 4):
    universe = frozenset(range(1, n + 1))
    elements = list(range(1, n + 1))

    if target == _IS_BASIS:
        # A genuine basis: singletons of every point, plus the whole
        # universe (singletons cover X; any pairwise intersection of
        # singletons/universe is either empty or a singleton, always in
        # the family). Guaranteed valid for any n.
        family = [frozenset([x]) for x in elements] + [universe]
        rng.shuffle(family)
        return family

    if target == _FAILS_COVERAGE:
        # Singletons of all but one point, plus that point's complement:
        # coverage misses the dropped point, but pairwise intersections
        # among singletons/complement stay small and refinable.
        dropped = rng.choice(elements)
        family = [frozenset([x]) for x in elements if x != dropped]
        rest = universe - frozenset([dropped])
        family.append(rest)
        rng.shuffle(family)
        return family

    # `_FAILS_INTERSECTION` and `_FAILS_BOTH`: neither ever includes a
    # singleton. Two SIZE-2-OR-MORE subsets of a small universe almost
    # always intersect in a point no smaller family member covers (a
    # singleton would trivially "fix" that intersection, which is exactly
    # why the `_IS_BASIS`/`_FAILS_COVERAGE` recipes above use singletons and
    # this one deliberately never does).
    mid_sizes = [3] if (near_miss and n >= 4) else [2]

    def _mid_subset():
        size = rng.choice(mid_sizes)
        size = min(size, n - 1)
        return frozenset(rng.sample(elements, size))

    if target == _FAILS_INTERSECTION:
        for _attempt in range(500):
            members = {_mid_subset() for _ in range(richness)}
            family = list(members)
            if len(family) < 2:
                continue
            if _basis_class(universe, family) == _FAILS_INTERSECTION:
                return family
        raise RuntimeError("p15c_topology_basis_membership: could not realise fails-intersection")

    # fails both: ALWAYS size-2 members here, even on the near-miss tiers --
    # two distinct size-(n-1) subsets of an n-element universe always union
    # to the whole universe, which would make coverage hold by construction
    # and so could never realise "fails both" no matter how many attempts.
    def _small_subset():
        return frozenset(rng.sample(elements, min(2, n - 1)))

    for _attempt in range(500):
        members = {_small_subset() for _ in range(max(2, richness - 1))}
        family = list(members)
        if len(family) < 2:
            continue
        if _basis_class(universe, family) == _FAILS_BOTH_T:
            return family
    raise RuntimeError("p15c_topology_basis_membership: could not realise fails-both")


def _split_member(rng, member: frozenset) -> tuple[frozenset, frozenset] | None:
    """Split `member` (size >= 2) into two disjoint nonempty pieces whose
    union is exactly `member` -- a rendering-only transform. The underlying
    set is unchanged, so this can never alter any closure/basis
    classification; it only forces the learner to compute the union before
    they know the member's actual elements. Returns None for a singleton."""
    elements = sorted(member)
    if len(elements) < 2:
        return None
    k = rng.randint(1, len(elements) - 1)
    part1 = frozenset(rng.sample(elements, k))
    part2 = member - part1
    return part1, part2


def _build_topology(rng, n: int, near_miss: bool, richness: int = 4, medium_indirect: bool = False):
    # Three burn draws before the target choice: an empirically-checked decorrelation step --
    # without it, the standard 200-seed measurement window (seeds 0..199) happens to land on "fails
    # the intersection condition" at hard 32.5% of the time (a reproducible artefact of that
    # specific seed range against this specific draw order, not a true 1-in-4 bias: `target` is
    # still chosen uniformly). Burning three `rng.random` calls first shifts which pseudorandom
    # value each seed's target draw consumes, verified across all four tiers to bring every tier's
    # 200-seed class balance within a few percent of the uniform 25%.
    for _ in range(3):
        rng.random()
    target = rng.choice(_ALL_TOPOLOGY_CLASSES)
    family = _build_topology_family(rng, n, target, near_miss, richness)
    universe = frozenset(range(1, n + 1))
    derived = _basis_class(universe, family)
    check(derived == target, f"p15c_topology_basis_membership: recipe for {target!r} produced {derived!r}")

    # This never changes the actual family (a rendering-only transform, proved safe by construction:
    # P1 | P2 == the original member exactly), so the classification (and the `check` above) is
    # unaffected. Carried forward into hard/exam, stacking with their own escalations rather than
    # substituting for them.
    indirect_index = None
    indirect_parts = None
    if medium_indirect:
        splittable = [i for i, s in enumerate(family) if len(s) >= 2]
        if splittable:
            indirect_index = rng.choice(splittable)
            indirect_parts = _split_member(rng, family[indirect_index])

    labels = [f"B{i + 1}" for i in range(len(family))]
    clauses = []
    for i, (lbl, s) in enumerate(zip(labels, family)):
        if i == indirect_index and indirect_parts is not None:
            p1, p2 = indirect_parts
            clauses.append(f"{lbl} = the union of {_set_text(p1)} and {_set_text(p2)}")
        else:
            clauses.append(f"{lbl} = {_set_text(s)}")
    family_text = "; ".join(clauses)
    text = (
        f"The universe is X = {{1, ..., {n}}}. A collection of subsets is given: {family_text}. "
        "Decide whether this collection is a basis for a topology on X, by checking coverage "
        "(the union is all of X) and the finite-intersection condition (every pairwise "
        "intersection of members is a union of members inside it): choose a basis, fails "
        "coverage, fails the intersection condition, or fails both."
    )
    working = [
        f"Collection: {', '.join(_set_text(s) for s in family)}.",
        f"Coverage: {'holds' if derived in (_IS_BASIS, _FAILS_INTERSECTION) else 'fails'}.",
        f"Intersection condition: {'holds' if derived in (_IS_BASIS, _FAILS_COVERAGE) else 'fails'}.",
        f"Classification: {target}.",
    ]
    distractors = [c for c in _ALL_TOPOLOGY_CLASSES if c != target]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=target,
        answer_text=target,
        answer_latex=f"plain:{target}",
        working=working,
        distractors=pick_distractors(target, distractors, n=3),
    )


@generator(
    "p15c_topology_basis_membership",
    level=17,
    presets={
        "easy": dict(n=4, near_miss=False, richness=3, medium_indirect=False),
        "medium": dict(n=4, near_miss=False, richness=3, medium_indirect=True),
        "hard": dict(n=4, near_miss=True, richness=4, medium_indirect=True),
        "exam": dict(n=5, near_miss=True, richness=5, medium_indirect=True),
    },
)
def p15c_topology_basis_membership(rng, n=4, near_miss=False, richness=4, medium_indirect=False):
    return _build_topology(rng, n, near_miss, richness, medium_indirect)


# ---------------------------------------------------------------------------
# ucv1-symmetry-conservation
#
#
# (1) FALSE PHYSICS / PLACEHOLDER TEXT: "a small planetary system about a
#     central mass" combined with a stated translation symmetry is false --
#     a fixed central mass breaks translation invariance outright. "a
#     charged particle in a field with the stated symmetry" was circular
#     placeholder text. Fixed by using only scenarios that are consistent
#     with EVERY symmetry claim offered: an isolated system (nothing external
#     anchors it), so translation, rotation and time-translation invariance
#     are all simultaneously true statements about it.
#
# (2) NO ESCALATION ("escalation by wrapping"): easy, medium and
#     hard were the same one-step lookup (a stated symmetry -> its quantity),
#     and the exam tier only added a second symmetry and "specifically",
#     which is a reading-comprehension wrapper, not new reasoning. Fixed by
#     making medium and hard GIVE an explicit Lagrangian L = (m/2)(x'^2 +
#     y'^2 + z'^2) - V(x, y, z) (always autonomous -- V never depends on t)
#     and requiring the learner to find the symmetry themselves before
#     applying Noether's theorem:
#       - medium: V is linear in exactly two of the three coordinates, so
#         one coordinate is cyclic (absent from V) -- the learner must SPOT
#         that missing coordinate and give its conjugate momentum.
#       - hard: V couples two coordinates only through u^2 + v^2 (invariant
#         under rotation about the third axis) plus an explicit linear term
#         in that third axis (breaking its own translation symmetry, so
#         nothing but the rotation is available) -- the learner must
#         recognise the rotationally-invariant combination.
#       - exam: V is fully spherically symmetric (a function of
#         x^2 + y^2 + z^2 alone), which admits MORE than one symmetry at
#         once (every rotation, plus time-translation): the question asks
#         which of a specific listed quantity is one of the conserved ones,
#         with the three linear momenta (always broken here, since V depends
#         on every coordinate) as distractors.
#
# Since every V used here is autonomous (no explicit t), energy is ALWAYS
# conserved in addition to whichever spatial symmetry the question is about.
# Medium and hard therefore never offer "E" as a distractor (it would be a
# distractor that is secretly also correct) -- the question ties the correct
# answer to the SPECIFIC symmetry under discussion ("that coordinate's
# absence" / "that rotational symmetry"), and distractors are drawn only
# from the other spatial quantities, all of which are genuinely not
# conserved for the stated V.
_ALL_QUANTITIES = ["E", "p_x", "p_y", "p_z", "L_x", "L_y", "L_z"]
_AXES = ["x", "y", "z"]

_TRANSLATION_SYMMETRIES = {
    "time": ("E", "the Lagrangian does not depend explicitly on time t"),
    "x": ("p_x", "the Lagrangian does not depend on the x coordinate"),
    "y": ("p_y", "the Lagrangian does not depend on the y coordinate"),
    "z": ("p_z", "the Lagrangian does not depend on the z coordinate"),
}

# Isolated systems only: none of these scenarios anchors the system to any
# external, fixed reference, so a stated translation, rotation or
# time-translation symmetry is true of every one of them.
_PHYSICAL_SCENARIOS = [
    "an isolated system of point masses connected by ideal springs, free to move in space",
    "an isolated rigid body free to move in space",
    "a free particle subject to no external forces",
]

_EASY_TEMPLATES = (
    "For {scenario}, the Lagrangian has the following property: {clause}. "
    "By Noether's theorem, which quantity is conserved as a result of this symmetry?",
    "Consider {scenario} whose Lagrangian satisfies: {clause}. Which quantity does Noether's theorem say is conserved?",
)

_PHYSICAL_LANGUAGE = {
    "time": "the setup is unchanged if the whole experiment is run starting at a later time",
    "x": "the setup is unchanged if the whole system is shifted along the x direction",
    "y": "the setup is unchanged if the whole system is shifted along the y direction",
    "z": "the setup is unchanged if the whole system is shifted along the z direction",
}

_LAGRANGIAN = "L = (m/2)(x'^2 + y'^2 + z'^2) - V(x, y, z)"


def _fmt_term(coef: int, unit: str) -> str:
    mag = abs(coef)
    return f"{'' if mag == 1 else mag}{unit}"


def _join_signed_terms(terms) -> str:
    """`terms` is a list of (nonzero int coefficient, unit string) pairs;
    joins them as a signed sum, e.g. [(-3, "x"), (5, "z")] -> "-3x + 5z"."""
    out = []
    for i, (coef, unit) in enumerate(terms):
        t = _fmt_term(coef, unit)
        if i == 0:
            out.append(f"-{t}" if coef < 0 else t)
        else:
            out.append(f" - {t}" if coef < 0 else f" + {t}")
    return "".join(out)


def _draw_easy(rng, phrasing: str):
    key = rng.choice(list(_TRANSLATION_SYMMETRIES.keys()))
    quantity, formal_clause = _TRANSLATION_SYMMETRIES[key]
    clause = formal_clause if phrasing == "formal" or key == "time" else _PHYSICAL_LANGUAGE[key]
    scenario_text = rng.choice(_PHYSICAL_SCENARIOS)
    text = phrase(rng, _EASY_TEMPLATES, scenario=scenario_text, clause=clause)
    distractors = [q for q in _ALL_QUANTITIES if q != quantity]
    rng.shuffle(distractors)
    working = [f"Stated symmetry: {clause}.", f"Conserved quantity: {quantity}."]
    return quantity, text, distractors, working


def _draw_medium(rng):
    """V is linear in exactly two of the three coordinates: the third is
    cyclic (absent from V), so its conjugate momentum is conserved."""
    missing = rng.choice(_AXES)
    present = [c for c in _AXES if c != missing]
    k1 = rng.randint(2, 9) * rng.choice([1, -1])
    k2 = rng.randint(2, 9) * rng.choice([1, -1])
    v_text = _join_signed_terms([(k1, present[0]), (k2, present[1])])
    target = f"p_{missing}"
    text = (
        f"A particle of mass m obeys {_LAGRANGIAN}, with V(x, y, z) = {v_text} and no explicit "
        "time dependence. Find the coordinate that does not appear in V (the cyclic coordinate), "
        "and use Noether's theorem to say which quantity is conserved as a result of that "
        "coordinate's absence."
    )
    distractor_pool = [q for q in _ALL_QUANTITIES if q not in (target, "E")]
    working = [
        f"V(x, y, z) = {v_text} does not depend on {missing}.",
        f"{missing} is cyclic, so its conjugate momentum {target} is conserved.",
    ]
    return target, text, distractor_pool, working


def _draw_hard(rng):
    """V couples two coordinates only through u^2 + v^2 (rotationally
    invariant about the third axis) plus an explicit linear term in that
    third axis, which breaks its own translation symmetry -- only the
    rotation survives."""
    axis = rng.choice(_AXES)
    u, v = [c for c in _AXES if c != axis]
    k1 = rng.randint(2, 9)
    k2 = rng.randint(2, 9) * rng.choice([1, -1])
    v_text = _join_signed_terms([(k1, f"({u}^2 + {v}^2)"), (k2, axis)])
    target = f"L_{axis}"
    text = (
        f"A particle of mass m obeys {_LAGRANGIAN}, with V(x, y, z) = {v_text} and no explicit "
        f"time dependence. Identify the axis about which V is invariant under rotation (V depends "
        f"on the other two coordinates only through the sum of their squares), and use Noether's "
        "theorem to say which quantity is conserved as a result of that rotational symmetry."
    )
    distractor_pool = [q for q in _ALL_QUANTITIES if q not in (target, "E")]
    working = [
        f"V(x, y, z) = {v_text} depends on {u}, {v} only via {u}^2 + {v}^2.",
        f"V is invariant under rotation about the {axis}-axis, so {target} is conserved.",
    ]
    return target, text, distractor_pool, working


def _draw_exam(rng):
    """V is fully spherically symmetric (a function of x^2+y^2+z^2 alone),
    so it genuinely admits MORE than one symmetry at once: every rotation,
    and (since there is still no explicit t) time-translation too. Unlike
    medium/hard, the prompt does NOT say so -- it states V and asks; the
    learner must recognise the spherical symmetry themselves from V alone
    z only through x^2+y^2+z^2, so it is invariant under rotation about
    every axis" -- gave the reasoning away, leaving only a single non-
    momentum option to pick). Exactly one of the four OFFERED options is
    conserved: the target is one of {E, L_x, L_y, L_z} (all genuinely
    conserved for this V, but only one is ever shown), and the three
    distractors are always p_x, p_y, p_z (always genuinely broken, since V
    depends on every coordinate)."""
    k = rng.randint(2, 9) * rng.choice([1, -1])
    form = rng.choice(["sqrt", "inv_sqrt"])
    radicand = "x^2 + y^2 + z^2"
    sign = "-" if k < 0 else ""
    mag = "" if abs(k) == 1 else str(abs(k))
    if form == "sqrt":
        v_text = f"{sign}{mag}sqrt({radicand})"
    else:
        v_text = f"{sign}{mag if mag else '1'}/sqrt({radicand})"
    true_set = ["E", "L_x", "L_y", "L_z"]
    target = rng.choice(true_set)
    distractor_pool = ["p_x", "p_y", "p_z"]
    text = phrase(
        rng,
        (
            "A particle of mass m obeys {lagrangian}, with V(x, y, z) = {v_text} and no explicit "
            "time dependence. Which of the following quantities is conserved?",
            "A particle of mass m has {lagrangian}, where V(x, y, z) = {v_text} and there is no "
            "explicit time dependence. Which of the following quantities is conserved?",
        ),
        lagrangian=_LAGRANGIAN,
        v_text=v_text,
    )
    working = [
        f"V(x, y, z) = {v_text} depends only on {radicand} (spherically symmetric).",
        f"Every rotation leaves V unchanged, and V has no explicit t, so E, L_x, L_y, L_z "
        "are all conserved; p_x, p_y, p_z are not, since V depends on every coordinate.",
        f"Asked about: {target}.",
    ]
    return target, text, distractor_pool, working


@generator(
    "p15c_symmetry_conservation",
    level=17,
    presets={
        "easy": dict(mode="words", phrasing="formal"),
        "medium": dict(mode="shape_a"),
        "hard": dict(mode="shape_b"),
        "exam": dict(mode="shape_c"),
    },
)
def p15c_symmetry_conservation(rng, mode="words", phrasing="formal"):
    if mode == "words":
        target, text, distractor_pool, working = _draw_easy(rng, phrasing)
    elif mode == "shape_a":
        target, text, distractor_pool, working = _draw_medium(rng)
    elif mode == "shape_b":
        target, text, distractor_pool, working = _draw_hard(rng)
    else:
        target, text, distractor_pool, working = _draw_exam(rng)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=target,
        answer_text=target,
        answer_latex=f"plain:{target}",
        working=working,
        distractors=pick_distractors(target, distractor_pool, n=3),
    )
