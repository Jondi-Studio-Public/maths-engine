"""Generates `ucv1-finite-subgroup-test` and `ucv1-isomorphism-invariant` questions on subgroup tests and
isomorphism invariants, both `answer: mcq`.

`ucv1-finite-subgroup-test` and `ucv1-isomorphism-invariant`. Both `answer: mcq`.

`ucv1-finite-subgroup-test`: decide whether an explicitly listed subset S of
a finite group G is a subgroup, by checking IDENTITY, then CLOSURE, then
INVERSES, in that stated order -- reporting the FIRST condition that fails.

Honest answer space, by construction (never invented -- see the note below): {"subgroup", "fails identity",
"fails closure"}. "fails inverses" is offered as a DISTRACTOR on every draw but is never the key: for a
nonempty subset of a FINITE group, containing the identity and closed under the operation FORCES closure under
inverses too (the standard "finite subgroup test" theorem -- for any a in S, the powers a, a^2, a^3... all lie
in S by closure, and since S is finite that sequence must repeat, giving a^k = e for some k >= 1, so a^(k-1) =
a^-1 also lies in S by closure). So "fails inverses" can never be the true state of a set that has already
passed identity and closure -- it is a mathematically IMPOSSIBLE key, and rules lesson (2) says an impossible
class may be a distractor, never the key.

`ucv1-isomorphism-invariant`: given the order and element-order profile of
two finite groups G, H, decide whether a claimed isomorphism G =~ H should be
retained (sizes and profiles agree) or rejected (sizes differ, or sizes
agree but some element-order count differs). 3-way answer space, uniform.

Construct-to-target throughout: the target class is drawn FIRST, uniformly,
then the data is built to realise it exactly, then re-verified via a fresh
computation independent of the construction branch.
"""

from __future__ import annotations

import itertools

from mathgen.core import check, generator, latex_answer
from mathgen.generators.p15a_group_data import GROUP_CATALOG, ORDER_TO_NAMES, DISPLAY_NAME

# ===========================================================================
# Skill A: p15a_finite_subgroup_test
# ===========================================================================

_SUBGROUP_CLASSES = ["subgroup", "fails identity", "fails closure"]


def _subgroup_label(cls: str, mode: str) -> str:
    if mode == "additive":
        identity_word, op_word = "0", "sum"
    else:
        identity_word, op_word = "the identity permutation ()", "composition"
    return {
        "subgroup": "S is a subgroup of G",
        "fails identity": f"not a subgroup: fails the identity condition ({identity_word} is not in S)",
        "fails closure": f"not a subgroup: fails the closure condition (some {op_word} of two elements of S is not in S)",
        "fails inverses": "not a subgroup: fails the inverse condition (some element of S has no inverse in S)",
    }[cls]


_ORDER_CONVENTION = (
    "Check the conditions in this order: identity, then closure, then inverses. "
    "If more than one condition would fail, report only the FIRST one that fails "
    "in that order."
)


def _classify_additive_subset(n: int, S: list[int]) -> str:
    """Fresh classification from the residues alone (identity=0, closure and
    inverses computed by direct modular arithmetic over ALL pairs -- never
    reusing whichever branch built S)."""
    if 0 not in S:
        return "fails identity"
    S_set = set(S)
    for a in S:
        for b in S:
            if (a + b) % n not in S_set:
                return "fails closure"
    for a in S:
        check((-a) % n in S_set, "finite subgroup test theorem violated: identity+closure without inverses")
    return "subgroup"


def _build_additive_subset(rng, n: int, target: str) -> list[int]:
    if target == "subgroup":
        divisors = [d for d in range(2, n) if n % d == 0]
        d = rng.choice(divisors)
        return sorted((k * d) % n for k in range(n // d))
    if target == "fails identity":
        size = rng.choice([2, 3])
        return sorted(rng.sample(range(1, n), size))
    # fails closure: retry random {0,a,b} triples until a genuine violation occurs
    for _ in range(200):
        a, b = rng.sample(range(1, n), 2)
        S = sorted({0, a, b})
        if _classify_additive_subset(n, S) == "fails closure":
            return S
    raise AssertionError("could not construct a fails-closure additive draw")


# --- permutation-group machinery for hard/exam (D4, then S4) --------------


def _compose(p: tuple, q: tuple) -> tuple:
    """(p o q)(i) = p(q(i))."""
    return tuple(p[q[i]] for i in range(len(p)))


def _inverse(p: tuple) -> tuple:
    inv = [0] * len(p)
    for j, pj in enumerate(p):
        inv[pj] = j
    return tuple(inv)


def _cycle_notation(p: tuple) -> str:
    """1-indexed cycle notation, e.g. (1 2 3), omitting fixed points; the
    identity renders as '()'."""
    n = len(p)
    seen = [False] * n
    cycles = []
    for start in range(n):
        if seen[start] or p[start] == start:
            seen[start] = True
            continue
        cyc = [start]
        seen[start] = True
        nxt = p[start]
        while nxt != start:
            cyc.append(nxt)
            seen[nxt] = True
            nxt = p[nxt]
        cycles.append(cyc)
    if not cycles:
        return "()"
    return "".join("(" + " ".join(str(x + 1) for x in cyc) + ")" for cyc in cycles)


def _d4_elements() -> list[tuple]:
    """Symmetries of a square on vertices 0..3: 4 rotations r^k: i -> (i+k)%4,
    and 4 reflections s_k: i -> (k-i)%4."""
    els = [tuple((i + k) % 4 for i in range(4)) for k in range(4)]
    els += [tuple((k - i) % 4 for i in range(4)) for k in range(4)]
    return els


def _s4_elements() -> list[tuple]:
    return [p for p in itertools.permutations(range(4))]


def _classify_perm_subset(n: int, S: list[tuple]) -> str:
    identity = tuple(range(n))
    if identity not in S:
        return "fails identity"
    S_set = set(S)
    for a in S:
        for b in S:
            if _compose(a, b) not in S_set:
                return "fails closure"
    for a in S:
        check(_inverse(a) in S_set, "finite subgroup test theorem violated: identity+closure without inverses")
    return "subgroup"


def _generate_subgroup(elements: list[tuple], gens: list[tuple]) -> list[tuple]:
    identity = tuple(range(len(elements[0])))
    S = {identity} | set(gens)
    changed = True
    while changed:
        changed = False
        for a in list(S):
            for b in list(S):
                p = _compose(a, b)
                if p not in S:
                    S.add(p)
                    changed = True
    return sorted(S)


def _build_perm_subset(rng, elements: list[tuple], target: str) -> list[tuple]:
    n = len(elements[0])
    identity = tuple(range(n))
    non_identity = [e for e in elements if e != identity]
    if target == "subgroup":
        n_gens = rng.choice([1, 2])
        for _ in range(50):
            gens = rng.sample(non_identity, n_gens)
            S = _generate_subgroup(elements, gens)
            if 2 <= len(S) < len(elements):
                return S
        return _generate_subgroup(elements, [rng.choice(non_identity)])
    if target == "fails identity":
        size = rng.choice([2, 3])
        return sorted(rng.sample(non_identity, size))
    for _ in range(300):
        size = rng.choice([2, 3])
        picks = rng.sample(non_identity, size)
        S = sorted(set([identity] + picks))
        if _classify_perm_subset(n, S) == "fails closure":
            return S
    raise AssertionError("could not construct a fails-closure permutation draw")


def _distractors_for(target: str) -> list[str]:
    """Exactly 3 distractors every draw: the other two honest classes, plus
    the always-wrong 'fails inverses' option (see module docstring)."""
    others = [c for c in _SUBGROUP_CLASSES if c != target]
    return others + ["fails inverses"]


@generator(
    "p15a_finite_subgroup_test",
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p15a_finite_subgroup_test(rng, tier="easy"):
    target = rng.choice(_SUBGROUP_CLASSES)

    if tier == "easy":
        n = rng.choice([6, 8])
        S = _build_additive_subset(rng, n, target)
        recomputed = _classify_additive_subset(n, S)
        check(recomputed == target, "easy subgroup-test construction mismatch")
        S_text = "{" + ", ".join(str(x) for x in S) + "}"
        text = (
            f"Let G be the set {{0, 1, ..., {n - 1}}} under addition modulo {n}, "
            f"with identity element 0. Let S = {S_text}. {_ORDER_CONVENTION} "
            f"Which statement about S is correct?"
        )
        working = [
            f"G = Z_{n} under addition mod {n}.",
            f"S = {S_text}.",
            f"Checking identity, closure, inverses in order: {target}.",
        ]
    elif tier == "medium":
        n = rng.choice([9, 10, 12])
        S = _build_additive_subset(rng, n, target)
        recomputed = _classify_additive_subset(n, S)
        check(recomputed == target, "medium subgroup-test construction mismatch")
        S_text = "{" + ", ".join(str(x) for x in S) + "}"
        text = (
            f"Let G be the set {{0, 1, ..., {n - 1}}} under addition modulo {n}, "
            f"with identity element 0. Let S = {S_text}. {_ORDER_CONVENTION} "
            f"Which statement about S is correct?"
        )
        working = [
            f"G = Z_{n} under addition mod {n}.",
            f"S = {S_text}.",
            f"Checking identity, closure, inverses in order: {target}.",
        ]
    elif tier == "hard":
        elements = _d4_elements()
        S = _build_perm_subset(rng, elements, target)
        recomputed = _classify_perm_subset(4, S)
        check(recomputed == target, "hard subgroup-test construction mismatch")
        S_text = ", ".join(_cycle_notation(p) for p in S)
        text = (
            f"G is the group of symmetries of a square, realised as permutations of "
            f"its four vertices {{1, 2, 3, 4}}, written in cycle notation (the identity "
            f"is written '()'). The group operation is composition of permutations, "
            f"applying the RIGHT permutation first. Let S = {{{S_text}}}. {_ORDER_CONVENTION} "
            f"Which statement about S is correct?"
        )
        working = [
            f"S (as permutations of the square's vertices) = {{{S_text}}}.",
            f"Checking identity, closure, inverses in order: {target}.",
        ]
    else:  # exam
        elements = _s4_elements()
        S = _build_perm_subset(rng, elements, target)
        recomputed = _classify_perm_subset(4, S)
        check(recomputed == target, "exam subgroup-test construction mismatch")
        S_text = ", ".join(_cycle_notation(p) for p in S)
        text = (
            f"G is the full symmetric group of permutations of {{1, 2, 3, 4}}, written in "
            f"cycle notation (the identity is written '()'). The group operation is "
            f"composition of permutations, applying the RIGHT permutation first. "
            f"Let S = {{{S_text}}}. {_ORDER_CONVENTION} Which statement about S is correct?"
        )
        working = [
            f"S (as permutations of {{1,2,3,4}}) = {{{S_text}}}.",
            f"Checking identity, closure, inverses in order: {target}.",
        ]

    mode = "additive" if tier in ("easy", "medium") else "permutation"
    distractors = _distractors_for(target)
    check(len(distractors) == 3, "subgroup-test must offer exactly 3 distractors")
    for d in distractors:
        check(d != target, "subgroup-test distractor collides with the answer")

    answer_text = _subgroup_label(target, mode)
    distractor_labels = [_subgroup_label(d, mode) for d in distractors]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractor_labels,
    )


# ===========================================================================
# Skill B: p15a_isomorphism_invariant
# ===========================================================================

_ISO_CLASSES = ["retain", "reject_size", "reject_profile"]
_ISO_LABEL = {
    "retain": "retain the claim: the sizes and element-order profiles agree",
    "reject_size": "reject the claim: |G| and |H| differ",
    "reject_profile": "reject the claim: |G| = |H| but the element-order counts differ for some order",
}

_ORDERS_WITH_MULTIPLE = [n for n, names in ORDER_TO_NAMES.items() if len(names) >= 2]


def _profile_text(profile: dict) -> str:
    return ", ".join(
        f"{k} element of order {v}" if k == 1 else f"{k} elements of order {v}"
        for v, k in sorted(profile.items(), key=lambda kv: kv[1])
    )


def _profile_raw_list(rng, profile: dict) -> list[int]:
    raw = []
    for order, count in profile.items():
        raw += [order] * count
    rng.shuffle(raw)
    return raw


def _tally(raw: list[int]) -> dict:
    out: dict[int, int] = {}
    for v in raw:
        out[v] = out.get(v, 0) + 1
    return out


def _decide(order_g: int, order_h: int, profile_g: dict, profile_h: dict) -> str:
    if order_g != order_h:
        return "reject_size"
    if profile_g == profile_h:
        return "retain"
    return "reject_profile"


def _pick_pair(rng, target: str, orders_pool: list[int]):
    if target == "retain":
        n = rng.choice(orders_pool)
        name = rng.choice(ORDER_TO_NAMES[n])
        return n, name, n, name
    if target == "reject_size":
        n1, n2 = rng.sample(orders_pool, 2)
        name1 = rng.choice(ORDER_TO_NAMES[n1])
        name2 = rng.choice(ORDER_TO_NAMES[n2])
        return n1, name1, n2, name2
    # reject_profile: same order, two structurally different named groups
    eligible = [n for n in orders_pool if len(ORDER_TO_NAMES[n]) >= 2]
    n = rng.choice(eligible)
    name1, name2 = rng.sample(ORDER_TO_NAMES[n], 2)
    return n, name1, n, name2


@generator(
    "p15a_isomorphism_invariant",
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p15a_isomorphism_invariant(rng, tier="easy"):
    target = rng.choice(_ISO_CLASSES)

    if tier == "easy":
        # Widened: [4, 6] alone gave only 16 distinct prompts over 200 seeds. 8 and 9 are already in
        # the catalogue (order 8 alone has 4 named groups: Z8, Z4xZ2, Z2xZ2xZ2, D4) and easy-medium
        # is documented as a range-only step (no reasoning escalation is owed between them, unlike
        # medium-hard-exam), so reusing medium's order values here does not blur any required
        # escalation boundary.
        pool = [4, 6, 8, 9]
    elif tier == "medium":
        pool = [8, 9]
    elif tier == "hard":
        pool = [8, 9, 10, 12]
    else:
        pool = [10, 12]

    n_g, name_g, n_h, name_h = _pick_pair(rng, target, pool)
    order_g, profile_g = GROUP_CATALOG[name_g]
    order_h, profile_h = GROUP_CATALOG[name_h]
    check(order_g == n_g and order_h == n_h, "catalogue order mismatch")
    recomputed = _decide(order_g, order_h, profile_g, profile_h)
    check(recomputed == target, "isomorphism-invariant construction mismatch")

    if tier in ("easy", "medium"):
        text = (
            f"Group G has order {order_g}, with {_profile_text(profile_g)}. "
            f"Group H has order {order_h}, with {_profile_text(profile_h)}. "
            f"A student claims G is isomorphic to H. Should the claim be retained or "
            f"rejected, and why?"
        )
    elif tier == "hard":
        raw_g = _profile_raw_list(rng, profile_g)
        raw_h = _profile_raw_list(rng, profile_h)
        check(_tally(raw_g) == profile_g and _tally(raw_h) == profile_h, "tally round-trip failed")
        raw_g_text = ", ".join(str(v) for v in raw_g)
        raw_h_text = ", ".join(str(v) for v in raw_h)
        text = (
            f"Group G has order {order_g}. The order of each of its {order_g} elements, listed "
            f"in no particular order, is: {raw_g_text}. Group H has order {order_h}, with element "
            f"orders (again in no particular order): {raw_h_text}. A student claims G is isomorphic "
            f"to H. Tally the element orders for each group, then decide whether the claim should "
            f"be retained or rejected, and why."
        )
    else:  # exam
        # The genuine extra step here is a DATA-EXTRACTION one, the same kind already used elsewhere
        # (a pair-list needing translation into an incidence grid): G's and H's element orders are
        # supplied as ONE shuffled, tagged list -- the two groups' entries interleaved together --
        # so the learner must first separate by tag before they can even tally either group, a step
        # hard's already-separated two lists never requires. Still strictly SUPPLIED data throughout
        # (neither tier ever asks the learner to compute an element order).
        raw_g = _profile_raw_list(rng, profile_g)
        raw_h = _profile_raw_list(rng, profile_h)
        check(_tally(raw_g) == profile_g and _tally(raw_h) == profile_h, "tally round-trip failed")
        tagged = [("G", v) for v in raw_g] + [("H", v) for v in raw_h]
        rng.shuffle(tagged)
        check(
            _tally([v for t, v in tagged if t == "G"]) == profile_g
            and _tally([v for t, v in tagged if t == "H"]) == profile_h,
            "exam tag/shuffle round-trip failed",
        )
        tagged_text = ", ".join(f"{t}:{v}" for t, v in tagged)
        text = (
            f"Group G has order {order_g} and group H has order {order_h}. The list below "
            f"has one entry per element of G or H, tagged by which group it belongs to, with "
            f"that element's order -- all {order_g + order_h} entries from both groups "
            f"shuffled together in no particular order: {tagged_text}. A student claims G is "
            f"isomorphic to H. Separate the entries by group, tally the element orders for "
            f"each group, then decide whether the claim should be retained or rejected, and "
            f"why."
        )

    working = [
        f"|G| = {order_g}, profile {dict(sorted(profile_g.items()))}.",
        f"|H| = {order_h}, profile {dict(sorted(profile_h.items()))}.",
        f"Decision: {target}.",
    ]

    distractors = [_ISO_LABEL[c] for c in _ISO_CLASSES if c != target]
    check(len(distractors) == 2, "isomorphism-invariant must offer exactly 2 distractors")
    answer_text = _ISO_LABEL[target]
    check(answer_text not in distractors, "isomorphism-invariant distractor collides with the answer")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
    )
