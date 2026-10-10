"""Number-theory + discrete-mathematics pair.

Two generators:

- ``chinese_remainder_two_congruences`` (``ucv1-chinese-remainder-two-
  congruences``, number-theory): find the least non-negative x satisfying two
  supplied COPRIME linear congruences x = r1 (mod m1), x = r2 (mod m2).
  Constructed via the standard modular-inverse CRT formula; independently
  re-checked by brute-force enumeration over the whole range [0, m1*m2) --
  a route that shares no arithmetic with the construction. None of this
  ``modular_arithmetic``, ``modular_multiplicative_inverse``, rendered in
  of them reduces or inverts a single value against a single modulus.

- ``inclusion_exclusion_two_sets`` (``ucv1-inclusion-exclusion-two-sets``,
  discrete-mathematics): |A union B| = |A| + |B| - |A intersect B|, from
  CARDINALITIES stated directly in a short word problem -- never from
  explicitly listed elements (that shape belongs exclusively to this
  like ``{1, 3, 5}``). Every draw is built from three independent, smaller
  "atomic" partition counts (``a_only``, ``b_only``, ``inter``, plus
  ``neither`` at the hard tier) and every displayed quantity (|A|, |B|,
  |A union B|, the total population) is a SUM of those atoms -- so no draw
  can ever realise a negative intersection, a union exceeding the total, or
  a "neither" count exceeding the population; consistency is a structural
  guarantee of the construction, not a post-hoc filter. Independently
  re-checked by building ACTUAL Python ``set`` objects of consecutive
  integers realising the drawn sizes and taking their union with Python's
  own ``|`` operator -- a route that shares no arithmetic with the formula.
"""

from __future__ import annotations

import math

from mathgen.core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    phrase,
    pick_distractors,
)

_RESAMPLE_ATTEMPTS = 200


# =============================================================================
# Skill 1: chinese_remainder_two_congruences
# =============================================================================

_CRT_TEMPLATES_STANDARD = (
    "Find the least non-negative integer x such that x ≡ {r1_raw} (mod {m1}) and x ≡ {r2} (mod {m2}).",
    "What is the smallest non-negative integer x satisfying x ≡ {r1_raw} (mod {m1}) and x ≡ {r2} (mod {m2})?",
    "Determine the least integer x, with x ≥ 0, for which x ≡ {r1_raw} (mod {m1}) and x ≡ {r2} (mod {m2}).",
)

_CRT_TEMPLATES_NEGATIVE = (
    "x ≡ {r1_raw} (mod {m1}) states a negative remainder, and x ≡ {r2} (mod {m2}) is the "
    "second congruence. Find the least non-negative integer x satisfying both.",
    "Given the negative-remainder congruence x ≡ {r1_raw} (mod {m1}) together with "
    "x ≡ {r2} (mod {m2}), determine the least non-negative integer x satisfying both.",
    "Reduce the negative remainder first, then combine: x ≡ {r1_raw} (mod {m1}) and "
    "x ≡ {r2} (mod {m2}). What is the least non-negative integer x satisfying both "
    "congruences?",
)


def _crt_combine(r1: int, m1: int, r2: int, m2: int) -> int:
    """The standard CRT construction via a modular inverse: the unique
    ``x`` in ``[0, m1*m2)`` with ``x = r1 (mod m1)`` and ``x = r2 (mod m2)``,
    for coprime ``m1``, ``m2``. ``r1``, ``r2`` must already be reduced
    (``0 <= r < m``)."""
    inv = pow(m1, -1, m2)
    t = ((r2 - r1) * inv) % m2
    return (r1 + m1 * t) % (m1 * m2)


def _verify_crt(x: int, r1_raw: int, m1: int, r2_raw: int, m2: int) -> None:
    """Independent route: brute-force enumerate every integer in
    ``[0, m1*m2)`` and confirm ``x`` is the UNIQUE one satisfying both RAW
    congruences reduced by ``%`` (so the hard tier's negative remainder is
    itself exercised here, exactly as given in the question) -- a different
    computation from the modular-inverse construction above, sharing no
    arithmetic with it."""
    solutions = [
        candidate for candidate in range(m1 * m2) if candidate % m1 == r1_raw % m1 and candidate % m2 == r2_raw % m2
    ]
    check(len(solutions) == 1, "a coprime 2x2 CRT system must have exactly one solution in [0, m1*m2)")
    check_equivalent(solutions[0], x, "constructed CRT solution disagrees with brute-force enumeration")


def _draw_coprime_moduli(rng, lo: int, hi: int) -> tuple[int, int]:
    """Draw two moduli in ``[lo, hi]``, resampling (never reducing) until
    they are coprime -- an excluded, non-coprime system is never silently
    turned into a solvable one."""
    for _ in range(_RESAMPLE_ATTEMPTS):
        m1 = rng.randint(lo, hi)
        m2 = rng.randint(lo, hi)
        if math.gcd(m1, m2) == 1:
            return m1, m2
    raise ValueError(f"chinese_remainder_two_congruences: no coprime pair found in [{lo}, {hi}]")


def _crt_question(rng, *, modulus_lo: int, modulus_hi: int, negative_first: bool) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        m1, m2 = _draw_coprime_moduli(rng, modulus_lo, modulus_hi)
        r1 = rng.randint(1, m1 - 1)
        r2 = rng.randint(1, m2 - 1)
        x = _crt_combine(r1, m1, r2, m2)

        first_only = r1
        second_only = r2
        adds = (r1 + r2) % (m1 * m2)

        if negative_first:
            if 2 * r1 == m1:
                continue  # sign-flip would coincide with the true answer -- resample
            sign_flip_value = m1 - r1
            x_sign_flip = _crt_combine(sign_flip_value, m1, r2, m2)
            candidates = [x_sign_flip, first_only, second_only, adds]
        else:
            candidates = [first_only, second_only, adds]

        try:
            check_pairwise_distinct(x, *candidates)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("chinese_remainder_two_congruences: no collision-free draw found")

    r1_raw = (r1 - m1) if negative_first else r1
    _verify_crt(x, r1_raw, m1, r2, m2)
    check_equivalent(_crt_combine(r1, m1, r2, m2), x, "CRT construction must reproduce the answer")

    answer_text = str(x)
    templates = _CRT_TEMPLATES_NEGATIVE if negative_first else _CRT_TEMPLATES_STANDARD
    text = phrase(rng, templates, r1_raw=r1_raw, m1=m1, r2=r2, m2=m2)
    latex = rf"x \equiv {r1_raw} \pmod{{{m1}}} \text{{ and }} x \equiv {r2} \pmod{{{m2}}}"
    working = (
        [f"x ≡ {r1_raw} (mod {m1}) reduces to the least non-negative residue x ≡ {r1} (mod {m1})."]
        if negative_first
        else []
    ) + [f"Combining x ≡ {r1} (mod {m1}) with x ≡ {r2} (mod {m2}) via the CRT construction gives x = {x}."]

    distractors = pick_distractors(answer_text, [str(c) for c in candidates], n=len(candidates))

    return dict(
        text=text,
        latex=latex,
        answer=x,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


@generator(
    "chinese_remainder_two_congruences",
    level=16,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def chinese_remainder_two_congruences(rng, tier="hard"):
    """Solve two coprime linear congruences for the least non-negative x.

    easy: moduli each drawn from 2..6 (so m1*m2 <= 30, the coprimality
    constraint on that range never lets the product exceed 30 -- 5x6 is the
    largest coprime pair available) -- small enough to find x by inspection.
    medium: moduli each drawn from 4..12 -- inspection is no longer the
    fastest route; the learner is expected to use the standard construction.
    hard: same modulus ranges as medium, but the first congruence's
    remainder is stated as a NEGATIVE integer (``x = r1 - m1``, which always
    reduces back to the true canonical r1), so the learner's first genuine
    extra step is reducing it before combining the two congruences.
    """
    if tier == "easy":
        return _crt_question(rng, modulus_lo=2, modulus_hi=6, negative_first=False)
    if tier == "medium":
        return _crt_question(rng, modulus_lo=4, modulus_hi=12, negative_first=False)
    if tier == "hard":
        return _crt_question(rng, modulus_lo=4, modulus_hi=12, negative_first=True)
    raise ValueError(f"unknown tier: {tier!r}")


# =============================================================================
# Skill 2: inclusion_exclusion_two_sets
# =============================================================================

#: (population noun, item A, item B, verb) -- deliberately non-exclusive
#: activities (liking a drink, playing a game, owning something, knowing a
#: language) so "or both" is always a natural, non-exclusive reading; never
#: a left-/right-handedness-style exclusive pair.
_CONTEXTS = [
    ("people", "tea", "coffee", "like"),
    ("students", "chess", "cards", "play"),
    ("residents", "cars", "bicycles", "own"),
    ("customers", "bread", "milk", "bought"),
    ("employees", "Python", "SQL", "know"),
    ("visitors", "the museum", "the gallery", "toured"),
]


def _verify_union_by_sets(a_only: int, b_only: int, inter: int, union: int) -> None:
    """Independent route: build ACTUAL finite sets of consecutive integers
    realising the drawn sizes -- overlap the first ``inter`` elements of B
    with the tail of A, then extend B with fresh elements for the rest of
    its own size -- and take Python's own ``|`` (set union) directly,
    comparing its length to the arithmetic formula's result. Shares no
    arithmetic with the construction above."""
    a_size = a_only + inter
    a = set(range(a_size))
    b = set(range(a_size - inter, a_size + b_only))
    check(len(a & b) == inter, "constructed sets do not realise the stated intersection size")
    check(len(b) == b_only + inter, "constructed set B does not realise the stated size")
    check_equivalent(len(a | b), union, "set-union enumeration disagrees with the arithmetic union")


def _verify_neither(total: int, union: int, neither: int) -> None:
    check_equivalent(total - union, neither, "total population minus the union must equal the neither count")


def _ie_easy(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        a_only = rng.randint(3, 20)
        b_only = rng.randint(3, 20)
        inter = rng.randint(2, 15)
        a_size, b_size = a_only + inter, b_only + inter
        union = a_only + b_only + inter
        forgets_overlap = a_size + b_size
        double_subtracts = a_size + b_size - 2 * inter
        try:
            check_pairwise_distinct(union, forgets_overlap, double_subtracts)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("inclusion_exclusion_two_sets: no collision-free easy draw found")

    _verify_union_by_sets(a_only, b_only, inter, union)
    check_equivalent(a_size + b_size - inter, union, "easy answer must equal the inclusion-exclusion formula")

    population, noun_a, noun_b, verb = rng.choice(_CONTEXTS)
    text = (
        f"Among a group of {population}, {a_size} {verb} {noun_a}, {b_size} {verb} {noun_b}, "
        f"and {inter} {verb} both {noun_a} and {noun_b}. "
        f"How many {population} {verb} {noun_a}, {noun_b}, or both?"
    )
    answer_text = str(union)
    distractors = pick_distractors(answer_text, [str(forgets_overlap), str(double_subtracts)])

    return dict(
        text=text,
        answer=union,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"|A ∪ B| = |A| + |B| - |A ∩ B| = {a_size} + {b_size} - {inter} = {union}."],
        distractors=distractors,
    )


def _ie_medium(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        a_only = rng.randint(3, 20)
        b_only = rng.randint(3, 20)
        inter = rng.randint(2, 15)
        a_size, b_size = a_only + inter, b_only + inter
        union = a_only + b_only + inter
        swaps = union
        try:
            check_pairwise_distinct(inter, swaps)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("inclusion_exclusion_two_sets: no collision-free medium draw found")

    _verify_union_by_sets(a_only, b_only, inter, union)
    check_equivalent(a_size + b_size - union, inter, "medium answer must equal the rearranged formula")

    population, noun_a, noun_b, verb = rng.choice(_CONTEXTS)
    text = (
        f"Among a group of {population}, {a_size} {verb} {noun_a}, {b_size} {verb} {noun_b}, "
        f"and {union} {verb} {noun_a}, {noun_b}, or both. "
        f"How many {population} {verb} both {noun_a} and {noun_b}?"
    )
    answer_text = str(inter)
    distractors = pick_distractors(answer_text, [str(swaps)])

    return dict(
        text=text,
        answer=inter,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"|A ∩ B| = |A| + |B| - |A ∪ B| = {a_size} + {b_size} - {union} = {inter}."],
        distractors=distractors,
    )


def _ie_hard(rng):
    for _ in range(_RESAMPLE_ATTEMPTS):
        a_only = rng.randint(3, 20)
        b_only = rng.randint(3, 20)
        inter = rng.randint(2, 15)
        neither = rng.randint(2, 15)
        a_size, b_size = a_only + inter, b_only + inter
        union = a_only + b_only + inter
        total = union + neither
        ignores_neither = a_size + b_size - total
        if ignores_neither < 0:
            continue  # required resample: this misconception can go negative
        try:
            check_pairwise_distinct(inter, ignores_neither)
        except VerificationError:
            continue
        break
    else:
        raise ValueError("inclusion_exclusion_two_sets: no collision-free hard draw found")

    _verify_union_by_sets(a_only, b_only, inter, union)
    _verify_neither(total, union, neither)
    check_equivalent(a_size + b_size - (total - neither), inter, "hard answer must equal the two-step chained formula")

    population, noun_a, noun_b, verb = rng.choice(_CONTEXTS)
    text = (
        f"In a survey of {total} {population}, {a_size} {verb} {noun_a}, {b_size} {verb} {noun_b}, "
        f"and {neither} {verb} neither {noun_a} nor {noun_b}. "
        f"How many {population} {verb} both {noun_a} and {noun_b}?"
    )
    answer_text = str(inter)
    distractors = pick_distractors(answer_text, [str(ignores_neither)])

    return dict(
        text=text,
        answer=inter,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"|A ∪ B| = {total} - {neither} = {union}.",
            f"|A ∩ B| = |A| + |B| - |A ∪ B| = {a_size} + {b_size} - {union} = {inter}.",
        ],
        distractors=distractors,
    )


@generator(
    "inclusion_exclusion_two_sets",
    level=14,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def inclusion_exclusion_two_sets(rng, tier="easy"):
    """Compute one quantity of a two-set inclusion-exclusion scenario from
    CARDINALITIES stated in a word problem (never listed elements).

    easy: |A|, |B|, |A intersect B| given; find |A union B| -- one-step
    formula application. medium: |A|, |B|, |A union B| given; find
    |A intersect B| -- the SAME formula rearranged for a different unknown.
    hard: a total population, |A|, |B| and the count in NEITHER set given;
    find |A intersect B| -- two chained steps (derive |A union B| from the
    total first, then rearrange).

    Bare-call default is "easy" (not "hard", unlike this module's other topic) because medium/hard each have
    only ONE named, tier-specific misconception available, so only "easy" reliably clears
    ``test_all_topics_usually_have_enough_distractors``'s >=2-distractors bar on a bare, no-difficulty call.
    """
    if tier == "easy":
        return _ie_easy(rng)
    if tier == "medium":
        return _ie_medium(rng)
    if tier == "hard":
        return _ie_hard(rng)
    raise ValueError(f"unknown tier: {tier!r}")
