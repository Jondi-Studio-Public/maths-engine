"""Generates `r4a_burnside_counting` questions (easy, medium, hard) and the exam chain `exd_x_r4a_burnside_counting`.

``uc2-burnside-counting`` (``abstract-algebra``, answer ``number``): count arrangements up to symmetry by averaging the
number of arrangements each symmetry fixes (the number of orbits is (1/|G|) × the sum of |Fix(g)|).

* easy: necklaces of n beads in k colours, where rotating a necklace gives the same necklace (n from 4 to 8).
* medium: bracelets of n beads in k colours, where a bracelet may also be turned over, so the group is D_n (n from 4 to 7).
* hard: necklaces (rotations) or bracelets (rotations and reflections) with exactly m beads of one colour and the others of
  a second colour, so each symmetry fixes only the colourings that put whole cycles in that colour.
* exam: bracelets of n beads (n = 6 or 8) in k colours: (a) how many colourings a rotation through two places fixes, (b) how
  many a reflection through two opposite beads fixes, (c) the number of bracelets (follows through from (a) and (b)),
  (d) the number of necklaces under rotation only (follows through from (a)).

Every key is computed from the fixed-colouring counts of the symmetries and checked by listing all the colourings and
counting the orbits directly (taking the least image of each colouring under the whole group).
"""

from __future__ import annotations

from itertools import combinations, product
from math import comb, gcd

from ..core import attempts, check, generator, phrase
from ._r4a_alg_kit import number_part, number_question, with_latex
from .r4a_orbit_stabiliser_count_build import _d_maps


def _cycle_lengths(g):
    seen, out = set(), []
    for start in range(len(g)):
        if start in seen:
            continue
        length, v = 0, start
        while v not in seen:
            seen.add(v)
            v = g[v]
            length += 1
        out.append(length)
    return out


def _group(n, with_reflections):
    rotations, reflections = _d_maps(n)
    return rotations + (reflections if with_reflections else [])


def _fixed_all_colours(g, k):
    return k ** len(_cycle_lengths(g))


def _fixed_exact(g, m):
    """Colourings of the positions in two colours with exactly m of the first colour that g fixes: choose whole cycles."""
    poly = [1] + [0] * m
    for length in _cycle_lengths(g):
        new = poly[:]
        for d in range(m + 1 - length):
            new[d + length] += poly[d]
        poly = new
    return poly[m]


def _apply(g, colouring):
    out = [None] * len(colouring)
    for v, c in enumerate(colouring):
        out[g[v]] = c
    return tuple(out)


def _count_orbits_all(group, n, k):
    """Orbits of colourings with k colours, found by generating each unvisited colouring's whole orbit."""
    seen, orbits = set(), 0
    for colouring in product(range(k), repeat=n):
        if colouring in seen:
            continue
        orbits += 1
        seen.update(_apply(g, colouring) for g in group)
    return orbits


def _count_orbits_exact(group, n, m):
    seen, orbits = set(), 0
    for black in combinations(range(n), m):
        colouring = tuple(1 if v in black else 0 for v in range(n))
        if colouring in seen:
            continue
        orbits += 1
        seen.update(_apply(g, colouring) for g in group)
    return orbits


def _burnside_all(group, k):
    total = sum(_fixed_all_colours(g, k) for g in group)
    check(total % len(group) == 0, "Burnside's sum is not divisible by the group order")
    return total // len(group), total


# -- easy ------------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "How many different necklaces can be made by threading {n} beads onto a circular string, each bead being one of "
    "{k} colours, if two necklaces are the same when one can be rotated into the other (but not turned over)?",
    "A necklace has {n} beads, each of {k} possible colours. Necklaces that differ only by a rotation are counted as one. "
    "How many different necklaces are there? (Turning a necklace over is not allowed.)",
    "Beads of {k} colours are strung on a loop to make a necklace of {n} beads. A necklace and any rotation of it count as "
    "the same necklace, and necklaces may not be flipped. How many distinct necklaces can be made?",
)


def _easy(rng):
    for _ in attempts(200, "a small necklace count"):
        n, k = rng.randint(4, 11), rng.randint(2, 6)
        if k**n <= 2200:
            break
    group = _group(n, False)
    key, total = _burnside_all(group, k)
    check(key == _count_orbits_all(group, n, k), "Burnside's count differs from counting orbits directly")
    fixed_rows = ", ".join(f"{_fixed_all_colours(g, k)}" for g in group)
    slips = [
        (k**n // n, "burnside-identity-only"),
        (total // (2 * n) if total % (2 * n) == 0 else None, "burnside-divided-by-wrong-group-order"),
        (
            sum(k ** (n // gcd(i, n)) for i in range(n)) // n
            if sum(k ** (n // gcd(i, n)) for i in range(n)) % n == 0
            else None,
            "burnside-fixed-count-uses-cycle-length",
        ),
        (k**n // n + 1, None),
        (key + 1, None),
    ]
    return number_question(
        phrase(rng, _EASY_STEMS, n=n, k=k),
        key,
        [
            f"Rotation through i places has gcd(i, {n}) cycles of beads, so it fixes k^gcd(i, {n}) colourings: {fixed_rows}.",
            f"Burnside: (1/{n}) × {total} = {key}.",
        ],
        slips,
        lo=1,
    )


# -- medium ----------------------------------------------------------------------------------------------------

_MED_STEMS = (
    "How many different bracelets can be made from {n} beads on a circular wire, each bead being one of {k} colours, if two "
    "bracelets are the same when one can be rotated or turned over to match the other?",
    "A bracelet has {n} beads, each of {k} possible colours. Bracelets that differ only by a rotation or a reflection (turning "
    "over) are counted as one. How many different bracelets are there?",
    "Count the bracelets that can be strung from {n} beads, each bead one of {k} colours, where a rotation or a flip of a "
    "bracelet gives the same bracelet.",
)


def _medium(rng):
    for _ in attempts(200, "a small bracelet count"):
        n, k = rng.randint(4, 10), rng.randint(2, 6)
        if k**n <= 2200 and _burnside_all(_group(n, True), k)[0] != _burnside_all(_group(n, False), k)[0]:
            break
    group = _group(n, True)
    key, total = _burnside_all(group, k)
    check(key == _count_orbits_all(group, n, k), "Burnside's count differs from counting orbits directly")
    necklaces, _tot = _burnside_all(_group(n, False), k)
    rot_total = sum(_fixed_all_colours(g, k) for g in _group(n, False))
    refl_total = total - rot_total
    slips = [
        (necklaces, "burnside-reflections-omitted"),
        (total // n if total % n == 0 else None, "burnside-divided-by-wrong-group-order"),
        (
            (rot_total + n * k ** ((n + 1) // 2)) // (2 * n)
            if (rot_total + n * k ** ((n + 1) // 2)) % (2 * n) == 0
            else None,
            "burnside-reflection-fixed-count-uses-one-type",
        ),
        (k**n // (2 * n), "burnside-identity-only"),
        (key + 1, None),
    ]
    return number_question(
        phrase(rng, _MED_STEMS, n=n, k=k),
        key,
        [
            f"The group is D_{n} with {2 * n} elements. The rotations fix {rot_total} colourings in total and the {n} reflections fix {refl_total}.",
            f"Burnside: (1/{2 * n}) × {total} = {key}.",
        ],
        slips,
        lo=1,
    )


# -- hard ------------------------------------------------------------------------------------------------------

_HARD_NECKLACE = (
    "How many different necklaces can be made from {n} beads on a circular string when exactly {m} of the beads are gold and "
    "the rest are silver, if necklaces that differ only by a rotation count as the same?",
    "A necklace has {n} beads, exactly {m} gold and {r} silver. Rotating a necklace does not change it (turning it over is "
    "not allowed). How many different necklaces are there?",
)
_HARD_BRACELET = (
    "How many different bracelets can be made from {n} beads on a circular wire when exactly {m} of the beads are gold and "
    "the rest are silver, if bracelets that differ only by a rotation or a reflection count as the same?",
    "A bracelet has {n} beads, exactly {m} gold and {r} silver. Bracelets that differ by a rotation or by turning over are "
    "counted as one. How many different bracelets are there?",
)


def _hard(rng):
    reflections = rng.random() < 0.5
    for _ in attempts(500, "an exact-count arrangement whose answer differs from the naive ones"):
        n = rng.randint(6, 12)
        m = rng.randint(2, n // 2 + 1)
        group = _group(n, reflections)
        total = sum(_fixed_exact(g, m) for g in group)
        if total % len(group):
            continue
        key = total // len(group)
        naive = comb(n, m) // n if comb(n, m) % n == 0 else None
        if key != naive and key != comb(n, m) // (2 * n if not reflections else n) and key > 1:
            break
    check(key == _count_orbits_exact(group, n, m), "Burnside's count differs from counting orbits directly")
    rotations_only = (
        sum(_fixed_exact(g, m) for g in _group(n, False)) // n
        if sum(_fixed_exact(g, m) for g in _group(n, False)) % n == 0
        else None
    )
    slips = [
        (comb(n, m) // len(group) if comb(n, m) % len(group) == 0 else None, "burnside-identity-only"),
        (rotations_only if reflections else None, "burnside-reflections-omitted"),
        (comb(n, m) // n if comb(n, m) % n == 0 else None, "burnside-identity-only"),
        (_burnside_all(group, 2)[0], "burnside-ignores-exact-count"),
        (total // (2 * len(group)) if total % (2 * len(group)) == 0 else None, "burnside-divided-by-wrong-group-order"),
        (total // n if reflections and total % n == 0 else None, "burnside-divided-by-wrong-group-order"),
        (key - 1, None),
        (key + 1, None),
    ]
    stems = _HARD_BRACELET if reflections else _HARD_NECKLACE
    fixed = [_fixed_exact(g, m) for g in group]
    return number_question(
        phrase(rng, stems, n=n, m=m, r=n - m),
        key,
        [
            f"A symmetry fixes an arrangement only if its {m} gold beads fill whole cycles of the symmetry's bead permutation.",
            f"The {len(group)} symmetries fix {', '.join(map(str, fixed))} arrangements, summing to {total}; {total}/{len(group)} = {key}.",
        ],
        slips,
        lo=1,
    )


@generator(
    "r4a_burnside_counting",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@with_latex
def r4a_burnside_counting(rng, tier="easy"):
    """Counting up to symmetry with Burnside's lemma; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


_EXAM_STEMS = (
    "A bracelet is made from {n} beads on a circular wire, each bead one of {k} colours. Two bracelets are the same if one "
    "can be rotated or turned over to match the other. The symmetry group is D_{n}, of order {g}.",
    "Bracelets are strung from {n} beads, each bead one of {k} colours, and two bracelets count as the same when a rotation or "
    "a flip takes one to the other. Their symmetry group is D_{n}, which has {g} elements.",
    "Each bead of a circular bracelet of {n} beads has one of {k} colours; bracelets that differ by a rotation or by turning "
    "the bracelet over are not counted as different. Let D_{n}, of order {g}, be the group of these symmetries.",
    "A wire ring carries {n} beads, each of one of {k} colours. A rotation of the ring or a flip of it gives the same bracelet, "
    "and these symmetries form the group D_{n} of order {g}.",
)


@generator("exd_x_r4a_burnside_counting", level=19, presets={"exam": dict(tier="exam")})
@with_latex
def exd_x_r4a_burnside_counting(rng, tier="exam"):
    """Exam chain: two fixed counts, then the bracelet and necklace counts that use them."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(100, "a bracelet problem small enough to enumerate"):
        n, k = rng.choice([4, 6, 6, 8, 10, 12]), rng.randint(2, 6)
        if k**n <= 5000:
            break
    rotations, reflections = _d_maps(n)
    two = rotations[2]
    through_vertices = next(g for g in reflections if sum(1 for v in range(n) if g[v] == v) == 2)
    fix_a = _fixed_all_colours(two, k)
    fix_b = _fixed_all_colours(through_vertices, k)
    check(fix_a == k ** gcd(2, n) and fix_b == k ** (n // 2 + 1), "fixed counts differ from the cycle-count formulas")
    group = rotations + reflections
    bracelets, total = _burnside_all(group, k)
    necklaces, rot_total = _burnside_all(rotations, k)
    check(
        bracelets == _count_orbits_all(group, n, k) and necklaces == _count_orbits_all(rotations, n, k),
        "Burnside differs from direct orbit counting",
    )
    # rotations through i places with gcd(i, n) = 2 fix as many as `two`; reflections through two beads fix as many as `through_vertices`
    same_as_a = [i for i in range(n) if gcd(i, n) == gcd(2, n) and i != 0]
    same_as_b = [g for g in reflections if sum(1 for v in range(n) if g[v] == v) == 2]
    const_rot = rot_total - len(same_as_a) * fix_a
    const_refl = total - rot_total - len(same_as_b) * fix_b
    ft_bracelets = f"({const_rot + const_refl} + {len(same_as_a)}*part_a + {len(same_as_b)}*part_b)/{2 * n}"
    ft_necklaces = f"({const_rot} + {len(same_as_a)}*part_a)/{n}"
    check(
        (const_rot + const_refl + len(same_as_a) * fix_a + len(same_as_b) * fix_b) == total
        and (const_rot + len(same_as_a) * fix_a) == rot_total,
        "the follow-through constants do not rebuild the sums",
    )
    return dict(
        text=phrase(rng, _EXAM_STEMS, n=n, k=k, g=2 * n),
        parts=[
            number_part(
                f"How many colourings of the {n} bead positions are fixed by a rotation through two places?",
                fix_a,
                2,
                [
                    f"The rotation splits the beads into {gcd(2, n)} cycles of {n // gcd(2, n)}, and each cycle is one colour: {k}^{gcd(2, n)} = {fix_a}."
                ],
                [
                    (k ** (n // gcd(2, n)), "burnside-fixed-count-uses-cycle-length"),
                    (k * gcd(2, n), None),
                    (k, None),
                    (k**n, None),
                ],
                sensible_range=(0, None),
            ),
            number_part(
                f"How many colourings are fixed by the reflection in the line through two opposite beads?",
                fix_b,
                2,
                [
                    f"The two beads on the line are fixed and the other {n - 2} beads swap in {(n - 2) // 2} pair{'s' if n > 4 else ''}, giving {n // 2 + 1} cycles: {k}^{n // 2 + 1} = {fix_b}."
                ],
                [
                    (k ** (n // 2), "burnside-reflection-fixed-count-uses-one-type"),
                    (k**n, None),
                    (k ** (n // 2 + 2), None),
                    (k * (n // 2), None),
                ],
                sensible_range=(0, None),
            ),
            number_part(
                f"Use Burnside's lemma to find the number of different bracelets. (Fixed counts for the other symmetries are for you to find.)",
                bracelets,
                3,
                [
                    f"Rotations fix {rot_total} colourings in all; reflections through two beads fix {fix_b} each and reflections through two gaps fix {k ** (n // 2)} each.",
                    f"Burnside: (1/{2 * n}) × {total} = {bracelets}.",
                ],
                [
                    (necklaces, "burnside-reflections-omitted"),
                    (total // n if total % n == 0 else None, "burnside-divided-by-wrong-group-order"),
                    (k**n // (2 * n), "burnside-identity-only"),
                    (bracelets - 1, None),
                ],
                follow_through=ft_bracelets,
                sensible_range=(0, None),
            ),
            number_part(
                f"How many different necklaces are there if only rotations are allowed (turning over is not)?",
                necklaces,
                2,
                [f"Burnside over the {n} rotations: (1/{n}) × {rot_total} = {necklaces}."],
                [
                    (bracelets, "burnside-reflections-included-for-necklaces"),
                    (
                        rot_total // (2 * n) if rot_total % (2 * n) == 0 else None,
                        "burnside-divided-by-wrong-group-order",
                    ),
                    (k**n // n, "burnside-identity-only"),
                    (necklaces + 1, None),
                ],
                follow_through=ft_necklaces,
                sensible_range=(0, None),
            ),
        ],
    )
