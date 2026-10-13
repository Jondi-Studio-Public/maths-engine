"""A particle in equilibrium under concurrent forces: `smech_equilibrium` (easy, medium, hard) and
`smech_equilibrium_exam`.

Skill: particle-equilibrium-concurrent-forces. The neighbouring skill on concurrent forces finds a resultant; this one
starts from the fact that there is none. Resolving in two perpendicular directions and setting each sum to zero gives
the unknown forces.

Tiers, each a new decision:

* easy: forces given in components. The unknown is a constant `k` in one component, or the magnitude of the third
  force (found from the negative of the sum of the other two).
* medium: a weight hangs from a slanting string and is pulled aside by a horizontal string. Resolve vertically first
  for the tension in the slanting string, then horizontally for the other.
* hard: two slanting strings at different angles. Both tensions appear in both equations, so the pair must be solved
  together, and the tempting symmetric answer W/(2 sin α) is wrong.
* exam: a weight on a string held aside by a horizontal force. Weight, tension, the horizontal force, then the
  greatest horizontal force before the string breaks (a new idea: the tension grows as the string is pulled aside).

Slopes use the 3-4-5, 5-12-13 and 4-3-5 triangles, so sines and cosines are exact. g = 9.8 m/s². Every answer is a
chosen exact `Fraction` shown at the stated places, kept only when it needs no rounding (the broken-string force is
the one rounded value); the sums of forces are re-added to zero as a second check.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, draw_until, generator, paren_negative
from ._smech_kit import G, G_NOTE, clear_of_boundary, dec, is_exact, number_fields, number_part

_TRIANGLES = (
    (Fraction(3, 5), Fraction(4, 5)),
    (Fraction(4, 5), Fraction(3, 5)),
    (Fraction(5, 13), Fraction(12, 13)),
    (Fraction(12, 13), Fraction(5, 13)),
)
_PAIRS = (  # (sin, cos) of the two string angles, both measured from the horizontal
    ((Fraction(3, 5), Fraction(4, 5)), (Fraction(5, 13), Fraction(12, 13))),
    ((Fraction(4, 5), Fraction(3, 5)), (Fraction(12, 13), Fraction(5, 13))),
    ((Fraction(3, 5), Fraction(4, 5)), (Fraction(4, 5), Fraction(3, 5))),
    ((Fraction(5, 13), Fraction(12, 13)), (Fraction(12, 13), Fraction(5, 13))),
)
_THINGS = ("lamp", "sign", "flower basket", "speaker", "banner")


def _frac(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


# ------------------------------------------------------------------------------------------------ easy


def _easy(rng):
    # Choose the third force first so its magnitude is whole (a Pythagorean pair), then two known forces that leave it.
    px, py = rng.choice(((3, 4), (4, 3), (6, 8), (8, 6), (5, 12), (12, 5), (9, 12), (12, 9)))
    px, py = px * rng.choice((-1, 1)), py * rng.choice((-1, 1))
    f1x, f1y = rng.randint(-8, 8), rng.randint(-8, 8)
    f2x, f2y = -px - f1x, -py - f1y  # so f1 + f2 + p = 0
    check((f1x + f2x + px, f1y + f2y + py) == (0, 0), "the forces do not sum to zero")
    if rng.random() < 0.5:
        # an unknown constant k in one component of the second force
        k_slot = rng.choice(("x", "y"))
        k = f2x if k_slot == "x" else f2y
        f2 = f"({'k' if k_slot == 'x' else f2x}, {'k' if k_slot == 'y' else f2y})"
        text = (
            f"A particle is in equilibrium under the three forces F1 = ({f1x}, {f1y}) N, F2 = {f2} N and "
            f"F3 = ({px}, {py}) N, where k is a constant. Find k."
        )
        if k_slot == "x":
            wrong = [
                (f1x + px, "mech-sign-error-in-balance"),
                (-f1x, "mech-ignored-one-force"),
                (-px, "mech-ignored-one-force"),
                (f1y + py, "mech-wrong-component"),
            ]
            working = [f"East: {f1x} + k + ({px}) = 0, so k = {k}."]
        else:
            wrong = [
                (f1y + py, "mech-sign-error-in-balance"),
                (-f1y, "mech-ignored-one-force"),
                (-py, "mech-ignored-one-force"),
                (f1x + px, "mech-wrong-component"),
            ]
            working = [f"North: {f1y} + k + ({py}) = 0, so k = {k}."]
        return dict(text=text, working=working, **number_fields(Fraction(k), 0, [(Fraction(v), t) for v, t in wrong]))
    text = (
        f"Three forces act at a point and keep a particle in equilibrium: F1 = ({f1x}, {f1y}) N, "
        f"F2 = ({f2x}, {f2y}) N and an unknown force P. Find the magnitude of P, in newtons."
    )
    mag = int(math.hypot(px, py))
    check(mag * mag == px * px + py * py, "|P| is not whole")
    wrong = [
        (Fraction(px * px + py * py), "mech-forgot-square-root"),
        (Fraction(abs(px) + abs(py)), "mech-added-components"),
        (Fraction(abs(f1x) + abs(f1y)), "mech-added-components"),
        (Fraction(abs(px)), "mech-used-one-component"),
    ]
    working = [
        f"F1 + F2 = ({f1x + f2x}, {f1y + f2y}) N, and the three forces sum to zero, so P = ({px}, {py}) N.",
        f"|P| = √({paren_negative(px)}² + {paren_negative(py)}²) = {mag} N.",
    ]
    return dict(text=text, working=working, **number_fields(Fraction(mag), 0, wrong))


# ------------------------------------------------------------------------------------------------ medium


def _medium(rng):
    thing = rng.choice(_THINGS)
    sin, cos = rng.choice(_TRIANGLES)
    m = draw_until(
        rng,
        lambda r: r.randint(2, 30),
        lambda x: is_exact(x * G / sin, 2) and is_exact(x * G * cos / sin, 2),
        what="a mass with exact tensions",
    )
    weight = m * G
    slanting, horizontal = weight / sin, weight * cos / sin
    # Second route: the three forces sum to zero.
    check(slanting * sin == weight and slanting * cos == horizontal, "forces do not balance")
    ask = rng.choice(("slanting", "horizontal"))
    base = (
        f"A {thing} of mass {m} kg hangs at rest from a point P. P is held by a light string fixed to a ceiling, "
        f"which makes an angle α with the horizontal where sin α = {_frac(sin)}, and by a light horizontal string "
        f"fixed to a wall. {G_NOTE}"
    )
    if ask == "slanting":
        text = base + " Find the tension in the slanting string, in newtons, correct to 2 decimal places."
        value = slanting
        wrong = [
            (weight * sin, "mech-sin-cos-swapped"),
            (weight / cos, "mech-sin-cos-swapped"),
            (weight, "mech-tension-equals-weight"),
            (horizontal, "mech-strings-swapped"),
        ]
        working = [
            f"Vertically: T sin α = {m}g = {dec(weight)}.",
            f"T = {dec(weight)} / ({_frac(sin)}) = {dec(slanting)} N.",
        ]
    else:
        text = base + " Find the tension in the horizontal string, in newtons, correct to 2 decimal places."
        value = horizontal
        wrong = [
            (slanting, "mech-strings-swapped"),
            (weight * cos, "mech-sin-cos-swapped"),
            (weight * sin / cos, "mech-tan-inverted-in-balance"),
            (weight, "mech-tension-equals-weight"),
        ]
        working = [
            f"Vertically: T₁ sin α = {m}g, so T₁ = {dec(slanting)} N.",
            f"Horizontally: T₂ = T₁ cos α = {dec(slanting)} × {_frac(cos)} = {dec(horizontal)} N.",
        ]
    return dict(text=text, working=working, **number_fields(value, 2, wrong))


# ------------------------------------------------------------------------------------------------ hard


def _solve_two_strings(weight, a, b):
    """Tensions (left, right) of two strings at angles with (sin, cos) `a` and `b` to the horizontal."""
    sin_sum = a[0] * b[1] + a[1] * b[0]
    return weight * b[1] / sin_sum, weight * a[1] / sin_sum


def _hard(rng):
    thing = rng.choice(_THINGS)
    a, b = rng.choice(_PAIRS)
    m = draw_until(
        rng,
        lambda r: r.randint(2, 40),
        lambda x: all(is_exact(t, 2) for t in _solve_two_strings(x * G, a, b)),
        what="a mass with exact tensions in both strings",
    )
    weight = m * G
    t1, t2 = _solve_two_strings(weight, a, b)
    # Second route: horizontal and vertical balances.
    check(t1 * a[1] == t2 * b[1] and t1 * a[0] + t2 * b[0] == weight, "the two strings do not balance the weight")
    ask = rng.choice(("left", "right"))
    base = (
        f"A {thing} of mass {m} kg hangs at rest from a point P on two light strings. One string runs to a "
        f"ceiling hook A on the left and makes an angle α with the horizontal, where sin α = {_frac(a[0])}. The "
        f"other runs to a hook B on the right and makes an angle β with the horizontal, where sin β = {_frac(b[0])}. "
        f"{G_NOTE}"
    )
    if ask == "left":
        text = base + " Find the tension in the string PA, in newtons, correct to 2 decimal places."
        value = t1
        wrong = [
            (weight / (2 * a[0]), "mech-assumed-symmetric"),
            (weight / a[0], "mech-other-string-ignored"),
            (t2, "mech-strings-swapped"),
            (weight / (a[0] + b[0]), "mech-horizontal-balance-skipped"),
        ]
        which = "T₁"
    else:
        text = base + " Find the tension in the string PB, in newtons, correct to 2 decimal places."
        value = t2
        wrong = [
            (weight / (2 * b[0]), "mech-assumed-symmetric"),
            (weight / b[0], "mech-other-string-ignored"),
            (t1, "mech-strings-swapped"),
            (weight / (a[0] + b[0]), "mech-horizontal-balance-skipped"),
        ]
        which = "T₂"
    working = [
        "Horizontally: T₁ cos α = T₂ cos β.",
        f"Vertically: T₁ sin α + T₂ sin β = {m}g = {dec(weight)}.",
        f"Solving together: T₁ = {dec(t1)} N and T₂ = {dec(t2)} N, so {which} = {dec(value)} N.",
    ]
    return dict(text=text, working=working, **number_fields(value, 2, wrong))


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("smech_equilibrium", level=17, presets={t: {"tier": t} for t in _TIERS})
def smech_equilibrium(rng, tier: str = "easy"):
    """Equilibrium of a particle: balance the components, then a hanging weight, then two slanting strings."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# ------------------------------------------------------------------------------------------------ exam


@generator("smech_equilibrium_exam", level=17, presets={"exam": {}})
def smech_equilibrium_exam(rng):
    """A weight on a string pulled aside by a horizontal force: weight, tension, force, greatest force before the string breaks."""
    thing = rng.choice(_THINGS)
    for _ in attempts(2000, "a hanging weight with exact tension and force"):
        sin, cos = rng.choice(_TRIANGLES)  # the string's angle θ from the vertical: sin θ, cos θ
        m = rng.randint(3, 30)
        weight = m * G
        tension, force = weight / cos, weight * sin / cos
        limit = Fraction(round(float(tension * Fraction(3, 2)) / 10) * 10)
        if limit <= tension + 5:
            continue
        h_max = math.sqrt(float(limit * limit - weight * weight))
        if (
            is_exact(tension, 2)
            and is_exact(force, 2)
            and clear_of_boundary(Fraction(h_max).limit_denominator(10**9), 1)
        ):
            break
    check(tension * cos == weight and tension * sin == force, "the three forces do not balance")
    check(abs(h_max**2 + float(weight) ** 2 - float(limit) ** 2) < 1e-6, "greatest force does not square back")
    tan_text = _frac(sin / cos)
    stem = (
        f"A {thing} of mass {m} kg hangs from a light string fixed to a ceiling. A horizontal force H holds it "
        f"in equilibrium with the string at an angle θ to the vertical, where tan θ = {tan_text}. {G_NOTE}"
    )
    part_a = number_part(
        f"Find the weight of the {thing}, in newtons, correct to 2 decimal places.",
        weight,
        2,
        1,
        [f"W = {m} × {dec(G)} = {dec(weight)} N."],
        [(Fraction(m), "mech-forgot-g"), (weight * G, "mech-g-applied-twice"), (weight / 2, "mech-halved-weight")],
        sensible=(0, None),
    )
    part_b = number_part(
        "Find the tension in the string, in newtons, correct to 2 decimal places.",
        tension,
        2,
        2,
        [f"Vertically: T cos θ = W, so T = {dec(weight)} / ({_frac(cos)}) = {dec(tension)} N."],
        [
            (weight * cos, "mech-sin-cos-swapped"),
            (weight / sin, "mech-sin-cos-swapped"),
            (weight, "mech-tension-equals-weight"),
            (force, "mech-strings-swapped"),
        ],
        follow=f"part_a * {cos.denominator} / {cos.numerator}",
        sensible=(0, None),
    )
    part_c = number_part(
        "Find the magnitude of the horizontal force H, in newtons, correct to 2 decimal places.",
        force,
        2,
        2,
        [f"Horizontally: H = T sin θ = {dec(tension)} × {_frac(sin)} = {dec(force)} N."],
        [
            (weight * cos / sin, "mech-tan-inverted-in-balance"),
            (weight * sin, "mech-sin-cos-swapped"),
            (tension, "mech-resultant-for-unknown"),
            (weight, "mech-tension-equals-weight"),
        ],
        follow=f"part_a * {(sin / cos).numerator} / {(sin / cos).denominator}",
        sensible=(0, None),
    )
    part_d = number_part(
        f"The string breaks if its tension exceeds {limit} N. The force H is increased slowly, so the {thing} "
        f"stays in equilibrium. Find the greatest value H can take before the string breaks, in newtons, "
        f"correct to 1 decimal place.",
        Fraction(h_max).limit_denominator(10**9),
        1,
        2,
        [f"T² = W² + H², so H² = {limit}² - {dec(weight)}².", f"H = {round(h_max, 1)} N."],
        [
            (limit - weight, "mech-subtracted-instead-of-pythagoras"),
            (
                Fraction(math.sqrt(float(limit * limit + weight * weight))).limit_denominator(10**9),
                "mech-added-squares",
            ),
            (limit * sin / cos, "mech-tan-inverted-in-balance"),
            (limit * sin, "mech-limit-resolved-as-fixed-angle"),
        ],
        follow=f"sqrt({limit}**2 - part_a**2)",
        sensible=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
