"""Generates `exd_x_r3j_residue_contour` questions: residues at three simple poles, then a contour integral.

Skill: `ucv1-residue-simple-pole` (exam tier). The practice tiers take one residue of c / ((z - a)(z - b)); a
first-year complex analysis exam instead gives a rational function with three simple poles and a numerator that
is not constant, asks for residues, and then uses them in the residue theorem:

(a), (b), (c) the residues at the three poles, in increasing order of the pole     fraction, 1 mark each
(d) k with (integral of f round the circle C) = 2 pi i k                           fraction, 2 marks,
    follow_through = the sum of the residues at the poles inside C

## Construction

The poles are three distinct integers and the numerator n1 z + n0 is non-zero at each of them, so all three poles
are genuinely simple. The circle |z - c0| = r + 1/2 has an integer centre and half-integer radius, so no pole
lies on it, and it is drawn so that exactly two poles are inside (with one, (d) would repeat that pole's residue).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, fmt_signed, generator, join_terms
from ._exd_x_r3_distinct import distinct_answers
from ._exd_x_r3j_kit import fraction_part


def _factor(pole: int) -> str:
    if pole == 0:
        return "z"
    return f"(z {'-' if pole > 0 else '+'} {abs(pole)})"


def _numerator(n2: int, n1: int, n0: int) -> str:
    return join_terms([(n2, "z^2"), (n1, "z"), (n0, "")])


def _at(n2: int, n1: int, n0: int, p: int) -> int:
    return n2 * p * p + n1 * p + n0


def _res(coefficients, poles, i) -> Fraction:
    p = poles[i]
    denominator = 1
    for j, q in enumerate(poles):
        if j != i:
            denominator *= p - q
    return Fraction(_at(*coefficients, p), denominator)


@generator("exd_x_r3j_residue_contour", level=17, presets={"exam": {}})
@distinct_answers
def exd_x_r3j_residue_contour(rng):
    for _ in attempts(2000, "three poles, a numerator and a circle separating them"):
        poles = sorted(rng.sample(range(-6, 7), 3))
        n2 = rng.choice([0, 0, 0, 1, 1, -1, 2])
        n1, n0 = rng.randint(-4, 4), rng.randint(-9, 9)
        if n2 == 0 and n1 == 0 and rng.random() < 0.7:
            continue
        if any(_at(n2, n1, n0, p) == 0 for p in poles):
            continue
        c0 = rng.randint(-3, 3)
        radius = Fraction(rng.randint(1, 8)) + Fraction(1, 2)
        inside = [i for i, p in enumerate(poles) if abs(p - c0) < radius]
        if len(inside) != 2:
            continue  # one pole inside would make (d) repeat that pole's residue
        residues = [_res((n2, n1, n0), poles, i) for i in range(3)]
        if sum(residues[i] for i in inside) != 0:
            break

    coefficients = (n2, n1, n0)
    residues = [_res(coefficients, poles, i) for i in range(3)]
    check(all(r != 0 for r in residues), "a pole is removable")
    total_inside = sum(residues[i] for i in inside)
    outside = [i for i in range(3) if i not in inside]
    total_outside = sum(residues[i] for i in outside)
    check(total_inside != 0, "the contour integral is zero")

    denominator_text = "".join(_factor(p) for p in poles)
    numerator = _numerator(n2, n1, n0)
    single_term = sum(1 for c in coefficients if c) == 1
    f_text = (
        f"f(z) = {numerator} / ({denominator_text})" if single_term else f"f(z) = ({numerator}) / ({denominator_text})"
    )
    pole_list = f"{poles[0]}, {poles[1]} and {poles[2]}"
    centre = "z" if c0 == 0 else f"z {'-' if c0 > 0 else '+'} {abs(c0)}"
    radius_text = f"{radius.numerator / radius.denominator:g}"
    stem = (
        f"The function {f_text} has simple poles at z = {pole_list}. "
        f"Let C be the circle |{centre}| = {radius_text}, traversed once anticlockwise."
    )

    def slips(i: int) -> list:
        p = poles[i]
        others = [q for j, q in enumerate(poles) if j != i]
        wrong_pole = poles[(i + 1) % 3]
        out = [
            (Fraction(1, (p - others[0]) * (p - others[1])), "ignored-numerator"),
            (Fraction(_at(*coefficients, wrong_pole), (p - others[0]) * (p - others[1])), "numerator-at-wrong-pole"),
            (-residues[i], "sign-slip-in-formula"),
            (Fraction(_at(*coefficients, p), p - others[0]), "dropped-denominator-factor"),
            (Fraction(_at(*coefficients, p), p - others[1]), "dropped-denominator-factor"),
        ]
        return out

    def res_part(i, marks):
        p = poles[i]
        others = [q for j, q in enumerate(poles) if j != i]
        return fraction_part(
            f"Find Res(f, {p}).",
            residues[i],
            marks,
            [
                f"Res(f, {p}) = N({p}) / (({p}) − ({others[0]}))(({p}) − ({others[1]})), where N({p}) = {_at(*coefficients, p)}; "
                f"so Res(f, {p}) = {residues[i]}.",
            ],
            slips(i),
        )

    inside_names = " + ".join(f"part_{'abc'[i]}" for i in inside)
    d_slips = [
        (sum(residues), "included-pole-outside-contour"),
        (total_outside, "included-pole-outside-contour"),
        (-total_inside, "sign-of-result-reversed"),
    ]
    d_slips += [(residues[i], "omitted-entry-in-total") for i in inside] if len(inside) == 2 else []
    d_slips += [(total_inside * 2, "dropped-factor-of-two")]

    parts = [
        res_part(0, 1),
        res_part(1, 1),
        res_part(2, 1),
        fraction_part(
            "Find the number k for which the integral of f(z) round C equals 2πik.",
            total_inside,
            2,
            [
                f"Only the poles at z = {', '.join(str(poles[i]) for i in inside)} lie inside C.",
                f"By the residue theorem k is the sum of their residues, {total_inside}.",
            ],
            d_slips,
            follow_through=inside_names,
        ),
    ]
    return dict(text=stem, parts=parts)
