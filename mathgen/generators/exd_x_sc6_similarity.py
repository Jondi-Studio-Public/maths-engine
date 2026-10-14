"""Exam-tier source for `senior-review-the-conditions-for-similarity-of-two-dimensional`.

Instead of classifying which condition applies, the learner uses a stated similarity condition to find an unknown
measurement and then carries the scale factor through. Three shapes, chosen per seed:

* ``sas``: two sides and the included angle are given for each triangle, with one side unknown. The sides must be in
  proportion, which fixes the unknown side.
* ``sss``: all three sides of one triangle and two sides of the other are given; the third side is the one that makes
  the three ratios equal.
* ``aa``: two angles of one triangle and two angles of the other are given, with one unknown; the learner finds the
  missing angle, then a scale factor from a pair of corresponding sides, then another side.

Every shape has three connected parts: (a) the missing measurement, (b) the scale factor, (c) a derived quantity
(the area, the perimeter, or another side). (b) and (c) follow through.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from math import gcd

from mathgen.answers import Distractor
from mathgen.core import (
    check,
    draw_until,
    exact_decimal_string,
    generator,
    latex_answer,
    pick_distractors,
    round_half_up,
    rounded_decimal_string,
    stated_precision_string,
)

_NAMES = [("PQR", "XYZ"), ("ABC", "DEF"), ("KLM", "RST"), ("GHJ", "UVW")]
# Scale factors p/q (larger over smaller) that terminate as decimals, so every answer is exact.
_SCALES = [(p, q) for q in (2, 4, 5) for p in range(q + 1, 3 * q + 1) if gcd(p, q) == 1]


def _dec(value: Fraction) -> str:
    return exact_decimal_string(value, 2)


def _dec_up(value: Fraction) -> str:
    return stated_precision_string(value, 2)


def _cand(value: Fraction, misconception: str) -> Distractor | None:
    if value <= 0:
        return None
    text = rounded_decimal_string(value, 2)
    return Distractor(text, misconception, Fraction(text))


def _tag(true: Fraction, cands, dp: int | None = None) -> list[Distractor]:
    cands = [c for c in cands if c is not None]
    key = stated_precision_string(true, dp) if dp else _dec(true)
    picked = pick_distractors(key, cands, 3)
    if dp:  # every option is written to the key's decimal places
        picked = [Distractor(stated_precision_string(Fraction(d.value), dp), d.misconception, d.value) for d in picked]
    check(len(picked) >= 2, "too few distractors")
    check(all(Fraction(d.value) != true for d in picked), "a distractor equals the key")
    return picked


def _part(prompt, value: Fraction, marks, dp: int | None = None, **extra):
    # A part that later parts follow through from states its precision, so a carried-forward value is marked at it.
    text = stated_precision_string(value, dp) if dp else _dec(value)
    return dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(value) if value.denominator == 1 else float(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        sensible_range=(0, None),
        **extra,
    )


def _area_for(rng: random.Random, p: int, q: int) -> int:
    """A whole-number area for the smaller triangle whose scaled area has at most 2 decimal places."""
    options = [a for a in range(12, 91) if (Fraction(a * p * p, q * q) * 100).denominator == 1]
    return rng.choice(options)


def _sas(rng: random.Random):
    n1, n2 = rng.choice(_NAMES)
    p, q = rng.choice(_SCALES)
    u, w = draw_until(rng, lambda r: (r.randint(1, 4), r.randint(1, 5)), lambda uw: uw[0] != uw[1])
    s1, s2 = q * u, q * w  # sides of the smaller triangle
    t1, t2 = p * u, p * w  # corresponding sides of the larger one
    area = _area_for(rng, p, q)
    k = Fraction(p, q)
    stem = (
        f"Triangles {n1} and {n2} are similar by the SAS condition, with {n1[0]} corresponding to {n2[0]}, "
        f"{n1[1]} to {n2[1]} and {n1[2]} to {n2[2]}. The angles at {n1[0]} and {n2[0]} are equal. "
        f"{n1[0]}{n1[1]} = {s1} cm, {n1[0]}{n1[2]} = {s2} cm, {n2[0]}{n2[1]} = {t1} cm and "
        f"{n2[0]}{n2[2]} = x cm. Triangle {n1} has an area of {area} cm²."
    )
    unknown = Fraction(t2)
    # Independent route: cross-multiplication of the two ratios.
    check(Fraction(t1, s1) == Fraction(unknown, s2), "sides not in proportion")
    a_cands = [
        _cand(Fraction(s2 + t1 - s1), "additive-scaling"),
        _cand(Fraction(s2 * s1, t1), "inverted-ratio"),
        _cand(Fraction(t1 * s1, s2), "cross-multiplies-wrongly"),
        _cand(Fraction(s2 * t1, s1 + 1) if s1 + 1 else None, "cross-multiplies-wrongly"),
    ]
    b_cands = [
        _cand(Fraction(q, p), "inverted-ratio"),
        _cand(Fraction(t2 - s2, s2) if t2 > s2 else Fraction(0), "difference-not-ratio"),
        _cand(k * k, "uses-area-ratio-as-scale-factor"),
    ]
    c_cands = [
        _cand(area * k, "area-scaled-by-linear-factor"),
        _cand(area / (k * k), "inverted-ratio"),
        _cand(area + (t1 - s1), "additive-scaling"),
    ]
    parts = [
        _part(
            f"Find the value of x, the length of {n2[0]}{n2[2]} in centimetres.",
            unknown,
            2,
            working=[
                f"The sides about the equal angle are in proportion: x/{s2} = {t1}/{s1}.",
                f"x = {_dec(unknown)}.",
            ],
            distractors=_tag(unknown, a_cands),
        ),
        _part(
            f"Hence find the scale factor of the enlargement from triangle {n1} to triangle {n2}, correct to 2 decimal places.",
            k,
            1,
            dp=2,
            follow_through=f"part_a / {s2}",
            working=[f"Scale factor = {n2[0]}{n2[2]} ÷ {n1[0]}{n1[2]} = {_dec(unknown)} ÷ {s2} = {_dec(k)}."],
            distractors=_tag(k, b_cands, 2),
        ),
        _part(
            f"Find the area of triangle {n2}, in cm², correct to 2 decimal places.",
            area * k * k,
            2,
            dp=2,
            follow_through=f"part_b**2 * {area}",
            working=[f"Areas scale by the square of the scale factor: {area} × {_dec(k)}² = {_dec(area * k * k)}."],
            distractors=_tag(area * k * k, c_cands, 2),
        ),
    ]
    return stem, parts


def _sss(rng: random.Random):
    n1, n2 = rng.choice(_NAMES)
    p, q = rng.choice(_SCALES)
    for _ in range(1000):
        t = sorted(rng.sample(range(2, 8), 3))
        if t[0] + t[1] > t[2]:
            break
    else:
        raise AssertionError("no triangle in 1000 draws")
    s = [q * v for v in t]
    big = [p * v for v in t]
    j = rng.randrange(3)  # which side of the larger triangle is unknown
    sides1 = [f"{n1[i]}{n1[(i + 1) % 3]}" for i in range(3)]
    sides2 = [f"{n2[i]}{n2[(i + 1) % 3]}" for i in range(3)]
    known = [f"{sides2[i]} = {big[i]} cm" for i in range(3) if i != j]
    k = Fraction(p, q)
    stem = (
        f"Triangle {n1} has sides {sides1[0]} = {s[0]} cm, {sides1[1]} = {s[1]} cm and {sides1[2]} = {s[2]} cm. "
        f"Triangle {n2} has {known[0]} and {known[1]}, and {sides2[j]} = y cm. "
        "The triangles are similar by the SSS condition, with the sides corresponding in the order written."
    )
    y = Fraction(big[j])
    check(all(Fraction(big[i], s[i]) == k for i in range(3)), "ratios differ")
    perimeter = sum(s)
    a_cands = [
        _cand(Fraction(s[j] + big[(j + 1) % 3] - s[(j + 1) % 3]), "additive-scaling"),
        _cand(Fraction(s[j] * s[(j + 1) % 3], big[(j + 1) % 3]), "inverted-ratio"),
        _cand(Fraction(s[j] * q, p), "inverted-ratio"),
        _cand(Fraction(big[(j + 2) % 3]), "uses-wrong-corresponding-side"),
    ]
    b_cands = [
        _cand(Fraction(q, p), "inverted-ratio"),
        _cand(Fraction(big[j] - s[j], s[j]), "difference-not-ratio"),
        _cand(k * k, "uses-area-ratio-as-scale-factor"),
    ]
    c_cands = [
        _cand(Fraction(perimeter), "gave-perimeter-of-original"),
        _cand(perimeter * k * k, "perimeter-scaled-by-k-squared"),
        _cand(perimeter + (big[j] - s[j]), "additive-scaling"),
    ]
    parts = [
        _part(
            f"Find the value of y, the length of {sides2[j]} in centimetres.",
            y,
            2,
            working=[f"Corresponding sides are in the same ratio: y/{s[j]} = {big[(j + 1) % 3]}/{s[(j + 1) % 3]}."],
            distractors=_tag(y, a_cands),
        ),
        _part(
            f"Hence find the scale factor from triangle {n1} to triangle {n2}, correct to 2 decimal places.",
            k,
            1,
            dp=2,
            follow_through=f"part_a / {s[j]}",
            working=[f"Scale factor = {_dec(y)} ÷ {s[j]} = {_dec(k)}."],
            distractors=_tag(k, b_cands, 2),
        ),
        _part(
            f"Find the perimeter of triangle {n2}, in centimetres, correct to 2 decimal places.",
            perimeter * k,
            2,
            dp=2,
            follow_through=f"part_b * {perimeter}",
            working=[f"Perimeters scale by the scale factor: {perimeter} × {_dec(k)} = {_dec(perimeter * k)}."],
            distractors=_tag(perimeter * k, c_cands, 2),
        ),
    ]
    return stem, parts


def _aa(rng: random.Random):
    n1, n2 = rng.choice(_NAMES)
    p, q = rng.choice(_SCALES)
    # The figure must exist, so the second given side comes from the sine rule: BC = AB sin A / sin C, shown to 1 decimal
    # place. Every later key is computed from the printed BC. No two angles are equal, so nothing is read off for free.
    for _ in range(1000):
        a_ang, b_ang = rng.randint(30, 80), rng.randint(35, 85)
        c_ang = 180 - a_ang - b_ang
        if len({a_ang, b_ang, c_ang}) < 3 or not 20 <= c_ang <= 110:
            continue
        ab = q * rng.randint(2, 5)
        exact_bc = ab * math.sin(math.radians(a_ang)) / math.sin(math.radians(c_ang))
        if abs(exact_bc * 10 % 1 - 0.5) < 1e-6:
            continue
        bc = round_half_up(Fraction(exact_bc), 1)
        ef_exact = Fraction(p, q) * bc
        # keep (c) clear of a rounding tie, and the sides distinct from each other and from the scale partner
        if bc == ab or (ef_exact * 100) % 1 == Fraction(1, 2) or bc <= 0:
            continue
        break
    else:
        raise AssertionError("no consistent angle figure in 1000 draws")
    de = p * ab // q
    k = Fraction(p, q)
    ef = round_half_up(ef_exact, 2)
    bc_text = exact_decimal_string(bc, 1)
    s1, s2 = f"{n1[0]}{n1[1]}", f"{n1[1]}{n1[2]}"
    t1, t2 = f"{n2[0]}{n2[1]}", f"{n2[1]}{n2[2]}"
    stem = (
        f"In triangle {n1}, ∠{n1[0]} = {a_ang}° and ∠{n1[1]} = {b_ang}°. In triangle {n2}, ∠{n2[0]} = {a_ang}° and "
        f"∠{n2[2]} = θ. The triangles are similar, with {n1[0]} corresponding to {n2[0]}, {n1[1]} to {n2[1]} and "
        f"{n1[2]} to {n2[2]}. Also {s1} = {ab} cm, {s2} = {bc_text} cm and {t1} = {de} cm."
    )
    theta = Fraction(c_ang)
    a_cands = [
        _cand(Fraction(180 - a_ang), "leaves-an-angle-out-of-the-sum"),
        _cand(Fraction(b_ang), None),
        _cand(Fraction(180 - b_ang), "leaves-an-angle-out-of-the-sum"),
        _cand(Fraction(a_ang + b_ang), None),
    ]
    b_cands = [
        _cand(Fraction(q, p), "inverted-ratio"),
        _cand(Fraction(de - ab, ab), "difference-not-ratio"),
        _cand(k * k, "uses-area-ratio-as-scale-factor"),
    ]
    c_cands = [
        _cand(Fraction(bc + de - ab), "additive-scaling"),
        _cand(Fraction(bc * q, p), "inverted-ratio"),
        _cand(bc * k * k, None),
    ]
    parts = [
        _part(
            "Find the size of θ, in degrees.",
            theta,
            1,
            working=[f"The angles of a triangle sum to 180°: θ = 180 − {a_ang} − {b_ang} = {c_ang}°."],
            distractors=_tag(theta, a_cands),
        ),
        _part(
            f"Find the scale factor from triangle {n1} to triangle {n2}, correct to 2 decimal places.",
            k,
            1,
            dp=2,
            working=[f"Scale factor = {t1} ÷ {s1} = {de} ÷ {ab} = {_dec(k)}."],
            distractors=_tag(k, b_cands, 2),
        ),
        _part(
            f"Hence find the length of {t2}, in centimetres, correct to 2 decimal places.",
            ef,
            2,
            dp=2,
            follow_through=f"part_b * {bc_text}",
            working=[f"{t2} = {_dec(k)} × {bc_text} = {_dec_up(ef)}."],
            distractors=_tag(ef, c_cands, 2),
        ),
    ]
    return stem, parts


_SHAPES = [_sas, _sss, _aa]


@generator("exd_x_sc6_similarity", level=10, presets={"exam": dict(tier="exam")})
def exd_x_sc6_similarity(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    stem, parts = rng.choice(_SHAPES)(rng)
    return dict(text=stem, parts=parts)
