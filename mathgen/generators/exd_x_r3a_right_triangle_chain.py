"""Exam source of `senior-review-sine-cosine-and-tangent-as-ratios-of`: one ratio and one side fix a whole triangle.

The practice tiers give a ratio and ask for another ratio. An exam question gives one ratio of an angle and ONE
side length of a right-angled triangle, and asks the learner to carry that through the whole triangle:

(a) the other ratio the learner needs (sine of A), from the given cosine or tangent;
(b) the hypotenuse, from the known side and the ratio in (a);
(c) the remaining leg, by Pythagoras on (b);
(d) the length of the perpendicular from the right angle to the hypotenuse, from (b) and (c).

Every later part follows through from the earlier ones, and every value is an exact fraction.

## Construction

Built backwards from a primitive Pythagorean triple (o, a, h), a scale k and a choice of which ratio is given.
The known side is the side opposite angle A, o*k cm. An independent route rebuilds the triangle from the given
ratio and side with SymPy and compares all four answers.
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Rational, Symbol, solve, sqrt

from mathgen.answers import Distractor, exact_value
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors, rational_text
from ._exd_distinct_parts import distinct_part_answers

_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (9, 40, 41), (12, 35, 37), (11, 60, 61)]
_WORLD = [
    "Triangle ABC has a right angle at B",
    "In triangle ABC, the angle at B is a right angle",
    "A steel brace is welded to make triangle ABC, right-angled at B",
]


def _t(value: Fraction) -> str:
    return rational_text(value)


def _fraction_part(prompt, value, marks, distractors, *, follow_through=None, working=(), rng_range=(0, None)):
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=_t(value),
        answer_latex=latex_answer(_t(value)),
        marks=marks,
        sensible_range=rng_range,
        working=list(working),
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _tag(texts: list[tuple[Fraction, str]], answer: Fraction, n: int = 3) -> list[Distractor]:
    cands = [Distractor(_t(v), mid, v) for v, mid in texts if v > 0]
    kept = pick_distractors(_t(answer), cands, n)
    return kept


@generator("exd_x_r3a_right_triangle_chain", level=13, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3a_right_triangle_chain(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a triangle whose keys do not appear in the prompt"):
        x, y, h = rng.choice(_TRIPLES)
        o, a = (x, y) if rng.random() < 0.5 else (y, x)
        k = rng.randint(1, 10)
        given = rng.choice(["tan", "cos"])
        given_value = Fraction(o, a) if given == "tan" else Fraction(a, h)
        length = o * k
        keys = {_t(Fraction(o, h)), str(h * k), str(a * k), _t(Fraction(a * o * k, h))}
        shown = {str(length), _t(given_value)}
        if keys & shown:
            continue
        break

    sin_a, cos_a, tan_a = Fraction(o, h), Fraction(a, h), Fraction(o, a)
    ac, ab, bc = Fraction(h * k), Fraction(a * k), Fraction(length)
    bd = ab * bc / ac

    # Independent route: rebuild the triangle with SymPy from the given ratio and side alone.
    s_, c_ = Symbol("s", positive=True), Symbol("c", positive=True)
    gv = Rational(given_value.numerator, given_value.denominator)
    eqs = [c_ - gv, s_**2 + c_**2 - 1] if given == "cos" else [s_ - gv * c_, s_**2 + c_**2 - 1]
    (sol,) = solve(eqs, [s_, c_], dict=True)
    s_val = sol[s_]
    hyp = Rational(length) / s_val
    leg = sqrt(hyp**2 - length**2)
    alt = leg * length / hyp
    check(
        s_val == Rational(o, h) and hyp == h * k and leg == a * k and alt == Rational(a * o * k, h), "SymPy disagrees"
    )

    # (a) sin A: the cosine, the reciprocal, the tangent, the cotangent.
    a_d = _tag(
        [
            (cos_a, "sine-cosine-swapped"),
            (Fraction(h, o), "reciprocal-ratio"),
            (tan_a, "wrong-ratio-for-sine"),
            (Fraction(a, o), "reciprocal-ratio"),
        ],
        sin_a,
    )
    # (b) AC: BC times sin A, BC over cos A, the adjacent leg.
    b_d = _tag(
        [
            (bc * sin_a, "ratio-multiplied-not-divided"),
            (bc / cos_a, "sine-cosine-swapped"),
            (ab, "other-leg-for-hypotenuse"),
        ],
        ac,
    )
    # (c) AB: AC - BC, AC + BC, BC.
    c_d = _tag(
        [
            (ac - bc if ac > bc else bc - ac, "legs-subtracted-for-pythagoras"),
            (ac + bc, "legs-added-for-pythagoras"),
            (bc, "given-side-repeated"),
            (ac * cos_a * cos_a, "cosine-applied-twice-for-leg"),
        ],
        ab,
    )
    # (d) BD: the area, the product with no division, the ratio inverted.
    d_d = _tag(
        [
            (ab * bc / 2, "area-for-altitude"),
            (ab * bc, "altitude-not-divided-by-hypotenuse"),
            (ab * ac / bc, "altitude-ratio-inverted"),
            (bc * sin_a, "sine-times-bc-for-altitude"),
        ],
        bd,
    )
    for name, ds in (("a", a_d), ("b", b_d), ("c", c_d), ("d", d_d)):
        check(len(ds) >= 2, f"too few ({name}) distractors: {ds}")

    ratio_phrase = f"tan(A) = {_t(given_value)}" if given == "tan" else f"cos(A) = {_t(given_value)}"
    stem = f"{rng.choice(_WORLD)}. {ratio_phrase} and BC = {length} cm, where BC is opposite angle A."
    ft_a = f"{length}/part_a"
    ft_b = f"sqrt(part_b**2 - {length}**2)"
    ft_c = f"part_c*{length}/part_b"
    return dict(
        text=stem,
        parts=[
            _fraction_part(
                "Find the exact value of sin(A), as a fraction in lowest terms.",
                sin_a,
                2,
                a_d,
                rng_range=(0, 1),
                working=[
                    f"cos(A) and tan(A) fix the triangle: opposite = {o}, adjacent = {a}, hypotenuse = {h}.",
                    f"sin(A) = {_t(sin_a)}.",
                ],
            ),
            _fraction_part(
                "Using your value of sin(A), find the length of AC, in cm.",
                ac,
                2,
                b_d,
                follow_through=ft_a,
                working=[f"AC = BC ÷ sin(A) = {length} ÷ {_t(sin_a)} = {_t(ac)} cm."],
            ),
            _fraction_part(
                "Using your length of AC, find the length of AB, in cm.",
                ab,
                2,
                c_d,
                follow_through=ft_b,
                working=[f"AB² = AC² - BC² = {_t(ac)}² - {length}², so AB = {_t(ab)} cm."],
            ),
            _fraction_part(
                "D is the point on AC where the perpendicular from B meets AC. "
                "Find the length of BD, in cm, as a fraction in lowest terms.",
                bd,
                3,
                d_d,
                follow_through=ft_c,
                working=[
                    "Area of triangle ABC = ½ × AB × BC = ½ × AC × BD.",
                    f"BD = AB × BC ÷ AC = {_t(ab)} × {length} ÷ {_t(ac)} = {_t(bd)} cm.",
                ],
            ),
        ],
    )
