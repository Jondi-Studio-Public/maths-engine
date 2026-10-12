"""Exam source for `senior-express-a-vector-in-component-form-using-the`: a resultant, its length, a vector along it.

The stem gives a = a1 i + a2 j and b = b1 i + b2 j and the vector r = a - k b, built backwards so that r is a
scaled Pythagorean-triple vector (its magnitude is a whole number).

(a) the i-coefficient of r;
(b) the j-coefficient of r;
(c) the magnitude of r, from (a) and (b), to 1 d.p. (a wrong coefficient does not give a whole magnitude);
(d) the j-coefficient of the vector of stated length L in the direction of r, from (b) and (c), to 1 d.p.

Parts (c) and (d) carry `follow_through`. The key is re-derived from r = a - k b with exact integers.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import check, generator, join_terms

from .exd_x_r3c_parts import number_part
from ._exd_r3_distinct import distinct_part_answers

_TRIPLES = ((3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29))
_MAX_SCALE = {5: 5, 13: 2, 17: 2, 25: 1, 29: 1}
_STEMS = (
    "The vectors a and b are given by a = {a} and b = {b}. The vector r is defined by r = a - {k}b.",
    "Let a = {a} and b = {b}. The vector r is given by r = a - {k}b.",
    "Two vectors are a = {a} and b = {b}, and r = a - {k}b.",
)


@generator("exd_x_r3c_vector_resultant", level=15, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3c_vector_resultant(rng, tier="exam"):
    """r = a - kb in components, |r|, and the j-coefficient of the vector of length L along r."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    p, q, h = rng.choice(_TRIPLES)
    if rng.random() < 0.5:
        p, q = q, p
    scale = rng.randint(1, _MAX_SCALE[h])
    r1, r2 = rng.choice((-1, 1)) * p * scale, rng.choice((-1, 1)) * q * scale
    magnitude = h * scale
    k = rng.choice((2, 3, 4))
    nonzero = [v for v in range(-8, 9) if v]
    b1, b2 = rng.choice(nonzero), rng.choice(nonzero)
    a1, a2 = r1 + k * b1, r2 + k * b2
    length = h * rng.randint(2, 6)
    ans = Fraction(length * r2, magnitude)
    check(ans.denominator == 1, "the length is not a multiple of the reduced magnitude")
    check(a1 - k * b1 == r1 and a2 - k * b2 == r2 and r1 * r1 + r2 * r2 == magnitude**2, "r is not a - kb")

    stem = rng.choice(_STEMS).format(a=join_terms([(a1, "i"), (a2, "j")]), b=join_terms([(b1, "i"), (b2, "j")]), k=k)
    part_a = number_part(
        "Write r in component form using the unit vectors i and j. What is the coefficient of i?",
        r1,
        working=[f"r = a - {k}b, so the i-coefficient is {a1} - {k} × ({b1}) = {r1}."],
        wrongs=[
            (a1 - b1, "forgot-scalar-multiple"),
            (a1 + k * b1, "reversed-vector-direction"),
            (k * b1 - a1, "reversed-subtraction"),
        ],
    )
    part_b = number_part(
        "What is the coefficient of j in r?",
        r2,
        working=[f"The j-coefficient is {a2} - {k} × ({b2}) = {r2}."],
        wrongs=[
            (a2 - b2, "forgot-scalar-multiple"),
            (a2 + k * b2, "reversed-vector-direction"),
            (k * b2 - a2, "reversed-subtraction"),
        ],
    )
    part_c = number_part(
        "Using your coefficients from (a) and (b), find the magnitude of r, correct to 1 decimal place.",
        magnitude,
        dp=1,
        marks=2,
        sensible_range=(0, None),
        working=[f"|r| = √(({r1})² + ({r2})²) = √{r1**2 + r2**2} = {magnitude}."],
        wrongs=[
            (abs(r1) + abs(r2), "magnitude-as-sum-of-components"),
            (r1 * r1 + r2 * r2, "magnitude-without-square-root"),
            (abs(r1), "magnitude-as-one-leg"),
            (abs(r2), "magnitude-as-one-leg"),
        ],
        follow_through="sqrt(part_a**2 + part_b**2)",
    )
    part_d = number_part(
        f"A vector v has magnitude {length} and points in the same direction as r. Using your answers to (b) and (c), "
        "what is the coefficient of j in v, correct to 1 decimal place?",
        ans,
        dp=1,
        marks=2,
        working=[f"v = ({length}/{magnitude}) r, so its j-coefficient is {length} × ({r2}) ÷ {magnitude} = {ans}."],
        wrongs=[
            (Fraction(length * r1, magnitude), "used-other-component-of-vector"),
            (Fraction(length * r2), "forgot-to-normalise"),
            (-ans, "unit-component-sign-slip"),
            (Fraction(length * r2, abs(r1) + abs(r2)), "magnitude-as-sum-of-components"),
        ],
        follow_through=f"{length}*part_b/part_c",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
