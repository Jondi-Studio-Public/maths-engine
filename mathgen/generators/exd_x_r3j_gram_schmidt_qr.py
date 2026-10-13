"""Generates `exd_x_r3j_gram_schmidt_qr` questions: the QR factorisation of a 2 x 2 matrix by Gram-Schmidt.

Skill: `ucv1-gram-schmidt-2d` (exam tier). The practice tiers orthonormalise two vectors in one step; a first-year
exam instead sets the factorisation A = QR and asks for its entries one at a time:

(a) r11, the length of the first column                          fraction, 1 mark
(b) r12, the component of the second column along e1            fraction, 2 marks, (u . v) / part_a
(c) r22, the length of the part of v orthogonal to e1           fraction, 2 marks, |det| / part_a
(d) one component of e2                                          fraction, 2 marks, (v_i - part_b e1_i) / part_c

## Construction: choose u from a Pythagorean triple

u = (a, b) has an integer length c, so r11 is a whole number and every later entry is an exact rational: r12 =
(u . v)/c and r22 = |det[u v]|/c. v is then drawn with small integer components so that u . v and det are non-zero
(r12 = 0 would make the follow-through from (b) vacuous).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ..core import paren_negative as pn
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3j_kit import fraction_part

_TRIPLES = [
    (3, 4, 5),
    (5, 12, 13),
    (8, 15, 17),
    (7, 24, 25),
    (20, 21, 29),
    (9, 40, 41),
    (12, 35, 37),
    (28, 45, 53),
    (11, 60, 61),
    (16, 63, 65),
    (33, 56, 65),
    (48, 55, 73),
    (13, 84, 85),
    (36, 77, 85),
    (39, 80, 89),
    (65, 72, 97),
]

_STEMS = [
    "The matrix A has columns u = ({a}, {b}) and v = ({p}, {q}). Applying the Gram-Schmidt process to its columns "
    "gives A = QR, where the columns of Q are the orthonormal vectors e1 and e2, and R is upper triangular with "
    "entries r11, r12 (first row) and r22 (second row), r11 and r22 being positive.",
    "Let A be the matrix with columns u = ({a}, {b}) and v = ({p}, {q}). Gram-Schmidt on the columns of A gives "
    "A = QR: Q has orthonormal columns e1 and e2, and R is upper triangular with positive diagonal entries r11 and "
    "r22 and with r12 above the diagonal.",
    "A QR factorisation A = QR is found for the matrix A whose columns are u = ({a}, {b}) and v = ({p}, {q}) by "
    "applying Gram-Schmidt to the columns. Q has orthonormal columns e1 and e2. R is upper triangular with "
    "entries r11 and r12 in its first row and r22 in its second, where r11 and r22 are positive.",
]


@generator("exd_x_r3j_gram_schmidt_qr", level=17, presets={"exam": {}})
@distinct_part_answers
def exd_x_r3j_gram_schmidt_qr(rng):
    for _ in attempts(1000, "a column pair with non-zero dot product and determinant"):
        a0, b0, c = rng.choice(_TRIPLES)
        if rng.random() < 0.5:
            a0, b0 = b0, a0
        a, b = a0 * rng.choice([-1, 1]), b0 * rng.choice([-1, 1])
        p, q = rng.randint(-9, 9), rng.randint(-9, 9)
        dot, det = a * p + b * q, a * q - b * p
        if dot != 0 and det != 0 and abs(dot) % c != 0:
            break

    r11 = Fraction(c)
    r12 = Fraction(dot, c)
    r22 = Fraction(abs(det), c)
    sign = 1 if det > 0 else -1
    # e2 = (v - r12 e1) / r22 and e1 = u / c, so e2 = sign * (-b, a) / c
    use_x = rng.random() < 0.5
    comp, v_comp, u_comp, e2_comp = ("x", p, a, Fraction(-sign * b, c)) if use_x else ("y", q, b, Fraction(sign * a, c))
    e2 = (v_comp - r12 * Fraction(u_comp, c)) / r22
    check(e2 == e2_comp, "e2 component disagrees with the perpendicular formula")
    check(r12**2 + r22**2 == Fraction(p * p + q * q), "R is not consistent with |v|")

    stem = rng.choice(_STEMS).format(a=a, b=b, p=p, q=q)
    other_comp = Fraction(u_comp, c)

    parts = [
        fraction_part(
            "Find r11.",
            r11,
            1,
            [f"r11 = |u| = √({pn(a)}² + {pn(b)}²) = {c}."],
            [
                (Fraction(c * c), "forgot-square-root"),
                (Fraction(abs(a) + abs(b)), "speed-adds-components"),
                (Fraction(abs(a * b)), "multiplied-components-for-length"),
            ],
            sensible_range=(0, None),
        ),
        fraction_part(
            "Find r12.",
            r12,
            2,
            [f"e1 = u / {c}, and r12 = e1 · v = ({pn(a)}·{pn(p)} + {pn(b)}·{pn(q)}) / {c} = {dot}/{c}."],
            [
                (Fraction(dot), "dot-product-not-normalised"),
                (Fraction(dot, c * c), "divided-by-squared-length"),
                (Fraction(det, c), "dot-and-cross-product-swapped"),
            ],
            follow_through=f"({dot})/part_a",
        ),
        fraction_part(
            "Find r22.",
            r22,
            2,
            [f"r22 = |v − r12 e1| = |det[u v]| / |u| = |{pn(a)}·{pn(q)} − {pn(b)}·{pn(p)}| / {c} = {abs(det)}/{c}."],
            [
                (Fraction(abs(det)), "area-for-height"),
                (Fraction(abs(det), c * c), "divided-by-squared-length"),
                (Fraction(abs(dot), c), "dot-and-cross-product-swapped"),
            ],
            follow_through=f"({abs(det)})/part_a",
            sensible_range=(0, None),
        ),
        fraction_part(
            f"Find the {comp}-component of e2.",
            e2,
            2,
            [
                f"v − r12 e1 has {comp}-component {v_comp} − ({dot}/{c})({u_comp}/{c}); dividing by r22 gives "
                f"{e2.numerator}/{e2.denominator}."
                if e2.denominator != 1
                else f"The {comp}-component of e2 is {e2.numerator}."
            ],
            [
                (-e2, "sign-of-result-reversed"),
                (other_comp, "gave-first-vector-component"),
                (Fraction(v_comp) / r22, "gram-schmidt-skipped-projection"),
                ((v_comp - r12 * other_comp), "forgot-final-normalisation"),
            ],
            follow_through=f"({v_comp} - part_b*({u_comp})/part_a)/part_c",
        ),
    ]
    return dict(text=stem, parts=parts)
