"""Exam tier for classifying a quadratic by its Hessian: a four-part chain that ends in the number of negative eigenvalues.

f(x, y) = a x^2 + c xy + b y^2 + d x + e y has the constant Hessian H = [[2a, c], [c, 2b]]; the linear terms play no
part. The question asks for the working a classification rests on, and the verdict as a count:

(a) det H = 4ab - c^2;
(b) trace H = 2(a + b);
(c) the smaller eigenvalue, (tr - sqrt(tr^2 - 4 det))/2, to 2 decimal places;
(d) the number of negative eigenvalues of H (0 convex, 1 indefinite, 2 concave), from the smaller eigenvalue and the
    trace.

Built backwards from integer eigenvalues: with a - b = u and cross term c a Pythagorean pair (u, c, w), the
eigenvalues of H are (a + b) +/- w. The outcome (convex, indefinite, concave) is drawn first so each is equally common.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, join_terms
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3h_kit import number_part

# (|u|, |c|, w) with u^2 + c^2 = w^2; u may be 0 (c = w) and the two legs may swap.
_TRIPLES = [
    (3, 4, 5),
    (4, 3, 5),
    (6, 8, 10),
    (8, 6, 10),
    (5, 12, 13),
    (12, 5, 13),
    (8, 15, 17),
    (15, 8, 17),
    (9, 12, 15),
    (12, 9, 15),
    (12, 16, 20),
    (16, 12, 20),
    (0, 2, 2),
    (0, 3, 3),
    (0, 4, 4),
    (0, 5, 5),
    (0, 6, 6),
    (0, 7, 7),
    (0, 8, 8),
]


@generator("exd_x_r3h_hessian_definiteness", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3h_hessian_definiteness(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a quadratic with integer eigenvalues in a drawn class"):
        u0, c0, w = rng.choice(_TRIPLES)
        u = rng.choice((-1, 1)) * u0
        c = rng.choice((-1, 1)) * c0
        outcome = rng.choice(("convex", "indefinite", "concave"))
        if outcome == "convex":
            s = rng.randint(w + 1, w + 8)
        elif outcome == "concave":
            s = -rng.randint(w + 1, w + 8)
        else:
            s = rng.randint(-(w - 1), w - 1)
        if (s + u) % 2 or c == 0 or s == 0:
            continue
        a, b = (s + u) // 2, (s - u) // 2
        if max(abs(a), abs(b)) > 12 or a * b == 0:
            continue
        break
    lo, hi = s - w, s + w
    det, trace = 4 * a * b - c * c, 2 * (a + b)
    check(det == lo * hi and trace == lo + hi, "eigenvalues from determinant and trace")
    negatives = (lo < 0) + (hi < 0)
    check(negatives == {"convex": 0, "indefinite": 1, "concave": 2}[outcome], "count of negative eigenvalues")
    d, e = rng.randint(-5, 5), rng.randint(-5, 5)
    poly = join_terms([(a, "x^2"), (c, "xy"), (b, "y^2"), (d, "x"), (e, "y")])
    stem = f"Let f(x, y) = {poly}, with Hessian matrix H."

    positives = (lo > 0) + (hi > 0)
    diag_negative = (a < 0) + (b < 0)
    return dict(
        text=stem,
        parts=[
            dict(
                **number_part(
                    "Find the determinant of H.",
                    Fraction(det),
                    0,
                    2,
                    [f"H = [[{2 * a}, {c}], [{c}, {2 * b}]]; det H = ({2 * a})({2 * b}) - ({c})^2 = {det}."],
                    [
                        (4 * a * b, "determinant-ignores-cross-term"),
                        (a * b - c * c, "used-coefficients-instead-of-second-derivatives"),
                        (4 * a * b + c * c, "determinant-adds-cross-term"),
                        (4 * a * b - 2 * c, "squared-as-doubled"),
                    ],
                )
            ),
            dict(
                **number_part(
                    "Find the trace of H.",
                    Fraction(trace),
                    0,
                    1,
                    [f"tr H = {join_terms([(2 * a, ''), (2 * b, '')])} = {trace}."],
                    [
                        (a + b, "used-coefficients-instead-of-second-derivatives"),
                        (trace + c, "trace-adds-off-diagonal"),
                        (2 * a, "trace-from-first-diagonal-entry-only"),
                        (trace - c, "trace-adds-off-diagonal"),
                    ],
                )
            ),
            dict(
                **number_part(
                    "Find the smaller eigenvalue of H, correct to 2 decimal places.",
                    Fraction(lo),
                    2,
                    3,
                    [
                        f"The eigenvalues satisfy t^2 - ({trace})t + ({det}) = 0.",
                        f"t = ({trace} - sqrt({trace * trace - 4 * det})) / 2 = {lo}.",
                    ],
                    [
                        (hi, "gave-other-root"),
                        (-lo, "sign-of-result-reversed"),
                        (2 * lo, "dropped-factor-of-two"),
                        (Fraction(trace, 2), "eigenvalue-is-half-the-trace"),
                    ],
                    follow_through="(part_b - sqrt(part_b**2 - 4*part_a))/2",
                )
            ),
            dict(
                **number_part(
                    "How many of the eigenvalues of H are negative?",
                    Fraction(negatives),
                    0,
                    1,
                    [f"The eigenvalues are {lo} and {hi}, so {negatives} of them are negative."],
                    [
                        (positives, "counted-positive-instead-of-negative-eigenvalues"),
                        (diag_negative, "counted-negative-diagonal-entries-instead-of-eigenvalues"),
                        (1 if det < 0 else 2, "counted-from-determinant-sign-alone"),
                        (0, "definiteness-assumed-from-diagonal-signs"),
                        (2, "definiteness-assumed-from-diagonal-signs"),
                        (1, "indefinite-assumed-from-cross-term-present"),
                    ],
                    follow_through="(1 - sign(part_c))/2 + (1 - sign(part_b - part_c))/2",
                )
            ),
        ],
    )
