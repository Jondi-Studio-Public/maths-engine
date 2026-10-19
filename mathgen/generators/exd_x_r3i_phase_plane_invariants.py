"""Exam-tier source for `ucv1-phase-plane-linear-classification`: the quantities that classify the origin of a
2-by-2 linear system, computed rather than named.

The stem gives x' = a x + b y, y' = c x + d y. The parts compute, in order, the pieces a classification rests on and
end on a number that carries the stability verdict:

(a) the trace T = a + d                                        number, 1 mark
(b) the determinant D = ad - bc                                number, 1 mark
(c) the discriminant T^2 - 4D                                  number, 2 marks, part_a^2 - 4 part_b
(d) the largest real part of the eigenvalues, 1 d.p.           number, 2 marks, (part_a + sqrt(part_c)) / 2
                                                               (or part_a / 2 when the eigenvalues are complex)

The sign of (d) is the stability verdict: negative is a sink or a spiral sink, positive a source or a saddle's
unstable direction, zero a centre or a degenerate case.

## Construction

Coefficients are drawn from -6..6 and kept when the discriminant is either negative (complex eigenvalues) or a
perfect square (rational eigenvalues), so (d) is a multiple of one half and exact to one decimal place. The draw
aims at a spread of cases: real distinct eigenvalues of both signs, complex ones with a positive, negative or zero
real part, and repeated ones.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator, join_terms, paren_negative
from ._exd_distinct_parts import distinct_part_answers
from .exd_x_r3i_support import number_part

_KINDS = ("real", "real", "complex", "complex", "complex-or-zero")


@generator("exd_x_r3i_phase_plane_invariants", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3i_phase_plane_invariants(rng, tier: str = "exam") -> dict:
    """Trace, determinant, discriminant and largest real eigenvalue part of a 2-by-2 linear system."""
    check(tier == "exam", f"unknown tier {tier!r}")
    want = rng.choice(_KINDS)
    for _ in attempts(5000, "a system with usable eigenvalues"):
        a, b, c, d = (rng.randint(-6, 6) for _ in range(4))
        T, D = a + d, a * d - b * c
        disc = T * T - 4 * D
        if (b == 0 and c == 0) or D == 0:
            continue
        if disc >= 0 and math.isqrt(disc) ** 2 != disc:
            continue
        if want == "real" and disc <= 0:
            continue
        if want.startswith("complex") and disc >= 0:
            continue
        if want == "complex" and T == 0:
            continue
        if abs(T) > 9 or abs(D) > 40:
            continue
        break
    root = Fraction(math.isqrt(disc)) if disc >= 0 else None
    top = (T + root) / 2 if root is not None else Fraction(T, 2)
    # Independent route: the eigenvalues solve l^2 - T l + D = 0.
    if root is not None:
        check(all(l * l - T * l + D == 0 for l in ((T + root) / 2, (T - root) / 2)), "eigenvalue does not solve")
    check(top * 2 == int(top * 2), "largest real part is not a multiple of one half")

    stem = (
        f"The linear system x' = {join_terms([(a, 'x'), (b, 'y')])}, y' = {join_terms([(c, 'x'), (d, 'y')])} "
        "can be written x' = Ax for a 2-by-2 matrix A."
    )
    part_a = number_part(
        "Find the trace of A.",
        Fraction(T),
        0,
        1,
        [f"T = {a} + ({d}) = {T}."],
        [
            ("subtracted-instead-of-added", Fraction(a - d)),
            ("trace-adds-off-diagonal", Fraction(b + c)),
            ("multiplied-instead-of-added", Fraction(a * d)),
            ("trace-is-determinant", Fraction(D)),
        ],
    )
    part_b = number_part(
        "Find the determinant of A.",
        Fraction(D),
        0,
        1,
        [f"D = ({a})({d}) - ({b})({c}) = {D}."],
        [
            ("determinant-adds-cross-term", Fraction(a * d + b * c)),
            ("sign-of-result-reversed", Fraction(-D)),
            ("determinant-ignores-cross-term", Fraction(a * d)),
            ("determinant-wrong-pairing", Fraction(a * c - b * d)),
            ("determinant-wrong-pairing", Fraction(a * b - c * d)),
            ("determinant-wrong-pairing", Fraction(a * d - b - c)),
        ],
    )
    part_c = number_part(
        "Find the value of T² - 4D, where T is the trace and D the determinant of A.",
        Fraction(disc),
        0,
        2,
        [f"T² - 4D = {paren_negative(T)}² - 4({D}) = {disc}."],
        [
            ("sign-slip-in-formula", Fraction(T * T + 4 * D)),
            ("discriminant-forgot-to-square-trace", Fraction(T - 4 * D)),
            ("wrong-constant-in-formula", Fraction(T * T - 2 * D)),
            ("discriminant-uses-trace-only", Fraction(T * T)),
        ],
        follow_through="part_a**2 - 4*part_b",
    )
    if root is not None:
        d_ft = "(part_a + sqrt(part_c))/2"
        d_cands = [
            ("gave-other-root", (T - root) / 2),
            ("eigenvalue-is-half-the-trace", Fraction(T, 2)),
            ("dropped-factor-of-two", T + root),
            ("ignores-trace-in-eigenvalue", root / 2),
        ]
        working = [f"λ = (T ± √(T² - 4D))/2 = ({T} ± {root})/2, so the larger is {float(top):.1f}."]
    else:
        d_ft = "part_a/2"
        d_cands = [
            ("dropped-factor-of-two", Fraction(T)),
            ("sign-of-result-reversed", Fraction(-T, 2)),
            ("real-part-from-wrong-invariant", Fraction(D, 2)),
            ("real-part-from-wrong-invariant", Fraction(disc, 2)),
        ]
        working = [f"The eigenvalues are complex, with real part T/2 = {float(top):.1f}."]
    part_d = number_part(
        "Find the largest real part of the eigenvalues of A, correct to 1 decimal place.",
        top,
        1,
        2,
        working,
        d_cands,
        follow_through=d_ft,
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
