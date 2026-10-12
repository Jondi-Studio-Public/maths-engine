"""Exam tier for a finite-basis least-squares approximation: a four-part chain through the normal equations.

A function is sampled at three nonzero integer points and fitted by y = c1 x^2 + c2 x^3. With Gram entries
G11 = sum x^4, G12 = sum x^5, G22 = sum x^6 and right-hand sides b1 = sum x^2 y, b2 = sum x^3 y:

(a) det G = G11 G22 - G12^2;
(b) c1 = (b1 G22 - b2 G12) / det G;
(c) c2, from the first normal equation, c2 = (b1 - G11 c1) / G12;
(d) the sum of squared residuals, an exact fraction.

Parts (b) to (d) follow through from the learner's own earlier answers. The residual is checked against the closed form
det([x^2, x^3, y])^2 / det G.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms
from ._exd_x_r3_distinct import distinct_answers
from ._exd_x_r3h_kit import fraction_part, number_part

_STEMS = (
    "A function f is sampled at {points}, giving {values}. It is approximated by the least-squares fit "
    "y = c1 x^2 + c2 x^3.",
    "The values of a function f are recorded at {points}: {values}. A least-squares fit of the form "
    "y = c1 x^2 + c2 x^3 is made to these three samples.",
    "Three samples of an unknown function f are available, taken at {points}, with {values}. The model "
    "y = c1 x^2 + c2 x^3 is fitted to them by least squares.",
)
_DET_PROMPTS = (
    "Find the determinant of the matrix G with entries G_ij equal to the sum, over the three sample points, of "
    "phi_i(x) phi_j(x), where phi_1 = x^2 and phi_2 = x^3.",
    "Let phi_1 = x^2 and phi_2 = x^3, and let G be the 2 by 2 matrix whose entries G_ij add phi_i(x) phi_j(x) over "
    "the three sample points. Find det G.",
    "The normal equations for the fit use the matrix G, where G_ij is the total of phi_i(x) phi_j(x) over the "
    "samples, with phi_1 = x^2 and phi_2 = x^3. Find the determinant of G.",
)


def _p(value) -> str:
    return f"({value})"


@generator("exd_x_r3h_least_squares_basis", level=19, presets={"exam": dict(tier="exam")})
@distinct_answers
def exd_x_r3h_least_squares_basis(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "sample points and values with a nonzero residual"):
        xs = rng.sample([-2, -1, 0, 1, 2], 3)
        ys = [rng.randint(-12, 30) for _ in xs]
        g11 = sum(x**4 for x in xs)
        g12 = sum(x**5 for x in xs)
        g22 = sum(x**6 for x in xs)
        b1 = sum(x * x * y for x, y in zip(xs, ys))
        b2 = sum(x**3 * y for x, y in zip(xs, ys))
        det = g11 * g22 - g12 * g12
        c1 = Fraction(b1 * g22 - b2 * g12, det)
        c2 = Fraction(g11 * b2 - g12 * b1, det)
        resid = [Fraction(y) - c1 * x * x - c2 * x**3 for x, y in zip(xs, ys)]
        rss = sum(r * r for r in resid)
        slips_d = [
            (sum(abs(r) for r in resid), "summed-sizes-not-squares"),
            (Fraction(sum(y * y for y in ys)), "total-sum-of-squares-for-residual-sum"),
            (sum((Fraction(y) - c2 * x * x - c1 * x**3) ** 2 for x, y in zip(xs, ys)), "swapped-coefficient-order"),
            (sum(resid), "summed-signed-values"),
        ]
        if (
            g12
            and c1
            and c2
            and rss
            and c1.denominator > 1
            and c2.denominator > 1
            and len({v for v, _ in slips_d} - {rss}) >= 3
        ):
            break
    # Independent routes: the normal equations solved by sympy, and the closed form for the residual.
    gram = sympy.Matrix([[g11, g12], [g12, g22]])
    solution = gram.LUsolve(sympy.Matrix([b1, b2]))
    check(solution[0] == sympy.Rational(c1.numerator, c1.denominator), "c1 from the normal equations")
    check(solution[1] == sympy.Rational(c2.numerator, c2.denominator), "c2 from the normal equations")
    d3 = sympy.Matrix([[x * x, x**3, y] for x, y in zip(xs, ys)]).det()
    check(Fraction(int(d3) ** 2, det) == rss, "residual from the bordered determinant")
    check(c2 == Fraction(b1 - g11 * c1, g12), "second coefficient from the first normal equation")

    points = ", ".join(f"x = {x}" for x in xs[:-1]) + f" and x = {xs[-1]}"
    values = ", ".join(f"f({x}) = {y}" for x, y in zip(xs[:-1], ys[:-1])) + f" and f({xs[-1]}) = {ys[-1]}"
    stem = rng.choice(_STEMS).format(points=points, values=values)
    det_prompt = rng.choice(_DET_PROMPTS)
    ft_b = f"({b1}*{_p(g22)} - {_p(b2)}*{_p(g12)})/part_a"
    ft_c = f"({b1} - {g11}*part_b)/{_p(g12)}"
    ft_d = " + ".join(f"({y} - part_b*{_p(x)}**2 - part_c*{_p(x)}**3)**2" for x, y in zip(xs, ys))
    return dict(
        text=stem,
        parts=[
            dict(
                **number_part(
                    det_prompt,
                    Fraction(det),
                    0,
                    2,
                    [f"G = [[{g11}, {g12}], [{g12}, {g22}]], so det G = {g11} x {g22} - ({g12})^2 = {det}."],
                    [
                        (g11 * g22, "determinant-ignores-cross-term"),
                        (g11 * g22 + g12 * g12, "determinant-adds-cross-term"),
                        (g11 * g22 - g12, "forgot-to-square-cross-gram-entry"),
                    ],
                    sensible_range=(0, None),
                )
            ),
            dict(
                **fraction_part(
                    "Find c1 exactly, as a fraction.",
                    c1,
                    3,
                    [f"b = ({b1}, {b2}). Cramer's rule: c1 = ({b1} x {g22} - ({b2}) x ({g12})) / {det} = {c1}."],
                    [
                        (Fraction(b1, g11), "ignored-coupling-between-basis-functions"),
                        (c2, "swapped-coefficient-order"),
                        (-c1, "sign-of-result-reversed"),
                        (Fraction(b1 * g22 + b2 * g12, det), "sign-slip-in-formula"),
                    ],
                    follow_through=ft_b,
                )
            ),
            dict(
                **fraction_part(
                    "Find c2 exactly, as a fraction.",
                    c2,
                    2,
                    [f"From the first normal equation, {join_terms([(g11, 'c1'), (g12, 'c2')])} = {b1}, so c2 = {c2}."],
                    [
                        (Fraction(b2, g22), "ignored-coupling-between-basis-functions"),
                        (c1, "swapped-coefficient-order"),
                        (-c2, "sign-of-result-reversed"),
                        (Fraction(g11 * b2 + g12 * b1, det), "sign-slip-in-formula"),
                    ],
                    follow_through=ft_c,
                )
            ),
            dict(
                **fraction_part(
                    "Find the sum of the squares of the residuals y_k - (c1 x_k^2 + c2 x_k^3) at the three sample "
                    "points, exactly, as a fraction.",
                    rss,
                    3,
                    ["Residuals: " + ", ".join(str(r) for r in resid) + ".", f"Sum of squares = {rss}."],
                    slips_d,
                    follow_through=ft_d,
                )
            ),
        ],
    )
