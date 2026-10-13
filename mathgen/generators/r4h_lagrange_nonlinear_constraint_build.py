"""Lagrange multipliers with a nonlinear constraint in two variables: `r4h_lagrange_nonlinear_constraint` (easy, medium,
hard) and the exam chain `exd_x_r4h_lagrange_nonlinear_constraint`.

``uc2-lagrange-multiplier-nonlinear-constraint`` (``optimisation``). Extremise ``f(x, y)`` subject to ``g(x, y) = c``
by solving ``∇f = λ∇g`` together with the constraint.

* easy: the maximum of a linear function on a circle ``x² + y² = c``.
* medium: the maximum of a linear function on an ellipse ``px² + qy² = c``, or of ``xy`` on an ellipse.
* hard: the maximum of a linear function on the conic ``x² + xy + y² = c`` (the cross term makes the two gradient
  equations couple the variables).
* exam: a linear function on an ellipse in four parts: ``λ``, the maximiser's ``x`` and ``y`` (follow-through from
  ``λ``) and the maximum value (follow-through from the coordinates).

Every instance is built backwards from a chosen extreme point, so all answers are exact fractions.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._r4h_kit import (
    Redraw,
    dtxt,
    finish_parts,
    fraction_part,
    ft_symbol,
    lin_text,
    rat,
    single_from,
    tx,
    with_redraw,
)

MISC = {
    "num2t-lagr-minimum-reported": "Reporting the minimum value (the negative of the maximum) when the maximum was asked for.",
    "num2t-lagr-radius-for-radius-squared": "Treating the constant on the right of the circle as the radius instead of the radius squared.",
    "num2t-lagr-radius-squared-factor": "Multiplying the length of the objective's gradient by the right-hand side instead of its square root.",
    "num2t-lagr-sum-of-coefficients": "Using the sum of the coefficients in place of the length of the objective's gradient.",
    "num2t-lagr-objective-at-gradient-point": "Evaluating the objective at the point (a, b) itself instead of at the point on the constraint.",
    "num2t-lagr-circle-value-unscaled": "Dividing by the right-hand side instead of multiplying by its square root.",
    "num2t-lagr-ellipse-weights-ignored": "Ignoring the weights of the squares when solving for the point on an ellipse.",
    "num2t-lagr-multiplier-reported": "Reporting the multiplier λ in place of the extreme value.",
    "num2t-lagr-factor-two-lost": "Losing the factor 2 in the value 2λc by writing the extreme value as λc.",
    "num2t-lagr-gradient-cross-term-missed": "Differentiating x² + xy + y² as 2x + 2y, missing the cross term.",
    "num2t-lagr-product-over-weights": "Writing the maximum of xy on px² + qy² = c as c/(2pq) instead of c/(2√(pq)).",
    "num2t-lagr-product-no-root": "Writing the maximum of xy as c/√(pq), a factor 2 too large.",
    "num2t-lagr-product-factor-four": "Writing the maximum of xy as c/(4√(pq)), a factor 2 too small.",
    "num2t-lagr-product-at-axis": "Evaluating xy at a point of the ellipse on an axis, where xy = 0.",
    "num2t-lagr-lambda-squared": "Giving λ² where λ was asked for, forgetting the square root.",
    "num2t-lagr-lambda-doubled": "Giving 2λ where λ was asked for.",
    "num2t-lagr-lambda-negative-root": "Giving the negative root λ, which belongs to the minimum.",
    "num2t-lagr-lambda-unweighted": "Solving for λ with the ellipse's weights p and q left out.",
    "num2t-lagr-coordinate-gradient-not-scaled": "Giving the coordinate as a/(2λ), leaving out the weight on the square.",
    "num2t-lagr-coordinate-sign": "Giving the coordinate of the minimiser instead of the maximiser.",
    "num2t-lagr-coordinate-swapped": "Giving the other variable's coordinate.",
    "num2t-lagr-coordinate-uses-doubled-lambda": "Dividing by λ rather than 2pλ when solving for the coordinate.",
    "num2t-lagr-value-sum-of-coordinates": "Adding the coordinates of the maximiser instead of evaluating the objective.",
    "num2t-lagr-value-first-term": "Evaluating only the first term of the objective at the maximiser.",
    "num2t-lagr-value-constraint": "Giving the constraint's right-hand side as the maximum.",
}
M_MIN = "num2t-lagr-minimum-reported"
M_R = "num2t-lagr-radius-for-radius-squared"
M_RSQ = "num2t-lagr-radius-squared-factor"
M_SUM = "num2t-lagr-sum-of-coefficients"
M_AT_AB = "num2t-lagr-objective-at-gradient-point"
M_DIV = "num2t-lagr-circle-value-unscaled"
M_LAM = "num2t-lagr-multiplier-reported"
M_TWO = "num2t-lagr-factor-two-lost"
M_CROSS = "num2t-lagr-gradient-cross-term-missed"
M_PQ = "num2t-lagr-product-over-weights"
M_NOROOT = "num2t-lagr-product-no-root"
M_FOUR = "num2t-lagr-product-factor-four"

_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (4, 3, 5), (12, 5, 13), (6, 8, 10)]


def quad_text(p, q, with_cross=False) -> str:
    out = []
    for coef, v in ((p, "x²"), (q, "y²")):
        out.append(v if coef == 1 else f"{coef}{v}")
    if with_cross:
        out.insert(1, "xy")
    return " + ".join(out)


@generator(
    "r4h_lagrange_nonlinear_constraint",
    level=22,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4h_lagrange_nonlinear_constraint(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return with_redraw(lambda: {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng), "a Lagrange question")


def _easy(rng):
    a0, b0, h0 = rng.choice(_TRIPLES)
    k = rng.choice([1, 2, 3])
    sa, sb = rng.choice([(1, 1), (1, -1), (-1, 1), (1, 1)])
    a, b, hyp = sa * a0 * k, sb * b0 * k, h0 * k
    r = rng.choice([Fraction(1), Fraction(2), Fraction(3), Fraction(1, 2), Fraction(3, 2), Fraction(5, 2), Fraction(4)])
    c = r * r
    value = r * hyp
    cands = [
        (M_MIN, -value),
        (M_R, c * hyp),
        (M_RSQ, c * hyp),
        (M_SUM, r * (abs(a) + abs(b))),
        (M_AT_AB, Fraction(a * a + b * b)),
        (M_DIV, hyp / r),
    ]
    part = fraction_part("", value, cands, 1, min_options=3)
    f_t = lin_text(a, b)
    ct = dtxt(c)
    stem = rng.choice(
        [
            f"Find the maximum value of «f(x, y) = {f_t}» on the circle «x² + y² = {ct}», as a fraction or whole number.",
            f"Use Lagrange multipliers to find the largest value taken by «{f_t}» at points of «x² + y² = {ct}». Give "
            "it as a fraction or whole number.",
            f"A point «(x, y)» moves on the circle «x² + y² = {ct}». What is the greatest value of «{f_t}»?",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            f"∇f = ({a}, {b}) = λ(2x, 2y), so (x, y) is a multiple of ({a}, {b}) with x² + y² = {ct}.",
            f"The maximum is {tx(value)}.",
        ],
    )


_PAIRS = [(1, 4), (4, 1), (1, 9), (9, 1), (4, 9), (9, 4), (1, 16), (9, 16)]


def _medium(rng):
    if rng.random() < 0.45:
        p, q = rng.choice(_PAIRS)
        root = {1: 1, 4: 2, 9: 3, 16: 4}
        rpq = root[p] * root[q]
        c = rng.choice([2, 4, 6, 8, 12, 18, 24])
        value = Fraction(c, 2 * rpq)
        cands = [
            (M_PQ, Fraction(c, 2 * p * q)),
            (M_NOROOT, Fraction(c, rpq)),
            (M_FOUR, Fraction(c, 4 * rpq)),
            ("num2t-lagr-product-at-axis", Fraction(0)),
            ("num2t-lagr-value-constraint", Fraction(c)),
        ]
        part = fraction_part("", value, cands, 1, min_options=3)
        qt = quad_text(p, q)
        stem = rng.choice(
            [
                f"Find the maximum value of «f(x, y) = xy» on the ellipse «{qt} = {c}», as a fraction or whole number.",
                f"The point «(x, y)» lies on the ellipse «{qt} = {c}». Use Lagrange multipliers to find the greatest "
                "value of «xy», as a fraction or whole number.",
                f"Maximise «xy» subject to «{qt} = {c}» and give the maximum as a fraction or whole number.",
            ]
        )
        return single_from(
            part,
            stem,
            working=[
                f"∇f = (y, x) = λ({2 * p}x, {2 * q}y) gives {p}x² = {q}y², so each term of the constraint is {tx(Fraction(c, 2))}.",
                f"The maximum of xy is {tx(value)}.",
            ],
        )
    p, q = rng.choice([(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (4, 1), (3, 4)])
    xs, ys = rng.randint(1, 4), rng.randint(1, 4)
    k = rng.choice([1, 2, 3])
    a, b = k * p * xs, k * q * ys
    c = p * xs * xs + q * ys * ys
    lam = Fraction(k, 2)
    value = 2 * lam * c
    check(value == a * xs + b * ys, "the extreme value disagrees with f at the point")
    cands = [
        (M_MIN, -value),
        (M_TWO, lam * c),
        (M_AT_AB, Fraction(a * a + b * b)),
        ("num2t-lagr-ellipse-weights-ignored", _unweighted(a, b, c)),
        (M_LAM, lam),
    ]
    part = fraction_part("", value, cands, 1, min_options=3)
    f_t, qt = lin_text(a, b), quad_text(p, q)
    stem = rng.choice(
        [
            f"Find the maximum value of «f(x, y) = {f_t}» on the ellipse «{qt} = {c}».",
            f"Use Lagrange multipliers to find the largest value of «{f_t}» for points on the ellipse «{qt} = {c}».",
            f"Maximise «{f_t}» subject to the constraint «{qt} = {c}».",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            f"∇f = ({a}, {b}) = λ({2 * p}x, {2 * q}y), so x = {a}/({2 * p}λ) and y = {b}/({2 * q}λ).",
            f"Substituting into the constraint gives λ = {tx(lam)} (for the maximum), the point ({xs}, {ys}) and f = {value}.",
        ],
    )


def _unweighted(a, b, c):
    """The maximum if the ellipse were read as a circle x² + y² = c: a value with a square root, rounded off."""
    import math

    return Fraction(round(math.sqrt(c * (a * a + b * b))))


def _hard(rng):
    xs, ys = rng.randint(-3, 4), rng.randint(-3, 4)
    mu = rng.choice([1, 2, 3])
    p, q = mu * (2 * xs + ys), mu * (xs + 2 * ys)
    c = xs * xs + xs * ys + ys * ys
    if c == 0 or p == 0 or q == 0 or xs == ys or p == q:
        raise Redraw
    value = 2 * mu * c
    cands = [
        (M_MIN, -value),
        (M_TWO, Fraction(mu * c)),
        (M_LAM, Fraction(1, mu)),
        (M_CROSS, Fraction(p * xs + q * ys) if False else Fraction(abs(p * ys + q * xs))),
        ("num2t-lagr-value-constraint", Fraction(c)),
        ("num2t-lagr-value-first-term", Fraction(abs(p * xs))),
    ]
    part = fraction_part("", value, cands, 1, min_options=3)
    f_t = lin_text(p, q)
    stem = rng.choice(
        [
            f"Find the maximum value of «f(x, y) = {f_t}» on the curve «x² + xy + y² = {c}».",
            f"Points «(x, y)» satisfy «x² + xy + y² = {c}». Use Lagrange multipliers to find the largest value of «{f_t}».",
            f"Maximise «{f_t}» subject to «x² + xy + y² = {c}».",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            f"∇f = ({p}, {q}) = λ(2x + y, x + 2y). Eliminating λ gives {p}(x + 2y) = {q}(2x + y).",
            f"The constraint then gives the point ({xs}, {ys}) (or its negative), so the maximum is {value}.",
        ],
    )


@generator("exd_x_r4h_lagrange_nonlinear_constraint", level=22, presets={"exam": dict(tier="exam")})
def exd_x_r4h_lagrange_nonlinear_constraint(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam Lagrange question")


def _exam(rng):
    p, q = rng.choice([(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (4, 1), (1, 1)])
    xs, ys = rng.randint(1, 4), rng.randint(1, 4)
    k = rng.choice([1, 2, 3, 4])
    a, b = k * p * xs, k * q * ys
    c = p * xs * xs + q * ys * ys
    lam = Fraction(k, 2)
    value = a * xs + b * ys
    la = ft_symbol("part_a")
    s2 = Fraction(a * a, p) + Fraction(b * b, q)
    check(lam * lam == s2 / (4 * c), "lambda formula")
    p_a = fraction_part(
        "At the maximum the multiplier «λ» in «∇f = λ∇g» is positive. Find it.",
        lam,
        [
            ("num2t-lagr-lambda-squared", lam * lam),
            ("num2t-lagr-lambda-doubled", 2 * lam),
            ("num2t-lagr-lambda-negative-root", -lam),
            ("num2t-lagr-lambda-unweighted", Fraction(a * a + b * b, 4 * c) if False else s2 / (2 * c)),
        ],
        1,
        working=[f"x = a/(2pλ), y = b/(2qλ); the constraint gives 4λ²c = a²/p + b²/q, so λ = {tx(lam)}."],
        value_range=(0, None),
    )
    p_b = fraction_part(
        "Use your «λ» to find the «x»-coordinate of the maximiser.",
        xs,
        [
            ("num2t-lagr-coordinate-gradient-not-scaled", Fraction(a) / (2 * lam)),
            ("num2t-lagr-coordinate-uses-doubled-lambda", Fraction(a) / lam),
            ("num2t-lagr-coordinate-sign", -Fraction(xs)),
            ("num2t-lagr-coordinate-swapped", Fraction(ys)),
        ],
        1,
        ft=str(rat(a) / (2 * rat(p) * la)),
        working=[f"x = {a}/({2 * p}·{tx(lam)}) = {xs}."],
    )
    p_c = fraction_part(
        "Use your «λ» to find the «y»-coordinate of the maximiser.",
        ys,
        [
            ("num2t-lagr-coordinate-gradient-not-scaled", Fraction(b) / (2 * lam)),
            ("num2t-lagr-coordinate-uses-doubled-lambda", Fraction(b) / lam),
            ("num2t-lagr-coordinate-sign", -Fraction(ys)),
            ("num2t-lagr-coordinate-swapped", Fraction(xs)),
        ],
        1,
        ft=str(rat(b) / (2 * rat(q) * la)),
        working=[f"y = {b}/({2 * q}·{tx(lam)}) = {ys}."],
    )
    xb, yc = ft_symbol("part_b"), ft_symbol("part_c")
    p_d = fraction_part(
        "Find the maximum value of «f», using your coordinates.",
        Fraction(value),
        [
            ("num2t-lagr-value-sum-of-coordinates", Fraction(xs + ys)),
            ("num2t-lagr-value-first-term", Fraction(a * xs)),
            ("num2t-lagr-value-constraint", Fraction(c)),
            ("num2t-lagr-minimum-reported", Fraction(-value)),
        ],
        2,
        ft=str(rat(a) * xb + rat(b) * yc),
        working=[f"f = {a}·{xs} + {b}·{ys} = {value}."],
    )
    f_t, qt = lin_text(a, b), quad_text(p, q)
    stem = rng.choice(
        [
            f"Maximise «f(x, y) = {f_t}» subject to «{qt} = {c}», using a Lagrange multiplier «λ» with «∇f = λ∇g» "
            f"for «g(x, y) = {qt}».",
            f"The function «f(x, y) = {f_t}» is to be maximised on the ellipse «{qt} = {c}». Write the constraint as "
            f"«g(x, y) = {qt}» and use «∇f = λ∇g».",
            f"On the ellipse «{qt} = {c}», find the point where «{f_t}» is largest, taking «g(x, y) = {qt}» and "
            "«∇f = λ∇g».",
        ]
    )
    return finish_parts([p_a, p_b, p_c, p_d], stem)
